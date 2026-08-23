# `ragorc.index` — loaders, splitters, ingest, and the optional stages

Documents in, vectors out, exactly once. Every stage this package drives already
exists elsewhere; what lives here is the **order**, the **memory policy** and the
**failure policy** — which is what actually decides whether an ingest is cheap,
restartable and correct.

Related: [ADR-0002 — late chunking](../adr/0002-late-chunking.md) ·
[docs/performance.md](../performance.md).

## Why graph construction is not an ingest stage

Entity resolution and community detection are only meaningful over the whole
corpus, and `IngestPipeline` deliberately holds one document's chunks at a time —
materialising a 4M-chunk corpus to cluster it is what the backpressure exists to
prevent. `GraphBuilder` also owns its own writes and returns a build report
rather than chunks, so there is no shape in which it could be a per-document
enrichment.

It used to be listed as one anyway, and silently failed to construct on every
run because nothing passed it a graph store. `graph.enabled` now records what it
needs on `IngestReport.warnings` instead:

```bash
ragorc ingest ./docs            # report.warnings names the second pass
ragorc graph build              # reads the chunks back and builds the graph
```

Or in process, which is what the command does:

```python
builder = GraphBuilder(llm, graph_store, settings=settings)
await builder.build(chunks)     # see examples/04_graphrag.py
```

`ragorc graph build` reads the chunks back out of the collection rather than
re-reading your sources, so it costs no loading and no embedding — only the
extraction calls. It exits non-zero when chunks were read and no entity came out
of them, because an exhausted API key and a corpus with no entities produce
identical counts.

The same reasoning applies to a corpus-wide RAPTOR tree; the per-document trees a
streaming ingest can build are what the `raptor` stage produces.

## Key classes

```python
IngestPipeline(*, vector_store=None, relational_store=None, splitter=None,
               dense_embedder=None, sparse_embedder=None, late_embedder=None,
               late_chunker=None, llm=None, validator=None, settings=None)
    async ingest(target) -> IngestReport      # a Document, a path, or an iterable of either
    async close() -> None                     # closes only the stores it created

IngestReport(documents_in, documents_indexed, documents_skipped, documents_rejected,
             documents_duplicate, documents_failed, documents_empty, chunks_created,
             vectors_written, points_in_store, strategy, total_ms, timings_ms, usage,
             rejected, failed, warnings)
    .cost_usd  .skip_rate  .summary()

```

**`vectors_written` vs `points_in_store`.** The first counts vectors *sent*, the
second is what the collection reports holding after the run's one flush. They are
separate numbers because Qdrant and Postgres share no transaction and
`qdrant.wait_on_upsert` is off, so the only honest answer to "did it land?" comes
from asking the store. Into an empty collection they should agree; on a re-ingest
`points_in_store` is lower, because points are overwritten by id while
`vectors_written` counts writes. `points_in_store` is `None` when the store could
not be asked, and the failure is recorded in `warnings` rather than passed off as
zero.

**Write order is a constraint, not a preference.** `chunks.document_id` is a
foreign key to `documents(id)` with `ON DELETE CASCADE`, so the order is: per
window, build chunks → purge the stale ones → write document rows → write chunks;
then once, at the end of the run, write everything the enrichment stages wanted
persisted (parents, summarised sources).

Parent-document and multi-representation writes are buffered during processing and
flushed last because three things have to be true at once, and that is the only
point where they all are: the document rows exist (foreign key), no purge is still
coming (cascade), and the leaves are already written. The third matters because a
parent *is* a chunk row, and chunk rows are the marker `_existing_checksums` reads
as "this document is ingested" — flush them before the leaves and a run that dies
on the vector write leaves parents behind, so the retry skips the document with
none of its searchable content indexed.

The buffer belongs to the run, not the pipeline: `ingest()` discards anything left
in it on the way out, because the server reuses one pipeline across requests and a
failed run's writes must not land in the next one.

```
build_splitter(name=None, *, embedder=None, settings=None) -> Splitter
load(source, **kw) -> list[Document]          # TextLoader, MarkdownLoader, PDFLoader, ...
GraphBuilder(llm, store: Neo4jStore, *, embedder=None, settings=None, ...)
    async build(chunks) -> GraphBuildReport
```

Optional submodules resolve on first attribute access, because each needs an extra
that may not be installed: `ragorc.index.raptor`, `.graph`, `.multirep`, `.colbert`.

## Six splitters, and how to choose

| Name | Use it for |
|---|---|
`semantic` (default) | cuts where the meaning changes; one embedding batch per document as a boundary detector |
`recursive` | no model, no network — the universal fallback and the right answer for unknown content |
`token` | the only strategy whose size guarantee is in the embedder's own unit; CJK and code |
`markdown` | keeps the authored outline: heading path per chunk, fences and tables never cut |
`code` | splits on definition boundaries, carries the enclosing class and imports |
`sentence_window` | one sentence per chunk for precision, surrounding sentences stored for generation |

Every splitter returns **boundaries, not vectors**. That is not tidiness: a splitter
that embedded as it split would foreclose late chunking.

## The three things the pipeline does that are easy to get wrong

**The checksum skip.** Before anything is split or embedded, the pipeline asks the
relational store for the stored `(id, checksum)` of the documents it is about to
write, in one batched query per few hundred documents, and drops the ones that did
not move. A nightly re-ingest of 100k documents with a 1% change rate costs 1k
documents of embedding instead of 100k.

**Nothing accumulates.** Documents are processed `max_concurrent_documents` at a
time and their chunks stream to the stores in `batch_size` batches. 4M chunks with
text plus vectors is tens of gigabytes if you materialize the list first — and one
all-or-nothing write at the end. Streaming bounds live memory independent of corpus
size, and every landed batch is durable.

**Two failure rules, opposite directions.** A *document* failure is recorded and the
run continues: one corrupt PDF must not discard 9,999 good files, and deterministic
ids mean a later run retries exactly that document. A *store* failure aborts: if
nothing can be written, continuing to embed spends money producing vectors with
nowhere to go.

The purge is mandatory, not hygiene: `chunk_id` folds content in, so an edited
document's chunks get *new* ids and an upsert alone would leave the previous version
indexed, retrievable, quotable and wrong.

## Usage

```python
from ragorc.index import IngestPipeline

pipeline = IngestPipeline(dense_embedder=dense, sparse_embedder=sparse)  # share the sessions
report = await pipeline.ingest("examples/corpus")
print(report.summary())  # strategy, skip_rate, chunks, vectors, cost_usd, timings_ms
await pipeline.close()
```

Runs above `BULK_LOAD_MIN_DOCUMENTS` (64) write inside Qdrant's bulk-load mode, which
defers HNSW construction so ingest is an append and each graph is built once.

## Settings

| Setting | Effect |
|---|---|
`indexing.splitter` · `chunk_size` · `chunk_overlap` · `min_chunk_size` · `max_chunk_size` | overlap costs storage but prevents an answer split across a boundary |
`indexing.semantic_breakpoint` · `semantic_threshold` · `semantic_buffer_size` | `gradient` finds boundaries in uniformly dense prose that `percentile` misses |
`indexing.chunking_strategy` | `auto` → LATE → CONTEXTUAL → EARLY (ADR-0002) |
`indexing.contextual_enabled` · `contextual_prefix_tokens` | one LLM call per chunk; opt-in |
`indexing.parent_document_enabled` · `summary_index_enabled` · `dense_x_enabled` | multi-representation stages |
`indexing.raptor_*` | UMAP → soft GMM → summarize → recurse; needs `ragorc[raptor]` |
`indexing.batch_size` · `max_concurrent_documents` | the backpressure window |
`indexing.skip_unchanged` | the checksum skip — the single biggest re-ingest saving |
`indexing.dedupe_chunks` · `dedupe_threshold` | drop duplicate chunks before writing |
`graph.enabled` and the `graph.*` block | GraphRAG construction; needs `ragorc[graphrag]` |
`qdrant.indexing_threshold` · `upsert_batch_size` · `parallel_upserts` | ingest write throughput |
