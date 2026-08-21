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

```python
report = await pipeline.ingest("./docs")
# report.warnings names the second pass when graph.enabled is set

builder = GraphBuilder(llm, graph_store, settings=settings)
await builder.build(chunks)      # see examples/04_graphrag.py
```

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
             documents_duplicate, documents_failed, chunks_created, vectors_written,
             strategy, total_ms, timings_ms, usage, rejected, failed, warnings)
    .cost_usd  .skip_rate  .summary()

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
