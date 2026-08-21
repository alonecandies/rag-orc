# Open items

Known gaps, each verified against the code at the commit that added its row. This
file exists because the working notes it replaces were machine-local, unlinked and
stale, so the same findings were rediscovered from scratch — twice. If you close
one, delete the row.

Nothing here is a secret or a live vulnerability. The security findings from
earlier audits are fixed and covered by tests in `tests/unit/test_security_guards.py`
and `tests/unit/test_guard_properties.py`.

## 1. Open: the confidence-gated model cascade is decided but not wired

`ModelRouter.should_escalate` implements the decision and reads
`cost.cascade_enabled` and `cost.cascade_confidence_threshold`. Nothing calls it.
The only `escalate=True` in the library is `construct/text_to_sql.py`'s guard
repair, which escalates unconditionally rather than on confidence, and
`_DEFAULT_TIERS` maps no `Task` to `ModelTier.STRONG`. So both settings are inert
and no answer is ever re-asked on `strong_model`.

This is the one item here that is a *decision*, not a defect to fix. Wiring it is
a spending change: `cascade_enabled` defaults to `true`, so implementing the gate
starts paying for `strong_model` on every answer scoring below 0.75 the moment
anyone upgrades. The two honest options are to wire it (and probably flip the
default to `false`), or to delete the settings and the method. Until one is
chosen, ADR-0005 and both settings docstrings say plainly that it is not wired,
so nothing in the repo claims otherwise.

## 2. Closed: derived units are searched the same way leaf chunks are

Sparse and ColBERT vectors are added *after* the enrichment stage rather than
before it, so a summary or proposition unit now carries every vector the
collection declares. Before, it carried only the dense vector its own indexer
computed: on a hybrid collection it was findable by vector search and invisible to
BM25 — half-indexed, with no symptom to notice. Dense stays before enrichment
because RAPTOR clusters on the leaf vectors and the summary indexer needs its
sources embedded.

## 3. Closed: the collection is declared from the real embedders

`_prepare` builds the sparse and ColBERT embedders before `_ensure_stores`, because
that is what the collection's named vectors are shaped from. Built lazily on first
use they arrived *after* the schema they decide, so `_colbert_dim()` read `None`
and fell back to ColBERTv2's 128 — correct for the default model, 32 too wide for
`answerai-colbert-small-v1`, whose every upsert the server then rejected. Sparse
has no width but does have `is_lexical`, which picks the IDF modifier.

## 4. Closed: the ingest path uses `ColBERTIndexer`

`_add_colbert` was a five-line inline re-implementation. It dropped token pruning
(so the storage dial bounded nothing on the largest field in the collection),
batching, the dimension check, the skip of chunks already indexed, and `embed_text`
— so ColBERT alone among the three vectors indexed the chunk without its
contextual prefix. The dial is now a real setting,
`embedding.late_interaction_max_tokens`.

## 5. Closed: nothing reaches the chunks table before its document row

`chunks.document_id` is a foreign key to `documents(id)` with `ON DELETE CASCADE`.
Parent-document indexing and the multi-representation stages both persisted chunks
from inside `_process_document`, which is one step before that document's row is
written and two before the stale purge — so on a fresh corpus the insert was
rejected outright (every document counted as failed, after paying for every
embedding), and on a re-ingest it succeeded and was then cascaded away, leaving
children pointing at parents that no longer existed. `_DeferredDocstore` buffers
those writes and `_run` flushes them after the rows and after the purge.

No unit test saw any of it because the doubles had no foreign key.
`tests.fakes.FakeDocumentStore` now enforces it and the cascade.

## 6. Closed: a failed write cancels its sibling instead of detaching it

`bounded_gather` wraps every coroutine in a task and, on failure, cancels the
siblings and awaits them before re-raising. Plain `gather(return_exceptions=False)`
left them running, unawaited and unowned — and the ingest's two chunk writes go to
Qdrant and Postgres, where the Postgres rows are the marker that says a document is
ingested. A Qdrant failure that let the Postgres write commit anyway produced chunk
rows with no vectors, which the next run skips as already done.

## 7. Closed: the promised final flush exists and is called

Two docstrings described "one final waiting flush" that no code performed.
`QdrantStore.flush()` waits for green and reads the exact count back;
`IngestPipeline` calls it once per run and reports it as
`IngestReport.points_in_store`, beside the `vectors_written` count of what was
*sent*.

## 8. Closed: bulk-load mode spans the run

It was entered inside `_run`, which is called once per document window, so a
directory ingest toggled HNSW construction off and on once per 512 documents and
every exit rebuilt the graph over everything written so far and waited for green —
the repeated rebuilding the mode exists to prevent, on a schedule. It is now
entered once, lazily, on the first batch over `BULK_LOAD_MIN_DOCUMENTS`.

## 9. Closed: every answer path honours its token budget

`generation.max_answer_tokens` reached only the plain-text path. The JSON-citation
path and self-consistency ran at the global `llm.max_tokens`, and self-consistency
is the path that multiplies. `structured()` read `max_tokens` with `kwargs.pop`
*inside* its retry loop, so the repair — the attempt most likely to run long — was
the only one uncapped.

## 10. Closed: report counters accumulate across windows

`_validate` runs once per document window and three of its report fields assigned
where every sibling accumulates, so a directory ingest reported the last window's
rejects against every window's list and `documents_in` stopped reconciling.

## 11. Closed: documentation that described code that does not exist

* `install_uvloop()` in `RAGPipeline.create` ran inside an already-running loop,
  where it can only detect that fact, return `False` and warn. Removed; the CLI
  and server already call it at their process entry points, which is the only
  place it works.
* `security/pii.py` told operators to "install Presidio and set
  `provider='presidio'`" — a parameter that never existed, in the module whose job
  is to be trusted. Replaced with what the regex engine does and does not catch.
* ADR-0007 listed retrieval results among what is cached, "each individually
  switchable". There is no retrieval-result cache and no switch; the four that
  exist are `cache_llm`, `cache_embeddings`, `cache_rerank`, `cache_schema`.
* `docs/modules/llm.md` documented `Prompt(name, system, template)`; the field
  order is `(name, template, system)`, so a positional call swapped the system
  prompt and the user template.
* `make bench` ran `ragorc bench` with no arguments, which exits 2 without
  measuring anything. It now names `examples/eval/bench-questions.txt` and takes
  `Q=` for a single question. `--queries` skips `#` comments, as the eval loader
  already did.
* `docs/performance.md` pointed at `make bench` for recall@k and nDCG. `bench` is
  latency only, by its own docstring; `make eval` is the quality one.

## 12. Open: an intermittent SIGABRT at interpreter teardown on macOS

Measured, pre-existing, and not caused by anything in this round: a clean copy of
the previous commit aborts the same way. Three of 38 `ragorc ingest` runs exited
134 with

    libc++abi: terminating due to uncaught exception of type std::__1::system_error:
    recursive_mutex lock failed: Invalid argument

*after* the command had printed its report and closed its stores. It has not
recurred in 44 consecutive runs since, and all three happened while the machine
was also running the test suite — consistent with a teardown race that only loses
under CPU contention. All three were on the semantic-splitter path, which does far
more concurrent ONNX work than the recursive one (0 of 24 there).

The work always completes; what is wrong is the exit code, which reads as a
failure to CI and files a macOS crash report. `ragorc/embed/_runtime.py` exists to
mitigate exactly this and is doing its job — `ORT_DISABLE_TELEMETRY=1` is set
before the runtime imports, and the release hooks are registered. Two gaps in it
*were* found and are fixed: `register_shutdown_hook` was guarded by one global
flag, so only the first module to import ever registered, and two of the three
model caches had no release function at all.

The remaining option, if it recurs, is to skip interpreter finalization at the CLI
entry points — flush the streams and `os._exit(code)` — which eliminates the whole
class of static-destructor race for CLI users. Not done here: it is unverifiable
against a fault that will not reproduce on demand, and it silently disables
`atexit`, coverage writers and buffered output for everyone.

## 13. Closed earlier, kept for the record

* Multi-representation indexing is wired: `summary_index_enabled` and
  `dense_x_enabled` run through `MultiRepresentationIndexer`; the stage builds and
  embeds, the pipeline writes. `parent_document_enabled` is handled at the splitter
  instead, because it splits the document twice and the child is the retrieval
  unit — running it beside the normal split would index every document twice.
* `ragorc graph build` runs the corpus-wide second pass. It remains a second pass
  on purpose: resolution and community detection are only meaningful over the whole
  corpus, and ingest holds one window at a time.
* The doubles model the stores. `FakeVectorStore.get` honours `with_vectors`; the
  multi-hop injection test uses a model that would comply if the instruction
  reached it in the clear; `GraphLocalRetriever._annotate` copies instead of
  mutating the chunk it was handed.
* Multi-hop preserves a caller-supplied vector at hop 0, and `_check_sufficiency`
  stops with the evidence it has rather than discarding it on an unparseable
  response.

## 14. Claims from earlier audits that did not survive checking

Recorded so they are not re-investigated a third time:

* "Enabling CRAG silently turns off reranking" — `builder.py` appends
  `nodes.rerank` whichever retrieval node was chosen, so reranking runs either way.
* "`filters` is silently dropped on the graph path" — `retrieve/graph.py` reads
  `kwargs["filters"]` then `query.filters`. Overstated; one call site does honour it.
* "Over-propagation in `_propagate`" — cannot reach anything the store's hop bound
  did not return, so that mutation surviving is correct behaviour.
* "Ingest embeds every leaf chunk and then discards those vectors under
  `summary_index_enabled`" — the leaf vectors are what the summary indexer's
  sources keep and what RAPTOR clusters on; they are not discarded.

## 15. Deliberately unused, kept on purpose

* `scope_sql_where` / `scope_cypher_where` are exported with no in-library caller.
  `security.generated_query_isolation` recommends PostgreSQL RLS instead —
  string-concatenated tenant predicates are the weaker mechanism, and these exist
  for a deployment that cannot use RLS.
* `nodes.multihop_retrieve` and `MultiHopRetriever` are wired into no shipped
  graph. They are the "multi-hop as one tool" shape, for a caller assembling their
  own graph.
* The `indexer` registry kind is registered and never resolved. `ragorc inspect`
  does not advertise it, so nothing promises otherwise.
* RAG-Fusion is reachable only through `translators=`, not a settings flag. Its
  fusion behaviour is on by default via `retrieval.fusion`.
