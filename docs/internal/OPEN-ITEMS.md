# Open items

Known gaps, each verified against the code at the commit that added its row. This
file exists because the working notes it replaces were machine-local, unlinked and
stale, so the same findings were rediscovered from scratch — twice. If you close
one, delete the row.

Nothing here is a secret or a live vulnerability. The security findings from
earlier audits are fixed and covered by tests in `tests/unit/test_security_guards.py`
and `tests/unit/test_guard_properties.py`.

## 1. Closed by deletion: the confidence-gated model cascade

`ModelRouter.should_escalate` and the two `cost.cascade_*` settings are removed.
The decision and its reasoning are recorded in ADR-0005 under "Rejected:
confidence-gated escalation on the answer path"; in short:

* There was no confidence signal to gate on. `Answer.confidence` is a literal
  `1.0` on the plain and JSON-citation paths, and with `check_groundedness` on
  (the default) it reduces to the groundedness score — so the gate would have
  measured grounding, not model uncertainty.
* That gate already exists. Abstention fires below `groundedness_threshold`
  (0.70) against a cascade threshold of 0.75, leaving a five-point band; placed
  after abstention it is nearly inert, placed before it, it re-runs the most
  expensive call in the pipeline on every ungrounded answer and forecloses
  streaming.
* A larger model is the weakest response to poor grounding, which usually means
  retrieval failed. CRAG, RRR and abstention all act on the cause instead.

`strong_model` stays: `model_for(..., escalate=True)` still reaches it from the
Text-to-SQL guard repair, which escalates unconditionally.

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

## 11b. Closed: the unit suite keeps its own no-infrastructure promise

`tests/fakes` states the design goal — "the entire unit suite runs with no
network, no containers, no API keys and no model downloads" — and two tests had
quietly stopped keeping it. `_stub_pipeline` passed no stores, so `_prepare` built
a real `QdrantStore` and `PostgresStore` from the default DSNs; they passed only
on a machine with the compose stack up and a `.env` pointing at it. On a clean
clone they did not fail with a useful message, they spent 30 seconds each timing
out. They inject fakes now.

The regression is prevented by a direct assertion on the helper
(`test_the_offline_pipeline_helper_injects_its_stores`), because that is where the
mistake is made.

There is also an autouse socket guard in `conftest`, but be clear about its reach:
it patches `socket.socket.connect`, so it only sees clients that connect from
Python. Measured against this stack, an HTTP Qdrant client is caught while gRPC —
the Qdrant default — and psycopg both connect inside C and bypass it entirely. It
would *not* have caught these two tests. It is kept because it costs nothing
measurable (11.7s vs 11.9s for the suite) and does catch the case the assertion
cannot: a unit test reaching a hosted API.

## 11c. Closed: a failed ingest does not leave writes for the next run

`_deferred` is instance state and `_LinearEngine` reuses one `IngestPipeline` for
every `POST /ingest`, so a run that died after a stage buffered its docstore
writes left them queued for the *next* run to flush against an unrelated document
set. `ingest()` now owns the buffer's lifetime in a `finally`.

The same test showed the flush was in the wrong place: it ran after
`_write_documents` but *before* the leaf chunks, and a parent is a chunk row —
which is the marker `_existing_checksums` reads as "this document is ingested". A
run dying on the vector write left parents behind, so the retry skipped the
document with none of its searchable content indexed. It flushes last now, the
only point where the rows exist, no purge is coming, and the leaves are in.

## 11d. Closed: the ranking and evaluation mathematics

A pass over the numbers, which no earlier audit had checked. A wrong formula here
is invisible — nothing crashes, results are merely worse or a benchmark lies.

* **`max_fusion` inverted its own weights on negative scores.** It multiplied raw
  scores by weights, and for a negative score a smaller weight moves it *up*:
  logits -5.0 at weight 1.0 versus -9.0 at weight 0.1 ranked the down-weighted
  junk first, and weight 0.0 — "disable this source" — produced -0.0, the largest
  value in an all-negative column. Cross-encoder logits are routinely negative
  (the shipped reranker prints -0.442 and -10.273 on the example corpus) and
  `FusionMethod.MAX` is reachable config with non-uniform default weights. A zero
  weight now drops the list; negative scores with non-uniform weights raise and
  name `fusion='relative'`. Shift-by-minimum was considered and rejected: with one
  element per list the minimum *is* the element, so weights would silently stop
  meaning anything.
* **Document-level retrieval metrics were computed and thrown away.** The shipped
  dataset has 20 cases, 0 with chunk labels and 18 with a source document, so
  `make eval` measured retrieval quality for 18 cases and then omitted it from
  `to_dict()`, excluded it from `series_names()` (so `--compare` could never A/B
  it), and printed "no chunk labels: retrieval metrics not computed" — which was
  false. All three surfaces carry it now.
* **`ragorc eval --compare` crashed.** `paired_bootstrap` correctly returns `None`
  for every statistic when two runs share no scored case; the table formatted
  `None` with `:.3f` and the TypeError took down the whole render, including the
  metrics that did compare. Undefined now prints as an em dash, which is not the
  same as zero.
* **The "95% CI" column was hard-zero.** `to_dict()` emits `ci` as a pair; the
  table read `ci_low`/`ci_high`, which never existed, so `.get(..., 0.0)` pinned
  the one number that says whether a difference is real to `+0.000 to +0.000`.
* **Lexical metrics were ASCII-only.** `_WORD` was `[a-z0-9]+`, so two identical
  Russian answers scored 0.0, `café` tokenized to `caf` and `Müller` to `m` +
  `ller`. Widened to Unicode word runs, with kana/Hangul/Han split per character
  since those scripts carry no spaces and would otherwise collapse to exact match.
* **The overflow decision did not price the framing.** `decide_strategy` compared
  bodies against the window while the packer also charges each passage its header,
  wrapper and separator — 23 tokens each on the default settings. A set that
  overflowed once framed was reported as fitting and the packer then dropped the
  tail. `ContextPacker.overhead()` is public now and the budgeter charges it.
* **`ContextSummarizer._clip` destroyed the caller's chunk**, truncating in place
  and returning the same object — the same aliasing hazard as
  `GraphLocalRetriever._annotate`. It returns a copy.
* Error panels ate the extra they told you to install: Rich reads `[...]` as a
  style tag, so `pip install 'ragorc[server]'` rendered as `pip install 'ragorc'`.
  Data is escaped before it reaches the markup parser now.

## 11e. Closed: the per-source budget split is enforced as a floor

`ContextPacker` reserves each contributing store its share before the open
density competition and hands anything unused to the free pass;
`AnswerGenerator` passes `plan.per_source`, and that wiring has its own test —
the packer's argument is optional, so a generator that forgot to pass it would
have restored the original bug with every packer test still green. That is how
the field went dead the first time.

A floor rather than a cap, deliberately. Shares are renormalized over the stores
that actually returned candidates, so a single-source query gives that source the
whole window and packs byte-identically to before. A hard cap would have made the
common case worse: most queries are single-source, and holding 45% of the window
for stores with nothing to say trades good evidence for none.

`_SOURCE_BUCKETS` maps each `RetrievalSource` onto a share name, because a store
speaks through several sources — Postgres as both `SQL` and `FULLTEXT`, the graph
as four. Anything unmapped falls to `vector`, the largest share, so a new source
is never starved by an oversight in that table.

## 11f. Verified clean: packaging and the clean-clone path

Checked for the first time and found nothing wrong. The wheel builds with all 154
modules and `py.typed`; it installs into a fresh virtualenv from base dependencies
alone; `import ragorc` and the `ragorc` console script both work. Every
third-party import is either declared in an extra or guarded with a working
fallback (`yaml`, `grpc`, `onnxruntime`), and the ones that are not declared
cannot be missing when their parent is (`tokenizers`, `starlette`).

## 11g. Closed: mechanisms that were reached but did not run

Twelve findings from the pipeline/graph/server audit, all reproduced before being
fixed. They share one shape, which is this codebase's characteristic defect and
worth naming: **the thing is written, wired, documented, reachable — and does not
do its job.** Every one of them was invisible to a suite that tests units in
isolation, and in six cases the fix was verified by mutating the *call site*
rather than the function body.

Security:

* `wrap_untrusted` escaped one exact string, so `</UNTRUSTED_DOCUMENT>`,
  `</untrusted_document >` and a forged *opening* tag all passed through. The
  Hypothesis property that was supposed to cover it counted the lowercase
  substring, so it held while the defence did not.
* The passage provenance line was rendered *above* the isolation fence. `source`
  comes from `metadata` on the ingest request, so that was attacker-controlled
  text in the region the system prompt calls instructions.
* `tenant_id` was never bound to a principal. `require_tenant` checks that a
  request *names* a tenant; the field is on the request body. Any authenticated
  caller could read or write any tenant. `server.api_key_tenants` binds it, and
  `/health` warns when isolation is on and nothing is bound.
* The injection scanner never saw an ingested document — only questions and web
  results, i.e. everything except the attack path its own module docstring opens
  by describing.
* A multipart upload was bounded three times and by nothing: the middleware reads
  `Content-Length` (absent on a chunked request), Starlette's `max_part_size`
  applies only where `file is None`, and `_staged_uploads` compared the total
  *after* `await value.read()` had materialized the part.

Behaviour:

* The agentic graph's collect step was `nodes.fuse`, which rebuilds the ranking
  from `per_store` — where CRAG publishes its *pre-grading* candidates. On
  `INCORRECT`, CRAG returned `[]` and the generator was handed all five
  documents. The abstention signal was computed, billed and discarded.
* `check_sufficiency` read `evidence`, built from `retrieval`, which `hop` does
  not write. So it re-judged the first retrieval on every iteration and
  `multihop_stop_on_sufficient` could not fire after hop 0.
* `_retrieve_for_stream` ran one fixed five-node list for every pipeline.
  Streaming `graphrag` performed no traversal at all; streaming `multihop` took
  no hops.
* Entity traversal was untyped, so a two-hop expansion walked through `Chunk` and
  `Community` nodes and returned co-occurrence as an extracted relationship —
  including from `paths()`, which *is* the answer to "how is A related to B".
* Community nodes were never pruned, so a changed document left its old
  community's report in global search forever.
* Local search's similarity term never fired: `query.dense` is set only by
  `QdrantStore._prepare`, and the GraphRAG path does not go through the vector
  store. The shipped ranking was two thirds of the documented one.
* `/health` reported Qdrant `unavailable` and the service `degraded` whenever
  `enforce_tenant_isolation` was on, because the probe was a tenant-scoped
  `count`.

## 11h. Closed: two cross-tenant leaks, and state shared by concurrent requests

Round nine, with four lenses not previously applied: concurrency, cache tenancy,
the relational store, and resource lifecycle. Nineteen findings, sixteen
surviving adversarial verification, three refuted. Both leaks were reproduced
end to end against the live stack.

**Cross-tenant reads.** The knowledge graph had no concept of a tenant at all —
`grep -c tenant ragorc/stores/neo4j/store.py` returned 0 — and the by-id chunk
reads in both stores were unscoped, so a query scoped to one tenant returned
another's chunk verbatim. The graph's *own* output leaked too, with forged
provenance: a subgraph chunk stamped with the querying tenant's id whose body was
a third party's private entity and description, with none of that tenant's chunks
even indexed. Separately, `RAGPipeline.query` consulted the semantic cache
*before* `require_tenant` ran (the guard lives in the graph's first node), and
with no tenant resolvable the cache drops its tenant predicate — so a request
that should have been refused returned another tenant's cached answer.

The same asymmetry `require_generated_query_isolation` was written for, one path
over: that guard covers Cypher an LLM *wrote*, and these legs run parameterized
traversals. Its docstring even names the mechanism — "`tenant_id` reached the SQL
path only to *stamp the resulting chunk*, never to filter the query" — which is
exactly what `graph.py:779` and `:940` do with it.

**Ceilings that were read and not enforced.** `CostLedger.check()` reads what has
been *recorded*, and a call records after its round trip, so all N pre-checks in
a fan-out passed against a zero ledger: 40 provider requests against a five-call
"hard ceiling". Replaced with a reservation. Worth recording that the first
measurement of this said the ceiling held — an artifact of a mock transport that
returned without yielding, which serialized the interleaving the bug needs.

**Data loss on retry.** `_select_changed` skips a document whose stored checksum
matches, so the checksum is the "indexed" marker — and it was written before any
vector existed. With the vector store down, run 1 raised with the row committed
and run 2 reported `skipped=1, indexed=0` over an index that never received the
document. A comment in `_run` described this failure and blamed chunk rows;
`_existing_checksums` reads the documents table, so the fix it justified narrowed
the window and left the cause.

**State shared by concurrent requests.** The docstore buffer was an attribute on
a pipeline the server reuses, so concurrent ingests flushed or discarded each
other's parents. `CircuitBreaker`'s documented "single probe" was every caller in
the window. `ingest(**kwargs)`, documented as "this run only", assigned
`self.settings`. `GraphGlobalRetriever` accumulated bills across requests,
reporting `[3, 6, 9]` calls for identical work.

**Configured and not connected.** The relational store failed open when told to
isolate. The Qdrant client cache's dead-loop eviction could not fire, because the
key contains `id(loop)`. `cache_rerank` reached no backend under `RAGPipeline`.
The HTTP cache stored degraded answers, omitted the pipeline from its key, and
kept full chunk bodies.

Six of these were only caught by mutating a *call site* rather than a function
body — the gap named in 11g, still the most productive place to look.

## 11i. Closed: round ten — the lifecycle, and what leaves the process

Four lenses never previously applied: data lifecycle, the answer path's outbound
guards, secret egress, and the public surface with the thinnest tests. Two
findings were reproduced end to end before being fixed.

**A foreign retrieval leg opted out of tenant isolation.**
`from_langchain_retriever` makes someone else's retriever one leg of an
ensemble, and it sat outside every mechanism in `docs/security.md`: it queries
its own store, so no filter of ours reaches it, and `from_langchain_documents`
reads `tenant_id` out of the foreign document's metadata — the retriever's
claim, not a fact. With `enforce_tenant_isolation` on, a query scoped to
`globex` came back with `id=acme-secret tenant_id='acme'`, and the wrapped
retriever had been handed `('what was revenue?', None)`. The security doc calls
that configuration "self-consistent — every query names a tenant, every store
filter carries it"; a foreign leg falsified it silently. Third instance of the
asymmetry behind `require_generated_query_isolation` and
`require_graph_tenant_isolation`, and the first with a defensible middle ground:
a label that is present can be checked and one that is absent can be dropped.

**A corrected document kept being answered from the old one.** The semantic
cache had no invalidation and *could not have had one*: the payload was
question, answer, timestamp, tenant and scope, so there was nothing to
invalidate by, and the only removal was `clear()` for every tenant at once. A
policy edited from "30 days" to "7 days" and re-ingested was still answered "30
days" from a cache hit carrying no citations. The module already contained the
argument for the fix, applied to the opposite case — it refuses to cache an
abstention because "it is a statement about the index at one moment" — and a
positive answer is the same statement with the sign flipped.

**A documented check that did not exist.** `validate/output.py` promised "the
answer must not contain PII that was redacted upstream, or the delimiters of our
own prompt scaffolding". `grep -ci pii` on that module returned 1: the
docstring. The half that existed only *detected* — `scaffold_leak` was set, read
nowhere, absent from the metadata, and the markup reached the reader. And
`PIIRedactor` ran on the inbound question and nowhere else, so redaction
protected the text the caller wrote and not the answer assembled from retrieved
documents, which is where the corpus's personal data is.

**Streaming skipped the guards that could have run.** Justified by "groundedness
can only be judged once the answer is complete" — true of groundedness, not of a
regex over emitted text. Both leakage checks now run on the stream behind a
96-character hold-back. Separately `_audit.answered` lives in `_finish`, which
only `query` calls, so the trail recorded that a streamed question was asked and
never that it was answered or what it cost.

**Nothing could be deleted.** No route, no CLI command, no graph delete at all.
See section 16.

**Secret egress.** `postgres.dsn` was a plain `str` while its two sibling
credentials were `SecretStr`, so `repr`, `model_dump` and `model_dump_json`
printed an embedded password that `summary()` was careful to mask.
`redact_identifiers` had no pattern for a credential inside a URL — though the
obvious exploit does not exist, and that is recorded in section 14.

**The unit suite was not hermetic.** Nine tests passed or failed depending on
whether a machine-local temp cache was warm: tiktoken and fastembed both default
into `tempfile.gettempdir()`, which macOS purges. A full green run, then nine
failures an hour later with no change in between, all reported as "unit test
opened a real connection" — a message naming neither the cause nor the fix.
Both caches now live under the repo's `.cache/` and two session-scoped fixtures
warm them ahead of the function-scoped network guard. Verified by blocking
AF_INET before pytest is imported: 943 tests, zero network.

Mutation verification found four tests that passed for the wrong reason, which
is the highest yield yet from that discipline: a tenant read only from the query
rather than the ensemble's kwarg; a probe swallowed by the very `except` it was
testing around; a split-delta test that never overflowed the buffer it existed
to exercise; and a redundancy that made a mutation genuinely invisible, where
the fix was to pin the property the redundancy protects rather than the
redundancy itself.

## 11j. Closed: round eleven — the loader, late chunking, and last round's work

Fourteen findings across three areas. Two of the three had never been audited;
the third was the previous round's own output, which turned out to be as prone
to this codebase's characteristic defect as the code it fixed.

**The loader was the worst single module found in eleven rounds.** 1400 lines,
no test file, and the place untrusted files enter. Six defects, two of which end
in an empty corpus and no error — the failure nobody notices, because it looks
exactly like ingesting an empty directory. One renamed `.docx` aborted a
twenty-file walk and recorded no failure, because the catch tuple missed
`PackageNotFoundError` and `FileDataError` — the two exceptions the class
docstring names when it promises in bold that this cannot happen. And any ingest
root *under* a directory called `build`, `env`, `dist`, `target` or `.cache`
loaded nothing, because the skip list was matched against the absolute path and
so vetoed ancestors above the root the caller named.

The other four: every multipart upload minted a fresh `document_id` (identity
came from the ephemeral staging path, so re-uploading duplicated forever and the
checksum skip could never fire); `IngestRequest.recursive` and `metadata` were
parsed, validated and dropped before any loader was built; `JSONLoader` built
documents on the event loop (676 ms stall on 16 MB, against its JSONL sibling's
52 ms); and `CSVLoader` used `splitlines()`, which deletes newlines inside
quoted cells and splits on six characters CSV does not treat as terminators.

**Late chunking was pooling through the model ADR-0002 rejects.** Three call
sites, all wiring rather than mathematics. `_late_chunker()` passed the ColBERT
embedder as the token source — the substitution the ADR records as already tried
and removed, "a different model in a different space at a different width" —
while `build_late_chunking_embedder`, which wires the dense model in correctly,
had no in-library caller at all. The route in was one the ADR does not mention:
enabling ColBERT *reranking* switched the *chunking strategy*, because
`FastEmbedLateInteraction` exposes token output. The disagreement was visible
without running anything, `supports_late_chunking()` False against
`resolve_strategy()` LATE, and the test the ADR says covers it passes the dense
embedder rather than the pipeline's chunker.

Alongside it, `query_side` — the variable whose whole purpose is to name the
object queries must use — was computed and read in one dead branch, so under
LATE the store went on embedding queries with the dense provider. And RAPTOR
summaries, multi-representation units and parent documents were embedded by the
dense provider too, landing in a different space from the leaves they share a
collection with.

**Round ten's own output.** A delete that removed nothing reported
`documents=1, complete=True, errors={}` — including when tenant isolation had
correctly refused it, which is a false confirmation for the one use case the
feature exists for. The CLI printed request counts under a column headed
"removed". The streaming leak filter cost ~310 ms of time-to-first-token, by
default, on the path documented as the one to use when latency matters. And
`ragorc delete` pointed at `ragorc query --json` to find an id, which needs a
model call — nothing could list what was indexed.

**Mutation verification found seven tests that passed for the wrong reason**,
the highest count yet, and six of the seven were one mistake: testing a helper
instead of the call site that uses it. Reverting `_ensure_stores` so it never
called `_bind_query_side`, and reverting both stage sites to `_dense()`, all
left the suite green because the helpers were still correct and still unreached.
That is the gap this codebase is named for, found by mutation for the third
round running. The other one overrode the very method under test, so the handler
it was checking never ran.

Two findings were nearly reported wrongly in the *other* direction: the
dataclass performance claim measured 1.62x under an unfair six-field comparison
and 2.87–3.25x field-for-field, and the ColBERT finding was nearly refuted
because `_late_chunker()` reads an injected attribute that is `None` on a bare
`IngestPipeline` and only becomes ColBERT once the builder wires it.

## 11k. Closed: round twelve — the graph write side, the scoring inputs, and a live regression

Nine findings. Four share one shape and two share a single cause, which is the
most useful thing this round produced.

**One hardcoded argument disabled two documented optimizations.**
`QdrantStore.search` passed `with_vectors=False`, so no chunk on any shipped
retrieval path carried a dense vector — and the two stages that read
`chunk.dense` both degrade to something indistinguishable from working. MMR
returned `chunks[:k]` on every query (verified against live Qdrant: every hit
`dense is None`, output byte-identical to relevance order), and
`EmbeddingFilterCompressor` re-embedded all 30 candidates at ~7 s a call against
the 2 µs its own matmul takes, while documenting the opposite.

Compounding it, `report.diversity_dropped` was assigned inside the
`mmr_enabled` branch whichever way the branch went, so plain truncation was
logged as `diversity=17`. That is what would have stopped anyone noticing.
Vectors are now requested when a stage will read them, by name so the sparse and
ColBERT vectors do not come too, and attribution is decided on the input —
MMR's fallback is indistinguishable from its success by the return value alone.

**Retrieval fetched ten and reranked ten.** `_fetch_k` exists for this and its
docstring states the failure verbatim; `store_node` called it and
`nodes.retrieve` read `state["top_k"]`, so `naive`, `self_rag` and `multihop`
handed the reranker ten candidates out of ten while `adaptive` and `graphrag`
gave it fifty. Same class, two call sites.

**Entity resolution merged what its own guard would have refused.**
`normalized_form` stripped leading articles in five languages and stage 2 unions
on `(type, normalized_form)` unconditionally — `_cluster_conflict` is reached
only from stage 3. `El Salvador` merged with `Salvador`, one node carrying both
descriptions and the graph asserting `El Salvador -[CAPITAL_OF]-> Bahia`. Stage
3 scored the pair 0.8167 against a 0.92 threshold. English articles only now:
an English leading "the" is usually droppable from a proper noun, a Romance or
German article usually is the proper noun's first word.

**And stage 3 never ran anyway on the shipped path.** `ragorc graph build`
omitted `embedder=`, so `_merge_by_embedding` returned `{}` on its first line,
`Entity.embedding` was always None, and `docs/operations.md` advised lowering a
`resolution_threshold` that did nothing. The engine had held a built dense
embedder since `build()` and the example passes it.

Also: a failed gleaning pass propagated out of `_extract_chunk`, whose caller
treats any exception as "this chunk failed", so a second-pass timeout discarded
a *successful* first extraction and the usage already paid for it.

**Round eleven's own output produced a live regression.** Adding `source_root=`
to the keyword set the directory walk passes each loader broke CSV, JSON, JSONL
and PDF on *every* ingest — three subclasses override `__init__` without
forwarding it — and the broad `except Exception` added in the same commit
recorded each as a per-file warning. A four-file directory reported success
having indexed one. Two changes that were each correct and together were not,
found by parameterizing a test that had used a single `.md` file.

The other two: the upload-identity fix covered six of nine label sites, because
`_read` bypassed `_label`; and `_LinearEngine.ingest` assigned loader options on
a shared pipeline and restored them in a `finally`, which under concurrency left
the first run's options behind permanently.

Mutation verification: 24 mutations, all caught, across four scripts. One
mutation had to be rewritten because deleting a line broke the module rather
than its behaviour — a syntax error is not a test of anything.

## 11l. Closed: round thirteen — the measuring instrument, and HyDE

Seven findings. Two were critical and both were in the harness every other
judgement about this library is made with; one was a whole feature with a
finished producer, a finished consumer contract and no wire between them.

**`ragorc eval --compare` crashed on the shipped dataset.** Round 11d taught
`series_names()` to advertise the document-level series so `--compare` could A/B
them and never taught `series()` to resolve them, so the first `doc_*` name
raised `KeyError` — with an error message listing the name it said it did not
have — and took the whole comparison down, including the metrics that would have
compared. On the shipped dataset those are the only retrieval signal there is.
The series pairs against a new `document_ids` field, because its vectors cover
the document-labelled subset: 18 of 20, so pairing against `scored_ids` raises
on `strict=True` for exactly the datasets it exists to grade.

**And when it did not crash it measured one pipeline twice.** `scope_key` takes
`pipeline=` and its docstring names the consequence verbatim — "whichever ran
first answered for both — so a benchmark comparing two pipelines measured one of
them twice". The HTTP layer passed it; `RAGPipeline._cache_get`/`_cache_set`,
the cache the eval harness actually runs through, did not and had no parameter
to forward. The candidate scored $0.00, zero LLM calls and 0.000 retrieval,
because a cached payload deliberately carries no chunks — and `cache_hit_rate`
could not expose it, since a semantic hit records no call at all.

**HyDE never embedded its hypothetical document.** The module opens with "HyDE:
embed a hypothetical answer instead of the question" and did the opposite: the
translator billed an LLM call, stored the document on `Query.hypothetical`, left
`Query.dense` as `None`, and the store embedded `query.text`. Both halves of the
contract were already written — `retrieve/vector.py` says twice that
`query.dense` "is how HyDE injects the embedding of a hypothetical document",
and `embed_for_search` implements the blend correctly with a docstring saying
"Called by the retriever instead of the plain query embedding". `grep -rn
embed_for_search` found the definition and two unit tests. Tested in isolation
at both ends, which is why it survived thirteen rounds.

**A rewrite carried the failed question forward.** `_respell` drops the vectors
"because they belong to the old text" and kept the variants, which a translator
produced *from the question that failed* — so with three of them the rewritten
question got one slot in four of the RRF window. It also kept
`metadata["hyde_documents"]`, which was harmless while nothing read that key and
stopped being harmless in the commit immediately before.

**graphrag paid for variants nothing on its path reads**, since `classify` sends
it to Neo4j traversals that read `query.text` and no store retriever is reached.
The audit made the same claim for `multihop` and it is wrong — `multihop` binds
`nodes.retrieve`, which does expand `all_texts`.

**And round twelve's fan-out fix over-applied.** Widening `nodes.retrieve` to
`fetch_k` was right for the two graphs that rerank and wrong for `naive`, the
one that does not: its generator was handed fifty passages where `top_k` — "What
the generator sees" — promises ten.

Mutation verification: 24 mutations across five scripts, all caught, with five
tests initially passing for the wrong reason. Three of those five were the same
mistake in different clothes — asserting on a helper, a source string, or a
parameter, rather than on what the caller does. One of them was defeated by a
docstring I had just written containing the word the test was grepping for.

## 11m. Closed: round fourteen — query construction and the loop decisions

Eight findings across two areas neither previously audited: the thing the SQL and
Cypher guards guard, and the decisions inside the feedback loops whose structure
round eight checked.

**Three renderers the stores made unreachable, for one shared reason.** Both
stores normalize result values for JSON-safety and that normalization runs
*before* the construct module's renderers. Each store's conversion is right for
its own purpose and wrong as a preprocessing step for code written to handle raw
types.

The worst returned the wrong rows. `execute_readonly` read with `dict_row`, and
a dict cannot hold two columns of the same name — so `SELECT * FROM a JOIN b`,
the commonest shape a text-to-SQL model produces, kept the *last* of each
duplicate and handed the generator a three-column table describing the
right-hand table's rows under the left-hand table's question. No error, and
nothing in the row count to notice.

The second lost digits: `_json_safe` called `float()` on NUMERIC, so `12.50`
printed as `12.5` and `1234567890123456789.99` as `1.2345678901234568e+18`, in
the one chunk class the generator is instructed to reproduce verbatim.
`text_to_sql._cell` had a Decimal branch documented for exactly this and the two
comments contradicted each other outright — the store won by position.

The third was the graph verbalizer. `to_chunks` exists because "handed to a model
as a repr, they are unreadable noise", and `Neo4jStore._serialize` flattens
`Node`/`Relationship`/`Path` to dicts first; every detector probes for
*attributes* a dict does not have however many matching keys it carries.

**A width, twice more.** `nodes.hop` and `nodes.bridge` read `state["top_k"]`
where their siblings call `_fetch_k`, so later hops fetched a fifth as wide as
hop 0 and fed the same reranker — on exactly the queries multi-hop exists for.
Third and fourth sites after `nodes.retrieve` (round twelve) and `naive` (round
thirteen), which makes the retrieval width the single most repeated defect in
this codebase. Now pinned per node rather than by banning the pattern, because
`validate` and `rerank` read the state's value correctly: the distinction is
whether the number bounds a search or bounds the answer.

**Two settings that did not do what they said.** `crag_web_fallback=False` left
`nodes.web_search` searching — the flag is read by CRAG's own fallback, the
linear engine and `describe()`, and not by the node — so an operator who
switched it off still sent every AMBIGUOUS query to a third party while
`/health` reported it disabled. And when it *was* on, the web was searched twice:
CRAG's internal fallback plus the graph's node. `generation.allow_abstention=False`
likewise did not stop either abstention path, so the policy's decision was made
and then overwritten.

**And a bill nobody collected.** `SelfRAGResult.usage` and `RRRResult.usage` had
no reader in `_LinearEngine`, so a successful Self-RAG run reported one LLM call
having made six — read by the cost ceiling, the eval `$` column and the metrics
histogram alike.

Mutation verification: 19 mutations across four scripts, all caught, with two
tests initially passing for the wrong reason — both source-grep assertions that a
behavioural mutation walked past, the same weakness round thirteen found.

## 12. Open: an intermittent SIGABRT at interpreter teardown on macOS

Still open, but no longer a mystery. A macOS crash report names the frames::

    onnxruntime_pybind11_state.so
      HttpClientManager::onHttpResponse
      HttpResponseDecoder::handleDecode
      LogManagerImpl::DispatchEvent
      DebugEventSource::DispatchEvent
    libc++  recursive_mutex::lock -> __throw_system_error -> abort

That is ONNX Runtime's bundled Microsoft 1DS telemetry client, on its own worker
thread, dispatching the *response* to a request it had sent — onto a mutex static
destruction had already destroyed. The work is always finished by then; what is
wrong is the exit code, which reads as a failure to CI and files a crash report.

Three things this settled:

* **`ORT_DISABLE_TELEMETRY` does not work on this build.** The crash happened
  with it set. `ragorc/embed/_runtime.py` had described it as the primary
  mitigation.
* **The documented fallback never ran.** `configure_onnx_runtime` guarded its
  `disable_telemetry_events()` call with `if "onnxruntime" in sys.modules`, which
  is false at provider-import time — the only time it was called. Fixed:
  `disable_onnx_telemetry()` runs where importing a FastEmbed class has just
  loaded the extension, and is verified to be reached.
* **Merely importing onnxruntime is enough.** The unit suite builds no model and
  still loads the extension (via `list_supported_models`), so the telemetry
  thread exists in a run that does no inference at all.

**The fix is not proven.** Measured over repeated full-suite runs the crash rate
tracks machine load far more than either mitigation: 8/10 and 6/10 under
sustained back-to-back runs, 6/6 and 0/4 clean on an idle machine, with and
without. Telemetry being off is worth having regardless — this library reads
internal documents — but do not record the abort as fixed.

If it needs to be closed for real, the remaining lever is to skip interpreter
finalization at the CLI entry points: flush the streams and `os._exit(code)`.
That eliminates the whole class of static-destructor race, and it silently
disables `atexit`, coverage writers and buffered output for everyone, so it is
not worth doing until someone is actually blocked.

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

Round ten:

* "Some settings are dead" — 292 fields across 15 groups, every one has a reader
  outside `settings.py`. Checked mechanically, not by eye.
* "An unknown ensemble leg gets weight 0 under weighted fusion" — `_weight_vector`
  defaults to 1.0, with a comment saying so.
* "A tenant can force a cross-tenant purge by colliding a document id" —
  `document_id()` folds the tenant in, and both the server route and the loaders
  derive it. `PostgresStore.delete_document` was still unscoped and has been
  fixed, but as a latent hazard, not a reachable one.
* "`error=str(exc)` leaks the DSN password across 104 log sites" — measured:
  neither psycopg nor the neo4j driver puts the password in a connection error,
  and no call site passes a DSN under an unhinted key. The redaction pattern was
  added anyway; the leak was not real.
* "The LangChain adapter synthesizes a fake similarity score" — it is rank-derived
  and the default fusion is RRF, which reads position.

Round eleven:

* Four documented performance claims, all measured and all holding: dataclass
  construction 2.87-3.25x with 86% less memory (the docs say 2-3x and ~40%, so the
  memory figure is understated), orjson 12x on dumps, float32 4.0 KiB against a
  ~32 KiB list, and `tiktoken.encode_batch` genuinely releasing the GIL.
* Prometheus label cardinality is bounded — two labels, 21 series, no tenant or
  question anywhere near a label.
* Symlink escape from an ingest root, `..` escape from the upload staging
  directory, the 20 MB `MAX_FILE_BYTES` ceiling being unenforced, and silent loss
  on decode failure: all four checked against real files, all clean.
* The late-chunking mathematics: no off-by-one in the char-to-token map, no NaN
  or unit-cosine collision from a zero-token span, no splitter span violations
  under Hypothesis, no RAPTOR non-termination or leaf/summary id collision, and no
  token drop in the long-document windowing. The defects were all in the wiring.

Round twelve:

* The scoring mathematics is sound. MMR's lambda is not inverted and its
  diversity term is computed against the selected set; BM25 has no negative-IDF
  case, no misplaced k1/b and no index/query tokenization drift; a rerank batch
  boundary does not change results and `argpartition` does not scramble order
  within the top-k; `maxsim` handles the padding mask and empty documents.
  Every defect was in what reached these stages.
* Community ids are stable across rebuilds, so `prune_communities` neither
  over-deletes nor orphans; community detection handles empty, single-node and
  disconnected graphs; a gleaning pass does not duplicate pass 1's entities;
  `normalize_relation_type` cannot emit a label the store's guard rejects.
* `near_dupe_threshold=0.93` is not too aggressive for the shipped embedder, and
  `NoiseFilter._threshold` does not fire on Qdrant's server-side RRF scores.
* The broad `except Exception` in `DirectoryLoader._load_one` does not swallow
  security refusals: `DocumentValidator` converts `GuardrailViolation` into
  `ValidationFailed` upstream, deliberately, and says why.
* The adaptive stream hold-back is complete — every PII pattern that spans
  whitespace contains a `_SPANNING` trigger, and every pattern that does not span
  whitespace is covered by holding the trailing run.

Round thirteen:

* A case whose pipeline call raised is *not* counted in the scored denominator,
  so a run that crashes on hard questions does not look better than one that
  answers them badly; `bounded_gather` does not misalign results against cases;
  the significance test is seeded; the per-case cost ledger is not cumulative.
* Translator and logical-router LLM calls all reach the cost ledger.
* `SemanticRouter` embeds exemplars with `embed_documents` and the query with
  `embed_query`, its `min_similarity` does fire, and a `none` route does not send
  a request to `generate` with zero chunks.
* Translated variants do all reach retrieval on the paths that read them, and one
  variant's results do not overwrite another's.
* `multihop` does consume `all_texts` — the audit's claim that it pays for unread
  variants was wrong, and only `graphrag` does.
* Narrowing `_ARTICLES` to English cost no legitimate merge: "The Acme
  Corporation"/"ACME Corp", "The Times"/"Times", "An Post"/"Post" and "A Tribe
  Called Quest"/"Tribe Called Quest" all still merge.
* Adding `truncated` to `NoiseReport.removed` changed no behaviour — every reader
  is a debug log line.

Round fourteen:

* The SQL guard's injected LIMIT cannot be escaped by a generated `UNION`, CTE,
  subquery or OFFSET; a repaired statement does go back through the guard, and the
  last failed attempt is not executed anyway; the repair's strong-model escalation
  is billed to the ledger.
* `self_query` does not let an LLM-invented filter key reach the store.
* CRAG's AMBIGUOUS branch does combine refined corpus results with web results
  rather than letting one win, and knowledge refinement does run and does reach
  the generator.
* Self-RAG's retry budget is bounded, decrements, and re-retrieves rather than
  re-generating from the same context; multi-hop evidence accumulates rather than
  overwriting, and the sufficiency check sees the newest hop (round eight's fix
  held, and the bridge path has the property too).
* No loop can exceed `cost.max_llm_calls_per_query`. RRR uses the rewritten
  query's results *in addition to* the original's.
* Round thirteen's own output holds: the newly-live HyDE path embeds once per
  query rather than once per variant or per store leg; moving `select_graph` above
  the cache did not disturb the tenant guard's precedence; dropping graphrag's
  variants breaks nothing downstream.
* The examples and docs are consistent with the API — every name the six examples
  import resolves, 64 call sites pass no rejected keyword, and every
  `from ragorc import X` in the docs exists.

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


## 16. Delete, and what it deliberately does not do

`RAGPipeline.delete`, `DELETE /documents` and `ragorc delete` remove a document
from Qdrant, Postgres, Neo4j and the answer cache. Three things about it are
decisions rather than implementation:

* **Chunk ids are read before anything is removed.** A `Chunk` node carries an
  id and no document reference, so the vector store is the only thing that knows
  which nodes belong to a document. Delete the vectors first and the graph nodes
  are unreachable, permanently and silently.
* **Entities go only when nothing mentions them.** Not every entity the deleted
  chunks mentioned — they merge on `name`. This is also what makes the graph
  delete tenant-safe in a graph that stores no tenant.
* **A failing store does not abort the rest.** `DeleteReport.errors` is the
  channel; `complete` is the answer. Stopping halfway leaves the caller believing
  the document is gone.

Not done, on purpose:

* **No filtered delete.** One typo from emptying an index, with no tombstone and
  no undo behind it. Ids only, on all three surfaces.
* **No cascade to derived units.** A RAPTOR summary or a multi-representation
  unit built from a deleted leaf keeps its own document id and is deleted by
  naming it. Inferring the parent-child closure would mean walking `parent_id`
  across two stores at delete time, and getting it subtly wrong deletes a summary
  another document still contributes to.
* **No re-detection of communities.** A community whose membership is entirely
  gone is removed; one that lost some members keeps its now-stale summary until
  the next `ragorc graph build`. Re-running detection inside a delete would make
  the cost of removing one document proportional to the corpus.
