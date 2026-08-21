# Open items

Known gaps, each verified against the code at the commit that added its row. This
file exists because the working notes it replaces were machine-local, unlinked and
stale, so the same findings were rediscovered from scratch — twice. If you close
one, delete the row.

Nothing here is a secret or a live vulnerability. The security findings from
earlier audits are fixed and covered by tests in `tests/unit/test_security_guards.py`
and `tests/unit/test_guard_properties.py`.

## 1. Multi-representation indexing is not wired into ingest

`indexing.summary_index_enabled` and `dense_x_enabled` do nothing through
`IngestPipeline`. `_OPTIONAL_STAGES` resolves the `multirep` stage by three factory
names that do not exist, so the stage is dropped — now with a report warning that
says so, rather than silently.

The implementations exist, are unit-tested, and work when driven directly. What is
missing is a decision, not a rename:

* `SummaryIndexer.index` and `PropositionIndexer.index` already embed their derived
  units *and* upsert them, plus write the source chunks to the docstore. Routing
  them through `_enrich`, which then writes whatever the stage returns, double-writes.
  Either the stage stops writing and returns, or `_enrich` stops writing what a
  stage returned. Both are defensible; they are not compatible.
* `ParentDocumentIndexer` performs its own two-level split. It is a chunking
  *mode*, not an enrichment, and wiring it as one would index every document twice.
  It belongs at the splitter, which is a larger change to the write path.

Until then, drive them directly — see `docs/modules/index.md`.

## 2. Corpus-wide graph construction is a manual second pass

`graph.enabled` records what to call on `IngestReport.warnings` and does not build
the graph itself. Entity resolution and community detection are only meaningful
over the whole corpus, and the ingest pipeline deliberately holds one window of
documents at a time. `examples/04_graphrag.py` shows the second pass. A
`ragorc graph build` command would make it a first-class step.

## 3. Seven tests survive their own mutation

Found by an adversarial pass over the coverage suites added in `b3c1690`. All are
cases where a test double is more accommodating than the real store, so the
assertion cannot fail:

* `test_graph_retrieval.py` — `with_vectors=True` can be dropped from the chunk
  read and all tests stay green, because the fake ignores the keyword. Against a
  real Qdrant that silently deletes the 0.35 similarity term.
* `test_graph_retrieval.py` — DRIFT `_merge` can keep the seed-side object instead
  of the graph-annotated one; the tests assert ids and scores but never the merged
  body, so the discarded verbalized-relationship prefix goes unnoticed.
* `test_graph_retrieval.py` — `_propagate`'s own hop bound can be raised (`hops+1`,
  or 99) because the fake store already returns a hop-bounded subgraph. Only the
  store call site is pinned.
* `test_multihop.py` — `test_a_passage_claiming_sufficiency_cannot_end_the_loop`
  passes with prompt isolation removed: the scripted LLM's verdict is fixed, so an
  injected passage has no causal channel to the decision. The safety claim in its
  docstring is untestable with that double.
* `test_multihop.py` — `test_each_hop_searches_for_what_the_previous_hop_found_missing`
  has no unique kill; its assertion is a subset of two other tests.

The fakes need to model the behaviour the assertion depends on: honour
`with_vectors`, and let a scripted LLM's verdict depend on its prompt.

## 4. Two undertested behaviours in multi-hop

Reported by the coverage pass, not asserted as intent because neither is clearly
intended:

* `_hop_query` is applied at hop 0, so a caller-supplied `dense`/`sparse`/`multi`
  vector on the incoming `Query` is dropped on the first retrieval too.
* `_check_sufficiency` has no `try`/`except`, so an unparseable structured response
  loses all evidence gathered so far. Every other stage in the codebase degrades.

## 5. Deliberately unused, kept on purpose

* `scope_sql_where` / `scope_cypher_where` are exported with no in-library caller.
  `security.generated_query_isolation` recommends PostgreSQL RLS instead —
  string-concatenated tenant predicates are the weaker mechanism, and these exist
  for a deployment that cannot use RLS.
* `nodes.multihop_retrieve` and `MultiHopRetriever` are wired into no shipped
  graph. They are the "multi-hop as one tool" shape, for a caller assembling their
  own graph; the docstring's claim that the agentic graph uses it was wrong and is
  corrected.
* The `indexer` registry kind is registered and never resolved. `ragorc inspect`
  does not advertise it, so nothing promises otherwise.
* RAG-Fusion is reachable only through `translators=`, not a settings flag. Its
  fusion behaviour is on by default via `retrieval.fusion`.
