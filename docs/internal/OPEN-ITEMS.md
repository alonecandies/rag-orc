# Open items

Known gaps, each verified against the code at the commit that added its row. This
file exists because the working notes it replaces were machine-local, unlinked and
stale, so the same findings were rediscovered from scratch — twice. If you close
one, delete the row.

Nothing here is a secret or a live vulnerability. The security findings from
earlier audits are fixed and covered by tests in `tests/unit/test_security_guards.py`
and `tests/unit/test_guard_properties.py`.

## 1. Closed: multi-representation indexing is wired

`summary_index_enabled` and `dense_x_enabled` run as enrichment stages through
`MultiRepresentationIndexer`. Write ownership is settled: the stage builds and
embeds its derived units and the pipeline writes them, because
`_process_document` already writes whatever a stage returns. The docstore write
stays with the indexer, since `expand_parents` reads those sources back at query
time.

`parent_document_enabled` is handled at the splitter rather than as an enrichment,
because it splits the document twice and the child is the retrieval unit — running
it beside the normal split would index every document twice.
`IngestPipeline._split` replaces the split with it and routes the parents to the
docstore alone.

Derived units carry the dense vector their indexer computed and no sparse or
ColBERT vector, because those are added before the enrichment stage runs. On a
hybrid collection they are findable by vector search and not by BM25 — the one
remaining rough edge here.

## 2. Closed: corpus-wide graph construction has a command

`ragorc graph build` runs the second pass, reading chunks back from the collection.
It remains a second pass on purpose — resolution and community detection are only
meaningful over the whole corpus, and ingest holds one window of documents at a
time.

Unlike ingest it does hold the corpus in memory, because extraction is per chunk
but resolution, detection and the write are global. `--limit` bounds a trial run.

## 3. Closed: the doubles now model the stores

Seven tests survived their own mutation. Four are fixed and two were misdiagnosed:

* `FakeVectorStore.get` accepted `with_vectors` and ignored it, so production that
  stopped asking for vectors got them anyway. It returns vectorless chunks now, and
  dropping `with_vectors=True` from the graph chunk read kills two tests.
* The multi-hop injection test ran on a `ScriptedLLM`, whose verdict ignores the
  prompt, so it passed with the isolation removed. It now uses a model that would
  comply if the instruction reached it in the clear and respects
  `<untrusted_document>` delimiters otherwise. Removing `isolate=True` turns it red.
* Chasing the DRIFT merge finding turned up the cause: `GraphLocalRetriever._annotate`
  mutated the chunk it was handed, so both merge halves held one object. It copies now.
* Misdiagnosed: over-propagation in `_propagate` cannot reach anything the store's
  hop bound did not return, so that mutation surviving is correct behaviour. And one
  multi-hop test is merely redundant with two others, which is not a defect.

## 4. One undertested behaviour in multi-hop

`_hop_query` is applied at hop 0, so a caller-supplied `dense`/`sparse`/`multi`
vector on the incoming `Query` is dropped on the first retrieval too. Whether that
is intended is unclear — it may be deliberate, since the hop query is a different
question from the caller's.

`_check_sufficiency` losing all gathered evidence on an unparseable response is
fixed: it now stops with what the hops found, which is what every other stage in
the module does.

## 5. Claims from earlier audits that did not survive checking

Reported by the coverage pass, not asserted as intent because neither is clearly
intended:

* `_hop_query` is applied at hop 0, so a caller-supplied `dense`/`sparse`/`multi`
  vector on the incoming `Query` is dropped on the first retrieval too.
* `_check_sufficiency` has no `try`/`except`, so an unparseable structured response
  loses all evidence gathered so far. Every other stage in the codebase degrades.

Recorded so they are not re-investigated a third time:

* "Enabling CRAG silently turns off reranking" — `builder.py:1090-1091` appends
  `nodes.rerank` whichever retrieval node was chosen, so reranking runs either way.
* "`filters` is silently dropped on the graph path" — `retrieve/graph.py:961`
  reads `kwargs["filters"]` then `query.filters`. Overstated; one call site does
  honour it.
* "`enable_late_interaction=true` makes every ingest fail" — not reproducible now.
  The late-chunking substitution that produced it was removed earlier, so this is
  most likely already fixed. Unverified against a live multivector collection.

## 6. Deliberately unused, kept on purpose

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
