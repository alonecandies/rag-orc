# `ragorc.retrieve` — hybrid search, fusion, noise, reranking, CRAG, graph, multi-hop

The largest package, and the one where the division of labour matters most: **this
layer buys recall, and the reranker buys precision.** A document the first stage does
not return can never be recovered — no reranker, compressor or generator can rank a
passage it never saw — so every retriever returns `fetch_k` (50) candidates, not
`top_k` (10).

Names resolve lazily (PEP 562), so `import ragorc.retrieve` does not load the Qdrant
client, the Postgres stack, the Neo4j driver and an ONNX session. Call `load_all()` once
at startup to populate the component registry.
Related: [ADR-0003 — server-side fusion](../adr/0003-server-side-fusion.md).

## Key classes

```python
HybridRetriever(store=None, *, postgres=None, dense=None, sparse=None, noise=None, settings=None)
    async retrieve(query, *, top_k=None, **kw) -> list[ScoredChunk]
    async retrieve_detailed(query, *, top_k=None, **kw) -> RetrievalResult

EnsembleRetriever(retrievers: Sequence | Mapping, *, weights=None, method=None, noise=None, settings=None)
    def add(name, retriever, weight=None)
MultiStoreRetriever(retrievers=None, *, vector=None, relational=None, graph=None, web=None, ...)
    async retrieve_detailed(query, *, route: RouteDecision | None = None, top_k=None, **kw)

CorrectiveRAG(base: Retriever, llm, settings=None, *, web=None, router=None, compressor=None, refine=True)
    async run(query, *, top_k=None, **kw) -> tuple[RetrievalResult, Usage]

GraphLocalRetriever(graph, chunks=None, *, weights=None, hop_decay=0.6, settings=None)
GraphGlobalRetriever(llm, graph, *, level=None, router=None, settings=None)
GraphDriftRetriever(vector, graph, chunks=None, *, local=None, seed_k=None, settings=None)
IterativeRetriever(...) / BridgeEntityRetriever(...) / MultiHopRetriever(...)

ParentDocumentRetriever(inner, store=None, *, settings=None, overfetch=3)
    Search the precise child, return the complete parent. Over-fetches children
    because several collapse into one parent; expansion writes
    metadata["parent_text"] and ContextPacker performs the swap after ranking.

VectorRetriever · SparseRetriever · InMemoryBM25Retriever · SQLRetriever ·
PgVectorRetriever · PgFullTextRetriever · CypherRetriever · WebSearchRetriever

CrossEncoderReranker · ColBERTReranker · RankGPTReranker · IdentityReranker
    build_reranker(name=None, *, llm=None, settings=None, **kw) -> BaseReranker
    async rerank_chunks(query, chunks, *, top_k=None) -> list[ScoredChunk]
    async rerank_with_usage(query, chunks, *, top_k=None) -> tuple[list[ScoredChunk], Usage]
EmbeddingFilterCompressor · LLMExtractCompressor · SentenceLevelCompressor · PipelineCompressor
NoiseFilter(settings=None).apply(chunks, *, top_k, query_vector=None) -> (list, NoiseReport)
fuse(result_lists, method=FusionMethod.RRF, *, weights=None, k=None, names=None,
     top_k=None, settings=None) -> list[ScoredChunk]
```

## Hybrid search: two paths, one contract

**Server-side (default)** — one `query_points` call whose `prefetch` runs dense and
sparse inside Qdrant and whose outer query is a `FusionQuery`. One round trip, fusion
in Rust next to the data.

**Client-side** — the two legs run concurrently and `fusion.py` merges them here. Two
round trips and a Python merge, kept for three real reasons: `server_side_fusion=false`
(what the offline suite uses), a fusion method Qdrant cannot do (`weighted`,
`relative`, `max`), and the case where the *per-modality rankings* are wanted rather
than their combination. Which path ran is logged on every query.

The Postgres full-text leg is always client-side by construction — it lives in
another database — and is weighted 0.5, because `ts_rank_cd` has no IDF term and is a
weaker ranker than either vector branch: useful as a third opinion, wrong as a
primary vote.

## Noise handling, cheapest filter first

1. exact dedupe by id **and** by word-normalized content hash
2. relative score cutoff — a fraction of the top score, because an *absolute* similarity
   floor is wrong per corpus and per embedding model, and fails in both directions
3. near-duplicate collapse — embedding cosine when vectors are present, SimHash otherwise
4. MMR diversity — one matmul for the pairwise matrix, then a greedy loop
5. lost-in-the-middle reordering — strongest evidence at *both* ends

It runs **after** fusion, which is what creates the duplicates and what gives the
relative cutoff one scale to be relative to.

## Degrading, not failing

Every leg runs under its own deadline (`per_store_timeout_s`) and, in
`MultiStoreRetriever`, its own circuit breaker. A deadline bounds one request; a
breaker bounds the next thousand. Failures land in `RetrievalResult.errors` by name
with their latency in `timings_ms`, and the query continues. The two exceptions are
`GuardrailViolation` and `BudgetExceeded`, which propagate — those are decisions, not
outages.

## Usage

```python
from ragorc.retrieve import HybridRetriever, load_all
from ragorc.retrieve.crag import CorrectiveRAG
from ragorc.retrieve.rerank import build_reranker

load_all()
hybrid = HybridRetriever(store)
result = await hybrid.retrieve_detailed(query, fetch_k=50)
print(result.per_store.keys(), result.timings_ms, result.errors)

reranked = await build_reranker(settings=settings).rerank_chunks(
    query.text, result.chunks, top_k=10
)

crag = CorrectiveRAG(hybrid, llm)
graded, usage = await crag.run(query, top_k=10)
print(graded.grade, graded.per_store.keys())  # CORRECT / AMBIGUOUS / INCORRECT
```

## Settings

| Setting | Effect |
|---|---|
`retrieval.top_k` · `fetch_k` | what the generator sees vs what recall is bought at |
`retrieval.hybrid_enabled` · `use_dense` · `use_sparse` · `use_fulltext` | which legs run. `hybrid_enabled=false` narrows the *defaults* to a single dense leg; the three finer flags and a per-call override still win |
`retrieval.server_side_fusion` | one round trip vs two |
`retrieval.fusion` · `rrf_k` · `fusion_weights` | RRF is rank-based, so it is the only merge unaffected by scores with no common scale |
`retrieval.rerank_enabled` · `reranker` · `rerank_top_k` · `rerank_batch_size` | the one retrieval stage worth tuning |
`retrieval.rankgpt_window` · `rankgpt_step` | sliding-window listwise reranking |
`retrieval.colbert_rerank` | MaxSim over multivectors, nested in the prefetch |
`retrieval.reranker = "colbert"` | client-side MaxSim. The store returns the stored multivectors so candidates are rescored without re-embedding — ~289 ms of ONNX saved per query at the default rerank width. A collection's named vectors are fixed at creation, so switching this on against an index built without ColBERT skips the stage and warns; the remedy is to re-index into a new `qdrant.collection` |
`retrieval.parent_expansion` | resolve a derived unit (child chunk, summary, proposition) back to its source. Gates both the fetch and the substitution; off means neither |
`security.enforce_tenant_isolation` | the parent fetch is scoped to the query's tenant, so a child whose parent belongs to another tenant comes back unexpanded rather than carrying that tenant's text |
`retrieval.relative_score_cutoff` · `score_threshold` · `dedupe_enabled` · `near_dupe_threshold` | noise filter |
`retrieval.mmr_enabled` · `mmr_lambda` · `reorder_lost_in_middle` | diversity and placement |
`retrieval.compression_enabled` · `compressor` · `compression_ratio` | post-retrieval refinement |
`retrieval.crag_enabled` · `crag_grade_top_k` · `crag_relevance_threshold` · `crag_web_fallback` · `web_search_provider` | CRAG |
`retrieval.per_store_timeout_s` · `max_concurrent_retrievers` | the difference between p99 and p99-of-the-slowest-store |
`graph.local_search_*` · `global_search_*` · `multihop_*` | GraphRAG and multi-hop |
