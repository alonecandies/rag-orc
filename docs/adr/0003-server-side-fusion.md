# ADR-0003: Hybrid search fused server-side in Qdrant, in one round trip

**Status:** accepted · **Date:** 2026-08-19

## Context

Hybrid retrieval needs at least dense (semantic) and sparse/BM25 (lexical)
results combined — dense search misses exact identifiers, error codes, SKUs and
rare proper nouns; lexical search misses paraphrase. Reciprocal Rank Fusion over
both is the standard combiner.

The conventional implementation runs a vector search, runs a keyword search
against a second system (Elasticsearch, or Postgres full-text), and merges in
application code. That is two network round trips, two systems to operate and
keep consistent, and a merge step that has to normalize incomparable score
scales.

## Decision

Express **every** representation as a vector inside Qdrant, and let Qdrant do the
fusion:

| Representation | Stored as | Purpose |
|---|---|---|
Semantic | named dense vector `dense` | paraphrase, concepts |
Lexical | **sparse vector from BM25** with `Modifier.IDF` | exact terms, identifiers |
Learned sparse | sparse vector from SPLADE (optional) | term expansion |
Late interaction | **multivector** `colbert`, `MaxSim` comparator | precise reranking |

One `query_points` call carries a `prefetch` list (dense + sparse, each with its
own limit) and a `FusionQuery(fusion=RRF)`, optionally nested inside a ColBERT
`MaxSim` rerank stage. The whole hybrid-and-rerank pipeline is a single gRPC
request.

Two details that make this work and are easy to miss:

- **`Modifier.IDF` on the sparse vector** makes Qdrant compute the IDF term at
  query time against the actual collection statistics. Without it, BM25-as-sparse
  is just term frequency and scores badly.
- **The `colbert` multivector must not be HNSW-indexed** (`hnsw_config.m = 0`).
  It is a reranking field reached through `prefetch`, never a first-stage index.
  Indexing it would cost 100x the storage for no benefit.

## Consequences

- **One network round trip instead of two**, and fusion happens in Rust next to
  the data instead of in Python after two transfers.
- **No second search engine.** No Elasticsearch cluster, no index-sync job, no
  consistency window between the vector store and the keyword store.
- Score normalization is Qdrant's problem, not ours. RRF operates on ranks, and
  DBSF is available for when score distributions are comparable.
- Client-side fusion is still implemented (`retrieve/fusion.py`) because
  cross-*store* fusion — merging Qdrant with Postgres and Neo4j results — has to
  happen in our process. `retrieval.server_side_fusion=false` also falls back to
  it, which is what the offline test suite uses.
