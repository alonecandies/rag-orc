# `ragorc.stores` — Qdrant, Postgres, Neo4j

Three backends, three protocols, one id space. Chunk ids are content-derived
(`ragorc.core.ids`), so the same chunk has the same id in all three stores and
cross-store joins need no mapping table.

The package imports **lazily per store**: a dense-only deployment never loads the
Postgres or Neo4j driver.

Related: [ADR-0003 — server-side fusion](../adr/0003-server-side-fusion.md) ·
[ADR-0006 — layered query guards](../adr/0006-layered-query-guards.md).

## Qdrant — the vector store

```python
QdrantStore(settings=None, client=None, dense_embedder=None,
            sparse_embedder=None, late_embedder=None)
    async ensure_collection(*, recreate=False) -> None
    async upsert(chunks) -> int
    async search(query, *, top_k=None, filters=None, fetch_k=None, tenant_id=None, **kw) -> list[ScoredChunk]
    async search_dense(...) / search_sparse(...) / search_colbert(...)
    async get(ids, *, with_vectors=False) -> list[Chunk]
    async scroll(...) / count(...) / delete(ids=None, *, filters=None, tenant_id=None)
    bulk_load()                                  # async context manager: indexing off during ingest
    async flush(*, timeout_s=300.0) -> int       # wait for green, then read back the exact count
```

`search` issues **one** `query_points` call whose `prefetch` list runs the dense and
sparse branches server-side and whose outer query is a `FusionQuery(RRF)`, optionally
nested inside a ColBERT `MaxSim` rerank. One round trip, fusion in Rust next to the
data. Embedders are injected rather than constructed so the store and the ingest
pipeline share one loaded ONNX session.

Two collection details are easy to get wrong and both are handled in
`qdrant/collections.py`: the sparse vector needs `Modifier.IDF` or BM25-as-sparse
degenerates into term frequency, and the ColBERT multivector needs `hnsw m=0`
because it is only ever reached through prefetch.

## Postgres — relational, pgvector, full-text, Text-to-SQL target

```python
PostgresStore(settings=None, *, cache: Cache | None = None)
    async ensure_schema(*, drop=False) -> None
    async execute_readonly(sql, params=None, *, limit=100) -> list[dict]
    async schema_summary(*, refresh=False) -> str       # cached DDL summary for the prompt
    async fulltext_search(query, *, top_k=None, **kw) -> list[ScoredChunk]   # top_k falls back to retrieval.top_k
    async vector_search(...) / hybrid_search(...)
    async upsert_documents(docs) / upsert_chunks(chunks)
    async get_chunks(ids) / get_children(parent_id) / delete_document(document_id) / count()
```

`execute_readonly` is the Text-to-SQL execution target: read-only transaction,
server-side `statement_timeout`, row cap, and — if `postgres.readonly_dsn` is set —
a role holding `SELECT` only. Three independent layers, because a guard that is the
only layer is a single point of failure.

## Neo4j — the graph

```python
Neo4jStore(driver=None, *, settings=None)
    async ensure_schema() -> list[str]
    async execute_readonly(cypher, params=None, *, limit=100) -> list[dict]
    async upsert_entities(entities) / upsert_relations(relations) / upsert_communities(...)
    async upsert_chunk_links(chunk_ids_by_entity)
    async fulltext_entities(query, *, limit=None) -> list[tuple[Entity, float]]   # capped by neo4j.max_cypher_rows
    async neighbors(names, *, hops=1, limit=50) -> tuple[list[Entity], list[Relation]]
    async paths(start, end, *, max_hops=3, limit=10) -> list[GraphPath]
    async communities(*, level=None, limit=None) -> list[Community]
    async degree_centrality(names) -> dict[str, float]
```

Every label interpolated into Cypher is validated once, at construction, so the
injection surface of the module is closed before any query is built.

## Usage

```python
from ragorc.stores import Neo4jStore, PostgresStore, QdrantStore

vector = QdrantStore(dense_embedder=dense, sparse_embedder=sparse)
await vector.ensure_collection()
await vector.upsert(chunks)  # chunks must already carry vectors
hits = await vector.search(query, top_k=10, fetch_k=50)

pg = PostgresStore()
await pg.ensure_schema()
graph = Neo4jStore()
await graph.ensure_schema()
```

## Settings

| Setting | Effect |
|---|---|
`qdrant.prefer_grpc` · `grpc_port` | protobuf instead of JSON float arrays: 2-3x faster |
`qdrant.hnsw_m` · `hnsw_ef_construct` · `hnsw_ef_search` | `ef_search` is the recall/latency dial at query time |
`qdrant.quantization` · `oversampling` · `rescore` | scalar int8 cuts memory 4x; without rescore it costs real accuracy |
`qdrant.on_disk_vectors` · `quantization_always_ram` | turn both on past ~5M vectors |
`qdrant.indexing_threshold` · `upsert_batch_size` · `parallel_upserts` · `wait_on_upsert` | ingest throughput |
`qdrant.use_multitenancy_index` | payload index with `is_tenant=True` co-locates a tenant's vectors on disk |
`postgres.dsn` · `readonly_dsn` | the second DSN is the `SELECT`-only role |
`postgres.statement_timeout_ms` · `max_sql_rows` · `allowed_tables` | fence in generated SQL |
`postgres.vector_index` · `hnsw_m` · `hnsw_ef_construction` | pgvector HNSW needs no training step |
`postgres.binary` · `prepare_threshold` · `max_pool_size` | psycopg3 binary protocol and plan reuse |
`neo4j.uri` · `user` · `password` · `database` | |
`neo4j.create_fulltext_index` | the entry point for GraphRAG local search |
`neo4j.max_cypher_rows` · `query_timeout_s` | bound a generated traversal |
