"""Configuration.

One nested settings tree, populated from environment variables, ``.env`` or an
explicit dict. Nesting uses ``__`` so every field is reachable from the
environment without a config file:

    RAGORC_LLM__API_KEY=sk-or-...
    RAGORC_QDRANT__PREFER_GRPC=true
    RAGORC_RETRIEVAL__FUSION=dbsf

Defaults are chosen for *production*, not for demos: gRPC on, quantization on,
caching on, guards on. Every non-obvious default carries the reason inline,
because a magic number without a rationale is a future outage.
"""

from __future__ import annotations

import functools
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragorc.core.models import ChunkingStrategy, FusionMethod

__all__ = [
    "CacheSettings",
    "CostSettings",
    "EmbeddingSettings",
    "GenerationSettings",
    "GraphSettings",
    "IndexingSettings",
    "LLMSettings",
    "Neo4jSettings",
    "ObservabilitySettings",
    "PostgresSettings",
    "QdrantSettings",
    "RetrievalSettings",
    "SecuritySettings",
    "ServerSettings",
    "Settings",
    "get_settings",
]


class LLMSettings(BaseModel):
    """OpenRouter. Also works against any OpenAI-compatible endpoint.

    The three model slots implement a **cost cascade** (ADR-0005). A single RAG
    query makes one synthesis call and 10-20 classification calls (routing,
    grading, filtering, extraction). Sending the classifiers to a frontier model
    is where RAG budgets die, so they get ``fast_model`` by default — typically
    20-50x cheaper — while only synthesis uses ``model``.
    """

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = SecretStr("")

    model: str = "anthropic/claude-sonnet-4.5"
    """Synthesis: the final answer, and anything that needs real reasoning."""
    fast_model: str = "google/gemini-2.5-flash-lite"
    """Classifiers, routers, graders, rewriters. High volume, low difficulty."""
    strong_model: str = "anthropic/claude-opus-4.5"
    """Escalation target, reached by `model_for(..., escalate=True)`.

    One caller: the Text-to-SQL guard repair, which escalates unconditionally
    because a query the guard rejected is worth one expensive retry rather than a
    failed answer. There is deliberately no confidence-gated escalation on the
    answer path — see ADR-0005."""

    temperature: float = 0.0
    """Zero by default: RAG answers should be reproducible, and every
    classifier in the pipeline wants argmax, not a sample."""
    max_tokens: int = 2048
    context_window: int = 128_000
    timeout_s: float = 90.0
    connect_timeout_s: float = 10.0

    max_concurrency: int = 16
    """Global cap on in-flight LLM requests. Bounds both provider rate-limit
    pressure and our own memory when a map stage fans out over 10k chunks."""
    max_concurrent_streams: int = 4
    """Separate cap for SSE streams, because a stream's duration is set by the
    *client* reading it, not by the provider.

    An async generator holds its permit across every ``yield``, so a permit taken
    from ``max_concurrency`` stays taken until the consumer finishes reading. With
    one shared pool, that many slow readers stall every non-streaming call in the
    process — grading, routing, synthesis, ingest. A separate pool keeps a slow
    client from starving work it has nothing to do with; streams then contend only
    with each other."""
    max_retries: int = 4
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 20.0

    requests_per_minute: int | None = None
    """Client-side token bucket. Set it to stay under a provider quota instead
    of discovering the limit through 429s."""
    tokens_per_minute: int | None = None

    http2: bool = True
    """Multiplexes concurrent requests over one TCP connection — measurably
    lower latency than opening 16 connections."""

    # --- OpenRouter-specific routing controls -----------------------------
    provider_order: list[str] = Field(default_factory=list)
    """Explicit provider preference, e.g. ``["deepinfra", "together"]``."""
    provider_sort: Literal["", "price", "throughput", "latency"] = "price"
    """``price`` selects the cheapest provider serving the model."""
    allow_fallbacks: bool = True
    require_parameters: bool = True
    """Only route to providers that support the parameters we send (notably
    ``response_format``) — otherwise structured output silently degrades."""
    data_collection: Literal["allow", "deny"] = "deny"
    """``deny`` excludes providers that train on prompts. Safe default for
    anything touching internal documents."""
    app_name: str = "ragorc"
    site_url: str = ""

    enable_prompt_cache: bool = True
    """Emit provider prompt-cache hints (Anthropic ``cache_control``). Large
    static system prompts then cost ~10% on repeat calls."""


class EmbeddingSettings(BaseModel):
    """Vectors. OpenRouter has no embeddings endpoint, so this is a separate
    provider; FastEmbed (ONNX, local, free) is the default (ADR-0004)."""

    provider: Literal["fastembed", "openai", "voyage", "cohere", "sentence_transformers"] = (
        "fastembed"
    )

    dense_model: str = "BAAI/bge-small-en-v1.5"
    """384-dim. Small, fast, and strong for its size. Swap to
    ``jinaai/jina-embeddings-v2-base-en`` (768-dim, 8192 ctx) for late chunking
    or ``BAAI/bge-m3`` for multilingual."""
    dense_dimension: int | None = None
    """Auto-detected from the model when left unset."""
    query_prefix: str = ""
    document_prefix: str = ""
    """Asymmetric models need these (E5: ``query: `` / ``passage: ``). Getting
    them wrong is a silent recall loss, so they are explicit."""

    sparse_model: str = "Qdrant/bm25"
    """BM25 as a sparse vector. Combined with Qdrant's IDF modifier this is
    true BM25 scoring *inside* the vector store — hybrid search with no second
    search engine to operate."""
    use_splade: bool = False
    splade_model: str = "prithivida/Splade_PP_en_v1"
    """Learned sparse: better than BM25 on paraphrase, ~4x the index size."""

    late_interaction_model: str = "colbert-ir/colbertv2.0"
    enable_late_interaction: bool = False
    """ColBERT multivectors are ~100x the storage of a single dense vector.
    Worth it as a *reranking* stage on the top ~200 candidates (see
    ``RetrievalSettings.colbert_rerank``), rarely worth it as a first-stage index."""
    late_interaction_max_tokens: int = 192
    """Token vectors kept per chunk, longest-norm first (``prune_tokens``).

    The dial that decides what ColBERT costs, since storage is linear in it: at
    128 dimensions and float32 this is 96 KB per chunk. Lowering it trades recall
    on long chunks for storage, and only tokens a query could have matched are
    lost. It was reachable only by constructing :class:`~ragorc.index.colbert.ColBERTIndexer`
    by hand until the ingest path started using it."""

    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    """ONNX cross-encoder. No torch required."""

    batch_size: int = 64
    """Tuned for ONNX CPU inference; raise to 256+ on GPU."""
    max_length: int = 512
    normalize: bool = True
    """L2-normalize so cosine similarity reduces to a dot product."""
    threads: int | None = None
    cache_embeddings: bool = True
    """Content-hash keyed. Re-ingesting an unchanged corpus then costs nothing."""

    api_key: SecretStr = SecretStr("")


class QdrantSettings(BaseModel):
    url: str = "http://localhost:6333"
    api_key: SecretStr = SecretStr("")
    collection: str = "ragorc"

    prefer_grpc: bool = True
    """gRPC is 2-3x faster than REST for vector payloads: protobuf instead of
    JSON-encoded float arrays."""
    grpc_port: int = 6334
    timeout_s: float = 30.0

    # --- HNSW ------------------------------------------------------------
    hnsw_m: int = 16
    """Edges per node. 16 is the accuracy/memory sweet spot; 32-64 for >10M
    points where recall matters more than RAM."""
    hnsw_ef_construct: int = 128
    hnsw_ef_search: int = 128
    """Search-time beam width. The main recall/latency dial at query time."""
    full_scan_threshold: int = 10_000
    """Below this many points, brute force beats HNSW — Qdrant switches itself."""

    # --- quantization ----------------------------------------------------
    quantization: Literal["none", "scalar", "binary", "product"] = "scalar"
    """``scalar`` (int8) cuts memory 4x for ~1% recall loss and is faster
    because of SIMD. ``binary`` cuts 32x and needs oversampling + rescoring;
    only viable for high-dimensional models (>=1024)."""
    quantization_always_ram: bool = True
    """Keep the quantized vectors in RAM even when originals are on disk —
    this is what makes on-disk storage fast."""
    oversampling: float = 2.0
    rescore: bool = True
    """Re-score the oversampled candidates with full-precision vectors. Without
    this, quantization costs real accuracy."""

    on_disk_payload: bool = True
    on_disk_vectors: bool = False
    """Turn on past ~5M vectors, together with ``quantization_always_ram``."""

    shard_number: int = 1
    replication_factor: int = 1
    write_consistency_factor: int = 1
    default_segment_number: int = 0
    """0 lets Qdrant choose (roughly CPU count) — good for parallel search."""

    indexing_threshold: int = 20_000
    """Defer index building during bulk load; set to 0 for a large ingest then
    restore, which is dramatically faster than indexing per batch."""

    upsert_batch_size: int = 256
    parallel_upserts: int = 4
    wait_on_upsert: bool = False
    """``False`` returns as soon as the write is queued. Fire-and-forget during
    bulk ingest; the ingest pipeline does one final ``wait=True`` flush."""

    use_multitenancy_index: bool = True
    """Payload index on ``tenant_id`` with ``is_tenant=True``: Qdrant then
    co-locates each tenant's vectors on disk, making filtered search fast
    instead of a filtered scan."""


class PostgresSettings(BaseModel):
    dsn: SecretStr = SecretStr("postgresql://ragorc:ragorc@localhost:5432/ragorc")
    """Connection string for the primary role.

    ``SecretStr`` because it carries a password inline, which is the one place in
    this settings tree where a credential hides inside a field that does not look
    like one. ``llm.api_key`` and ``neo4j.password`` were secrets and this was a
    plain ``str``, so ``repr(settings)``, ``model_dump()`` and
    ``model_dump_json()`` masked those two and printed this one in full — into a
    debugger, a crash reporter, or any handler that serializes its configuration.
    ``Settings.summary()`` was already careful; nothing else was.

    Read it with ``.get_secret_value()``. There are three call sites and two of
    them only want the host part."""

    readonly_dsn: SecretStr = SecretStr("")
    """Separate DSN for a ``SELECT``-only role used by Text-to-SQL. Defence in
    depth: even if the SQL guard is bypassed, the connection cannot write."""

    min_pool_size: int = 2
    max_pool_size: int = 16
    max_idle_s: float = 300.0
    timeout_s: float = 30.0
    statement_timeout_ms: int = 15_000
    """Server-side cap so a pathological generated query cannot pin a worker."""

    binary: bool = True
    """Binary protocol: no text encode/decode of float arrays or timestamps."""
    prepare_threshold: int = 5
    """Server-side prepare after 5 executions — plan reuse for hot queries."""

    schema_name: str = "public"
    chunks_table: str = "ragorc_chunks"
    documents_table: str = "ragorc_documents"

    vector_dimension: int = 384
    vector_index: Literal["hnsw", "ivfflat", "none"] = "hnsw"
    """pgvector HNSW: better recall/latency than IVFFlat and no training step."""
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    ivf_lists: int = 100

    fulltext_config: str = "english"
    use_pg_search: bool = False
    """ParadeDB's ``pg_search`` gives true BM25 in Postgres. Off by default
    because it needs an extension; ``ts_rank`` is the portable fallback."""

    allowed_tables: list[str] = Field(default_factory=list)
    """Allowlist for Text-to-SQL. Empty = every table the role can read."""
    max_sql_rows: int = 200


class Neo4jSettings(BaseModel):
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("ragorcpass")
    database: str = "neo4j"

    max_connection_pool_size: int = 50
    connection_timeout_s: float = 30.0
    max_transaction_retry_time_s: float = 15.0
    fetch_size: int = 1000

    node_label: str = "Entity"
    chunk_label: str = "Chunk"
    community_label: str = "Community"
    max_cypher_rows: int = 200
    query_timeout_s: float = 20.0

    create_fulltext_index: bool = True
    """Full-text index on entity name/description — the entry point for
    GraphRAG local search, which matches question entities to graph nodes."""
    create_vector_index: bool = False
    """Neo4j 5.13+ native vector index. Useful if you want entity-embedding
    search in the graph itself rather than in Qdrant."""


class CacheSettings(BaseModel):
    """Three tiers, checked cheapest-first (ADR-0007)."""

    enabled: bool = True
    memory_max_items: int = 20_000
    memory_ttl_s: float = 900.0

    redis_url: str = ""
    """Empty disables the shared tier. Set it whenever more than one process
    serves traffic, otherwise each worker warms its own cache."""
    redis_ttl_s: float = 86_400.0
    redis_prefix: str = "ragorc"

    semantic_enabled: bool = True
    """Answer-level cache keyed by *embedding proximity* rather than by hash.

    It used to be documented as making "what is X" hit the entry for "explain X",
    at 20-40% of traffic. Neither survives measurement with the shipped default
    embedder (``BAAI/bge-small-en-v1.5``, 384-dim, cosine, ~20 question pairs from
    this repo's corpus): that exact pair scores **0.9596**, well under the
    threshold below, and paraphrases in general land at 0.93-0.97 — the same band
    as questions that have *different answers*. What does clear the threshold is
    the same question re-asked with different case, punctuation or whitespace
    (0.9953-1.0000), which is common enough in a chat UI or a retried request to
    be worth caching, but is not a paraphrase cache and cannot be budgeted at a
    hit rate taken from someone else's traffic. Measure yours before counting on
    it, and see ``semantic_threshold`` for the part that is model-dependent."""
    semantic_threshold: float = 0.97
    """Strict, and strict is not the same as safe — this number belongs to the
    *model*, not to an abstract notion of similarity.

    Measured on the shipped embedder: "Who approves expenses over $500?" against
    "...under $500?" scores **0.9924**, i.e. above this threshold with the
    opposite meaning, because a one-word inversion barely moves a mean-pooled
    384-dim vector while a rephrasing moves it a lot. So the ordering the cache
    needs is not the ordering the model provides. In the same run the highest
    wrong pair was 0.9924 and the lowest surface-form variant of the *same*
    question was 0.9953: 0.995 is the value that separates them, and it is what to
    set if serving the opposite question's answer is unacceptable in your domain.

    Lowering it does not buy paraphrase hits — those sit at 0.93-0.97, tangled up
    with questions whose answers differ — it only buys wrong ones. Any change here
    is a correctness change, and it needs re-measuring against the model you
    actually run, not against this number."""
    semantic_collection: str = "ragorc_semantic_cache"
    semantic_ttl_s: float = 3600.0

    cache_llm: bool = True
    cache_embeddings: bool = True
    cache_rerank: bool = True
    cache_schema: bool = True


class SecuritySettings(BaseModel):
    """Defaults are restrictive. A Text-to-SQL feature is a remote code
    execution primitive unless it is fenced in."""

    enable_sql_guard: bool = True
    sql_allow_statements: list[str] = Field(default_factory=lambda: ["SELECT", "WITH"])
    sql_forbid_functions: list[str] = Field(
        default_factory=lambda: [
            "pg_read_file",
            "pg_read_binary_file",
            "pg_ls_dir",
            "lo_import",
            "lo_export",
            "dblink",
            "pg_sleep",
            "copy",
        ]
    )
    sql_max_joins: int = 8
    """Bounds the damage of a generated cartesian product."""
    sql_require_limit: bool = True

    enable_cypher_guard: bool = True
    cypher_forbid_keywords: list[str] = Field(
        default_factory=lambda: [
            "CREATE",
            "MERGE",
            "DELETE",
            "DETACH",
            "SET",
            "REMOVE",
            "DROP",
            "LOAD CSV",
            "CALL DBMS",
            "CALL APOC.TRIGGER",
            "CALL APOC.LOAD",
            "TERMINATE",
            "GRANT",
            "DENY",
            "REVOKE",
        ]
    )
    cypher_explain_dryrun: bool = True
    """Run EXPLAIN first: catches syntax errors and unbounded expansions
    without touching data."""

    enable_injection_detection: bool = True
    """Retrieved documents are untrusted input. A document containing
    "ignore previous instructions" is an attack on the generator, and it
    arrives through the *data* path where nobody thinks to look."""
    injection_action: Literal["block", "sanitize", "flag"] = "sanitize"
    max_query_length: int = 4000
    min_query_length: int = 1

    enable_pii_redaction: bool = False
    pii_entities: list[str] = Field(
        default_factory=lambda: ["EMAIL", "PHONE", "CREDIT_CARD", "SSN", "IBAN", "IP"]
    )
    pii_action: Literal["redact", "hash", "flag"] = "redact"

    enable_rate_limit: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 20

    enforce_tenant_isolation: bool = True
    """When on, a query without a ``tenant_id`` is rejected rather than
    silently searching every tenant's data.

    Covers the vector store by construction (every filter is built through
    :func:`~ragorc.security.tenancy.scope_filter`). It does **not** cover the
    knowledge graph: nothing written to Neo4j carries a tenant, so
    ``graph_tenant_isolation`` governs that leg. Nor can it cover *generated*
    SQL or Cypher: those run against the operator's own schema, and
    this library cannot know which column of ``orders`` carries a tenant — or
    whether the concept exists there at all. So those legs **fail closed** unless
    ``generated_query_isolation`` declares how isolation is enforced."""

    graph_tenant_isolation: Literal["reject", "trusted"] = "reject"
    """How tenant isolation is enforced for the **knowledge graph** legs.

    A separate setting from ``generated_query_isolation``, because it is a
    separate problem with the same shape. That one covers Cypher an LLM *wrote*;
    this covers the parameterized traversals in
    :class:`~ragorc.stores.neo4j.store.Neo4jStore` that GraphRAG local, global
    and DRIFT search, and the multi-hop bridge, all run.

    Neo4j holds no tenant at all. Entities are merged on ``name``, communities on
    a membership hash, and chunk links on a chunk id — none of them namespaced —
    so two tenants writing about the same company converge on one node and a
    traversal from it reaches both. The chunk ids it yields then went through an
    unscoped by-id fetch, and the verbalized subgraph it returns is built from
    every tenant's entity descriptions while being *stamped* with the querying
    tenant's id.

    ``reject``   — refuse the graph legs while tenant isolation is on. The
                   default, because it is the only setting that is true without
                   the operator having done something.
    ``trusted``  — the graph holds one tenant's data: a Neo4j instance or
                   database per tenant, or a single-tenant deployment that has
                   isolation on for other reasons. An explicit assertion, so it
                   cannot be arrived at by accident.

    There is deliberately no ``rls``-equivalent. Making the graph multi-tenant is
    a schema change — entity identity has to become ``(tenant, name)``, which
    changes every MERGE, every traversal predicate and the fulltext index, and
    needs a migration for graphs already built — not a filter this library can
    add at query time. Pretending otherwise would provide the appearance of
    isolation, which is worse than a refusal an operator can see.
    """

    foreign_retriever_tenant_isolation: Literal["reject", "filter", "trusted"] = "reject"
    """How tenant isolation is enforced for a retriever this library does not own.

    :func:`~ragorc.adapters.langchain.from_langchain_retriever` makes someone
    else's retriever one leg of an ensemble, fused with ours. That leg is outside
    every mechanism the other settings here describe: it runs its own query
    against its own store, so no filter of ours reaches it, and the chunks it
    returns declare their own ``tenant_id`` — read out of the foreign document's
    metadata, which is the retriever's claim rather than anything we verified.

    Unlike the graph, there *is* a defensible middle ground, which is why this has
    three modes rather than two. A returned document either carries a tenant label
    or does not, and both cases can be decided without a schema change.

    ``reject``  — refuse the leg while tenant isolation is on. The default: a
                  foreign retriever has no way to prove it scoped anything.
    ``filter``  — pass the tenant down (so a capable retriever can scope itself)
                  and drop every returned chunk that does not *declare* the
                  querying tenant. Unlabelled chunks are dropped too: an absent
                  label is not a match, and stamping one with the querying
                  tenant's id would forge exactly the provenance
                  ``graph_tenant_isolation`` exists to prevent.
    ``trusted`` — the wrapped retriever holds one tenant's data. An explicit
                  assertion, so it cannot be arrived at by accident.
    """

    generated_query_isolation: Literal["reject", "database", "rls", "trusted"] = "reject"
    """How tenant isolation is enforced for *generated* SQL and Cypher.

    ``reject``    — refuse the relational and graph legs while tenant isolation is
                    on. The default, because it is the only setting that is true
                    without the operator having done something.
    ``database``  — each tenant has its own database/schema and the connection is
                    already scoped, so a predicate would be redundant.
    ``rls``       — PostgreSQL row-level security is enabled on the queried
                    tables. This is the *correct* mechanism: it is enforced by the
                    database on every statement, including ones this library never
                    sees, which no amount of query rewriting can match.
    ``trusted``   — single-tenant data, or isolation handled upstream. An explicit
                    opt-out, so it appears in a config review rather than being
                    the silent default.

    Deliberately not offered: injecting a ``WHERE tenant_id = …`` into generated
    SQL. Placing a predicate correctly across joins, CTEs, subqueries and set
    operations is exactly the problem row-level security already solves in the
    database, and a rewriter that gets it subtly wrong provides the *appearance*
    of isolation — which is worse than refusing."""
    audit_log_enabled: bool = True
    audit_log_path: str = ""
    redact_secrets_in_logs: bool = True


class IndexingSettings(BaseModel):
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    """AUTO resolves to LATE when the embedder exposes token vectors, else
    CONTEXTUAL if enabled, else EARLY (ADR-0002)."""
    splitter: Literal["recursive", "token", "semantic", "markdown", "code", "sentence_window"] = (
        "semantic"
    )

    chunk_size: int = 512
    chunk_overlap: int = 64
    """Overlap costs storage but prevents an answer being split across the
    boundary of two chunks, which no retriever can recover from."""
    min_chunk_size: int = 64
    max_chunk_size: int = 2048

    # --- semantic splitting ----------------------------------------------
    semantic_breakpoint: Literal["percentile", "stddev", "interquartile", "gradient"] = "percentile"
    semantic_threshold: float = 95.0
    semantic_buffer_size: int = 1
    """Sentences of context on each side when embedding for boundary
    detection — a lone sentence is too noisy to compare reliably."""
    semantic_min_sentences: int = 3

    # --- contextual retrieval --------------------------------------------
    contextual_enabled: bool = False
    contextual_max_doc_tokens: int = 60_000
    contextual_prefix_tokens: int = 100

    # --- multi-representation --------------------------------------------
    parent_document_enabled: bool = False
    parent_chunk_size: int = 2048
    child_chunk_size: int = 256
    """Search the small chunk for precision, return the large one for context.
    The retrieval unit and the generation unit should not be the same thing."""
    summary_index_enabled: bool = False
    dense_x_enabled: bool = False
    """Dense-X / propositions: rewrite each chunk into standalone factual
    statements and index those. Best precision of the multi-rep options, and
    the most expensive (one LLM call per chunk)."""

    @property
    def multirep_enabled(self) -> bool:
        """Whether the index holds *derived* units rather than source text.

        Defined once and read from both sides, which is the point. All three
        representations index something that stands in for a chunk — a child, a
        summary, a proposition — and every one of them needs the query side to
        resolve the stand-in back to the source before the generator sees it. The
        index side had this predicate inline and the query side did not have it at
        all, so switching a representation on produced an index whose retriever
        did not know the representation existed.
        """
        return (
            self.parent_document_enabled or self.summary_index_enabled or self.dense_x_enabled
        )

    # --- RAPTOR ----------------------------------------------------------
    raptor_enabled: bool = False
    raptor_max_levels: int = 3
    raptor_min_cluster_size: int = 2
    raptor_max_cluster_size: int = 12
    raptor_umap_neighbors: int = 10
    raptor_umap_components: int = 8
    raptor_gmm_threshold: float = 0.1
    """Soft-clustering probability floor: a chunk may belong to several
    clusters, which is the point — topics overlap."""
    raptor_collapse_tree: bool = True
    """Query all levels at once rather than traversing top-down. Simpler and
    empirically better in the RAPTOR paper."""

    # --- ingest pipeline -------------------------------------------------
    batch_size: int = 128
    max_concurrent_documents: int = 8
    document_window: int = 512
    """How many documents to load before indexing any of them, when ingesting a
    directory.

    The chunk stream is bounded, but the *document* list was not: loading a
    directory materialized every document's text before the first vector was
    written, so peak memory scaled with the corpus even though the module's memory
    policy claims a bound independent of it. At 100k documents the document list is
    the larger number, not the chunk stream.

    512 is chosen so anything smaller behaves exactly as before — one window, one
    pass — while a large corpus is held a window at a time. Only directory ingests
    stream; an explicit list of documents is already in the caller's memory."""
    skip_unchanged: bool = True
    """Checksum comparison before embedding. Turns a full re-ingest into a
    no-op for unchanged documents."""
    dedupe_chunks: bool = True
    dedupe_threshold: float = 0.98


class GraphSettings(BaseModel):
    """GraphRAG. Off by default: graph construction costs one LLM call per
    chunk, so it is opt-in per corpus."""

    enabled: bool = False
    extract_entities: bool = True
    entity_types: list[str] = Field(
        default_factory=lambda: [
            "PERSON",
            "ORGANIZATION",
            "LOCATION",
            "PRODUCT",
            "EVENT",
            "CONCEPT",
            "TECHNOLOGY",
            "DATE",
        ]
    )
    max_gleanings: int = 1
    """Extra extraction passes asking "what did you miss?". Each pass finds
    fewer entities at the same cost, so 1 is usually the right trade."""
    extraction_batch_size: int = 8

    resolve_entities: bool = True
    resolution_threshold: float = 0.92
    """Embedding similarity above which two entity names are merged. Without
    resolution the graph fragments into near-duplicate nodes and traversal
    stops finding anything."""

    detect_communities: bool = True
    community_algorithm: Literal["leiden", "louvain", "label_propagation"] = "leiden"
    leiden_resolution: float = 1.0
    max_community_levels: int = 3
    min_community_size: int = 3
    summarize_communities: bool = True
    community_summary_max_tokens: int = 500

    # --- search modes ----------------------------------------------------
    local_search_hops: int = 2
    local_search_top_entities: int = 10
    local_search_top_chunks: int = 10
    global_search_top_communities: int = 8

    multihop_enabled: bool = True
    multihop_max_iterations: int = 3
    """IRCoT-style retrieve-reason-retrieve. Cap it: each iteration is a full
    retrieval plus an LLM call, and gains flatten after ~3."""
    multihop_max_path_length: int = 4
    multihop_beam_width: int = 5
    multihop_stop_on_sufficient: bool = True
    """Ask the model whether the evidence already answers the question and
    stop early if so — most questions need one hop, not three."""


class RetrievalSettings(BaseModel):
    top_k: int = 10
    """What the generator sees."""
    fetch_k: int = 50
    """What each retriever fetches before fusion and reranking. Recall is set
    here; precision is set by the reranker. Fetching only ``top_k`` and then
    reranking cannot recover a document the first stage missed."""

    # --- hybrid ----------------------------------------------------------
    hybrid_enabled: bool = True
    use_dense: bool = True
    use_sparse: bool = True
    use_fulltext: bool = False
    server_side_fusion: bool = True
    """Qdrant's Query API prefetch+fusion does dense+sparse in ONE round trip.
    Client-side fusion needs two, plus a merge in Python."""
    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    """The RRF constant from the original paper. Larger flattens the
    contribution of rank position."""
    fusion_weights: dict[str, float] = Field(
        default_factory=lambda: {"dense": 1.0, "sparse": 0.7, "colbert": 1.0, "fulltext": 0.5}
    )

    # --- reranking -------------------------------------------------------
    rerank_enabled: bool = True
    rerank_top_k: int = 20
    reranker: Literal["cross_encoder", "rankgpt", "colbert", "none"] = "cross_encoder"
    rerank_batch_size: int = 32
    rankgpt_window: int = 10
    rankgpt_step: int = 5
    """Sliding-window listwise reranking: lets a 10-doc window rank 50 docs
    without exceeding context."""
    colbert_rerank: bool = False

    # --- noise handling --------------------------------------------------
    score_threshold: float | None = None
    relative_score_cutoff: float | None = 0.35
    """Drop anything below 35% of the top score. Adapts to the query instead
    of hard-coding an absolute similarity floor that is wrong per-corpus."""
    dedupe_enabled: bool = True
    near_dupe_threshold: float = 0.93
    mmr_enabled: bool = False
    mmr_lambda: float = 0.6
    """0 = maximum diversity, 1 = pure relevance."""
    reorder_lost_in_middle: bool = True
    """Models attend most to the beginning and end of context. Placing the
    strongest evidence at both ends measurably improves answer accuracy."""

    # --- compression -----------------------------------------------------
    compression_enabled: bool = False
    compressor: Literal["extract", "embedding_filter", "sentence", "both", "none"] = (
        "embedding_filter"
    )
    """``sentence`` selects :class:`~ragorc.retrieve.compress.SentenceLevelCompressor`,
    which `build_compressor` has always supported and this Literal used to reject —
    the one option documented in the README that no configuration could reach."""
    compression_ratio: float = 0.5

    # --- CRAG ------------------------------------------------------------
    crag_enabled: bool = False
    crag_grade_top_k: int = 5
    crag_relevance_threshold: float = 0.6
    crag_web_fallback: bool = True
    web_search_provider: Literal["tavily", "ddgs", "none"] = "ddgs"
    web_search_results: int = 5

    parent_expansion: bool = True
    sentence_window_size: int = 3
    per_store_timeout_s: float = 10.0
    """A slow store is dropped rather than allowed to define the request's
    latency. This is the difference between p99 and p99-of-the-slowest-store."""
    max_concurrent_retrievers: int = 8


class GenerationSettings(BaseModel):
    prompt_name: str = "answer_default"
    """Must name a prompt registered in ``ragorc.llm.prompts``; validated at
    startup rather than discovered as a KeyError on the first request."""
    cite_sources: bool = True
    citation_style: Literal["inline", "footnote", "json"] = "inline"

    # --- hallucination control -------------------------------------------
    check_groundedness: bool = True
    groundedness_method: Literal["llm", "nli", "both"] = "llm"
    groundedness_threshold: float = 0.7
    verify_citations: bool = True
    """Confirm each cited span actually exists in the cited chunk. Catches the
    most common fabrication: a real-looking quote from a real document that the
    document does not contain."""
    decompose_claims: bool = False
    """Split the answer into atomic claims and grade each. Highest-fidelity
    check available, and the most expensive."""

    self_consistency_samples: int = 1
    """>1 samples the answer N times and measures agreement. Use for
    high-stakes answers; it multiplies cost by N."""
    self_consistency_threshold: float = 0.6

    allow_abstention: bool = True
    abstain_message: str = (
        "I could not find enough supporting information in the available sources "
        "to answer this reliably."
    )
    min_context_chunks: int = 1

    # --- Self-RAG / RRR loops --------------------------------------------
    self_rag_enabled: bool = False
    self_rag_max_retries: int = 2
    rrr_enabled: bool = False
    rrr_max_rewrites: int = 2

    stream: bool = False
    max_answer_tokens: int = 1024
    reserved_output_tokens: int = 1200


_COLBERT_RERANKER_NAMES: frozenset[str] = frozenset({"colbert", "late_interaction", "maxsim"})
"""The user-facing spellings ``build_reranker`` resolves to :class:`ColBERTReranker`.
Shared so a name added there cannot become a name this predicate does not know."""


class CostSettings(BaseModel):
    track_costs: bool = True
    max_cost_per_query_usd: float | None = 0.50
    """Hard ceiling. Exceeding it raises ``BudgetExceeded`` rather than
    quietly running up a bill on a pathological query."""
    max_llm_calls_per_query: int = 40
    max_tokens_per_query: int | None = 200_000

    # --- ingest ----------------------------------------------------------
    max_llm_calls_per_ingest: int | None = None
    max_cost_per_ingest_usd: float | None = None
    max_tokens_per_ingest: int | None = None
    """Ceilings for an ingest, which are separate because an ingest is not a query.

    The HTTP ingest route ran the whole corpus inside the *per-query* ledger, so a
    60-document corpus with RAPTOR on stopped after ``max_llm_calls_per_query``
    (40) documents and reported success:

        documents that got a RAPTOR summary: 40 of 60
        warnings: ['raptor stage disabled: LLM call budget exhausted']

    ``None`` means "bounded by the corpus, not by a request ceiling", which is the
    honest default: an ingest's size is known in advance, and
    :meth:`~ragorc.index.raptor.RaptorIndexer.estimate_llm_calls` forecasts and
    refuses an over-budget build *before* the first call rather than halfway
    through. Picking an arbitrary larger number here would only move the same
    silent truncation to a different corpus size.

    Set them when an ingest is caller-triggered and you need a hard stop. Every
    stage now propagates :class:`~ragorc.core.errors.BudgetExceeded` instead of
    degrading per chunk, so a ceiling that is reached fails the run visibly."""
    price_table_path: str = ""
    refresh_prices: bool = True
    """Pull live per-model prices from OpenRouter's ``/models`` endpoint so
    cost accounting stays correct as prices change."""


class ObservabilitySettings(BaseModel):
    log_level: str = "INFO"
    log_json: bool = True
    trace_enabled: bool = True
    log_prompts: bool = False
    """Off by default: prompts contain retrieved customer data."""
    slow_query_ms: float = 5000.0


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"  # noqa: S104 - containers need to bind all interfaces
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    api_keys: list[str] = Field(default_factory=list)
    """Empty = open. Set at least one before exposing the service."""
    api_key_tenants: dict[str, str] = Field(default_factory=dict)
    """Bind an API key to the one tenant it may read: ``{key: tenant_id}``.

    Without a binding, ``security.enforce_tenant_isolation`` only enforces that a
    request *names* a tenant — not that it owns it — so any authenticated caller
    can read any tenant by putting a different id in the request body. A key
    listed here may use its own tenant and no other; a key absent from this map
    stays unrestricted, which is what keeps an existing single-tenant deployment
    working unchanged.

    Keys are written out in full because that is what the operator has in hand;
    they are hashed to the same principal form the audit log uses before being
    compared, and never logged.
    """
    request_timeout_s: float = 120.0
    max_body_bytes: int = 10_000_000

    @model_validator(mode="after")
    def _bindings_reference_known_keys(self) -> ServerSettings:
        """A binding for a key that cannot authenticate is a typo, not a policy.

        It fails open in the most misleading way available — the operator reads
        the config and sees the tenant restricted, while the key that is actually
        in use is unbound — so it is rejected at load rather than at the first
        cross-tenant read.
        """
        unknown = sorted(set(self.api_key_tenants) - set(self.api_keys))
        if unknown:
            raise ValueError(
                f"server.api_key_tenants binds {len(unknown)} key(s) that are not in "
                "server.api_keys, so they can never authenticate"
            )
        return self


class Settings(BaseSettings):
    """Root configuration object.

    ``Settings()`` reads the environment; ``Settings(llm={"model": ...})``
    overrides programmatically; both can be mixed.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGORC_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "ragorc"
    environment: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    tenant_id: str | None = None

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @field_validator("environment")
    @classmethod
    def _prod_hardening(cls, v: str) -> str:
        return v

    @property
    def late_interaction_needed(self) -> bool:
        """Whether anything in this deployment will ask for ColBERT vectors.

        Three consumers, and each wiring knew about a different subset. The
        factory already read two of them (``enable_late_interaction`` or
        ``colbert_rerank``); the pipeline builder and the server read only the
        first, so turning on ``retrieval.colbert_rerank`` alone built no embedder,
        left ``QdrantStore._has_colbert`` false and silently dropped the stage:

            enable_late_interaction = False | colbert_rerank = True
            builder.late_embedder -> None
            store._has_colbert -> False
            search() want_colbert -> False

        The third consumer — ``retrieval.reranker == "colbert"`` — was in nobody's
        condition, so selecting the ColBERT reranker by name built a second,
        uncached embedder inside the reranker itself.
        """
        return (
            self.embedding.enable_late_interaction
            or self.retrieval.colbert_rerank
            or self.retrieval.reranker in _COLBERT_RERANKER_NAMES
        )

    def model_post_init(self, __context: Any) -> None:
        # Keep the two vector dimensions in lockstep. A mismatch between
        # Qdrant and pgvector surfaces as an opaque insert error much later.
        if self.embedding.dense_dimension:
            self.postgres.vector_dimension = self.embedding.dense_dimension
        # Reserve output tokens in one place.
        self.generation.reserved_output_tokens = max(
            self.generation.reserved_output_tokens, self.generation.max_answer_tokens + 128
        )
        # Fail at startup on an unknown prompt name. Discovering it as a
        # KeyError inside the generator means the first real request 500s.
        try:
            from ragorc.llm.prompts import PROMPTS

            if self.generation.prompt_name not in PROMPTS:
                from ragorc.core.errors import ConfigError

                raise ConfigError(
                    f"unknown generation.prompt_name {self.generation.prompt_name!r}",
                    known=sorted(PROMPTS),
                )
        except ImportError:  # pragma: no cover - prompts is a base module
            pass

        if self.environment == "prod":
            # Production must not silently run with the guards off.
            self.security.enable_sql_guard = True
            self.security.enable_cypher_guard = True
            self.observability.log_prompts = False

    def summary(self) -> dict[str, Any]:
        """Redacted snapshot for logs and the ``/health`` endpoint."""
        return {
            "environment": self.environment,
            "llm": {
                "model": self.llm.model,
                "fast_model": self.llm.fast_model,
                "has_key": bool(self.llm.api_key.get_secret_value()),
            },
            "embedding": {
                "provider": self.embedding.provider,
                "dense_model": self.embedding.dense_model,
                "sparse_model": self.embedding.sparse_model,
            },
            "stores": {
                "qdrant": self.qdrant.url,
                "postgres": self.postgres.dsn.get_secret_value().split("@")[-1],
                "neo4j": self.neo4j.uri,
            },
            "features": {
                "hybrid": self.retrieval.hybrid_enabled,
                "rerank": self.retrieval.rerank_enabled,
                "crag": self.retrieval.crag_enabled,
                "self_rag": self.generation.self_rag_enabled,
                "graphrag": self.graph.enabled,
                "multihop": self.graph.multihop_enabled,
                "chunking": self.indexing.chunking_strategy.value,
                "raptor": self.indexing.raptor_enabled,
            },
        }


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in
    tests that need a different configuration."""
    return Settings()
