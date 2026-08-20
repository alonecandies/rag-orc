"""Structural interfaces for every pluggable part of the system.

These are :class:`typing.Protocol` classes, not ABCs, on purpose: a component
satisfies an interface by *shape*, so third-party objects (a LangChain
retriever, your own store) work without inheriting from us and without an
adapter registry. ``@runtime_checkable`` lets the factory layer verify at
config-load time instead of failing mid-request.

Everything is async. There is no sync variant: a RAG query fans out to three
databases plus N LLM calls, and the whole point is to overlap them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from ragorc.core.models import (
    Answer,
    Chunk,
    Community,
    Document,
    Entity,
    FloatArray,
    GradeLabel,
    GraphPath,
    Query,
    Relation,
    RetrievalResult,
    RouteDecision,
    ScoredChunk,
    SparseVector,
    Usage,
)

__all__ = [
    "LLM",
    "BatchStructuredLLM",
    "Cache",
    "Compressor",
    "ContextPacker",
    "DenseEmbedder",
    "Generator",
    "Grader",
    "GraphStore",
    "LateInteractionEmbedder",
    "Loader",
    "QueryConstructor",
    "QueryTranslator",
    "RelationalStore",
    "Reranker",
    "Retriever",
    "Router",
    "SparseEmbedder",
    "Splitter",
    "VectorStore",
]


# ---------------------------------------------------------------------------
# Language models
# ---------------------------------------------------------------------------
@runtime_checkable
class LLM(Protocol):
    """A chat model. Implemented by :class:`ragorc.llm.openrouter.OpenRouterLLM`.

    ``complete`` returns the text plus a :class:`Usage` so cost accounting is
    impossible to forget — every call site gets the bill.

    ``structured`` is the workhorse: the routers, graders, self-query
    constructor, entity extractor and every classifier in the pipeline are
    schema-constrained calls, not free text to be regex-parsed.
    """

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> tuple[str, Usage]: ...

    async def structured(
        self,
        prompt: str,
        schema: type,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Usage]:
        """Return an instance of ``schema`` (a pydantic ``BaseModel``)."""
        ...

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...

    async def batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, Usage]]:
        """Concurrent fan-out under a shared semaphore. Used by every
        map-style stage (multi-query, per-chunk grading, RAPTOR summaries)."""
        ...


@runtime_checkable
class BatchStructuredLLM(Protocol):
    """An :class:`LLM` that can fan out schema-constrained calls itself.

    Deliberately separate from ``LLM``: a fan-out is a convenience over
    ``structured``, not a new capability, and every caller here has a working
    sequential fallback. Requiring it would tax anyone plugging in their own
    client for a method they can get for free. Callers narrow with ``isinstance``
    — cheaper than the ``getattr`` probe it replaces, and typed.
    """

    async def batch_structured(
        self,
        prompts: Sequence[str],
        schema: type,
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Any | None, Usage]]:
        """One ``(instance | None, Usage)`` per prompt, in order. A failed item is
        ``None`` rather than an exception: one unparseable grade must not lose the
        other forty-nine."""
        ...


# ---------------------------------------------------------------------------
# Embeddings — three kinds, because hybrid search needs all three
# ---------------------------------------------------------------------------
@runtime_checkable
class DenseEmbedder(Protocol):
    """Standard pooled embeddings.

    ``embed_documents`` and ``embed_query`` are separate because asymmetric
    models (E5, BGE, GTE) require different instruction prefixes for the two
    sides; using one method for both silently costs several points of recall.
    """

    dimension: int
    model_name: str
    max_tokens: int

    async def embed_documents(self, texts: Sequence[str]) -> list[FloatArray]: ...

    async def embed_query(self, text: str) -> FloatArray: ...

    async def embed_queries(self, texts: Sequence[str]) -> list[FloatArray]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """BM25 or SPLADE, expressed as Qdrant sparse vectors so lexical and
    semantic search happen in a single server-side query."""

    model_name: str
    is_lexical: bool

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    async def embed_query(self, text: str) -> SparseVector: ...


@runtime_checkable
class LateInteractionEmbedder(Protocol):
    """ColBERT-style token-level embeddings, shape ``(n_tokens, dim)``.

    Also the engine behind late chunking (ADR-0002): the per-token output is
    pooled over chunk spans to give context-aware chunk vectors.
    """

    dimension: int
    model_name: str

    async def embed_documents(self, texts: Sequence[str]) -> list[FloatArray]: ...

    async def embed_query(self, text: str) -> FloatArray: ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder or listwise LLM reranker.

    Returns ``(index, score)`` pairs against the *input* order so the caller
    can attribute scores without relying on object identity.
    """

    model_name: str

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]: ...


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorStore(Protocol):
    """Qdrant. ``search`` is expected to run dense + sparse + late-interaction
    as one server-side hybrid query, not three client-side ones."""

    async def ensure_collection(self, *, recreate: bool = False) -> None: ...

    async def upsert(self, chunks: Sequence[Chunk]) -> int: ...

    async def search(
        self,
        query: Query,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]: ...

    async def get(self, ids: Sequence[str], *, with_vectors: bool = False) -> list[Chunk]:
        """``with_vectors`` is part of the contract, not a Qdrant extra.

        :func:`ragorc.retrieve.graph.load_chunks` always passes it, because DRIFT
        search reuses the vector store as its
        :class:`~ragorc.retrieve.graph.ChunkStore` and the similarity term needs
        the vectors back. A store that omits the keyword raises ``TypeError``
        there rather than failing anywhere a checker would have caught.
        """
        ...

    async def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> int: ...

    async def count(self, **kwargs: Any) -> int: ...

    async def close(self) -> None: ...


@runtime_checkable
class RelationalStore(Protocol):
    """Postgres. Doubles as a pgvector store, a full-text index, and the
    Text-to-SQL execution target — with a read-only guard in front."""

    async def execute_readonly(
        self, sql: str, params: Sequence[Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    async def schema_summary(self, *, refresh: bool = False) -> str:
        """Compact DDL summary for the Text-to-SQL prompt. Cached: re-reading
        ``information_schema`` on every query is a needless round trip."""
        ...

    async def fulltext_search(
        self, query: str, *, top_k: int = 10, **kwargs: Any
    ) -> list[ScoredChunk]: ...

    async def close(self) -> None: ...


@runtime_checkable
class GraphStore(Protocol):
    """Neo4j. Serves Text-to-Cypher, GraphRAG local/global search and
    multi-hop path finding."""

    async def execute_readonly(
        self, cypher: str, params: dict[str, Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]: ...

    async def schema_summary(self, *, refresh: bool = False) -> str: ...

    async def upsert_entities(self, entities: Sequence[Entity]) -> int: ...

    async def upsert_relations(self, relations: Sequence[Relation]) -> int: ...

    async def neighbors(
        self, names: Sequence[str], *, hops: int = 1, limit: int = 50
    ) -> tuple[list[Entity], list[Relation]]: ...

    async def paths(
        self, start: Sequence[str], end: Sequence[str], *, max_hops: int = 3, limit: int = 10
    ) -> list[GraphPath]: ...

    async def communities(self, *, level: int | None = None) -> list[Community]: ...

    async def close(self) -> None: ...


@runtime_checkable
class Cache(Protocol):
    """Tiered cache. ``get``/``set`` are bytes-in/bytes-out; callers serialize
    with orjson so the cache never needs to know about our types."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, *, ttl: float | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def clear(self, prefix: str | None = None) -> int: ...


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
@runtime_checkable
class Loader(Protocol):
    """Turns a path/URL/bytes into documents."""

    async def load(self, source: Any, **kwargs: Any) -> list[Document]: ...


@runtime_checkable
class Splitter(Protocol):
    """Document -> chunks. Split boundaries only; embedding happens later so
    late chunking stays possible."""

    async def split(self, document: Document) -> list[Chunk]: ...

    async def split_many(self, documents: Sequence[Document]) -> list[Chunk]: ...


@runtime_checkable
class QueryTranslator(Protocol):
    """Multi-query, RAG-Fusion, step-back, decomposition, HyDE.

    Returns an enriched :class:`Query` rather than a bare list of strings so
    downstream stages can see *why* each variant exists.
    """

    name: str

    async def translate(self, query: Query) -> tuple[Query, Usage]: ...


@runtime_checkable
class Router(Protocol):
    """Chooses datastores (logical routing) and/or prompt (semantic routing)."""

    name: str

    async def route(self, query: Query) -> tuple[RouteDecision, Usage]: ...


@runtime_checkable
class QueryConstructor(Protocol):
    """Natural language -> SQL / Cypher / metadata filter.

    ``construct`` returns the artifact *and* whatever the guard needs to
    validate it; execution is a separate step so nothing runs unvalidated.
    """

    name: str
    target: str

    async def construct(self, query: Query, **kwargs: Any) -> tuple[Any, Usage]: ...


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a query into scored chunks."""

    name: str

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Compressor(Protocol):
    """Post-retrieval refinement: extract, filter or summarize to fit budget."""

    name: str

    async def compress(
        self, query: Query, chunks: Sequence[ScoredChunk], **kwargs: Any
    ) -> tuple[list[ScoredChunk], Usage]: ...


@runtime_checkable
class Grader(Protocol):
    """Relevance / groundedness / utility grading (CRAG and Self-RAG)."""

    name: str

    async def grade(
        self, query: Query, target: Any, **kwargs: Any
    ) -> tuple[GradeLabel, float, Usage]: ...


@runtime_checkable
class ContextPacker(Protocol):
    """Fits chunks into a token budget and decides their order."""

    async def pack(
        self, query: Query, chunks: Sequence[ScoredChunk], *, budget: int
    ) -> tuple[list[ScoredChunk], str]: ...


@runtime_checkable
class Generator(Protocol):
    """Produces the final answer with citations."""

    name: str

    async def generate(self, query: Query, retrieval: RetrievalResult, **kwargs: Any) -> Answer: ...

    async def stream(
        self, query: Query, retrieval: RetrievalResult, **kwargs: Any
    ) -> AsyncIterator[str]: ...
