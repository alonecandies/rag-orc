"""Core data model.

Design note — why dataclasses and not pydantic
----------------------------------------------
The objects in this module are *hot path*: a single ingest run can create
millions of :class:`Chunk` instances and a single query creates thousands of
:class:`ScoredChunk` instances.  ``@dataclass(slots=True)`` instantiates
roughly 2-3x faster than a pydantic ``BaseModel`` and, because ``__slots__``
removes the per-instance ``__dict__``, uses ~40% less memory per object.

Pydantic is still used — but only at *boundaries*, where its Rust validator
earns its cost: settings (:mod:`ragorc.core.settings`), LLM structured output
(:mod:`ragorc.core.schemas`) and the HTTP API (:mod:`ragorc.server`).

Vectors are stored as ``numpy`` arrays rather than ``list[float]``: a 1024-dim
float32 array is 4 KiB, the equivalent Python list of floats is ~40 KiB.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "Answer",
    "Chunk",
    "ChunkingStrategy",
    "Citation",
    "Community",
    "DataStore",
    "Document",
    "Entity",
    "FusionMethod",
    "GradeLabel",
    "GraphPath",
    "Modality",
    "Query",
    "Relation",
    "RetrievalResult",
    "RetrievalSource",
    "RouteDecision",
    "ScoredChunk",
    "SparseVector",
    "StepTrace",
    "Usage",
    "utcnow",
]

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]


def utcnow() -> datetime:
    """Timezone-aware ``now``. Naive datetimes are a recurring source of bugs."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DataStore(str, enum.Enum):
    """The three retrieval backends plus the two synthetic ones."""

    VECTOR = "vector"  # Qdrant
    RELATIONAL = "relational"  # Postgres
    GRAPH = "graph"  # Neo4j
    WEB = "web"  # CRAG active-retrieval fallback
    NONE = "none"  # answer without retrieval


class RetrievalSource(str, enum.Enum):
    """Which concrete retriever produced a result. Used for fusion + telemetry."""

    DENSE = "dense"
    SPARSE = "sparse"  # SPLADE learned sparse
    BM25 = "bm25"  # lexical
    COLBERT = "colbert"  # late interaction
    FULLTEXT = "fulltext"  # Postgres tsvector / Neo4j fulltext index
    SQL = "sql"
    CYPHER = "cypher"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"
    GRAPH_PATH = "graph_path"
    RAPTOR = "raptor"
    PARENT = "parent"
    WEB = "web"
    CACHE = "cache"
    FUSED = "fused"


class ChunkingStrategy(str, enum.Enum):
    """See ADR-0002. The ladder degrades LATE -> CONTEXTUAL -> EARLY."""

    EARLY = "early"  # chunk, then embed each chunk in isolation
    LATE = "late"  # embed whole doc once, then pool per chunk span
    CONTEXTUAL = "contextual"  # LLM-written situating prefix, then embed
    AUTO = "auto"  # pick the best the provider can support


class FusionMethod(str, enum.Enum):
    RRF = "rrf"  # reciprocal rank fusion
    DBSF = "dbsf"  # distribution-based score fusion
    WEIGHTED = "weighted"  # normalized weighted score sum
    RELATIVE_SCORE = "relative"  # min-max normalized then max
    MAX = "max"


class GradeLabel(str, enum.Enum):
    """Output of the CRAG / Self-RAG graders."""

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class Modality(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    CODE = "code"
    IMAGE_CAPTION = "image_caption"
    SUMMARY = "summary"
    PROPOSITION = "proposition"


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------
@dataclass(slots=True, eq=False)
class SparseVector:
    """A sparse vector in Qdrant's ``indices``/``values`` form.

    Produced by SPLADE (learned) or BM25 (lexical) embedders. Kept as numpy
    arrays so ``to_qdrant`` is a zero-copy ``tolist`` and dot products stay in C.
    """

    indices: IntArray
    values: FloatArray

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"sparse vector length mismatch: {len(self.indices)} indices "
                f"vs {len(self.values)} values"
            )

    @classmethod
    def from_dict(cls, mapping: dict[int, float]) -> SparseVector:
        if not mapping:
            return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
        vals = np.fromiter(mapping.values(), dtype=np.float32, count=len(mapping))
        return cls(keys, vals)

    def dot(self, other: SparseVector) -> float:
        """Sparse dot product via sorted intersection — O(n+m), no Python loop."""
        common, ia, ib = np.intersect1d(
            self.indices, other.indices, assume_unique=True, return_indices=True
        )
        if common.size == 0:
            return 0.0
        return float(np.dot(self.values[ia], other.values[ib]))

    def top_k(self, k: int) -> SparseVector:
        """Prune to the k largest weights — cuts payload size on long documents."""
        if k >= len(self.values):
            return self
        keep = np.argpartition(self.values, -k)[-k:]
        return SparseVector(self.indices[keep], self.values[keep])

    def __len__(self) -> int:
        return len(self.indices)


# ---------------------------------------------------------------------------
# Documents & chunks
# ---------------------------------------------------------------------------
@dataclass(slots=True, eq=False)
class Document:
    """A source document, before splitting."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str | None = None
    title: str | None = None
    modality: Modality = Modality.TEXT
    checksum: str | None = None
    """Content hash. Ingest is idempotent on ``(id, checksum)``: unchanged
    documents are skipped without re-embedding, which is the single biggest
    cost saving in a re-ingest."""
    created_at: datetime = field(default_factory=utcnow)
    tenant_id: str | None = None

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(slots=True, eq=False)
class Chunk:
    """A retrievable unit.

    Holds up to three vector representations simultaneously because the hybrid
    retriever queries all of them in a single Qdrant round trip:

    * ``dense``  — semantic similarity (default 1024-dim float32)
    * ``sparse`` — lexical/learned-sparse for exact-term matching
    * ``multi``  — ``(n_tokens, dim)`` late-interaction matrix for ColBERT MaxSim
    """

    id: str
    content: str
    document_id: str = ""
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- multi-representation indexing -----------------------------------
    parent_id: str | None = None
    """Set by the parent-document retriever: this chunk is a small, precise
    search key whose *parent* is the text actually handed to the LLM."""
    level: int = 0
    """RAPTOR tree level. 0 = leaf (original text), >0 = LLM cluster summary."""
    children_ids: tuple[str, ...] = ()
    modality: Modality = Modality.TEXT
    contextual_prefix: str | None = None
    """Contextual-retrieval blurb, prepended before embedding but *not* shown
    to the generator (it would duplicate information already in the prompt)."""

    # --- vectors ----------------------------------------------------------
    dense: FloatArray | None = None
    sparse: SparseVector | None = None
    multi: FloatArray | None = None

    # --- bookkeeping ------------------------------------------------------
    token_count: int | None = None
    tenant_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def embed_text(self) -> str:
        """Exactly the text that gets embedded — prefix included when present."""
        if self.contextual_prefix:
            return f"{self.contextual_prefix}\n\n{self.content}"
        return self.content

    def payload(self) -> dict[str, Any]:
        """Flat payload for the vector store. Vectors are excluded on purpose."""
        p: dict[str, Any] = {
            "content": self.content,
            "document_id": self.document_id,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "level": self.level,
            "modality": self.modality.value,
            **self.metadata,
        }
        if self.parent_id:
            p["parent_id"] = self.parent_id
        if self.children_ids:
            p["children_ids"] = list(self.children_ids)
        if self.tenant_id:
            p["tenant_id"] = self.tenant_id
        if self.token_count is not None:
            p["token_count"] = self.token_count
        return p

    @classmethod
    def from_payload(cls, chunk_id: str, payload: dict[str, Any]) -> Chunk:
        payload = dict(payload)
        known = {
            "content",
            "document_id",
            "index",
            "start_char",
            "end_char",
            "level",
            "modality",
            "parent_id",
            "children_ids",
            "tenant_id",
            "token_count",
        }
        return cls(
            id=chunk_id,
            content=payload.get("content", ""),
            document_id=payload.get("document_id", ""),
            index=int(payload.get("index", 0)),
            start_char=int(payload.get("start_char", 0)),
            end_char=int(payload.get("end_char", 0)),
            level=int(payload.get("level", 0)),
            modality=Modality(payload.get("modality", "text")),
            parent_id=payload.get("parent_id"),
            children_ids=tuple(payload.get("children_ids") or ()),
            tenant_id=payload.get("tenant_id"),
            token_count=payload.get("token_count"),
            metadata={k: v for k, v in payload.items() if k not in known},
        )


@dataclass(slots=True, eq=False)
class ScoredChunk:
    """A chunk with a relevance score and full provenance.

    ``component_scores`` keeps every contributing retriever's score so fusion,
    reranking and debugging are all inspectable after the fact. Without it,
    "why did this document rank third?" is unanswerable.
    """

    chunk: Chunk
    score: float
    source: RetrievalSource = RetrievalSource.DENSE
    rank: int = 0
    component_scores: dict[str, float] = field(default_factory=dict)
    explain: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.chunk.id)

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def content(self) -> str:
        return self.chunk.content

    def with_score(self, score: float, source: RetrievalSource | None = None) -> ScoredChunk:
        return ScoredChunk(
            chunk=self.chunk,
            score=score,
            source=source or self.source,
            rank=self.rank,
            component_scores=dict(self.component_scores),
            explain=dict(self.explain),
        )


# ---------------------------------------------------------------------------
# Query & routing
# ---------------------------------------------------------------------------
@dataclass(slots=True, eq=False)
class Query:
    """A question in flight, enriched as it moves through the pipeline."""

    text: str
    original: str | None = None
    """The user's untouched input, preserved across rewrites so the final
    answer can be graded against what was actually asked."""
    variants: tuple[str, ...] = ()
    """Multi-query / step-back / decomposition outputs."""
    hypothetical: str | None = None  # HyDE pseudo-document
    filters: dict[str, Any] = field(default_factory=dict)
    top_k: int = 10
    dense: FloatArray | None = None
    sparse: SparseVector | None = None
    multi: FloatArray | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.original is None:
            self.original = self.text

    @property
    def all_texts(self) -> tuple[str, ...]:
        """The query plus every variant, de-duplicated, order preserved."""
        seen: dict[str, None] = {self.text: None}
        for v in self.variants:
            seen.setdefault(v, None)
        return tuple(seen)


@dataclass(slots=True, eq=False)
class RouteDecision:
    """Output of the routing layer (logical and/or semantic)."""

    stores: tuple[DataStore, ...]
    prompt_name: str | None = None
    confidence: float = 1.0
    reasoning: str | None = None
    method: str = "logical"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(slots=True, eq=False)
class RetrievalResult:
    """Everything the retrieval stage produced, with per-store diagnostics."""

    chunks: list[ScoredChunk] = field(default_factory=list)
    per_store: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    """A store that failed is recorded here rather than raising: one dead
    backend degrades the answer, it does not fail the request."""
    grade: GradeLabel | None = None
    total_candidates: int = 0

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterable[ScoredChunk]:
        return iter(self.chunks)

    def texts(self) -> list[str]:
        return [c.content for c in self.chunks]

    def top(self, k: int) -> list[ScoredChunk]:
        return self.chunks[:k]


@dataclass(slots=True, eq=False)
class Citation:
    """A span-level attribution. ``support`` is the entailment score of the
    claim against the quoted span, which is what makes citations verifiable
    rather than decorative."""

    chunk_id: str
    document_id: str = ""
    quote: str = ""
    claim: str = ""
    support: float = 1.0
    source: str | None = None
    start_char: int | None = None
    end_char: int | None = None


@dataclass(slots=True, eq=False)
class Usage:
    """Token and cost accounting for one LLM call or one whole request."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    calls: int = 0
    cached: int = 0
    """Number of calls served from cache — these cost nothing and are the
    headline number in the cost report."""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            model=self.model or other.model,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
            calls=self.calls + other.calls,
            cached=self.cached + other.cached,
        )

    @classmethod
    def sum(cls, items: Iterable[Usage]) -> Usage:
        total = cls()
        for item in items:
            total = total + item
        return total


@dataclass(slots=True, eq=False)
class StepTrace:
    """One pipeline step, for the trace returned alongside every answer."""

    name: str
    duration_ms: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    usage: Usage | None = None


@dataclass(slots=True, eq=False)
class Answer:
    """The terminal object of the pipeline."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[ScoredChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    grounded: bool = True
    groundedness: float = 1.0
    confidence: float = 1.0
    abstained: bool = False
    """True when the guardrails decided that saying "I don't know" beats
    guessing. An abstention is a success, not a failure."""
    abstain_reason: str | None = None
    trace: list[StepTrace] = field(default_factory=list)
    route: RouteDecision | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph model (GraphRAG)
# ---------------------------------------------------------------------------
@dataclass(slots=True, eq=False)
class Entity:
    """A node extracted from text. ``name`` is the canonical (deduplicated)
    surface form; ``aliases`` keeps the variants that were merged into it."""

    name: str
    type: str = "Entity"
    description: str = ""
    aliases: tuple[str, ...] = ()
    source_chunk_ids: tuple[str, ...] = ()
    degree: int = 0
    community_id: int | None = None
    embedding: FloatArray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Case-folded identity key used for cross-chunk deduplication."""
        return self.name.strip().casefold()

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass(slots=True, eq=False)
class Relation:
    """A typed, weighted edge. ``weight`` accumulates across chunks, so an
    edge asserted by many documents outranks one asserted once."""

    source: str
    target: str
    type: str = "RELATED_TO"
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source.casefold(), self.type, self.target.casefold())

    def __hash__(self) -> int:
        return hash(self.key)


@dataclass(slots=True, eq=False)
class Community:
    """A Leiden community with its LLM-written summary — the unit of
    GraphRAG *global* search."""

    id: int
    level: int
    entity_names: tuple[str, ...] = ()
    relation_keys: tuple[tuple[str, str, str], ...] = ()
    title: str = ""
    summary: str = ""
    rank: float = 0.0
    parent_id: int | None = None
    embedding: FloatArray | None = None


@dataclass(slots=True, eq=False)
class GraphPath:
    """A multi-hop path. Used both as evidence and as the explanation of how
    two entities in the question are connected."""

    nodes: tuple[str, ...]
    relations: tuple[Relation, ...] = ()
    score: float = 0.0

    @property
    def hops(self) -> int:
        return max(len(self.nodes) - 1, 0)

    def verbalize(self) -> str:
        """Render as ``A -[TYPE]-> B -[TYPE]-> C`` for the LLM prompt."""
        if not self.relations:
            return " -> ".join(self.nodes)
        parts: list[str] = [self.nodes[0]]
        for rel, node in zip(self.relations, self.nodes[1:], strict=False):
            parts.append(f"-[{rel.type}]->")
            parts.append(node)
        return " ".join(parts)


def dedupe_scored(items: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Keep the highest-scoring occurrence of each chunk id, order preserved."""
    best: dict[str, ScoredChunk] = {}
    for item in items:
        prev = best.get(item.id)
        if prev is None or item.score > prev.score:
            best[item.id] = item
    return list(best.values())
