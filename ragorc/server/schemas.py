"""HTTP request and response bodies — the one place validation is the point.

ADR-0008 splits the data model by trust boundary: ``@dataclass(slots=True)`` in
the hot path where the fields come from our own code, pydantic where they come
from somewhere else. This module is the *somewhere else*. Every field here was
typed by a client, so every field here carries a constraint, and the constraints
are the reason the module exists rather than a decoration on top of it.

Three rules shape what follows.

**Bound everything that a caller controls the size of.** A question, a filter
dict, a path list and an inline document body are all attacker-sized inputs, and
each one feeds a stage whose cost scales with it: the question is embedded and
sent to a model, the filters become a Qdrant filter tree, the paths become
filesystem walks. An unbounded field is a denial-of-service primitive that reads
like a convenience. The ceilings here are *transport* ceilings, deliberately
looser than the policy limits in :class:`~ragorc.core.settings.SecuritySettings`:
policy belongs to configuration and can be tightened per deployment, while these
exist so that a request too large to be legitimate is rejected before it reaches
the first allocation. Where both apply, the settings value wins because it is
checked second (see :class:`~ragorc.validate.input.QueryValidator`).

**Forbid unknown fields.** ``extra="forbid"`` turns a misspelled ``tenant_id``
into a 422 instead of a query that quietly searches every tenant's data. A typo
in a field whose whole job is to scope access must never be silently discarded.

**Responses are models too.** The pipeline's terminal object is an
:class:`~ragorc.core.models.Answer` dataclass holding numpy arrays and enum
members; ``from_answer`` is the single conversion point where that becomes JSON.
Having one converter — rather than a serializer per endpoint — is what keeps the
semantic cache able to store a response and hand it back verbatim, because the
cached payload and the live payload are produced by the same code.
"""

from __future__ import annotations

import contextlib
import enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragorc.core.models import Answer, Citation, ScoredChunk, StepTrace, Usage

__all__ = [
    "MAX_CHUNK_CONTENT_CHARS",
    "MAX_FILTER_KEYS",
    "MAX_INLINE_TEXT_CHARS",
    "MAX_PATHS",
    "MAX_QUESTION_CHARS",
    "MAX_TOP_K",
    "ChunkModel",
    "CitationModel",
    "ErrorResponse",
    "EvalItem",
    "EvalMetrics",
    "EvalRequest",
    "EvalResponse",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "PipelineName",
    "QueryRequest",
    "QueryResponse",
    "RouteModel",
    "StepModel",
    "StoreHealth",
    "UsageModel",
]

# ---------------------------------------------------------------------------
# Ceilings
#
# Each of these is a transport bound with a cost behind it, not a round number.
# ---------------------------------------------------------------------------
MAX_QUESTION_CHARS = 8_000
"""Hard transport ceiling on a question.

Above ``security.max_query_length`` (default 4000) on purpose: that setting is
policy and *truncates* rather than rejects, because an over-long question is
usually a paste accident whose useful part is at the front. This bound is the
backstop for the case that is not an accident — 8k characters is already ~2k
tokens of embedding and prompt on every retrieval leg."""

MAX_TOP_K = 100
"""No reranker improves an answer by being handed more than this, and each
additional chunk is a payload from the store plus tokens in the prompt. A caller
asking for 10,000 is asking to pay for a scan."""

MAX_FILTER_KEYS = 32
"""Filter clauses become a Qdrant filter tree evaluated per candidate point.
Thirty-two predicates is far past any real metadata schema; beyond it the filter
is the query's cost centre rather than its scope."""

MAX_INLINE_TEXT_CHARS = 4_000_000
"""Inline ingest body. Matches the order of magnitude of the loaders' own file
ceiling (``MAX_FILE_BYTES``), so posting a document and loading the same document
from disk fail at the same size rather than at two surprising ones."""

MAX_PATHS = 512
"""Paths per ingest request. Each one can expand to a directory walk, so the
bound is on the number of *roots*, not the number of files."""

MAX_CHUNK_CONTENT_CHARS = 4_000
"""Per-chunk body returned to the client. Ten chunks of unbounded document text
is a multi-hundred-kilobyte response for a two-sentence answer; the id is
returned alongside so a client that wants the full body can ask for it, and
``truncated`` says when it should."""

_TENANT_PATTERN = r"^[A-Za-z0-9._:@-]{1,128}$"
"""Tenant ids reach a Qdrant payload filter, a SQL parameter and a Cypher
parameter. All three are parameterized, so this is not the injection defence —
it is the check that keeps an id from being a smuggled JSON document or a
kilobyte of whitespace."""


class PipelineName(str, enum.Enum):
    """Which composition answers the request.

    These are the graphs of :mod:`ragorc.pipeline.graphs`, not free-form strings.
    Enumerating them is what turns "unknown pipeline" into a 422 at the edge
    rather than a ``ConfigError`` raised mid-request, after the query has already
    been validated, embedded and billed.

    ``AUTO`` is the default and is not a graph of its own: it hands the choice to
    the orchestrator, which picks from the configuration (CRAG on, Self-RAG on,
    GraphRAG on) instead of asking every client to know which features this
    deployment enabled.
    """

    AUTO = "auto"
    NAIVE = "naive"
    ADAPTIVE = "adaptive"
    CRAG = "crag"
    SELF_RAG = "self_rag"
    GRAPHRAG = "graphrag"
    MULTIHOP = "multihop"
    AGENTIC = "agentic"


class _Body(BaseModel):
    """Shared configuration for every body on the wire.

    ``extra="forbid"`` is the load-bearing one; see the module docstring.
    ``str_strip_whitespace`` matters because a question of pure whitespace would
    otherwise satisfy ``min_length=1`` and reach the embedder.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
class QueryRequest(_Body):
    """One question."""

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        description="The user's question, as asked.",
    )
    tenant_id: str | None = Field(
        default=None,
        pattern=_TENANT_PATTERN,
        description=(
            "Tenant to scope retrieval to. Required when "
            "security.enforce_tenant_isolation is on — a missing tenant is "
            "refused rather than read as 'all tenants'."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=MAX_TOP_K,
        description="Chunks handed to the generator. Defaults to retrieval.top_k.",
    )
    pipeline: PipelineName = Field(
        default=PipelineName.AUTO,
        description="Graph to run. An explicit choice overrides the configured default.",
    )
    stream: bool = Field(
        default=False,
        description=(
            "Advisory on POST /query, which always returns a complete verified "
            "answer. Use POST /query/stream for token deltas."
        ),
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        max_length=MAX_FILTER_KEYS,
        description="Metadata predicates ANDed into every store's filter.",
    )

    @model_validator(mode="after")
    def _reject_tenant_in_filters(self) -> Self:
        """Refuse a tenant predicate smuggled through ``filters``.

        :func:`ragorc.security.tenancy.scope_filter` already raises on a
        *conflicting* value, but an *agreeing* one is worse than useless: it
        looks like the caller set the scope while the scope actually came from
        ``tenant_id``, so a later change to one and not the other reads as
        working. One field owns tenancy.
        """
        if "tenant_id" in self.filters:
            raise ValueError("set tenant_id as its own field, not inside filters")
        return self


class UsageModel(_Body):
    """Token and cost accounting for the whole request."""

    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    calls: int = 0
    cached_calls: int = 0

    @classmethod
    def from_usage(cls, usage: Usage) -> UsageModel:
        return cls(
            model=usage.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=round(usage.cost_usd, 6),
            latency_ms=round(usage.latency_ms, 2),
            calls=usage.calls,
            cached_calls=usage.cached,
        )


class CitationModel(_Body):
    """A span-level attribution.

    ``support`` is the entailment score of the claim against the quote, which is
    what separates a verifiable citation from a decorative one.
    """

    chunk_id: str
    document_id: str = ""
    quote: str = ""
    claim: str = ""
    support: float = 1.0
    source: str | None = None
    start_char: int | None = None
    end_char: int | None = None

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationModel:
        return cls(
            chunk_id=citation.chunk_id,
            document_id=citation.document_id,
            quote=citation.quote[:MAX_CHUNK_CONTENT_CHARS],
            claim=citation.claim[:MAX_CHUNK_CONTENT_CHARS],
            support=round(citation.support, 4),
            source=citation.source,
            start_char=citation.start_char,
            end_char=citation.end_char,
        )


class ChunkModel(_Body):
    """One retrieved chunk, as the client sees it.

    ``component_scores`` is carried through rather than collapsed into ``score``
    because once fusion has flattened several rankings into one number, "why did
    this rank third?" is unanswerable — and that question is the whole of
    retrieval debugging.
    """

    id: str
    document_id: str = ""
    content: str = ""
    truncated: bool = False
    score: float = 0.0
    source: str = ""
    rank: int = 0
    level: int = 0
    parent_id: str | None = None
    component_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_scored(cls, scored: ScoredChunk) -> ChunkModel:
        body = scored.chunk.content
        return cls(
            id=scored.chunk.id,
            document_id=scored.chunk.document_id,
            content=body[:MAX_CHUNK_CONTENT_CHARS],
            truncated=len(body) > MAX_CHUNK_CONTENT_CHARS,
            score=round(scored.score, 6),
            source=scored.source.value,
            rank=scored.rank,
            level=scored.chunk.level,
            parent_id=scored.chunk.parent_id,
            component_scores={k: round(v, 6) for k, v in scored.component_scores.items()},
            metadata=_json_safe(scored.chunk.metadata),
        )


class StepModel(_Body):
    """One traced pipeline stage."""

    name: str
    duration_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_step(cls, step: StepTrace) -> StepModel:
        return cls(
            name=step.name,
            duration_ms=round(step.duration_ms, 3),
            detail=_json_safe(step.detail),
        )


class RouteModel(_Body):
    """What the router decided, and how sure it was."""

    stores: list[str] = Field(default_factory=list)
    prompt: str | None = None
    confidence: float = 1.0
    reasoning: str | None = None
    method: str = "logical"


class QueryResponse(_Body):
    """The answer, with everything needed to check it.

    Groundedness, resolvable citations, an honest ``abstained`` flag and the cost
    ledger are not optional extras a caller has to remember to request — they are
    the contract, because an answer without them cannot be audited.
    """

    request_id: str = ""
    question: str = ""
    answer: str = ""
    citations: list[CitationModel] = Field(default_factory=list)
    chunks: list[ChunkModel] = Field(default_factory=list)
    grounded: bool = True
    groundedness: float = 1.0
    confidence: float = 1.0
    abstained: bool = False
    abstain_reason: str | None = None
    usage: UsageModel = Field(default_factory=UsageModel)
    trace: list[StepModel] = Field(default_factory=list)
    route: RouteModel | None = None
    pipeline: PipelineName = PipelineName.AUTO
    cached: bool = False
    """True when the semantic cache served this without running the pipeline.
    Surfaced rather than hidden: a client comparing latencies deserves to know
    which numbers came from a 2 ms nearest-neighbour lookup."""
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_answer(
        cls,
        answer: Answer,
        *,
        request_id: str,
        question: str,
        pipeline: PipelineName,
        warnings: list[str] | None = None,
    ) -> QueryResponse:
        route = answer.route
        return cls(
            request_id=request_id,
            question=question,
            answer=answer.text,
            citations=[CitationModel.from_citation(c) for c in answer.citations],
            chunks=[ChunkModel.from_scored(c) for c in answer.chunks],
            grounded=answer.grounded,
            groundedness=round(answer.groundedness, 4),
            confidence=round(answer.confidence, 4),
            abstained=answer.abstained,
            abstain_reason=answer.abstain_reason,
            usage=UsageModel.from_usage(answer.usage),
            trace=[StepModel.from_step(s) for s in answer.trace],
            route=RouteModel(
                stores=[s.value for s in route.stores],
                prompt=route.prompt_name,
                confidence=round(route.confidence, 4),
                reasoning=route.reasoning,
                method=route.method,
            )
            if route is not None
            else None,
            pipeline=pipeline,
            warnings=list(warnings or ()),
            metadata=_json_safe(answer.metadata),
        )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
class IngestRequest(_Body):
    """Documents to index: inline text, server-side paths, or both.

    Deliberately absent: the chunking strategy and the optional index stages
    (RAPTOR, GraphRAG, multi-representation). Those decide which models and
    clustering libraries the ingest pipeline loads, so they are resolved once
    when the pipeline is built. Honouring them per request would mean rebuilding
    the pipeline per request — the exact thing the lifespan exists to avoid. They
    are configuration (and CLI flags), not request fields.
    """

    text: str | None = Field(
        default=None,
        max_length=MAX_INLINE_TEXT_CHARS,
        description="A document body to index directly.",
    )
    paths: list[str] = Field(
        default_factory=list,
        max_length=MAX_PATHS,
        description="Server-side files or directories to load.",
    )
    source: str | None = Field(
        default=None,
        max_length=1_024,
        description=(
            "Label for inline text. Ingest is idempotent on (id, checksum) and "
            "the id derives from this, so a stable label makes re-posting the "
            "same document a no-op instead of a duplicate."
        ),
    )
    title: str | None = Field(default=None, max_length=1_024)
    tenant_id: str | None = Field(default=None, pattern=_TENANT_PATTERN)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=MAX_FILTER_KEYS)
    recursive: bool = Field(default=True, description="Walk directories in `paths` recursively.")

    @model_validator(mode="after")
    def _needs_something_to_ingest(self) -> Self:
        if not self.text and not self.paths:
            raise ValueError("provide `text`, `paths`, or an uploaded file")
        return self


class IngestResponse(_Body):
    """What one ingest run did.

    ``skip_rate`` is the number to watch: on a steady-state corpus the checksum
    comparison should skip almost everything, and a sudden drop means something
    upstream is rewriting content that did not change.
    """

    request_id: str = ""
    documents_in: int = 0
    indexed: int = 0
    skipped: int = 0
    rejected: int = 0
    duplicate: int = 0
    failed: int = 0
    chunks: int = 0
    vectors: int = 0
    strategy: str = ""
    skip_rate: float = 0.0
    cost_usd: float = 0.0
    llm_calls: int = 0
    total_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    rejections: list[list[str]] = Field(default_factory=list)
    """``[document_id, reason]`` pairs. A list of pairs rather than a mapping
    because the same reason recurs across documents and a dict keyed by reason
    would lose which documents it applied to."""
    failures: list[list[str]] = Field(default_factory=list)

    @classmethod
    def from_report(cls, report: Any, *, request_id: str) -> IngestResponse:
        summary = report.summary()
        return cls(
            request_id=request_id,
            documents_in=summary["documents_in"],
            indexed=summary["indexed"],
            skipped=summary["skipped"],
            rejected=summary["rejected"],
            duplicate=summary["duplicate"],
            failed=summary["failed"],
            chunks=summary["chunks"],
            vectors=summary["vectors"],
            strategy=summary["strategy"],
            skip_rate=summary["skip_rate"],
            cost_usd=summary["cost_usd"],
            llm_calls=summary["llm_calls"],
            total_ms=summary["total_ms"],
            timings_ms=summary["timings_ms"],
            warnings=list(report.warnings),
            rejections=[[doc, reason] for doc, reason in report.rejected],
            failures=[[doc, reason] for doc, reason in report.failed],
        )


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
class EvalItem(_Body):
    """One graded question.

    The field names mirror :class:`ragorc.eval.dataset.EvalCase`'s JSON form
    exactly, so the lines of a stored dataset can be posted here verbatim. A
    second, prettier spelling for the same thing would guarantee that the file on
    disk and the request body eventually disagree.

    Both label fields are optional and each unlocks different metrics: chunk ids
    give recall and nDCG, a reference answer gives correctness, and a dataset with
    neither still measures abstention, groundedness, latency and cost — which are
    the four that regress first and the four nobody labels for.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    expected_answer: str = Field(default="", max_length=MAX_INLINE_TEXT_CHARS)
    expected_chunk_ids: list[str] = Field(default_factory=list, max_length=MAX_TOP_K)
    id: str = Field(default="", max_length=128)
    """Stable case id. Left empty it is derived from the question text, which is
    what lets an A/B comparison pair results across two runs of a regenerated
    dataset."""
    tenant_id: str | None = Field(default=None, pattern=_TENANT_PATTERN)
    metadata: dict[str, Any] = Field(default_factory=dict, max_length=MAX_FILTER_KEYS)


class EvalRequest(_Body):
    """Run the eval harness over a dataset.

    ``compare`` is what makes the endpoint worth having: a single pipeline's
    numbers are uninterpretable without a baseline, and running two
    configurations over the same questions in the same process removes every
    confound except the one under test.
    """

    dataset: str | None = Field(
        default=None,
        max_length=4_096,
        description="Server-side path to a JSON array or JSONL file of items.",
    )
    items: list[EvalItem] = Field(
        default_factory=list,
        max_length=2_000,
        description="Inline dataset. Mutually sufficient with `dataset`.",
    )
    pipeline: PipelineName = PipelineName.AUTO
    compare: list[PipelineName] = Field(
        default_factory=list,
        max_length=6,
        description="Additional pipelines to score over the same items.",
    )
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    limit: int | None = Field(
        default=None, ge=1, le=2_000, description="Score only the first N items."
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description=(
            "Questions in flight. Bounded because each one is a full pipeline "
            "run: unbounded fan-out spends the cost ceiling before it finishes."
        ),
    )
    tenant_id: str | None = Field(default=None, pattern=_TENANT_PATTERN)

    @model_validator(mode="after")
    def _needs_a_dataset(self) -> Self:
        if not self.dataset and not self.items:
            raise ValueError("provide `dataset` or `items`")
        return self


class EvalMetrics(_Body):
    """Scores for one pipeline over one dataset.

    Percentiles, not means, for latency: a mean hides the tail users complain
    about. Means for the quality scores, where the distribution is bounded and
    the average is the quantity of interest.

    The retrieval and answer metrics are open maps rather than named fields.
    :func:`ragorc.eval.retrieval_metrics.evaluate_retrieval` reports every metric
    at every *k* — ``recall@1``, ``ndcg@10``, ``map`` and a dozen more, with the
    set of *k*s configurable — and pinning a subset of those into fixed fields
    here would mean this schema silently dropping whichever metric the harness
    added last. ``labelled`` says how many items could be scored at all, which is
    the number that makes the rest interpretable.
    """

    pipeline: PipelineName = PipelineName.AUTO
    items: int = 0
    labelled: int = 0
    errors: int = 0
    abstain_rate: float = 0.0
    grounded_rate: float = 0.0
    groundedness_mean: float = 0.0
    confidence_mean: float = 0.0
    citation_coverage: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cost_usd_total: float = 0.0
    cost_usd_per_query: float = 0.0
    llm_calls_total: int = 0
    cache_hit_rate: float = 0.0
    retrieval: dict[str, float] = Field(default_factory=dict)
    """recall@k / precision@k / hit_rate@k / ndcg@k / mrr / map. Empty when no
    item carried chunk labels — empty rather than zeroed, because a zero recall
    and an unmeasurable recall are different findings."""
    answer: dict[str, float] = Field(default_factory=dict)
    """Answer quality, keyed by the names in
    :data:`ragorc.eval.answer_metrics.ALL_METRICS` — ``lexical_overlap``,
    ``faithfulness``, ``answer_relevance``, ``context_precision``,
    ``context_recall``, ``answer_correctness``.

    Which of them appear depends on the dataset and the configuration, and each
    absence means something different: the reference-based scores need
    ``expected_answer``, and the judged scores need judging to be enabled. A
    metric is omitted rather than zeroed when it could not be computed, for the
    same reason ``retrieval`` is empty rather than zeroed."""


class EvalResponse(_Body):
    request_id: str = ""
    dataset: str = ""
    items: int = 0
    results: list[EvalMetrics] = Field(default_factory=list)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    """Paired bootstrap of each compared pipeline against the first one.

    Present only when ``compare`` was non-empty, and the reason ``compare`` exists:
    two means cannot say whether a difference is real. The pairing removes the
    between-question variance that otherwise dominates the estimate, so a verdict
    of ``inconclusive`` here means "consistent with noise at this sample size" —
    not "the two are equal"."""
    harness: str = "ragorc.eval.runner"
    """Which harness produced the numbers. Recorded because scores from two
    different harnesses are not comparable, and this field is what makes that
    visible in a results file read six months later."""
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Health & errors
# ---------------------------------------------------------------------------
class StoreHealth(_Body):
    """One backend's reachability.

    Probed with a real, cheap query rather than a TCP connect: a store that
    accepts connections and refuses queries is exactly the failure a health
    check exists to catch.
    """

    name: str
    status: str = "ok"
    latency_ms: float = 0.0
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class HealthResponse(_Body):
    """Service and dependency status, plus a redacted configuration summary.

    The settings summary comes from :meth:`Settings.summary`, which redacts by
    construction — it reports whether an API key is present, never the key, and
    reduces each DSN to its host. Health endpoints get scraped into dashboards
    and pasted into tickets, so the redaction has to be a property of the
    producer, not of the caller's discipline.
    """

    status: str = "ok"
    """``ok`` when every probed store answered, ``degraded`` when at least one
    did not. Degraded is deliberately not ``error``: one dead store degrades
    answers rather than failing the service, which is the documented behaviour
    and therefore what the status should say."""
    version: str = ""
    environment: str = "dev"
    uptime_s: float = 0.0
    pipelines: list[PipelineName] = Field(default_factory=list)
    stores: list[StoreHealth] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    cache: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ErrorResponse(_Body):
    """The only error shape this service emits.

    No stack traces, no exception ``repr``, no credentials: the body carries the
    error class, its message, and the structured ``detail`` the exception was
    raised with, scrubbed of anything key-shaped. A traceback in a response body
    is a map of the application handed to whoever asked for it.
    """

    error: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)
    request_id: str = ""


def _json_safe(value: Any) -> Any:
    """Coerce trace and metadata payloads into something JSON can hold.

    These dicts are assembled by a dozen pipeline stages and legitimately contain
    enum members, tuples, numpy scalars and dataclasses. Serializing them is not
    the stage's job — it never knew it was heading for HTTP — so the conversion
    happens once, here, and unknown objects degrade to ``repr`` rather than
    failing a response that is otherwise correct.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, enum.Enum):
        return _json_safe(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # numpy scalars are the common case here — a score or a count that came out of
    # a vectorized stage — and ``.item()`` is what turns one into a Python number.
    # Anything else exposing ``.item`` falls through to the repr below rather than
    # being reported as a failure: this function's contract is that it always
    # returns something serializable.
    item = getattr(value, "item", None)
    if callable(item):
        with contextlib.suppress(TypeError, ValueError):
            return _json_safe(item())
    return repr(value)[:1_000]
