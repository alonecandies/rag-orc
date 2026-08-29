"""The HTTP service.

This module is the composition layer. Everything below it — retrieval,
generation, the guards, the caches, the stores — already works and knows nothing
about HTTP. What is left, and what actually decides whether the service is
operable, is four things: **when components are built**, **what wraps a request**,
**what a failure is allowed to reveal**, and **what a client is allowed to ask
for**. That is this file (plus :mod:`ragorc.server.schemas` for the last one).

Build once, at startup
----------------------
The lifespan builds one :class:`RagService` and closes it on shutdown. Nothing is
constructed per request, and that is not a micro-optimization:

* The dense embedder loads an ONNX model — hundreds of milliseconds and hundreds
  of megabytes of resident memory. Per request it would dominate every latency
  number and multiply memory by the concurrency.
* The Qdrant, Postgres and Neo4j clients own **connection pools**. A pool built
  per request is not a pool; it is a connection storm plus a TLS handshake on the
  request path, and Postgres will start refusing connections long before the
  embedder finishes loading.
* The circuit breakers in the retrieval layer are *stateful by design*. A breaker
  rebuilt per request has no memory, which is the one property it exists to have,
  so a dead store would cost every request its full timeout forever.

The corollary is that anything a client could change which would require
rebuilding is **not a request field**. Chunking strategy and the optional index
stages are the clearest case; see :class:`~ragorc.server.schemas.IngestRequest`.

Every request runs inside a request context
-------------------------------------------
:func:`~ragorc.core.telemetry.new_request_context` installs a fresh trace and a
fresh :class:`~ragorc.core.telemetry.CostLedger` in contextvars for the duration
of one request. Three consequences, all of which the service depends on:

* the per-stage trace returned with every answer is populated without threading a
  trace object through forty signatures;
* the cost ledger is checked *before* each model call, so the configured ceiling
  is enforced before the money is spent rather than reported after — which is the
  only defence a pipeline with loops and retries has against unbounded spend;
* concurrent requests in one event loop keep separate traces and separate
  budgets, because contextvars are per-task.

A request that escapes this wrapper has no budget. So the wrapper is inside
:class:`RagService`, not in a middleware a future endpoint could forget to use.

Delegation, and why there is a linear fallback
----------------------------------------------
A RAG query is a state machine with feedback loops (ADR-0001), and the state
machines live in :mod:`ragorc.pipeline` as LangGraph graphs. When that layer is
importable this service delegates to it — cycles, checkpointing and the agentic
branch are properties of a graph and cannot be faked by calling stages in order.

When it is not importable, :class:`_LinearEngine` answers instead: validate,
retrieve, rerank, generate, with the Self-RAG and RRR loops wired through the
callables they already accept. It is a real engine, not a placeholder, and it
covers every pipeline whose only cycle is one those two loops provide. What it
cannot do it says: the response carries ``metadata["orchestrator"]`` and a warning
naming the substitution, because a service that quietly answered ``agentic``
queries with a linear pipeline would be lying about which system produced the
number someone is about to trust.

Failures reveal the decision, never the machine
-----------------------------------------------
Every :class:`~ragorc.core.errors.RagOrcError` carries a structured ``detail``
that exists to be shown: *which* rule blocked the query, *which* store is down,
*which* budget is spent. That is returned. What is never returned is a traceback,
an exception ``repr``, a DSN or anything key-shaped — a stack trace in a response
body is a map of the application handed to whoever asked for one, and a redacted
``detail`` is strictly more useful to a legitimate caller anyway.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import importlib
import inspect
import math
import os
import time
import uuid
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson
import structlog

from ragorc.cache.semantic import scope_key
from ragorc.core.concurrency import gather_dict, install_uvloop
from ragorc.core.errors import (
    BudgetExceeded,
    ConfigError,
    ConstructionError,
    EmbeddingError,
    GuardrailViolation,
    LLMError,
    RagOrcError,
    RateLimited,
    RetrievalError,
    StoreUnavailable,
    TransientError,
    ValidationFailed,
)
from ragorc.core.ids import content_hash, document_id
from ragorc.core.models import (
    Answer,
    Document,
    Query,
    RetrievalResult,
    RouteDecision,
    Usage,
)
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import (
    CostLedger,
    configure_logging,
    new_request_context,
    redact_identifiers,
    timed,
    trace_step,
)
from ragorc.security.audit import AuditLog
from ragorc.security.ratelimit import KeyedRateLimiter
from ragorc.security.tenancy import (
    ANONYMOUS,
    graph_isolation_warning,
    principal_for_key,
    resolve_tenant,
    scope_filter,
    tenant_bindings,
    unbound_principals_warning,
)
from ragorc.server.schemas import (
    MAX_QUESTION_CHARS,
    DeleteRequest,
    DeleteResponse,
    DocumentsResponse,
    DocumentSummary,
    ErrorResponse,
    EvalItem,
    EvalMetrics,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PipelineName,
    QueryRequest,
    QueryResponse,
    StoreHealth,
)
from ragorc.validate.input import QueryValidator

if TYPE_CHECKING:  # pragma: no cover - see the note in create_app
    from fastapi import Request

log = structlog.get_logger(__name__)

__all__ = ["RagService", "create_app", "load_eval_items", "service_dependency"]

_VERSION = "0.1.0"

_HEALTH_PROBE_TIMEOUT_S = 3.0
"""Deadline for one store probe.

Short on purpose: a health endpoint that blocks on an unresponsive store has
become the outage it was meant to report, and every orchestrator treats a slow
probe as a failed one anyway. Three seconds is longer than any of these queries
takes when the store is alive."""

_MISSING = object()
"""Sentinel distinguishing "declares no such attribute" from "declares it as
``None``". ``getattr(x, "cache", None)`` cannot tell those apart, and only the
second one is an invitation to fill it in."""

_SECRET_HINTS = ("api_key", "apikey", "password", "token", "secret", "authorization", "dsn")
"""Substrings that mark a value as unfit for a response body. Mirrors the log
redactor in :mod:`ragorc.core.telemetry`; the two lists are short and duplicating
them is cheaper than a shared import that ties error formatting to logging."""


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency, naming the extra when it is missing.

    The bare ``ModuleNotFoundError`` for ``sse_starlette`` tells an operator
    nothing about which extra ships it, and they will guess wrong at least once.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            f"{module} is required for the HTTP service: pip install 'ragorc[{extra}]'"
        ) from exc


def _new_request_id() -> str:
    """A short, sortable-enough id. Not a UUID string: this value is echoed in a
    header, logged on every line of the request and printed by the CLI, and 32
    hex characters of it is noise in all three places."""
    return uuid.uuid4().hex[:16]


def _safe_detail(detail: dict[str, Any]) -> dict[str, Any]:
    """Scrub an exception's ``detail`` for a response body.

    Two rules. Anything whose *key* looks like a credential is dropped outright
    rather than masked — a masked value still confirms the field exists and how
    long it is. Everything else is stringified and clipped, because ``detail``
    legitimately carries a rejected SQL statement or a store's error text, and an
    unbounded one of those becomes the response.
    """
    out: dict[str, Any] = {}
    for key, value in detail.items():
        lowered = str(key).lower()
        if any(hint in lowered for hint in _SECRET_HINTS):
            continue
        # Values are redacted as well as keys. Key filtering alone lets a provider
        # error body through under `body` or `message` with a key-management URL
        # and an account id inside it — the credential quoted in someone else's
        # prose, which is exactly the shape a key-name rule cannot see.
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = redact_identifiers(value[:500]) if isinstance(value, str) else value
        elif isinstance(value, (list, tuple)):
            out[str(key)] = [redact_identifiers(str(v)[:200]) for v in list(value)[:20]]
        else:
            out[str(key)] = redact_identifiers(str(value)[:500])
    return out


# ---------------------------------------------------------------------------
# Resolving the orchestration layer
#
# The graphs in ``ragorc.pipeline`` are developed alongside this module, so this
# service depends on their *shape* rather than on their presence. The mechanism
# is the one ``ragorc.index.pipeline`` already uses for its optional stages:
# several module and attribute names are accepted, the first that resolves wins,
# and calls are filtered by signature so a collaborator's parameter list can
# change without breaking the call site.
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Plugin:
    label: str
    modules: tuple[str, ...]
    factories: tuple[str, ...]


_ORCHESTRATOR = _Plugin(
    label="orchestrator",
    modules=("ragorc.pipeline.builder", "ragorc.pipeline"),
    factories=("build_pipeline", "RAGPipeline", "build"),
)

_QUERY_METHODS = ("query", "run", "answer", "ainvoke")
_STREAM_METHODS = ("stream", "astream", "stream_answer")
_INGEST_METHODS = ("ingest", "index", "add")
_CLOSE_METHODS = ("aclose", "close", "shutdown")


def _load_plugin(plugin: _Plugin) -> Any | None:
    """Resolve a collaborator's factory, or ``None`` when it is not present.

    Only a genuinely absent module yields ``None``. Any other ``ImportError`` —
    a broken relative import, an optional dependency imported at module scope —
    is left to propagate, because swallowing those turns a real bug into a
    mysteriously missing feature.
    """
    for module_name in plugin.modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if (exc.name or "") in (module_name, module_name.rsplit(".", 1)[0]):
                continue
            raise
        for attr in plugin.factories:
            factory = getattr(module, attr, None)
            if factory is not None:
                return factory
    return None


def _accepted(target: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Filter ``candidates`` down to the parameters ``target`` declares.

    A callable declaring ``**kwargs`` gets everything; one whose signature cannot
    be read gets nothing. This is what lets the call sites below stay stable while
    the thing they call is still being written.
    """
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # pragma: no cover - C-implemented callable
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(candidates)
    return {k: v for k, v in candidates.items() if k in signature.parameters}


def _method(obj: Any, names: Sequence[str]) -> Any | None:
    for name in names:
        candidate = getattr(obj, name, None)
        if callable(candidate):
            return candidate
    return None


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


# ---------------------------------------------------------------------------
# The linear engine
# ---------------------------------------------------------------------------
_LINEAR_UNSUPPORTED = frozenset({PipelineName.AGENTIC})
"""Pipelines whose defining feature is a cycle the loops cannot express. An
agentic graph chooses its own next tool from the state; approximating that with a
fixed fan-out would not be a degraded version of it, it would be a different
system wearing its name."""


class _LinearEngine:
    """Validate, retrieve, rerank, generate — with the two loops wired in.

    The stage order is the one :mod:`ragorc.generate.answer` documents and every
    guarantee depends on. The loops are the interesting part, and they compose the
    way :mod:`ragorc.generate.rrr` says they do:

    * **RRR wraps retrieval.** It rewrites the conversational question into a
      search query *before* retrieving, and retries on weak retrieval — paying one
      cheap rewrite instead of a full synthesis to discover the query was bad.
    * **Self-RAG wraps the pair.** It grades the finished answer for support and
      utility and, on failure, rewrites and retrieves again. Because it takes a
      ``retrieve_fn``, the function it is handed is the RRR-wrapped one, so a
      Self-RAG retry gets a rewritten query for free.

    Which retriever runs is the only thing the pipeline name selects, because CRAG,
    GraphRAG and multi-hop are all ``Retriever`` implementations — the corrective
    loop, the graph traversal and the hop iteration are inside them. That is why
    a linear driver can cover them at all.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm: Any = None
        self.dense: Any = None
        self.sparse: Any = None
        self.colbert: Any = None
        self.vector: Any = None
        self.relational: Any = None
        self.graph: Any = None
        self.hybrid: Any = None
        self.vector_leg: Any = None
        self.reranker: Any = None
        self.generator: Any = None
        self.ingest_pipeline: Any = None
        self.router: Any = None
        self.cache: Any = None
        self._retrievers: dict[PipelineName, Any] = {}

    # -- construction ------------------------------------------------------
    async def build(self) -> None:
        """Construct every component. Called once, from the lifespan."""
        from ragorc.cache.tiered import build_cache
        from ragorc.embed.cache import EmbeddingCache
        from ragorc.generate.answer import AnswerGenerator
        from ragorc.index.pipeline import IngestPipeline
        from ragorc.llm.cache import LLMCache
        from ragorc.llm.openrouter import OpenRouterLLM
        from ragorc.retrieve import HybridRetriever, build_reranker
        from ragorc.retrieve.parent import parent_leg
        from ragorc.stores.postgres.store import PostgresStore
        from ragorc.stores.qdrant.store import QdrantStore

        s = self.settings
        self.cache = build_cache(s.cache)
        embedding_cache = EmbeddingCache(self.cache, s)
        self.llm = OpenRouterLLM(s.llm, cache=LLMCache(self.cache, s.cache))

        self.dense = _embedder("dense_embedder", s.embedding.provider, embedding_cache, s)
        # Sparse and ColBERT follow the *retrieval* configuration rather than
        # being built unconditionally: each is an extra ONNX session in resident
        # memory, and a dense-only deployment should not pay for a lexical index
        # it has switched off.
        self.sparse = (
            _embedder("sparse_embedder", "fastembed", embedding_cache, s)
            if s.retrieval.use_sparse
            else None
        )
        self.colbert = (
            _embedder("late_interaction_embedder", "fastembed", embedding_cache, s)
            if s.embedding.enable_late_interaction
            else None
        )

        self.vector = QdrantStore(
            s,
            dense_embedder=self.dense,
            sparse_embedder=self.sparse,
            late_embedder=self.colbert,
        )
        # The width the embedder emits, not the settings default: changing
        # `embedding.dense_model` moves Qdrant and leaves `vector_dimension`
        # at 384, which fails every pgvector write with a message that names
        # neither the model nor the setting.
        self.relational = PostgresStore(
            s, cache=self.cache, dimension=int(getattr(self.dense, "dimension", 0) or 0) or None
        )

        self.hybrid = HybridRetriever(self.vector, postgres=self.relational, settings=s)
        # The vector leg as every route should see it. `parent_leg` is the
        # identity unless an indexing representation is on; when one is, the
        # collection holds children/summaries/propositions and this resolves
        # them back to the source text. Assigned to its own attribute rather
        # than folded into `self.hybrid`, so the name keeps describing the
        # object — and so the five routes below read one thing, not two.
        self.vector_leg = parent_leg(self.hybrid, self.relational, settings=s)
        self.reranker = build_reranker(llm=self.llm, settings=s)
        # Only the cross-encoder stage has scores worth memoizing, and it is the
        # one that declares a ``cache``. Attaching it after construction rather
        # than passing it through the factory avoids restating the factory's alias
        # table here — which would silently stop matching the day a spelling is
        # added, leaving ``cache.cache_rerank`` on and doing nothing.
        if getattr(self.reranker, "cache", _MISSING) is None:
            self.reranker.cache = self.cache
        self.generator = AnswerGenerator(self.llm, s)
        # The ingest pipeline shares the stores and the embedders, so an ingest
        # served by this process reuses the ONNX sessions and the pools the query
        # path already has open instead of doubling both.
        self.ingest_pipeline = IngestPipeline(
            vector_store=self.vector,
            relational_store=self.relational,
            dense_embedder=self.dense,
            sparse_embedder=self.sparse,
            late_embedder=self.colbert,
            llm=self.llm,
            settings=s,
        )

    async def aclose(self) -> None:
        """Release everything, in dependency order, without raising.

        Shutdown runs while the process is already going away, so a failure here
        can only obscure the reason it is going away. Each step is logged and
        suppressed.
        """
        from ragorc.stores.qdrant.client import close_all_clients

        for label, closer in (
            ("ingest", getattr(self.ingest_pipeline, "close", None)),
            ("llm", getattr(self.llm, "aclose", None)),
            ("qdrant", getattr(self.vector, "close", None)),
            ("postgres", getattr(self.relational, "close", None)),
            ("neo4j", getattr(self.graph, "close", None)),
            ("cache", getattr(self.cache, "close", None)),
            # The embedders and the reranker were omitted, on a method whose
            # docstring is about shutting the engine down completely. A local
            # ONNX session leaks little at process exit, but a *hosted* embedding
            # provider holds an httpx pool and its own connections, and the
            # orchestration layer closes this engine while the process keeps
            # running — so the leak is per-swap, not per-process.
            ("dense_embedder", getattr(self.dense, "aclose", None)),
            ("sparse_embedder", getattr(self.sparse, "aclose", None)),
            ("late_embedder", getattr(self.colbert, "aclose", None)),
            ("reranker", getattr(self.reranker, "aclose", None)),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("close_failed", component=label, error=str(exc)[:200])
        with contextlib.suppress(Exception):
            await close_all_clients()

    # -- retriever selection ----------------------------------------------
    def graph_store(self) -> Any:
        """The Neo4j store, built on first use.

        Lazy rather than eager because the graph pipelines are opt-in: a
        deployment with no Neo4j should not hold a driver, and building one here
        costs nothing since the driver connects lazily too.
        """
        if self.graph is None:
            from ragorc.stores.neo4j.store import Neo4jStore

            self.graph = Neo4jStore(settings=self.settings)
        return self.graph

    def _resolve(self, pipeline: PipelineName) -> tuple[PipelineName, list[str]]:
        """Turn ``auto`` into a concrete pipeline, and refuse what cannot run.

        ``auto`` reads the configuration rather than asking the client which
        features this deployment enabled. The order is by specificity: a
        corrective loop is a stronger statement of intent than a graph index,
        which is a stronger statement than plain hybrid search.
        """
        warnings: list[str] = []
        if pipeline in _LINEAR_UNSUPPORTED:
            warnings.append(
                f"pipeline {pipeline.value!r} needs the orchestration layer; "
                f"answered with {PipelineName.ADAPTIVE.value!r} instead"
            )
            pipeline = PipelineName.ADAPTIVE
        if pipeline is not PipelineName.AUTO:
            return pipeline, warnings
        if self.settings.retrieval.crag_enabled:
            return PipelineName.CRAG, warnings
        if self.settings.graph.enabled:
            return PipelineName.GRAPHRAG, warnings
        return PipelineName.NAIVE, warnings

    def notes_for(self, pipeline: PipelineName) -> list[str]:
        """The substitutions answering ``pipeline`` here would require, named.

        :meth:`query` returns these alongside the answer, where they end up in
        ``QueryResponse.warnings``. A stream has no response object to hang them
        on, so :meth:`RagService.stream` reads them from here and emits them as
        ``warning`` events before the first token.
        """
        return self._resolve(pipeline)[1]

    def _retriever_for(self, pipeline: PipelineName) -> Any:
        """Build (and memoize) the retriever a pipeline selects.

        A missing optional extra is re-raised as a :class:`ConfigError` carrying
        the original message. The underlying ``ImportError`` already names the
        package to install — losing that text to a generic 500 would send the
        operator hunting for a dependency the exception could have told them.
        """
        existing = self._retrievers.get(pipeline)
        if existing is not None:
            return existing
        try:
            built = self._build_retriever(pipeline)
        except ImportError as exc:
            raise ConfigError(
                f"pipeline {pipeline.value!r} needs an optional dependency",
                reason=str(exc)[:300],
            ) from exc
        self._retrievers[pipeline] = built
        return built

    def _build_retriever(self, pipeline: PipelineName) -> Any:
        """The retriever each pipeline selects.

        ``naive`` is a *graph shape* — retrieve, then generate, no loops — not a
        retriever choice, so it gets the configured default, which is hybrid
        search. Calling it "naive" and quietly downgrading it to dense-only would
        make the benchmark control measure something nobody deploys.
        """
        from ragorc.retrieve import (
            CorrectiveRAG,
            EnsembleRetriever,
            GraphGlobalRetriever,
            GraphLocalRetriever,
            MultiHopRetriever,
            MultiStoreRetriever,
            PgFullTextRetriever,
            make_web_retriever,
        )

        s = self.settings
        if pipeline is PipelineName.CRAG:
            return CorrectiveRAG(self.vector_leg, self.llm, s)
        if pipeline is PipelineName.GRAPHRAG:
            graph = self.graph_store()
            # Local and global are not alternatives: local answers "what about
            # X", global answers "what are the themes". Fusing both and letting
            # the ranking decide is cheaper than asking a model which the
            # question is, and it degrades to whichever one found something.
            return EnsembleRetriever(
                {
                    "graph_local": GraphLocalRetriever(graph, self.vector, settings=s),
                    "graph_global": GraphGlobalRetriever(self.llm, graph, settings=s),
                    "vector": self.vector_leg,
                },
                settings=s,
            )
        if pipeline is PipelineName.MULTIHOP:
            return MultiHopRetriever(
                self.llm, self.vector_leg, self.graph_store(), self.vector, settings=s
            )
        if pipeline is PipelineName.ADAPTIVE:
            graph = self.graph_store() if s.graph.enabled else None
            return MultiStoreRetriever(
                vector=self.vector_leg,
                relational=PgFullTextRetriever(self.relational, settings=s),
                graph=GraphLocalRetriever(graph, self.vector, settings=s) if graph else None,
                web=make_web_retriever(s) if s.retrieval.crag_web_fallback else None,
                settings=s,
            )
        return self.vector_leg

    def _router(self) -> Any:
        if self.router is None:
            from ragorc.route import build_router

            self.router = build_router(
                "hybrid", llm=self.llm, embedder=self.dense, settings=self.settings
            )
        return self.router

    # -- the query path ----------------------------------------------------
    async def query(
        self,
        query: Query,
        *,
        pipeline: PipelineName = PipelineName.AUTO,
        prompt_name: str | None = None,
    ) -> tuple[Answer, PipelineName, list[str]]:
        """Run one query and return the answer, the pipeline that ran, and notes."""
        resolved, warnings = self._resolve(pipeline)
        retriever = self._retriever_for(resolved)
        gen = self.settings.generation

        route: RouteDecision | None = None
        if resolved is PipelineName.ADAPTIVE:
            route, usage = await self._route(query)
            query = await self._expand(query)
            trace_step("route", stores=[s.value for s in route.stores], usage=usage)

        async def retrieve(current: Query) -> RetrievalResult:
            return await self._retrieve_and_rerank(retriever, current, route=route)

        # Every loop's bill, collected as it is spent. `SelfRAGResult.usage` and
        # `RRRResult.usage` are computed by their owners and had no reader here,
        # so a successful Self-RAG run — two answers, two groundedness grades, two
        # utility grades, one rewrite — reported the single generation call the
        # answer happened to carry, and the cost ceiling and every `$` in the eval
        # output were understated by the whole loop.
        loop_usage: list[Usage] = []

        async def retrieve_with_rrr(current: Query) -> RetrievalResult:
            if not gen.rrr_enabled:
                return await retrieve(current)
            from ragorc.generate.rrr import RRR

            outcome = await RRR(self.llm, self.settings).run(current, retrieve)
            loop_usage.append(outcome.usage)
            # Named keys rather than splatting ``report()``: a trace detail is a
            # few scalars, and splatting a collaborator's dict into a function
            # that also takes ``duration_ms`` and ``usage`` is a collision waiting
            # for that dict to gain a field.
            trace_step(
                "rrr",
                rewrites=len(outcome.rewrites),
                succeeded=outcome.succeeded,
                chunks=len(outcome.retrieval.chunks),
            )
            return outcome.retrieval

        async def generate(current: Query, retrieval: RetrievalResult) -> Answer:
            return await self.generator.generate(
                current, retrieval, route=route, prompt_name=prompt_name
            )

        if resolved is PipelineName.SELF_RAG or gen.self_rag_enabled:
            from ragorc.generate.self_rag import SelfRAG

            outcome = await SelfRAG(self.llm, self.settings).run(query, retrieve_with_rrr, generate)
            trace_step(
                "self_rag",
                iterations=len(outcome.attempts),
                accepted_at=outcome.accepted_iteration,
                abstained=outcome.answer.abstained,
            )
            answer = outcome.answer
            loop_usage.append(outcome.usage)
            # The full report goes on the answer, where a caller can read the
            # per-iteration verdicts. The trace gets the summary.
            answer.metadata.setdefault("self_rag", outcome.report())
        else:
            retrieval = await retrieve_with_rrr(query)
            answer = await generate(query, retrieval)

        if loop_usage:
            # Summed rather than assigned: the answer already carries the usage of
            # the generation that produced it, and the loops' totals cover the
            # calls around it.
            answer.usage = Usage.sum([answer.usage, *loop_usage])

        answer.metadata["orchestrator"] = "linear"
        answer.metadata["pipeline"] = resolved.value
        return answer, resolved, warnings

    async def stream(
        self,
        query: Query,
        *,
        pipeline: PipelineName = PipelineName.AUTO,
        prompt_name: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream the answer text.

        The verification loops are deliberately absent: Self-RAG grades a
        *finished* answer, so running it here would mean generating the whole
        answer before emitting the first token — which is not streaming with
        extra steps, it is a slower non-streaming request. The pre-generation
        abstention gate still applies inside the generator, so a stream is never
        started with no evidence at all.
        """
        resolved, _warnings = self._resolve(pipeline)
        retriever = self._retriever_for(resolved)
        route: RouteDecision | None = None
        if resolved is PipelineName.ADAPTIVE:
            route, _usage = await self._route(query)
            query = await self._expand(query)
        retrieval = await self._retrieve_and_rerank(retriever, query, route=route)
        async for delta in self.generator.stream(
            query, retrieval, route=route, prompt_name=prompt_name
        ):
            yield delta

    async def ingest(
        self,
        target: Any,
        *,
        force: bool = False,
        loader_options: Mapping[str, Any] | None = None,
    ) -> Any:
        """Index a target, honouring the request's loader options.

        Declared explicitly rather than left to ``**kwargs``: :func:`_accepted`
        filters a call down to the parameters the target *declares*, so an engine
        that omits this parameter has its options silently dropped rather than
        raising — which is how ``recursive`` and ``metadata`` went unnoticed in the
        first place.

        Passed through, not assigned. This used to set the attribute on the
        long-lived shared pipeline and restore it in a ``finally``, which is worse
        than not scoping at all once two ingests overlap: both runs saw whichever
        options were assigned last, and the restore put the *first* run's options
        back permanently, so every later request inherited them. Measured with two
        concurrent calls — both observed caller B's metadata, and the pipeline was
        left holding caller A's.
        """
        return await self.ingest_pipeline.ingest(
            target, force=force, loader_options=loader_options
        )

    # -- stages ------------------------------------------------------------
    async def _route(self, query: Query) -> tuple[RouteDecision, Usage]:
        """Route, degrading to "query everything" on failure.

        A router that cannot decide is not a reason to fail a request that a
        full fan-out would have answered — it only costs the saving the router
        exists to make.
        """
        try:
            return await self._router().route(query)
        except Exception as exc:  # noqa: BLE001 - routing is an optimization
            log.warning("route_failed", error=str(exc)[:200])
            return RouteDecision(stores=(), method="fallback", reasoning=str(exc)[:200]), Usage()

    async def _expand(self, query: Query) -> Query:
        """Add query variants before a multi-store fan-out.

        Only here, and only multi-query. Every retriever in this library already
        consumes ``Query.variants`` (``all_texts``), and a fan-out is exactly
        where several phrasings pay: the lexical leg and the graph leg are
        sensitive to wording in ways the dense leg is not. Multi-query is also the
        one translator that needs no configuration beyond a model, so enabling it
        here invents no policy.
        """
        try:
            from ragorc.translate.multi_query import MultiQueryTranslator

            expanded, usage = await MultiQueryTranslator(self.llm, self.settings).translate(query)
            trace_step("translate", variants=len(expanded.variants), usage=usage)
            return expanded
        except Exception as exc:  # noqa: BLE001 - translation is an enhancement
            log.warning("translate_failed", error=str(exc)[:200])
            return query

    async def _retrieve_and_rerank(
        self, retriever: Any, query: Query, *, route: RouteDecision | None
    ) -> RetrievalResult:
        result = await self._retrieve(retriever, query, route=route)
        if not self.settings.retrieval.rerank_enabled or not result.chunks:
            return result
        with timed("rerank", candidates=len(result.chunks)):
            ranked, usage = await self.reranker.rerank_with_usage(query, result.chunks)
        if usage.calls:
            trace_step("rerank_usage", usage=usage)
        limit = int(query.top_k or self.settings.retrieval.top_k)
        result.chunks = ranked[:limit]
        return result

    async def _retrieve(
        self, retriever: Any, query: Query, *, route: RouteDecision | None
    ) -> RetrievalResult:
        """Call a retriever through whichever of its three shapes it offers.

        The ``Retriever`` protocol returns a bare list, which has nowhere to put
        per-store timings, per-store errors or a CRAG grade — so the richer
        methods are preferred when a retriever has them, and the bare one is
        wrapped when it does not. ``CorrectiveRAG.run`` additionally returns its
        bill, which the protocol also cannot carry.
        """
        kwargs: dict[str, Any] = {"filters": query.filters or None, "tenant_id": query.tenant_id}
        run = getattr(retriever, "run", None)
        if callable(run):
            result, usage = await run(query, top_k=query.top_k, **kwargs)
            if usage.calls:
                trace_step("retrieve_usage", retriever=retriever.name, usage=usage)
            return result
        detailed = getattr(retriever, "retrieve_detailed", None)
        if callable(detailed):
            if route is not None and "route" in _accepted(detailed, {"route": route}):
                kwargs["route"] = route
            return await detailed(query, top_k=query.top_k, **kwargs)
        chunks = await retriever.retrieve(query, top_k=query.top_k, **kwargs)
        usage = getattr(retriever, "usage", None)
        if isinstance(usage, Usage) and usage.calls:
            trace_step("retrieve_usage", retriever=retriever.name, usage=usage)
        return RetrievalResult(chunks=list(chunks), total_candidates=len(chunks))


def _embedder(kind: str, provider: str, cache: Any, settings: Settings) -> Any:
    """Resolve an embedder class through the registry, importing its provider.

    The registry only knows a class once the module defining it has been
    imported, and importing all five providers eagerly would require every hosted
    SDK (openai, voyageai, cohere, torch) to be installed. Deriving the module
    name from the provider name means a missing extra fails with that provider's
    own ``ImportError``, which names the package to install.
    """
    from ragorc.core.registry import resolve

    importlib.import_module(f"ragorc.embed.{provider}_provider")
    return resolve(kind, provider)(cache=cache, settings=settings)


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------
class RagService:
    """Everything one process needs to answer, ingest and evaluate.

    Built once per process. The cross-cutting concerns live here rather than in
    the endpoints because every one of them is a correctness property, not a
    feature: an endpoint that forgot the request context would have no cost
    ceiling, one that forgot :func:`require_tenant` would answer tenant A's
    question with tenant B's documents, and neither omission is visible in a
    passing test.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.started_at = time.monotonic()
        self.engine: Any = None
        self.orchestrated = False
        self.linear = _LinearEngine(self.settings)
        self.validator = QueryValidator(self.settings)
        self.audit = AuditLog(self.settings)
        self.limiter = KeyedRateLimiter.from_settings(self.settings.security)
        self.tenant_bindings = tenant_bindings(self.settings)
        self.semantic: Any = None
        self.warnings: list[str] = []
        self._built = False

    # -- lifecycle ---------------------------------------------------------
    async def build(self) -> RagService:
        """Construct the engine and the caches. Idempotent."""
        if self._built:
            return self
        self._built = True

        with timed("service_build"):
            await self.linear.build()
            self.engine = self.linear
            await self._attach_orchestrator()
            self._attach_semantic_cache()

        log.info(
            "service_ready",
            orchestrator="graph" if self.orchestrated else "linear",
            environment=self.settings.environment,
            embedding_model=self.settings.embedding.dense_model,
            sparse=self.linear.sparse is not None,
            semantic_cache=self.semantic is not None,
            warnings=self.warnings,
        )
        return self

    async def _attach_orchestrator(self) -> None:
        """Prefer the LangGraph layer when it is importable.

        The graph is handed the components the linear engine already built, so
        both paths share one ONNX session and one set of pools no matter which
        one answers. A factory that cannot accept them builds its own, which is
        correct but doubles the resident model — logged, because that is a real
        memory difference an operator should see once at startup.
        """
        factory = _load_plugin(_ORCHESTRATOR)
        if factory is None:
            self.warnings.append(
                "orchestration layer unavailable; answering with the linear engine"
            )
            log.info("orchestrator_unavailable", fallback="linear")
            return

        candidates = {
            "settings": self.settings,
            "llm": self.linear.llm,
            "vector_store": self.linear.vector,
            "relational_store": self.linear.relational,
            "dense_embedder": self.linear.dense,
            "sparse_embedder": self.linear.sparse,
            "late_embedder": self.linear.colbert,
            "retriever": self.linear.hybrid,
            "reranker": self.linear.reranker,
            "generator": self.linear.generator,
            "cache": self.linear.cache,
        }
        try:
            engine = await _maybe_await(factory(**_accepted(factory, candidates)))
            # A *class* was instantiated, so an async initialization step may still
            # be pending — ``__init__`` cannot await, which is why this codebase
            # pairs constructors with a ``build()``. A factory *function* owns its
            # own initialization by definition (``build_pipeline`` is awaited
            # above), and calling a second method on its result would be guessing.
            if isinstance(factory, type):
                starter = _method(engine, ("build", "prepare", "warmup"))
                if starter is not None:
                    await _maybe_await(starter(**_accepted(starter, candidates)))
        except Exception as exc:  # noqa: BLE001 - degrade to the engine that works
            self.warnings.append(f"orchestration layer failed to build: {type(exc).__name__}")
            log.warning(
                "orchestrator_build_failed",
                error=str(exc)[:300],
                error_type=type(exc).__name__,
                fallback="linear",
            )
            return
        if _method(engine, _QUERY_METHODS) is None:
            self.warnings.append("orchestration layer exposes no query method; using linear")
            log.warning("orchestrator_unusable", methods=_QUERY_METHODS)
            return
        self.engine = engine
        self.orchestrated = True

    def _attach_semantic_cache(self) -> None:
        """Wire the answer-level cache.

        A hit here skips the *entire* pipeline — no retrieval, no reranking, no
        synthesis — which is why its hit rate matters more than the exact cache's:
        an exact hit saves one call, a semantic hit saves twenty. It is also the
        one tier that can be wrong, so the threshold lives in settings with a
        docstring telling operators to be conservative with it.
        """
        cfg = self.settings.cache
        if not (cfg.enabled and cfg.semantic_enabled):
            return
        from ragorc.cache.semantic import SemanticCache

        self.semantic = SemanticCache(self.linear.dense, settings=self.settings)

    async def aclose(self) -> None:
        orchestrated_close = _method(self.engine, _CLOSE_METHODS) if self.orchestrated else None
        if orchestrated_close is not None:
            try:
                await _maybe_await(orchestrated_close())
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("orchestrator_close_failed", error=str(exc)[:200])
        await self.linear.aclose()
        log.info("service_closed", uptime_s=round(time.monotonic() - self.started_at, 1))

    # -- query -------------------------------------------------------------
    async def query(
        self,
        request: QueryRequest,
        *,
        request_id: str = "",
        principal: str = "anonymous",
    ) -> QueryResponse:
        """Answer one question, inside a fresh trace and cost ledger."""
        request_id = request_id or _new_request_id()
        with self._request_context(request_id) as ledger:
            query, warnings = self.prepare(request, principal=principal)

            cached = await self._cache_get(query, request_id=request_id, pipeline=request.pipeline)
            if cached is not None:
                return cached

            answer, pipeline, notes = await self._dispatch(request, query)
            response = QueryResponse.from_answer(
                answer,
                request_id=request_id,
                question=request.question,
                pipeline=pipeline,
                warnings=[*warnings, *notes],
            )
            response.metadata["cost"] = ledger.report()
            response.metadata.setdefault("orchestrator", "graph" if self.orchestrated else "linear")

            await self._cache_set(query, response, answer=answer, pipeline=request.pipeline)
            self.audit.answered(
                tenant_id=query.tenant_id,
                cost_usd=ledger.total.cost_usd,
                chunks=len(answer.chunks),
                grounded=answer.grounded,
            )
            log.info(
                "query_answered",
                request_id=request_id,
                pipeline=pipeline.value,
                chunks=len(answer.chunks),
                abstained=answer.abstained,
                grounded=answer.grounded,
                groundedness=round(answer.groundedness, 3),
                cost_usd=round(ledger.total.cost_usd, 6),
                calls=ledger.total.calls,
            )
            return response

    async def stream(
        self,
        request: QueryRequest,
        *,
        request_id: str = "",
        principal: str = "anonymous",
        prepared: tuple[Query, list[str]] | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """Yield ``(event, data)`` pairs for a server-sent event stream.

        ``prepared`` accepts the output of :meth:`prepare` from a caller that has
        already validated. That is not an optimization — it is how the endpoint
        gets a real status code. A server-sent event response commits its ``200``
        with the headers, before the generator is ever resumed, so a rejected
        tenant discovered in here can only be reported as an event on a successful
        response. Validating first lets the guard answer with a 400.

        Streaming and verification are in genuine tension: groundedness can only
        be judged once the answer is complete, but the point of streaming is to
        emit before then. The honest resolution — the one
        :meth:`AnswerGenerator.stream` also takes — is to stream the text and
        *say* that it is unverified, never to stream while implying it was
        checked. The terminal ``done`` event carries the bill and that
        disclaimer; a client that needs verification calls ``POST /query``.

        The request context is opened inside the generator because the cost
        ceiling has to be live while tokens are being produced. Contextvars set
        in an async generator belong to whoever resumes it, and a server-sent
        event response is resumed by exactly one task, so the trace and ledger
        stay scoped to this request.
        """
        request_id = request_id or _new_request_id()
        with self._request_context(request_id) as ledger:
            query, warnings = prepared or self.prepare(request, principal=principal)
            for warning in [*warnings, *self._stream_notes(request)]:
                yield "warning", warning

            emitted = 0
            async for delta in self._deltas(request, query):
                emitted += 1
                yield "token", delta

            report = ledger.report()
            yield (
                "done",
                orjson.dumps(
                    {
                        "request_id": request_id,
                        "tokens_emitted": emitted,
                        "verified": False,
                        "note": (
                            "streamed text is not groundedness-checked; "
                            "POST /query returns a verified answer"
                        ),
                        "cost": report,
                    }
                ).decode(),
            )
            log.info(
                "query_streamed",
                request_id=request_id,
                deltas=emitted,
                cost_usd=report["total_cost_usd"],
            )

    def _stream_notes(self, request: QueryRequest) -> list[str]:
        """What answering this request as a stream had to substitute, named.

        ``POST /query`` carries its substitutions in ``QueryResponse.warnings``;
        a stream has nowhere to put them once the ``200`` is on the wire, so they
        go out as ``warning`` events ahead of the first token. Emitting nothing
        would mean a client asking to stream ``agentic`` received the adaptive
        linear pipeline with no way to tell — the same silent misreporting this
        module refuses to do on the non-streaming path.
        """
        if self.orchestrated and _method(self.engine, _STREAM_METHODS) is not None:
            return []
        notes = self.linear.notes_for(request.pipeline)
        if self.orchestrated:
            # The graph answers ``POST /query`` but exposes no streaming method,
            # so this one request bypasses it. Worth saying: the two endpoints are
            # being served by two different engines.
            notes.append("the orchestration layer cannot stream; streamed with the linear engine")
        return notes

    async def _deltas(self, request: QueryRequest, query: Query) -> AsyncIterator[str]:
        """Token deltas from whichever engine can produce them.

        The linear engine is used when the graph layer exposes no streaming
        method: streaming is a property of the generator, not of the graph, so a
        graph that only has ``ainvoke`` is a reason to bypass it rather than a
        reason to refuse the request.
        """
        streamer = _method(self.engine, _STREAM_METHODS) if self.orchestrated else None
        if streamer is None:
            source: Any = self.linear.stream(query, pipeline=request.pipeline)
        else:
            source = streamer(**_accepted(streamer, self._engine_kwargs(request, query)))
        if not inspect.isasyncgen(source):
            # A coroutine returning the iterator, rather than an async generator
            # function. Both are legitimate and only one can be iterated directly.
            source = await _maybe_await(source)
        async for delta in source:
            yield delta

    async def _dispatch(
        self, request: QueryRequest, query: Query
    ) -> tuple[Answer, PipelineName, list[str]]:
        """Run the graph when there is one, the linear engine otherwise."""
        if not self.orchestrated:
            return await self.linear.query(query, pipeline=request.pipeline)

        method = _method(self.engine, _QUERY_METHODS)
        assert method is not None  # guaranteed by _attach_orchestrator
        try:
            result = await _maybe_await(
                method(**_accepted(method, self._engine_kwargs(request, query)))
            )
        except RagOrcError:
            # A pipeline error is the pipeline's answer: guards, budgets and dead
            # stores all arrive this way and each maps to a specific status code.
            # Retrying them on the linear engine would spend the budget twice to
            # reach the same refusal.
            raise
        except Exception as exc:  # noqa: BLE001 - a broken graph must not 500 forever
            log.warning(
                "orchestrator_query_failed",
                error=str(exc)[:300],
                error_type=type(exc).__name__,
                fallback="linear",
            )
            answer, pipeline, notes = await self.linear.query(query, pipeline=request.pipeline)
            return answer, pipeline, [*notes, f"graph failed ({type(exc).__name__}); ran linear"]

        answer = _as_answer(result)
        pipeline = request.pipeline
        chosen = str(answer.metadata.get("pipeline") or "")
        with contextlib.suppress(ValueError):
            pipeline = PipelineName(chosen)
        return answer, pipeline, []

    def _engine_kwargs(self, request: QueryRequest, query: Query) -> dict[str, Any]:
        return {
            "question": query.text,
            "query": query,
            "tenant_id": query.tenant_id,
            "top_k": query.top_k,
            "pipeline": request.pipeline.value,
            "filters": query.filters or None,
        }

    # -- shared request plumbing ------------------------------------------
    @contextlib.contextmanager
    def _request_context(self, request_id: str) -> Iterator[CostLedger]:
        """Open the trace and the ledger, with this deployment's ceilings.

        The ceilings come from ``settings.cost`` and are checked *before* every
        model call. With ``track_costs`` off the cost ceiling is lifted but the
        call and token ceilings are not: those bound a runaway loop's latency and
        context usage, which is a liveness property rather than a billing one.

        Only the ledger is yielded. The trace is reachable from
        :func:`~ragorc.core.telemetry.current_trace` — and is already attached to
        every answer by the generator — so handing it out here would only invite a
        second, divergent copy.
        """
        cost = self.settings.cost
        with new_request_context(
            request_id=request_id,
            max_cost_usd=cost.max_cost_per_query_usd if cost.track_costs else None,
            max_calls=cost.max_llm_calls_per_query,
            max_tokens=cost.max_tokens_per_query,
            trace=self.settings.observability.trace_enabled,
        ) as (_trace, ledger):
            yield ledger

    def prepare(self, request: QueryRequest, *, principal: str) -> tuple[Query, list[str]]:
        """Validate, scope to a tenant, and audit — in that order.

        Order matters and it is cheapest-first: length and encoding checks reject
        for free, tenancy is a dict lookup, and only a request that survived both
        is worth writing to the audit log.

        Scoping is :func:`resolve_tenant`, not ``require_tenant``: ``tenant_id``
        is a request-*body* field, so checking only that one was named leaves any
        authenticated caller able to read any tenant by naming it. The principal
        is already here for the audit line — it just was not being used for the
        decision the audit line records.
        """
        validated = self.validator.validate(
            request.question, tenant_id=request.tenant_id, top_k=request.top_k
        )
        query = validated.query
        tenant = resolve_tenant(
            query.tenant_id,
            principal=principal,
            bindings=self.tenant_bindings,
            settings=self.settings.security,
        )
        query.tenant_id = tenant
        query.filters = scope_filter(request.filters, tenant, self.settings.security)
        self.audit.query(tenant_id=tenant, principal=principal, length=len(request.question))
        if validated.injection_risk:
            log.info(
                "query_injection_risk",
                risk=round(validated.injection_risk, 3),
                tenant_id=tenant,
            )
        return query, list(validated.warnings)

    async def _cache_get(
        self, query: Query, *, request_id: str, pipeline: PipelineName | None = None
    ) -> QueryResponse | None:
        if self.semantic is None:
            return None
        hit = await self.semantic.get(
            query.text,
            tenant_id=query.tenant_id,
            scope=scope_key(query.filters, query.top_k, pipeline=pipeline),
        )
        if hit is None:
            return None
        try:
            response = QueryResponse.model_validate(hit.answer)
        except Exception as exc:  # noqa: BLE001 - a stale shape is a miss
            # The cached payload was written by an older version of this schema.
            # Treating it as a miss is correct; failing the request because a
            # cache entry is out of date would turn a deploy into an outage.
            log.warning("semantic_cache_shape_stale", error=str(exc)[:200])
            return None
        response.request_id = request_id
        response.cached = True
        response.metadata["cache"] = {
            "tier": "semantic",
            "score": round(hit.score, 4),
            "cached_question": hit.question[:200],
        }
        log.info("query_cache_hit", request_id=request_id, score=round(hit.score, 4))
        return response

    async def _cache_set(
        self,
        query: Query,
        response: QueryResponse,
        *,
        answer: Answer | None = None,
        pipeline: PipelineName | None = None,
    ) -> None:
        """Store an answer, unless something about it makes it unrepeatable.

        ``SemanticCache.set`` declines abstentions itself, and that is the right
        place for that rule: an abstention is a statement about the index at one
        moment, and replaying it later hides content that has since been added.

        Three things it could not know, all of which this layer can:

        * **A degraded answer.** ``RAGPipeline._cache_set`` already refuses one —
          "the answer produced while a store was unreachable is not the answer this
          question has, it is the answer it had during an outage" — and the HTTP
          path implemented no such rule, so it served the outage for the whole
          TTL, long after the store came back.
        * **The pipeline.** ``graphrag`` and ``naive`` answer the same question
          differently *on purpose*; that is the entire reason a caller names one.
          Keyed without it, whichever ran first answered for both, so a benchmark
          comparing two pipelines measured one of them twice. Keyed on the
          *requested* pipeline rather than the resolved one, because that is what
          the read side knows — resolution happens after dispatch — and ``auto``
          resolves identically for identical settings.
        * **The chunk bodies.** ``chunks`` carries the retrieved passages in full.
          Storing them puts the whole retrieved context in the cache payload for
          every entry — the cost the payload projection exists to avoid — and
          re-serves passages whose source may since have been deleted. The text is
          reconstructible by re-running the query; the answer is the expensive part.
        """
        if self.semantic is None or response.cached:
            return
        if answer is not None and answer.metadata.get("errors"):
            log.info("semantic_cache_skipped", reason="degraded_answer")
            return
        payload = response.model_dump(mode="json", exclude={"chunks", "trace"})
        await self.semantic.set(
            query.text,
            payload,
            tenant_id=query.tenant_id,
            scope=scope_key(query.filters, query.top_k, pipeline=pipeline),
        )

    # -- ingest ------------------------------------------------------------
    async def documents(
        self,
        *,
        tenant_id: str | None = None,
        source: str | None = None,
        limit: int = 100,
        request_id: str = "",
        principal: str = ANONYMOUS,
    ) -> DocumentsResponse:
        """List indexed documents, scoped to the caller's tenant."""
        request_id = request_id or _new_request_id()
        with self._request_context(request_id):
            tenant = resolve_tenant(
                tenant_id or self.settings.tenant_id,
                principal=principal,
                bindings=self.tenant_bindings,
                settings=self.settings.security,
            )
            rows = await self.engine.documents(tenant_id=tenant, source=source, limit=limit)
        return DocumentsResponse(
            request_id=request_id,
            documents=[DocumentSummary(**row) for row in rows],
        )

    async def delete(
        self,
        request: DeleteRequest,
        *,
        request_id: str = "",
        principal: str = ANONYMOUS,
    ) -> DeleteResponse:
        """Remove documents from every store that holds them.

        Tenancy is resolved the same way ingest resolves it, and for the same
        reason stated there: a caller able to file documents under another
        tenant's id is a caller able to have them answered back to that tenant —
        and a caller able to *delete* under another tenant's id is worse, because
        that one does not need a second step.
        """
        request_id = request_id or _new_request_id()
        with self._request_context(request_id):
            tenant = resolve_tenant(
                request.tenant_id or self.settings.tenant_id,
                principal=principal,
                bindings=self.tenant_bindings,
                settings=self.settings.security,
            )
            report = await self.engine.delete(request.document_ids, tenant_id=tenant)
        return DeleteResponse(
            request_id=request_id,
            documents=report.documents,
            found=report.found,
            deleted=report.deleted,
            vectors=report.vectors,
            rows=report.rows,
            entities=report.entities,
            communities=report.communities,
            answers_invalidated=report.answers_invalidated,
            complete=report.complete,
            errors=dict(report.errors),
        )

    async def ingest(
        self,
        request: IngestRequest,
        *,
        request_id: str = "",
        principal: str = ANONYMOUS,
        staged_root: Path | None = None,
    ) -> IngestResponse:
        """Index inline text and/or server-side paths.

        Inline text gets a deterministic id derived from ``source`` and a checksum
        derived from its content, which is what makes re-posting the same document
        a skip instead of a duplicate. A caller that supplies no ``source`` gets
        one derived from the content, so the document is still idempotent — it
        just cannot be *updated*, because nothing identifies it across edits.

        ``staged_root`` is the *one* path root a request may carry that no caller
        chose: :func:`_staged_uploads` writes an upload to a temporary directory
        and then asks for that directory by path, so the upload transport would
        otherwise be refused by its own confinement check (see
        :func:`_resolve_paths`). Passed explicitly, per request, rather than
        allowlisting the system temp directory — which is world-writable, and
        would let a caller ingest anything another process had left in it.
        """
        request_id = request_id or _new_request_id()
        with self._request_context(request_id):
            # Writes are bound to the credential for the same reason reads are:
            # unbound, a caller could file documents under another tenant's id and
            # have them answered back to that tenant.
            tenant = resolve_tenant(
                request.tenant_id or self.settings.tenant_id,
                principal=principal,
                bindings=self.tenant_bindings,
                settings=self.settings.security,
            )
            targets: list[Any] = []
            if request.text:
                targets.append(_inline_document(request, tenant))
            roots = [staged_root] if staged_root is not None else _ingest_roots()
            targets.extend(_resolve_paths(request.paths, roots=roots))

            ingester = _method(self.engine, _INGEST_METHODS) or self.linear.ingest
            # `recursive` and `metadata` are fields on the request that reached no
            # loader: the service turns `paths` into bare `Path` objects and the
            # loader was built from settings alone, so a caller asking for a
            # non-recursive ingest got a recursive one and per-request metadata
            # never touched a document.
            #
            # `source_root` is the upload transport's fix for identity. A staged
            # file's absolute path carries a fresh random component per request, so
            # a document id derived from it was new every time: re-uploading the
            # same file produced a second document instead of a skip, forever.
            # Labelling relative to the staging root makes the id depend on the
            # filename the user sent, which is what they mean by "the same file".
            loader_options: dict[str, Any] = {"recursive": request.recursive}
            if request.metadata:
                loader_options["metadata"] = dict(request.metadata)
            if staged_root is not None:
                loader_options["source_root"] = staged_root
            report = _as_report(
                await _maybe_await(
                    ingester(targets, **_accepted(ingester, {"loader_options": loader_options}))
                )
            )
            response = IngestResponse.from_report(report, request_id=request_id)
            log.info("ingest_complete", request_id=request_id, **report.summary())
            return response

    # -- eval --------------------------------------------------------------
    async def evaluate(
        self,
        request: EvalRequest,
        *,
        request_id: str = "",
        principal: str = ANONYMOUS,
        dataset_roots: Sequence[Path] | None = None,
    ) -> EvalResponse:
        """Score one or more pipelines over the same dataset.

        The harness itself is :class:`ragorc.eval.runner.EvalRunner`, which already
        owns the metrics, the per-case error isolation and — the part worth
        delegating for — the *paired bootstrap*. Two means cannot say whether a
        difference is real; pairing by case id removes the between-question
        variance that otherwise dominates the estimate. Re-implementing any of
        that here would produce a second set of numbers that look like the first
        and are not.

        Same dataset, same process, same index: running the variants here rather
        than as separate invocations removes every confound except the one under
        test. Pipelines are scored **sequentially** while cases within a pipeline
        run concurrently — interleaving variants would have them competing for one
        provider concurrency limit, so the latency percentiles would measure the
        harness rather than the pipeline.
        """
        from ragorc.eval.dataset import EvalDataset
        from ragorc.eval.runner import EvalRunner, compare_runs

        request_id = request_id or _new_request_id()
        # `dataset_roots` is the caller saying who supplied the path. Left unset —
        # which is what the HTTP handler does — the loader confines it to the
        # ingest allowlist, because over HTTP the path is caller-supplied and the
        # response quotes the dataset back. The CLI passes the directory the
        # operator named, since there the path came from their own shell.
        items = await load_eval_items(request, roots=dataset_roots)
        if request.limit:
            items = items[: request.limit]
        if not items:
            raise ValidationFailed("eval dataset is empty", dataset=request.dataset or "inline")

        name = Path(request.dataset).stem if request.dataset else "inline"
        dataset = EvalDataset(
            cases=[_as_case(item) for item in items],
            name=name,
            source=request.dataset or "inline",
        )

        reports = []
        for pipeline in _unique([request.pipeline, *request.compare]):
            runner = EvalRunner(
                self.answer_fn(pipeline, request, principal=principal),
                self.settings,
                concurrency=request.concurrency,
            )
            reports.append((pipeline, await runner.run(dataset, name=pipeline.value)))

        comparisons = [
            compare_runs(reports[0][1], report).to_dict() for _pipeline, report in reports[1:]
        ]
        return EvalResponse(
            request_id=request_id,
            dataset=dataset.source or name,
            items=len(items),
            results=[_metrics_of(pipeline, report) for pipeline, report in reports],
            comparisons=comparisons,
            warnings=_unique(
                warning
                for _pipeline, report in reports
                for warning in (
                    [f"{report.name}: {report.n_errors} case(s) failed"] if report.n_errors else []
                )
            ),
        )

    def answer_fn(
        self, pipeline: PipelineName, request: EvalRequest, *, principal: str
    ) -> Callable[[str], Awaitable[Answer]]:
        """Bind one pipeline into the ``question -> Answer`` shape the runner wants.

        Each case gets its own trace and ledger, but **without the request-path
        ceilings**. That is deliberate and it follows the harness's own reasoning:
        an eval case is a query *plus* its judges, so enforcing a per-query cost
        cap here would abort legitimate cases and turn a budget knob into flaky
        measurements. The run is bounded by the size of the dataset instead, and
        the bill is aggregated from the ``Usage`` every call already returns.
        """

        async def answer(question: str) -> Answer:
            body = QueryRequest(
                question=question[:MAX_QUESTION_CHARS],
                tenant_id=request.tenant_id,
                top_k=request.top_k,
                pipeline=pipeline,
            )
            with new_request_context(request_id=_new_request_id()):
                query, _warnings = self.prepare(body, principal=principal)
                result, _pipeline, _notes = await self._dispatch(body, query)
                return result

        return answer

    # -- health ------------------------------------------------------------
    async def health(self) -> HealthResponse:
        """Probe every wired store, then report with a redacted config summary.

        Each store is probed with a real query under its own short deadline, and
        a failure is recorded rather than raised: one dead store degrades answers
        instead of failing the service, so the honest status is ``degraded`` and
        the honest response code is 200. An orchestrator that needs to take the
        pod out of rotation should key off the individual store entries.
        """
        probes: dict[str, Any] = {
            "qdrant": _probe("qdrant", self._probe_qdrant()),
            "postgres": _probe("postgres", self._probe_postgres()),
        }
        if self.linear.graph is not None or self.settings.graph.enabled:
            probes["neo4j"] = _probe("neo4j", self._probe_neo4j())

        probed = await gather_dict(probes, limit=len(probes))
        stores = [
            value
            if isinstance(value, StoreHealth)
            else StoreHealth(name=name, status="unavailable", error=str(value)[:300])
            for name, value in probed.items()
        ]
        degraded = [store.name for store in stores if store.status != "ok"]
        warnings = list(self.warnings)
        if not self.settings.server.api_keys:
            warnings.append("server.api_keys is empty: this service is unauthenticated")
        unbound = unbound_principals_warning(self.settings)
        if unbound:
            warnings.append(unbound)
        refused = graph_isolation_warning(self.settings)
        if refused:
            warnings.append(refused)

        return HealthResponse(
            status="degraded" if degraded else "ok",
            version=_VERSION,
            environment=self.settings.environment,
            uptime_s=round(time.monotonic() - self.started_at, 1),
            pipelines=list(PipelineName),
            stores=stores,
            settings=self.settings.summary(),
            cache=self._cache_stats(),
            warnings=warnings,
        )

    async def _probe_qdrant(self) -> StoreHealth:
        # Through the store's own ``health``, not ``count``. A count applies the
        # tenant filter, which fails closed — so with tenant isolation enabled and
        # no service-wide default tenant, this probe used to raise and report a
        # healthy Qdrant as ``unavailable``. Collection info answers "is the server
        # up and is the collection there" without a tenant being part of the
        # question, and is cheaper than a count besides. Same shape as
        # :meth:`_probe_neo4j`.
        return StoreHealth(name="qdrant", detail=await self.linear.vector.health())

    async def _probe_postgres(self) -> StoreHealth:
        count = await self.linear.relational.count()
        return StoreHealth(name="postgres", detail={"chunks": count})

    async def _probe_neo4j(self) -> StoreHealth:
        detail = await self.linear.graph_store().health()
        return StoreHealth(name="neo4j", detail=dict(detail))

    def _cache_stats(self) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        tier_stats = getattr(self.linear.cache, "stats", None)
        if callable(tier_stats):
            reported = tier_stats()
            stats.extend(reported if isinstance(reported, list) else [reported])
        if self.semantic is not None:
            stats.append(self.semantic.stats())
        return stats


# ---------------------------------------------------------------------------
# Helpers used by the service
# ---------------------------------------------------------------------------
def _as_answer(result: Any) -> Answer:
    """Normalize whatever a graph returned into an :class:`Answer`.

    LangGraph's ``ainvoke`` returns the final *state*, a plain dict, while a
    convenience wrapper returns the answer directly. Both are accepted so this
    service does not care which one the orchestration layer settled on.
    """
    if isinstance(result, Answer):
        return result
    if isinstance(result, dict):
        answer = result.get("answer")
        if isinstance(answer, Answer):
            return answer
        if isinstance(answer, str):
            return Answer(text=answer)
    answer = getattr(result, "answer", None)
    if isinstance(answer, Answer):
        return answer
    raise ConfigError(
        "the orchestration layer returned no Answer",
        returned=type(result).__name__,
        hint="a query method must return an Answer, or a state dict containing one",
    )


def _as_report(result: Any) -> Any:
    """Confirm an ingest returned something reportable.

    Checked by shape rather than by class: an orchestrated ingest may wrap the
    :class:`~ragorc.index.pipeline.IngestReport` or return its own equivalent, and
    what this service needs is the two accessors the response is built from. A
    clear error here beats an ``AttributeError`` from inside the serializer, which
    would report the ingest as an internal failure after it had already written
    every vector.
    """
    if callable(getattr(result, "summary", None)) and hasattr(result, "warnings"):
        return result
    raise ConfigError(
        "the ingest returned no report",
        returned=type(result).__name__,
        hint="an ingest must return an IngestReport",
    )


def _inline_document(request: IngestRequest, tenant: str | None) -> Document:
    text = request.text or ""
    source = request.source or f"inline:{content_hash(text, size=8)}"
    return Document(
        id=document_id(source, tenant_id=tenant),
        content=text,
        metadata={**request.metadata, "loader": "inline"},
        source=source,
        title=request.title,
        checksum=content_hash(text),
        tenant_id=tenant,
    )


_MAX_EVAL_DATASET_BYTES = 64 * 1024 * 1024
"""Ceiling on a server-side eval dataset. Generous — a dataset with reference
answers is legitimately megabytes — but finite, because the file is read whole
into a process that is also serving queries."""

_INGEST_ROOTS_ENV = "RAGORC_INGEST_ROOTS"
"""Env var holding the directories an HTTP caller may ingest from.

``os.pathsep``-separated absolute paths (``/srv/corpus:/srv/uploads``). Read from
the environment rather than from :class:`~ragorc.core.settings.Settings` because
it is a property of the *deployment's filesystem*, like a chroot or a systemd
``ReadOnlyPaths=``, not a tunable of the RAG pipeline — and because a confinement
boundary a request body could reach (settings are echoed by ``/health``, and
``Settings`` is constructible from any dict) is not a boundary. Unset means "no
server-side path is ingestible over HTTP", which is the safe default and the one
every deployment gets until an operator names a root.
"""


def _ingest_roots() -> list[Path]:
    """The configured ingest roots. Empty when none are configured.

    Read per request, not memoized: the value is one environment lookup, and
    caching it would mean a root added to a restarted-in-place process still
    reads as "no roots" — a confinement rule that silently ignores its own
    configuration is the failure mode to avoid here.
    """
    raw = os.environ.get(_INGEST_ROOTS_ENV, "")
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def _resolve_paths(paths: Iterable[str], *, roots: Sequence[Path]) -> list[Path]:
    """Confine each path to ``roots``, and check it exists, before any is loaded.

    Confinement first, because without it this endpoint is an arbitrary file read
    with a read-back channel: ``POST /ingest {"paths": ["/etc"]}`` indexed
    whatever the service user could open, and ``POST /query`` returns chunk text
    verbatim in ``QueryResponse.chunks[]``. Anything the process can read, a
    caller could then extract — and ``server.api_keys`` is empty by default, so
    "a caller" means anyone who can reach the port.

    ``roots`` is the allowlist, and an **empty allowlist rejects every path**
    rather than allowing every path. That is the inversion this had backwards: the
    common deployment ingests through inline ``text`` and multipart uploads (both
    unaffected — neither goes through here with a caller-supplied path), so
    refusing server-side paths until an operator sets :data:`_INGEST_ROOTS_ENV`
    costs those deployments nothing and closes the hole for the ones that never
    knew they had it.

    ``resolve()`` on **both** sides before comparing, so ``..`` traversal and a
    symlink pointing out of the root are settled against the real path rather than
    the spelling of it (:meth:`~pathlib.Path.is_relative_to` on unresolved paths
    would pass both), and so a root that is itself a symlink — ``/var`` on macOS,
    where the upload staging directory lives — still contains its own children.

    Existence is checked up front rather than per file so a typo in the third of
    four paths fails before the first three have been embedded, which is the
    cheapest possible moment to discover it.
    """
    bounds = [root.expanduser().resolve() for root in roots]
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not bounds:
            raise ValidationFailed(
                "ingesting a server-side path is disabled",
                path=str(path),
                hint=(
                    f"upload the file instead, post it as `text`, or set {_INGEST_ROOTS_ENV} "
                    "to the directories this service may read"
                ),
            )
        if not any(path.is_relative_to(root) for root in bounds):
            # The rejection names the roots: the caller cannot enumerate the
            # filesystem with this, and an operator debugging a legitimate 422
            # otherwise has to guess which of their roots was meant to match.
            raise ValidationFailed(
                "ingest path is outside the configured ingest roots",
                path=str(path),
                roots=[str(root) for root in bounds],
            )
        if not path.exists():
            raise ValidationFailed("ingest path does not exist", path=str(path))
        resolved.append(path)
    return resolved


async def _probe(name: str, coro: Awaitable[StoreHealth]) -> StoreHealth:
    """Run one store probe under a deadline, turning any failure into a report."""
    started = time.perf_counter()
    try:
        health = await asyncio.wait_for(coro, timeout=_HEALTH_PROBE_TIMEOUT_S)
    except TimeoutError:
        return StoreHealth(
            name=name,
            status="unavailable",
            latency_ms=_HEALTH_PROBE_TIMEOUT_S * 1000.0,
            error=f"probe exceeded {_HEALTH_PROBE_TIMEOUT_S}s",
        )
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return StoreHealth(
            name=name,
            status="unavailable",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
    health.latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return health


# ---------------------------------------------------------------------------
# Dataset plumbing for the eval harness
# ---------------------------------------------------------------------------
async def load_eval_items(
    request: EvalRequest, *, roots: Sequence[Path] | None = None
) -> list[EvalItem]:
    """Read a dataset from disk, or take the inline one.

    Both JSONL (one case per line, as :class:`ragorc.eval.dataset.EvalDataset`
    writes) and a JSON array are accepted, because the first is what the synthetic
    generator produces and the second is what a human writes by hand.

    Every record goes through :class:`EvalItem` rather than straight into
    ``EvalCase.from_json``. That is the point of having a pydantic boundary at all:
    a malformed dataset fails naming the field that is wrong, and the length
    ceilings apply to a file on disk exactly as they do to a request body — a
    dataset is an untrusted input too, and one case with a megabyte question
    would otherwise be discovered by the embedder.

    The read happens in a thread: a dataset with reference answers is megabytes,
    and blocking the loop on it inside a process that is also serving queries is
    exactly what this codebase forbids.
    """
    if not request.dataset:
        return list(request.items)
    location = request.dataset

    def _read() -> tuple[Path, bytes]:
        # Expansion, confinement, existence and the read all happen in the thread.
        # Every one of them is a syscall — on a network filesystem, a slow one —
        # and doing them one at a time from the loop would stall it three times to
        # answer one question, with a race between the check and the read for free.
        #
        # Confined to the same roots as POST /ingest, and for the same reason:
        # over HTTP, `dataset` is a caller-supplied server-side path and an eval
        # response quotes the dataset's contents back, which made it an arbitrary
        # local file read with a read-back channel — the exact hole /ingest was
        # fixed for, reachable through a different door. An empty allowlist
        # refuses every path rather than allowing every path.
        #
        # `roots` is explicit because that reasoning is about *who supplied the
        # path*, and it does not transfer to the CLI: there the path came from the
        # operator's own shell and they can already read their own files.
        # Defaulting it to the environment allowlist confined `ragorc eval` too,
        # which made `make eval` — a documented headline command — fail every
        # time. The default stays server-shaped so a new HTTP caller is safe by
        # omission; the CLI passes the dataset's own directory.
        (path,) = _resolve_paths([location], roots=_ingest_roots() if roots is None else roots)
        if not path.is_file():
            raise ValidationFailed("eval dataset is not a file", path=str(path))
        size = path.stat().st_size
        if size > _MAX_EVAL_DATASET_BYTES:
            # Bounded for the same reason request bodies are: this is read whole
            # into memory in a process that is also serving queries.
            raise ValidationFailed(
                "eval dataset is too large",
                path=str(path),
                bytes=size,
                limit_bytes=_MAX_EVAL_DATASET_BYTES,
            )
        return path, path.read_bytes()

    path, raw = await asyncio.to_thread(_read)

    records: list[Any]
    stripped = raw.lstrip()
    if stripped.startswith(b"["):
        try:
            records = orjson.loads(stripped)
        except orjson.JSONDecodeError as exc:
            raise ValidationFailed("malformed JSON eval dataset", path=str(path)) from exc
    else:
        records = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            text = line.strip()
            if not text or text.startswith(b"#"):
                continue
            try:
                records.append(orjson.loads(text))
            except orjson.JSONDecodeError as exc:
                raise ValidationFailed(
                    "malformed JSONL eval dataset", path=str(path), line=lineno
                ) from exc

    items: list[EvalItem] = []
    for index, record in enumerate(records):
        try:
            items.append(EvalItem.model_validate(record))
        except Exception as exc:  # noqa: BLE001 - report the record, not a stack
            raise ValidationFailed(
                "invalid eval case", path=str(path), index=index, reason=str(exc)[:300]
            ) from exc
    return items


def _as_case(item: EvalItem) -> Any:
    """Cross the boundary: a validated request model becomes the harness's case.

    One direction only, and only here. ``EvalCase`` derives its id from the
    question text when none is given, which is what lets a comparison pair two
    runs of a regenerated dataset — so an empty ``id`` is passed through rather
    than filled in on this side.
    """
    from ragorc.eval.dataset import EvalCase

    return EvalCase(
        question=item.question,
        expected_answer=item.expected_answer,
        expected_chunk_ids=tuple(item.expected_chunk_ids),
        metadata=dict(item.metadata),
        id=item.id,
    )


def _metrics_of(pipeline: PipelineName, report: Any) -> EvalMetrics:
    """Project a :class:`ragorc.eval.runner.RunReport` onto the response model.

    ``operational()`` already returns the field names below — the mapping is a
    projection, not a translation — but it is written out rather than splatted so
    that a metric the harness renames fails here, at a name, instead of silently
    reporting a default of zero to whoever reads the results file.
    """
    operational = report.operational()

    def value(key: str) -> float:
        return round(float(operational.get(key, 0.0)), 6)

    retrieval = report.retrieval
    documents = report.document_retrieval
    # Chunk- and document-labelled cases are disjoint: the document pass grades
    # exactly the cases that carry a source document and no chunk ids. Summing
    # them is the count that makes the rest interpretable, and reading only the
    # chunk one reported `labelled 0` on the shipped dataset — where 18 of 20
    # cases are graded — under a caption saying "labelled cases only".
    labelled = int(getattr(retrieval, "n_labelled", 0) or 0) + int(
        getattr(documents, "n_labelled", 0) or 0
    )
    return EvalMetrics(
        pipeline=pipeline,
        items=int(value("items")),
        labelled=labelled,
        errors=int(value("errors")),
        abstain_rate=value("abstain_rate"),
        grounded_rate=value("grounded_rate"),
        groundedness_mean=value("groundedness_mean"),
        confidence_mean=value("confidence_mean"),
        citation_coverage=value("citation_coverage"),
        latency_p50_ms=value("latency_p50_ms"),
        latency_p95_ms=value("latency_p95_ms"),
        cost_usd_total=value("cost_usd_total"),
        cost_usd_per_query=value("cost_usd_per_query"),
        llm_calls_total=int(value("llm_calls_total")),
        cache_hit_rate=value("cache_hit_rate"),
        retrieval={**report.retrieval_metrics(), **report.document_retrieval_metrics()},
        answer=report.answer_metrics(),
    )


def _unique(values: Iterable[Any]) -> list[Any]:
    """Order-preserving dedupe. ``dict`` rather than ``set`` because the order of
    a comparison table is the order the operator asked for."""
    return list(dict.fromkeys(values))


# ---------------------------------------------------------------------------
# Reaching the service from a handler
# ---------------------------------------------------------------------------
def service_dependency(request: Request) -> RagService:
    """The one way a handler obtains the :class:`RagService`.

    A dependency rather than ``request.app.state.service`` read inline, for two
    reasons that are both about a wrong answer rather than about style.

    A service that is not there is a **503**, not a 500. ``app.state.service`` is
    set by the lifespan, so the attribute is missing in exactly one situation: the
    process is starting up (or an ASGI host mounted the app without running the
    lifespan). Reading it directly raises ``AttributeError``, which the backstop
    handler correctly reports as an internal error — and "internal error" tells a
    load balancer to page someone about a pod that just needed another second.

    It is also the seam FastAPI's ``dependency_overrides`` needs. Nothing else in
    this file can be stubbed: the components are built by the lifespan, so a test
    of the HTTP layer — routing, tenancy, the error mapping, the redaction — would
    otherwise have to start Qdrant, Postgres and an ONNX session to assert that a
    guardrail returns 400. Overriding this one function replaces the entire
    pipeline with a stub and leaves every line of the request path under test.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise TransientError(
            "the service is still starting",
            hint="retry shortly; the lifespan builds the embedder and the pools once",
        )
    return service


# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Metrics:
    """The five signals docs/operations.md says predict incidents.

    Deliberately not one metric per pipeline stage: a dashboard with sixty series
    gets one glance and then ignored, and the stage breakdown already exists on
    every answer's trace where it can be read for the request that needs it.
    """

    queries: Any
    latency: Any
    cost: Any
    groundedness: Any
    render: Any
    content_type: str


_METRICS: _Metrics | None = None
_METRICS_TRIED = False


def _service_metrics() -> _Metrics | None:
    """Build the collectors once per process.

    Once, because ``prometheus_client`` refuses a duplicate metric name and a
    second :func:`create_app` in the same interpreter — which is what a test
    suite does — would otherwise raise on import of the second app.
    """
    global _METRICS, _METRICS_TRIED
    if _METRICS_TRIED:
        return _METRICS
    _METRICS_TRIED = True
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    except ImportError:
        log.info("metrics_unavailable", hint="pip install 'ragorc[otel]'")
        return None

    _METRICS = _Metrics(
        queries=Counter(
            "ragorc_queries_total",
            "Queries answered, by pipeline and outcome.",
            ["pipeline", "outcome"],
        ),
        latency=Histogram(
            "ragorc_query_latency_seconds",
            "End-to-end query latency.",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
        ),
        cost=Histogram(
            "ragorc_query_cost_usd",
            "Model spend per query. Bucketed low: the interesting question is "
            "which queries cost 100x the median, not what the median is.",
            buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
        ),
        groundedness=Histogram(
            "ragorc_groundedness",
            "Groundedness score of returned answers.",
            buckets=(0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0),
        ),
        render=generate_latest,
        content_type=CONTENT_TYPE_LATEST,
    )
    return _METRICS


def _record(metrics: _Metrics | None, response: QueryResponse, elapsed_s: float) -> None:
    if metrics is None:
        return
    outcome = "cached" if response.cached else "abstained" if response.abstained else "answered"
    metrics.queries.labels(pipeline=response.pipeline.value, outcome=outcome).inc()
    metrics.latency.observe(elapsed_s)
    metrics.cost.observe(response.usage.cost_usd)
    if not response.abstained:
        metrics.groundedness.observe(response.groundedness)


# ---------------------------------------------------------------------------
# Error mapping
#
# Ordered most-specific-first, because the hierarchy overlaps: RateLimited and
# StoreUnavailable are both TransientError, and a dict keyed by class would pick
# whichever the MRO walk reached first rather than the one that carries meaning.
# ---------------------------------------------------------------------------
_STATUS_BY_ERROR: tuple[tuple[type[RagOrcError], int], ...] = (
    (GuardrailViolation, 400),
    (ValidationFailed, 422),
    (RateLimited, 429),
    (BudgetExceeded, 429),
    (StoreUnavailable, 503),
    (LLMError, 502),
    (ConfigError, 500),
    (ConstructionError, 422),
    (EmbeddingError, 500),
    (RetrievalError, 500),
    (TransientError, 503),
    (RagOrcError, 500),
)


_HTTP_ERROR_NAMES = {
    401: "Unauthorized",
    403: "Forbidden",
    404: "NotFound",
    405: "MethodNotAllowed",
    413: "PayloadTooLarge",
    422: "RequestInvalid",
}
"""Machine-readable names for the errors the framework raises rather than the
pipeline. A client should be able to branch on ``error`` without string-matching a
human-readable message that is free to change."""


def _status_for(exc: RagOrcError) -> int:
    for error_type, status in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return status
    return 500


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------
def create_app(settings: Settings | None = None) -> Any:
    """Build the FastAPI application.

    A factory rather than a module-level app so that a test, a second worker
    configuration or an embedding host can construct one with explicit settings.
    ``ragorc.server.app:app`` still resolves — via a module ``__getattr__`` — for
    ``uvicorn ragorc.server.app:app``, which is what docs/operations.md documents.
    """
    resolved = settings or get_settings()
    obs = resolved.observability
    configure_logging(obs.log_level, obs.log_json, resolved.security.redact_secrets_in_logs)
    # Before the loop exists, which is the only moment the policy can be swapped.
    # Harmless when uvicorn has already installed uvloop itself.
    install_uvloop()

    _require("fastapi", "server")
    from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import PlainTextResponse
    from starlette.exceptions import HTTPException as StarletteHTTPException
    from starlette.middleware.cors import CORSMiddleware

    # FastAPI resolves a handler's parameter types with ``typing.get_type_hints``,
    # which evaluates annotations against the *module* globals — not this
    # function's locals. Because this module uses ``from __future__ import
    # annotations``, every annotation is a string, and ``Request`` exists only
    # under ``TYPE_CHECKING``. Without this injection, resolution raises
    # ``NameError``, FastAPI silently falls back to treating ``request: Request``
    # as a required *query parameter*, and every endpoint answers 422 — including
    # ``/health``, whose whole job is to be reachable by a load balancer.
    #
    # Publishing the names the handler annotations reference is what keeps the
    # lazy import compatible with eager hint resolution — and the lazy import is
    # what keeps ``import ragorc.server.app`` free of the web framework, because
    # the CLI imports RagService from this module and must not need the ``server``
    # extra to run ``ragorc query``. Idempotent, so a second create_app() in one
    # interpreter is fine.
    globals().update(
        Request=Request,
        Response=Response,
        HealthResponse=HealthResponse,
        QueryRequest=QueryRequest,
        QueryResponse=QueryResponse,
        IngestRequest=IngestRequest,
        IngestResponse=IngestResponse,
        EvalRequest=EvalRequest,
        EvalResponse=EvalResponse,
    )

    metrics = _service_metrics()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Build the service on startup, release it on shutdown."""
        service = RagService(resolved)
        try:
            await service.build()
        except Exception:
            # A build failure must not leave half-opened pools behind: the
            # process is about to exit and uvicorn will not call shutdown for a
            # startup that raised.
            await service.aclose()
            raise
        app.state.service = service
        if not resolved.server.api_keys:
            log.warning(
                "api_keys_empty",
                effect="every endpoint is unauthenticated",
                hint="set RAGORC_SERVER__API_KEYS before exposing this service",
                environment=resolved.environment,
            )
        try:
            yield
        finally:
            await service.aclose()

    app = FastAPI(
        title="ragorc",
        version=_VERSION,
        summary="Retrieval-augmented generation over Qdrant, Postgres and Neo4j.",
        lifespan=lifespan,
        # No `default_response_class`. It used to be `ORJSONResponse`, justified
        # against stdlib json — which is no longer the alternative. FastAPI now
        # serializes a route's declared `response_model` straight to bytes with
        # Pydantic's Rust serializer, skipping the intermediate `model_dump()`
        # dict that orjson would then have to walk. Measured on a realistic
        # 13.1 KB QueryResponse: 0.016 ms native against 0.019 ms via
        # ORJSONResponse, so removing the deprecated class is also the faster
        # option. Every route that returns a body declares a response model,
        # which is the condition that path requires.
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.server.cors_origins),
        # Credentialed requests cannot be paired with a wildcard origin: browsers
        # reject the combination outright, so a service configured with "*" would
        # appear to allow everything and work for nothing.
        allow_credentials="*" not in resolved.server.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    _install_middleware(app, resolved)
    _install_error_handlers(app, RequestValidationError, StarletteHTTPException)

    require_key = _api_key_dependency(resolved, HTTPException, status)

    async def guard(
        request: Request,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(require_key),
    ) -> str:
        """Authenticate, then rate limit — keyed by API key, falling back to IP.

        Key first because it is the identity that actually costs money and the one
        an operator can revoke; IP only for an unauthenticated deployment, where
        it is the only handle available and a weak one (every caller behind one
        NAT shares a budget). Ordering matters the other way too: rate limiting
        before authenticating would let an anonymous flood consume the budget of
        the key it is guessing.
        """
        if not resolved.security.enable_rate_limit:
            return principal
        key = principal if principal != "anonymous" else f"ip:{_client_ip(request)}"
        await service.limiter.check(key)
        return principal

    # -- endpoints ---------------------------------------------------------
    @app.post(
        "/query",
        response_model=QueryResponse,
        summary="Answer a question, verified.",
        responses=_error_responses(),
    )
    async def query(
        body: QueryRequest,
        request: Request,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(guard),
    ) -> QueryResponse:
        """Run the pipeline and return a complete, groundedness-checked answer.

        The deadline is applied here and not in a middleware because it is only
        correct here: ingest and eval are long-running by nature and a 120-second
        cap would abort them mid-run, while a query that has taken two minutes has
        already lost its caller.
        """
        started = time.perf_counter()
        async with asyncio.timeout(resolved.server.request_timeout_s):
            response = await service.query(
                body, request_id=_request_id(request), principal=principal
            )
        _record(metrics, response, time.perf_counter() - started)
        return response

    @app.post(
        "/query/stream",
        summary="Stream answer tokens over server-sent events.",
        responses=_error_responses(),
    )
    async def query_stream(
        body: QueryRequest,
        request: Request,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(guard),
    ) -> Any:
        """Server-sent events: ``warning``, ``token``, ``done``, ``error``.

        Validation and tenancy run *before* the response is constructed, so a
        rejected query still gets a 400 or a 422 like every other endpoint. Once
        the stream has started that is no longer possible — the ``200`` went out
        with the headers — so anything that fails during generation arrives as a
        final ``error`` event instead. Closing the stream silently would leave the
        client unable to tell a failure from a very short answer.
        """
        sse = _require("sse_starlette.sse", "server")
        request_id = _request_id(request)
        prepared = service.prepare(body, principal=principal)

        async def events() -> AsyncIterator[Any]:
            try:
                async for event, data in service.stream(
                    body, request_id=request_id, principal=principal, prepared=prepared
                ):
                    yield sse.ServerSentEvent(event=event, data=data)
            except asyncio.CancelledError:
                # The client hung up. Not an error, and not something to report.
                raise
            except RagOrcError as exc:
                log.warning("stream_failed", request_id=request_id, error=str(exc)[:300])
                yield sse.ServerSentEvent(
                    event="error",
                    data=orjson.dumps(
                        ErrorResponse(
                            error=type(exc).__name__,
                            message=exc.message,
                            detail=_safe_detail(exc.detail),
                            request_id=request_id,
                        ).model_dump()
                    ).decode(),
                )
            except Exception as exc:  # noqa: BLE001 - the stream must close cleanly
                log.exception("stream_crashed", request_id=request_id)
                yield sse.ServerSentEvent(
                    event="error",
                    data=orjson.dumps(
                        ErrorResponse(
                            error="InternalError",
                            message="the stream failed",
                            request_id=request_id,
                        ).model_dump()
                    ).decode(),
                )
                del exc

        return sse.EventSourceResponse(events(), headers={"X-Request-ID": request_id})

    @app.post(
        "/ingest",
        response_model=IngestResponse,
        summary="Index text, server-side paths, or uploaded files.",
        responses=_error_responses(),
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {
                        "schema": IngestRequest.model_json_schema(
                            ref_template="#/components/schemas/{model}"
                        )
                    },
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "binary"},
                                },
                                "tenant_id": {"type": "string"},
                                "metadata": {"type": "string", "description": "JSON object"},
                            },
                        }
                    },
                }
            }
        },
    )
    async def ingest(
        request: Request,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(guard),
    ) -> IngestResponse:
        """Ingest, dispatching on content type.

        One route rather than two because "index this" is one operation with two
        transports, and a client should not have to know that a file upload is
        multipart while a path list is JSON. FastAPI cannot declare both shapes in
        one signature, so the body is parsed here and the OpenAPI document is
        supplied explicitly above — which keeps the generated docs honest about
        both forms.
        """
        request_id = _request_id(request)
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/"):
            # The staging directory is handed over as the one allowed root: the
            # service confines every other path to the operator's ingest roots,
            # and an upload's temporary directory is by definition not one of
            # them. See :func:`_resolve_paths`.
            async with _staged_uploads(request, resolved) as (body, staged_root):
                return await service.ingest(
                    body,
                    request_id=request_id,
                    principal=principal,
                    staged_root=staged_root,
                )
        return await service.ingest(
            await _json_body(request, IngestRequest, resolved),
            request_id=request_id,
            principal=principal,
        )

    @app.get(
        "/documents",
        response_model=DocumentsResponse,
        summary="List indexed documents.",
        responses=_error_responses(),
    )
    async def list_documents(
        request: Request,
        source: str | None = None,
        limit: int = 100,
        tenant_id: str | None = None,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(guard),
    ) -> DocumentsResponse:
        """The read that makes ``DELETE /documents`` usable.

        Authenticated and tenant-scoped like every other read: a list of what is
        indexed is a list of what exists, which is exactly the enumeration the
        400-not-403 choice elsewhere in this service exists to avoid leaking.
        """
        return await service.documents(
            tenant_id=tenant_id,
            source=source,
            limit=limit,
            request_id=_request_id(request),
            principal=principal,
        )

    @app.delete(
        "/documents",
        response_model=DeleteResponse,
        summary="Remove documents from every store that holds them.",
        responses=_error_responses(),
    )
    async def delete_documents(
        body: DeleteRequest,
        request: Request,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(guard),
    ) -> DeleteResponse:
        """The only way to take a document out of the index.

        Authenticated like ingest, and bound to the credential's tenant for a
        sharper version of the same reason: a caller who can write under another
        tenant's id needs a second step to cause harm, and one who can delete
        under it does not.

        A body on a ``DELETE`` rather than ids in the path or query, because a
        delete is naturally a batch — a re-ingest that supersedes a directory
        removes hundreds — and a URL is the wrong place for hundreds of ids and
        the wrong place for anything that ends up in an access log.
        """
        return await service.delete(body, request_id=_request_id(request), principal=principal)

    @app.post(
        "/eval",
        response_model=EvalResponse,
        summary="Score one or more pipelines over a dataset.",
        responses=_error_responses(),
    )
    async def evaluate(
        body: EvalRequest,
        request: Request,
        service: RagService = Depends(service_dependency),
        principal: str = Depends(guard),
    ) -> EvalResponse:
        """Run the eval harness. Long-running and deliberately untimed."""
        return await service.evaluate(body, request_id=_request_id(request), principal=principal)

    @app.get("/health", response_model=HealthResponse, summary="Store health and configuration.")
    async def health(service: RagService = Depends(service_dependency)) -> HealthResponse:
        """Unauthenticated on purpose.

        A health endpoint behind an API key cannot be scraped by the thing that
        needs it most — the load balancer — and every field it returns is already
        redacted by :meth:`Settings.summary`. What it must never do is grow a
        parameter, because an unauthenticated endpoint that takes input is an
        unauthenticated endpoint that does work.
        """
        return await service.health()

    @app.get("/metrics", summary="Prometheus exposition.")
    async def prometheus() -> Response:
        """Prometheus text format, when ``ragorc[otel]`` is installed.

        501 rather than an empty 200 when it is not: an empty exposition looks
        like a service with nothing to report, and a scraper will happily record
        that as healthy silence for a week.
        """
        if metrics is None:
            return PlainTextResponse(
                "metrics require the otel extra: pip install 'ragorc[otel]'",
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
            )
        body = await asyncio.to_thread(metrics.render)
        return Response(content=body, media_type=metrics.content_type)

    log.info(
        "app_created",
        environment=resolved.environment,
        authenticated=bool(resolved.server.api_keys),
        rate_limited=resolved.security.enable_rate_limit,
        metrics=metrics is not None,
        cors_origins=list(resolved.server.cors_origins),
    )
    return app


# ---------------------------------------------------------------------------
# Cross-cutting wiring
# ---------------------------------------------------------------------------
def _install_middleware(app: Any, settings: Settings) -> None:
    """Request ids, structlog binding, and a body-size ceiling."""

    @app.middleware("http")
    async def request_context(request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        # An id supplied by the caller is honoured so a trace spans a gateway and
        # this service, but it is length-clipped: it is echoed in a header and
        # written to every log line of the request, and an unbounded one is a log
        # injection vector dressed as a correlation id.
        incoming = request.headers.get("x-request-id", "")
        request_id = "".join(c for c in incoming if c.isalnum() or c in "-_")[:64] or (
            _new_request_id()
        )
        request.state.request_id = request_id

        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.server.max_body_bytes:
            return _json_error(
                413,
                ErrorResponse(
                    error="PayloadTooLarge",
                    message="request body exceeds server.max_body_bytes",
                    detail={"limit_bytes": settings.server.max_body_bytes},
                    request_id=request_id,
                ),
                request_id,
            )

        # Bound before the task that runs the endpoint is created, so the
        # endpoint's context inherits it; unbinding afterwards happens in this
        # task's own context and cannot disturb the child's.
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            # Logged while the contextvar is still bound. Outside the try it would
            # emit the one line carrying method, path and status *without* the
            # request id — leaving the access log the only place in the file that
            # cannot be joined to the request it describes, which is precisely
            # the join anyone reading it is trying to make.
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
            )
            return response
        finally:
            structlog.contextvars.unbind_contextvars("request_id")


def _install_error_handlers(
    app: Any, validation_error: type[Exception], http_error: type[Exception]
) -> None:
    """Map exceptions to status codes without ever leaking a stack trace."""

    @app.exception_handler(RagOrcError)
    async def ragorc_error(request: Any, exc: RagOrcError) -> Any:
        status_code = _status_for(exc)
        request_id = getattr(request.state, "request_id", "")
        # 5xx is ours and gets a full log; 4xx is the caller's and gets one line,
        # because a client looping on a rejected query would otherwise fill the
        # log with our own error reports.
        event = "request_failed" if status_code >= 500 else "request_rejected"
        getattr(log, "error" if status_code >= 500 else "info")(
            event,
            error=type(exc).__name__,
            message=exc.message,
            status=status_code,
            path=request.url.path,
        )
        headers = {}
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            # The client is being told to slow down; telling it *how long* is the
            # difference between backoff and a retry storm.
            #
            # Rounded *up*, and that is the whole point of the ceiling: the header
            # is whole seconds, so rounding 2.4 to the nearest would advertise 2 —
            # and a compliant client's first retry would land before the window
            # reopened and be rejected again. An under-reported Retry-After
            # manufactures the retry storm the header exists to prevent.
            headers["Retry-After"] = str(max(1, math.ceil(float(retry_after))))
        return _json_error(
            status_code,
            ErrorResponse(
                error=type(exc).__name__,
                message=exc.message,
                detail=_safe_detail(exc.detail),
                request_id=request_id,
            ),
            request_id,
            headers=headers,
        )

    @app.exception_handler(validation_error)
    async def body_invalid(request: Any, exc: Any) -> Any:
        """422 in this service's own error shape.

        FastAPI's default body echoes the offending ``input`` back. That is the
        caller's own data, but a field name misspelled in a request that also
        carried a credential would put the credential in the response — so the
        loc/msg/type are kept and the values are dropped.
        """
        request_id = getattr(request.state, "request_id", "")
        problems = [
            {"field": ".".join(str(p) for p in err.get("loc", ())), "problem": err.get("msg", "")}
            for err in exc.errors()[:20]
        ]
        return _json_error(
            422,
            ErrorResponse(
                error="RequestInvalid",
                message="the request body did not validate",
                detail={"problems": problems},
                request_id=request_id,
            ),
            request_id,
        )

    @app.exception_handler(http_error)
    async def http_failed(request: Any, exc: Any) -> Any:
        """Render framework errors — 401, 404, 405 — in this service's own shape.

        Without this, an unauthenticated call gets FastAPI's ``{"detail": ...}``
        while a rejected query gets :class:`ErrorResponse`, and a client has to
        parse two error formats from one API. The OpenAPI document declares one
        shape, so there had better be one shape.
        """
        request_id = getattr(request.state, "request_id", "")
        detail = getattr(exc, "detail", "")
        return _json_error(
            int(getattr(exc, "status_code", 500)),
            ErrorResponse(
                error=_HTTP_ERROR_NAMES.get(int(getattr(exc, "status_code", 500)), "HTTPError"),
                message=str(detail) if detail else "request failed",
                request_id=request_id,
            ),
            request_id,
            headers=dict(getattr(exc, "headers", None) or {}),
        )

    @app.exception_handler(TimeoutError)
    async def timed_out(request: Any, exc: TimeoutError) -> Any:
        request_id = getattr(request.state, "request_id", "")
        log.warning("request_timeout", path=request.url.path)
        return _json_error(
            504,
            ErrorResponse(
                error="Timeout",
                message="the request exceeded server.request_timeout_s",
                request_id=request_id,
            ),
            request_id,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Any, exc: Exception) -> Any:
        """The backstop. Full detail to the log, nothing but an id to the client.

        An unexpected exception is by definition one whose message was never
        reviewed for what it contains — a connection string, a prompt, a row of
        customer data. The request id is what connects the caller's report to the
        server-side traceback, and it is the only thing that needs to cross.
        """
        request_id = getattr(request.state, "request_id", "")
        log.exception("unhandled_exception", path=request.url.path, error=type(exc).__name__)
        return _json_error(
            500,
            ErrorResponse(
                error="InternalError",
                message="an unexpected error occurred; quote the request id",
                request_id=request_id,
            ),
            request_id,
        )


def _json_error(
    status_code: int,
    body: ErrorResponse,
    request_id: str,
    *,
    headers: dict[str, str] | None = None,
) -> Any:
    from fastapi.responses import JSONResponse

    # Built by hand rather than returned from a route, so FastAPI's serialization
    # does not apply and a response class has to be named. `JSONResponse` and its
    # stdlib encoder are correct here for the reason they are wrong on `/query`:
    # an error body is a few hundred bytes, where the 6x difference measured
    # against orjson is microseconds.
    merged = {"X-Request-ID": request_id, **(headers or {})}
    return JSONResponse(status_code=status_code, content=body.model_dump(), headers=merged)


def _error_responses() -> dict[int | str, dict[str, Any]]:
    """Declare the error shape once so the OpenAPI document is truthful."""
    return {
        code: {"model": ErrorResponse, "description": description}
        for code, description in (
            (400, "a guardrail rejected the request"),
            (401, "missing or invalid API key"),
            (422, "the request body or query did not validate"),
            (429, "rate limit or cost budget exhausted"),
            (500, "internal error"),
            (502, "the model provider failed"),
            (503, "a datastore is unavailable"),
        )
    }


def _api_key_dependency(settings: Settings, http_exception: Any, status: Any) -> Any:
    """Build the authentication dependency.

    Keys are compared with :func:`hmac.compare_digest` against every configured
    key. Not paranoia: ``==`` on strings returns as soon as it finds a differing
    byte, and that timing difference is enough to recover a key one byte at a
    time over enough requests. The comparison is also done on bytes, because
    ``compare_digest`` refuses non-ASCII ``str`` and a caller can put anything in
    a header.

    The principal returned is a *hash prefix* of the key, never the key: it lands
    in the audit log and in the rate-limiter's bucket names, and a credential in
    either is a credential in a backup.
    """
    keys = tuple(k.encode() for k in settings.server.api_keys if k)

    async def require_api_key(request: Request) -> str:
        if not keys:
            return "anonymous"
        presented = _presented_key(request)
        if presented and any(hmac.compare_digest(presented, candidate) for candidate in keys):
            # Through the shared helper, not an inline format string: the tenant
            # bindings look a principal up from the *configured* key, so the two
            # have to agree. If they drifted, every binding would quietly stop
            # matching and every key would go back to being unrestricted.
            return principal_for_key(presented.decode())
        log.info("auth_failed", path=request.url.path, presented=bool(presented))
        raise http_exception(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": 'Bearer realm="ragorc"'},
        )

    return require_api_key


def _presented_key(request: Any) -> bytes | None:
    """Read the key from ``X-API-Key`` or a bearer ``Authorization`` header.

    Both, because the two ecosystems disagree: gateways and curl users reach for
    ``X-API-Key``, generated SDK clients send ``Authorization: Bearer``. Rejecting
    either would be a support ticket rather than a security property.
    """
    header = request.headers.get("x-api-key")
    if header:
        return header.strip().encode()
    authorization = request.headers.get("authorization", "")
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() == "bearer" and credential.strip():
        return credential.strip().encode()
    return None


def _client_ip(request: Any) -> str:
    """The caller's address, preferring the first ``X-Forwarded-For`` hop.

    Trusting that header is only safe behind a proxy that rewrites it, which is
    the documented deployment (docs/operations.md puts uvicorn behind a load
    balancer). Directly exposed, it is spoofable — which is precisely why the API
    key is the primary rate-limit key and this is only the fallback.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()[:64]
    client = getattr(request, "client", None)
    return getattr(client, "host", "unknown") or "unknown"


def _request_id(request: Any) -> str:
    return getattr(request.state, "request_id", "") or _new_request_id()


async def _read_bounded(request: Any, limit: int) -> bytes:
    """Read a request body, stopping as soon as it passes ``limit``.

    ``await request.body()`` materializes first and checks second. For a chunked
    request that is the whole exposure: it declares no ``Content-Length``, so the
    middleware's early check never fires, and the only remaining limit was applied
    to a body already resident in memory. Accumulating from the stream keeps the
    rejection ahead of the allocation, which is the point of having a limit.
    """
    parts: list[bytes] = []
    total = 0
    async for part in request.stream():
        total += len(part)
        if total > limit:
            raise ValidationFailed(
                "request body exceeds server.max_body_bytes",
                limit_bytes=limit,
                received_bytes=total,
            )
        parts.append(part)
    return b"".join(parts)


async def _json_body(request: Any, model: type[Any], settings: Settings) -> Any:
    """Parse and validate a JSON body that the route could not declare.

    Read under a bound rather than read then measured: ``Content-Length`` is a
    claim, and a chunked request does not make one at all.
    """
    raw = await _read_bounded(request, settings.server.max_body_bytes)
    try:
        payload = orjson.loads(raw or b"{}")
    except orjson.JSONDecodeError as exc:
        raise ValidationFailed("request body is not valid JSON") from exc
    try:
        return model.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - reported as a 422, not a 500
        raise ValidationFailed(
            f"invalid {model.__name__}", reason=str(exc).splitlines()[0][:300]
        ) from exc


_UPLOAD_CHUNK_BYTES = 1 << 20
"""How much of one upload is held in memory at a time.

A megabyte, because that is Starlette's own spool threshold, so a part smaller
than this is copied from memory and a larger one is copied from the temporary
file it already spilled into."""


def _capped_receive(request: Any, budget: int) -> Any:
    """Wrap the ASGI receive channel so the body cannot exceed ``budget``.

    Placed here rather than in the middleware because the middleware runs before
    the body arrives and can only read the declared ``Content-Length``. This
    counts what actually turns up, so a chunked upload that declares nothing is
    bounded by the same number, and it raises on the chunk that crosses the line
    rather than after the whole body has been spooled to disk.
    """
    received = 0

    async def receive() -> Any:
        nonlocal received
        message = await request.receive()
        if message.get("type") == "http.request":
            received += len(message.get("body", b""))
            if received > budget:
                raise ValidationFailed(
                    "upload exceeds server.max_body_bytes",
                    limit_bytes=budget,
                    received_bytes=received,
                )
        return message

    return receive


async def _copy_part(value: Any, destination: Path, *, budget: int, written: int) -> int:
    """Copy one uploaded part to disk in bounded chunks, returning the new total.

    ``await value.read()`` with no argument returns the whole part as one
    ``bytes``, so the previous version materialized an entire file in memory and
    only then compared the running total against the budget — the check fired
    after the allocation it existed to prevent. Reading in fixed chunks makes the
    ceiling apply to the copy as well as to the transfer, which matters as soon
    as an operator raises ``max_body_bytes`` to take large PDFs.
    """

    def _write(handle: Any, data: bytes) -> None:
        handle.write(data)

    with destination.open("wb") as handle:
        while chunk := await value.read(_UPLOAD_CHUNK_BYTES):
            written += len(chunk)
            if written > budget:
                raise ValidationFailed(
                    "uploaded files exceed server.max_body_bytes",
                    limit_bytes=budget,
                    received_bytes=written,
                )
            await asyncio.to_thread(_write, handle, chunk)
    return written


@contextlib.asynccontextmanager
async def _staged_uploads(
    request: Any, settings: Settings
) -> AsyncIterator[tuple[IngestRequest, Path]]:
    """Stage uploaded files on disk, then hand the loaders a directory.

    Files rather than bytes because the loaders dispatch on *suffix* — a PDF and a
    CSV need different parsers and the extension is what selects them — and
    because a 20 MB PDF held in memory while it is parsed is memory this process
    does not have to spend. The directory is temporary and removed on the way out
    even when the ingest raised, so a failed upload cannot fill the disk.

    Two things are enforced here rather than trusted: the filename is reduced to
    its basename (an upload named ``../../etc/passwd`` writes ``passwd`` inside the
    temporary directory and nothing else), and the byte ceiling is applied at both
    points where bytes accumulate — on the wire before the parser spools them
    (:func:`_capped_receive`) and again as each part is copied out
    (:func:`_copy_part`). ``Content-Length`` is a claim about the whole envelope,
    so the middleware's check on it is a courtesy to honest clients and not a
    bound.

    Yields the request *and* the staging directory, because the directory is the
    only root the resulting path is allowed to be under: the caller of this
    context manager passes it to :meth:`RagService.ingest` as ``staged_root``, and
    :func:`_resolve_paths` refuses every path outside it. Returning the request
    alone would leave the upload transport indistinguishable from a caller asking
    for a path on the server.
    """
    import tempfile

    from starlette.requests import Request as _Request

    _require("multipart", "server")
    budget = settings.server.max_body_bytes
    # Counted off the wire, before the parser sees it. The middleware's ceiling
    # reads ``Content-Length``, which is a *claim*: a chunked request omits it
    # and a lying one is free. Starlette's own ``max_part_size`` does not cover
    # this either — ``on_part_data`` applies it only when ``file is None``, so a
    # field is bounded and an uploaded file is not.
    form = await _Request(request.scope, receive=_capped_receive(request, budget)).form()
    written = 0
    try:
        with tempfile.TemporaryDirectory(prefix="ragorc-upload-") as staging:
            root = Path(staging)
            names: list[str] = []
            for value in form.getlist("files"):
                filename = getattr(value, "filename", None)
                if not filename:
                    continue
                safe = Path(str(filename)).name
                if not safe or safe in {".", ".."}:
                    continue
                written = await _copy_part(value, root / safe, budget=budget, written=written)
                names.append(safe)
            if not names:
                raise ValidationFailed("no files were uploaded", field="files")

            metadata_raw = form.get("metadata")
            metadata: dict[str, Any] = {}
            if isinstance(metadata_raw, str) and metadata_raw.strip():
                try:
                    parsed = orjson.loads(metadata_raw)
                except orjson.JSONDecodeError as exc:
                    raise ValidationFailed("metadata is not valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise ValidationFailed("metadata must be a JSON object")
                metadata = parsed

            tenant = form.get("tenant_id")
            log.info("uploads_staged", files=len(names), bytes=written)
            yield (
                IngestRequest(
                    paths=[str(root)],
                    tenant_id=str(tenant) if isinstance(tenant, str) and tenant else None,
                    metadata=metadata,
                    recursive=False,
                ),
                root,
            )
    finally:
        await form.close()


# ---------------------------------------------------------------------------
# ``uvicorn ragorc.server.app:app``
# ---------------------------------------------------------------------------
_APP: Any = None


def __getattr__(name: str) -> Any:
    """Materialize the module-level ``app`` on first access (:pep:`562`).

    docs/operations.md runs this service as ``uvicorn ragorc.server.app:app``,
    which imports the module and then reads that attribute. Building the app at
    import time instead would mean ``import ragorc.server.app`` required FastAPI —
    breaking the CLI, which imports :class:`RagService` from here and never needs
    a web framework to run ``ragorc query``.
    """
    if name == "app":
        global _APP
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
