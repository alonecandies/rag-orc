"""The public facade: one object most users ever touch.

    async with await build_pipeline() as rag:
        await rag.ingest("./docs")
        answer = await rag.query("why is late chunking cheaper?")

Everything below this line is composition. There is no logic here that is not either
*wiring* (which component gets which collaborator) or *policy* (which graph runs, what
the request budget is, what gets cached). The retrieval, generation and verification
behaviour all lives in the layers underneath, and this file is deliberately the least
clever one in the library.

Laziness is a correctness property, not an optimization
-------------------------------------------------------
A deployment with only Qdrant configured must not open Postgres and Neo4j
connections — and the reason is not tidiness. Those settings have *defaults*
(``localhost:5432``, ``bolt://localhost:7687``), so eagerly constructing their stores
would mean: a driver import and a connection pool for a database nobody configured, a
health check that fails against whatever happens to be listening on that port, and a
startup that breaks when a user's local Postgres is stopped — for a feature they never
enabled. Worse, the failure would look like a ragorc bug rather than an unset setting.

So the relational and graph legs are wrapped in :class:`_LazyRetriever`, which builds
its store on the *first request that actually routes to it* and raises
:class:`~ragorc.core.errors.StoreUnavailable` if it cannot. That error is the shape
:class:`~ragorc.retrieve.multi_store.MultiStoreRetriever` already handles: the store is
recorded in ``RetrievalResult.errors``, its circuit breaker opens after a few attempts,
and the query is answered from the backends that do work.

The one exception is ``retrieval.use_fulltext``. Turning it on is an explicit statement
that Postgres is part of this deployment, so its store is built eagerly — the *route*
asking for Postgres is speculative, a settings flag saying so is not.

What ``create()`` does and does not do at startup
-------------------------------------------------
It warms the embedder (local, no network, and it is what pins the vector dimension), and
it refreshes the OpenRouter price table so cost accounting is correct rather than zero.

It deliberately does **not** create the Qdrant collection. Creating a collection as a
side effect of building a *query* client is how a typo'd collection name becomes a
permanent, silent "no documents found": the empty collection gets created, every query
succeeds, and nothing ever says why the answers are empty. Schema creation belongs to
:class:`~ragorc.index.pipeline.IngestPipeline`, which is the component whose job is to
put something in it.

Every request is a costed request
---------------------------------
:meth:`query` and :meth:`stream` both run inside
:func:`~ragorc.core.telemetry.new_request_context`, which installs a fresh trace and a
fresh :class:`~ragorc.core.telemetry.CostLedger` in contextvars. That is what makes the
per-stage timing, the itemized bill and — most importantly — the *hard ceilings* work:
the ledger is checked before every model call, so a runaway loop raises
:class:`~ragorc.core.errors.BudgetExceeded` instead of quietly spending money. A graph
with three feedback loops has no natural upper bound on spend, and this is the bound.

The semantic cache sits in front of all of it
---------------------------------------------
A semantic hit skips the entire pipeline — no retrieval, no reranking, no synthesis — so
it saves twenty calls where an exact cache hit saves one. That makes it the largest
single cost lever in a production deployment and the one setting to be conservative
with; the default threshold of 0.97 is strict on purpose, because below ~0.95 you start
answering questions nobody asked.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import orjson
import structlog

from ragorc.cache.semantic import scope_key
from ragorc.cache.tiered import build_cache
from ragorc.core.errors import ConfigError, StoreUnavailable
from ragorc.core.models import (
    Answer,
    Citation,
    DataStore,
    Document,
    Query,
    RetrievalResult,
    ScoredChunk,
)
from ragorc.core.protocols import (
    DenseEmbedder,
    LateInteractionEmbedder,
    Retriever,
    SparseEmbedder,
)
from ragorc.core.registry import resolve
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging, current_trace, new_request_context
from ragorc.generate.answer import AnswerGenerator
from ragorc.generate.self_rag import SelfRAG
from ragorc.llm.cache import LLMCache
from ragorc.llm.openrouter import OpenRouterLLM
from ragorc.llm.router import ModelRouter
from ragorc.pipeline.graphs import GRAPHS, build_graph, graph_names
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState, evidence, initial_state, total_usage
from ragorc.retrieve.compress import build_compressor
from ragorc.retrieve.hybrid import HybridRetriever
from ragorc.retrieve.rerank import build_reranker
from ragorc.security.audit import AuditLog
from ragorc.security.ratelimit import KeyedRateLimiter
from ragorc.translate import build_translators

log = structlog.get_logger(__name__)

__all__ = ["RAGPipeline", "build_pipeline"]

_DEFAULT_TRANSLATORS: tuple[str, ...] = ("multi_query",)
"""One translator by default.

Multi-query is the highest value per call: it covers the vocabulary mismatch that
sinks most dense retrieval, and the variants fan out inside one batched embed. Adding
step-back, decomposition and HyDE on every query multiplies translation cost by four
for a recall gain that most corpora do not show — so the full set is reserved for the
agentic graph, which is already paying for depth."""

_FULL_TRANSLATORS: tuple[str, ...] = ("multi_query", "step_back", "decomposition", "hyde")
"""Every strategy, chained. Used only by the agentic pipeline.

Order matters and is not alphabetical: step-back and decomposition are meant to see the
*original* question, and HyDE goes last because its output is a pseudo-document rather
than a question and should not be the input to another rewriter."""

_ACCUMULATORS: tuple[str, ...] = ("candidates", "usages", "errors", "warnings", "web_chunks")
"""State keys that append rather than replace, for the hand-driven streaming path."""


def _resolve_settings(settings: Settings | None, overrides: Mapping[str, Any]) -> Settings:
    """Apply ``**overrides`` on top of a settings tree.

    Nested keys use the same ``__`` separator as the environment
    (``llm__model="..."``, ``retrieval__crag_enabled=True``) so a programmatic override
    is spelled the same way as its environment variable. A whole submodel can also be
    replaced with a dict, which is what pydantic already accepts.

    Re-validating through ``Settings`` rather than mutating in place is what keeps
    ``model_post_init`` honest — the vector-dimension alignment and the production
    hardening both live there, and a mutated instance would silently skip them.
    """
    base = settings or get_settings()
    if not overrides:
        return base
    payload: dict[str, Any] = base.model_dump()
    for key, value in overrides.items():
        path = key.split("__")
        cursor = payload
        for part in path[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                raise ConfigError(
                    f"cannot override {key!r}: {part!r} is not a settings section",
                    available=sorted(k for k, v in cursor.items() if isinstance(v, dict)),
                )
            cursor = nxt
        leaf = path[-1]
        if leaf not in cursor:
            raise ConfigError(
                f"unknown setting {key!r}",
                hint="settings are fixed; see ragorc.core.settings.Settings",
            )
        if isinstance(cursor[leaf], dict) and isinstance(value, Mapping):
            cursor[leaf] = {**cursor[leaf], **value}
        else:
            cursor[leaf] = value
    return Settings.model_validate(payload)


def _component_params() -> frozenset[str]:
    """Keyword-only parameters of ``RAGPipeline.__init__`` that inject a component.

    Read from the signature instead of being listed by hand, so a new constructor
    parameter is injectable through ``create()`` without a second list to remember.
    ``settings`` is excluded: it is the positional configuration argument, not a
    component.
    """
    import inspect

    return frozenset(
        name
        for name, param in inspect.signature(RAGPipeline.__init__).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY and name != "settings"
    )


class _LazyRetriever:
    """A :class:`~ragorc.core.protocols.Retriever` that builds its store on first use.

    This is what makes "no Postgres connection unless a query routes to Postgres" true
    rather than aspirational. The store, its driver and its pool are all created inside
    :meth:`retrieve`, under a lock so a burst of concurrent first requests builds one
    store rather than eight.

    A build failure is converted to :class:`StoreUnavailable` because that is
    semantically what it is — the store is not available — and because it is the one
    error shape the multi-store fan-out already degrades around: the store lands in
    ``RetrievalResult.errors``, its breaker opens after a few attempts, and the query is
    answered from the backends that work. Failures are *not* cached: an unreachable
    database at 09:00 is often a reachable one at 09:01, and the breaker is the thing
    that stops us retrying too eagerly.
    """

    __slots__ = ("_factory", "_inner", "_lock", "label", "name")

    def __init__(self, name: str, label: str, factory: Callable[[], Awaitable[Retriever]]) -> None:
        self.name = name
        self.label = label
        self._factory = factory
        self._inner: Retriever | None = None
        self._lock = asyncio.Lock()

    @property
    def built(self) -> bool:
        return self._inner is not None

    @property
    def usage(self) -> Any:
        """Forward the inner retriever's published bill, if it has one.

        Without this, wrapping a text-to-SQL or multi-hop retriever in the proxy would
        silently drop its cost from the request ledger — the one thing this library
        promises never to do.
        """
        return getattr(self._inner, "usage", None)

    async def _resolve(self) -> Retriever:
        if self._inner is not None:
            return self._inner
        async with self._lock:
            if self._inner is not None:  # another waiter won the race
                return self._inner
            try:
                inner = await self._factory()
            except Exception as exc:
                log.warning(
                    "lazy_store_unavailable",
                    store=self.label,
                    error=str(exc)[:200],
                    error_type=type(exc).__name__,
                )
                raise StoreUnavailable(
                    self.label, f"could not build the {self.label} retriever: {exc}"
                ) from exc
            log.info("lazy_store_built", store=self.label, retriever=self.name)
            self._inner = inner
            return inner

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        inner = await self._resolve()
        return await inner.retrieve(query, top_k=top_k, **kwargs)


class RAGPipeline:
    """The assembled system: components, graphs, cache and request lifecycle.

    Construction is synchronous and does no I/O — every component is built on first use
    — so a :class:`RAGPipeline` can be created in a constructor, a fixture or a
    dependency-injection provider. :meth:`create` is the async factory that additionally
    warms what benefits from warming.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: Any | None = None,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        late_embedder: LateInteractionEmbedder | None = None,
        cache: Any | None = None,
        vector_store: Any | None = None,
        relational_store: Any | None = None,
        graph_store: Any | None = None,
        generator: AnswerGenerator | None = None,
        retriever: Any | None = None,
        reranker: Any | None = None,
        constructor: Any | None = None,
        translators: Sequence[str] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self._dense = dense_embedder
        self._sparse = sparse_embedder
        self._late = late_embedder
        self._cache = cache
        self._vector = vector_store
        self._relational = relational_store
        self._graph = graph_store
        self._generator = generator
        self._translator_names = tuple(translators) if translators else _DEFAULT_TRANSLATORS
        # Pre-built retriever and reranker seed the lazy slots below rather than
        # sitting in separate fields. An embedding host that already constructed
        # them — the HTTP service does, to answer with its linear engine before the
        # graphs are wired — must not cause a second ONNX session to be loaded for
        # the same model.
        self._injected_retriever = retriever
        self._injected_reranker = reranker

        self._embedding_cache: Any | None = None
        self._semantic_cache: Any | None = None
        self._model_router: ModelRouter | None = None
        self._hybrid: HybridRetriever | None = retriever
        self._multi_store: Any | None = None
        self._reranker: Any | None = reranker
        self._compressor: Any | None = None
        self._crag: Any | None = None
        self._self_rag: SelfRAG | None = None
        self._web: Any | None = None
        self._router: Any | None = None
        # Injected, not just lazily built: the property below constructs a
        # SelfQueryConstructor with an empty attribute schema, which is a
        # deliberate no-op, and its docstring told the reader to "inject it" when
        # there was no parameter to inject through. `_COMPONENT_PARAMS` is derived
        # from this signature, so adding it here also makes `create(constructor=…)`
        # work rather than being read as a settings path.
        self._constructor: Any | None = constructor
        self._ingest: Any | None = None
        self._node_bundles: dict[bool, PipelineNodes] = {}
        self._compiled: dict[str, Any] = {}
        self._lazy_legs: dict[str, _LazyRetriever] = {}
        self._ratelimiter = KeyedRateLimiter.from_settings(self.settings.security)
        self._audit = AuditLog(self.settings)
        self._closed = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    async def create(cls, settings: Settings | None = None, **overrides: Any) -> RAGPipeline:
        """Build a pipeline from settings, applying ``**overrides``.

        Logging is configured here rather than at import: a library that reconfigures
        logging on import is a library that fights the application embedding it, and
        ``configure_logging`` is idempotent so an application that already called it
        keeps its own configuration.
        """
        # `**overrides` carries two different kinds of thing, and conflating them
        # was a real bug: a caller that has already built an LLM, a store or an
        # embedder wants to *inject* it, while a caller tweaking configuration wants
        # a settings path. Routing everything to settings made
        # `create(vector_store=store)` raise "unknown setting 'vector_store'" — and
        # because the server catches that and degrades, the entire LangGraph layer
        # silently fell back to a linear engine with only a warning to show for it.
        #
        # The split is by name against the constructor's own signature, so the two
        # cannot drift: adding a component parameter to __init__ makes it injectable
        # here automatically.
        components = {k: overrides.pop(k) for k in _COMPONENT_PARAMS if k in overrides}

        resolved = _resolve_settings(settings, overrides)
        obs = resolved.observability
        configure_logging(obs.log_level, obs.log_json, resolved.security.redact_secrets_in_logs)
        # uvloop is deliberately *not* installed here. This is an `async def`, so a
        # loop is already running, and a policy cannot replace a running loop:
        # `install_uvloop` detects exactly that, returns False and warns
        # `uvloop_not_in_use` — so the call could never do anything but tell the
        # operator it had not worked. It belongs at the process entry point, before
        # the loop starts, which is where the CLI and the server already call it
        # (`ragorc/cli.py`, `ragorc/server/app.py`); an embedding application does
        # the same in its own `main`, as `install_uvloop`'s docstring shows.
        pipeline = cls(resolved, **components)
        if components:
            log.info("pipeline_components_injected", names=sorted(components))
        await pipeline._warmup()
        log.info("pipeline_ready", **pipeline.describe()["features"])
        return pipeline

    async def _warmup(self) -> None:
        """Pay the one-time costs now instead of on the first user's query.

        Two things, both best-effort. The embedder warmup loads the ONNX session and
        pins the true vector dimension — it is local, so it either works or the
        deployment is broken. The price table is a network call to OpenRouter, and a
        failure there degrades cost *estimation* (the pre-flight budget check) while
        leaving the *actual* per-call cost, which the provider reports on every
        response, entirely correct. Failing startup over it would be the wrong trade.
        """
        embedder = self.dense_embedder
        warmup = getattr(embedder, "warmup", None)
        if callable(warmup):
            with contextlib.suppress(Exception):
                await warmup()
        # A local table takes precedence over the network: a deployment that pins
        # its prices wants the pinned numbers, and it also wants a cost estimate
        # without an outbound call at startup. `cost.price_table_path` was
        # documented and read by nothing, so pinning silently did nothing.
        if self.settings.cost.price_table_path:
            configured = self.settings.cost.price_table_path

            def _read() -> Any:
                # Expansion and the read both happen in the thread: they are
                # syscalls, and splitting them puts one of them on the loop.
                return json.loads(Path(configured).expanduser().read_text())

            path = configured
            try:
                loaded = await asyncio.to_thread(_read)
                self.model_router.prices = {
                    str(model): {str(k): float(v) for k, v in entry.items()}
                    for model, entry in loaded.items()
                }
                log.info("price_table_loaded", path=str(path), models=len(loaded))
            except Exception as exc:  # noqa: BLE001 - estimation only, never fatal
                log.warning("price_table_unreadable", path=str(path), error=str(exc)[:200])
        elif self.settings.cost.refresh_prices:
            fetch = getattr(self.llm, "fetch_model_prices", None)
            if callable(fetch):
                try:
                    self.model_router.prices = await fetch()
                except Exception as exc:  # noqa: BLE001 - estimation only, never fatal
                    log.warning("price_table_unavailable", error=str(exc)[:200])

    # ------------------------------------------------------------------
    # Components — every one built on first access
    # ------------------------------------------------------------------
    @property
    def llm(self) -> Any:
        if self._llm is None:
            cache = (
                LLMCache(self.cache, self.settings.cache) if self.settings.cache.cache_llm else None
            )
            self._llm = OpenRouterLLM(self.settings.llm, cache=cache)
        return self._llm

    @property
    def cache(self) -> Any:
        if self._cache is None:
            self._cache = build_cache(self.settings.cache)
        return self._cache

    @property
    def model_router(self) -> ModelRouter:
        if self._model_router is None:
            self._model_router = ModelRouter(self.settings.llm)
        return self._model_router

    @property
    def dense_embedder(self) -> DenseEmbedder:
        if self._dense is None:
            cls = self._provider_class("dense_embedder", self.settings.embedding.provider)
            self._dense = cls(cache=self._embeddings(), settings=self.settings)
        return self._dense

    @property
    def sparse_embedder(self) -> SparseEmbedder | None:
        """BM25/SPLADE vectors, unless hybrid search is off.

        FastEmbed regardless of ``embedding.provider``: no hosted embedding API returns
        sparse vectors, so the sparse side is always local. A deployment on OpenAI
        embeddings still gets true BM25 scoring inside Qdrant.
        """
        if self._sparse is None and self.settings.retrieval.use_sparse:
            cls = self._provider_class("sparse_embedder", "fastembed")
            self._sparse = cls(cache=self._embeddings(), settings=self.settings)
        return self._sparse

    @property
    def late_embedder(self) -> LateInteractionEmbedder | None:
        if self._late is None and self.settings.embedding.enable_late_interaction:
            cls = self._provider_class("late_interaction_embedder", "fastembed")
            self._late = cls(cache=self._embeddings(), settings=self.settings)
        return self._late

    @property
    def vector_store(self) -> Any:
        """Qdrant. The one store built without hesitation: it is the baseline index.

        The embedders are injected rather than left to the store so the ingest path and
        the query path share one loaded ONNX session instead of paying for two.
        """
        if self._vector is None:
            from ragorc.stores.qdrant.store import QdrantStore

            self._vector = QdrantStore(
                self.settings,
                dense_embedder=self.dense_embedder,
                sparse_embedder=self.sparse_embedder,
                late_embedder=self.late_embedder,
            )
        return self._vector

    async def relational_store(self) -> Any:
        """Postgres, built on demand. Async because building it is a connection.

        Shared between the text-to-SQL leg, the pgvector leg and the ingest pipeline's
        document table, so the first of them to need it pays and the rest reuse it.
        """
        if self._relational is None:
            from ragorc.stores.postgres.store import PostgresStore

            self._relational = PostgresStore(self.settings, cache=self.cache)
        return self._relational

    async def graph_store(self) -> Any:
        """Neo4j, built on demand."""
        if self._graph is None:
            from ragorc.stores.neo4j.store import Neo4jStore

            self._graph = Neo4jStore(settings=self.settings)
        return self._graph

    @property
    def semantic_cache(self) -> Any | None:
        if self._semantic_cache is None and self.settings.cache.semantic_enabled:
            from ragorc.cache.semantic import SemanticCache

            self._semantic_cache = SemanticCache(self.dense_embedder, settings=self.settings)
        return self._semantic_cache

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            self._reranker = build_reranker(llm=self.llm, settings=self.settings)
        return self._reranker

    @property
    def compressor(self) -> Any:
        if self._compressor is None:
            self._compressor = build_compressor(
                llm=self.llm, embedder=self.dense_embedder, settings=self.settings
            )
        return self._compressor

    @property
    def web_retriever(self) -> Any:
        if self._web is None:
            from ragorc.retrieve.web import make_web_retriever

            # Resolved through the factory so ``web_search_provider="none"`` costs
            # nothing: the null retriever needs no ``[web]`` extra installed.
            self._web = make_web_retriever(self.settings)
        return self._web

    @property
    def hybrid_retriever(self) -> HybridRetriever:
        """Dense + sparse (+ optional Postgres full-text) over Qdrant."""
        if self._hybrid is None:
            postgres = None
            if self.settings.retrieval.use_fulltext:
                # An explicit opt-in, so the store is built now rather than lazily: the
                # user has said Postgres is part of this deployment.
                from ragorc.stores.postgres.store import PostgresStore

                if self._relational is None:
                    self._relational = PostgresStore(self.settings, cache=self.cache)
                postgres = self._relational
            self._hybrid = HybridRetriever(
                self.vector_store, postgres=postgres, settings=self.settings
            )
        return self._hybrid

    @property
    def retriever(self) -> Any:
        """The route-driven fan-out every graph retrieves through.

        Built with all four legs registered, but only the vector leg is a real object:
        the other three are :class:`_LazyRetriever` proxies, so registering them costs
        nothing and a route that never selects them never builds them.
        """
        if self._multi_store is None:
            from ragorc.retrieve.multi_store import MultiStoreRetriever

            self._multi_store = MultiStoreRetriever(
                vector=self.hybrid_retriever,
                relational=self._lazy_relational(),
                graph=self._lazy_graph(),
                web=self.web_retriever,
                settings=self.settings,
            )
        return self._multi_store

    @property
    def crag(self) -> Any:
        """The graded-retrieval stage, wrapping the fan-out."""
        if self._crag is None:
            from ragorc.retrieve.crag import CorrectiveRAG

            self._crag = CorrectiveRAG(
                self.retriever,
                self.llm,
                self.settings,
                web=self.web_retriever,
                router=self.model_router,
                compressor=None,
            )
        return self._crag

    @property
    def self_rag(self) -> SelfRAG:
        """Owns the groundedness checker the verification nodes grade with."""
        if self._self_rag is None:
            self._self_rag = SelfRAG(self.llm, self.settings, router=self.model_router)
        return self._self_rag

    @property
    def router(self) -> Any:
        """Hybrid routing: rules first, then the LLM and the exemplar embeddings.

        Rules first because a greeting needs neither a model call nor a prompt lookup,
        and that is a measurable share of real traffic.
        """
        if self._router is None:
            from ragorc.route import build_router

            self._router = build_router(
                "hybrid", llm=self.llm, embedder=self.dense_embedder, settings=self.settings
            )
        return self._router

    @property
    def constructor(self) -> Any:
        """Self-query: split structured constraints out of the question.

        Built with no attribute schema, which makes it a deliberate no-op: with no
        schema there is nothing to filter on, and asking a model to invent field names
        produces filters that match nothing — a silent empty result set.

        To turn the stage on, describe your metadata and inject the constructor::

            from ragorc.construct.self_query import SelfQueryConstructor

            rag = await RAGPipeline.create(
                constructor=SelfQueryConstructor(llm, attributes=my_attributes)
            )

        ``constructor=`` also works on ``RAGPipeline(...)`` directly.
        """
        if self._constructor is None:
            from ragorc.construct.self_query import SelfQueryConstructor

            self._constructor = SelfQueryConstructor(
                self.llm, router=self.model_router, settings=self.settings
            )
        return self._constructor

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator(self.llm, self.settings, router=self.model_router)
        return self._generator

    # ------------------------------------------------------------------
    # Lazy store legs
    # ------------------------------------------------------------------
    def _leg(
        self, key: str, label: str, factory: Callable[[], Awaitable[Retriever]]
    ) -> _LazyRetriever:
        """Memoize one lazy leg by key.

        Memoized so the graph leg the router selects and the GraphRAG local-search leg
        share one Neo4j store rather than opening a second driver for the same database.
        """
        proxy = self._lazy_legs.get(key)
        if proxy is None:
            proxy = _LazyRetriever(key, label, factory)
            self._lazy_legs[key] = proxy
        return proxy

    def _lazy_relational(self) -> _LazyRetriever:
        async def factory() -> Retriever:
            from ragorc.retrieve.sql import SQLRetriever

            return SQLRetriever(self.llm, await self.relational_store(), settings=self.settings)

        return self._leg("sql", "postgres", factory)

    def _lazy_graph(self) -> _LazyRetriever:
        """The graph leg, whose *shape* depends on whether a GraphRAG index exists.

        With ``graph.enabled`` there are entities, relations and communities in Neo4j,
        so entity-anchored local search is the right retriever. Without it there may
        still be a graph — someone else's — and the only safe way to query an unknown
        schema is generated Cypher behind the guard. Picking text-to-Cypher when local
        search would work would spend a model call to reinvent a traversal; picking
        local search when nothing was extracted returns nothing at all.
        """

        async def factory() -> Retriever:
            store = await self.graph_store()
            if self.settings.graph.enabled:
                from ragorc.retrieve.graph import GraphLocalRetriever

                return GraphLocalRetriever(store, self.vector_store, settings=self.settings)
            from ragorc.retrieve.cypher import CypherRetriever

            return CypherRetriever(self.llm, store, settings=self.settings)

        return self._leg("graph", "neo4j", factory)

    def _graph_search_retrievers(self) -> dict[str, Retriever]:
        """The three GraphRAG modes, each behind its own lazy proxy."""

        async def local() -> Retriever:
            from ragorc.retrieve.graph import GraphLocalRetriever

            return GraphLocalRetriever(
                await self.graph_store(), self.vector_store, settings=self.settings
            )

        async def global_() -> Retriever:
            from ragorc.retrieve.graph import GraphGlobalRetriever

            return GraphGlobalRetriever(
                self.llm, await self.graph_store(), router=self.model_router, settings=self.settings
            )

        async def drift() -> Retriever:
            from ragorc.retrieve.graph import GraphDriftRetriever

            return GraphDriftRetriever(
                self.vector_store,
                await self.graph_store(),
                self.vector_store,
                settings=self.settings,
            )

        return {
            "local": self._leg("graph_local", "neo4j", local),
            "global": self._leg("graph_global", "neo4j", global_),
            "drift": self._leg("graph_drift", "neo4j", drift),
        }

    def _lazy_bridge(self) -> _LazyRetriever:
        async def factory() -> Retriever:
            from ragorc.retrieve.multihop import BridgeEntityRetriever

            return BridgeEntityRetriever(
                await self.graph_store(), self.vector_store, settings=self.settings
            )

        return self._leg("bridge", "neo4j", factory)

    def _lazy_multihop(self) -> _LazyRetriever:
        async def factory() -> Retriever:
            from ragorc.retrieve.multihop import MultiHopRetriever

            return MultiHopRetriever(
                self.llm,
                self.retriever,
                await self.graph_store(),
                self.vector_store,
                router=self.model_router,
                settings=self.settings,
            )

        return self._leg("multihop", "neo4j", factory)

    # ------------------------------------------------------------------
    # Nodes and graphs
    # ------------------------------------------------------------------
    def nodes(self, *, full_translation: bool = False) -> PipelineNodes:
        """The component bundle the graphs are compiled against.

        Two bundles at most, differing only in the translator: the agentic pipeline runs
        every translation strategy, and every other pipeline runs one. Two bundles rather
        than a per-graph translator setting because the choice is a property of *how much
        depth the pipeline is buying*, not of the deployment.
        """
        bundle = self._node_bundles.get(full_translation)
        if bundle is not None:
            return bundle
        names = _FULL_TRANSLATORS if full_translation else self._translator_names
        bundle = PipelineNodes(
            llm=self.llm,
            generator=self.generator,
            retriever=self.retriever,
            settings=self.settings,
            store_retrievers={
                DataStore.VECTOR: self.hybrid_retriever,
                DataStore.RELATIONAL: self._lazy_relational(),
                DataStore.GRAPH: self._lazy_graph(),
                DataStore.WEB: self.web_retriever,
            },
            translator=build_translators(names, self.llm, self.settings),
            router=self.router,
            constructor=self.constructor,
            reranker=self.reranker,
            compressor=self.compressor,
            crag=self.crag,
            self_rag=self.self_rag,
            web=self.web_retriever,
            graph_retrievers=self._graph_search_retrievers(),
            bridge_retriever=self._lazy_bridge(),
            multihop=self._lazy_multihop(),
            model_router=self.model_router,
        )
        self._node_bundles[full_translation] = bundle
        return bundle

    def graph(self, name: str) -> Any:
        """Compile one pipeline by name, memoized.

        Per-name rather than all-at-once so a ``naive`` query never pays to compile the
        agentic graph — and so a deployment that only uses one pipeline only ever builds
        the components that pipeline's nodes touch.
        """
        compiled = self._compiled.get(name)
        if compiled is None:
            if name not in GRAPHS:
                raise ConfigError(f"unknown pipeline {name!r}", available=graph_names())
            compiled = build_graph(
                name, self.nodes(full_translation=name == "agentic"), settings=self.settings
            )
            self._compiled[name] = compiled
        return compiled

    @property
    def graphs(self) -> Mapping[str, Any]:
        """Every compiled pipeline, by name.

        Compiles whatever has not been compiled yet, so this is the inspection surface
        rather than the request path — ``rag.graphs["crag"].get_graph().draw_mermaid()``
        prints the control flow as a diagram, which is the point of putting it in a
        graph engine at all.
        """
        for name in GRAPHS:
            self.graph(name)
        return dict(self._compiled)

    def select_graph(self, pipeline: str = "auto") -> str:
        """Resolve ``pipeline`` to a graph name, logging the choice and why.

        The resolution order for ``"auto"``, in full:

        =====================================  ==========  ============================
         flags                                  graph       reasoning
        =====================================  ==========  ============================
         crag + self_rag                        agentic     both loops asked for
         graph + (crag or self_rag)             agentic     a graph *and* a loop
         graph + detect_communities             graphrag    global search is available
         graph + multihop                       multihop    no summaries: traversal only
         graph                                  graphrag    local and DRIFT still work
         crag                                   crag        document-side loop only
         self_rag                               self_rag    answer-side loop only
         none of the above                      adaptive    route and fan out
        =====================================  ==========  ============================

        ``graph.multihop_enabled`` never selects a pipeline on its own. It defaults to
        *on*, so letting it decide would silently change the baseline for every
        deployment that never asked for multi-hop; it earns a graph only when a graph
        exists and community summaries do not, which is exactly the case where traversal
        is all the graph can offer.
        """
        if pipeline and pipeline != "auto":
            if pipeline not in GRAPHS:
                raise ConfigError(
                    f"unknown pipeline {pipeline!r}", available=graph_names(), requested=pipeline
                )
            return pipeline

        s = self.settings
        crag = s.retrieval.crag_enabled
        self_rag = s.generation.self_rag_enabled
        graph_on = s.graph.enabled

        if crag and self_rag:
            name, reason = "agentic", "crag and self_rag both enabled"
        elif graph_on and (crag or self_rag):
            name, reason = "agentic", "graph enabled alongside a feedback loop"
        elif graph_on and s.graph.detect_communities:
            name, reason = "graphrag", "graph enabled with community summaries"
        elif graph_on and s.graph.multihop_enabled:
            name, reason = "multihop", "graph enabled without community detection"
        elif graph_on:
            name, reason = "graphrag", "graph enabled"
        elif crag:
            name, reason = "crag", "crag_enabled"
        elif self_rag:
            name, reason = "self_rag", "self_rag_enabled"
        else:
            name, reason = "adaptive", "no feedback loop enabled"

        log.info(
            "pipeline_selected",
            pipeline=name,
            reason=reason,
            crag=crag,
            self_rag=self_rag,
            graph=graph_on,
            multihop=s.graph.multihop_enabled,
        )
        return name

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    @property
    def ingest_pipeline(self) -> Any:
        """The ingest orchestrator, sharing this pipeline's embedders and stores.

        Sharing is the point rather than a saving: the query path and the ingest path
        must embed in the *same space*, and two independently constructed embedders with
        different asymmetric prefixes silently index one space and search another.
        """
        if self._ingest is None:
            from ragorc.index.pipeline import IngestPipeline

            self._ingest = IngestPipeline(
                vector_store=self.vector_store,
                relational_store=self._relational,
                dense_embedder=self.dense_embedder,
                sparse_embedder=self.sparse_embedder,
                late_embedder=self.late_embedder,
                llm=self.llm,
                settings=self.settings,
            )
        return self._ingest

    async def ingest(
        self,
        source: Any = None,
        *,
        documents: Sequence[Document] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Index documents or a source, and report what happened.

        ``source`` is a path, a URL, a :class:`~ragorc.core.models.Document`, or an
        iterable mixing them; ``documents=`` is the explicit spelling for objects you
        built yourself. Everything real happens in
        :class:`~ragorc.index.pipeline.IngestPipeline` — the checksum skip, the streamed
        batches, the failure policy — and this method exists so a user does not have to
        assemble it.

        ``**kwargs`` are settings overrides applied to *this run only*
        (``chunking_strategy=...``, ``indexing__batch_size=256``). They rebuild the
        ingest pipeline, since the strategy decides which embedder is the query-side one
        and therefore what the collection's vector dimension has to be.
        """
        self._ensure_open()
        target = list(documents) if documents is not None else source
        if target is None:
            raise ConfigError("ingest needs a source or documents=", hint="rag.ingest('./docs')")
        if kwargs:
            self.settings = _resolve_settings(self.settings, self._normalize_ingest_kwargs(kwargs))
            self._ingest = None
        return await self.ingest_pipeline.ingest(target)

    @staticmethod
    def _normalize_ingest_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Let ``ingest(chunk_size=256)`` mean ``indexing__chunk_size=256``.

        Unprefixed keys are resolved against the indexing section because that is the
        only section an ingest call could plausibly mean; anything already carrying a
        section prefix is passed through untouched.
        """
        return {key if "__" in key else f"indexing__{key}": value for key, value in kwargs.items()}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    async def query(
        self,
        question: str,
        *,
        tenant_id: str | None = None,
        top_k: int | None = None,
        pipeline: str = "auto",
        stream: bool = False,
    ) -> Answer:
        """Answer a question. The one method most callers ever use.

        Wrapped in :func:`~ragorc.core.telemetry.new_request_context`, so the returned
        answer always carries a complete per-stage trace and an itemized bill, and the
        cost ceilings are enforced *before* each model call rather than discovered
        afterwards.

        ``stream=True`` is accepted for interface symmetry with the HTTP layer, which
        flips one flag rather than calling a different method. It delegates to
        :meth:`stream`, joins the deltas, and marks the result ``verified=False``:
        groundedness can only be judged once an answer is complete, and an answer that
        was streamed was already sent. Never returns a *claim* of verification it did
        not perform.
        """
        self._ensure_open()
        if stream:
            return await self._collect_stream(
                question, tenant_id=tenant_id, top_k=top_k, pipeline=pipeline
            )

        tenant = tenant_id or self.settings.tenant_id
        request_id = uuid.uuid4().hex[:16]
        cost = self.settings.cost
        with new_request_context(
            request_id=request_id,
            max_cost_usd=cost.max_cost_per_query_usd if cost.track_costs else None,
            max_calls=cost.max_llm_calls_per_query,
            max_tokens=cost.max_tokens_per_query,
            trace=self.settings.observability.trace_enabled,
        ) as (trace, ledger):
            await self._rate_limit(tenant)
            self._audit.query(tenant_id=tenant, principal=None, length=len(question))

            hit = await self._cache_get(question, tenant, top_k)
            if hit is not None:
                return hit

            name = self.select_graph(pipeline)
            spec = GRAPHS[name]
            final: RAGState = await self.graph(name).ainvoke(
                initial_state(question, tenant_id=tenant, top_k=top_k, pipeline=name),
                # The recursion limit is the safety net above each graph's own iteration
                # counters; it is derived per graph so raising a retry budget cannot turn
                # the net into the primary bound.
                config={"recursion_limit": spec.recursion_limit(self.settings)},
            )
            answer = self._finish(
                final, name=name, request_id=request_id, trace=trace, ledger=ledger
            )
            await self._cache_set(question, tenant, answer, final, top_k)
            return answer

    async def stream(
        self,
        question: str,
        *,
        tenant_id: str | None = None,
        top_k: int | None = None,
        pipeline: str = "auto",
    ) -> AsyncIterator[str]:
        """Stream the answer's tokens as they are produced.

        Streaming does not go through a compiled graph, and that is a deliberate
        limitation rather than an oversight. The retrieval-side stages are driven here by
        calling the very same nodes the graphs are built from — which is the payoff of
        every node being an independently callable coroutine — and then the generator
        streams. What cannot be included is the *answer-side* loop: Self-RAG regenerates
        an answer it judged inadequate, and you cannot un-emit a token you have already
        sent. CRAG still applies, because it is retrieval-side and runs before the first
        token.

        So a streamed answer is unverified by construction. Use :meth:`query` when the
        groundedness score, the citation validation or the abstention gates matter, and
        this when latency to first token does.

        The request context is installed for the *lifetime of the generator*, which is
        what keeps the streamed call inside the cost ledger and the budget ceilings. The
        alternative — an uncosted stream — is worse in a library whose central claim is
        that every model call is accounted for.
        """
        self._ensure_open()
        tenant = tenant_id or self.settings.tenant_id
        request_id = uuid.uuid4().hex[:16]
        cost = self.settings.cost
        with new_request_context(
            request_id=request_id,
            max_cost_usd=cost.max_cost_per_query_usd if cost.track_costs else None,
            max_calls=cost.max_llm_calls_per_query,
            max_tokens=cost.max_tokens_per_query,
            trace=self.settings.observability.trace_enabled,
        ):
            await self._rate_limit(tenant)
            self._audit.query(tenant_id=tenant, principal=None, length=len(question))
            state, nodes = await self._retrieve_for_stream(
                question, tenant=tenant, top_k=top_k, pipeline=pipeline
            )
            query = state.get("query")
            if query is None:  # pragma: no cover - validate raises before this
                return
            retrieval = RetrievalResult(chunks=evidence(state))
            async for delta in self.generator.stream(
                query,
                retrieval,
                route=state.get("route"),
                prompt_name=nodes.prompt_for(state),
            ):
                yield delta

    async def _retrieve_for_stream(
        self, question: str, *, tenant: str | None, top_k: int | None, pipeline: str
    ) -> tuple[RAGState, PipelineNodes]:
        """Run the retrieval-side nodes by hand, in the order the graph would.

        Graded retrieval is used when CRAG is enabled, because that decision happens
        before the first token and is therefore compatible with streaming; the
        answer-side loops are not, and are skipped rather than half-applied.
        """
        name = self.select_graph(pipeline)
        nodes = self.nodes(full_translation=name == "agentic")
        state: RAGState = initial_state(question, tenant_id=tenant, top_k=top_k, pipeline=name)
        steps = [nodes.validate, nodes.translate, nodes.route]
        steps.append(nodes.grade if self.settings.retrieval.crag_enabled else nodes.retrieve)
        steps.append(nodes.rerank)
        for step in steps:
            _merge(state, await step(state))
        return state, nodes

    async def _collect_stream(
        self, question: str, *, tenant_id: str | None, top_k: int | None, pipeline: str
    ) -> Answer:
        parts = [
            delta
            async for delta in self.stream(
                question, tenant_id=tenant_id, top_k=top_k, pipeline=pipeline
            )
        ]
        text = "".join(parts).strip()
        return Answer(
            text=text,
            grounded=False,
            groundedness=0.0,
            confidence=0.0 if not text else 1.0,
            metadata={"streamed": True, "verified": False},
        )

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------
    def _finish(
        self,
        state: RAGState,
        *,
        name: str,
        request_id: str,
        trace: list[Any],
        ledger: Any,
    ) -> Answer:
        """Attach the request-level facts the graph could not know.

        The bill comes from the **ledger**, not from summing the state's usages: the
        ledger is written by the LLM client itself on every call, so it includes calls a
        node forgot to report and never double-counts one that reported twice. The
        state's usage list is per-call attribution for the trace; the ledger is the
        total.

        The one exception is a ledger that recorded nothing while nodes did report
        usage — which means the injected LLM satisfies the protocol without writing to
        the ledger, as a third-party client legitimately might. Then the nodes' own
        accounting is used instead. The two are never *added*: they describe the same
        calls, so summing them would double the reported bill.
        """
        answer = state.get("answer")
        if answer is None:
            # No node produced an answer — every branch failed, or the graph ended early.
            # An abstention with the recorded errors is the only honest result, and it
            # keeps the promise that ``query`` returns an Answer.
            answer = Answer(
                text=self.settings.generation.abstain_message,
                grounded=False,
                groundedness=0.0,
                confidence=0.0,
                abstained=True,
                abstain_reason="the pipeline produced no answer",
                chunks=evidence(state),
            )
        billed = ledger.total
        answer.usage = billed if billed.calls else total_usage(state)
        answer.trace = list(trace)
        answer.metadata = {
            **answer.metadata,
            "pipeline": name,
            "request_id": request_id,
            "cost": ledger.report(),
            "errors": list(state.get("errors") or ()),
            "warnings": list(state.get("warnings") or ()),
            "rewrites": list(state.get("rewrites") or ()),
            "tools_used": list(state.get("tools_used") or ()),
            "retrieve_iterations": int(state.get("retrieve_iterations") or 0),
            "generate_iterations": int(state.get("generate_iterations") or 0),
            "search_mode": state.get("search_mode"),
        }
        self._audit.answered(
            tenant_id=state.get("tenant_id"),
            cost_usd=answer.usage.cost_usd,
            chunks=len(answer.chunks),
            grounded=answer.grounded,
        )
        log.info(
            "query_answered",
            pipeline=name,
            request_id=request_id,
            abstained=answer.abstained,
            grounded=answer.grounded,
            groundedness=round(answer.groundedness, 3),
            chunks=len(answer.chunks),
            calls=answer.usage.calls,
            cost_usd=round(answer.usage.cost_usd, 6),
            errors=len(state.get("errors") or ()),
        )
        slow_ms = self.settings.observability.slow_query_ms
        elapsed_ms = sum(step.duration_ms for step in current_trace())
        if slow_ms and elapsed_ms > slow_ms:
            # A separate line at WARNING, because that is what an operator alerts
            # on. The threshold was configurable and compared against nothing, so
            # a deployment could not say what "slow" meant for it.
            log.warning(
                "query_slow",
                request_id=request_id,
                duration_ms=round(elapsed_ms, 1),
                threshold_ms=slow_ms,
                pipeline=name,
                calls=answer.usage.calls,
            )
        return answer

    # ------------------------------------------------------------------
    # Semantic cache
    # ------------------------------------------------------------------
    async def _cache_get(
        self, question: str, tenant: str | None, top_k: int | None = None
    ) -> Answer | None:
        cache = self.semantic_cache
        if cache is None:
            return None
        # `top_k` is part of the identity: it changes how much evidence the
        # answer was built from. This path takes no filters, so the scope is
        # top_k alone; the HTTP path passes both.
        hit = await cache.get(question, tenant_id=tenant, scope=scope_key(None, top_k))
        if hit is None:
            return None
        answer = _answer_from_payload(hit.answer)
        answer.metadata = {
            **answer.metadata,
            "cache": {
                "tier": "semantic",
                "score": round(hit.score, 4),
                "cached_question": hit.question,
            },
        }
        log.info("semantic_cache_served", score=round(hit.score, 4))
        return answer

    async def _cache_set(
        self,
        question: str,
        tenant: str | None,
        answer: Answer,
        state: RAGState,
        top_k: int | None = None,
    ) -> None:
        """Populate the cache, with two refusals.

        Abstentions are refused by :class:`~ragorc.cache.semantic.SemanticCache` itself:
        they are statements about the index at one moment, and serving one later hides
        content that has since been added.

        Degraded answers are refused here. An answer produced while a store was
        unreachable is not the answer this question has — it is the answer it had during
        an outage — and caching it would serve the outage for the whole TTL, long after
        the store came back.
        """
        cache = self.semantic_cache
        if cache is None:
            return
        if state.get("errors"):
            log.info("semantic_cache_skipped", reason="degraded_answer")
            return
        await cache.set(
            question,
            _answer_to_payload(answer),
            tenant_id=tenant,
            scope=scope_key(None, top_k),
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """The resolved configuration and which features are actually on.

        Reports what is *built* rather than what is *configured* where the two differ,
        because "Postgres is configured" and "Postgres has been connected to" are
        different facts and the second one is the one that answers "why is my SQL leg
        empty?".
        """
        s = self.settings
        return {
            "settings": s.summary(),
            "features": {
                "pipeline_auto": self.select_graph("auto"),
                "hybrid": s.retrieval.hybrid_enabled,
                "sparse": s.retrieval.use_sparse,
                "fulltext": s.retrieval.use_fulltext,
                "late_interaction": s.embedding.enable_late_interaction,
                "rerank": s.retrieval.rerank_enabled and s.retrieval.reranker,
                "compression": s.retrieval.compression_enabled and s.retrieval.compressor,
                "crag": s.retrieval.crag_enabled,
                "web_fallback": s.retrieval.crag_web_fallback and s.retrieval.web_search_provider,
                "self_rag": s.generation.self_rag_enabled,
                "rrr": s.generation.rrr_enabled,
                "graphrag": s.graph.enabled,
                "multihop": s.graph.multihop_enabled,
                "raptor": s.indexing.raptor_enabled,
                "chunking": s.indexing.chunking_strategy.value,
                "groundedness": s.generation.check_groundedness
                and s.generation.groundedness_method,
                "abstention": s.generation.allow_abstention,
                # Read from settings rather than from ``self.semantic_cache``: touching
                # that property would *construct* the cache and its embedder, and a
                # describe() call that opens a Qdrant collection is a surprising
                # description.
                "semantic_cache": s.cache.enabled and s.cache.semantic_enabled,
                "tenant_isolation": s.security.enforce_tenant_isolation,
            },
            "pipelines": {name: spec.summary for name, spec in GRAPHS.items()},
            "models": self.model_router.describe(),
            "budgets": {
                "max_cost_per_query_usd": s.cost.max_cost_per_query_usd,
                "max_llm_calls_per_query": s.cost.max_llm_calls_per_query,
                "max_tokens_per_query": s.cost.max_tokens_per_query,
                "recursion_limits": {
                    name: spec.recursion_limit(s) for name, spec in GRAPHS.items()
                },
            },
            "components": {
                "translators": list(self._translator_names),
                "reranker": getattr(self._reranker, "name", None),
                "compressor": getattr(self._compressor, "name", None),
                "embedder": {
                    "provider": s.embedding.provider,
                    "model": s.embedding.dense_model,
                    "dimension": getattr(self._dense, "dimension", None),
                },
            },
            "stores": {
                "vector": {"backend": "qdrant", "built": self._vector is not None},
                "relational": {"backend": "postgres", "built": self._relational is not None},
                "graph": {"backend": "neo4j", "built": self._graph is not None},
                "lazy_legs": {key: proxy.built for key, proxy in self._lazy_legs.items()},
            },
            "compiled_graphs": sorted(self._compiled),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Close every store and client this pipeline created.

        Idempotent, and every close is suppressed independently: a driver that fails to
        shut down cleanly must not prevent the other three from releasing their sockets,
        because the caller is on their way out and a leaked connection pool outlives the
        process that could have reported it.
        """
        if self._closed:
            return
        self._closed = True
        closers: list[Any] = [self._vector, self._relational, self._graph]
        for proxy in self._lazy_legs.values():
            inner = getattr(proxy, "_inner", None)
            store = getattr(inner, "store", None)
            if store is not None and store not in closers:
                closers.append(store)
        if self._ingest is not None:
            closers.append(self._ingest)
        # The embedders and the reranker were missing from this list, so a
        # pipeline on a hosted embedding provider leaked one connection pool per
        # instance — invisible in a script that exits, fatal in a long-lived
        # service that rebuilds pipelines per tenant.
        closers.extend([self._dense, self._sparse, self._late, self._reranker])
        closers.append(self._cache)
        closers.append(self._llm)
        for component in closers:
            if component is None:
                continue
            closer = getattr(component, "aclose", None) or getattr(component, "close", None)
            if closer is None:
                continue
            with contextlib.suppress(Exception):
                await closer()
        log.info("pipeline_closed")

    async def __aenter__(self) -> RAGPipeline:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise ConfigError(
                "this RAGPipeline has been closed", hint="build a new one with build_pipeline()"
            )

    def _embeddings(self) -> Any:
        """One embedding cache shared by every embedder this pipeline builds.

        Shared deliberately: the dense, sparse and late-interaction embedders key their
        entries by content hash *and* model name, so one backend holds all three without
        collisions and a query embedded once is embedded once.
        """
        if self._embedding_cache is None:
            from ragorc.embed.cache import EmbeddingCache

            self._embedding_cache = EmbeddingCache(self.cache, self.settings)
        return self._embedding_cache

    @staticmethod
    def _provider_class(kind: str, provider: str) -> type:
        """Resolve an embedder class, importing its provider module first.

        The registry only knows a class once the module defining it has been imported,
        and importing every provider eagerly would require every hosted SDK to be
        installed. Deriving the module name from the provider means a missing extra
        fails with that provider's own ``ImportError`` and its own pip hint.
        """
        importlib.import_module(f"ragorc.embed.{provider}_provider")
        return resolve(kind, provider)

    async def _rate_limit(self, tenant: str | None) -> None:
        """Per-tenant admission control, before anything expensive runs.

        Keyed by tenant rather than globally: one tenant's burst must not throttle
        another's. An unauthenticated caller shares the ``anonymous`` bucket, which is
        the conservative reading of "we do not know who this is".
        """
        if not self.settings.security.enable_rate_limit:
            return
        await self._ratelimiter.check(tenant or "anonymous")


def _merge(state: RAGState, partial: Mapping[str, Any]) -> None:
    """Apply a node's partial state by hand, honouring the accumulating keys.

    Used only by the streaming path, which drives the nodes directly and therefore has
    no LangGraph channels to apply reducers for it. Kept explicit and tiny rather than
    reimplementing the reducer machinery: the streaming path is linear, so the only
    reducer semantics it needs is "these five keys append".
    """
    # Every write below is by a *runtime* key, which a TypedDict cannot express:
    # ``state[key]`` is only checkable when ``key`` is a literal. The cast states
    # the thing that is already true — ``RAGState`` is a plain dict — and keeps the
    # checker live on the interesting part, the reducer semantics.
    target = cast(dict[str, Any], state)
    for key, value in partial.items():
        if key in _ACCUMULATORS and isinstance(value, list):
            previous = target.get(key) or []
            target[key] = [*previous, *value]
        elif key == "per_store" and isinstance(value, dict):
            merged = dict(state.get("per_store") or {})
            merged.update(value)
            state["per_store"] = merged
        else:
            target[key] = value


def _answer_to_payload(answer: Answer) -> dict[str, Any]:
    """The cacheable projection of an answer.

    The evidence is deliberately **not** cached. Chunk bodies would multiply the cache
    entry by the size of the context window, and — the real reason — a cached
    :class:`ScoredChunk` is a snapshot of a document that may since have been edited or
    deleted. Served as a citation, it would attribute the answer to text that no longer
    exists, which is a fabricated citation produced by our own cache. Citation metadata
    is kept because it names ids and quotes that a caller can re-resolve against the
    live store.
    """
    return {
        "text": answer.text,
        "grounded": answer.grounded,
        "groundedness": answer.groundedness,
        "confidence": answer.confidence,
        "abstained": answer.abstained,
        "abstain_reason": answer.abstain_reason,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "quote": c.quote,
                "claim": c.claim,
                "support": c.support,
                "source": c.source,
            }
            for c in answer.citations
        ],
        "metadata": _json_safe(answer.metadata),
    }


def _answer_from_payload(payload: Mapping[str, Any]) -> Answer:
    return Answer(
        text=str(payload.get("text", "")),
        citations=[Citation(**c) for c in payload.get("citations") or ()],
        grounded=bool(payload.get("grounded", True)),
        groundedness=float(payload.get("groundedness", 1.0)),
        confidence=float(payload.get("confidence", 1.0)),
        abstained=bool(payload.get("abstained", False)),
        abstain_reason=payload.get("abstain_reason"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _json_safe(value: Any) -> Any:
    """Drop anything the cache cannot serialize instead of failing the write.

    Answer metadata carries whatever the stages put there, including the occasional
    numpy scalar or dataclass. A cache write is an optimization, so an unserializable
    field must degrade to a missing field rather than to an exception on the request
    path.
    """
    try:
        return orjson.loads(orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY))
    except (TypeError, ValueError):
        return {k: str(v)[:500] for k, v in dict(value).items()} if isinstance(value, dict) else {}


async def build_pipeline(settings: Settings | None = None, **overrides: Any) -> RAGPipeline:
    """Build a ready pipeline. The one-line entry point.

        rag = await build_pipeline(retrieval__crag_enabled=True)

    Equivalent to :meth:`RAGPipeline.create`; it exists because a module-level function
    is what a reader reaches for first, and because ``from ragorc import build_pipeline``
    keeps the import in the root package lazy.
    """
    return await RAGPipeline.create(settings, **overrides)


#: Resolved once, after the class body, because the signature is read from it.
_COMPONENT_PARAMS: frozenset[str] = _component_params()
