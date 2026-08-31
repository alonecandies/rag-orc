"""The three-way fan-out: query the routed stores at once, then fuse.

This is the retriever the whole architecture is shaped around. Vector, relational
and graph are not three interchangeable indexes — they answer three different
kinds of question, and the router decides which of them a question needs. This
module is what turns that decision into concurrent I/O and one ranked list.

Why concurrency is the entire point
-----------------------------------
Serial fan-out makes a query's latency the *sum* of its backends: a 120 ms Qdrant
search, a 400 ms generated SQL statement and a 250 ms graph traversal is 770 ms of
waiting. Run together it is 400 ms — the slowest one — and adding a fourth store
costs nothing unless it is slower than the current worst. That is the difference
between a three-store architecture being a feature and being a tax.

Why every store gets its own deadline *and* its own circuit breaker
-------------------------------------------------------------------
They solve different failures and neither substitutes for the other.

A **deadline** (``retrieval.per_store_timeout_s``) bounds one request. Without it
the request's latency is defined by its worst backend, so p99 becomes
p99-of-the-slowest-store and one degraded dependency drags down every query,
including the ones two healthy stores had already answered. A store whose retriever
already applies that deadline to each of *its* own legs is the one exception, and
:meth:`MultiStoreRetriever._guarded` explains why a second timer on the same budget
recreates the failure this one prevents.

A **circuit breaker** bounds the *next thousand* requests. A deadline alone means
every single request still pays the full timeout for a store that is down —
10 seconds each, forever. The breaker trips after a few consecutive failures and
then fails instantly, so a dead backend costs the first few requests their latency
and the rest nothing, with a single half-open probe after the cooldown to notice
recovery automatically.

What counts as a breaker failure is deliberately narrow: an outage or a timeout,
never a rejected generated query. A text-to-SQL statement the guard refuses proves
Postgres is *alive*; counting it would open the relational breaker and take out
pgvector search over a prompt-quality problem. The stores themselves draw the same
line internally, and this layer must not undo it.

Degrading is explicit, and visible
----------------------------------
A store that fails does not fail the query. It is recorded — by name — in
:attr:`~ragorc.core.models.RetrievalResult.errors`, its latency is recorded in
``timings_ms``, its (empty) contribution is recorded in ``per_store``, and the
query continues with whatever the other stores returned. Every degradation path
produces an entry, including the ones that are not exceptions at all: a store the
route selected but nobody configured, and a store whose breaker is already open,
both land in ``errors`` rather than disappearing. That is what makes the behaviour
testable — assert on ``errors`` and ``per_store``, not on a log line.

The one thing that does *not* degrade is a
:class:`~ragorc.core.errors.GuardrailViolation` or a
:class:`~ragorc.core.errors.BudgetExceeded` raised by a leg. Those are decisions,
not outages, and continuing past a security guard is how a guard becomes a data
leak. :func:`~ragorc.retrieve.ensemble.run_leg` already enforces that distinction,
which is why leg execution is borrowed from there rather than reimplemented.

Why fusion is rank-based by default
-----------------------------------
The lists arriving here have no common scale — cosine similarity, ``ts_rank_cd``
cover density, a graph blend, and (from the SQL and Cypher legs) a *constant*.
:mod:`ragorc.retrieve.sql` argues the constant case in full; the short version is
that a constant has no magnitude relative to a distribution, so any score-based
combiner either always prefers the structured answer or never surfaces it,
depending on the query. RRF reads only position, so it is the only merge whose
behaviour does not depend on a number that carries no information. Score-based
methods stay configurable through ``retrieval.fusion`` because they are the right
choice when every routed store happens to be a vector store.

No per-request state is kept on the instance: one retriever object serves
concurrent requests, and per-request state on ``self`` is a data race that produces
plausible wrong answers rather than a crash. Callers who want the diagnostics call
:meth:`MultiStoreRetriever.retrieve_detailed`, which returns them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Mapping, Sequence
from typing import Any

import structlog

from ragorc.core.concurrency import CircuitBreaker
from ragorc.core.errors import StoreUnavailable
from ragorc.core.models import (
    DataStore,
    FusionMethod,
    Query,
    RetrievalResult,
    RouteDecision,
    ScoredChunk,
)
from ragorc.core.protocols import Retriever
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.retrieve.ensemble import LegResult, run_legs
from ragorc.retrieve.fusion import Weights, fuse
from ragorc.retrieve.noise import NoiseFilter

log = structlog.get_logger(__name__)

__all__ = ["MultiStoreRetriever"]

_BREAKER_THRESHOLD = 4
"""Consecutive failures before a store's breaker opens.

Low, because the cost of being wrong is small in both directions: a false trip
costs one store's contribution for ``reset_timeout_s`` while the half-open probe
is already scheduled, and a late trip costs every request the full per-store
deadline. Four is enough that a single transient blip does not open it."""

_BREAKER_RESET_S = 30.0
"""Cooldown before a half-open probe. Long enough for a rolling restart or a
leader election to finish, short enough that a recovered store rejoins within one
user's session rather than one deploy cycle."""


def _as_store(value: DataStore | str) -> DataStore:
    return value if isinstance(value, DataStore) else DataStore(value)


@register("retriever", "multi_store")
class MultiStoreRetriever:
    """Route-driven concurrent fan-out over the configured datastores."""

    name = "multi_store"

    def __init__(
        self,
        retrievers: Mapping[DataStore | str, Retriever] | None = None,
        *,
        vector: Retriever | None = None,
        relational: Retriever | None = None,
        graph: Retriever | None = None,
        web: Retriever | None = None,
        method: FusionMethod | str | None = None,
        weights: Weights = None,
        noise: NoiseFilter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        resolved: dict[DataStore, Retriever] = {
            _as_store(key): value for key, value in (retrievers or {}).items()
        }
        # The named arguments are the ergonomic path (this is how the pipeline
        # wires it); the mapping is the general one. An explicit keyword overrides
        # the mapping's entry for the same store.
        named: dict[DataStore, Retriever | None] = {
            DataStore.VECTOR: vector,
            DataStore.RELATIONAL: relational,
            DataStore.GRAPH: graph,
            DataStore.WEB: web,
        }
        resolved.update({s: r for s, r in named.items() if r is not None})
        self.retrievers = resolved
        self.method = FusionMethod(method) if method is not None else self.settings.retrieval.fusion
        self.weights: Weights = (
            weights if weights is not None else self.settings.retrieval.fusion_weights
        )
        self.noise = noise or NoiseFilter(self.settings)
        # One breaker per store, built once and *kept*: a breaker rebuilt per
        # request has no memory, which is the one property it exists to have.
        self.breakers: dict[DataStore, CircuitBreaker] = {
            store: CircuitBreaker(
                name=f"store:{store.value}",
                failure_threshold=_BREAKER_THRESHOLD,
                reset_timeout_s=_BREAKER_RESET_S,
            )
            for store in DataStore
        }

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """:class:`~ragorc.core.protocols.Retriever` entry point.

        Pass ``route=`` a :class:`~ragorc.core.models.RouteDecision` to restrict the
        fan-out; without one every configured store is queried, which is the right
        default for a caller that has not run a router.
        """
        result = await self.retrieve_detailed(
            query, route=kwargs.pop("route", None), top_k=top_k, **kwargs
        )
        return result.chunks

    async def retrieve_detailed(
        self,
        query: Query,
        *,
        route: RouteDecision | None = None,
        top_k: int | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Fan out, fuse, and report per-store diagnostics.

        The full :class:`~ragorc.core.models.RetrievalResult` is the point of this
        method: once fusion has flattened several scores into one number, "which
        store contributed this, how long did it take, and what failed" is
        unrecoverable after the fact.
        """
        rs = self.settings.retrieval
        limit = int(top_k or query.top_k or rs.top_k)
        # Each leg fetches ``fetch_k``, not ``top_k``. Recall is set here and
        # precision downstream: fusion and reranking can reorder what a leg
        # returned, and cannot recover what it never fetched.
        fetch_k = max(int(rs.fetch_k), limit)

        result = RetrievalResult()
        stores = self._plan(route, result)
        if not stores:
            log.info(
                "multi_store_no_stores", route=[s.value for s in (route.stores if route else ())]
            )
            return result

        passthrough = {k: v for k, v in kwargs.items() if k not in ("route", "top_k")}
        jobs: dict[str, Coroutine[Any, Any, list[ScoredChunk]]] = {
            store.value: self._guarded(store, query, fetch_k, passthrough, result)
            for store in stores
        }

        started = time.perf_counter()
        legs = await run_legs(
            jobs,
            # The deadline is applied inside ``_guarded`` so the breaker can
            # observe a timeout as a failure; a second one here would only be able
            # to cancel the first.
            timeout_s=None,
            limit=max(1, rs.max_concurrent_retrievers),
            label="multi_store",
        )
        self._record(legs, result)

        fuse_started = time.perf_counter()
        result.chunks = self._fuse(query, legs, limit)
        result.timings_ms["fuse"] = (time.perf_counter() - fuse_started) * 1000.0
        result.timings_ms["total"] = (time.perf_counter() - started) * 1000.0

        log.info(
            "multi_store_retrieved",
            stores=[s.value for s in stores],
            per_store={name: len(chunks) for name, chunks in result.per_store.items()},
            errors=result.errors,
            candidates=result.total_candidates,
            returned=len(result.chunks),
            fusion=self.method.value,
            timings_ms={k: round(v, 1) for k, v in result.timings_ms.items()},
        )
        return result

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _plan(self, route: RouteDecision | None, result: RetrievalResult) -> list[DataStore]:
        """Decide which stores to query, recording every store that is skipped.

        Three skip reasons, all of which land in ``errors`` rather than vanishing:
        the route selected a store nobody wired up, the route selected a store
        whose breaker is open, and ``DataStore.NONE`` — which is not a failure but a
        routing decision that this question needs no retrieval at all, and is
        therefore logged rather than recorded as an error.
        """
        selected = list(route.stores) if route is not None else list(self.retrievers)
        planned: list[DataStore] = []
        seen: set[DataStore] = set()
        for raw in selected:
            store = _as_store(raw)
            if store in seen:
                continue
            seen.add(store)
            if store is DataStore.NONE:
                log.info("multi_store_route_none", reasoning=route.reasoning if route else None)
                continue
            retriever = self.retrievers.get(store)
            if retriever is None:
                # Visible on purpose. A route that asks for the graph on a
                # deployment with no graph retriever is a configuration error, and
                # silently answering from two stores hides it indefinitely.
                result.errors[store.value] = "no retriever configured for this store"
                result.per_store.setdefault(store.value, [])
                continue
            try:
                self.breakers[store].check()
            except StoreUnavailable as exc:
                result.errors[store.value] = str(exc)
                result.per_store.setdefault(store.value, [])
                result.timings_ms[store.value] = 0.0
                log.warning("multi_store_circuit_open", store=store.value)
                continue
            planned.append(store)
        return planned

    async def _guarded(
        self,
        store: DataStore,
        query: Query,
        fetch_k: int,
        passthrough: Mapping[str, Any],
        result: RetrievalResult,
    ) -> list[ScoredChunk]:
        """Run one store's retriever under its deadline and its breaker.

        The breaker is fed only outage-shaped failures. A timeout is one of them —
        a store that cannot answer inside the deadline is not answering — and it is
        converted to :class:`StoreUnavailable` so the diagnostics name the store
        rather than reporting a bare ``TimeoutError``. Everything else (a rejected
        generated query, a missing index, a malformed filter) propagates untouched
        to :func:`~ragorc.retrieve.ensemble.run_leg`, which records it without
        opening the circuit: those failures prove the store is up.

        A store whose retriever is a fan-out in its own right — one exposing
        ``retrieve_detailed``, which is
        :class:`~ragorc.retrieve.hybrid.HybridRetriever` on the default wiring — is
        exempt from the deadline, because it already applies *this same setting* to
        each of its own legs. Two timers spending one budget do not reinforce each
        other: the outer one is armed first, so it can only ever expire first, and
        when it does it cancels the whole inner fan-out. That turned a vector store
        which had already answered from dense search into zero chunks, a
        ``StoreUnavailable``, and — worse here than in the graph, because this layer
        also holds the breakers — a recorded failure, so four requests through one
        wedged optional leg cut a healthy store out for the whole cooldown. Its
        per-leg failures are adopted under ``store/leg`` rather than dropped, since
        surviving the outage silently would trade one bug for another. What the
        exemption gives up is a bound on that retriever's *pre-leg* work (embedding
        the query), which is why it is granted only to retrievers that demonstrably
        bound their legs themselves.
        """
        breaker = self.breakers[store]
        retriever = self.retrievers[store]
        detailed = getattr(retriever, "retrieve_detailed", None)
        timeout_s = self.settings.retrieval.per_store_timeout_s
        try:
            if detailed is not None:
                nested = await detailed(query, top_k=fetch_k, **dict(passthrough))
                for leg, message in nested.errors.items():
                    result.errors[f"{store.value}/{leg}"] = message
                chunks = list(nested.chunks)
                if nested.errors and not chunks:
                    # Every leg failed and nothing came back — an outage, however
                    # gracefully it was reported. This branch is exempt from the
                    # outer deadline because the retriever bounds its own legs, and
                    # `HybridRetriever.retrieve_detailed` also converts *all* leg
                    # failures into `RetrievalResult.errors` and returns normally.
                    # So neither `except` below could fire here and
                    # `record_success()` ran on a total outage: the breaker never
                    # opened, and a wedged Qdrant cost every request the full
                    # per-store deadline forever — measured, six requests at
                    # 1001 ms each with `failures=0`.
                    #
                    # Recorded, not raised: the fan-out already has the per-leg
                    # errors and degrades on them. What was missing is the breaker
                    # learning, so the *next* request fails fast instead of
                    # spending the deadline again.
                    breaker.record_failure()
                    return chunks
            else:
                coro = retriever.retrieve(query, top_k=fetch_k, **dict(passthrough))
                chunks = (
                    await asyncio.wait_for(coro, timeout_s)
                    if timeout_s and timeout_s > 0
                    else await coro
                )
        except TimeoutError as exc:
            breaker.record_failure()
            raise StoreUnavailable(
                store.value, f"timed out after {timeout_s}s", retriever=retriever.name
            ) from exc
        except StoreUnavailable:
            breaker.record_failure()
            raise
        breaker.record_success()
        return chunks

    @staticmethod
    def _record(legs: Sequence[LegResult], result: RetrievalResult) -> None:
        """Fold leg outcomes into the diagnostics.

        A failed leg still gets a ``per_store`` entry (empty) and a ``timings_ms``
        entry: "consulted, returned nothing" and "consulted, blew up after 10
        seconds" are different facts, and only recording the second one for
        successes makes the timing table lie about where the latency went.
        """
        for leg in legs:
            result.per_store[leg.name] = leg.chunks
            result.timings_ms[leg.name] = leg.ms
            if leg.error is not None:
                result.errors[leg.name] = leg.error
        result.total_candidates = sum(len(chunks) for chunks in result.per_store.values())

    def _fuse(self, query: Query, legs: Sequence[LegResult], limit: int) -> list[ScoredChunk]:
        """Fuse the contributing legs, then apply the noise filter.

        Named lists rather than positional ones, so the fusion audit trail carries
        real store names and ``retrieval.fusion_weights`` can be looked up by them.

        A single contributing store skips fusion entirely: aligning one list onto
        one id axis and combining it with nothing is work with no effect, and for
        ``max``/``relative`` fusion the per-list normalization would rescale a
        perfectly good ranking for no reason.

        The noise filter runs *after* fusion, never before. Its relative cutoff is a
        fraction of the top score, and before fusion the top score may be a constant
        from the SQL leg — measuring cosine similarities against a number that is not
        a similarity, which discards the entire dense list on a hard query.
        """
        contributing = {leg.name: leg.chunks for leg in legs if leg.chunks}
        if not contributing:
            return []

        if len(contributing) == 1:
            fused = list(next(iter(contributing.values())))
        else:
            fused = fuse(
                contributing,
                self.method,
                weights=self.weights,
                settings=self.settings,
            )

        filtered, report = self.noise.apply(fused, top_k=limit, query_vector=query.dense)
        # The noise filter already re-stamps rank; re-stamping here would be
        # redundant, and rewriting ``source`` would be wrong — the fusion layer
        # marks a chunk FUSED only when more than one list actually contributed it,
        # which is the provenance the context packer prints and citations rely on.
        if report.removed:
            log.debug(
                "multi_store_filtered",
                kept=report.kept,
                exact_duplicates=report.exact_duplicates,
                near_duplicates=report.near_duplicates,
                below_threshold=report.below_threshold,
                diversity_dropped=report.diversity_dropped,
            )
        return filtered
