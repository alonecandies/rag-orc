"""Fusion across *arbitrary* retrievers — the general case.

The hybrid retriever knows what it is combining: dense and sparse vectors in one
Qdrant collection, optionally Postgres full-text. This one knows nothing about
its inputs beyond the :class:`~ragorc.core.protocols.Retriever` shape, which is
what lets it combine things that have no common substrate — a vector search, a
generated SQL query, a Cypher traversal, a GraphRAG community search, a web
search, a cached answer store, someone's LangChain retriever passed straight in.

That generality forces three design decisions.

**Isolation.** Any of those can be slow, broken, or absent. A Text-to-SQL leg can
hit a lock; a web search can hang on DNS; a Neo4j instance can be mid-restart.
Every retriever therefore runs inside its own deadline
(``retrieval.per_store_timeout_s``) and its own exception boundary, and a leg
that blows either is *dropped* — logged, recorded in the diagnostics, and left
out of the fusion. One dead backend degrades the answer; it must never fail a
query that three other sources could have answered. The two exceptions are
deliberate: a :class:`~ragorc.core.errors.GuardrailViolation` and a
:class:`~ragorc.core.errors.BudgetExceeded` propagate, because those are
decisions rather than outages and degrading past a security guard is how a guard
becomes a data leak.

**Per-source normalization.** Scores arriving here have no common scale: cosine
in [0, 1], unbounded BM25, ``ts_rank_cd`` cover density, a graph path weight, a
cross-encoder logit that can be negative. Each source is min-max normalized
against its own results before fusion. That is exactly free for four of the five
combiners — min-max is a positive affine map per list, and RRF (ranks don't
move), DBSF (z-scores are affine-invariant) and both min-max combiners
(idempotent) are all invariant under one. ``max_fusion`` is the single exception,
because taking a maximum over raw scores is the whole point of it, so that method
is handed raw input. Normalizing anyway keeps the ``raw_*`` provenance uniform
across every method for free.

**Weights are per retriever, not per store.** ``retrieval.fusion_weights`` names
the four standard modalities; anything else defaults to 1.0 rather than to 0, so
adding a custom retriever cannot silently produce an unweighted no-op.

``retrieve_detailed`` returns the full :class:`RetrievalResult` — per-store
results, per-store latency, per-store errors — because that is the object the
pipeline's diagnostics and the eval harness need, and because reconstructing
"which store contributed this" after fusion has flattened the scores is
impossible. ``retrieve`` is the protocol-shaped wrapper over it. Neither stores
per-request state on the instance: one retriever object serves concurrent
requests, and per-request state on ``self`` is a data race that produces
plausible wrong answers rather than a crash.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import BudgetExceeded, GuardrailViolation, StoreUnavailable
from ragorc.core.models import FusionMethod, Query, RetrievalResult, ScoredChunk
from ragorc.core.protocols import Retriever
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.retrieve.fusion import Weights, fuse
from ragorc.retrieve.noise import NoiseFilter, normalize_scores

log = structlog.get_logger(__name__)

__all__ = ["EnsembleRetriever", "LegResult", "run_leg", "run_legs"]


# ---------------------------------------------------------------------------
# Leg execution: shared with the hybrid retriever
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class LegResult:
    """One retriever's outcome: results, latency, and how it failed if it did."""

    name: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def run_leg(
    name: str,
    coro: Coroutine[Any, Any, list[ScoredChunk]],
    *,
    timeout_s: float | None,
    label: str = "ensemble",
) -> LegResult:
    """Await one leg under a deadline, converting failure into a diagnostic.

    The deadline is not a nicety. Without it the request's latency is defined by
    its slowest backend, so p99 becomes p99-of-the-worst-store and one degraded
    dependency drags the whole service down with it. With it, a slow store simply
    does not contribute — which is the correct trade for a system whose other
    legs already found something.

    Cancellation is re-raised rather than recorded: a cancelled leg means the
    *caller* is going away, and swallowing that turns a shutdown into a hang.
    """
    start = time.perf_counter()
    try:
        chunks = (
            await asyncio.wait_for(coro, timeout_s) if timeout_s and timeout_s > 0 else await coro
        )
    except TimeoutError:
        elapsed = (time.perf_counter() - start) * 1000.0
        log.warning("retriever_timeout", retriever=name, label=label, timeout_s=timeout_s)
        return LegResult(name=name, ms=elapsed, error=f"timeout after {timeout_s}s")
    except asyncio.CancelledError:
        raise
    except (GuardrailViolation, BudgetExceeded):
        # Not an outage: a decision. Degrading past it is the failure mode the
        # guard exists to prevent.
        raise
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        level = log.warning if isinstance(exc, StoreUnavailable) else log.error
        level(
            "retriever_failed",
            retriever=name,
            label=label,
            error=str(exc)[:300],
            error_type=type(exc).__name__,
        )
        return LegResult(name=name, ms=elapsed, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    return LegResult(name=name, chunks=list(chunks), ms=(time.perf_counter() - start) * 1000.0)


async def run_legs(
    jobs: Mapping[str, Coroutine[Any, Any, list[ScoredChunk]]],
    *,
    timeout_s: float | None,
    limit: int,
    label: str = "ensemble",
) -> list[LegResult]:
    """Run every leg concurrently under a shared concurrency ceiling.

    ``bounded_gather`` rather than a bare ``asyncio.gather``: the ensemble can be
    handed fifty retrievers, and fifty simultaneous connections to four backends
    is how a query rate-limits itself.

    ``return_exceptions=True`` even though :func:`run_leg` already converts
    failures into results — the two exceptions it deliberately re-raises would
    otherwise leave the sibling legs running as orphans whose exceptions are never
    retrieved. Collecting them and re-raising the first keeps the abort clean.
    """
    if not jobs:
        return []
    names = list(jobs)
    outcomes = await bounded_gather(
        (run_leg(name, jobs[name], timeout_s=timeout_s, label=label) for name in names),
        limit=max(1, limit),
        return_exceptions=True,
    )
    legs: list[LegResult] = []
    raised: BaseException | None = None
    for outcome in outcomes:
        if isinstance(outcome, LegResult):
            legs.append(outcome)
        elif raised is None:
            raised = outcome
    if raised is not None:
        raise raised
    return legs


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------
@register("retriever", "ensemble", "multi")
class EnsembleRetriever:
    """Weighted fusion over any collection of retrievers."""

    name = "ensemble"

    def __init__(
        self,
        retrievers: Sequence[Retriever] | Mapping[str, Retriever] | None = None,
        *,
        weights: Weights = None,
        method: FusionMethod | str | None = None,
        noise: NoiseFilter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrievers: dict[str, Retriever] = _named(retrievers)
        self.method = FusionMethod(method) if method is not None else self.settings.retrieval.fusion
        # Explicit weights win; otherwise the configured per-modality table, which
        # only names the four standard retrievers and defaults the rest to 1.0.
        self.weights: Weights = (
            weights if weights is not None else self.settings.retrieval.fusion_weights
        )
        self.noise = noise

    def add(self, name: str, retriever: Retriever, weight: float | None = None) -> None:
        """Register another leg. Used by the pipeline builder when routing has
        decided which stores are in play for this query."""
        self.retrievers[name] = retriever
        if weight is not None:
            table = dict(self.weights) if isinstance(self.weights, Mapping) else {}
            table[name] = float(weight)
            self.weights = table

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> list[ScoredChunk]:
        result = await self.retrieve_detailed(query, top_k=top_k, **kw)
        return result.chunks

    async def retrieve_detailed(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> RetrievalResult:
        """Run every leg, fuse the survivors, and report what happened.

        Keyword arguments are forwarded verbatim to each leg (``filters``,
        ``tenant_id``, ...), because a leg that does not understand one has a
        ``**kw`` in its signature by protocol and will ignore it.
        """
        rs = self.settings.retrieval
        k = int(top_k or query.top_k or rs.top_k)
        # Candidates, not final results: recall is this stage's job and precision
        # is the reranker's, so the ensemble hands over the wider set.
        fetch_k = max(int(kw.pop("fetch_k", None) or rs.fetch_k), k)

        if not self.retrievers:
            log.warning("ensemble_no_retrievers", hint="add legs before querying")
            return RetrievalResult()

        jobs = {
            name: retriever.retrieve(query, top_k=fetch_k, **kw)
            for name, retriever in self.retrievers.items()
        }
        with timed("retrieve.ensemble", legs=len(jobs), fetch_k=fetch_k):
            legs = await run_legs(
                jobs,
                timeout_s=rs.per_store_timeout_s,
                limit=rs.max_concurrent_retrievers,
                label=self.name,
            )

        result = RetrievalResult()
        lists: dict[str, list[ScoredChunk]] = {}
        for leg in legs:
            result.timings_ms[leg.name] = round(leg.ms, 2)
            if leg.error is not None:
                result.errors[leg.name] = leg.error
                continue
            result.per_store[leg.name] = leg.chunks
            if leg.chunks:
                lists[leg.name] = self._normalize(leg.chunks)
        result.total_candidates = sum(len(v) for v in result.per_store.values())

        if not lists:
            log.warning(
                "ensemble_all_legs_empty",
                legs=len(legs),
                errors=len(result.errors),
                # Every leg failing is an outage worth alerting on; every leg
                # returning nothing is a legitimate "no results".
                failed_all=len(result.errors) == len(legs),
            )
            return result

        fused = fuse(
            lists,
            self.method,
            weights=self.weights,
            names=None,
            top_k=fetch_k,
            settings=self.settings,
        )
        result.chunks = self._denoise(fused, query, fetch_k)
        log.debug(
            "ensemble_fused",
            method=self.method.value,
            legs=len(lists),
            dropped=len(result.errors),
            candidates=len(result.chunks),
        )
        return result

    # -- helpers -----------------------------------------------------------
    def _normalize(self, chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """Min-max each leg against itself, except under ``max`` fusion.

        See the module docstring: this is order-preserving for every combiner, and
        score-preserving for none — which is why ``max_fusion``, the one method
        that reads raw magnitudes, opts out.
        """
        if self.method is FusionMethod.MAX:
            return chunks
        return normalize_scores(chunks, method="minmax")

    def _denoise(self, chunks: list[ScoredChunk], query: Query, fetch_k: int) -> list[ScoredChunk]:
        if self.noise is None:
            return chunks
        kept, report = self.noise.apply(chunks, top_k=fetch_k, query_vector=query.dense)
        if report.removed:
            log.debug("ensemble_denoised", kept=report.kept, removed=report.removed)
        return kept


def _named(
    retrievers: Sequence[Retriever] | Mapping[str, Retriever] | None,
) -> dict[str, Retriever]:
    """Key retrievers by name, keeping duplicates distinguishable.

    Two instances of the same class (two vector stores, two web providers) would
    otherwise collide on ``name`` and the second would silently replace the first.
    """
    if retrievers is None:
        return {}
    if isinstance(retrievers, Mapping):
        return dict(retrievers)
    out: dict[str, Retriever] = {}
    for i, retriever in enumerate(retrievers):
        base = getattr(retriever, "name", None) or type(retriever).__name__.lower()
        key = base if base not in out else f"{base}~{i}"
        out[key] = retriever
    return out
