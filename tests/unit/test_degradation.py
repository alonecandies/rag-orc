"""What the library does when one thing is broken, against what it says it does.

Three claims, each stated in a docstring, each false in a different way:

* `MultiStoreRetriever._guarded` exempts a self-bounding retriever from the outer
  deadline and then calls `breaker.record_success()` — but that retriever converts
  *all* leg failures into `RetrievalResult.errors` and returns normally, so a total
  outage was recorded as a success. The breaker never opened and a wedged Qdrant
  cost every request the full per-store deadline forever: six requests, 1001 ms
  each, `failures=0`.
* `_LinearEngine._route` says it degrades to "query everything". It returned
  `RouteDecision(stores=())`, and an empty-but-present route is the *opposite*
  branch in `_plan`: nothing planned, zero chunks, no error recorded.
* `RagService._cache_set` refuses to cache a degraded answer by reading
  `answer.metadata["errors"]`, and only the graph path ever wrote that key — so the
  HTTP path cached the outage for the full TTL, which is what its own docstring
  says it was written to stop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ragorc.core.models import Chunk, DataStore, Query, RetrievalResult, ScoredChunk
from ragorc.core.settings import Settings


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
        "security": {"enforce_tenant_isolation": False},
    }
    base.update(over)
    return Settings(**base)


class _GracefullyDead:
    """A self-bounding retriever whose legs all fail — the shipped vector leg's
    shape. It reports rather than raises, which is correct and is exactly what
    made the outage invisible to the breaker."""

    name = "hybrid"

    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: Query, **kw: Any) -> list[ScoredChunk]:
        result = await self.retrieve_detailed(query, **kw)
        return result.chunks

    async def retrieve_detailed(self, query: Query, **kw: Any) -> RetrievalResult:
        self.calls += 1
        await asyncio.sleep(0)
        result = RetrievalResult()
        result.errors = {"dense": "timed out after 10s"}
        return result


class _Healthy(_GracefullyDead):
    async def retrieve_detailed(self, query: Query, **kw: Any) -> RetrievalResult:
        self.calls += 1
        result = RetrievalResult()
        result.chunks = [ScoredChunk(chunk=Chunk(id="c1", content="ok"), score=1.0)]
        return result


def _multi(vector: Any) -> Any:
    from ragorc.retrieve.multi_store import MultiStoreRetriever

    return MultiStoreRetriever(vector=vector, settings=_settings())


async def test_a_total_outage_is_recorded_as_a_breaker_failure() -> None:
    """However gracefully it was reported. The breaker is what stops the next
    request paying the same deadline."""
    retriever = _multi(_GracefullyDead())
    breaker = retriever.breakers[DataStore.VECTOR]

    await retriever.retrieve_detailed(Query(text="q"), top_k=3)

    assert breaker._failures > 0, "a total outage was recorded as a success"


async def test_a_partial_result_is_still_a_success() -> None:
    """One leg down and chunks returned is degradation, not an outage — tripping
    the breaker on it would take out a store that is answering."""
    class _Partial(_GracefullyDead):
        async def retrieve_detailed(self, query: Query, **kw: Any) -> RetrievalResult:
            result = RetrievalResult()
            result.errors = {"sparse": "down"}
            result.chunks = [ScoredChunk(chunk=Chunk(id="c1", content="ok"), score=1.0)]
            return result

    retriever = _multi(_Partial())
    await retriever.retrieve_detailed(Query(text="q"), top_k=3)

    assert retriever.breakers[DataStore.VECTOR]._failures == 0


async def test_a_healthy_store_is_never_penalised() -> None:
    retriever = _multi(_Healthy())
    await retriever.retrieve_detailed(Query(text="q"), top_k=3)
    assert retriever.breakers[DataStore.VECTOR]._failures == 0


async def test_the_breaker_eventually_stops_paying_the_deadline() -> None:
    """The point of the accounting: repeated outages open the breaker, and an open
    breaker fails fast instead of spending the per-store timeout again."""
    inner = _GracefullyDead()
    retriever = _multi(inner)
    breaker = retriever.breakers[DataStore.VECTOR]

    for _ in range(12):
        await retriever.retrieve_detailed(Query(text="q"), top_k=3)

    assert breaker.is_open, f"still closed after 12 outages (failures={breaker._failures})"
    before = inner.calls
    await retriever.retrieve_detailed(Query(text="q"), top_k=3)
    assert inner.calls == before, "an open breaker still called the wedged store"


# ---------------------------------------------------------------------------
# The router's fallback
# ---------------------------------------------------------------------------
async def test_a_failed_route_queries_everything() -> None:
    """"Query everything" is what the fan-out does for *no* route. A route that is
    present and names no store is the opposite branch."""
    from ragorc.server.app import _LinearEngine

    engine = object.__new__(_LinearEngine)
    engine.settings = _settings()

    class _Broken:
        async def route(self, query: Query) -> Any:
            raise RuntimeError("router backend down")

    engine.router = _Broken()
    engine._router = lambda: _Broken()  # type: ignore[method-assign]

    decision, _usage = await engine._route(Query(text="q"))

    assert decision is None, f"an empty-but-present route plans nothing: {decision}"


async def test_no_route_reaches_every_store() -> None:
    """The behaviour the fallback is named for, asserted on the fan-out itself."""
    retriever = _multi(_Healthy())

    result = await retriever.retrieve_detailed(Query(text="q"), top_k=3, route=None)

    assert result.chunks, "route=None retrieved nothing"


# ---------------------------------------------------------------------------
# The degraded answer must not be cached
# ---------------------------------------------------------------------------
def test_the_linear_path_stamps_store_errors_on_the_answer() -> None:
    """`_cache_set`'s guard reads `answer.metadata["errors"]`, and only the graph
    path wrote it — so on the HTTP path the predicate was never true and the
    outage answer was served for the whole TTL."""
    import inspect

    from ragorc.server.app import _LinearEngine

    source = inspect.getsource(_LinearEngine.query)
    assert 'answer.metadata["errors"]' in source, "the cache guard has no writer on this path"
    assert "store_errors" in source


def test_the_guard_still_reads_what_the_path_now_writes() -> None:
    """Two halves of one rule, in two classes. A writer whose key the reader does
    not check is the same defect pointed the other way."""
    import inspect

    from ragorc.server.app import RagService, _LinearEngine

    writer = inspect.getsource(_LinearEngine.query)
    reader = inspect.getsource(RagService._cache_set)
    assert 'metadata["errors"]' in writer
    assert 'metadata.get("errors")' in reader
