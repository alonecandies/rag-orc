"""The OpenRouter client's concurrency behaviour.

Driven through ``httpx.MockTransport`` rather than a stubbed method, because the
property under test is about *when permits are held*, and that only shows up when
the real generator plumbing runs.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from ragorc.core.settings import LLMSettings
from ragorc.llm.openrouter import OpenRouterLLM

_CHAT = {
    "choices": [{"message": {"content": "ok"}}],
    "model": "test/model",
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}


def _handler(request: httpx.Request) -> httpx.Response:
    """Answer a streaming request with SSE deltas and anything else with a chat."""
    if json.loads(request.content).get("stream"):
        deltas = [
            f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in "abcdef"
        ]
        return httpx.Response(
            200, text="".join(deltas), headers={"content-type": "text/event-stream"}
        )
    return httpx.Response(200, json=_CHAT)


def _client(**overrides: object) -> OpenRouterLLM:
    settings = LLMSettings(api_key="sk-test", **overrides)  # type: ignore[arg-type]
    llm = OpenRouterLLM(settings=settings)
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_handler), base_url="http://provider.invalid"
    )
    return llm


async def test_a_half_read_stream_does_not_starve_other_llm_work() -> None:
    """An async generator holds its permit across every ``yield``.

    So a permit taken from the shared pool stays taken for as long as the *client*
    takes to read the response — and with ``max_concurrency`` slow readers, every
    other call in the process stops: grading, routing, synthesis, ingest. Streams
    therefore draw on their own pool, and contend only with each other.
    """
    llm = _client(max_concurrency=1, max_concurrent_streams=1)
    stream = llm.stream("tell me a story")
    try:
        assert await stream.__anext__() == "a", "the stream must actually be mid-flight"
        # The generator is now parked at a yield with its permit held. Unrelated,
        # non-streaming work must still get through.
        text, _ = await asyncio.wait_for(llm.complete("unrelated"), timeout=5.0)
        assert text == "ok"
    finally:
        await stream.aclose()
        await llm.aclose()


async def test_streams_still_bound_each_other() -> None:
    """The fix must not remove the bound, only stop it being shared: an unbounded
    stream pool is the denial of service the ceiling exists to prevent."""
    llm = _client(max_concurrency=8, max_concurrent_streams=1)
    first = llm.stream("one")
    try:
        await first.__anext__()
        second = llm.stream("two")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(second.__anext__(), timeout=0.5)
        await second.aclose()
    finally:
        await first.aclose()
        await llm.aclose()


async def test_a_finished_stream_returns_its_permit() -> None:
    """Draining one stream fully must leave the pool exactly as it was."""
    llm = _client(max_concurrency=4, max_concurrent_streams=1)
    try:
        for _ in range(3):
            assert "".join([delta async for delta in llm.stream("go")]) == "abcdef"
    finally:
        await llm.aclose()


async def test_a_structured_repair_keeps_the_caller_s_token_cap() -> None:
    """`max_tokens` was read with `kwargs.pop` *inside* the retry loop.

    So the first attempt got the caller's cap and every repair after it got
    `None`, falling back to the global `llm.max_tokens` — the repair, which is the
    attempt most likely to run long, was the only attempt running uncapped. It
    also makes the cap unenforceable: a caller who sets it to keep a structured
    extraction small cannot rely on it surviving one malformed response.
    """
    from pydantic import BaseModel

    class Verdict(BaseModel):
        ok: bool

    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        # First response is unparseable, forcing exactly one repair round trip.
        content = "not json at all" if len(bodies) == 1 else json.dumps({"ok": True})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "model": "test/model",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    settings = LLMSettings(api_key="sk-test", max_tokens=4096)
    llm = OpenRouterLLM(settings=settings)
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://provider.invalid"
    )

    result, _usage = await llm.structured("give a verdict", Verdict, max_tokens=64)

    assert result.ok is True
    assert len(bodies) >= 2, "the test needs a repair round trip to say anything"
    caps = [b.get("max_tokens") for b in bodies]
    assert caps == [64] * len(bodies), f"every attempt must carry the caller's cap, saw {caps}"


# ---------------------------------------------------------------------------
# The per-request ceilings, under fan-out
# ---------------------------------------------------------------------------
def _yielding_client(served: list[int], *, fail_after: int | None = None, **overrides: object):
    """A client whose transport actually awaits.

    This is the load-bearing detail. `MockTransport` with a synchronous handler
    returns without yielding, which serializes the coroutines and hides the
    defect entirely: the first measurement of this bug read "ceiling held" for
    exactly that reason. A real provider always yields.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        served.append(1)
        await asyncio.sleep(0.01)
        if fail_after is not None and len(served) > fail_after:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_CHAT)

    settings = LLMSettings(api_key="k", max_retries=1, **overrides)  # type: ignore[arg-type]
    llm = OpenRouterLLM(settings=settings)
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://provider.invalid"
    )
    return llm


async def test_the_call_ceiling_bounds_a_fan_out_not_just_a_sequence() -> None:
    """`check()` reads what has been *recorded*, and a call records only after its
    round trip — so every coroutine in a fan-out passed the gate against a ledger
    still reading zero. Measured before the fix: 40 requests against a five-call
    ceiling. The ceiling was read forty times and enforced none."""
    from ragorc.core.telemetry import new_request_context

    served: list[int] = []
    llm = _yielding_client(served, max_concurrency=64)
    try:
        with new_request_context(request_id="r", max_calls=5, trace=False):
            await llm.batch([f"p{i}" for i in range(40)])
        assert len(served) == 5, f"{len(served)} provider requests against a ceiling of 5"
    finally:
        await llm.aclose()


async def test_the_ceiling_still_admits_exactly_what_it_permits() -> None:
    """The other half: a reservation that never releases would strangle the
    request instead of bounding it."""
    from ragorc.core.telemetry import new_request_context

    served: list[int] = []
    llm = _yielding_client(served, max_concurrency=8)
    try:
        with new_request_context(request_id="r", max_calls=10, trace=False):
            results = await llm.batch([f"p{i}" for i in range(6)])
        assert len(served) == 6
        assert [r for r in results if r] and len(results) == 6
    finally:
        await llm.aclose()


async def test_a_failed_call_returns_its_reservation() -> None:
    """Released in a `finally`, so a failure gives the budget back rather than
    wedging the ceiling for the rest of the request."""
    from ragorc.core.telemetry import new_request_context

    served: list[int] = []
    llm = _yielding_client(served, fail_after=0, max_concurrency=4)
    try:
        with new_request_context(request_id="r", max_calls=3, trace=False) as (_t, ledger):
            await llm.batch(["a", "b"])
            assert ledger._reserved_calls == 0, "a failed call kept its claim"
    finally:
        await llm.aclose()


def test_reservations_count_as_spent_while_in_flight() -> None:
    """A call that has passed the gate and not yet returned is money committed.
    Treating it as zero is precisely what let the fan-out through."""
    from ragorc.core.errors import BudgetExceeded
    from ragorc.core.telemetry import CostLedger

    ledger = CostLedger(max_calls=2)
    with ledger.reserve():
        ledger.check()  # one claimed, one left
        with ledger.reserve(), pytest.raises(BudgetExceeded, match="call budget"):
            ledger.check()
    ledger.check()  # both released


def test_the_token_ceiling_is_reserved_from_max_tokens() -> None:
    """`max_tokens` is the most a call can spend and is known before it is made,
    so it can be claimed exactly rather than discovered afterwards."""
    from ragorc.core.errors import BudgetExceeded
    from ragorc.core.telemetry import CostLedger

    ledger = CostLedger(max_tokens=100)
    with ledger.reserve(tokens=90), pytest.raises(BudgetExceeded, match="token budget"):
        ledger.check(projected_tokens=20)
    ledger.check(projected_tokens=20)
