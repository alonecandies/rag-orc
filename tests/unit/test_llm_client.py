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
