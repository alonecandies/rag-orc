"""Two requests at once, and a ceiling that never fired.

`SemanticCache._ensure` had two awaits between reading `_ready` and setting it, so
two concurrent cold requests both created the collection and the loser's write
failed with `409 Collection already exists` — silently, since `set` degrades:

    gather: [None, None]
    points stored after two concurrent cold sets: 1 (expected 2)

`llm.tokens_per_minute` was a fully implemented token bucket that nothing debited:
`RateLimiter.acquire` defaults to `tokens=0` and `_post_once` called it bare.

    20 calls x max_tokens=100 = 2000 tokens against tokens_per_minute=60
    elapsed 0.003s; allowance after the traffic: 60.000 (started at 60)
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ragorc.core.settings import Settings


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "embedding": {"dense_dimension": 32},
        "security": {"enforce_tenant_isolation": False},
    }
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# The lazy init
# ---------------------------------------------------------------------------
class _RacyClient:
    """A Qdrant client that rejects a second create, as the real one does."""

    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.creates = 0

    async def collection_exists(self, name: str) -> bool:
        await asyncio.sleep(0)  # the await that made the check-then-act a race
        return name in self.collections

    async def create_collection(self, collection_name: str, **kw: Any) -> None:
        self.creates += 1
        await asyncio.sleep(0)
        if collection_name in self.collections:
            raise RuntimeError(f"Collection `{collection_name}` already exists!")
        self.collections.add(collection_name)


async def test_two_cold_requests_create_the_collection_once() -> None:
    from ragorc.cache.semantic import SemanticCache

    cache = SemanticCache(
        embedder=type("E", (), {"dimension": 32})(),
        settings=_settings(cache={"enabled": True, "semantic_enabled": True}),
    )
    cache._client = _RacyClient()

    await asyncio.gather(*(cache._ensure() for _ in range(8)))

    assert cache._client.creates == 1, f"created {cache._client.creates} times"


async def test_the_init_is_still_lazy() -> None:
    """The lock must not turn construction into I/O."""
    from ragorc.cache.semantic import SemanticCache

    cache = SemanticCache(
        embedder=type("E", (), {"dimension": 32})(),
        settings=_settings(cache={"enabled": True, "semantic_enabled": True}),
    )
    assert cache._client is None
    assert not cache._ready


def test_the_guard_covers_the_whole_initialization() -> None:
    """A `collection_exists` retry would fix the create and leave the client
    build racing. The race is the whole of `_ensure`, not just its last await."""
    import inspect

    from ragorc.cache.semantic import SemanticCache

    source = inspect.getsource(SemanticCache._ensure)
    assert "async with self._init_lock" in source
    assert "_ensure_locked" in source


# ---------------------------------------------------------------------------
# The token ceiling
# ---------------------------------------------------------------------------
def test_a_request_declares_what_it_may_spend() -> None:
    """Prompt plus the reserved completion. A limiter that debits after the fact
    cannot stop the call that breaches the ceiling — the same reason
    `CostLedger.reserve` claims budget before a request rather than after."""
    from ragorc.core.tokens import count_tokens
    from ragorc.llm.openrouter import OpenRouterLLM

    llm = object.__new__(OpenRouterLLM)
    body = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "why is late chunking cheaper?"},
        ],
        "max_tokens": 256,
    }

    cost = llm._token_cost(body)

    prompt = count_tokens("you are helpfulwhy is late chunking cheaper?")
    assert cost == prompt + 256, f"{cost} != {prompt} + 256"
    assert cost > 256, "the prompt side is not counted"


def test_the_limiter_is_debited_with_that_cost() -> None:
    """The call site. A `_token_cost` nothing passes is the defect this round is
    named for, and `acquire()` defaults to `tokens=0` — so the bare call type-checks,
    runs, and throttles nothing."""
    import ast
    import inspect
    import textwrap

    from ragorc.llm.openrouter import OpenRouterLLM

    tree = ast.parse(textwrap.dedent(inspect.getsource(OpenRouterLLM._post_once)))
    acquires = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "acquire"
    ]
    assert len(acquires) == 1, "expected one limiter acquire"
    assert acquires[0].args, "acquire() was called bare, so the token bucket never debits"


async def test_the_token_ceiling_actually_throttles() -> None:
    """Behaviour, not the argument: the bucket is real and was already tested —
    what was missing was anything spending from it."""
    from ragorc.core.concurrency import RateLimiter

    limiter = RateLimiter(requests_per_minute=None, tokens_per_minute=60)
    await limiter.acquire(60)
    assert limiter._tok_allowance == pytest.approx(0.0, abs=1e-6)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await limiter.acquire(30)
    assert loop.time() - started > 0.5, "a spent token budget did not wait"


def test_both_provider_paths_debit_the_token_ceiling() -> None:
    """The gap in the previous round's fix, and its shape.

    Two methods reach the provider — `_post_once` and the streaming generator —
    and the fix landed in one. Streaming spends the same tokens and, held across
    every yield, spends them for longer, so the *more* expensive path was the one
    still unbounded. Enumerating every site that reaches the resource is the
    checklist item; this is the test that makes it stick.
    """
    import ast
    import inspect
    import textwrap

    from ragorc.llm.openrouter import OpenRouterLLM

    source = textwrap.dedent(inspect.getsource(OpenRouterLLM))
    bare: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "acquire"
            and not node.args
        ):
            bare.append(node.lineno)
    assert not bare, f"a provider call acquires the limiter without a token cost: lines {bare}"
