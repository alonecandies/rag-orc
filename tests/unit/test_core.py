"""Core contracts: ids, tokens, models, concurrency, telemetry, registry."""

from __future__ import annotations

import asyncio
import pathlib

import numpy as np
import pytest

from ragorc.core.concurrency import (
    CircuitBreaker,
    RateLimiter,
    bounded_gather,
    gather_dict,
    retry_async,
    run_with_timeout,
    safe_gather,
)
from ragorc.core.errors import BudgetExceeded, StoreUnavailable, TransientError
from ragorc.core.ids import cache_key, chunk_id, content_hash, document_id, stable_uuid
from ragorc.core.models import (
    Chunk,
    GraphPath,
    Relation,
    SparseVector,
    Usage,
    dedupe_scored,
)
from ragorc.core.registry import available, register, resolve
from ragorc.core.telemetry import (
    CostLedger,
    _redact_secrets,
    new_request_context,
    redact_identifiers,
    timed,
)
from ragorc.core.tokens import TokenBudget, count_tokens, count_tokens_batch, truncate_to_tokens


# ---------------------------------------------------------------------------
# ids — determinism is what makes ingest idempotent
# ---------------------------------------------------------------------------
def test_ids_are_deterministic() -> None:
    assert chunk_id("d1", 0, "hello") == chunk_id("d1", 0, "hello")
    assert document_id("a.md", "body") == document_id("a.md", "body")


def test_ids_change_with_content() -> None:
    """An edited chunk must get a new id so the stale vector is replaced."""
    assert chunk_id("d1", 0, "hello") != chunk_id("d1", 0, "hello!")
    assert chunk_id("d1", 0, "hello") != chunk_id("d1", 1, "hello")
    assert chunk_id("d1", 0, "hello", 0) != chunk_id("d1", 0, "hello", 1)


def test_content_hash_uses_a_separator() -> None:
    """('ab','c') must not collide with ('a','bc')."""
    assert content_hash("ab", "c") != content_hash("a", "bc")


def test_stable_uuid_is_a_valid_uuid() -> None:
    import uuid

    value = stable_uuid("chunk", "d1", 0)
    assert uuid.UUID(value)  # Qdrant point ids must be a UUID or an unsigned int


def test_cache_key_is_namespaced() -> None:
    assert cache_key("llm", "a").startswith("llm:")
    assert cache_key("llm", "a") != cache_key("embed", "a")


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------
def test_count_tokens_batch_matches_individual() -> None:
    texts = ["hello world", "a much longer sentence with more tokens in it", ""]
    assert count_tokens_batch(texts) == [count_tokens(t) for t in texts]


def test_truncate_respects_token_boundary() -> None:
    text = "word " * 200
    out = truncate_to_tokens(text, 10)
    assert count_tokens(out) <= 10
    assert truncate_to_tokens(text, 0) == ""


def test_token_budget_reserves_output_first() -> None:
    """The classic failure: context fills the window and the model has no room
    left to answer. Reserving output first is what prevents it."""
    budget = TokenBudget(total=1000, reserved_output=200, reserved_system=100, safety_margin=0.0)
    assert budget.available_context == 700
    assert budget.fits(700)
    assert not budget.fits(701)


def test_token_budget_split_normalizes_shares() -> None:
    budget = TokenBudget(total=1000, reserved_output=0, safety_margin=0.0)
    split = budget.split({"a": 2.0, "b": 2.0})  # does not sum to 1
    assert split["a"] == split["b"] == 500


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def test_sparse_vector_dot_is_intersection_based() -> None:
    a = SparseVector.from_dict({1: 0.5, 5: 2.0})
    b = SparseVector.from_dict({5: 1.0, 9: 3.0})
    assert a.dot(b) == pytest.approx(2.0)
    assert a.dot(SparseVector.from_dict({})) == 0.0


def test_sparse_vector_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        SparseVector(np.array([1, 2]), np.array([1.0], dtype=np.float32))


def test_sparse_top_k_keeps_largest() -> None:
    vector = SparseVector.from_dict({1: 0.1, 2: 5.0, 3: 3.0})
    pruned = vector.top_k(2)
    assert set(pruned.values.tolist()) == {5.0, 3.0}


def test_chunk_embed_text_includes_prefix_but_payload_does_not() -> None:
    """The contextual prefix is embedded but must not be shown to the generator —
    it would duplicate context the prompt already carries."""
    chunk = Chunk(id="c", content="Revenue grew 40%.", contextual_prefix="From Acme's 2024 report.")
    assert chunk.embed_text.startswith("From Acme's 2024 report.")
    assert "Revenue grew 40%." in chunk.embed_text
    assert chunk.payload()["content"] == "Revenue grew 40%."


def test_chunk_payload_roundtrip_preserves_metadata() -> None:
    chunk = Chunk(
        id="c1",
        content="body",
        document_id="d1",
        index=3,
        start_char=10,
        end_char=14,
        level=2,
        parent_id="p1",
        metadata={"custom": "value", "source": "a.md"},
    )
    restored = Chunk.from_payload("c1", chunk.payload())
    assert restored.document_id == "d1"
    assert restored.index == 3
    assert restored.level == 2
    assert restored.parent_id == "p1"
    assert restored.metadata["custom"] == "value"
    assert "content" not in restored.metadata, "known keys must not leak into metadata"


def test_usage_sums() -> None:
    total = Usage.sum(
        [
            Usage(prompt_tokens=10, cost_usd=0.1, calls=1),
            Usage(completion_tokens=5, cost_usd=0.2, calls=1),
        ]
    )
    assert total.total_tokens == 15
    assert total.cost_usd == pytest.approx(0.3)
    assert total.calls == 2


def test_graph_path_verbalizes() -> None:
    path = GraphPath(
        nodes=("A", "B", "C"),
        relations=(Relation("A", "B", "WORKS_FOR"), Relation("B", "C", "OWNS")),
    )
    assert path.hops == 2
    assert path.verbalize() == "A -[WORKS_FOR]-> B -[OWNS]-> C"


def test_dedupe_scored_keeps_highest() -> None:
    from ragorc.core.models import ScoredChunk

    chunk = Chunk(id="same", content="x")
    out = dedupe_scored([ScoredChunk(chunk=chunk, score=0.4), ScoredChunk(chunk=chunk, score=0.9)])
    assert len(out) == 1
    assert out[0].score == 0.9


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------
async def test_bounded_gather_preserves_order_and_bounds() -> None:
    active = 0
    peak = 0

    async def work(i: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return i

    result = await bounded_gather((work(i) for i in range(10)), limit=3)
    assert result == list(range(10)), "order must follow input, not completion"
    assert peak <= 3, f"concurrency ceiling breached: {peak}"


async def test_safe_gather_separates_failures() -> None:
    async def ok() -> str:
        return "fine"

    async def boom() -> str:
        raise ValueError("nope")

    good, errors = await safe_gather([ok(), boom(), ok()], limit=2)
    assert good == ["fine", "fine"]
    assert len(errors) == 1


async def test_gather_dict_labels_results() -> None:
    async def value(v: int) -> int:
        return v

    out = await gather_dict({"a": value(1), "b": value(2)})
    assert out == {"a": 1, "b": 2}


async def test_run_with_timeout_returns_default() -> None:
    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    assert await run_with_timeout(slow(), 0.01, default="fallback") == "fallback"


async def test_retry_async_retries_then_succeeds() -> None:
    attempts = 0

    @retry_async(max_attempts=4, base_delay=0.001)
    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientError("not yet")
        return "ok"

    assert await flaky() == "ok"
    assert attempts == 3


async def test_retry_async_does_not_retry_other_errors() -> None:
    attempts = 0

    @retry_async(max_attempts=3, base_delay=0.001)
    async def broken() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError, match="permanent"):
        await broken()
    assert attempts == 1, "a non-transient error must not be retried"


async def test_circuit_breaker_opens_and_recovers() -> None:
    breaker = CircuitBreaker("test", failure_threshold=2, reset_timeout_s=0.05)

    for _ in range(2):
        breaker.record_failure()
    assert breaker.is_open
    with pytest.raises(StoreUnavailable):
        breaker.check()

    await asyncio.sleep(0.06)
    assert not breaker.is_open, "the breaker must half-open after the cooldown"
    breaker.record_success()
    assert not breaker.is_open


async def test_rate_limiter_delays() -> None:
    limiter = RateLimiter(requests_per_minute=600)  # 10/s
    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(4):
        await limiter.acquire()
    assert loop.time() - start >= 0.0  # bucket starts full; no false delay


# ---------------------------------------------------------------------------
# telemetry
# ---------------------------------------------------------------------------
def test_cost_ledger_enforces_ceiling_before_spend() -> None:
    ledger = CostLedger(max_cost_usd=0.10)
    ledger.record(Usage(model="m", cost_usd=0.09, calls=1), stage="answer")
    ledger.check()  # still under
    with pytest.raises(BudgetExceeded):
        ledger.check(projected_cost=0.05)


def test_cost_ledger_enforces_call_ceiling() -> None:
    ledger = CostLedger(max_calls=2)
    for _ in range(2):
        ledger.record(Usage(model="m", calls=1), stage="grade")
    with pytest.raises(BudgetExceeded):
        ledger.check()


def test_cost_ledger_reports_by_stage_and_model() -> None:
    ledger = CostLedger()
    ledger.record(Usage(model="cheap", cost_usd=0.001, calls=1, prompt_tokens=100), stage="grade")
    ledger.record(Usage(model="cheap", cost_usd=0.001, calls=1, prompt_tokens=100), stage="grade")
    ledger.record(Usage(model="strong", cost_usd=0.05, calls=1, prompt_tokens=2000), stage="answer")
    ledger.record(Usage(model="cheap", calls=1, cached=1), stage="route")

    report = ledger.report()
    assert report["calls"] == 4
    assert report["cached_calls"] == 1
    assert report["cache_hit_rate"] == pytest.approx(0.25)
    assert set(report["by_stage"]) == {"grade", "answer", "route"}
    assert report["by_stage"]["answer"]["cost_usd"] == pytest.approx(0.05)


def test_request_context_isolates_trace_and_ledger() -> None:
    with new_request_context(request_id="r1", max_cost_usd=1.0) as (trace, ledger):
        with timed("stage_one", detail="x"):
            pass
        ledger.record(Usage(model="m", calls=1), stage="stage_one")
        assert len(trace) == 1
        assert trace[0].name == "stage_one"
        assert ledger.total.calls == 1


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_resolves_and_lists() -> None:
    @register("widget", "alpha", "a")
    class Alpha:
        pass

    assert resolve("widget", "alpha") is Alpha
    assert resolve("widget", "a") is Alpha
    assert "alpha" in available("widget")["widget"]


def test_registry_rejects_unknown_name() -> None:
    """Two distinct failures, both at config-load time rather than mid-request:
    an unknown name within a known kind, and an entirely unknown kind."""
    from ragorc.core.errors import ConfigError

    @register("gadget", "known")
    class Known:
        pass

    with pytest.raises(ConfigError, match="unknown gadget"):
        resolve("gadget", "does-not-exist")

    with pytest.raises(ConfigError, match="no components registered"):
        resolve("no-such-kind", "whatever")


# ---------------------------------------------------------------------------
# uvloop: a policy is not a running loop
#
# `install_uvloop` used to set the policy and return True unconditionally, so
# every caller inside a coroutine — which is where a library is called from —
# was told it had uvloop while running on the stdlib selector loop, and
# docs/performance.md priced in a 2-4x that never arrived.
# ---------------------------------------------------------------------------
async def test_install_uvloop_admits_it_cannot_upgrade_a_running_loop() -> None:
    from structlog.testing import capture_logs

    import ragorc.core.concurrency as concurrency

    pytest.importorskip("uvloop")
    policy = asyncio.get_event_loop_policy()
    running = asyncio.get_running_loop()
    assert type(running).__module__.split(".")[0] != "uvloop", (
        "this test needs a non-uvloop loop to be the running one"
    )

    concurrency._uvloop_warned = False
    try:
        with capture_logs() as logs:
            first = concurrency.install_uvloop()
            second = concurrency.install_uvloop()
        assert first is False, "a policy change cannot replace the loop already running"
        assert second is False
        # Not mutated: the only thing setting it here could do is change loops
        # this call has nothing to do with, in the caller's process.
        assert type(asyncio.get_event_loop_policy()) is type(policy)
    finally:
        asyncio.set_event_loop_policy(policy)

    warned = [event for event in logs if event["event"] == "uvloop_not_in_use"]
    assert len(warned) == 1, f"expected exactly one advisory, got {warned}"
    assert "uvloop.run" in warned[0]["hint"]


def test_install_uvloop_takes_effect_before_a_loop_exists() -> None:
    """The recipe that does work, and the reason the return value is worth
    reading: with no loop yet, the next ``asyncio.run`` gets a uvloop loop."""
    uvloop = pytest.importorskip("uvloop")
    policy = asyncio.get_event_loop_policy()
    try:
        from ragorc.core.concurrency import install_uvloop

        assert install_uvloop() is True
        assert isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy)

        async def probe() -> str:
            return type(asyncio.get_running_loop()).__module__.split(".")[0]

        assert asyncio.run(probe()) == "uvloop"
    finally:
        asyncio.set_event_loop_policy(policy)


# ---------------------------------------------------------------------------
# Documented claims
#
# Three of these were wrong in a way no test could catch, because the claim
# lived in a doc and the code was fine: a cost recipe that always printed zero,
# a latency table that was 4-18x optimistic on the reranker, and a cache example
# the shipped embedder cannot produce. The docs are part of the contract, so the
# assertions live here rather than in a reviewer's memory.
# ---------------------------------------------------------------------------
DOCS = pathlib.Path(__file__).resolve().parents[2] / "docs"


def test_nested_request_context_hides_calls_from_the_outer_ledger() -> None:
    """Why an ambient ledger around ``query()`` can never work: ``query()``
    installs its own context, and the inner one *replaces* the contextvars, so
    the caller's ledger records nothing. Correct isolation — two concurrent
    requests must not share a budget — which is exactly why the documented
    recipe had to change rather than the code."""
    with new_request_context(request_id="outer", max_cost_usd=0.50) as (outer_trace, outer):
        with new_request_context(request_id="inner") as (_, inner):
            with timed("answer"):
                pass
            inner.record(Usage(model="m", cost_usd=0.01, calls=1), stage="answer")

        assert inner.total.calls == 1
        assert outer.total.calls == 0, "the outer ledger cannot see the inner request's calls"
        assert outer.report()["total_cost_usd"] == 0.0
        assert outer_trace == [], "nor its trace steps"


def test_cost_docs_read_the_bill_off_the_answer() -> None:
    """Both documented recipes were broken: ``pipeline.run()`` does not exist,
    and the ambient-ledger snippet printed ``{'total_cost_usd': 0.0, 'calls': 0}``
    for a query that made 46 real calls."""
    cost = (DOCS / "cost.md").read_text()
    performance = (DOCS / "performance.md").read_text()

    for name, text in (("cost.md", cost), ("performance.md", performance)):
        assert "pipeline.run(" not in text, f"{name} still documents a method that does not exist"
        assert "print(ledger.report())" not in text, f"{name} still reads the caller's ledger"
    assert 'answer.metadata["cost"]' in cost
    assert "answer.usage.cost_usd" in cost
    assert "answer.trace" in performance


def test_performance_docs_give_rerank_cost_per_count_and_size() -> None:
    """Measured on an M1 Pro: 50 candidates x 600 chars is ~690 ms, not the
    40-150 ms the table claimed, and at 2000 chars it is 2.3 s. A single number
    for a stage that scales with two inputs cannot be right for either."""
    performance = (DOCS / "performance.md").read_text()

    assert "40-150 ms" not in performance, "the old single-number rerank claim survives"
    assert "2-8 ms" not in performance, "the old single-number embed claim survives"
    assert "measured on one machine" in performance.lower()
    # The cost has to be given against both axes it depends on.
    assert "n=50" in performance and "chars per candidate" in performance
    assert "chunk_size" in performance and "max_chunk_size" in performance


def test_uvloop_claim_names_the_two_recipes_that_work() -> None:
    performance = (DOCS / "performance.md").read_text()
    assert "uvloop.run(" in performance
    assert "install_uvloop()" in performance and "asyncio.run(main())" in performance


def test_semantic_cache_settings_document_what_the_model_can_do() -> None:
    """The docstring offered "what is X" / "explain X" as a hit at 0.97. Measured
    with the shipped ``BAAI/bge-small-en-v1.5``, that pair is 0.9596 — a miss —
    while "over $500?" vs "under $500?" is 0.9924, a hit with the opposite
    meaning. Both numbers belong in the docstring, because the threshold is a
    property of the model and not of the word "similar"."""
    import inspect

    from ragorc.core.settings import CacheSettings

    source = inspect.getsource(CacheSettings)

    assert "hit rates are 20-40% on real traffic" not in source.lower()
    assert "single largest cost lever in a production" not in source.lower()
    assert "0.9596" in source, "the docstring must state the measured paraphrase score"
    assert "0.9924" in source, "and the wrong-answer pair that clears the threshold"
    # The default itself is unchanged: five other files quote 0.97, and the
    # measurement argues for raising it, never for lowering it.
    assert CacheSettings().semantic_threshold == 0.97

    # ADR-0007 is where the figure came from, and a retraction that leaves the
    # source standing is not a retraction — the number gets copied back out of it.
    adr = (DOCS / "adr" / "0007-cache-tiers.md").read_text().lower()
    assert "hit rates of 20-40% are typical" not in adr
    assert "the single largest cost lever in the system" not in adr
    assert "0.9924" in adr, "the ADR must carry the measurement that replaced the claim"


def test_late_chunking_docs_do_not_call_it_the_default() -> None:
    """The claim was corrected in README, ADR-0002 and architecture.md and left
    standing in two files, which is the failure mode a doc claim has: it is copied
    faster than it is retracted. Against the shipped resolver
    (``chunking_strategy_resolved chosen=early reason=no_better_option``) both were
    false — FastEmbed returns pooled vectors only, so the zero-dependency install
    gets ``early`` and an explicit ``late`` is downgraded to it.

    Asserted on normalized text rather than on the exact sentences, because the
    sentences will be rewritten and the claim must stay retracted: emphasis markers
    stripped so ``**the default**`` cannot hide the phrase, whitespace collapsed so
    a line wrap cannot either.
    """
    import re

    cost = (DOCS / "cost.md").read_text()
    embed = (DOCS / "modules" / "embed.md").read_text()

    def normalized(text: str) -> str:
        return " ".join(re.sub(r"[*`_]", "", text).lower().split())

    for name, text in (("cost.md", cost), ("modules/embed.md", embed)):
        flat = normalized(text)
        assert "late chunking is the default" not in flat, f"{name} still claims the old default"
        assert "not the default" in flat, f"{name} must say late chunking is not what you get"
    # And each must name what the default install *does* get, in the resolver's own
    # terms — "preferred, but not running" is only actionable if it says so.
    assert "resolves to early" in normalized(cost)
    assert "chosen=early" in embed


# ---------------------------------------------------------------------------
# Redaction: the credential quoted inside somebody else's prose
# ---------------------------------------------------------------------------
def test_redact_identifiers_strips_operator_identity_from_free_text() -> None:
    """Key-name filtering cannot see a secret that arrives inside a value.

    A provider's 4xx body is a sentence, and OpenRouter's carries a
    key-management URL with the key id in it plus the account user id. That body
    is attached to the raised error and travels to the abstention reason, the
    HTTP error detail and the CLI, so it reaches whoever called the API.
    """
    body = (
        '{"error":{"message":"Insufficient credit. To purchase more, visit '
        'https://openrouter.ai/workspaces/default/keys/0f1e2d3c4b5a69788796a5b4c3d2e1f0"},'
        '"user_id":"user_EXAMPLEaccountEXAMPLE00000"}'
    )
    out = redact_identifiers(body)
    assert "0f1e2d3c4b5a69788796a5b4c3d2e1f0" not in out
    assert "user_EXAMPLEaccountEXAMPLE00000" not in out
    assert "Insufficient credit" in out, "the useful sentence must survive"


@pytest.mark.parametrize(
    "text",
    [
        "Authorization: Bearer sk-or-v1-c985a601a9ecbf64a186ea7b0d695b3abc3340db",
        "key=sk-or-v1-c985a601a9ecbf64a186ea7b0d695b3abc3340db failed",
        "token ghp_16CharactersAtLeastHere00",
    ],
)
def test_redact_identifiers_removes_credential_shapes(text: str) -> None:
    out = redact_identifiers(text)
    assert "c985a601" not in out
    assert "16CharactersAtLeastHere00" not in out


def test_redact_identifiers_leaves_ordinary_text_alone() -> None:
    """It runs on every log line, so it must not mangle normal messages."""
    for text in (
        'relation "orders" does not exist',
        "retrieved 12 chunks in 43ms",
        "SELECT id FROM orders WHERE sku = 'sk-9'",
    ):
        assert redact_identifiers(text) == text


def test_log_redactor_scrubs_values_not_only_secret_keys() -> None:
    """`body` is not a credential-shaped key, and that is the whole problem."""
    event = _redact_secrets(
        None,
        "info",
        {
            "event": "provider_error",
            "body": "visit https://openrouter.ai/workspaces/default/keys/abcdef0123456789",
            "api_key": "sk-or-v1-should-be-masked-by-the-key-rule",
        },
    )
    assert "abcdef0123456789" not in event["body"]
    assert event["api_key"].endswith("***"), "the key-name rule must still apply"
    assert event["event"] == "provider_error"
