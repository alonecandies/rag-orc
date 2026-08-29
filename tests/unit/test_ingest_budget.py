"""What bounds an ingest, and what a spent budget is allowed to do quietly.

The HTTP ingest route opened a *per-query* cost ledger and ran the whole corpus
inside it. With RAPTOR on, ``cost.max_llm_calls_per_query`` (40) stopped the
enrichment 40 documents in and the response still reported success:

    documents indexed: 60
    documents that got a RAPTOR summary: 40 of 60
    warnings: ['raptor stage disabled: LLM call budget exhausted (calls=40 limit=40)']

RAPTOR at least says something. The two multi-representation stages caught the
same ``BudgetExceeded`` per chunk and returned the source text as its own unit,
so 680 of 720 chunks were silently indexed unenriched with ``usage.calls == 0``
and nothing on the report at all.

A per-query ceiling is a liveness bound on one request. Applied to a corpus it is
a silent truncation, and the truncated part is exactly the expensive part the
operator turned on.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.errors import BudgetExceeded, LLMError
from ragorc.core.models import Chunk, Usage
from ragorc.core.settings import Settings


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {"llm": {"api_key": "k"}, "cache": {"enabled": False}}
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# The ceilings exist and are separate
# ---------------------------------------------------------------------------
def test_an_ingest_is_unbounded_by_default() -> None:
    """Bounded by the corpus, not by a request ceiling. RAPTOR forecasts and
    refuses an over-budget build before the first call, so an arbitrary number
    here would only move the same silent truncation to a different corpus size."""
    cost = _settings().cost
    assert cost.max_llm_calls_per_ingest is None
    assert cost.max_cost_per_ingest_usd is None
    assert cost.max_tokens_per_ingest is None


def test_the_query_ceiling_is_unchanged() -> None:
    """The per-query bound is a real liveness property and must keep its value."""
    assert _settings().cost.max_llm_calls_per_query == 40


def test_the_ingest_route_does_not_use_the_query_ledger() -> None:
    """The call site, which is the whole defect: `_ingest_context` existing and
    `ingest` still opening `_request_context` would read exactly the same."""
    import inspect

    from ragorc.server.app import RagService

    source = inspect.getsource(RagService.ingest)
    assert "self._ingest_context(request_id)" in source
    assert "self._request_context" not in source


def test_the_query_route_still_uses_the_query_ledger() -> None:
    """The fix must not lift the per-query ceiling as a side effect."""
    import inspect

    from ragorc.server.app import RagService

    assert "self._request_context(request_id)" in inspect.getsource(RagService.query)


async def test_the_ingest_context_reads_the_ingest_ceilings() -> None:
    """Behaviour, not just the argument: a context that accepted the settings and
    installed the query numbers anyway would satisfy a source check."""
    from ragorc.core.telemetry import current_ledger
    from ragorc.server.app import RagService

    service = object.__new__(RagService)
    service.settings = _settings(cost={"max_llm_calls_per_ingest": 5000})  # type: ignore[misc]

    with service._ingest_context("rid") as ledger:
        assert ledger.max_calls == 5000, f"installed the query ceiling: {ledger.max_calls}"
        assert current_ledger() is ledger


async def test_the_query_context_still_reads_the_query_ceiling() -> None:
    from ragorc.server.app import RagService

    service = object.__new__(RagService)
    service.settings = _settings(  # type: ignore[misc]
        cost={"max_llm_calls_per_query": 7, "max_llm_calls_per_ingest": 5000}
    )

    with service._request_context("rid") as ledger:
        assert ledger.max_calls == 7


# ---------------------------------------------------------------------------
# A spent budget stops the stage
# ---------------------------------------------------------------------------
class _ExhaustedLLM:
    """An LLM whose budget is already gone, as the ledger reports it."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def structured(self, *a: Any, **kw: Any) -> Any:
        self.calls += 1
        raise self.error

    async def complete(self, *a: Any, **kw: Any) -> tuple[str, Usage]:
        self.calls += 1
        raise self.error


def _source(cid: str = "c0") -> Chunk:
    """Long enough to be worth summarizing.

    `_summarize` returns early below `_SUMMARIZE_FLOOR_TOKENS` without calling the
    model at all, so a short fixture makes a stage that never reached the LLM
    indistinguishable from one that reached it and swallowed the budget error.
    """
    return Chunk(
        id=cid,
        content=(
            "Refunds are available for thirty days after delivery of the item. "
            "After that window the item is yours to keep and no refund is issued. "
            "Shipping costs are never refundable in either direction, and the "
            "original packaging must be intact for the return to be accepted at "
            "the depot. Items marked as final sale are excluded from this policy "
            "entirely, as are perishable goods, personalised engravings and any "
            "product whose security seal has been broken by the customer. Where a "
            "refund is approved, funds are returned to the original payment method "
            "within five to seven working days of the depot confirming receipt."
        ),
        document_id="d1",
    )


async def test_a_spent_budget_stops_the_summary_stage() -> None:
    """Not one chunk's bad luck: every remaining call would raise too. Swallowing
    it per chunk is how 680 of 720 chunks got indexed as their own source text
    with `usage.calls == 0`."""
    from ragorc.index.multirep.summary import SummaryIndexer

    llm = _ExhaustedLLM(BudgetExceeded("LLM call budget exhausted", calls=40, limit=40))
    indexer = SummaryIndexer(llm, settings=_settings())

    with pytest.raises(BudgetExceeded):
        await indexer._summarize(_source())


async def test_a_spent_budget_stops_the_proposition_stage() -> None:
    from ragorc.index.multirep.dense_x import PropositionIndexer

    llm = _ExhaustedLLM(BudgetExceeded("LLM call budget exhausted", calls=40, limit=40))
    indexer = PropositionIndexer(llm, settings=_settings())

    with pytest.raises(BudgetExceeded):
        await indexer._decompose(_source())


@pytest.mark.parametrize("stage", ["summary", "dense_x"])
async def test_an_ordinary_model_failure_still_degrades(stage: str) -> None:
    """The behaviour the re-raise must not swallow. One chunk the model refused
    is not a reason to fail an ingest — the chunk still retrieves as its own
    text, which is what `_as_source_unit` is for."""
    llm = _ExhaustedLLM(LLMError("content filter"))
    settings = _settings()

    if stage == "summary":
        from ragorc.index.multirep.summary import SummaryIndexer

        unit, usage = await SummaryIndexer(llm, settings=settings)._summarize(_source())
        units = [unit]
    else:
        from ragorc.index.multirep.dense_x import PropositionIndexer

        units, usage = await PropositionIndexer(llm, settings=settings)._decompose(_source())

    assert units, "the chunk was dropped from the corpus entirely"
    assert units[0].metadata.get("summary_skipped") == "llm_error" or units[0].metadata.get(
        "dense_x_skipped"
    ) == "llm_error", units[0].metadata
    assert usage.calls == 0


def test_raptor_already_treated_it_this_way() -> None:
    """The precedent the two stages diverged from, asserted so the three cannot
    drift apart again."""
    import inspect

    from ragorc.index.raptor import RaptorIndexer

    source = inspect.getsource(RaptorIndexer._summarize_level)
    assert "isinstance(result, BudgetExceeded)" in source
    assert "raise result" in source
