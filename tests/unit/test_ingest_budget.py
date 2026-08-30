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
def test_an_ingest_is_bounded_by_spend_not_by_call_count() -> None:
    """Spend is the quantity that does not scale with the corpus, so it is the one
    with a default. Call and token counts do scale and have no defensible number.

    The first version of this left all three at None and justified it with
    "RAPTOR forecasts and refuses an over-budget build before the first call".
    That is false: `RaptorIndexer._check_budget` returns early when
    `ledger.max_calls is None`, so the cited compensating control does not run and
    POST /ingest had no ceiling of any kind.
    """
    cost = _settings().cost
    assert cost.max_cost_per_ingest_usd == 10.0, "an ingest may not be unbounded in spend"
    assert cost.max_cost_per_ingest_usd > (cost.max_cost_per_query_usd or 0), (
        "an ingest ceiling below the per-query one would truncate corpora again"
    )
    assert cost.max_llm_calls_per_ingest is None
    assert cost.max_tokens_per_ingest is None


def test_the_raptor_forecast_is_not_what_bounds_an_ingest() -> None:
    """Asserted because a docstring claimed the opposite. A guard that returns
    early on the default configuration is not a compensating control."""
    import inspect

    from ragorc.index.raptor import RaptorIndexer

    source = inspect.getsource(RaptorIndexer._check_budget)
    assert "if ledger.max_calls is None:\n            return" in source
    assert "not what bounds an ingest" in source, (
        "the early return is undocumented, so it will be cited as a guard again"
    )


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
    service.settings = _settings(  # type: ignore[misc]
        cost={
            "max_llm_calls_per_ingest": 5000,
            "max_cost_per_ingest_usd": 25.0,
            "max_tokens_per_ingest": 9_000_000,
        }
    )

    with service._ingest_context("rid") as ledger:
        # All three, because the first version asserted only `max_calls` — so
        # swapping the cost and token ceilings back to their per-query values
        # (0.50 USD, 200k tokens) restored the corpus truncation along those two
        # axes and the suite did not notice.
        assert ledger.max_calls == 5000, f"installed the query ceiling: {ledger.max_calls}"
        assert ledger.max_cost_usd == 25.0, f"cost ceiling is {ledger.max_cost_usd}"
        assert ledger.max_tokens == 9_000_000, f"token ceiling is {ledger.max_tokens}"
        assert current_ledger() is ledger


async def test_the_query_context_still_reads_the_query_ceiling() -> None:
    from ragorc.server.app import RagService

    service = object.__new__(RagService)
    service.settings = _settings(  # type: ignore[misc]
        cost={"max_llm_calls_per_query": 7, "max_llm_calls_per_ingest": 5000}
    )

    with service._request_context("rid") as ledger:
        assert ledger.max_calls == 7
        assert ledger.max_cost_usd == 0.5
        assert ledger.max_tokens == 200_000


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


# ---------------------------------------------------------------------------
# ...and survives its caller
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("stage", ["summary", "dense_x"])
async def test_the_stop_signal_escapes_the_gather(stage: str) -> None:
    """The half the first fix missed, and the shape this round is named for.

    `_summarize` and `_decompose` re-raise correctly and the original tests
    asserted on exactly those two methods — but their only caller runs them
    through `map_concurrent(..., return_exceptions=True)` and then reclassifies
    any BaseException as one chunk's bad luck. Measured before this fix:

        build() returned 5 units, usage.calls=0, NO EXCEPTION
        metadata[0]: representation=source summary_skipped=error

    which is verbatim the outcome the re-raise was added to prevent.
    """
    llm = _ExhaustedLLM(BudgetExceeded("LLM call budget exhausted", calls=40, limit=40))
    sources = [_source(f"c{i}") for i in range(5)]

    if stage == "summary":
        from ragorc.index.multirep.summary import SummaryIndexer

        indexer: Any = SummaryIndexer(llm, settings=_settings())
    else:
        from ragorc.index.multirep.dense_x import PropositionIndexer

        indexer = PropositionIndexer(llm, settings=_settings())

    with pytest.raises(BudgetExceeded):
        await indexer.build(sources)


@pytest.mark.parametrize("stage", ["summary", "dense_x"])
async def test_build_still_degrades_on_an_ordinary_failure(stage: str) -> None:
    """The distinction the re-raise must preserve at this level too."""
    llm = _ExhaustedLLM(LLMError("content filter"))
    sources = [_source(f"c{i}") for i in range(3)]

    if stage == "summary":
        from ragorc.index.multirep.summary import SummaryIndexer

        indexer: Any = SummaryIndexer(llm, settings=_settings())
    else:
        from ragorc.index.multirep.dense_x import PropositionIndexer

        indexer = PropositionIndexer(llm, settings=_settings())

    units, usage = await indexer.build(sources)
    assert len(units) == 3, "chunks were dropped from the corpus"
    assert usage.calls == 0


async def test_the_ingest_pipeline_does_not_downgrade_a_ceiling_to_a_warning() -> None:
    """`_enrich` wraps every stage in `except Exception` and turns it into
    `report.warnings.append("<stage> disabled: ...")`. BudgetExceeded is an
    Exception, so even RAPTOR — which raises it correctly — came back as a warning
    string on a field most callers never read, and the ingest reported success:

        warnings: ['raptor stage disabled: LLM call budget exhausted (calls=2 limit=2)']

    Character for character the symptom this round set out to remove. Driven
    through the real `_enrich`, because the defect is which handler catches it.
    """
    from ragorc.core.models import Document
    from ragorc.index.pipeline import IngestPipeline, IngestReport, _Plugin

    class _Ceiling:
        async def run(self, **kw: Any) -> Any:
            raise BudgetExceeded("LLM call budget exhausted", calls=2, limit=2)

    class _Ordinary:
        async def run(self, **kw: Any) -> Any:
            raise RuntimeError("the clustering library is not installed")

    def _pipeline(stage: Any) -> Any:
        pipeline = object.__new__(IngestPipeline)
        pipeline._stages = [(_Plugin(label="raptor", modules=(), factories=()), stage)]
        pipeline.vector = None
        pipeline.relational = None
        return pipeline

    document = Document(id="d1", content="x")
    chunk = Chunk(id="c1", content="x", document_id="d1")

    report = IngestReport()
    with pytest.raises(BudgetExceeded):
        await _pipeline(_Ceiling())._enrich(document, [chunk], report)
    assert not report.warnings, f"the ceiling was reported instead of raised: {report.warnings}"

    # And an ordinary stage failure still degrades, which is what the blanket
    # handler is for — the leaves are already written and losing an enrichment is
    # recoverable where losing them is not.
    report = IngestReport()
    await _pipeline(_Ordinary())._enrich(document, [chunk], report)
    assert report.warnings and "raptor stage disabled" in report.warnings[0]


def test_the_ledger_counts_each_ingest_call_once() -> None:
    """`OpenRouterLLM._record` already recorded every call; `_charge` recorded the
    stage aggregate again, so `ledger.total.calls` was 12 for 6 real calls:

        ledger.by_stage: {'summary_index': 6, 'ingest.multirep': 6}

    Harmless while an ingest borrowed the query ledger, a factor-of-two error in
    the operator's ceiling now that the ingest ceilings are load-bearing.

    Asserted on the ledger, not on the source: a grep for `ledger.record` is
    defeated by any other spelling, and this one was.
    """
    from ragorc.core.telemetry import new_request_context
    from ragorc.index.pipeline import IngestPipeline, IngestReport

    stage_usage = Usage(model="m", calls=6, prompt_tokens=100, cost_usd=0.06)
    with new_request_context(
        request_id="r", max_calls=None, max_tokens=None, max_cost_usd=None, trace=False
    ) as (_trace, ledger):
        # What the LLM client already recorded while making the calls.
        ledger.record(stage_usage, stage="summary_index")
        report = IngestReport()
        IngestPipeline._charge(None, report, stage_usage, "multirep")  # type: ignore[arg-type]

        assert ledger.total.calls == 6, f"each call counted {ledger.total.calls / 6:g} times"
        assert report.usage.calls == 6, "the report must still be charged"
