"""Token budgeting.

Context overflow is not an edge case; it is the default state of a RAG pipeline
that retrieves anything useful. 50 candidate chunks at 512 tokens is 25k tokens
before the query, the system prompt, the graph evidence or the SQL results are
counted.

The failure mode when nobody budgets is specific and bad: the provider truncates
the *end* of the request, which is where the question usually sits, so the model
answers a question it never saw — or it has no room left to reply and returns an
empty completion.

So the budget is allocated top-down and reserved *before* packing:

    total window
      - safety margin (tokenizer drift between our count and the provider's)
      - output reservation (the answer needs room to exist)
      - system prompt
      - the question
      = what retrieved context may consume

and that remainder is divided between sources by share, so no single store can
crowd the others out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import structlog

from ragorc.core.models import ScoredChunk
from ragorc.core.settings import Settings, get_settings
from ragorc.core.tokens import TokenBudget, count_tokens, count_tokens_batch

log = structlog.get_logger(__name__)

__all__ = ["DEFAULT_SHARES", "BudgetPlan", "ContextBudgeter"]

#: How the context window is divided when several stores contribute.
#: Vector text is the bulk of the evidence; SQL rows and graph paths are dense
#: and small, so they need proportionally little room to be useful.
DEFAULT_SHARES: dict[str, float] = {
    "vector": 0.55,
    "graph": 0.20,
    "relational": 0.10,
    "summary": 0.10,
    "web": 0.05,
}


@dataclass(slots=True)
class BudgetPlan:
    """A concrete allocation for one request."""

    budget: TokenBudget
    per_source: dict[str, int] = field(default_factory=dict)
    """Budget shares per retrieval source. **Computed on every request and read
    by nothing.**

    `plan()` fills it from `DEFAULT_SHARES` and `to_dict()` reports it, but no
    packer, retriever or generator consults it, so one store can still take the
    whole context window. Kept rather than deleted because enforcing it is a
    behaviour change — it would start dropping evidence that is currently
    packed — and that is a decision for the operator, not a tidy-up. Tracked in
    docs/internal/OPEN-ITEMS.md."""
    context_tokens: int = 0
    overflow: bool = False
    dropped_chunks: int = 0
    strategy: str = "fit"
    """``fit`` (everything fit), ``truncate`` (tail dropped), or ``summarize``
    (compressed to fit)."""

    def report(self) -> dict[str, object]:
        return {
            "window": self.budget.total,
            "available_context": self.budget.available_context,
            "used": self.context_tokens,
            "per_source": self.per_source,
            "overflow": self.overflow,
            "dropped": self.dropped_chunks,
            "strategy": self.strategy,
        }


class ContextBudgeter:
    """Computes the budget and decides how to respond to overflow."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()

    def plan(
        self,
        *,
        system_prompt: str = "",
        question: str = "",
        shares: dict[str, float] | None = None,
        window: int | None = None,
    ) -> BudgetPlan:
        gen = self.settings.generation
        budget = TokenBudget(
            total=window or self.settings.llm.context_window,
            reserved_output=gen.reserved_output_tokens,
            reserved_system=count_tokens(system_prompt) if system_prompt else 0,
            reserved_query=count_tokens(question) if question else 0,
        )
        plan = BudgetPlan(budget=budget)
        plan.per_source = budget.split(shares or DEFAULT_SHARES)
        return plan

    def measure(self, chunks: Sequence[ScoredChunk]) -> list[int]:
        """Token counts, computed in one batched call.

        Counting 50 chunks individually crosses the Python/Rust boundary 50
        times; ``encode_batch`` does it once and releases the GIL while it works.
        Cached on the chunk so a later stage does not recount.
        """
        missing = [i for i, c in enumerate(chunks) if c.chunk.token_count is None]
        if missing:
            counts = count_tokens_batch([chunks[i].chunk.content for i in missing])
            for i, n in zip(missing, counts, strict=True):
                chunks[i].chunk.token_count = n
        return [c.chunk.token_count or 0 for c in chunks]

    def _total(self, chunks: Sequence[ScoredChunk], overhead: Sequence[int] | None) -> int:
        """Bodies plus the framing the packer will add around them.

        ``overhead`` comes from :meth:`~ragorc.context.pack.ContextPacker.overhead`,
        which measures it exactly. Omitted, this prices bodies alone — which is
        what it always did, and why a set that overflowed once framed was
        reported as fitting.
        """
        total = sum(self.measure(chunks))
        if overhead:
            total += sum(overhead[: len(chunks)])
        return total

    def fits(
        self,
        chunks: Sequence[ScoredChunk],
        plan: BudgetPlan,
        *,
        overhead: Sequence[int] | None = None,
    ) -> bool:
        return self._total(chunks, overhead) <= plan.budget.available_context

    def decide_strategy(
        self,
        chunks: Sequence[ScoredChunk],
        plan: BudgetPlan,
        *,
        overhead: Sequence[int] | None = None,
    ) -> str:
        """Choose how to handle overflow.

        Truncating is free and loses the tail; summarizing costs LLM calls and
        loses precision. The rule: if the overflow is small, drop the weakest
        chunks — they were ranked last for a reason. If the overflow is large,
        the tail is too much evidence to discard, so compress instead.
        """
        total = self._total(chunks, overhead)
        available = plan.budget.available_context
        if total <= available:
            return "fit"
        ratio = total / max(available, 1)
        if ratio <= 1.6:
            return "truncate"
        return "summarize"
