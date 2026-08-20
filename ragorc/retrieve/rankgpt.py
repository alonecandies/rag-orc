"""RankGPT: listwise reranking with an LLM, and the sliding window that makes it
affordable.

Why listwise at all
-------------------
A cross-encoder is *pointwise*: it scores one ``(query, passage)`` pair at a
time, so its logits are only comparable within a query and it can never express
"passage 3 is redundant given passage 1" or "passage 7 answers the question
directly while 2 merely mentions the topic". Ranking is a comparative judgement,
and a pointwise scorer is structurally unable to make one — it has never seen the
alternatives.

A listwise reranker shows the model the whole candidate list and asks for a
permutation. That is the question users actually have, the model can weigh the
candidates against each other, and on the reasoning-heavy queries where a
cross-encoder plateaus this is where the remaining precision lives. It is also
100-1000x more expensive per query, which is why it is opt-in and why it runs on
a tier from :class:`~ragorc.llm.router.Task` rather than the frontier model.

Why a sliding window, and why it runs backwards
-----------------------------------------------
A listwise call needs every candidate *in the prompt*. Fifty passages of 400
tokens is 20k tokens of context in which the model must hold fifty items in
working memory — and long-context attention degrades well before the context
limit is reached (the same "lost in the middle" effect that shapes context
packing applies to a numbered passage list). The paper's answer is to rank a
window of ``rankgpt_window`` passages at a time and slide it by
``rankgpt_step``, writing each window's permutation back in place. The overlap
(``window - step``) is what carries information between windows: passages that
survived one window are re-judged against the next batch.

The direction is the subtle part. Windows are processed **from the end of the
list toward the front**. Within one pass a passage can only travel as far as its
window reaches, so starting at the tail lets a strong candidate at position 49
move up by ``step`` per window and keep climbing — each pass bubbles the best
candidates toward the front, and every later window therefore re-ranks a head
that the windows behind it have already improved. Front-to-back does the
opposite: it fixes the head first, using only the information in the first
window, and never revisits it, so a strong passage in the tail can never reach
the top no matter how many calls are spent.

Calls are ``ceil((n - window) / step) + 1`` and are strictly **sequential**: each
window's input is the previous window's output, so there is nothing to fan out.
At the defaults (50 candidates, window 10, step 5) that is 9 small calls instead
of one call that would not fit.

Why the repair code is not optional
-----------------------------------
The output is a permutation produced by a language model, and a malformed
permutation is the single most common way RankGPT fails in production: a missing
index, a duplicated index, an index past the end of the window, a 0-based list
when the prompt numbered from 1, or a truncated list. Treating that as an error
would make the reranker *less* reliable than not reranking at all. So the
permutation is repaired instead: valid unique entries keep the model's order, and
everything the model failed to mention is appended in its original relative
order. No passage is ever dropped, no exception ever escapes, and every repair is
logged so the failure rate of the configured model is visible rather than
inferred.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import structlog

from ragorc.core.models import Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import RankOrder
from ragorc.core.settings import Settings
from ragorc.core.tokens import truncate_to_tokens
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.retrieve.rerank import BaseReranker

log = structlog.get_logger(__name__)

__all__ = ["RankGPTReranker", "repair_permutation", "sliding_windows"]

_MIN_PASSAGE_TOKENS = 96
"""Floor for the per-passage clip. Below roughly this length a passage carries
too little to be ranked against its neighbours, so clipping further would trade
a context problem for a ranking problem."""


def sliding_windows(total: int, window: int, step: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, end)`` slices from the end of the list toward the front.

    With ``total=50, window=10, step=5``: ``(40,50), (35,45), (30,40) ... (0,10)``.
    The final window is clamped to start at 0 so the head is always ranked, even
    when ``(total - window)`` is not a multiple of ``step`` — otherwise the first
    few positions, the ones that matter most, would be the only ones never
    reconsidered. Clamping can make that last window narrower than ``window``,
    which is the harmless direction: a smaller final call, never a gap.
    """
    if total <= 0:
        return
    if total <= window:
        yield (0, total)  # everything fits in one call; no sliding needed
        return
    end = total
    while True:
        start = max(end - window, 0)
        yield (start, end)
        if start == 0:
            return
        end -= step


def repair_permutation(raw: Sequence[int], size: int) -> tuple[list[int], dict[str, int]]:
    """Coerce model output into a genuine permutation of ``range(size)``.

    Policy: keep the model's ordering for every entry that is in range and not
    already used, then append whatever it omitted in the original order. That
    preserves the judgement the model did make while guaranteeing the output is
    a permutation, which is what the caller's slice assignment requires.

    Also tolerates a 0-based reply. The prompt numbers passages from 1, but a
    model that answers ``[0..size-1]`` is expressing a perfectly good ranking in
    the wrong dialect; treating it as 1-based would drop its first choice and
    shift every other one by a position. The heuristic only fires when the reply
    cannot be 1-based (it contains 0) and fits 0-based bounds exactly.

    Returns the permutation plus counts of what was wrong, for logging.
    """
    values = [int(v) for v in raw]
    stats = {"returned": len(values), "duplicates": 0, "out_of_range": 0, "missing": 0}
    zero_based = bool(values) and min(values) == 0 and max(values) <= size - 1
    offset = 0 if zero_based else 1
    stats["zero_based"] = int(zero_based)

    order: list[int] = []
    seen: set[int] = set()
    for value in values:
        index = value - offset
        if not 0 <= index < size:
            stats["out_of_range"] += 1
            continue
        if index in seen:
            stats["duplicates"] += 1
            continue
        seen.add(index)
        order.append(index)
    if len(order) < size:
        remainder = [i for i in range(size) if i not in seen]
        stats["missing"] = len(remainder)
        order.extend(remainder)
    return order, stats


@register("reranker", "rankgpt")
class RankGPTReranker(BaseReranker):
    """Listwise LLM reranking over a sliding window.

    Callers should reach this through
    :meth:`~ragorc.retrieve.rerank.BaseReranker.rerank_with_usage`: it is the one
    reranker in the package that spends money, and discarding its
    :class:`~ragorc.core.models.Usage` under-states a query's bill by the largest
    single retrieval-side item.
    """

    name = "rankgpt"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        super().__init__(settings)
        self.llm = llm
        self.router = router or ModelRouter(self.settings.llm)
        self.model_name = self.router.model_for(Task.RANK_GPT)

    async def _order(
        self,
        question: str,
        texts: list[str],
        ids: list[str] | None,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float]], Usage]:
        total = len(texts)
        if total < 2:
            # Nothing to compare. Spending a model call to rank one passage is
            # pure cost, and the answer is known.
            return self._as_pairs(list(range(total)), top_k), Usage()

        cfg = self.settings.retrieval
        # A window of one cannot reorder anything, and a step wider than the
        # window would leave gaps of passages no call ever looks at.
        window = max(min(cfg.rankgpt_window, total), 2)
        step = max(min(cfg.rankgpt_step, window), 1)

        order = list(range(total))
        usages: list[Usage] = []
        calls = 0
        for start, end in sliding_windows(total, window, step):
            selected = order[start:end]
            permutation, usage = await self._rank_window(
                question, [texts[i] for i in selected], window_index=calls
            )
            usages.append(usage)
            calls += 1
            # Write the permutation back *in place*: the next window overlaps
            # this one, so it must see the improved ordering, not the original.
            order[start:end] = [selected[j] for j in permutation]

        log.debug(
            "rankgpt_done",
            candidates=total,
            window=window,
            step=step,
            calls=calls,
            model=self.model_name,
        )
        pairs = self._as_pairs(order, top_k)
        return pairs, Usage.sum(usages)

    # -- internals ---------------------------------------------------------
    def _as_pairs(self, order: Sequence[int], top_k: int | None) -> list[tuple[int, float]]:
        """Turn positions into scores.

        A listwise reranker produces an *ordering*, not a calibrated relevance —
        there is no score to report. The rank is therefore mapped linearly onto
        ``(0, 1]``, which keeps the library's higher-is-better invariant and
        makes ``relative_score_cutoff`` behave as "keep the best fraction",
        the only sensible reading of a relative threshold over positions.
        """
        total = len(order)
        keep = total if top_k is None else max(min(top_k, total), 1)
        return [(index, 1.0 - rank / total) for rank, index in enumerate(order[:keep])]

    async def _rank_window(
        self, question: str, passages: Sequence[str], *, window_index: int
    ) -> tuple[list[int], Usage]:
        """One listwise call. Never raises; the worst case is the identity
        permutation, which leaves this window's slice exactly as it was."""
        size = len(passages)
        prompt = get_prompt("rank_gpt")
        rendered = prompt.render(
            question=question,
            passages=self._render(passages),
            n=size,
        )
        try:
            result, usage = await self.llm.structured(
                rendered,
                RankOrder,
                system=prompt.system,
                model=self.model_name,
                stage="rank_gpt",
            )
        except Exception as exc:  # noqa: BLE001 - one bad window must not fail the query
            log.warning(
                "rankgpt_window_failed",
                window=window_index,
                passages=size,
                model=self.model_name,
                error=str(exc)[:200],
                error_type=type(exc).__name__,
            )
            return list(range(size)), Usage()

        order, stats = repair_permutation(result.order, size)
        if stats["duplicates"] or stats["out_of_range"] or stats["missing"] or stats["zero_based"]:
            log.warning(
                "rankgpt_permutation_repaired",
                window=window_index,
                expected=size,
                model=self.model_name,
                **stats,
            )
        return order, usage

    def _render(self, passages: Sequence[str]) -> str:
        """Number the passages from 1 and clip each to its share of the window.

        The clip is a guard, not a compression step: parent expansion or a table
        chunk can produce one passage of tens of thousands of tokens, and without
        a per-passage ceiling that single passage evicts the other nine from the
        window — the reranker would then be ranking a list it cannot see.
        """
        share = int(self.settings.llm.context_window * 0.6 / max(len(passages), 1))
        limit = max(share, _MIN_PASSAGE_TOKENS)
        return "\n\n".join(
            f"[{i}] {truncate_to_tokens(text.strip(), limit)}"
            for i, text in enumerate(passages, start=1)
        )
