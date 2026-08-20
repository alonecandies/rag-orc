"""Query decomposition: split a compound question into answerable parts.

Two distinct patterns, and conflating them is a common mistake.

**Parallel decomposition** — the sub-questions are independent ("compare X and Y"
becomes "what is X", "what is Y"). Retrieve for all of them concurrently and
answer once from the union. Cheap, and the default.

**Sequential decomposition with answer chaining** — a later sub-question cannot be
*asked* until an earlier one is answered ("who directed the highest-grossing film
of 2019?" needs the film before it can ask about the director). This requires a
serial retrieve-answer loop, which costs one retrieval and one generation per hop,
and is why :class:`RecursiveDecomposer` is separate and opt-in.

Both respect ``is_decomposable=false``. Forcing a split on an already-atomic
question produces near-duplicate sub-questions and multiplies cost for nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import structlog

from ragorc.core.models import Query, ScoredChunk, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import DecompositionOutput
from ragorc.core.settings import Settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.translate.base import BaseTranslator

log = structlog.get_logger(__name__)

__all__ = ["DecompositionTranslator", "RecursiveDecomposer", "SubAnswer"]


@register("translator", "decomposition", "decompose")
class DecompositionTranslator(BaseTranslator):
    name = "decomposition"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        max_sub: int = 4,
    ) -> None:
        super().__init__(llm, settings, router=router)
        self.max_sub = max_sub

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        prompt = get_prompt("decomposition")
        result, usage = await self.llm.structured(
            prompt.render(question=query.text, max_sub=self.max_sub),
            DecompositionOutput,
            system=prompt.system,
            model=self.router.model_for(Task.DECOMPOSE),
            stage="decomposition",
        )
        if not result.is_decomposable:
            log.debug("decomposition_skipped", reason="atomic_question")
            return query, usage
        subs = [s.strip() for s in result.sub_questions[: self.max_sub] if s.strip()]
        if not subs:
            return query, usage
        out = self._extend(query, subs, sub_questions=subs, decomposed=True)
        log.debug("decomposed", count=len(subs))
        return out, usage


@dataclass(slots=True)
class SubAnswer:
    question: str
    answer: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


class RecursiveDecomposer:
    """Sequential decomposition with answer chaining.

    Takes ``retrieve_fn`` and ``answer_fn`` as callables rather than a retriever
    and a generator, so this class stays decoupled from both and is trivially
    testable with stubs. The prior question-answer pairs are threaded into each
    subsequent prompt, which is what makes the chain work: sub-question 2 is
    only answerable once sub-answer 1 exists.

    Cost is linear in the number of sub-questions and cannot be parallelized by
    construction, so ``max_steps`` is a hard cap rather than a suggestion.
    """

    name = "recursive_decomposition"

    def __init__(
        self,
        translator: DecompositionTranslator,
        *,
        max_steps: int = 4,
    ) -> None:
        self.translator = translator
        self.max_steps = max_steps

    async def run(
        self,
        query: Query,
        retrieve_fn: Callable[[Query], Awaitable[Sequence[ScoredChunk]]],
        answer_fn: Callable[
            [str, Sequence[ScoredChunk], Sequence[SubAnswer]], Awaitable[tuple[str, Usage]]
        ],
    ) -> tuple[list[SubAnswer], Usage]:
        expanded, usage = await self.translator.translate(query)
        subs: list[str] = list(expanded.metadata.get("sub_questions") or [])
        if not subs:
            # Not decomposable: answer the original question as a single step, so
            # the caller gets a uniform result shape either way.
            subs = [query.text]

        usages: list[Usage] = [usage]
        answers: list[SubAnswer] = []
        for step, question in enumerate(subs[: self.max_steps]):
            sub_query = Query(
                text=question,
                original=query.original,
                filters=dict(query.filters),
                top_k=query.top_k,
                tenant_id=query.tenant_id,
                metadata={"decomposition_step": step, "parent_question": query.text},
            )
            chunks = list(await retrieve_fn(sub_query))
            text, step_usage = await answer_fn(question, chunks, answers)
            usages.append(step_usage)
            answers.append(
                SubAnswer(question=question, answer=text, chunks=chunks, usage=step_usage)
            )
            log.debug("sub_answered", step=step, question=question[:70], chunks=len(chunks))
        return answers, Usage.sum(usages)

    @staticmethod
    def render_chain(answers: Sequence[SubAnswer]) -> str:
        """Format prior sub-answers for the next prompt."""
        if not answers:
            return "(none yet)"
        return "\n\n".join(f"Q: {a.question}\nA: {a.answer}" for a in answers)
