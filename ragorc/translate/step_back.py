"""Step-back prompting: ask the general question before the specific one.

The failure this fixes is subtle. "Which team did Messi play for in 2005?" is a
narrow question, and the passage that answers it is almost certainly a broad one —
a career history that happens to contain the 2005 line. Retrieval on the narrow
question competes against every other narrow mention of 2005; retrieval on
"What is Messi's club career history?" lands on the passage that contains the
answer.

So the step-back question is not a replacement, it is an *addition*: the specific
question stays in the variant set because sometimes the corpus really does contain
a sentence answering it directly. The step-back question is also recorded in
metadata, so the generator can place the general context first — background before
specifics reads better and grounds better.

The failure mode to avoid is over-generalizing until the topic is lost
("What is football?"), which the prompt explicitly guards against.
"""

from __future__ import annotations

import structlog

from ragorc.core.models import Query, Usage
from ragorc.core.registry import register
from ragorc.core.schemas import StepBackOutput
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import Task
from ragorc.translate.base import BaseTranslator

log = structlog.get_logger(__name__)

__all__ = ["StepBackTranslator"]


@register("translator", "step_back", "stepback")
class StepBackTranslator(BaseTranslator):
    name = "step_back"

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        prompt = get_prompt("step_back")
        result, usage = await self.llm.structured(
            prompt.render(question=query.text),
            StepBackOutput,
            system=prompt.system,
            model=self.router.model_for(Task.STEP_BACK),
            stage="step_back",
        )
        step_back = result.step_back_question.strip()
        if not step_back:
            return query, usage
        out = self._extend(
            query,
            [step_back],
            step_back=step_back,
            step_back_reasoning=result.reasoning,
        )
        log.debug("step_back", question=step_back[:80])
        return out, usage
