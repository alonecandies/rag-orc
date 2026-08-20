"""Multi-Query: retrieve with several phrasings and union the results.

The cheapest reliable recall improvement available. One question is one point in
embedding space; the passage that answers it may sit near a differently-worded
point. Asking three ways covers more of that space.

Distinct from RAG-Fusion (:mod:`ragorc.translate.rag_fusion`) in how the results
are *combined*, not in how the variants are made: multi-query concatenates and
deduplicates, RAG-Fusion ranks by reciprocal rank across the per-variant lists so
a document found by several variants is promoted. Multi-query is the right choice
when you want coverage; RAG-Fusion when you want consensus.
"""

from __future__ import annotations

import structlog

from ragorc.core.models import Query, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import MultiQueryOutput
from ragorc.core.settings import Settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.translate.base import BaseTranslator

log = structlog.get_logger(__name__)

__all__ = ["MultiQueryTranslator"]


@register("translator", "multi_query", "multiquery")
class MultiQueryTranslator(BaseTranslator):
    name = "multi_query"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        n: int = 3,
    ) -> None:
        super().__init__(llm, settings, router=router)
        self.n = n

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        prompt = get_prompt("multi_query")
        result, usage = await self.llm.structured(
            prompt.render(question=query.text, n=self.n),
            MultiQueryOutput,
            system=prompt.system,
            model=self.router.model_for(Task.MULTI_QUERY),
            stage="multi_query",
        )
        out = self._extend(query, result.queries, multi_query_requested=self.n)
        log.debug("multi_query", requested=self.n, kept=len(out.variants))
        return out, usage
