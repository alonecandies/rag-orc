"""RAG-Fusion: expand the query, retrieve per variant, fuse by reciprocal rank.

The expansion is the same as multi-query. The difference — and the whole point —
is the combination step.

Concatenating per-variant results ranks by whichever variant happened to score
highest, so one lucky variant can dominate. Reciprocal Rank Fusion instead sums
``1 / (k + rank)`` across the variant result lists, which means a document that
*several* independent phrasings surfaced outranks one that a single phrasing
ranked first. That is a consensus signal, and it is more robust than any
individual similarity score because it is scale-free: it never has to compare a
cosine to a BM25 score.

The fusion primitives themselves live in :mod:`ragorc.retrieve.fusion` and are
re-exported here so the classic RAG-Fusion import path works.
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
from ragorc.retrieve.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from ragorc.translate.base import BaseTranslator

log = structlog.get_logger(__name__)

__all__ = ["DEFAULT_RRF_K", "RAGFusionTranslator", "reciprocal_rank_fusion"]


@register("translator", "rag_fusion", "ragfusion")
class RAGFusionTranslator(BaseTranslator):
    """Query expansion whose contract is that the retriever fuses with RRF.

    It sets ``metadata["fusion"] = "rrf"`` so the retriever knows to fuse
    per-variant rather than concatenate. A retriever that ignores that flag
    degrades to plain multi-query behaviour, which is a reasonable fallback rather
    than a failure.
    """

    name = "rag_fusion"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        n: int = 4,
        rrf_k: int | None = None,
    ) -> None:
        super().__init__(llm, settings, router=router)
        self.n = n
        self.rrf_k = rrf_k

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        prompt = get_prompt("multi_query")
        result, usage = await self.llm.structured(
            prompt.render(question=query.text, n=self.n),
            MultiQueryOutput,
            system=prompt.system,
            model=self.router.model_for(Task.MULTI_QUERY),
            stage="rag_fusion",
        )
        out = self._extend(
            query,
            result.queries,
            fusion="rrf",
            rrf_k=self.rrf_k if self.rrf_k is not None else self.settings.retrieval.rrf_k,
            fuse_per_variant=True,
        )
        log.debug("rag_fusion", variants=len(out.variants))
        return out, usage
