"""RRR — Rewrite, Retrieve, Read.

The narrowest of the three loops, and the cheapest. RRR (Ma et al., 2023) makes
one observation: **the user's question is written for a human, and retrieval wants
a search query.** Those are different artifacts. "I've been waiting ages, when do I
actually get my money back?" contains conversational framing that adds nothing to
a vector search and actively dilutes it.

So RRR rewrites *before* retrieving, rather than after failing:

    rewrite -> retrieve -> read     (as opposed to Self-RAG's
                                     retrieve -> read -> grade -> rewrite -> retry)

The difference matters in cost. Self-RAG pays for a full generation before it
discovers the query was bad; RRR pays one cheap rewrite up front and never
generates from bad retrieval in the first place. They compose: RRR on the way in,
Self-RAG as the safety net.

The retry loop here is driven by *retrieval* signal, not answer quality — if
retrieval comes back empty or weak, rewrite and try again, without having spent a
synthesis call to find out.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import structlog

from ragorc.core.models import Query, RetrievalResult, Usage
from ragorc.core.protocols import LLM
from ragorc.core.schemas import RewriteOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import trace_step
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.security.injection import render_untrusted_passages

log = structlog.get_logger(__name__)

__all__ = ["RRR", "RRRResult"]

RetrieveFn = Callable[[Query], Awaitable[RetrievalResult]]


@dataclass(slots=True)
class RRRResult:
    query: Query
    retrieval: RetrievalResult
    usage: Usage = field(default_factory=Usage)
    rewrites: list[str] = field(default_factory=list)
    succeeded: bool = True

    def report(self) -> dict[str, object]:
        return {
            "rewrites": self.rewrites,
            "final_query": self.query.text,
            "chunks": len(self.retrieval.chunks),
            "succeeded": self.succeeded,
        }


class RRR:
    """Rewrite-then-retrieve, with retrieval-driven retries."""

    name = "rrr"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        min_chunks: int = 1,
        min_top_score: float = 0.0,
    ) -> None:
        self.llm = llm
        self.settings: Settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self.min_chunks = min_chunks
        self.min_top_score = min_top_score

    async def run(self, query: Query, retrieve: RetrieveFn) -> RRRResult:
        max_rewrites = max(self.settings.generation.rrr_max_rewrites, 0)
        usages: list[Usage] = []
        rewrites: list[str] = []

        current, usage = await self._rewrite(query, reason="initial")
        usages.append(usage)
        if current.text != query.text:
            rewrites.append(current.text)

        retrieval = await retrieve(current)

        for attempt in range(max_rewrites):
            if self._sufficient(retrieval):
                break
            trace_step(
                "rrr_retry",
                attempt=attempt,
                chunks=len(retrieval.chunks),
                reason="weak_retrieval",
            )
            current, usage = await self._rewrite(
                current, reason="weak_retrieval", previous_result=retrieval
            )
            usages.append(usage)
            rewrites.append(current.text)
            retrieval = await retrieve(current)

        succeeded = self._sufficient(retrieval)
        if not succeeded:
            log.info("rrr_exhausted", rewrites=len(rewrites), chunks=len(retrieval.chunks))
        return RRRResult(
            query=current,
            retrieval=retrieval,
            usage=Usage.sum(usages),
            rewrites=rewrites,
            succeeded=succeeded,
        )

    def _sufficient(self, retrieval: RetrievalResult) -> bool:
        if len(retrieval.chunks) < self.min_chunks:
            return False
        if self.min_top_score > 0.0 and retrieval.chunks:
            return retrieval.chunks[0].score >= self.min_top_score
        return True

    async def _rewrite(
        self, query: Query, *, reason: str, previous_result: RetrievalResult | None = None
    ) -> tuple[Query, Usage]:
        prompt = get_prompt("rewrite_query")
        if previous_result is not None and previous_result.chunks:
            # Fenced, for the reason `nodes.rewrite` gives: this excerpt steers a
            # rewrite, and the rewrite steers the next retrieval.
            retrieved = render_untrusted_passages(
                [c.chunk.content[:100] for c in previous_result.chunks[:3]]
            )
        elif previous_result is not None:
            retrieved = "(nothing was retrieved)"
        else:
            retrieved = (
                "(not yet retrieved) Rewrite the user's conversational question into a "
                "precise search query: strip pleasantries and framing, keep every entity, "
                "identifier and constraint, and use the vocabulary documents would use."
            )

        try:
            result, usage = await self.llm.structured(
                prompt.render(
                    question=query.original or query.text,
                    previous=query.text,
                    retrieved=retrieved,
                ),
                RewriteOutput,
                system=prompt.system,
                model=self.router.model_for(Task.REWRITE),
                stage="rrr_rewrite",
            )
        except Exception as exc:  # noqa: BLE001
            # A failed rewrite degrades to the original query, which is still
            # searchable. Never let an optional enhancement fail the request.
            log.warning("rrr_rewrite_failed", reason=reason, error=str(exc)[:160])
            return query, Usage()

        rewritten = result.rewritten_query.strip()
        if not rewritten or rewritten.lower() == query.text.lower():
            return query, usage

        log.debug("rrr_rewritten", reason=reason, query=rewritten[:80])
        return (
            Query(
                text=rewritten,
                original=query.original,
                variants=query.variants,
                hypothetical=query.hypothetical,
                filters=dict(query.filters),
                top_k=query.top_k,
                tenant_id=query.tenant_id,
                metadata={**query.metadata, "rrr_rewrite_of": query.text, "rrr_reason": reason},
            ),
            usage,
        )
