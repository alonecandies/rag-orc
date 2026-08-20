"""Overflow compression: fitting more evidence than the window holds.

When retrieval returns 3x the context window of genuinely relevant material,
truncating discards evidence the answer needs. The alternative is to compress,
and there are three established shapes with different trade-offs:

* **map-reduce** — summarize each group independently, then combine. Fully
  parallel, so its latency is one map call plus one reduce call regardless of
  input size. Loses cross-group relationships, because no map call sees more
  than its own group.
* **refine** — carry a running summary through the groups in sequence. Preserves
  narrative and cross-group relationships; strictly serial, so latency grows
  linearly with the number of groups.
* **hierarchical** — map-reduce applied repeatedly until the result fits. The
  right choice when the overflow is large (10x+), where a single reduce step
  would itself overflow.

Default is map-reduce: the parallelism matters more than the lost cross-group
context, because the *reduce* call still sees all the summaries together and can
relate them there.

Every strategy is query-aware. A generic summary throws away exactly the details
the question needed; the prompts here compress *toward the question*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import Chunk, Modality, ScoredChunk, Usage
from ragorc.core.protocols import LLM
from ragorc.core.settings import Settings, get_settings
from ragorc.core.tokens import count_tokens
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["ContextSummarizer", "SummarizationResult"]


@dataclass(slots=True)
class SummarizationResult:
    chunks: list[ScoredChunk]
    usage: Usage
    strategy: str
    input_tokens: int
    output_tokens: int

    @property
    def compression_ratio(self) -> float:
        return self.output_tokens / self.input_tokens if self.input_tokens else 1.0


class ContextSummarizer:
    """Compresses a chunk set until it fits a token budget."""

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        self.llm = llm
        self.settings: Settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)

    async def fit(
        self,
        question: str,
        chunks: Sequence[ScoredChunk],
        *,
        budget: int,
        strategy: str = "map_reduce",
        max_levels: int = 3,
    ) -> SummarizationResult:
        """Compress ``chunks`` to at most ``budget`` tokens."""
        total = sum(c.chunk.token_count or count_tokens(c.chunk.content) for c in chunks)
        if total <= budget or not chunks:
            return SummarizationResult(list(chunks), Usage(), "fit", total, total)

        if strategy == "refine":
            out, usage = await self._refine(question, chunks, budget)
        elif strategy == "hierarchical":
            out, usage = await self._hierarchical(question, chunks, budget, max_levels)
        else:
            out, usage = await self._map_reduce(question, chunks, budget)

        produced = sum(c.chunk.token_count or count_tokens(c.chunk.content) for c in out)
        log.info(
            "context_compressed",
            strategy=strategy,
            input_tokens=total,
            output_tokens=produced,
            budget=budget,
            ratio=round(produced / total, 3) if total else 1.0,
            cost_usd=round(usage.cost_usd, 6),
        )
        return SummarizationResult(out, usage, strategy, total, produced)

    # -- strategies --------------------------------------------------------
    async def _map_reduce(
        self, question: str, chunks: Sequence[ScoredChunk], budget: int
    ) -> tuple[list[ScoredChunk], Usage]:
        groups = self._group(chunks, budget)
        prompt = get_prompt("compress_extract")
        model = self.router.model_for(Task.COMPRESS)

        # Per-group budget leaves headroom: N summaries at exactly budget/N will
        # overflow once the passage headers and separators are added.
        per_group = max(int(budget / max(len(groups), 1) * 0.8), 120)

        async def summarize(group: list[ScoredChunk]) -> tuple[str, Usage]:
            joined = "\n\n".join(c.chunk.content for c in group)
            text, usage = await self.llm.complete(
                prompt.render(question=question, document=joined),
                system=prompt.system,
                model=model,
                max_tokens=per_group,
                stage="compress_map",
            )
            return text.strip(), usage

        results = await bounded_gather(
            (summarize(g) for g in groups),
            limit=self.settings.retrieval.max_concurrent_retrievers,
            return_exceptions=True,
        )

        out: list[ScoredChunk] = []
        usages: list[Usage] = []
        for group, result in zip(groups, results, strict=True):
            if isinstance(result, BaseException):
                # A failed map call must not lose the group entirely: fall back
                # to the highest-ranked original chunk, clipped.
                log.warning("compress_map_failed", error=str(result)[:200])
                out.append(self._clip(group[0], per_group))
                continue
            text, usage = result
            usages.append(usage)
            if text:
                out.append(self._as_summary_chunk(group, text))
        return out, Usage.sum(usages)

    async def _refine(
        self, question: str, chunks: Sequence[ScoredChunk], budget: int
    ) -> tuple[list[ScoredChunk], Usage]:
        groups = self._group(chunks, budget)
        prompt = get_prompt("summarize_chunk")
        model = self.router.model_for(Task.SUMMARIZE)
        usages: list[Usage] = []
        running = ""
        for group in groups:
            joined = "\n\n".join(c.chunk.content for c in group)
            instruction = (
                f"Question: {question}\n\n"
                f"Summary so far:\n{running or '(none yet)'}\n\n"
                f"New material:\n{joined}\n\n"
                "Produce an updated summary that keeps everything relevant to the "
                "question from both the existing summary and the new material. "
                "Preserve entities, numbers and dates verbatim."
            )
            text, usage = await self.llm.complete(
                instruction,
                system=prompt.system,
                model=model,
                max_tokens=int(budget * 0.9),
                stage="compress_refine",
            )
            usages.append(usage)
            running = text.strip() or running
        merged = self._as_summary_chunk(list(chunks), running)
        return [merged], Usage.sum(usages)

    async def _hierarchical(
        self, question: str, chunks: Sequence[ScoredChunk], budget: int, max_levels: int
    ) -> tuple[list[ScoredChunk], Usage]:
        current = list(chunks)
        usages: list[Usage] = []
        for level in range(max_levels):
            total = sum(c.chunk.token_count or count_tokens(c.chunk.content) for c in current)
            if total <= budget or len(current) <= 1:
                break
            current, usage = await self._map_reduce(question, current, budget)
            usages.append(usage)
            log.debug("hierarchical_level", level=level, chunks=len(current), tokens=total)
        return current, Usage.sum(usages)

    # -- helpers -----------------------------------------------------------
    def _group(self, chunks: Sequence[ScoredChunk], budget: int) -> list[list[ScoredChunk]]:
        """Bin chunks into groups that each fit one model call.

        Grouped in rank order rather than by similarity, so a group's summary
        covers a coherent slice of the ranking and the strongest evidence is
        never diluted by being averaged with the weakest.
        """
        window = max(int(self.settings.llm.context_window * 0.4), budget)
        groups: list[list[ScoredChunk]] = []
        current: list[ScoredChunk] = []
        running = 0
        for chunk in chunks:
            cost = chunk.chunk.token_count or count_tokens(chunk.chunk.content)
            if current and running + cost > window:
                groups.append(current)
                current, running = [], 0
            current.append(chunk)
            running += cost
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _as_summary_chunk(group: Sequence[ScoredChunk], text: str) -> ScoredChunk:
        """Wrap compressed text, keeping provenance so citations still resolve.

        ``source_chunk_ids`` records everything that went into the summary, which
        is what lets the citation validator accept a claim supported by a
        compressed passage.
        """
        head = group[0]
        chunk = Chunk(
            id=f"{head.chunk.id}:sum",
            content=text,
            document_id=head.chunk.document_id,
            index=head.chunk.index,
            level=max(c.chunk.level for c in group) + 1,
            modality=Modality.SUMMARY,
            metadata={
                **head.chunk.metadata,
                "compressed_from": [c.chunk.id for c in group],
                "compressed_count": len(group),
            },
            tenant_id=head.chunk.tenant_id,
            token_count=count_tokens(text),
        )
        return ScoredChunk(
            chunk=chunk,
            score=max(c.score for c in group),
            source=head.source,
            rank=head.rank,
            component_scores=dict(head.component_scores),
            explain={"compressed": True, "from": len(group)},
        )

    @staticmethod
    def _clip(scored: ScoredChunk, budget: int) -> ScoredChunk:
        from ragorc.core.tokens import truncate_to_tokens

        clipped = truncate_to_tokens(scored.chunk.content, budget)
        scored.chunk.content = clipped
        scored.chunk.token_count = count_tokens(clipped)
        scored.explain["truncated"] = True
        return scored
