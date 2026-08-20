"""Context packing: what goes in the prompt, and in what order.

Two decisions, both of which measurably change answer quality.

**What** — this is a knapsack: each chunk has a token cost and a relevance
value, and the window is the capacity. Filling greedily by score alone wastes
capacity, because a 900-token chunk scoring 0.81 displaces three 250-token
chunks scoring 0.79. Selecting by *relevance per token* fits more evidence into
the same window. The rank order is preserved for presentation; only the
selection is density-driven. The capacity is only respected if the cost counted
is the cost *rendered* — a passage's numbered header and its untrusted-document
wrapper are 20-25 tokens that the model has to read like any other, so they are
part of what a chunk costs, not free packaging.

**Where** — "lost in the middle" (Liu et al., 2023): transformer attention over
long contexts is U-shaped, so evidence placed in the middle of a long prompt is
markedly less likely to be used than the same evidence at either end. Packing in
plain rank order therefore buries the second- and third-best passages in exactly
the dead zone. The fix is to interleave outward from both ends, so the strongest
chunks occupy the positions the model actually attends to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import structlog

from ragorc.core.models import ScoredChunk
from ragorc.core.settings import Settings, get_settings
from ragorc.core.tokens import count_tokens, count_tokens_batch, truncate_to_tokens
from ragorc.security.injection import wrap_untrusted

log = structlog.get_logger(__name__)

__all__ = ["ContextPack", "ContextPacker", "reorder_lost_in_middle"]

_PASSAGE_SEPARATOR = "\n\n"


def _repacked(scored: ScoredChunk, content: str, **explain: Any) -> ScoredChunk:
    """A copy of ``scored`` carrying ``content``, leaving the caller's objects alone.

    Both places that change a body — clipping the oversized top hit and swapping
    in a parent/window span — used to assign ``scored.chunk.content`` directly.
    ``ScoredChunk.with_score`` shares the underlying ``Chunk``, so that edited the
    object the caller still holds: after ``build(budget=40)`` an 801-token chunk in
    the caller's ``RetrievalResult`` was permanently 32 tokens, and a second
    ``build`` with a huge budget still saw the clipped text — packing was not
    idempotent, which bites on the ``generate()``-then-``stream()`` path where the
    same chunks are packed twice. It also left ``chunk.id``, which is derived from
    the content at index time, describing text the chunk no longer held.

    The copy keeps the id on purpose: citations resolve ``[n]`` back to these
    chunks and the rerank cache is keyed on the id, so re-deriving it here would
    break attribution. What the packer hands back is meant to be the packed text
    (that is what the model saw); it is the *input* that must survive untouched.
    """
    return ScoredChunk(
        chunk=replace(scored.chunk, content=content, token_count=count_tokens(content)),
        score=scored.score,
        source=scored.source,
        rank=scored.rank,
        component_scores=dict(scored.component_scores),
        explain={**scored.explain, **explain},
    )


def reorder_lost_in_middle(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Place the strongest evidence at both ends of the context.

    Given ranks ``[0,1,2,3,4,5]`` the result is ``[0,2,4,5,3,1]``: rank 0 first,
    rank 1 last, rank 2 second, rank 3 second-to-last, and so on. Both extremes
    — the positions with the highest attention — are occupied by the highest
    ranked passages, and the weakest end up in the middle where being ignored
    costs least.
    """
    if len(chunks) <= 2:
        return list(chunks)
    head: list[ScoredChunk] = []
    tail: list[ScoredChunk] = []
    for i, chunk in enumerate(chunks):
        (head if i % 2 == 0 else tail).append(chunk)
    return head + tail[::-1]


@dataclass(slots=True)
class ContextPack:
    """The packed context, ready for a prompt."""

    text: str
    chunks: list[ScoredChunk]
    tokens: int
    dropped: int = 0
    truncated: int = 0
    order: str = "rank"

    def __len__(self) -> int:
        return len(self.chunks)


class ContextPacker:
    """Selects, orders and renders retrieved chunks into prompt text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()

    async def pack(
        self,
        query: object,
        chunks: Sequence[ScoredChunk],
        *,
        budget: int,
        isolate: bool = True,
    ) -> tuple[list[ScoredChunk], str]:
        """:class:`ragorc.core.protocols.ContextPacker` entry point."""
        result = self.build(chunks, budget=budget, isolate=isolate)
        return result.chunks, result.text

    def build(
        self,
        chunks: Sequence[ScoredChunk],
        *,
        budget: int,
        isolate: bool = True,
        expand_parents: bool | None = None,
    ) -> ContextPack:
        if not chunks:
            return ContextPack(text="", chunks=[], tokens=0)

        working = self._expand(chunks, expand_parents)
        # `isolate` reaches selection because it decides whether `_render` emits the
        # <untrusted_document> wrapper, and that wrapper is most of what a passage
        # costs beyond its body text. Pricing it wrong is what put the pack over
        # `budget`; see `_select`.
        selected, dropped, truncated = self._select(working, budget, isolate=isolate)

        if self.settings.retrieval.reorder_lost_in_middle and len(selected) > 2:
            ordered = reorder_lost_in_middle(selected)
            order = "lost_in_middle"
        else:
            ordered = selected
            order = "rank"

        text = self._render(ordered, isolate=isolate)
        tokens = count_tokens(text)
        log.debug(
            "context_packed",
            selected=len(selected),
            dropped=dropped,
            truncated=truncated,
            tokens=tokens,
            budget=budget,
            order=order,
        )
        return ContextPack(
            text=text,
            chunks=ordered,
            tokens=tokens,
            dropped=dropped,
            truncated=truncated,
            order=order,
        )

    # -- selection ---------------------------------------------------------
    def _select(
        self, chunks: Sequence[ScoredChunk], budget: int, *, isolate: bool
    ) -> tuple[list[ScoredChunk], int, int]:
        """Density-first knapsack with a rank-order guarantee for the top hit.

        The single best chunk is admitted unconditionally (truncated if it alone
        exceeds the budget) — returning nothing because the top result is large
        is worse than returning it clipped. Everything after that competes on
        relevance per token.

        A chunk's cost is what the *renderer* will spend on it, body plus
        scaffolding, not the length of its text.
        """
        body_costs = [c.chunk.token_count or count_tokens(c.chunk.content) for c in chunks]
        for chunk, cost in zip(chunks, body_costs, strict=True):
            # Writing this back memoizes a value derived from the content already
            # there, so it stays true for the caller's chunk. Rewriting a *body*
            # does not, which is what `_repacked` exists to avoid.
            chunk.chunk.token_count = cost

        # Charge each passage for the scaffolding `_render` will wrap it in.
        #
        # This used to budget `chunk.content` alone, while `_render` prepends a
        # numbered provenance header and wraps the body in <untrusted_document>
        # tags — 20-25 tokens a passage that nobody paid for. Measured on an
        # 8192-token window: `available_context=6572` came back as a 9563-token
        # pack (45% over), and prompt + `reserved_output` then no longer fit the
        # window at all. The overshoot lands hardest on the `truncate` strategy,
        # whose entire job is to make the context fit; at the 128k default it is a
        # harmless ~4.5%, which is why the suite never noticed.
        #
        # The separator is charged on every passage rather than on n-1 of them:
        # the spare token absorbs the boundary token that a scaffolding-only
        # render loses where the header meets the body (22 measured vs 23 real).
        # Indices are input positions, which are >= the render index a chunk ends
        # up at — selection only ever removes passages — so `[n]` is priced at or
        # above what it costs. Erring high is the safe direction for a budget.
        separator = count_tokens(_PASSAGE_SEPARATOR)
        overheads = [n + separator for n in self._scaffold_tokens(chunks, isolate=isolate)]
        costs = [body + over for body, over in zip(body_costs, overheads, strict=True)]

        remaining = budget
        chosen: list[tuple[int, ScoredChunk]] = []
        truncated = 0

        first, first_cost = chunks[0], costs[0]
        if first_cost > remaining:
            clipped = truncate_to_tokens(first.chunk.content, max(remaining - overheads[0], 0))
            if not clipped and remaining > 0:
                # The budget is smaller than this passage's own scaffolding, so
                # there is no honest room for any body at all. The documented
                # guarantee is that the top hit still ships, and a sub-25-token
                # window is unusable either way, so spend what is left on body text
                # and let the scaffolding overshoot rather than silently dropping
                # rank 0 and handing its slot to a weaker chunk.
                clipped = truncate_to_tokens(first.chunk.content, remaining)
            if clipped:
                truncated += 1
                chosen.append((0, _repacked(first, clipped, truncated=True)))
                remaining = 0
        else:
            chosen.append((0, first))
            remaining -= first_cost

        # Density is computed on scores lifted to clear zero — but only when
        # something is below it.
        #
        # Dividing a raw score by token count inverts the ordering as soon as any
        # score is negative — and the default reranker emits negatives routinely:
        # a live `Xenova/ms-marco-MiniLM-L-6-v2` run scores [8.42, -11.29, -11.31].
        # With negatives, "score per token" rewards *longer* chunks, because
        # dividing a negative by a larger denominator moves it closer to zero.
        # Measured: a 401-token chunk scoring -5.00 was selected over a 31-token
        # chunk scoring -0.50, i.e. the packer preferred a passage 10x worse and
        # 13x longer.
        #
        # The condition is the point. Subtracting the minimum *unconditionally*
        # would pin the worst candidate at density 0 and invert this module's own
        # headline example: 0.81/900 against 0.79/250 picks the small chunks on raw
        # scores, but as margins above the minimum — 0.02/900 against 0.00/250 —
        # the fat chunk wins instead. A batch that never dips below zero therefore
        # keeps the raw ratio the design argument is stated in; only a batch that
        # reaches below zero gets lifted.
        #
        # So the guarantee here is not invariance to score offset — the ordering
        # does shift as the batch minimum crosses zero — but the weaker property
        # selection actually needs: density never falls as score rises, and never
        # rises as cost rises.
        floor = min((c.score for c in chunks), default=0.0)
        shift = -floor if floor < 0.0 else 0.0

        candidates = sorted(
            ((i, chunks[i], costs[i]) for i in range(1, len(chunks))),
            # Higher score per token first; on a tie prefer the better rank.
            key=lambda t: (-((t[1].score + shift) / max(t[2], 1)), t[0]),
        )
        for index, chunk, cost in candidates:
            if cost <= remaining:
                chosen.append((index, chunk))
                remaining -= cost

        chosen.sort(key=lambda t: t[0])  # restore rank order for presentation
        selected = [c for _, c in chosen]
        return selected, len(chunks) - len(selected), truncated

    def _expand(
        self, chunks: Sequence[ScoredChunk], expand_parents: bool | None
    ) -> list[ScoredChunk]:
        """Swap in the wider text a chunk points at.

        Two multi-representation patterns converge here: the sentence-window
        splitter stores its surrounding sentences in ``metadata["window_text"]``,
        and the parent-document retriever stores the parent body in
        ``metadata["parent_text"]``. Both search a *precise* unit and generate
        from a *wide* one, and this is where the substitution happens — after
        ranking, so precision is preserved where it matters.
        """
        if expand_parents is None:
            expand_parents = self.settings.retrieval.parent_expansion
        if not expand_parents:
            return list(chunks)

        out: list[ScoredChunk] = []
        seen_parents: set[str] = set()
        for scored in chunks:
            meta = scored.chunk.metadata
            wider = meta.get("parent_text") or meta.get("window_text")
            if not wider or wider == scored.chunk.content:
                out.append(scored)
                continue
            key = scored.chunk.parent_id or scored.chunk.id
            if key in seen_parents:
                # Several child chunks can share one parent; emitting the parent
                # once per child would duplicate the same text in the prompt.
                continue
            seen_parents.add(key)
            # A new ScoredChunk *and* a new Chunk. Wrapping alone was not enough:
            # the wrapper shared the caller's `Chunk`, and assigning the wider body
            # to it rewrote the retrieved text under the caller — this substitution
            # is meant to change what the prompt shows, not what was retrieved.
            out.append(_repacked(scored, wider, expanded=True))
        return out

    # -- rendering ---------------------------------------------------------
    def _render(self, chunks: Sequence[ScoredChunk], *, isolate: bool) -> str:
        """Render numbered passages with provenance.

        Numbering is 1-based and stable, because the generator cites ``[n]`` and
        the citation validator resolves those numbers back to these chunks. The
        source line matters too: a model that can see where a passage came from
        writes better attributions and hedges appropriately on weak sources.
        """
        parts = [
            self._render_passage(scored, i, scored.chunk.content.strip(), isolate=isolate)
            for i, scored in enumerate(chunks, start=1)
        ]
        return _PASSAGE_SEPARATOR.join(parts)

    def _render_passage(self, scored: ScoredChunk, index: int, body: str, *, isolate: bool) -> str:
        """One passage, rendered. Split out of ``_render`` so ``_select`` can price
        the scaffolding by rendering it with an empty body."""
        chunk = scored.chunk
        meta = chunk.metadata
        label_bits = [f"[{index}]"]
        source = meta.get("source") or meta.get("title") or chunk.document_id
        if source:
            label_bits.append(f"source: {source}")
        if scored.source.value not in ("dense", "fused"):
            label_bits.append(f"via: {scored.source.value}")
        if chunk.level:
            label_bits.append(f"summary-level: {chunk.level}")
        header = " | ".join(label_bits)
        if isolate:
            body = wrap_untrusted(body, index=index)
        return f"{header}\n{body}"

    def _scaffold_tokens(self, chunks: Sequence[ScoredChunk], *, isolate: bool) -> list[int]:
        """What each passage costs before a single token of its body is added.

        Measured rather than assumed: the header grows with whatever provenance the
        chunk carries (``source``, ``via:``, ``summary-level:``) and ``isolate=False``
        emits no wrapper at all, so any constant would be wrong in both directions —
        3 tokens for a bare numbered passage, 22 for a wrapped one with a source.
        Rendering the real scaffolding with an empty body prices each chunk exactly.
        Batched, because these are N short strings and ``encode_batch`` counts them
        in Rust in one call.
        """
        scaffolds = [
            self._render_passage(scored, i, "", isolate=isolate)
            for i, scored in enumerate(chunks, start=1)
        ]
        return count_tokens_batch(scaffolds)
