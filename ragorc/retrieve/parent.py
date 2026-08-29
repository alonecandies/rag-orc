"""Parent-document retrieval: search the precise unit, return the complete one.

The index-side half of this pattern already exists
(:mod:`ragorc.index.multirep.parent_document`): documents are split twice, only
the small child chunks are embedded, and each child records the parent it came
from. This is the query-side half — the retriever that closes the loop, so
``RetrievalSource.PARENT`` is reachable from configuration rather than only by
hand-wiring ``expand_parents`` at a call site.

:func:`parent_leg` is what does the reaching. Having the class and not calling it
was the entire defect for three representations at once: writing the query-side
half is not the same as wiring it, and only the wiring is observable.

The idea in one line: **the unit that matches best is rarely the unit that reads
best.** A 200-token child is precise enough to win a similarity contest on a
specific phrase; the 2000-token parent around it is what actually answers the
question. Embedding the parent instead would dilute the phrase across too much
text to rank; returning the child instead hands the model a fragment.

Three details that make it work, and that a naive implementation gets wrong:

* **Retrieve *more* children than you want parents.** Several children of one
  parent collapse into a single result, so asking for ``top_k`` children yields
  fewer than ``top_k`` parents. This over-fetches and trims afterwards.
* **Expansion writes metadata; it does not substitute.** ``expand_parents``
  records ``metadata["parent_text"]``, and :mod:`ragorc.context.pack` performs the
  swap *after* selection and reordering. That ordering is deliberate: substituting
  earlier would let a large parent be weighed by the packer as though its full
  length were the thing that matched.
* **Sibling evidence is merged, not discarded.** When four children of one parent
  match, that is a stronger signal than one child matching, and the merged
  ``component_scores`` keep it visible.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragorc.core.models import Query, RetrievalResult, RetrievalSource, ScoredChunk
from ragorc.core.protocols import Retriever
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed

log = structlog.get_logger(__name__)

__all__ = ["ParentDocumentRetriever", "parent_leg"]

#: How many extra children to fetch per requested parent. Children of one parent
#: collapse to a single result, so fetching exactly ``top_k`` children reliably
#: returns fewer than ``top_k`` parents — and on a corpus that chunks finely, far
#: fewer. Three is enough for the common case without tripling the rerank cost.
_CHILD_OVERFETCH = 3


@register("retriever", "parent_document", "parent")
class ParentDocumentRetriever:
    """Retrieve child chunks, return them with their parents attached."""

    name = "parent_document"

    def __init__(
        self,
        inner: Retriever,
        store: Any = None,
        *,
        settings: Settings | None = None,
        overfetch: int = _CHILD_OVERFETCH,
    ) -> None:
        self.settings: Settings = settings or get_settings()
        self.inner = inner
        #: Where parent bodies live. Postgres in the default deployment: the
        #: parents are deliberately *not* in the vector store, since nothing ever
        #: searches them and their embeddings would be storage spent on a query
        #: that is never issued.
        self.store = store
        self.overfetch = max(overfetch, 1)

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        wanted = self._wanted(top_k, query)
        with timed("parent_document_retrieve", wanted=wanted):
            children = await self.inner.retrieve(query, top_k=wanted * self.overfetch, **kwargs)
            if not children:
                return []
            return self._finish(await self._expand(children), wanted, len(children))

    async def retrieve_detailed(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> RetrievalResult:
        """Expand in place, preserving the inner leg's diagnostics.

        Five call sites choose their code path with
        ``getattr(retriever, "retrieve_detailed", None)`` — the pipeline's two
        retrieval nodes among them. A wrapper without this method does not merely
        lose ``per_store`` and ``timings_ms``: it silently moves those callers onto
        their fallback branch, so wrapping the vector leg would have changed how
        the pipeline retrieves, not just what it returns.

        ``per_store`` keeps the *children*, because that is what each leg actually
        found. Collapsing it to parents would make the dense and sparse counts
        disagree with the searches that produced them.
        """
        wanted = self._wanted(top_k, query)
        inner_detailed = getattr(self.inner, "retrieve_detailed", None)
        if inner_detailed is None:
            result = RetrievalResult()
            result.chunks = await self.retrieve(query, top_k=top_k, **kwargs)
            result.total_candidates = len(result.chunks)
            return result

        with timed("parent_document_retrieve", wanted=wanted):
            result = await inner_detailed(query, top_k=wanted * self.overfetch, **kwargs)
            children = list(result.chunks)
            if children:
                result.chunks = self._finish(await self._expand(children), wanted, len(children))
        return result

    def _wanted(self, top_k: int | None, query: Query) -> int:
        return int(top_k or query.top_k or self.settings.retrieval.top_k)

    def _finish(self, expanded: list[ScoredChunk], wanted: int, children: int) -> list[ScoredChunk]:
        """Trim to the requested width and restamp rank and source."""
        out = expanded[:wanted]
        for rank, item in enumerate(out):
            item.rank = rank
            item.source = RetrievalSource.PARENT
        log.debug(
            "parent_document_retrieved",
            children=children,
            parents=len(expanded),
            returned=len(out),
        )
        return out

    async def _expand(self, children: list[ScoredChunk]) -> list[ScoredChunk]:
        """Attach parent bodies, degrading to the children if that is impossible.

        A missing store or a failed lookup returns the children unchanged rather
        than raising. They are still relevant text and still correctly ranked —
        the answer is merely narrower than it would have been, which is a far
        better outcome than failing a query that had already found the right
        passage.
        """
        if self.store is None:
            log.debug("parent_expansion_skipped", reason="no parent store configured")
            return children
        try:
            from ragorc.index.multirep.parent_document import expand_parents

            return await expand_parents(children, self.store)
        except Exception as exc:  # noqa: BLE001 - degrade, never fail the query
            log.warning(
                "parent_expansion_failed",
                error=str(exc)[:200],
                effect="returning child chunks without their parents",
            )
            return children


def parent_leg(inner: Retriever, store: Any, *, settings: Settings | None = None) -> Retriever:
    """Wrap a vector leg in parent expansion when the index holds derived units.

    The one call every wiring makes, because the alternative was for each wiring
    to *remember*. Parent-document, summary-index and dense-X all put a stand-in
    for the source into the vector store — a child chunk, an LLM summary, a
    rewritten proposition — and all three ship a query-time step that resolves the
    stand-in back. None of the three had a caller: the pipeline built a
    ``HybridRetriever`` over a collection full of summaries and handed the
    summaries to the generator, which then answered from a paraphrase of the
    document while citing the document.

    Nothing here fires unless an indexing representation is on, so a default
    deployment builds the same object graph it did before. ``parent_expansion``
    gates it too: that flag is what the packer asks before substituting, and
    fetching bodies nobody will substitute is work with no output.

    Wraps the *vector* leg specifically. Graph, web and relational results have no
    ``parent_id`` to resolve, and the over-fetch this applies is only correct for
    the leg whose hits collapse.
    """
    resolved = settings or get_settings()
    if not resolved.indexing.multirep_enabled or not resolved.retrieval.parent_expansion:
        return inner
    log.info(
        "parent_expansion_enabled",
        parent_document=resolved.indexing.parent_document_enabled,
        summary_index=resolved.indexing.summary_index_enabled,
        dense_x=resolved.indexing.dense_x_enabled,
        store=type(store).__name__ if store is not None else None,
    )
    return ParentDocumentRetriever(inner, store, settings=resolved)
