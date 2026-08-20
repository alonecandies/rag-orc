"""Parent-document retrieval: search the precise unit, return the complete one.

The index-side half of this pattern already exists
(:mod:`ragorc.index.multirep.parent_document`): documents are split twice, only
the small child chunks are embedded, and each child records the parent it came
from. What was missing is the query-side half — a retriever that closes the loop,
so ``RetrievalSource.PARENT`` is reachable from configuration rather than only by
hand-wiring ``expand_parents`` at a call site.

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

from ragorc.core.models import Query, RetrievalSource, ScoredChunk
from ragorc.core.protocols import Retriever
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed

log = structlog.get_logger(__name__)

__all__ = ["ParentDocumentRetriever"]

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
        wanted = top_k or query.top_k or self.settings.retrieval.top_k
        with timed("parent_document_retrieve", wanted=wanted):
            children = await self.inner.retrieve(query, top_k=wanted * self.overfetch, **kwargs)
            if not children:
                return []

            expanded = await self._expand(children)
            out = expanded[:wanted]

        for rank, item in enumerate(out):
            item.rank = rank
            item.source = RetrievalSource.PARENT
        log.debug(
            "parent_document_retrieved",
            children=len(children),
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
