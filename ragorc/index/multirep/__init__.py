"""Multi-representation indexing: the retrieval unit is not the generation unit.

One idea, four implementations. Search is best served by a small, precise, single
topic representation; generation is best served by a large, complete one. Nothing
requires them to be the same object, and every technique here exploits that by
indexing a *derived* representation and substituting the original at generation
time through ``parent_id``.

They differ in what the derived representation is, what it costs, and what it
breaks:

======================  =========================  ==============  ==================
Indexer                 Search unit                Ingest cost     Best for
======================  =========================  ==============  ==================
``parent_document``     a slice of the parent      second split    almost everything
``contextual``          chunk + situating prefix   1 call/chunk    hosted embedders
``summary``             an LLM summary             1 call/chunk    transcripts, prose
``dense_x``             one atomic proposition     1 call/chunk    fact lookup
======================  =========================  ==============  ==================

Read that table as a ladder and start at the top. ``parent_document`` involves no
model calls at all and captures most of the available gain, so it is the default
choice; the other three buy precision with an LLM call per chunk at ingest, which
is permanent spend on a permanent artifact. ``contextual`` is the one to reach for
when the embedder is a hosted API, because it recovers the same information late
chunking would have pooled in (ADR-0002) and late chunking is unavailable there.

``dense_x`` is the precision ceiling and the cost ceiling together, and it is the
one that fails ugliest when misapplied: a proposition that is not
context-independent is worthless the moment it is retrieved alone, which is the
only way it is ever retrieved.

All four are registered under the ``indexer`` kind, so a deployment selects one by
name, and all four resolve their pointers through
:func:`~ragorc.index.multirep.parent_document.expand_parents` — one batched query
per result set, deduplicated so several matching children of a parent yield that
parent exactly once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from ragorc.core.models import Chunk, Document, Usage
from ragorc.core.settings import Settings, get_settings
from ragorc.index.contextual import ContextualEnricher
from ragorc.index.multirep.dense_x import PropositionIndexer
from ragorc.index.multirep.parent_document import (
    ParentDocumentIndex,
    ParentDocumentIndexer,
    expand_parents,
    scoped_settings,
)
from ragorc.index.multirep.summary import SummaryIndexer

log = structlog.get_logger(__name__)

__all__ = [
    "ContextualEnricher",
    "MultiRepresentationIndexer",
    "ParentDocumentIndex",
    "ParentDocumentIndexer",
    "PropositionIndexer",
    "SummaryIndexer",
    "expand_parents",
    "scoped_settings",
]


class MultiRepresentationIndexer:
    """The ingest-stage face of this package.

    `IngestPipeline` looks for this name; until it existed, three documented
    settings resolved to a stage that could not be loaded and did nothing.

    Write ownership
    ---------------
    The stage does **not** upsert. `_process_document` embeds the leaf chunks,
    calls this, and then writes whatever comes back, so an indexer that also wrote
    would write twice — idempotent but wasteful, and it would race the pipeline's
    own vectors for the same ids. So `vector_store` is deliberately not forwarded:
    the indexers build and embed their derived units and hand them over, and the
    pipeline writes.

    The docstore *is* forwarded, because that write is not a duplicate. The
    derived unit replaces its source in the vector store, and
    :func:`expand_parents` reads the source back from the docstore at query time;
    if nobody persists it, retrieval returns units whose sources do not exist.

    Not parent-document
    -------------------
    `ParentDocumentIndexer` performs its own two-level split, which makes it a
    chunking *mode* rather than an enrichment: running it here would index every
    document twice, once as the pipeline's chunks and once as its own children.
    `indexing.parent_document_enabled` therefore reports that it must be driven
    directly rather than quietly doing half of it. See docs/internal/OPEN-ITEMS.md.

    Derived units carry the dense vector their indexer computed and no sparse or
    ColBERT vector, because those are added before this stage runs. On a hybrid
    collection they are therefore dense-only — findable by vector search, not by
    BM25.
    """

    name = "multirep"

    def __init__(
        self,
        llm: Any,
        *,
        embedder: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.settings = settings or get_settings()

    async def enrich(
        self,
        document: Document,
        chunks: Sequence[Chunk],
        *,
        relational_store: Any | None = None,
    ) -> tuple[list[Chunk], Usage]:
        config = self.settings.indexing
        current = list(chunks)
        usages: list[Usage] = []

        if config.summary_index_enabled:
            summariser = SummaryIndexer(self.llm, embedder=self.embedder, settings=self.settings)
            current, usage = await summariser.index(current, docstore=relational_store)
            usages.append(usage)

        if config.dense_x_enabled:
            propositions = PropositionIndexer(
                self.llm, embedder=self.embedder, settings=self.settings
            )
            current, usage = await propositions.index(current, docstore=relational_store)
            usages.append(usage)

        if config.parent_document_enabled:
            log.info(
                "parent_document_not_an_ingest_stage",
                document_id=document.id,
                hint=(
                    "ParentDocumentIndexer re-splits the document, so running it here "
                    "would index it twice; drive it directly — see docs/modules/index.md"
                ),
            )

        return current, Usage.sum(usages) if usages else Usage()
