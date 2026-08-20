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

from ragorc.index.contextual import ContextualEnricher
from ragorc.index.multirep.dense_x import PropositionIndexer
from ragorc.index.multirep.parent_document import (
    ParentDocumentIndex,
    ParentDocumentIndexer,
    expand_parents,
    scoped_settings,
)
from ragorc.index.multirep.summary import SummaryIndexer

__all__ = [
    "ContextualEnricher",
    "ParentDocumentIndex",
    "ParentDocumentIndexer",
    "PropositionIndexer",
    "SummaryIndexer",
    "expand_parents",
    "scoped_settings",
]
