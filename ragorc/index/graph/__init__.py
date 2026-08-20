"""GraphRAG construction: text -> entities -> graph -> communities -> reports.

The ingest half of GraphRAG. Four stages, each one useful on its own and each one
correcting a specific way the previous stage's output breaks traversal:

* :class:`~ragorc.index.graph.extract.EntityExtractor` — entities and typed,
  weighted relations per chunk, with gleaning passes and endpoint validation.
* :class:`~ragorc.index.graph.resolve.EntityResolver` — collapses the mentions of
  one thing into one node, without which the graph fragments into near-duplicates
  and traversal stops finding anything.
* :class:`~ragorc.index.graph.community.CommunityDetector` — hierarchical Leiden
  over the resolved graph, producing the clusters that global search reads.
* :class:`~ragorc.index.graph.summarize.CommunitySummarizer` — one report per
  community, written bottom-up so a parent reuses its children's summaries.
* :class:`~ragorc.index.graph.build.GraphBuilder` — the orchestrator, including
  the write ordering into Neo4j that the stages depend on.

The retrieval half lives in :mod:`ragorc.retrieve.graph` and reads what this
package writes.
"""

from __future__ import annotations

from ragorc.index.graph.build import GraphBuilder, GraphBuildReport
from ragorc.index.graph.community import CommunityDetector, CommunityHierarchy, community_id
from ragorc.index.graph.extract import (
    EntityExtractor,
    GraphExtraction,
    normalize_entity_name,
    normalize_relation_type,
)
from ragorc.index.graph.resolve import EntityResolver, ResolutionReport, normalized_form
from ragorc.index.graph.summarize import CommunitySummarizer, SummarizationReport

__all__ = [
    "CommunityDetector",
    "CommunityHierarchy",
    "CommunitySummarizer",
    "EntityExtractor",
    "EntityResolver",
    "GraphBuildReport",
    "GraphBuilder",
    "GraphExtraction",
    "ResolutionReport",
    "SummarizationReport",
    "community_id",
    "normalize_entity_name",
    "normalize_relation_type",
    "normalized_form",
]
