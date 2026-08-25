"""GraphRAG construction: chunks in, a summarized knowledge graph in Neo4j out.

The orchestrator for the four stages that turn text into a graph — extract,
resolve, detect, summarize — plus the writes that persist them. The ordering is
not arbitrary and is the main thing this module contributes:

    extract ──▶ resolve ──▶ detect ──▶ summarize ──▶ write
                                                     entities
                                                     relations
                                                     chunk links
                                                     communities

**Why detection happens before any write.** Community detection needs only the
resolved entities and relations, both of which are already in memory. Running it
before the write means each entity is written *once*, with its final ``degree``
and ``community_id`` already on it. The alternative — write, detect, then write
again to attach community membership — doubles the most expensive write in the
pipeline to add two properties.

**Why communities are written last.** ``Neo4jStore.upsert_communities`` attaches
membership with ``MATCH`` on the entity nodes, not ``MERGE``. An entity that has
not been written yet is silently not attached, and the community's report then
describes members it is not linked to — a failure that produces no error and is
invisible until a global-search answer cites entities that cannot be reached.

**Why the write batches are sequential.** Every write here is already a single
``UNWIND`` round trip for its whole batch, so the network cost is already
amortized and concurrency would buy only overlap. What it would also buy is
deadlocks: ``MERGE`` takes locks on the nodes it touches, relation writes
``MERGE`` their endpoints, and two concurrent batches that touch the same entity
in opposite order deadlock. Neo4j reports that as a transient error and the
driver retries it, so the failure mode is a slow, occasionally-failing ingest
rather than a broken one — which is exactly the kind of bug that survives to
production. Batches go one at a time; the store already fans out per
relationship type inside a batch, where the locks are disjoint.

**Idempotency.** A second run over the same corpus converges instead of
duplicating, because every identity in the chain is content-derived: chunk ids
come from ``ids.chunk_id``, entity identity is the resolved canonical name,
community ids are a hash of level plus membership, and the store's ``MERGE``
statements accumulate descriptions with a ``CONTAINS`` guard and skip re-counting
edge weight for chunk ids already recorded. Re-running costs the LLM calls again
(extraction is not cached at this layer — the LLM cache handles that) but the
graph it lands on is the same graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from ragorc.core.models import Chunk, Community, Entity, Relation, Usage
from ragorc.core.protocols import LLM, DenseEmbedder
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.index.graph.community import CommunityDetector
from ragorc.index.graph.extract import EntityExtractor
from ragorc.index.graph.resolve import EntityResolver
from ragorc.index.graph.summarize import CommunitySummarizer
from ragorc.llm.router import ModelRouter
from ragorc.validate.schema import DocumentValidator

if TYPE_CHECKING:
    from ragorc.stores.neo4j.store import Neo4jStore

log = structlog.get_logger(__name__)

__all__ = ["GraphBuildReport", "GraphBuilder"]


@dataclass(slots=True)
class GraphBuildReport:
    """Counts, timings and cost for one construction run."""

    chunks_in: int = 0
    chunks_used: int = 0
    entity_mentions: int = 0
    entities: int = 0
    relations: int = 0
    merged_entities: int = 0
    dangling_relations: int = 0
    communities: int = 0
    community_levels: int = 0
    entities_written: int = 0
    relations_written: int = 0
    chunk_links_written: int = 0
    communities_written: int = 0
    communities_pruned: int = 0
    """Community nodes this build superseded and removed. Reported because a
    non-zero value means the previous partition of these entities is gone, which
    is exactly what an operator wants to see after re-ingesting changed
    documents — and a persistent zero across such a run is the symptom of the bug
    this counter was added with."""
    schema_applied: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())

    def summary(self) -> dict[str, Any]:
        return {
            "chunks_in": self.chunks_in,
            "chunks_used": self.chunks_used,
            "entity_mentions": self.entity_mentions,
            "entities": self.entities,
            "relations": self.relations,
            "merged_entities": self.merged_entities,
            "dangling_relations": self.dangling_relations,
            "communities": self.communities,
            "community_levels": self.community_levels,
            "entities_written": self.entities_written,
            "relations_written": self.relations_written,
            "chunk_links_written": self.chunk_links_written,
            "communities_written": self.communities_written,
            "communities_pruned": self.communities_pruned,
            "total_ms": round(self.total_ms, 1),
            "llm_calls": self.usage.calls,
            "cached_calls": self.usage.cached,
            "tokens": self.usage.total_tokens,
            "cost_usd": round(self.usage.cost_usd, 6),
        }


class GraphBuilder:
    """Runs the GraphRAG ingest pipeline and persists it to Neo4j.

    Every stage is injectable so a caller can substitute one (a cached
    extractor, a resolver with a different embedder) without reimplementing the
    ordering, which is the part that is easy to get wrong.

    The per-stage ``graph.*`` toggles are honoured; ``graph.enabled`` is not
    consulted, because that flag is the *pipeline's* gate on whether to build a
    graph at all and calling this class is already that decision.
    """

    def __init__(
        self,
        llm: LLM,
        store: Neo4jStore,
        *,
        embedder: DenseEmbedder | None = None,
        settings: Settings | None = None,
        router: ModelRouter | None = None,
        extractor: EntityExtractor | None = None,
        resolver: EntityResolver | None = None,
        detector: CommunityDetector | None = None,
        summarizer: CommunitySummarizer | None = None,
        validator: DocumentValidator | None = None,
    ) -> None:
        self.llm = llm
        self.store = store
        self.settings = settings or get_settings()
        self.cfg = self.settings.graph
        self.router = router or ModelRouter(self.settings.llm)
        self.extractor = extractor or EntityExtractor(llm, self.settings, router=self.router)
        self.resolver = resolver or EntityResolver(embedder, self.settings)
        self.detector = detector or CommunityDetector(self.settings)
        self.summarizer = summarizer or CommunitySummarizer(llm, self.settings, router=self.router)
        self.validator = validator or DocumentValidator(self.settings)

    async def build(self, chunks: Sequence[Chunk]) -> GraphBuildReport:
        """Build and persist the graph for ``chunks``."""
        report = GraphBuildReport(chunks_in=len(chunks))
        usable = self._usable_chunks(chunks, report)
        report.chunks_used = len(usable)
        if not usable:
            log.info("graph_build_skipped", reason="no_usable_chunks", chunks_in=len(chunks))
            return report

        entities, relations = await self._extract(usable, report)
        if not entities:
            log.warning("graph_build_empty", chunks=len(usable), reason="no_entities_extracted")
            return report

        entities, relations = await self._resolve(entities, relations, report)
        communities = await self._communities(entities, relations, report)
        self._attach_communities(entities, communities)
        await self._persist(entities, relations, communities, report)

        log.info("graph_built", **report.summary())
        return report

    # -- stages ------------------------------------------------------------
    def _usable_chunks(self, chunks: Sequence[Chunk], report: GraphBuildReport) -> list[Chunk]:
        """Drop chunks that cannot yield a graph before paying for them.

        Extraction is one LLM call per chunk, so the cheapest possible filter
        runs first: the ingest validator already rejects whitespace-only and
        below-minimum chunks, and spending a model call to discover that a chunk
        is three punctuation marks is pure waste.
        """
        with timed("graph_validate_chunks") as timer:
            usable = self.validator.validate_chunks(list(chunks))
        report.timings_ms["validate"] = timer.elapsed_ms
        if len(usable) != len(chunks):
            log.info(
                "graph_chunks_filtered",
                chunks_in=len(chunks),
                chunks_used=len(usable),
                dropped=len(chunks) - len(usable),
            )
        return usable

    async def _extract(
        self, chunks: Sequence[Chunk], report: GraphBuildReport
    ) -> tuple[list[Entity], list[Relation]]:
        with timed("graph_extract", chunks=len(chunks)) as timer:
            extraction, usage = await self.extractor.extract(chunks)
        report.timings_ms["extract"] = timer.elapsed_ms
        report.usage = report.usage + usage
        report.entity_mentions = len(extraction.entities)
        report.dangling_relations += extraction.dangling_dropped
        report.stages["extract"] = extraction.summary()
        return extraction.entities, extraction.relations

    async def _resolve(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        report: GraphBuildReport,
    ) -> tuple[list[Entity], list[Relation]]:
        with timed("graph_resolve", entities=len(entities)) as timer:
            resolution = await self.resolver.resolve(entities, relations)
        report.timings_ms["resolve"] = timer.elapsed_ms
        report.entities = len(resolution.entities)
        report.relations = len(resolution.relations)
        report.merged_entities = resolution.merged
        report.dangling_relations += resolution.dangling_dropped
        report.stages["resolve"] = resolution.summary()
        return resolution.entities, resolution.relations

    async def _communities(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        report: GraphBuildReport,
    ) -> list[Community]:
        with timed("graph_detect_communities", entities=len(entities)) as timer:
            hierarchy = await self.detector.detect(entities, relations)
        report.timings_ms["detect"] = timer.elapsed_ms
        report.communities = len(hierarchy.communities)
        report.community_levels = hierarchy.levels
        report.stages["detect"] = hierarchy.summary()
        if not hierarchy.communities:
            return []

        with timed("graph_summarize_communities", communities=len(hierarchy.communities)) as timer:
            summarization, usage = await self.summarizer.summarize(
                hierarchy.communities, entities, relations
            )
        report.timings_ms["summarize"] = timer.elapsed_ms
        report.usage = report.usage + usage
        report.stages["summarize"] = summarization.summary()
        return summarization.communities

    @staticmethod
    def _attach_communities(entities: Sequence[Entity], communities: Sequence[Community]) -> None:
        """Stamp each entity with its finest-grained community.

        The deepest level wins: a level-2 community is a more specific answer to
        "what is this entity part of" than the level-0 blob that contains it, and
        the coarser memberships are still reachable through the ``IN_COMMUNITY``
        edges and the communities' own ``parent_id`` chain.
        """
        best: dict[str, tuple[int, int]] = {}
        for community in communities:
            for name in community.entity_names:
                key = name.casefold()
                current = best.get(key)
                if current is None or community.level > current[0]:
                    best[key] = (community.level, community.id)
        for entity in entities:
            found = best.get(entity.key)
            if found is not None:
                entity.community_id = found[1]

    # -- persistence -------------------------------------------------------
    async def _persist(
        self,
        entities: Sequence[Entity],
        relations: Sequence[Relation],
        communities: Sequence[Community],
        report: GraphBuildReport,
    ) -> None:
        with timed("graph_ensure_schema") as timer:
            # Uniqueness on ``Entity.name`` is what makes every ``MERGE`` below
            # an index lookup instead of a label scan, so the schema is applied
            # before the first write rather than assumed.
            report.schema_applied = await self.store.ensure_schema()
        report.timings_ms["ensure_schema"] = timer.elapsed_ms

        batch = max(1, self.settings.indexing.batch_size)

        with timed("graph_write_entities", entities=len(entities)) as timer:
            for window in _windows(entities, batch):
                report.entities_written += await self.store.upsert_entities(window)
        report.timings_ms["write_entities"] = timer.elapsed_ms

        with timed("graph_write_relations", relations=len(relations)) as timer:
            for window in _windows(relations, batch):
                report.relations_written += await self.store.upsert_relations(window)
        report.timings_ms["write_relations"] = timer.elapsed_ms

        links = _chunk_links(entities)
        with timed("graph_write_chunk_links", entities=len(links)) as timer:
            for link_window in _map_windows(links, batch):
                report.chunk_links_written += await self.store.upsert_chunk_links(link_window)
        report.timings_ms["write_chunk_links"] = timer.elapsed_ms

        if communities:
            with timed("graph_write_communities", communities=len(communities)) as timer:
                for window in _windows(communities, batch):
                    report.communities_written += await self.store.upsert_communities(window)
                # After the loop, not inside it: the windows are message-size
                # slices of one partition, so "which communities does this build
                # produce" is only answerable once every window has been written.
                # Pruning per window would delete the communities the next window
                # is about to write.
                report.communities_pruned = await self.store.prune_communities(
                    keep_ids=[c.id for c in communities],
                    entity_names=[e.name for e in entities],
                )
            report.timings_ms["write_communities"] = timer.elapsed_ms
        elif entities:
            # Detection produced nothing. That is not evidence the existing
            # communities are stale — an extraction failure looks identical from
            # here — so nothing is pruned, and the reason is logged rather than
            # inferred later from an unexpectedly empty global search.
            log.info("graph_communities_not_pruned", reason="no_communities_detected")


def _windows(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
    """Fixed-size slices. Slices, not copies of the elements.

    The batch ceiling is about message size, not round trips: the store happily
    merges 10 000 rows in one statement, but 10 000 entity rows carrying 384-float
    embeddings is a several-hundred-megabyte Bolt message, and the driver buffers
    it whole.
    """
    return [items[start : start + size] for start in range(0, len(items), size)]


def _chunk_links(entities: Sequence[Entity]) -> dict[str, tuple[str, ...]]:
    """``entity name -> chunk ids`` for the ``(:Chunk)-[:MENTIONS]->(:Entity)`` edges.

    These edges are what makes a graph hit citable: local search matches an
    entity, then walks ``MENTIONS`` backwards to the text that asserted it. Built
    from the *resolved* entities, so a chunk that mentioned "Acme Corp" links to
    the canonical "Acme Corporation" node — linking the pre-resolution name would
    create the phantom node that resolution just finished eliminating.
    """
    return {
        entity.name: entity.source_chunk_ids
        for entity in entities
        if entity.name and entity.source_chunk_ids
    }


def _map_windows(
    mapping: Mapping[str, tuple[str, ...]], size: int
) -> list[dict[str, tuple[str, ...]]]:
    items = list(mapping.items())
    return [dict(items[start : start + size]) for start in range(0, len(items), size)]
