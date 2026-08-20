"""Community detection: turning a knowledge graph into a summarizable hierarchy.

GraphRAG's *global* search does not read the graph. It reads community reports —
one summary per cluster of densely connected entities — because a question like
"what are the main themes in this corpus?" has no entry point: there is no entity
to match and no chunk that contains the answer. The answer only exists in the
graph's shape, and communities are how that shape is made readable.

**Why Leiden and not Louvain.** Louvain has a known defect: it can produce
internally *disconnected* communities — a member with no path to the rest of its
own cluster (Traag et al., 2019). For us that is not a theoretical blemish, it is
a bad summary, because the report for that community describes entities that have
nothing to do with each other. Leiden guarantees connected communities and
converges to a partition where every subset is locally optimal. It is also faster
on large graphs, so there is no trade to make.

**Why hierarchical.** One flat partition forces a resolution choice that is wrong
at one end or the other: coarse communities are too broad to summarize usefully,
fine ones are too numerous for global search to read. Re-partitioning each
community that is large enough to have internal structure gives both — level 0
for "what is this corpus about", deeper levels for "what are the parts of this
theme". Re-running on the *induced subgraph* is what makes the split possible at
all: the RB-configuration null model is defined against the graph's total edge
weight, so the same resolution parameter is effectively finer once the rest of
the graph is removed.

**Why the whole thing runs in a thread.** Leiden on 100k nodes is seconds of pure
C, and the recursion adds a graph construction per community. Left on the event
loop it stalls every other coroutine in the process for that whole time, which
during an ingest means every concurrent document stops.

**Why ids are content-derived.** ``Community.id`` is a hash of the level plus the
sorted member names, not a counter. A counter shifts every id when one entity
joins the graph, which orphans every stored summary and forces a full
re-summarization — the most expensive stage in the pipeline. Hashing means an
unchanged community keeps its id, its ``MERGE`` lands on the existing node, and
its summary survives the re-run. Every randomized step is seeded, so identical
input yields an identical partition and therefore an identical id.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import structlog

from ragorc.core.ids import content_hash
from ragorc.core.models import Community, Entity, Relation
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["CommunityDetector", "CommunityHierarchy", "community_id"]

_IMPORT_HINT = 'community detection needs the graphrag extra: pip install "ragorc[graphrag]"'

#: Fixed seed for every randomized step. Community ids are derived from
#: membership, so an unseeded partition would renumber half the graph on a
#: re-run and discard every community summary that was paid for.
_SEED = 20_240_919

#: 53 bits: comfortably inside Neo4j's signed 64-bit integer, and inside the
#: exactly-representable range of a float64, so a JSON round trip through any
#: client cannot corrupt an id.
_ID_MASK = (1 << 53) - 1

_EdgeKey = tuple[int, int]
_RelationKey = tuple[str, str, str]


@dataclass(slots=True)
class CommunityHierarchy:
    """Detected communities plus the accounting for the detection run."""

    communities: list[Community]
    algorithm: str
    backend: str
    levels: int
    nodes: int
    edges: int
    dropped_small: int

    def by_level(self) -> dict[int, list[Community]]:
        """Communities grouped by level, deepest level last."""
        out: dict[int, list[Community]] = {}
        for community in self.communities:
            out.setdefault(community.level, []).append(community)
        return dict(sorted(out.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "communities": len(self.communities),
            "algorithm": self.algorithm,
            "backend": self.backend,
            "levels": self.levels,
            "nodes": self.nodes,
            "edges": self.edges,
            "dropped_small": self.dropped_small,
        }


class _Partitioner(Protocol):
    """Partitions an induced subgraph, in *global* node indices."""

    name: str

    def partition(self, nodes: Sequence[int]) -> list[list[int]]: ...


class CommunityDetector:
    """Builds the community hierarchy from resolved entities and relations."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.graph

    async def detect(
        self, entities: Sequence[Entity], relations: Sequence[Relation]
    ) -> CommunityHierarchy:
        """Detect communities across ``graph.max_community_levels`` levels."""
        if not self.cfg.detect_communities or not entities:
            return CommunityHierarchy([], self.cfg.community_algorithm, "none", 0, 0, 0, 0)

        hierarchy = await asyncio.to_thread(self._detect_sync, list(entities), list(relations))
        log.info("graph_communities_detected", **hierarchy.summary())
        return hierarchy

    # -- everything below runs in the worker thread ------------------------
    def _detect_sync(self, entities: list[Entity], relations: list[Relation]) -> CommunityHierarchy:
        names = [entity.name for entity in entities]
        types = [entity.type for entity in entities]
        index_of = {name.casefold(): i for i, name in enumerate(names)}

        graph = _WeightedGraph.build(relations, index_of, node_count=len(names))
        if graph.skipped:
            # Post-resolution this should be zero. A non-zero count means an
            # edge survived with an endpoint that is not an entity, and that
            # edge is invisible to every community it should have joined.
            log.warning(
                "graph_community_edges_skipped", skipped=graph.skipped, edges=len(graph.edges)
            )
        if not graph.edges:
            log.info("graph_community_no_edges", nodes=len(names))
            return CommunityHierarchy([], self.cfg.community_algorithm, "none", 0, len(names), 0, 0)

        partitioner, backend = _build_partitioner(
            len(names),
            graph.edges,
            graph.weights,
            self.cfg.community_algorithm,
            self.cfg.leiden_resolution,
        )

        max_levels = max(1, self.cfg.max_community_levels)
        min_size = max(1, self.cfg.min_community_size)
        communities: list[Community] = []
        internal_weights: list[float] = []
        dropped = 0

        # Iterative rather than recursive: a pathological graph could nest
        # deeper than Python's recursion limit, and the queue keeps the level
        # bookkeeping explicit.
        queue: list[tuple[list[int], int, int | None]] = [(list(range(len(names))), 0, None)]
        while queue:
            members, level, parent_id = queue.pop(0)
            parts = partitioner.partition(members)
            if level > 0 and len(parts) <= 1:
                # The parent has no internal structure left. Recording it again
                # one level down duplicates a community and doubles the
                # summarization bill for no extra information.
                continue
            for part in parts:
                if len(part) < min_size:
                    dropped += 1
                    continue
                keys, weight = graph.internals(part)
                communities.append(
                    Community(
                        id=community_id(level, [names[i] for i in part]),
                        level=level,
                        entity_names=tuple(names[i] for i in part),
                        relation_keys=keys,
                        title=_provisional_title(part, names, types),
                        parent_id=parent_id,
                    )
                )
                internal_weights.append(weight)
                # Splitting a community smaller than twice the floor can only
                # produce sub-communities below the floor, so the partition
                # would be computed and then thrown away in full.
                if level + 1 < max_levels and len(part) >= 2 * min_size:
                    queue.append((part, level + 1, communities[-1].id))

        _assign_ranks(communities, internal_weights)
        levels = 1 + max((c.level for c in communities), default=-1)
        return CommunityHierarchy(
            communities=communities,
            algorithm=self.cfg.community_algorithm,
            backend=backend,
            levels=levels,
            nodes=len(names),
            edges=len(graph.edges),
            dropped_small=dropped,
        )


# ---------------------------------------------------------------------------
# Graph plumbing
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _WeightedGraph:
    """The typed multigraph collapsed into a weighted simple graph.

    Community detection is about connection strength, not relationship
    semantics: two entities linked by both WORKS_FOR and FOUNDED are more
    strongly connected than two linked once, and modularity can only express
    that as a summed weight. The original typed keys are kept per edge so each
    community can still report exactly which relations it contains.

    Adjacency is materialized once and shared by every level. A community's
    internal weight is an intersection of its members with each member's
    neighbours; doing that against a flat edge list would rescan every edge once
    per community, which is the dominant cost at level 2 of a large graph.
    """

    edges: list[_EdgeKey]
    weights: list[float]
    neighbours: list[list[int]]
    weight_of: dict[_EdgeKey, float]
    keys_of: dict[_EdgeKey, tuple[_RelationKey, ...]]
    skipped: int

    @classmethod
    def build(
        cls,
        relations: Sequence[Relation],
        index_of: Mapping[str, int],
        *,
        node_count: int,
    ) -> _WeightedGraph:
        merged: dict[_EdgeKey, float] = {}
        keys: dict[_EdgeKey, list[_RelationKey]] = {}
        skipped = 0
        for relation in relations:
            source = index_of.get(relation.source.casefold())
            target = index_of.get(relation.target.casefold())
            if source is None or target is None:
                skipped += 1
                continue
            if source == target:
                continue
            pair = (source, target) if source < target else (target, source)
            merged[pair] = merged.get(pair, 0.0) + max(float(relation.weight), 0.0)
            keys.setdefault(pair, []).append(relation.key)

        # Sorted edge order makes the igraph/networkx construction — and hence
        # the partition — a pure function of the input set rather than of dict
        # iteration order.
        edges = sorted(merged)
        neighbours: list[list[int]] = [[] for _ in range(node_count)]
        for source, target in edges:
            neighbours[source].append(target)
            neighbours[target].append(source)
        return cls(
            edges=edges,
            weights=[merged[pair] for pair in edges],
            neighbours=neighbours,
            weight_of=merged,
            keys_of={pair: tuple(items) for pair, items in keys.items()},
            skipped=skipped,
        )

    def internals(self, members: Sequence[int]) -> tuple[tuple[_RelationKey, ...], float]:
        """Relation keys and summed weight of edges *inside* a member set.

        Boundary edges are excluded: they describe the community's relationship
        to the rest of the graph, and feeding them to the report as evidence
        would let a summary make claims about entities that are not members.
        """
        inside = set(members)
        keys: list[_RelationKey] = []
        total = 0.0
        for node in members:
            for neighbour in self.neighbours[node]:
                if neighbour <= node or neighbour not in inside:
                    continue
                pair = (node, neighbour)
                keys.extend(self.keys_of.get(pair, ()))
                total += self.weight_of.get(pair, 0.0)
        return tuple(keys), total


def _build_partitioner(
    node_count: int,
    edges: Sequence[_EdgeKey],
    weights: Sequence[float],
    algorithm: str,
    resolution: float,
) -> tuple[_Partitioner, str]:
    """Pick a backend, degrading explicitly rather than silently.

    Leiden needs both ``python-igraph`` (the C core) and ``leidenalg``. When
    either is missing the request degrades to networkx greedy modularity — the
    only community algorithm networkx offers that needs nothing beyond networkx
    itself — and says so in the log, because a partition produced by a different
    algorithm than the configured one is something an operator has to know.
    """
    if algorithm == "leiden":
        try:
            return _IGraphLeiden(node_count, edges, weights, resolution), "igraph+leidenalg"
        except ImportError as exc:
            log.warning(
                "graph_leiden_unavailable",
                error=str(exc)[:200],
                fallback="networkx_greedy_modularity",
                hint=_IMPORT_HINT,
            )
            return (
                _NetworkxPartitioner(node_count, edges, weights, "greedy", resolution),
                "networkx",
            )
    return _NetworkxPartitioner(node_count, edges, weights, algorithm, resolution), "networkx"


class _IGraphLeiden:
    """Hierarchical Leiden over an igraph core.

    The whole graph is built once; each level partitions an induced subgraph.
    igraph returns induced-subgraph vertices in ascending order of their id in
    the parent graph, *not* in the order they were requested, so the member list
    is sorted first and results are mapped back through that sorted list.
    Getting this wrong silently permutes community membership — every community
    would contain the right number of the wrong entities.
    """

    name = "leiden"

    def __init__(
        self,
        node_count: int,
        edges: Sequence[_EdgeKey],
        weights: Sequence[float],
        resolution: float,
    ) -> None:
        import igraph
        import leidenalg

        self._leidenalg = leidenalg
        self._resolution = float(resolution)
        graph = igraph.Graph(n=node_count, edges=list(edges), directed=False)
        graph.es["weight"] = list(weights)
        self._graph = graph

    def partition(self, nodes: Sequence[int]) -> list[list[int]]:
        members = sorted(nodes)
        if len(members) < 2:
            return [list(members)]
        sub = self._graph.induced_subgraph(members)
        found = self._leidenalg.find_partition(
            sub,
            self._leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=self._resolution,
            n_iterations=2,
            seed=_SEED,
        )
        return _order_parts([[members[i] for i in group] for group in found])


class _NetworkxPartitioner:
    """networkx backend: louvain, label propagation, or greedy modularity.

    Label propagation uses the *semi-synchronous* variant deliberately.
    networkx's asynchronous variant shuffles node order and takes a random seed;
    the semi-synchronous one is deterministic by construction, and determinism is
    what keeps community ids — and therefore the summaries already paid for —
    stable across runs.
    """

    def __init__(
        self,
        node_count: int,
        edges: Sequence[_EdgeKey],
        weights: Sequence[float],
        algorithm: str,
        resolution: float,
    ) -> None:
        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - guarded by the extra
            raise ImportError(_IMPORT_HINT) from exc
        self._nx = nx
        self.name = algorithm
        self._resolution = float(resolution)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_count))
        graph.add_weighted_edges_from(
            (source, target, value) for (source, target), value in zip(edges, weights, strict=True)
        )
        self._graph = graph

    def partition(self, nodes: Sequence[int]) -> list[list[int]]:
        members = sorted(nodes)
        if len(members) < 2:
            return [list(members)]
        # A subgraph *view*: no copy of node or edge data, which matters because
        # the level walk takes one of these per community.
        sub = self._graph.subgraph(members)
        community = self._nx.community
        if self.name == "louvain":
            groups = community.louvain_communities(
                sub, weight="weight", resolution=self._resolution, seed=_SEED
            )
        elif self.name == "label_propagation":
            groups = community.label_propagation_communities(sub)
        else:
            groups = community.greedy_modularity_communities(
                sub, weight="weight", resolution=self._resolution
            )
        return _order_parts([[int(node) for node in group] for group in groups])


def _order_parts(parts: list[list[int]]) -> list[list[int]]:
    """Sort members and parts so the partition is stable across runs."""
    ordered = [sorted(part) for part in parts if part]
    ordered.sort(key=lambda part: part[0])
    return ordered


# ---------------------------------------------------------------------------
# Community metadata
# ---------------------------------------------------------------------------
def community_id(level: int, member_names: Sequence[str]) -> int:
    """Deterministic, content-derived community id.

    Level is part of the digest: a community whose only child has identical
    membership is a different object at each level, and colliding the two would
    make a node its own parent.
    """
    digest = content_hash("community", level, sorted(member_names), size=8)
    return int(digest, 16) & _ID_MASK


def _provisional_title(members: Sequence[int], names: Sequence[str], types: Sequence[str]) -> str:
    """A title before the LLM writes one.

    Not decoration: when summarization is disabled or a report call fails, this
    is the only human-readable handle the community has, and an untitled
    community is unusable in a global-search trace.
    """
    head = [names[i] for i in members[:3]]
    kinds = {types[i] for i in members if types[i]}
    label = ", ".join(head)
    if len(members) > 3:
        label = f"{label} +{len(members) - 3} more"
    kind = next(iter(sorted(kinds))) if len(kinds) == 1 else ""
    return f"{kind}: {label}" if kind else label


def _assign_ranks(communities: Sequence[Community], internal_weights: Sequence[float]) -> None:
    """Structural importance, from size and summed internal edge weight.

    Both inputs are heavy-tailed — one hub community can hold a tenth of the
    graph — so they are compressed with ``log1p`` before scaling. Without that,
    the largest community takes the whole [0, 1] range and every other community
    flattens to approximately zero, which makes rank useless as a priority for
    global search.

    The two signals are weighted equally because each is wrong alone: size
    favours sprawling, weakly connected blobs, and weight favours a tight
    triangle of three heavily corroborated entities over a genuinely central
    fifty-entity cluster. Scaled to 0-10 to match the LLM's own significance
    rating, which the summarizer blends with this.
    """
    if not communities:
        return
    sizes = np.fromiter(
        (len(c.entity_names) for c in communities), dtype=np.float64, count=len(communities)
    )
    weights = np.asarray(internal_weights, dtype=np.float64)
    rank = 10.0 * (0.5 * _scale(np.log1p(sizes)) + 0.5 * _scale(np.log1p(weights)))
    for community, value in zip(communities, rank, strict=True):
        community.rank = float(value)


def _scale(values: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a degenerate range scores everything equally."""
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)
