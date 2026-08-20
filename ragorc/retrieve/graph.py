"""GraphRAG retrieval: three search modes, because one does not cover the space.

A knowledge graph answers questions a vector index structurally cannot, but *how*
you query it depends entirely on the shape of the question, and the three shapes
need three different algorithms. Microsoft's GraphRAG names them local, global and
DRIFT; all three are implemented here over the same Neo4j graph.

**Local** — "what do we know about X, and what is X connected to?" The question
names an entity, so the graph is entered at that node and the answer is assembled
from its neighbourhood: the chunks that mention it, plus the relationships that
explain *why* those chunks are relevant together. This is the mode that fixes
vector search's blind spot on connective facts — a passage saying "Acme acquired
Beta" and a passage saying "Beta built the payments engine" are individually poor
matches for "who owns the payments engine", and no reranker recovers the join.

**Global** — "what are the main themes?" No entity is named, and no single chunk
contains the answer, because the answer is a property of the corpus rather than of
any passage in it. Vector search over chunks cannot answer it at any ``top_k``:
retrieving 10 of 50 000 chunks tells you about 10 chunks. Global search instead
reads the *community summaries* — the LLM-written reports over Leiden clusters —
and map-reduces over them, so every part of the corpus is represented by something
short enough to fit.

**DRIFT** — the gap between the two. Local search needs the question to name an
entity the index can match; it fails on descriptive questions ("what caused the
latency spike in checkout") where no proper noun appears. Global search never
descends below a community summary, so it answers thematically and loses the
specific, citable fact. DRIFT seeds with vector search — which needs no entity
name and finds the passage by meaning — then expands the graph around whatever
entities that passage turned out to be about. It gets local search's relationship
evidence on questions local search cannot start, and chunk-level evidence on
questions only global search could otherwise reach.

Why entity matching does not get an LLM call
--------------------------------------------
The obvious implementation of "find the entities in the question" is an extraction
call. It is the wrong tool, for four reasons that compound:

1. **It is on the critical path of every graph query.** 300-800 ms and a token
   bill added to every request, before any retrieval has started.
2. **Its output has to be reconciled anyway.** An extractor returns surface forms
   from the question. The graph is keyed on *canonical* names produced by entity
   resolution during ingest. "the Acme acquisition" has to become the node
   ``Acme Corporation``, and an extractor cannot know that — only the graph's own
   name index can.
3. **The index already is the answer.** Neo4j's full-text index over entity name,
   aliases and description is a Lucene index: it stems, it tokenizes, it scores by
   relevance, and it matches the alias variants that entity resolution merged. One
   round trip, no tokens, and the score it returns is a usable match confidence.
4. **Its failure mode is silent.** A hallucinated or non-canonical name matches no
   node, and the retriever returns nothing while looking like it worked.

So: full-text index for matching, LLM calls only where reasoning is actually
required — the global map step, and the multi-hop sufficiency check in
:mod:`ragorc.retrieve.multihop`.

Why the subgraph is verbalized into the chunk text
--------------------------------------------------
The graph's whole contribution is *edges*, and an edge is not in any chunk's prose
in a form the generator can use — the sentence that asserted it has been through
extraction and normalization since. Returning only the matched chunks therefore
throws away the reason they were matched: the generator sees three passages about
three entities and has to re-derive the connection that the traversal already
found. So each returned chunk is prefixed with the entities and relationships tied
to *it*, and the traversal as a whole is emitted as its own leading chunk. The
second one is not redundant: a two-hop path runs through edges asserted by
different documents, so the *path* is evidence that no single chunk carries.

Scoring
-------
Every score in this module is higher-is-better. Lucene relevance and relationship
weight already are; cosine similarity is computed directly (never a distance).
Signals are combined by max-scaling each one and taking a weighted mean over the
terms that are actually available — a chunk store that cannot return vectors makes
the similarity term unavailable rather than zero, and its weight is redistributed
instead of penalizing every candidate equally.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import structlog

from ragorc.core.concurrency import safe_gather
from ragorc.core.errors import RetrievalError, StoreUnavailable
from ragorc.core.ids import chunk_id
from ragorc.core.models import (
    Chunk,
    Community,
    Entity,
    FloatArray,
    GraphPath,
    Modality,
    Query,
    Relation,
    RetrievalSource,
    ScoredChunk,
    Usage,
)
from ragorc.core.protocols import LLM, VectorStore
from ragorc.core.registry import register
from ragorc.core.schemas import MapAnswer
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import Timer, timed, trace_step
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = [
    "ChunkStore",
    "GraphDriftRetriever",
    "GraphGlobalRetriever",
    "GraphLocalRetriever",
    "GraphSearchStore",
    "load_chunks",
    "max_scale",
    "verbalize_entities",
    "verbalize_relations",
    "verbalize_subgraph",
]

# --- blend weights ---------------------------------------------------------
# Local search ranks a chunk by three signals with genuinely different meanings,
# so the weights encode an ordering of trust rather than a tuned optimum:
#
#   entity match      how confident are we that the question is about an entity
#                     this chunk mentions. Strongest signal, because it is the
#                     only one that connects the chunk to the *question*.
#   relationship      how well attested are the edges this chunk asserted. An
#                     edge asserted by twenty documents outranks one asserted
#                     once, which is exactly what makes graph evidence citable.
#   chunk similarity  ordinary dense relevance. Kept, and kept last, because the
#                     traversal can reach a chunk that mentions the right entity
#                     while discussing something else entirely; similarity is the
#                     term that demotes it.
#
# Override per deployment through the constructor; there is no settings field for
# a graph blend and inventing one is not this module's call.
_W_ENTITY = 0.40
_W_RELATION = 0.25
_W_SIMILARITY = 0.35

_HOP_DECAY = 0.6
"""Per-hop attenuation of an entity's match weight.

Relevance falls off fast with graph distance: at two hops from a hub node you are
already reachable from most of the corpus, so an undecayed traversal ranks by
connectivity instead of by relevance. 0.6 leaves a two-hop neighbour at ~36% of
its seed, which keeps it in contention without letting it outrank a direct hit."""

_CHUNK_OVERSAMPLE = 3
"""Candidate chunks loaded per chunk returned.

The similarity term can reorder the ranking, so the top-k cannot be decided before
the chunk bodies are in hand — but a hub entity is mentioned by thousands of
chunks and loading all of them to return ten is the dominant cost of local search.
Three times the output is enough headroom for reordering to matter."""

_MAX_ENTITY_DESC_CHARS = 240
"""Per-entity description budget in the verbalized subgraph. Descriptions
accumulate across every chunk an entity appeared in, so an unbudgeted node can be
several kilobytes on its own and crowd the prose out of the context window."""

_DRIFT_PROBE_CHARS = 2000
"""How much seed text is fed to the entity index in DRIFT. The index scores by
term overlap, so a longer probe raises recall and flattens precision; two thousand
characters is roughly the top three chunks, which is where the seeds are still
about the question rather than about the corpus."""

_DRIFT_UNCONFIRMED_WEIGHT = 0.6
"""Match weight for an entity the index matched but whose ``source_chunk_ids`` do
not intersect the seed chunks. It is a plausible mention rather than a proven one,
so it seeds the traversal at a discount instead of being dropped — dropping it
would make DRIFT return nothing whenever the graph was built before the current
chunk set."""

_W_DRIFT_SEED = 0.45
_W_DRIFT_GRAPH = 0.55
"""Merge weights for DRIFT's two halves.

Neither sums to 1 alone, on purpose: a chunk found by *both* the vector seed and
the graph expansion can reach 1.0, and one found by a single path cannot. Agreement
between two independent retrieval mechanisms is real evidence, and this is where it
is priced in."""


# ---------------------------------------------------------------------------
# Store surfaces
# ---------------------------------------------------------------------------
@runtime_checkable
class GraphSearchStore(Protocol):
    """The graph operations GraphRAG search needs.

    Deliberately a structural protocol rather than an import of
    :class:`~ragorc.stores.neo4j.store.Neo4jStore`: two of these four methods
    (``fulltext_entities``, the ``limit`` on ``communities``) are not on
    :class:`~ragorc.core.protocols.GraphStore`, and hard-wiring the concrete class
    would make GraphRAG untestable without a Bolt driver and unusable over any
    other graph backend.
    """

    async def fulltext_entities(
        self, query: str, *, limit: int | None = None
    ) -> list[tuple[Entity, float]]: ...

    async def neighbors(
        self, names: Sequence[str], *, hops: int = 1, limit: int = 50
    ) -> tuple[list[Entity], list[Relation]]: ...

    async def communities(
        self, *, level: int | None = None, limit: int | None = None
    ) -> list[Community]: ...

    async def paths(
        self,
        start: Sequence[str],
        end: Sequence[str],
        *,
        max_hops: int = 3,
        limit: int = 10,
    ) -> list[GraphPath]: ...


@runtime_checkable
class ChunkStore(Protocol):
    """Whatever holds the chunk *text*.

    The graph stores chunk ids on entities and relationships, not chunk bodies —
    duplicating the prose into Neo4j would make it a third copy to keep in sync for
    no retrieval benefit. So local and DRIFT search resolve ids against the store
    that already has the text warm, which is normally Qdrant.
    """

    async def get(self, ids: Sequence[str], *, with_vectors: bool = False) -> list[Chunk]: ...


async def load_chunks(
    source: ChunkStore | Any, ids: Sequence[str], *, with_vectors: bool = False
) -> list[Chunk]:
    """Resolve chunk ids to chunk bodies from either store.

    Qdrant spells this ``get`` (the :class:`~ragorc.core.protocols.VectorStore`
    protocol); Postgres spells it ``get_chunks`` and has no vectors to hand back
    from the read path. Accepting both means a deployment can serve graph chunk
    text from whichever store it already runs, rather than being forced to keep
    Qdrant on the graph path.
    """
    wanted = list(dict.fromkeys(ids))
    if not wanted:
        return []
    getter = getattr(source, "get", None)
    if getter is not None:
        return await getter(wanted, with_vectors=with_vectors)
    getter = getattr(source, "get_chunks", None)
    if getter is None:
        raise RetrievalError(
            "chunk store exposes neither get() nor get_chunks()",
            store=type(source).__name__,
        )
    return await getter(wanted)


# ---------------------------------------------------------------------------
# Verbalization
# ---------------------------------------------------------------------------
def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit]}..."


def verbalize_relations(relations: Iterable[Relation]) -> str:
    """Render edges as ``A -[TYPE]-> B`` lines with weight and description.

    The arrow form matches :meth:`~ragorc.core.models.GraphPath.verbalize` so a
    relationship reads identically whether it arrived from a traversal or from a
    path search. The weight is included because it is the model's only cue that
    one assertion is better attested than another, and it is the difference
    between "sources agree" and "one source claims".
    """
    lines: list[str] = []
    for rel in relations:
        line = f"{rel.source} -[{rel.type}]-> {rel.target} (weight {rel.weight:.1f})"
        if rel.description:
            line = f"{line}: {_clip(rel.description, _MAX_ENTITY_DESC_CHARS)}"
        lines.append(line)
    return "\n".join(lines)


def verbalize_entities(entities: Iterable[Entity]) -> str:
    """Render nodes as ``Name [TYPE]: description`` lines."""
    lines: list[str] = []
    for entity in entities:
        line = f"{entity.name} [{entity.type}]"
        if entity.description:
            line = f"{line}: {_clip(entity.description, _MAX_ENTITY_DESC_CHARS)}"
        lines.append(line)
    return "\n".join(lines)


def verbalize_subgraph(
    entities: Sequence[Entity], relations: Sequence[Relation], *, title: str
) -> str:
    """One prompt-ready block for a traversal result."""
    parts = [title]
    if entities:
        parts.append(f"Entities ({len(entities)}):\n{verbalize_entities(entities)}")
    if relations:
        parts.append(f"Relationships ({len(relations)}):\n{verbalize_relations(relations)}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def max_scale(values: np.ndarray) -> np.ndarray:
    """Scale by the maximum, not min-max.

    Min-max normalization pins the weakest candidate at exactly 0, which asserts
    that it is irrelevant — false when every candidate is strong, and it destroys
    the ratios the blend is trying to combine. Dividing by the maximum keeps
    "half as good as the best" meaning half as good.
    """
    peak = float(values.max()) if values.size else 0.0
    return values / peak if peak > 0.0 else np.zeros_like(values)


def _propagate(
    seed_weights: Mapping[str, float], relations: Sequence[Relation], *, hops: int, decay: float
) -> dict[str, float]:
    """Spread seed match weight outward along the traversed edges.

    Relaxation rather than a plain BFS depth: a node's weight is the best
    ``seed_weight * decay ** distance`` over every seed that reaches it, and with
    a uniform decay ``hops`` rounds of max-relaxation compute that exactly. It
    matters because the seeds are *not* equally good — a weak seed one hop away
    should not outrank a strong seed two hops away just because it is closer.

    Traversal is undirected: the extractor's choice of edge direction is an
    artifact of sentence order, not a semantic constraint on "what is X related to".
    """
    adjacency: dict[str, set[str]] = {}
    for rel in relations:
        source, target = rel.source, rel.target
        if not source or not target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    weights = dict(seed_weights)
    frontier = set(seed_weights)
    for _ in range(max(hops, 0)):
        if not frontier:
            break
        nxt: set[str] = set()
        for node in frontier:
            candidate = weights.get(node, 0.0) * decay
            if candidate <= 0.0:
                continue
            for neighbour in adjacency.get(node, ()):
                if candidate > weights.get(neighbour, 0.0):
                    weights[neighbour] = candidate
                    nxt.add(neighbour)
        frontier = nxt
    return weights


def _cosine(query_vector: FloatArray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a stack of chunk vectors.

    One matmul over the whole stack rather than a loop, and clipped at 0 below:
    a negative cosine means "unrelated", and letting it stay negative would let
    the similarity term *subtract* from a chunk the graph strongly supports, which
    is a veto the weight was never meant to grant it.
    """
    q = np.asarray(query_vector, dtype=np.float32).ravel()
    q = q / max(float(np.linalg.norm(q)), 1e-9)
    norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    return np.clip((matrix / norms) @ q, 0.0, None)


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
@register("retriever", "graph_local")
class GraphLocalRetriever:
    """Entity-anchored search: match, expand, collect, rank."""

    name = "graph_local"

    def __init__(
        self,
        graph: GraphSearchStore,
        chunks: ChunkStore | None = None,
        *,
        weights: tuple[float, float, float] | None = None,
        hop_decay: float = _HOP_DECAY,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.graph = graph
        self.chunks = chunks
        self.weights = weights or (_W_ENTITY, _W_RELATION, _W_SIMILARITY)
        self.hop_decay = float(hop_decay)

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """Match question entities against the graph, then expand and rank.

        Returns nothing — rather than raising — when the question names no entity
        the graph knows. That is the expected outcome for a purely descriptive
        question, and it is the case DRIFT exists to cover; failing here would
        take down a fan-out that the vector leg was going to answer anyway.
        """
        cfg = self.settings.graph
        seeds = await self.graph.fulltext_entities(query.text, limit=cfg.local_search_top_entities)
        if not seeds:
            log.info("graph_local_no_entity_match", query_length=len(query.text))
            return []
        chunks, _ = await self.expand(query, seeds, top_k=top_k)
        return chunks

    async def expand(
        self,
        query: Query,
        seeds: Sequence[tuple[Entity, float]],
        *,
        top_k: int | None = None,
    ) -> tuple[list[ScoredChunk], dict[str, Any]]:
        """Expand from pre-matched seed entities and rank the chunks they reach.

        Split out from :meth:`retrieve` because DRIFT arrives with its seeds
        already chosen by a vector search, and everything after the match step is
        identical. Returns the chunks plus a diagnostics dict, so a composite
        retriever can report what the traversal found and what it lost.
        """
        cfg = self.settings.graph
        limit = int(top_k or cfg.local_search_top_chunks)
        seed_weights = self._seed_weights(seeds)
        detail: dict[str, Any] = {
            "seed_entities": list(seed_weights),
            "hops": cfg.local_search_hops,
            "degraded": [],
        }

        with timed("retrieve.graph_local", seeds=len(seed_weights), hops=cfg.local_search_hops):
            entities, relations = await self._neighbors(list(seed_weights), detail)
            # The seed nodes themselves must be in the pool even when the
            # expansion failed or the seed has no edges: a lone matched entity
            # with a good description is still the answer to "who is X".
            by_name = {e.name: e for e, _ in seeds if e.name}
            for entity in entities:
                by_name.setdefault(entity.name, entity)

            weights = _propagate(
                seed_weights, relations, hops=cfg.local_search_hops, decay=self.hop_decay
            )
            candidates = self._candidates(by_name, relations, weights, limit)
            detail["candidate_chunks"] = len(candidates)

            bodies = await self._load(list(candidates), detail)
            scored = self._rank(query, candidates, bodies, by_name, limit)

        subgraph = self._subgraph_chunk(query, by_name, relations, scored)
        out = ([subgraph] if subgraph is not None else []) + scored
        for rank, item in enumerate(out):
            item.rank = rank
        detail["returned"] = len(out)
        detail["relations"] = len(relations)
        log.info(
            "graph_local_retrieved",
            seeds=len(seed_weights),
            entities=len(by_name),
            relations=len(relations),
            candidates=len(candidates),
            returned=len(out),
            degraded=detail["degraded"],
        )
        return out, detail

    # -- steps -------------------------------------------------------------
    @staticmethod
    def _seed_weights(seeds: Sequence[tuple[Entity, float]]) -> dict[str, float]:
        """Normalize Lucene relevance into a per-seed match weight in ``(0, 1]``.

        Max-scaled rather than used raw: Lucene scores are corpus- and
        query-dependent with no upper bound, so the absolute value is not
        comparable across queries, while the ratio between two hits on the *same*
        query is exactly the signal wanted here.
        """
        named = [(entity.name, float(score)) for entity, score in seeds if entity.name]
        if not named:
            return {}
        raw = np.fromiter((s for _, s in named), dtype=np.float32, count=len(named))
        scaled = max_scale(raw)
        # A zero-relevance match still entered the graph, so it keeps a floor
        # rather than being silently excluded from propagation.
        return {
            name: max(float(value), 1e-3) for (name, _), value in zip(named, scaled, strict=True)
        }

    async def _neighbors(
        self, names: Sequence[str], detail: dict[str, Any]
    ) -> tuple[list[Entity], list[Relation]]:
        """Ego-network expansion, degrading to the seeds alone if it fails.

        This is a composite retriever: the seed match already succeeded, so a
        failed expansion should cost the *edges*, not the query. The loss is
        recorded so it is visible in the trace rather than looking like a graph
        with no relationships in it.
        """
        try:
            return await self.graph.neighbors(
                names,
                hops=self.settings.graph.local_search_hops,
                limit=self.settings.neo4j.max_cypher_rows,
            )
        except StoreUnavailable as exc:
            detail["degraded"].append("neighbors")
            log.warning("graph_local_expansion_degraded", error=str(exc)[:200])
            return [], []

    def _candidates(
        self,
        entities: Mapping[str, Entity],
        relations: Sequence[Relation],
        weights: Mapping[str, float],
        limit: int,
    ) -> dict[str, tuple[float, float, list[Relation]]]:
        """Collect candidate chunk ids with their entity and relationship signals.

        A chunk is a candidate if some weighted entity names it as a source, or
        some traversed edge was asserted by it. The two signals are kept apart
        (rather than pre-summed) because they are normalized independently: a hub
        entity can contribute a hundred chunks with identical entity weight, and
        only the edge evidence separates them.

        Truncated to ``_CHUNK_OVERSAMPLE * limit`` before the bodies are fetched,
        ranked by the signals already in hand — loading every chunk a hub entity
        mentions would dominate the cost of the whole query.
        """
        entity_signal: dict[str, float] = {}
        relation_signal: dict[str, float] = {}
        by_chunk: dict[str, list[Relation]] = {}

        for name, entity in entities.items():
            weight = weights.get(name, 0.0)
            if weight <= 0.0:
                continue
            for cid in entity.source_chunk_ids:
                if weight > entity_signal.get(cid, 0.0):
                    entity_signal[cid] = weight

        for rel in relations:
            endpoint = max(weights.get(rel.source, 0.0), weights.get(rel.target, 0.0))
            if endpoint <= 0.0:
                continue
            contribution = float(rel.weight) * endpoint
            for cid in rel.source_chunk_ids:
                relation_signal[cid] = relation_signal.get(cid, 0.0) + contribution
                by_chunk.setdefault(cid, []).append(rel)
                # An edge asserted by a chunk is also evidence that the chunk is
                # about the endpoints, even if the entity rows did not list it.
                entity_signal.setdefault(cid, endpoint * self.hop_decay)

        ranked = sorted(
            entity_signal,
            key=lambda cid: (entity_signal[cid], relation_signal.get(cid, 0.0)),
            reverse=True,
        )[: max(limit, 1) * _CHUNK_OVERSAMPLE]
        return {
            cid: (entity_signal[cid], relation_signal.get(cid, 0.0), by_chunk.get(cid, []))
            for cid in ranked
        }

    async def _load(self, ids: Sequence[str], detail: dict[str, Any]) -> dict[str, Chunk]:
        """Fetch chunk bodies, degrading to the verbalized subgraph if we cannot.

        Vectors are requested because the similarity term needs them; a store that
        does not carry them simply makes that term unavailable and its weight is
        redistributed over the other two.
        """
        if not ids or self.chunks is None:
            if ids and self.chunks is None:
                detail["degraded"].append("no_chunk_store")
            return {}
        try:
            bodies = await load_chunks(self.chunks, ids, with_vectors=True)
        except StoreUnavailable as exc:
            detail["degraded"].append("chunk_store")
            log.warning("graph_local_chunk_load_degraded", error=str(exc)[:200])
            return {}
        return {chunk.id: chunk for chunk in bodies}

    def _rank(
        self,
        query: Query,
        candidates: Mapping[str, tuple[float, float, list[Relation]]],
        bodies: Mapping[str, Chunk],
        entities: Mapping[str, Entity],
        limit: int,
    ) -> list[ScoredChunk]:
        """Blend the three signals and cut to ``limit``."""
        resolved = [cid for cid in candidates if cid in bodies]
        if not resolved:
            return []

        count = len(resolved)
        entity_term = np.fromiter(
            (candidates[cid][0] for cid in resolved), dtype=np.float32, count=count
        )
        relation_term = np.fromiter(
            (candidates[cid][1] for cid in resolved), dtype=np.float32, count=count
        )
        w_entity, w_relation, w_similarity = self.weights
        terms: list[tuple[float, np.ndarray]] = [
            (w_entity, max_scale(entity_term)),
            (w_relation, max_scale(relation_term)),
        ]

        vectors = [bodies[cid].dense for cid in resolved]
        similarity: np.ndarray | None = None
        if query.dense is not None and not any(v is None for v in vectors):
            similarity = _cosine(query.dense, np.asarray(vectors, dtype=np.float32))
            terms.append((w_similarity, max_scale(similarity)))

        total = sum(weight for weight, _ in terms) or 1.0
        # ``terms`` is never empty — the entity and relation terms are added
        # unconditionally above — so ``sum`` never falls back to the integer 0
        # that its signature also admits.
        weighted = cast("np.ndarray", sum(weight * values for weight, values in terms))
        blended = weighted / total

        order = np.argsort(-blended)[: max(limit, 0)]
        out: list[ScoredChunk] = []
        for rank, index in enumerate(order.tolist()):
            cid = resolved[index]
            chunk = bodies[cid]
            entity_score = float(entity_term[index])
            relation_score = float(relation_term[index])
            component: dict[str, float] = {
                "graph_entity": entity_score,
                "graph_relation": relation_score,
            }
            if similarity is not None:
                component["dense"] = float(similarity[index])
            related = candidates[cid][2]
            self._annotate(chunk, related, entities)
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(blended[index]),
                    source=RetrievalSource.GRAPH_LOCAL,
                    rank=rank,
                    component_scores=component,
                    explain={
                        "retriever": self.name,
                        "graph_relations": len(related),
                        "similarity_available": similarity is not None,
                    },
                )
            )
        return out

    def _annotate(
        self,
        chunk: Chunk,
        relations: Sequence[Relation],
        entities: Mapping[str, Entity],
    ) -> None:
        """Prefix a chunk's text with the graph context tied to that chunk.

        Mutating ``content`` is safe here: these ``Chunk`` objects were built by
        this call's own store read and are owned by it, exactly as the context
        packer's parent expansion relies on. The header is also kept in metadata so
        a citation validator can tell the graph preamble from the source prose.
        """
        if not relations:
            return
        names = {rel.source for rel in relations} | {rel.target for rel in relations}
        mentioned = [entities[name] for name in sorted(names) if name in entities]
        header = verbalize_subgraph(
            mentioned, list(relations), title="Graph context for this passage:"
        )
        chunk.metadata["graph_context"] = header
        chunk.content = f"{header}\n\n{chunk.content}"
        # The body changed, so any cached token count is now a lie the budgeter
        # would trust.
        chunk.token_count = None

    def _subgraph_chunk(
        self,
        query: Query,
        entities: Mapping[str, Entity],
        relations: Sequence[Relation],
        scored: Sequence[ScoredChunk],
    ) -> ScoredChunk | None:
        """Emit the whole traversal as its own leading chunk.

        Not redundant with the per-chunk annotations: a two-hop connection runs
        through edges asserted by different documents, so the shape of the
        neighbourhood is evidence that no individual chunk carries. It leads the
        result because it is the connective tissue for everything below it, and
        because it is the only evidence for edges whose source chunks were not
        retrieved.
        """
        if not entities and not relations:
            return None
        content = verbalize_subgraph(
            sorted(entities.values(), key=lambda e: e.name),
            list(relations),
            title=f"Knowledge graph neighbourhood for: {query.text}",
        )
        document = "graph:local"
        chunk = Chunk(
            id=chunk_id(document, 0, content),
            content=content,
            document_id=document,
            modality=Modality.SUMMARY,
            tenant_id=query.tenant_id or self.settings.tenant_id,
            metadata={
                "source": "graph_local",
                "entities": len(entities),
                "relations": len(relations),
                "hops": self.settings.graph.local_search_hops,
            },
        )
        # Sits at the top of the returned list by construction; the score matches
        # the best prose chunk so it does not distort a relative-score cutoff
        # applied downstream.
        score = max((s.score for s in scored), default=1.0)
        return ScoredChunk(
            chunk=chunk,
            score=score,
            source=RetrievalSource.GRAPH_LOCAL,
            rank=0,
            component_scores={"graph_subgraph": score},
            explain={"retriever": self.name, "verbalized_subgraph": True},
        )


# ---------------------------------------------------------------------------
# Global search
# ---------------------------------------------------------------------------
@register("retriever", "graph_global")
class GraphGlobalRetriever:
    """Map-reduce over community summaries.

    Only the **map** half lives here, and that is a deliberate seam. Map is
    retrieval: N independent, cheap, parallel calls that each ask "what does this
    community contribute to the question", producing scored partial answers that
    are exactly ``ScoredChunk`` shaped. Reduce is *generation* — one call that
    synthesizes the partials into prose — and it belongs to the generator, which
    already owns citation formatting, groundedness checking and abstention. Doing
    the reduce here would fork all of that.

    So this retriever returns the surviving partials, ranked, and the pipeline's
    generator combines them with the ``global_reduce`` prompt.
    """

    name = "graph_global"

    def __init__(
        self,
        llm: LLM,
        graph: GraphSearchStore,
        *,
        level: int | None = None,
        router: ModelRouter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.graph = graph
        self.level = level
        """Which community level to read. ``None`` takes the store's rank
        ordering across all levels, which is the right default: the useful level
        depends on how coarse the corpus's clustering came out, and rank already
        encodes importance."""
        self.router = router or ModelRouter(self.settings.llm)
        self.prompt = get_prompt("global_map")
        self.usage = Usage()

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        cfg = self.settings.graph
        cap = int(cfg.global_search_top_communities)
        self.usage = Usage()

        communities = [
            c for c in await self.graph.communities(level=self.level, limit=cap) if c.summary
        ]
        if not communities:
            log.info(
                "graph_global_no_communities",
                level=self.level,
                hint="run community detection and summarization during indexing",
            )
            return []
        # The store returns them rank-ordered, so the cap keeps the communities
        # that matter. The cap is the cost dial for this retriever: it is one LLM
        # call per community, every query.
        communities = communities[:cap]

        model = self.router.model_for(Task.GLOBAL_MAP)
        with Timer("retrieve.graph_global") as timer:
            results, errors = await safe_gather(
                [self._map_one(query, community, model) for community in communities],
                limit=max(1, self.settings.llm.max_concurrency),
                label="graph_global_map",
            )

        partials: list[ScoredChunk] = []
        for community, answer, usage in results:
            self.usage = self.usage + usage
            # A zero score is the prompt's explicit "this community contributes
            # nothing". Keeping it would poison the reduce step with confident
            # irrelevance, which is the failure mode the map prompt is written to
            # avoid — so the drop is not an optimization, it is the contract.
            if answer.score <= 0.0 or not answer.answer.strip():
                continue
            partials.append(self._to_chunk(query, community, answer))

        partials.sort(key=lambda s: s.score, reverse=True)
        limit = int(top_k or self.settings.retrieval.top_k)
        partials = partials[:limit]
        for rank, item in enumerate(partials):
            item.rank = rank

        trace_step(
            "retrieve.graph_global",
            duration_ms=timer.elapsed_ms,
            usage=self.usage,
            communities=len(communities),
            partials=len(partials),
            failed=len(errors),
        )
        log.info(
            "graph_global_retrieved",
            communities=len(communities),
            partials=len(partials),
            dropped=len(results) - len(partials),
            failed=len(errors),
            cost_usd=round(self.usage.cost_usd, 6),
        )
        return partials

    async def _map_one(
        self, query: Query, community: Community, model: str
    ) -> tuple[Community, MapAnswer, Usage]:
        """One community, one structured call.

        ``safe_gather`` rather than ``bounded_gather``: with eight or more
        independent calls per query, one provider hiccup would otherwise discard
        seven good partial answers. Global search degrades gracefully by
        construction — it is an aggregation, so it is still meaningful over the
        subset that returned.
        """
        report = community.summary
        if community.title:
            report = f"{community.title}\n\n{report}"
        answer, usage = await self.llm.structured(
            self.prompt.render(question=query.text, report=report),
            MapAnswer,
            system=self.prompt.system,
            model=model,
            stage="global_map",
        )
        return community, answer, usage

    def _to_chunk(self, query: Query, community: Community, answer: MapAnswer) -> ScoredChunk:
        content = answer.answer.strip()
        document = f"community:{community.id}"
        chunk = Chunk(
            id=chunk_id(document, 0, content, community.level),
            content=content,
            document_id=document,
            level=community.level,
            modality=Modality.SUMMARY,
            tenant_id=query.tenant_id or self.settings.tenant_id,
            metadata={
                "source": "graph_global",
                "community_id": community.id,
                "community_title": community.title,
                "community_rank": community.rank,
                "community_level": community.level,
                "entity_count": len(community.entity_names),
            },
        )
        # MapAnswer.score is the prompt's 0-10 helpfulness rating; divided by 10
        # it lands on the same [0, 1] higher-is-better scale as every other
        # retriever in the library, so fusion does not have to special-case it.
        score = float(answer.score) / 10.0
        return ScoredChunk(
            chunk=chunk,
            score=score,
            source=RetrievalSource.GRAPH_GLOBAL,
            rank=0,
            component_scores={"graph_global_map": score},
            explain={
                "retriever": self.name,
                "map_score_0_10": float(answer.score),
                "reduce_with": "global_reduce",
            },
        )


# ---------------------------------------------------------------------------
# DRIFT search
# ---------------------------------------------------------------------------
@register("retriever", "graph_drift")
class GraphDriftRetriever:
    """Vector seeds, then graph expansion around whatever they were about.

    The two halves cover each other's blind spot. Vector search finds the right
    passage without needing the question to name anything, but it cannot see the
    edge that joins two passages. The graph has the edges but has to be entered at
    a named node. Running vector first and using its hits as the entry point means
    neither limitation applies: an unnamed, descriptive question still lands on the
    graph, and the answer still comes back with the relationship evidence and the
    citable prose attached.
    """

    name = "graph_drift"

    def __init__(
        self,
        vector: VectorStore,
        graph: GraphSearchStore,
        chunks: ChunkStore | None = None,
        *,
        local: GraphLocalRetriever | None = None,
        seed_k: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector = vector
        self.graph = graph
        # The chunk store defaults to the vector store: DRIFT already holds one
        # that has the bodies and, unlike Postgres, the vectors the similarity
        # term wants.
        self.local = local or GraphLocalRetriever(graph, chunks or vector, settings=self.settings)
        self.seed_k = seed_k

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        cfg = self.settings.graph
        limit = int(top_k or self.settings.retrieval.top_k)
        seed_k = int(self.seed_k or cfg.local_search_top_chunks)

        with timed("retrieve.graph_drift", seed_k=seed_k):
            seeds = await self.vector.search(
                query,
                top_k=seed_k,
                filters=kwargs.get("filters") or query.filters or None,
                tenant_id=kwargs.get("tenant_id") or query.tenant_id,
            )
            if not seeds:
                log.info("graph_drift_no_seeds", seed_k=seed_k)
                return []

            entity_seeds, confirmed = await self._seed_entities(query, seeds)
            graph_hits: list[ScoredChunk] = []
            detail: dict[str, Any] = {}
            if entity_seeds:
                graph_hits, detail = await self.local.expand(query, entity_seeds, top_k=limit)

        merged = self._merge(seeds, graph_hits, limit)
        log.info(
            "graph_drift_retrieved",
            seeds=len(seeds),
            entities=len(entity_seeds),
            confirmed_entities=confirmed,
            graph_hits=len(graph_hits),
            returned=len(merged),
            degraded=detail.get("degraded", []),
        )
        return merged

    async def _seed_entities(
        self, query: Query, seeds: Sequence[ScoredChunk]
    ) -> tuple[list[tuple[Entity, float]], int]:
        """Find the graph entities the seed chunks are about.

        The mapping from chunk to entity is stored on the entity
        (``source_chunk_ids``), not on the chunk, so there is no direct lookup —
        and adding a Cypher traversal here would put query text in the retrieve
        layer, which is the store's job. Instead the seed *text* goes through the
        same entity full-text index local search uses: one round trip, no tokens,
        and it returns each candidate together with its source chunk ids.

        Those ids are then the confirmation. An entity whose ``source_chunk_ids``
        intersect the seed chunks is provably mentioned by the passage the vector
        search found, and seeds the traversal at full weight. One that merely
        matched the text is plausible but unproven and seeds at a discount, so
        DRIFT still works on a graph built from a different chunk generation
        instead of silently returning nothing.
        """
        probe_parts = [query.text]
        budget = _DRIFT_PROBE_CHARS
        for seed in seeds:
            if budget <= 0:
                break
            text = seed.content[:budget]
            probe_parts.append(text)
            budget -= len(text)
        probe = "\n".join(probe_parts)

        cap = max(1, self.settings.graph.local_search_top_entities) * 2
        try:
            hits = await self.graph.fulltext_entities(probe, limit=cap)
        except StoreUnavailable as exc:
            # Degrade to a plain vector result rather than failing: the seeds are
            # already a usable answer, they just have no relationship evidence.
            log.warning("graph_drift_entity_match_degraded", error=str(exc)[:200])
            return [], 0

        if not hits:
            return [], 0
        seed_ids = {s.chunk.id for s in seeds}
        confirmed = [
            (entity, score)
            for entity, score in hits
            if seed_ids.intersection(entity.source_chunk_ids)
        ]
        if confirmed:
            return confirmed, len(confirmed)
        keep = max(1, self.settings.graph.local_search_top_entities)
        return [(entity, score * _DRIFT_UNCONFIRMED_WEIGHT) for entity, score in hits[:keep]], 0

    def _merge(
        self,
        seeds: Sequence[ScoredChunk],
        graph_hits: Sequence[ScoredChunk],
        limit: int,
    ) -> list[ScoredChunk]:
        """Weighted merge of the vector and graph halves.

        Both sides are max-scaled first, because a cosine similarity and a graph
        blend are not on the same scale and a raw sum would let whichever happened
        to run hotter decide the ranking. A chunk reached by both paths keeps both
        component scores and can score up to 1.0; a chunk reached by one path is
        capped at that path's weight. That asymmetry is the point — agreement
        between two independent mechanisms is evidence, and it is priced here
        rather than left to a downstream reranker to rediscover.
        """
        seed_scaled = max_scale(
            np.fromiter((s.score for s in seeds), dtype=np.float32, count=len(seeds))
        )
        graph_scaled = max_scale(
            np.fromiter((s.score for s in graph_hits), dtype=np.float32, count=len(graph_hits))
        )

        merged: dict[str, ScoredChunk] = {}
        for scored, value in zip(seeds, seed_scaled.tolist(), strict=True):
            scored.score = _W_DRIFT_SEED * float(value)
            scored.component_scores.setdefault("drift_seed", float(value))
            scored.explain["retriever"] = self.name
            merged[scored.chunk.id] = scored

        for scored, value in zip(graph_hits, graph_scaled.tolist(), strict=True):
            contribution = _W_DRIFT_GRAPH * float(value)
            existing = merged.get(scored.chunk.id)
            if existing is None:
                scored.score = contribution
                scored.component_scores.setdefault("drift_graph", float(value))
                scored.explain["retriever"] = self.name
                merged[scored.chunk.id] = scored
                continue
            # Keep the graph-side object: its content already carries the
            # verbalized relationships, which is the whole reason to expand.
            scored.score = existing.score + contribution
            scored.component_scores.update(existing.component_scores)
            scored.component_scores["drift_graph"] = float(value)
            scored.source = RetrievalSource.FUSED
            scored.explain = {**existing.explain, **scored.explain, "retriever": self.name}
            merged[scored.chunk.id] = scored

        out = sorted(merged.values(), key=lambda s: s.score, reverse=True)[: max(limit, 0)]
        for rank, item in enumerate(out):
            item.rank = rank
        return out
