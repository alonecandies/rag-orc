"""GraphRAG query-side search: local, global and DRIFT.

The graph's whole contribution is the *edge*, and an edge is only worth paying for
if the traversal that found it is bounded, ordered and citable. So the assertions
here are on ids, order and hand-computed scores rather than on non-emptiness: a
traversal that quietly runs one hop too far, ranks by connectivity instead of
relevance, or loses the chunk ids that let an answer cite its source still returns
plausible-looking results, and no smoke test notices.

The three modes fail in three different ways, and each is covered:

* **local** enters at a matched entity, so its risks are hop-bound and decay;
* **global** is a map-reduce, so its risk is a partial answer silently replacing
  the others instead of joining them;
* **DRIFT** merges two independent halves, so its risk is one half's scale eating
  the other's ranking.

Degradation is tested as a first-class outcome, not an afterthought: every one of
these retrievers is a composite, and a dead traversal or a dead chunk store must
cost the *edges*, never the query.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ragorc.core.errors import RetrievalError, StoreUnavailable
from ragorc.core.models import (
    Chunk,
    Community,
    Entity,
    Modality,
    Query,
    Relation,
    RetrievalSource,
)
from ragorc.core.settings import Settings
from ragorc.retrieve.graph import (
    GraphDriftRetriever,
    GraphGlobalRetriever,
    GraphLocalRetriever,
    load_chunks,
    max_scale,
    verbalize_relations,
)
from tests.fakes import FakeGraphStore, FakeVectorStore, ScriptedLLM, StubLLM

# ---------------------------------------------------------------------------
# Doubles
#
# Both are thin extensions of the shared fakes rather than new ones: the shared
# graph fake was written for the ingest path and does not carry the two surfaces
# ``GraphSearchStore`` adds on top of ``GraphStore``, and the shared vector fake's
# ``get`` predates the ``with_vectors`` flag the graph chunk read passes.
# ---------------------------------------------------------------------------
_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    """Tokens the fake entity index matches on: lowercased, four characters or
    more, so a stop word shared by every description cannot score a hit."""
    return {t for t in _WORD.findall(text.lower()) if len(t) > 3}


class SearchableGraph(FakeGraphStore):
    """The shared graph fake plus a full-text entity index and a community cap.

    Matching is real term overlap over name, aliases and description, scored and
    ranked, rather than a canned hit list — a canned list would make every seeding
    and ordering assertion below a statement about the fixture instead of about the
    retriever. ``fail`` injects the store outage that the degradation paths exist
    for.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fail: set[str] = set()
        self.probes: list[str] = []
        self.expansions: list[list[str]] = []

    async def fulltext_entities(
        self, query: str, *, limit: int | None = None
    ) -> list[tuple[Entity, float]]:
        if "fulltext" in self.fail:
            raise StoreUnavailable("neo4j", "entity index offline")
        self.probes.append(query)
        terms = _terms(query)
        hits: list[tuple[Entity, float]] = []
        for entity in self.entities.values():
            names = _terms(entity.name) | {t for alias in entity.aliases for t in _terms(alias)}
            score = 2.0 * len(terms & names) + 0.5 * len(terms & _terms(entity.description))
            if score > 0.0:
                hits.append((entity, score))
        hits.sort(key=lambda hit: (-hit[1], hit[0].name))
        return hits[:limit] if limit else hits

    async def neighbors(
        self, names: Sequence[str], *, hops: int = 1, limit: int = 50
    ) -> tuple[list[Entity], list[Relation]]:
        if "neighbors" in self.fail:
            raise StoreUnavailable("neo4j", "traversal offline")
        self.expansions.append(list(names))
        return await super().neighbors(names, hops=hops, limit=limit)

    async def communities(
        self, *, level: int | None = None, limit: int | None = None
    ) -> list[Community]:
        found = sorted(await super().communities(level=level), key=lambda c: -c.rank)
        return found[:limit] if limit else found


class ChunkBodies(FakeVectorStore):
    """The shared vector fake, recording the ids requested and honouring
    ``with_vectors``, plus an outage switch for the degradation path."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = False
        self.requested: list[list[str]] = []
        self.requested_tenants: list[str | None] = []

    async def get(
        self,
        ids: Sequence[str],
        *,
        with_vectors: bool = False,
        tenant_id: str | None = None,
    ) -> list[Chunk]:
        if self.fail:
            raise StoreUnavailable("qdrant", "chunk store offline")
        self.requested.append(list(ids))
        self.requested_tenants.append(tenant_id)
        # Forwarded, not swallowed. Ignoring it made a real bug untestable: code
        # that stops asking for vectors still got them here, so removing the
        # similarity term left every test green.
        return await super().get(ids, with_vectors=with_vectors, tenant_id=tenant_id)


def graph_settings(**graph: Any) -> Settings:
    """Offline settings with the graph leg enabled; ``graph`` overrides the search
    knobs the retrievers read."""
    return Settings(
        environment="dev",
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "test-key", "max_concurrency": 4},
        embedding={"dense_dimension": 32},
        graph={"enabled": True, **graph},
    )


def _chunk(cid: str, document: str, content: str, dense: list[float] | None = None) -> Chunk:
    return Chunk(
        id=cid,
        content=content,
        document_id=document,
        dense=None if dense is None else np.asarray(dense, dtype=np.float32),
        token_count=11,
    )


async def build_chain() -> tuple[SearchableGraph, ChunkBodies]:
    """A four-node chain, one chunk per node, plus one chunk reachable only
    through an edge.

    ``Acme Corporation -[ACQUIRED]-> Beta Systems -[BUILT]-> Payments Engine
    -[EXPERIENCED]-> Latency Spike``. Every node names exactly one chunk, so
    traversal depth is directly observable as a chunk id, and ``ACQUIRED`` is
    asserted by ``c-filing``, which no entity row lists.
    """
    graph = SearchableGraph()
    await graph.upsert_entities(
        [
            Entity(
                name="Acme Corporation",
                type="ORGANIZATION",
                description="A logistics holding company.",
                source_chunk_ids=("c-acme",),
            ),
            Entity(
                name="Beta Systems",
                type="ORGANIZATION",
                description="A subsidiary writing billing software.",
                source_chunk_ids=("c-beta",),
            ),
            Entity(
                name="Payments Engine",
                type="PRODUCT",
                description="Settles card transactions.",
                source_chunk_ids=("c-engine",),
            ),
            Entity(
                name="Latency Spike",
                type="EVENT",
                description="A production incident during checkout.",
                source_chunk_ids=("c-spike",),
            ),
        ]
    )
    await graph.upsert_relations(
        [
            Relation(
                source="Acme Corporation",
                target="Beta Systems",
                type="ACQUIRED",
                description="Share purchase completed in 2019.",
                weight=3.0,
                source_chunk_ids=("c-filing",),
            ),
            Relation(
                source="Beta Systems",
                target="Payments Engine",
                type="BUILT",
                weight=2.0,
                source_chunk_ids=("c-beta",),
            ),
            Relation(
                source="Payments Engine",
                target="Latency Spike",
                type="EXPERIENCED",
                weight=1.0,
                source_chunk_ids=("c-engine",),
            ),
        ]
    )
    chunks = ChunkBodies()
    await chunks.upsert(
        [
            _chunk("c-acme", "doc-acquisition", "Acme Corporation acquired Beta Systems in 2019."),
            _chunk("c-filing", "doc-filing", "The filing confirms the transaction closed."),
            _chunk("c-beta", "doc-engineering", "Beta Systems built the payments engine."),
            _chunk("c-engine", "doc-engineering", "The payments engine settles card charges."),
            _chunk("c-spike", "doc-incident", "Checkout latency rose sharply on Friday."),
        ]
    )
    return graph, chunks


async def build_fan(vectors: dict[str, list[float]]) -> tuple[SearchableGraph, ChunkBodies]:
    """One entity naming several chunks and no edges at all.

    With the entity and relationship signals identical across every candidate,
    the similarity term is the only thing left that can order them — which is
    what makes the cosine assertions below assertions about the cosine.
    """
    graph = SearchableGraph()
    await graph.upsert_entities(
        [
            Entity(
                name="Ledger Service",
                type="PRODUCT",
                description="Posts double entry rows.",
                source_chunk_ids=tuple(vectors),
            )
        ]
    )
    chunks = ChunkBodies()
    await chunks.upsert(
        [_chunk(cid, "doc-ledger", f"ledger note {cid}", dense=v) for cid, v in vectors.items()]
    )
    return graph, chunks


def prose(out: Sequence[Any]) -> list[str]:
    """Ids of the retrieved passages, dropping the verbalized-subgraph chunk the
    traversal synthesizes."""
    return [c.chunk.id for c in out if c.chunk.document_id != "graph:local"]


ACME_QUERY = "What did Acme acquire?"
ENGINE_QUERY = "Which team owns the payments engine?"


class RecordingEmbedder:
    """Embeds to a fixed vector, recording that it was asked.

    ``dimension`` and ``name`` are part of the protocol the retriever's
    collaborators expect; only ``embed_query`` is exercised here.
    """

    name = "recording"
    dimension = 4

    def __init__(self, vector: list[float]) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> Any:
        self.queries.append(text)
        return self.vector

    async def embed_documents(self, texts: Sequence[str]) -> Any:  # pragma: no cover
        return np.asarray([self.vector for _ in texts], dtype=np.float32)


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
async def test_local_search_enters_the_graph_at_the_entity_the_question_names() -> None:
    """Local search's entry point is the question's entity, not the corpus's most
    connected node. Asking about the payments engine must rank the engine's own
    chunk first even though the acquisition edge is three times better attested —
    a traversal that ranked by edge weight would answer a different question.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=2))

    out = await retriever.retrieve(Query(text=ENGINE_QUERY))

    assert prose(out) == ["c-engine", "c-beta", "c-filing", "c-spike", "c-acme"]
    assert out[0].chunk.document_id == "graph:local", "the traversal leads its own results"


@pytest.mark.parametrize(
    ("hops", "expected"),
    [
        (1, ["c-filing", "c-acme", "c-beta"]),
        (2, ["c-filing", "c-acme", "c-beta", "c-engine"]),
        (3, ["c-filing", "c-acme", "c-beta", "c-engine", "c-spike"]),
    ],
)
async def test_local_hop_bound_is_the_traversal_depth(hops: int, expected: list[str]) -> None:
    """The hop setting has to bound the walk, not merely be passed to the store.

    Each node of the chain is one hop further from the seed, so an off-by-one in
    hop counting shows up as exactly one extra chunk id. Two hops from a hub node
    already reaches most of a real corpus, so an unbounded traversal ranks by
    connectivity and the failure is invisible without an assertion on depth.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=hops))

    out = await retriever.retrieve(Query(text=ACME_QUERY))

    assert prose(out) == expected


async def test_local_decays_the_match_weight_once_per_hop() -> None:
    """A neighbour's entity weight must fall geometrically with distance.

    Without decay every node reachable from the seed carries the seed's own
    confidence, so a three-hop stranger ties with a direct hit and the ranking
    collapses into a connectivity ranking. 0.6 per hop is the documented rate; the
    exact values are the only way to tell decay from mere ordering.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=3))

    out = await retriever.retrieve(Query(text=ACME_QUERY))
    entity_weight = {c.chunk.id: c.component_scores["graph_entity"] for c in out[1:]}

    assert entity_weight["c-acme"] == pytest.approx(1.0)
    assert entity_weight["c-beta"] == pytest.approx(0.6)
    assert entity_weight["c-engine"] == pytest.approx(0.36)
    assert entity_weight["c-spike"] == pytest.approx(0.216)


async def test_local_blend_matches_the_documented_weights() -> None:
    """The blend is 0.40 entity + 0.25 relationship over max-scaled signals.

    Hand-computed because a blend that "looks right" but drops or double-counts a
    term reorders results silently. One hop from Acme: entity weights 1.0/0.6/0.6
    and relationship signals 0.0/3.0/0.0, each max-scaled, divided by the 0.65 of
    weight actually in play.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))

    out = await retriever.retrieve(Query(text=ACME_QUERY))
    scores = {c.chunk.id: c.score for c in out[1:]}

    assert scores["c-filing"] == pytest.approx((0.40 * 0.6 + 0.25 * 1.0) / 0.65)
    assert scores["c-acme"] == pytest.approx((0.40 * 1.0) / 0.65)
    assert scores["c-beta"] == pytest.approx((0.40 * 0.6) / 0.65)
    assert [c.rank for c in out] == [0, 1, 2, 3], "ranks must be contiguous after the subgraph"


async def test_local_treats_the_chunk_that_asserted_an_edge_as_evidence() -> None:
    """A chunk that asserted a traversed edge is about the endpoints even when no
    entity row lists it, which is the normal state of affairs: entity rows are
    written by extraction, edges by a later pass, and only the edge remembers the
    sentence it came from. ``c-filing`` is reachable *only* through the ACQUIRED
    edge, and dropping it would lose the only text supporting the join.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))

    out = await retriever.retrieve(Query(text=ACME_QUERY))
    filing = next(c for c in out if c.chunk.id == "c-filing")

    assert prose(out)[0] == "c-filing", "edge evidence outranks a bare entity mention"
    assert filing.component_scores["graph_entity"] == pytest.approx(0.6), "one hop of decay"
    assert filing.component_scores["graph_relation"] == pytest.approx(3.0), "the edge weight"


async def test_local_redistributes_the_similarity_weight_when_vectors_are_missing() -> None:
    """A chunk store that cannot return vectors makes similarity *unavailable*,
    not zero. Scoring the missing term as zero would penalize every candidate by
    0.35 identically — harmless to the order but a lie about the score, which is
    what a relative-score cutoff downstream then acts on.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))
    query = Query(text=ACME_QUERY, dense=np.ones(4, dtype=np.float32))

    out = await retriever.retrieve(query)
    filing = next(c for c in out if c.chunk.id == "c-filing")

    assert filing.explain["similarity_available"] is False
    assert "dense" not in filing.component_scores
    assert filing.score == pytest.approx((0.40 * 0.6 + 0.25 * 1.0) / 0.65), "0.65 of weight, not 1"


async def test_local_similarity_orders_candidates_the_graph_cannot_separate() -> None:
    """A hub entity contributes many chunks with identical graph signals, and the
    traversal can reach a chunk that mentions the right entity while discussing
    something else. Similarity is the term that separates them, so it has to be a
    real cosine against the query vector, in the right order.
    """
    graph, chunks = await build_fan(
        {
            "c-half": [0.5, 0.8660254, 0.0, 0.0],
            "c-exact": [1.0, 0.0, 0.0, 0.0],
            "c-near": [0.8, 0.6, 0.0, 0.0],
        }
    )
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings())
    query = Query(text="ledger service latency", dense=np.asarray([1, 0, 0, 0], dtype=np.float32))

    out = await retriever.retrieve(query)

    assert prose(out) == ["c-exact", "c-near", "c-half"]
    assert [c.component_scores["dense"] for c in out[1:]] == [
        pytest.approx(1.0),
        pytest.approx(0.8),
        pytest.approx(0.5),
    ]
    assert out[1].score == pytest.approx(0.40 + 0.35 * 1.0), "graph 0.40 + full similarity"


async def test_local_similarity_cannot_veto_graph_evidence() -> None:
    """A negative cosine means "unrelated", and letting it stay negative would let
    the similarity term *subtract* from a chunk the graph strongly supports — a
    veto the 0.35 weight was never meant to grant it. Clipped at zero, the
    anti-aligned chunk keeps exactly its graph score.
    """
    graph, chunks = await build_fan(
        {"c-aligned": [1.0, 0.0, 0.0, 0.0], "c-opposed": [-1.0, 0.0, 0.0, 0.0]}
    )
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings())
    query = Query(text="ledger service latency", dense=np.asarray([1, 0, 0, 0], dtype=np.float32))

    out = await retriever.retrieve(query)
    scores = {c.chunk.id: c.score for c in out[1:]}
    dense = {c.chunk.id: c.component_scores["dense"] for c in out[1:]}

    assert dense["c-opposed"] == pytest.approx(0.0), "a negative cosine is floored, not kept"
    assert scores["c-opposed"] == pytest.approx(0.40), "the graph term survives intact"
    assert scores["c-aligned"] == pytest.approx(0.75)


async def test_local_annotates_a_chunk_with_only_the_edges_it_asserted() -> None:
    """The generator sees prose, and the edge that made the prose relevant is not
    in the prose — extraction and normalization happened since. So each chunk is
    prefixed with the graph context tied to *that* chunk, and a chunk that
    asserted no edge is left exactly as the store returned it: an unrelated header
    would be fabricated provenance.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))

    out = await retriever.retrieve(Query(text=ACME_QUERY))
    filing = next(c for c in out if c.chunk.id == "c-filing").chunk
    acme = next(c for c in out if c.chunk.id == "c-acme").chunk

    assert filing.content.startswith("Graph context for this passage:")
    assert "Acme Corporation -[ACQUIRED]-> Beta Systems (weight 3.0)" in filing.content
    assert filing.content.endswith("The filing confirms the transaction closed.")
    assert filing.metadata["graph_context"] in filing.content
    assert filing.token_count is None, "a rewritten body invalidates the cached token count"
    assert acme.content == "Acme Corporation acquired Beta Systems in 2019."
    assert acme.token_count == 11, "an untouched body keeps its budgeted token count"
    assert "graph_context" not in acme.metadata


async def test_local_leads_with_the_verbalized_neighbourhood() -> None:
    """The traversal itself is evidence no single chunk carries: a two-hop join
    runs through edges asserted by different documents. It leads the list because
    it is the connective tissue for everything under it, and it carries the best
    prose chunk's score so a relative-score cutoff downstream does not cut the
    passages away from their explanation.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))

    out = await retriever.retrieve(Query(text=ACME_QUERY))
    summary = out[0].chunk

    assert summary.document_id == "graph:local"
    assert summary.modality is Modality.SUMMARY
    assert summary.metadata == {
        "source": "graph_local",
        "entities": 2,
        "relations": 1,
        "hops": 1,
    }
    assert summary.content.startswith(f"Knowledge graph neighbourhood for: {ACME_QUERY}")
    assert "Acme Corporation [ORGANIZATION]: A logistics holding company." in summary.content
    assert "Acme Corporation -[ACQUIRED]-> Beta Systems (weight 3.0)" in summary.content
    assert out[0].score == pytest.approx(out[1].score), "the summary matches the best passage"
    assert out[0].component_scores == {"graph_subgraph": pytest.approx(out[1].score)}


async def test_local_results_still_cite_their_source_documents() -> None:
    """Graph retrieval that loses the chunk ids produces an answer that cannot be
    cited, which in a RAG pipeline is an ungrounded answer. Every passage returned
    keeps the store's own id and document, and the synthesized subgraph chunk is
    labelled as a summary so a citation validator can tell the two apart.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))

    out = await retriever.retrieve(Query(text=ACME_QUERY))

    assert [(c.chunk.id, c.chunk.document_id) for c in out[1:]] == [
        ("c-filing", "doc-filing"),
        ("c-acme", "doc-acquisition"),
        ("c-beta", "doc-engineering"),
    ]
    assert {c.source for c in out} == {RetrievalSource.GRAPH_LOCAL}
    assert out[0].explain == {"retriever": "graph_local", "verbalized_subgraph": True}
    assert chunks.requested == [["c-acme", "c-filing", "c-beta"]], "one batched read, deduped"


async def test_local_loads_three_candidate_chunks_per_chunk_returned() -> None:
    """A hub entity is mentioned by thousands of chunks, and loading all of them to
    return one is the dominant cost of local search. The similarity term can still
    reorder the ranking, so the pool cannot be cut to ``top_k`` before the bodies
    are read — three times the output is the documented headroom, and it has to
    bind or the cost bound is decorative.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=3))

    out = await retriever.retrieve(Query(text=ACME_QUERY), top_k=1)

    assert chunks.requested == [["c-acme", "c-filing", "c-beta"]], "3 candidates for 1 result"
    assert prose(out) == ["c-filing"]


async def test_local_returns_nothing_when_the_graph_knows_no_entity() -> None:
    """The right answer for a purely descriptive question, and for a corpus that
    was never graph-indexed, is nothing — not an exception. This retriever runs
    inside a fan-out whose vector leg was going to answer anyway, and raising here
    would take that answer down with it. DRIFT exists to cover this case.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings())

    assert await retriever.retrieve(Query(text="How long do refunds take?")) == []
    assert graph.expansions == [], "an unanswerable question costs one lookup, not a traversal"
    assert chunks.requested == []
    assert (
        await GraphLocalRetriever(SearchableGraph(), chunks, settings=graph_settings()).retrieve(
            Query(text=ACME_QUERY)
        )
        == []
    ), "an empty graph is an empty result"


async def test_local_survives_a_failed_traversal() -> None:
    """The seed match already succeeded, so a dead traversal costs the edges, not
    the query: a lone matched entity with a good description still answers "who is
    X". The loss is recorded so the trace shows a degraded expansion rather than a
    graph that appears to have no relationships in it.
    """
    graph, chunks = await build_chain()
    graph.fail.add("neighbors")
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=2))
    seed = graph.entities["acme corporation"]

    out, detail = await retriever.expand(Query(text=ACME_QUERY), [(seed, 4.0)])

    assert prose(out) == ["c-acme"]
    # ``no_query_embedder`` because this retriever was built without one, and the
    # similarity term is therefore permanently unavailable rather than
    # transiently failed. Reported alongside the traversal loss because both
    # reduce the ranking to less than the three terms it documents.
    assert detail["degraded"] == ["neighbors", "no_query_embedder"]
    assert detail["relations"] == 0
    assert out[1].score == pytest.approx(0.40 / 0.65), "the entity term carries the whole blend"


async def test_local_survives_a_failed_chunk_read() -> None:
    """No bodies is still an answer: the verbalized subgraph names the entities and
    the edges, which is exactly the evidence the vector index could never produce.
    Failing instead would trade a partial answer for no answer.
    """
    graph, chunks = await build_chain()
    chunks.fail = True
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings(local_search_hops=1))
    seed = graph.entities["acme corporation"]

    out, detail = await retriever.expand(Query(text=ACME_QUERY), [(seed, 4.0)])

    assert prose(out) == [], "no bodies means no passages"
    assert detail["degraded"] == ["chunk_store"]
    assert detail["candidate_chunks"] == 3
    assert detail["returned"] == 1
    assert "Acme Corporation -[ACQUIRED]-> Beta Systems (weight 3.0)" in out[0].chunk.content


async def test_local_without_a_chunk_store_returns_the_subgraph_alone() -> None:
    """A deployment can wire the graph without a chunk store, and it must degrade
    to graph-only evidence rather than to a crash. The distinction from a *failed*
    read is recorded, because one is a configuration and the other is an outage.
    """
    graph, _ = await build_chain()
    retriever = GraphLocalRetriever(graph, None, settings=graph_settings(local_search_hops=1))
    seed = graph.entities["acme corporation"]

    out, detail = await retriever.expand(Query(text=ACME_QUERY), [(seed, 4.0)])

    assert [c.chunk.document_id for c in out] == ["graph:local"]
    assert detail["degraded"] == ["no_chunk_store"]


async def test_local_ignores_a_seed_that_has_no_canonical_name() -> None:
    """The graph is keyed on the canonical names entity resolution produced, so a
    nameless match cannot be looked up. Seeding the traversal from the empty string
    would expand from a node that does not exist and dress the result up as
    evidence; returning nothing is the only honest answer.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings())

    out, detail = await retriever.expand(Query(text=ACME_QUERY), [(Entity(name=""), 9.0)])

    assert out == []
    assert detail["seed_entities"] == []
    assert detail["returned"] == 0
    assert chunks.requested == [], "nothing to expand from is nothing to read"


async def test_local_hop_decay_of_zero_keeps_only_the_seed_and_its_edges() -> None:
    """``hop_decay`` is a real dial, not decoration: at zero, a neighbour inherits
    no match weight, so it contributes no chunks of its own and only the edges
    incident to the seed still carry evidence. Verifying the extreme is how we know
    the default 0.6 is being applied per hop rather than once.
    """
    graph, chunks = await build_chain()
    retriever = GraphLocalRetriever(
        graph, chunks, hop_decay=0.0, settings=graph_settings(local_search_hops=2)
    )

    out = await retriever.retrieve(Query(text=ACME_QUERY))

    assert prose(out) == ["c-acme", "c-filing"], "the one-hop and two-hop entities drop out"
    assert out[1].score == pytest.approx(0.40 / 0.65)
    assert out[2].score == pytest.approx(0.25 / 0.65), "edge evidence, with no entity weight left"


# ---------------------------------------------------------------------------
# Global search
# ---------------------------------------------------------------------------
def _community(cid: int, title: str, summary: str, rank: float, level: int = 1) -> Community:
    return Community(
        id=cid,
        level=level,
        title=title,
        summary=summary,
        rank=rank,
        entity_names=("Acme Corporation", "Beta Systems"),
    )


async def build_global(*communities: Community) -> SearchableGraph:
    graph = SearchableGraph()
    graph.communities_list = list(communities)
    return graph


async def test_global_combines_every_community_it_mapped() -> None:
    """Map-reduce means *reduce*: the partial answers join, they do not overwrite.
    A map loop that keeps only the last result still returns a fluent answer about
    one corner of the corpus, which is precisely the failure global search exists
    to prevent. Ranking is by map score, not by the store's community order.
    """
    graph = await build_global(
        _community(11, "Payments platform", "Acme owns Beta, which built the engine.", rank=9.0),
        _community(12, "Incident history", "Checkout latency spiked in March.", rank=7.0),
        _community(13, "Vendor contracts", "Catering renewals for the London office.", rank=5.0),
    )
    llm = ScriptedLLM(
        script=[
            {"answer": "Acme is the parent company.", "score": 4.0},
            {"answer": "The March incident hit checkout.", "score": 9.0},
            {"answer": "Catering is renewed annually.", "score": 6.0},
        ]
    )
    retriever = GraphGlobalRetriever(llm, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="What are the main themes?"))

    assert [c.chunk.content for c in out] == [
        "The March incident hit checkout.",
        "Catering is renewed annually.",
        "Acme is the parent company.",
    ]
    assert [c.score for c in out] == [pytest.approx(0.9), pytest.approx(0.6), pytest.approx(0.4)]
    assert [c.chunk.metadata["community_id"] for c in out] == [12, 13, 11]
    assert [c.rank for c in out] == [0, 1, 2]


async def test_global_drops_communities_that_contribute_nothing() -> None:
    """A zero score is the map prompt's explicit "this community has nothing to
    add". Keeping it would feed the reduce step confident irrelevance, which is
    the exact failure the prompt is written to avoid — so the drop is the
    contract, not an optimization.
    """
    graph = await build_global(
        _community(21, "Relevant", "Beta built the engine.", rank=9.0),
        _community(22, "Irrelevant", "Catering renewals.", rank=8.0),
        _community(23, "Empty", "Unrelated meeting notes.", rank=7.0),
    )
    llm = ScriptedLLM(
        script=[
            {"answer": "Beta built the payments engine.", "score": 7.0},
            {"answer": "Nothing in this report is relevant.", "score": 0.0},
            {"answer": "   ", "score": 8.0},
        ]
    )
    retriever = GraphGlobalRetriever(llm, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="Who built the engine?"))

    assert [c.chunk.content for c in out] == ["Beta built the payments engine."]
    assert llm.call_count == 3, "all three were mapped; two were dropped after answering"


async def test_global_pays_for_at_most_the_top_ranked_communities() -> None:
    """One LLM call per community per query makes the cap the cost dial for this
    retriever, so it has to bind — and it has to keep the *highest-ranked*
    communities, since rank is the store's importance ordering and a cap that
    sliced arbitrarily would silently answer from the corpus's footnotes.
    """
    graph = await build_global(
        _community(31, "Third", "Third report.", rank=3.0),
        _community(32, "First", "First report.", rank=9.0),
        _community(33, "Fourth", "Fourth report.", rank=2.0),
        _community(34, "Second", "Second report.", rank=8.0),
        _community(35, "Fifth", "Fifth report.", rank=1.0),
    )
    llm = StubLLM(responses={"MapAnswer": {"answer": "a partial answer", "score": 5.0}})
    retriever = GraphGlobalRetriever(
        llm, graph, settings=graph_settings(global_search_top_communities=2)
    )

    out = await retriever.retrieve(Query(text="What are the main themes?"))

    assert llm.call_count == 2, f"paid for {llm.call_count} communities, budget was 2"
    assert [c.chunk.metadata["community_id"] for c in out] == [32, 34]


async def test_global_returns_at_most_top_k_partials() -> None:
    """The reduce step is one prompt, so the partials have to fit in it. Cutting
    by score keeps the communities that actually answered."""
    graph = await build_global(
        _community(41, "One", "First report.", rank=9.0),
        _community(42, "Two", "Second report.", rank=8.0),
        _community(43, "Three", "Third report.", rank=7.0),
    )
    llm = ScriptedLLM(
        script=[
            {"answer": "weakest", "score": 2.0},
            {"answer": "strongest", "score": 9.0},
            {"answer": "middling", "score": 5.0},
        ]
    )
    retriever = GraphGlobalRetriever(llm, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="themes?"), top_k=2)

    assert [c.chunk.content for c in out] == ["strongest", "middling"]
    assert [c.rank for c in out] == [0, 1]


async def test_global_accounts_for_the_whole_map_fan_out() -> None:
    """N calls per query is the reason global search needs a visible bill. Usage
    that recorded only the last call would under-report the cost by a factor of N,
    which is how a cost ceiling stops being a ceiling.
    """
    graph = await build_global(
        _community(51, "One", "First report.", rank=9.0),
        _community(52, "Two", "Second report.", rank=8.0),
        _community(53, "Three", "Third report.", rank=7.0),
    )
    llm = StubLLM(
        responses={"MapAnswer": {"answer": "a partial answer", "score": 5.0}}, cost_per_call=0.002
    )
    retriever = GraphGlobalRetriever(llm, graph, settings=graph_settings())

    await retriever.retrieve(Query(text="What are the main themes?"))

    assert retriever.usage.calls == 3
    assert retriever.usage.cost_usd == pytest.approx(0.006)


async def test_global_survives_a_failed_map_call() -> None:
    """Global search is an aggregation, so it is still meaningful over the subset
    that returned. With eight or more independent calls per query, letting one
    provider hiccup discard seven good partial answers would make the mode
    unusable in production.
    """

    class Failing(StubLLM):
        async def structured(self, prompt: str, schema: type, **kwargs: Any) -> Any:  # type: ignore[override]
            if "Second report." in prompt:
                raise RuntimeError("provider hiccup")
            return await super().structured(prompt, schema, **kwargs)

    graph = await build_global(
        _community(61, "One", "First report.", rank=9.0),
        _community(62, "Two", "Second report.", rank=8.0),
        _community(63, "Three", "Third report.", rank=7.0),
    )
    llm = Failing(responses={"MapAnswer": {"answer": "a partial answer", "score": 5.0}})
    retriever = GraphGlobalRetriever(llm, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="What are the main themes?"))

    assert [c.chunk.metadata["community_id"] for c in out] == [61, 63]


async def test_global_partials_cite_the_community_they_came_from() -> None:
    """The reduce prompt is told to attribute each point to a community, and a
    partial that lost its provenance cannot be attributed. The 0-10 map score is
    also rescaled to [0, 1] here so fusion does not have to special-case this
    retriever against every other one in the library.
    """
    graph = await build_global(
        _community(7, "Payments platform", "Beta built it.", rank=3.5, level=2)
    )
    llm = StubLLM(responses={"MapAnswer": {"answer": "Beta built the engine.", "score": 10.0}})
    retriever = GraphGlobalRetriever(llm, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="Who built the engine?"))
    partial = out[0]

    assert partial.chunk.document_id == "community:7"
    assert partial.chunk.level == 2
    assert partial.chunk.modality is Modality.SUMMARY
    assert partial.chunk.metadata == {
        "source": "graph_global",
        "community_id": 7,
        "community_title": "Payments platform",
        "community_rank": 3.5,
        "community_level": 2,
        "entity_count": 2,
    }
    assert partial.score == pytest.approx(1.0), "a 10/10 map score is 1.0 on the shared scale"
    assert partial.component_scores == {"graph_global_map": pytest.approx(1.0)}
    assert partial.explain["map_score_0_10"] == pytest.approx(10.0)
    assert partial.explain["reduce_with"] == "global_reduce"
    assert partial.source is RetrievalSource.GRAPH_GLOBAL


async def test_global_buys_nothing_without_community_reports() -> None:
    """A corpus indexed without community summarization has nothing for global
    search to read. It must say so for free: one wasted LLM call per unsummarized
    community, on every query, is a bill for a guaranteed empty answer.
    """
    graph = await build_global(
        Community(id=71, level=1, title="No summary", summary="", rank=9.0),
    )
    llm = StubLLM(responses={"MapAnswer": {"answer": "a partial answer", "score": 5.0}})

    out = await GraphGlobalRetriever(llm, graph, settings=graph_settings()).retrieve(
        Query(text="What are the main themes?")
    )

    assert out == []
    assert llm.call_count == 0
    empty = GraphGlobalRetriever(llm, await build_global(), settings=graph_settings())
    assert await empty.retrieve(Query(text="What are the main themes?")) == []
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# DRIFT search
# ---------------------------------------------------------------------------
async def build_drift() -> tuple[SearchableGraph, ChunkBodies]:
    """A graph whose entities are all provable from the seed chunks, over a store
    that also holds a passage the graph knows nothing about."""
    graph = SearchableGraph()
    await graph.upsert_entities(
        [
            Entity(name="Acme Corporation", type="ORGANIZATION", source_chunk_ids=("c-acme",)),
            Entity(name="Beta Systems", type="ORGANIZATION", source_chunk_ids=("c-beta",)),
            Entity(name="Payments Engine", type="PRODUCT", source_chunk_ids=("c-beta",)),
        ]
    )
    await graph.upsert_relations(
        [
            Relation(
                source="Acme Corporation",
                target="Beta Systems",
                type="ACQUIRED",
                weight=3.0,
                source_chunk_ids=("c-acme",),
            ),
            Relation(
                source="Beta Systems",
                target="Payments Engine",
                type="BUILT",
                weight=2.0,
                source_chunk_ids=("c-beta",),
            ),
        ]
    )
    store = ChunkBodies()
    await store.upsert(
        [
            _chunk("c-beta", "doc-engineering", "Beta Systems built the payments engine"),
            _chunk(
                "c-acme", "doc-acquisition", "Acme Corporation acquired the Beta Systems engine"
            ),
            _chunk("c-noise", "doc-ops", "payments team catering schedule"),
        ]
    )
    return graph, store


async def test_drift_prices_agreement_between_its_two_halves() -> None:
    """A chunk both halves found is better evidence than a chunk either found
    alone, and the weights say so arithmetically: 0.45 for the vector seed, 0.55
    for the graph expansion, up to 1.0 together. ``c-acme`` and ``c-noise`` are
    equally weak vector hits; only one of them has graph support, and that has to
    be the difference between them.
    """
    graph, store = await build_drift()
    retriever = GraphDriftRetriever(store, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="payments engine"))
    summary = next(c for c in out if c.chunk.document_id == "graph:local")
    scores = {c.chunk.id: c.score for c in out}

    assert [c.chunk.id for c in out] == ["c-beta", "c-acme", summary.chunk.id, "c-noise"]
    assert scores["c-acme"] == pytest.approx(0.45 * 0.5 + 0.55 * 1.0)
    assert scores["c-noise"] == pytest.approx(0.45 * 0.5), "a seed-only hit is capped at 0.45"
    assert scores[summary.chunk.id] == pytest.approx(0.55), "a graph-only hit is capped at 0.55"
    assert out[0].source is RetrievalSource.FUSED
    assert out[0].component_scores["drift_seed"] == pytest.approx(1.0)
    assert out[0].component_scores["drift_graph"] == pytest.approx((0.4 + 0.25 * 2 / 3) / 0.65)
    assert out[3].source is RetrievalSource.DENSE, "the seed-only hit keeps its own provenance"

    # The merge must keep the *graph-annotated* body, not the seed's copy of the
    # same chunk. Both objects share an id and a score, so asserting only those
    # cannot tell them apart — and the annotation is the verbalized relationship
    # the generator needs to explain why the passage is relevant.
    both_halves = next(c for c in out if c.chunk.id == "c-beta")
    assert both_halves.chunk.content.startswith("Graph context for this passage:"), (
        f"merged body lost its graph annotation: {both_halves.chunk.content[:80]!r}"
    )
    assert out[3].chunk.content == "payments team catering schedule", (
        "a seed-only hit must keep its plain body"
    )


async def test_drift_enters_the_graph_from_the_passage_not_only_the_question() -> None:
    """This is the whole point of DRIFT: a descriptive question names no entity, so
    local search cannot start, but the passage the vector search found does name
    one. Here the question matches nothing in the graph and nothing in the corpus
    lexically, and the graph still gets entered — the seed *text* is what probes
    the entity index.
    """
    graph, store = await build_drift()
    retriever = GraphDriftRetriever(store, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="quarterly figures"))
    scores = {c.chunk.id: c.score for c in out if c.chunk.document_id != "graph:local"}

    assert scores == {
        "c-acme": pytest.approx(0.55),
        "c-beta": pytest.approx(0.55 * (0.4 + 0.25 * 2 / 3) / 0.65),
        "c-noise": pytest.approx(0.0),
    }
    assert out[0].component_scores["drift_seed"] == pytest.approx(0.0), (
        "the vector leg found nothing"
    )
    assert out[0].component_scores["drift_graph"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("filler", "reached", "expected"),
    [(400, False, 0.45), (10, True, 1.0)],
)
async def test_drift_bounds_how_much_seed_text_probes_the_index(
    filler: int, reached: bool, expected: float
) -> None:
    """The probe is budgeted at 2000 characters because the index scores by term
    overlap: a longer probe raises recall and flattens precision until the seeds
    describe the corpus rather than the question. The bound is only real if names
    beyond it go genuinely unmatched — both the tail of an over-long first seed and
    everything in the seeds that come after it, whose budget it has spent.
    """
    graph = SearchableGraph()
    await graph.upsert_entities(
        [
            Entity(name="Zeta Holdings", type="ORGANIZATION", source_chunk_ids=("c-long",)),
            Entity(name="Omega Trust", type="ORGANIZATION", source_chunk_ids=("c-tail",)),
        ]
    )
    store = ChunkBodies()
    await store.upsert(
        [
            _chunk("c-long", "doc-long", "alpha " * filler + "Zeta Holdings"),
            _chunk("c-tail", "doc-tail", "alpha Omega Trust"),
        ]
    )
    retriever = GraphDriftRetriever(store, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="alpha"))
    scores = {c.chunk.id: c.score for c in out if c.chunk.document_id != "graph:local"}

    assert any(c.chunk.document_id == "graph:local" for c in out) is reached
    assert scores == {"c-long": pytest.approx(expected), "c-tail": pytest.approx(expected)}


async def build_split() -> tuple[SearchableGraph, ChunkBodies, ChunkBodies]:
    """Seeds and bodies in separate stores, so a chunk can be readable without
    being retrievable — which is what makes "the seeds prove it" testable."""
    graph = SearchableGraph()
    seeds = ChunkBodies()
    await seeds.upsert(
        [_chunk("c-beta", "doc-engineering", "Beta Systems built the payments engine")]
    )
    bodies = ChunkBodies()
    await bodies.upsert(
        [
            _chunk("c-beta", "doc-engineering", "Beta Systems built the payments engine"),
            _chunk("c-archive", "doc-archive", "Ghost Registry rows from an older archive"),
        ]
    )
    return graph, seeds, bodies


async def test_drift_prefers_the_entities_its_seed_chunks_prove() -> None:
    """The seed chunk ids are the confirmation. An entity the index merely matched
    on text is a plausible mention; one whose ``source_chunk_ids`` intersect the
    seeds is provably what the retrieved passage is about, and when any entity is
    proven the plausible ones must not dilute the traversal.
    """
    graph, seeds, bodies = await build_split()
    await graph.upsert_entities(
        [
            Entity(name="Beta Systems", type="ORGANIZATION", source_chunk_ids=("c-beta",)),
            Entity(name="Ghost Registry", type="PRODUCT", source_chunk_ids=("c-archive",)),
        ]
    )
    retriever = GraphDriftRetriever(seeds, graph, bodies, settings=graph_settings())

    out = await retriever.retrieve(Query(text="beta systems ghost registry"))

    assert [c.chunk.id for c in out if c.chunk.document_id != "graph:local"] == ["c-beta"]
    assert out[0].score == pytest.approx(1.0), "both halves found it"


async def test_drift_still_expands_when_nothing_can_be_confirmed() -> None:
    """A graph built from an earlier chunk generation confirms nothing, because no
    entity's ``source_chunk_ids`` match the current ids. Dropping the unproven
    matches would make DRIFT return a bare vector result for the entire corpus, so
    they seed at a discount instead.
    """
    graph, seeds, bodies = await build_split()
    await graph.upsert_entities(
        [Entity(name="Ghost Registry", type="PRODUCT", source_chunk_ids=("c-archive",))]
    )
    retriever = GraphDriftRetriever(seeds, graph, bodies, settings=graph_settings())

    out = await retriever.retrieve(Query(text="ghost registry payments"))
    scores = {c.chunk.id: c.score for c in out}

    assert scores["c-archive"] == pytest.approx(0.55), "unproven, but still expanded"
    assert scores["c-beta"] == pytest.approx(0.45)
    assert len(out) == 3, "the seed, the expanded chunk and the verbalized subgraph"


async def test_drift_degrades_to_the_vector_result_when_the_graph_is_down() -> None:
    """The seeds are already a usable answer; they just have no relationship
    evidence. Failing the query instead would make a graph outage look like a
    search outage, when the vector index is up and holding the passage.
    """
    graph, store = await build_drift()
    graph.fail.add("fulltext")
    retriever = GraphDriftRetriever(store, graph, settings=graph_settings())

    out = await retriever.retrieve(Query(text="payments engine"))

    assert [c.chunk.id for c in out] == ["c-beta", "c-acme", "c-noise"]
    assert [c.score for c in out] == [
        pytest.approx(0.45),
        pytest.approx(0.225),
        pytest.approx(0.225),
    ]
    assert [c.rank for c in out] == [0, 1, 2]


async def test_drift_does_not_probe_the_graph_with_nothing() -> None:
    """No seeds means no entry point, and an entity probe built from an empty seed
    set is a round trip that can only return noise."""
    graph, _ = await build_drift()
    retriever = GraphDriftRetriever(ChunkBodies(), graph, settings=graph_settings())

    assert await retriever.retrieve(Query(text="payments engine")) == []
    assert graph.probes == []


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------
def test_max_scale_keeps_the_ratios_between_signals() -> None:
    """Min-max normalization pins the weakest candidate at exactly 0, asserting it
    is irrelevant — false when every candidate is strong, and it destroys the very
    ratios the blend is combining. Dividing by the maximum keeps "half as good as
    the best" meaning half as good.
    """
    scaled = max_scale(np.asarray([4.0, 2.0, 1.0], dtype=np.float32))

    assert scaled.tolist() == [pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.25)]


def test_max_scale_survives_a_signal_nobody_produced() -> None:
    """An all-zero term is the normal case for a traversal with no edges, and
    dividing it by its own maximum would put NaN into every blended score."""
    scaled = max_scale(np.zeros(3, dtype=np.float32))

    assert scaled.tolist() == [0.0, 0.0, 0.0]
    assert max_scale(np.asarray([], dtype=np.float32)).tolist() == []


def test_verbalized_edge_carries_its_weight_and_a_budgeted_description() -> None:
    """Weight is the model's only cue that one assertion is better attested than
    another — the difference between "sources agree" and "one source claims". The
    description is clipped because it accumulates across every chunk the edge was
    seen in and would otherwise crowd the prose out of the context window.
    """
    rendered = verbalize_relations(
        [
            Relation(source="Acme", target="Beta", type="ACQUIRED", weight=12.0),
            Relation(source="Beta", target="Engine", type="BUILT", description="x " * 300),
        ]
    )
    first, second = rendered.split("\n")

    assert first == "Acme -[ACQUIRED]-> Beta (weight 12.0)"
    assert second.startswith("Beta -[BUILT]-> Engine (weight 1.0): x x")
    assert len(second.split(": ", 1)[1]) == 243, "240 characters plus the elision"


async def test_load_chunks_asks_for_each_id_once_in_first_seen_order() -> None:
    """Entities and edges cite the same chunk constantly, so the candidate list
    arrives with duplicates. Reading a hub chunk five times is five round trips
    for one body, and order is preserved so the ranking's tie-breaks stay stable.
    """
    store = ChunkBodies()
    await store.upsert([_chunk(cid, "doc", cid) for cid in ("a", "b", "c")])

    out = await load_chunks(store, ["b", "a", "b", "c", "a"])

    assert store.requested == [["b", "a", "c"]]
    assert [c.id for c in out] == ["b", "a", "c"]
    assert await load_chunks(store, []) == [], "no ids is not a store call"
    assert store.requested == [["b", "a", "c"]]


async def test_load_chunks_reads_a_store_that_only_speaks_get_chunks() -> None:
    """Qdrant spells this ``get`` and Postgres spells it ``get_chunks``. Accepting
    both is what lets a deployment serve graph chunk text from the store it already
    runs, instead of being forced to keep a vector store on the graph path — and
    the Postgres shape takes no vector flag, so passing one must not reach it.
    """

    class PostgresShaped:
        def __init__(self) -> None:
            self.requested: list[list[str]] = []

        async def get_chunks(self, ids: Sequence[str]) -> list[Chunk]:
            self.requested.append(list(ids))
            return [_chunk(cid, "doc", f"body of {cid}") for cid in ids]

    store = PostgresShaped()

    out = await load_chunks(store, ["b", "a", "b"], with_vectors=True)

    assert store.requested == [["b", "a"]]
    assert [c.id for c in out] == ["b", "a"]
    assert [c.dense for c in out] == [None, None], "this store has no vectors to hand back"


async def test_load_chunks_rejects_a_store_it_cannot_read() -> None:
    """A misconfigured chunk store has to fail loudly at the read, not return an
    empty list that reads downstream as "the graph found nothing"."""

    class Opaque:
        pass

    with pytest.raises(RetrievalError, match="neither get"):
        await load_chunks(Opaque(), ["a"])


async def test_graph_annotation_does_not_mutate_the_store_s_own_chunk() -> None:
    """Annotating a chunk must not write into the object it was handed.

    The old code prefixed `content` in place and justified it by saying the chunk
    was built by this call's own store read and therefore owned by it. That holds
    for Postgres and Qdrant, which deserialize a fresh object per read, and fails
    for any store handing back a reference it also keeps — a cache tier, or a
    third-party `VectorStore`, which the protocol invites. The annotation then
    leaks into whatever reads that chunk next.
    """
    graph, store = await build_drift()
    before = (await store.get(["c-beta"], with_vectors=True))[0].content

    retriever = GraphLocalRetriever(graph, store, settings=graph_settings())
    out = await retriever.retrieve(Query(text="payments engine"))

    annotated = next(c for c in out if c.chunk.id == "c-beta")
    assert annotated.chunk.content.startswith("Graph context for this passage:"), (
        "the returned chunk must carry the annotation"
    )
    after = (await store.get(["c-beta"], with_vectors=True))[0].content
    assert after == before, "the store's own copy must be untouched"
    assert "graph_context" not in (await store.get(["c-beta"]))[0].metadata


# ---------------------------------------------------------------------------
# The third ranking term
# ---------------------------------------------------------------------------
_FAN = {
    "c-exact": [1.0, 0.0, 0.0, 0.0],
    "c-near": [0.8, 0.6, 0.0, 0.0],
    "c-half": [0.5, 0.8660254, 0.0, 0.0],
}


async def test_local_embeds_the_question_so_the_similarity_term_fires() -> None:
    """Local search blends three signals, and the third never contributed.

    ``query.dense`` is populated in exactly one place — ``QdrantStore._prepare``
    — and the GraphRAG graph never goes through the vector store: ``classify``
    routes straight to a graph node, which calls this retriever with the query
    as validated. So the term was skipped, its weight was silently redistributed
    over entity and relation signal, and the ranking that shipped was two-thirds
    of the one this module documents.

    The existing cosine test sets ``query.dense`` by hand, which is why it passed
    throughout: it covers the arithmetic, and nothing covered whether anything
    ever supplies the input.
    """
    graph, chunks = await build_fan(_FAN)
    embedder = RecordingEmbedder([1.0, 0.0, 0.0, 0.0])
    retriever = GraphLocalRetriever(graph, chunks, embedder=embedder, settings=graph_settings())

    out = await retriever.retrieve(Query(text="ledger service latency"))

    assert embedder.queries == ["ledger service latency"]
    passages = [c for c in out if c.chunk.document_id != "graph:local"]
    assert [c.chunk.id for c in passages] == ["c-exact", "c-near", "c-half"], (
        "the ranking is not ordered by similarity, so the term did not contribute"
    )
    assert all(c.explain["similarity_available"] for c in passages)
    assert [c.component_scores["dense"] for c in passages] == [
        pytest.approx(1.0),
        pytest.approx(0.8),
        pytest.approx(0.5),
    ]


async def test_without_an_embedder_the_graph_terms_carry_the_whole_blend() -> None:
    """The state this shipped in, pinned as the contrast.

    Identical graph signals across three chunks and no query vector: the
    similarity weight is redistributed, every score collapses to the same value,
    and the term that exists to separate them cannot.
    """
    graph, chunks = await build_fan(_FAN)
    retriever = GraphLocalRetriever(graph, chunks, settings=graph_settings())

    out = await retriever.retrieve(Query(text="ledger service latency"))

    passages = [c for c in out if c.chunk.document_id != "graph:local"]
    assert not any(c.explain["similarity_available"] for c in passages)
    assert len({round(c.score, 9) for c in passages}) == 1, (
        "without the third term these chunks are indistinguishable"
    )


async def test_local_reuses_a_query_vector_someone_else_already_computed() -> None:
    """DRIFT seeds from a vector search, so the vector is already on the query by
    the time ``expand`` runs. Re-embedding would be a second model call for a
    value we are holding."""
    graph, chunks = await build_fan(_FAN)
    embedder = RecordingEmbedder([1.0, 0.0, 0.0, 0.0])
    retriever = GraphLocalRetriever(graph, chunks, embedder=embedder, settings=graph_settings())
    query = Query(text="ledger service latency")
    query.dense = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    await retriever.retrieve(query)

    assert embedder.queries == [], "the query was embedded twice"


async def test_local_still_ranks_when_the_embedder_fails() -> None:
    """A failed embedding must cost the tie-breaker, not the query: the entity and
    relation terms are an answer on their own."""

    class Broken(RecordingEmbedder):
        async def embed_query(self, text: str) -> Any:
            raise RuntimeError("provider down")

    graph, chunks = await build_fan(_FAN)
    retriever = GraphLocalRetriever(
        graph, chunks, embedder=Broken([1.0, 0.0, 0.0, 0.0]), settings=graph_settings()
    )

    out, detail = await retriever.expand(
        Query(text="ledger service latency"), [(graph.entities["ledger service"], 4.0)]
    )

    assert [c.chunk.id for c in out if c.chunk.document_id != "graph:local"], (
        "a failed embedding must not empty the result"
    )
    assert "query_embedding" in detail["degraded"]
