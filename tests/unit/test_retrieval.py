"""Fusion, reranking and compression.

These are the arithmetic-heavy parts of the pipeline, so the assertions are
numeric where the answer is derivable by hand — a fusion function that "looks
right" and quietly loses a document is the kind of bug that only shows up as
mysteriously poor retrieval quality months later.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragorc.core.models import Chunk, FusionMethod, Query, RetrievalSource, ScoredChunk, Usage
from ragorc.core.settings import Settings
from ragorc.core.telemetry import current_ledger, new_request_context
from ragorc.retrieve.bm25 import InMemoryBM25Retriever
from ragorc.retrieve.crag import CorrectiveRAG
from ragorc.retrieve.fusion import (
    DEFAULT_RRF_K,
    distribution_based_score_fusion,
    fuse,
    max_fusion,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from ragorc.retrieve.rankgpt import repair_permutation, sliding_windows
from ragorc.retrieve.rerank import maxsim
from tests.fakes.llm import StubLLM


def sc(
    cid: str, score: float, *, source: RetrievalSource = RetrievalSource.DENSE, text: str = ""
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=cid, content=text or f"content of {cid}"), score=score, source=source
    )


def ranked(*ids: str, base: float = 0.9) -> list[ScoredChunk]:
    return [sc(cid, base - i * 0.1) for i, cid in enumerate(ids)]


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------
def test_rrf_promotes_cross_retriever_consensus() -> None:
    """A document two retrievers both found must outrank one only a single
    retriever ranked first. This is the entire point of RRF."""
    dense = ranked("consensus", "dense_only")
    sparse = ranked("sparse_only", "consensus")
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].chunk.id == "consensus"


def test_rrf_matches_hand_computation() -> None:
    """score = sum over lists of 1 / (k + rank + 1), rank 0-based."""
    k = DEFAULT_RRF_K
    fused = reciprocal_rank_fusion([ranked("a", "b"), ranked("b", "a")])
    expected = 1.0 / (k + 1) + 1.0 / (k + 2)
    by_id = {f.chunk.id: f.score for f in fused}
    assert by_id["a"] == pytest.approx(expected)
    assert by_id["b"] == pytest.approx(expected)


def test_rrf_is_scale_free() -> None:
    """Multiplying one list's scores by 1000 must not change the fusion, because
    RRF reads rank, not magnitude. This is why it is the default for cross-store
    fusion where a cosine and a BM25 score are not comparable."""
    small = [sc("a", 0.9), sc("b", 0.8)]
    huge = [sc("b", 900.0), sc("a", 800.0)]
    a = [f.chunk.id for f in reciprocal_rank_fusion([small, huge])]
    scaled = [sc("a", 0.0009), sc("b", 0.0008)]
    b = [f.chunk.id for f in reciprocal_rank_fusion([scaled, huge])]
    assert a == b


def test_rrf_merges_duplicates_once() -> None:
    fused = reciprocal_rank_fusion([ranked("x", "y"), ranked("x", "z"), ranked("x")])
    ids = [f.chunk.id for f in fused]
    assert ids.count("x") == 1
    assert len(ids) == len(set(ids))


def test_rrf_records_per_list_contribution() -> None:
    """Fusion must be auditable: 'why did this rank third' has to be answerable."""
    fused = reciprocal_rank_fusion({"dense": ranked("a", "b"), "sparse": ranked("b", "a")})
    top = fused[0]
    assert set(top.component_scores) >= {"dense", "sparse"}


def test_rrf_weights_shift_the_ranking() -> None:
    """Weights reorder the single-list documents relative to each other.

    Note what they do *not* do: ``shared`` appears in both lists and therefore
    still wins on consensus regardless of weighting. That is correct behaviour —
    weighting tilts how much a retriever's opinion counts, it does not override
    agreement between two retrievers.
    """
    dense = ranked("d", "shared")
    sparse = ranked("s", "shared")

    dense_heavy = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 5.0, "sparse": 0.1}
    )
    by_id = {f.chunk.id: f.score for f in dense_heavy}
    assert by_id["d"] > by_id["s"], "the dense-only hit must outrank the sparse-only hit"
    assert dense_heavy[0].chunk.id == "shared", "consensus across lists still wins"

    sparse_heavy = reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse}, weights={"dense": 0.1, "sparse": 5.0}
    )
    by_id = {f.chunk.id: f.score for f in sparse_heavy}
    assert by_id["s"] > by_id["d"], "weighting must be symmetric"


def test_rrf_assigns_contiguous_ranks() -> None:
    fused = reciprocal_rank_fusion([ranked("a", "b", "c")])
    assert [f.rank for f in fused] == [0, 1, 2]


@pytest.mark.parametrize(
    "combiner",
    [reciprocal_rank_fusion, distribution_based_score_fusion, weighted_score_fusion, max_fusion],
)
def test_fusion_handles_degenerate_input(combiner) -> None:
    assert combiner([]) == []
    assert combiner([[]]) == []
    assert combiner([[], []]) == []
    single = combiner([ranked("only")])
    assert len(single) == 1


def test_fusion_handles_very_uneven_lists() -> None:
    long_list = ranked(*[f"l{i}" for i in range(40)])
    short_list = ranked("l39")  # only the long list's worst hit
    fused = reciprocal_rank_fusion([long_list, short_list])
    assert len(fused) == 40
    assert fused[0].chunk.id in {"l0", "l39"}


def test_dbsf_distinguishes_what_rrf_ties() -> None:
    """The real difference between the two combiners.

    Two documents that each top their own list are *ordinally identical*, so RRF
    scores them exactly equally — rank is all it sees. DBSF reads each document's
    position within its list's score distribution, so it separates them.

    Deliberately not asserted: that the larger *absolute* margin wins. Z-scoring
    divides by the standard deviation, so a tight list amplifies a small lead —
    the limitation the function's own docstring names. This test pins the
    property DBSF actually has, not the one the name suggests.
    """
    a = [sc("X", 0.95), sc("a2", 0.20), sc("a3", 0.10)]
    b = [sc("Y", 0.52), sc("b2", 0.50), sc("b3", 0.49)]

    rrf = reciprocal_rank_fusion([a, b])
    top_two = {f.chunk.id: f.score for f in rrf[:2]}
    assert set(top_two) == {"X", "Y"}
    assert top_two["X"] == pytest.approx(top_two["Y"]), "RRF cannot separate two rank-1 hits"

    dbsf = distribution_based_score_fusion([a, b])
    scores = {f.chunk.id: f.score for f in dbsf}
    assert scores["X"] != pytest.approx(scores["Y"]), "DBSF must separate them"
    assert dbsf[0].chunk.id in {"X", "Y"}


def test_fuse_dispatches_by_method() -> None:
    lists = [ranked("a", "b"), ranked("b", "a")]
    for method in (FusionMethod.RRF, FusionMethod.DBSF, FusionMethod.WEIGHTED, FusionMethod.MAX):
        out = fuse(lists, method, settings=Settings(security={"enforce_tenant_isolation": False}))
        assert len(out) == 2, f"{method} lost a document"


def test_fuse_respects_top_k() -> None:
    out = fuse(
        [ranked("a", "b", "c", "d")],
        FusionMethod.RRF,
        top_k=2,
        settings=Settings(security={"enforce_tenant_isolation": False}),
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# ColBERT MaxSim
# ---------------------------------------------------------------------------
def test_maxsim_matches_hand_computation() -> None:
    """Q has 2 tokens, D has 3. With orthonormal basis vectors the answer is
    countable by hand: each query token's best match is 1.0 when the document
    contains that basis vector, 0.0 otherwise."""
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    both = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    only_first = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    scores = maxsim(q, [both, only_first])
    assert scores[0] == pytest.approx(2.0), "both query tokens matched exactly"
    assert scores[1] == pytest.approx(1.0), "only the first query token matched"


def test_maxsim_handles_ragged_documents() -> None:
    """Padding must not let a shorter document borrow score from the pad rows."""
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    docs = [
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.array([[-1.0, 0.0]], dtype=np.float32),
    ]
    scores = maxsim(q, docs)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(1.0), "extra identical tokens must not inflate the score"
    assert scores[2] == pytest.approx(-1.0), (
        "a padded row must not raise the max above the real one"
    )


def test_maxsim_empty_input() -> None:
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    assert list(maxsim(q, [])) == []


# ---------------------------------------------------------------------------
# RankGPT
# ---------------------------------------------------------------------------
def test_sliding_windows_run_backwards() -> None:
    """Each pass bubbles the best candidates forward, so later windows re-rank an
    already-improved head — which only works if it starts at the end."""
    windows = list(sliding_windows(50, 10, 5))
    assert windows[0] == (40, 50)
    assert windows[-1][0] == 0
    assert all(end <= 50 for _, end in windows)


def test_sliding_windows_degenerate_sizes() -> None:
    assert list(sliding_windows(0, 10, 5)) == []
    assert list(sliding_windows(5, 10, 5)) == [(0, 5)]


@pytest.mark.parametrize(
    "raw",
    [
        [1, 1, 999],  # duplicate and out of range
        [],  # empty
        [0, 1, 2, 3, 4],  # 0-based when 1-based was requested
        [3, 2],  # incomplete
        [-1, 2, 2, 7],  # negative, duplicate, out of range
        [2, 1, 0],
    ],
)
def test_repair_permutation_never_loses_a_passage(raw: list[int]) -> None:
    """The most common RankGPT failure is a malformed permutation. Every input
    passage must survive exactly once, whatever the model returned."""
    size = 4
    order, stats = repair_permutation(raw, size)
    assert sorted(order) == list(range(size)), f"{raw} -> {order} is not a permutation"
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
@pytest.fixture
def offline_settings() -> Settings:
    return Settings(security={"enforce_tenant_isolation": False})


@pytest.fixture
def bm25_corpus() -> list[Chunk]:
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the dog sleeps in the sun all day long",
        "quantum chromodynamics describes the strong interaction",
        "the cat sat on the mat with the dog and the fox",
        "a lazy afternoon with the dog in the garden",
    ]
    return [Chunk(id=f"c{i}", content=t) for i, t in enumerate(texts)]


async def test_bm25_prefers_the_rare_term(
    bm25_corpus: list[Chunk], offline_settings: Settings
) -> None:
    """IDF is the whole point: 'quantum' appears once and must dominate 'the',
    which appears everywhere."""
    from ragorc.core.models import Query

    retriever = InMemoryBM25Retriever(bm25_corpus, settings=offline_settings)
    results = await retriever.retrieve(Query(text="quantum the"), top_k=3)
    assert results, "BM25 returned nothing"
    assert results[0].chunk.id == "c2", (
        f"rare term lost to a stopword: {[r.chunk.id for r in results]}"
    )


async def test_bm25_length_normalization(
    bm25_corpus: list[Chunk], offline_settings: Settings
) -> None:
    """Between two documents matching the same term, the shorter should not be
    penalized for its brevity."""
    from ragorc.core.models import Query

    retriever = InMemoryBM25Retriever(bm25_corpus, settings=offline_settings)
    results = await retriever.retrieve(Query(text="lazy"), top_k=5)
    ids = [r.chunk.id for r in results]
    assert "c0" in ids and "c4" in ids


async def test_bm25_unknown_term_returns_nothing(
    bm25_corpus: list[Chunk], offline_settings: Settings
) -> None:
    from ragorc.core.models import Query

    retriever = InMemoryBM25Retriever(bm25_corpus, settings=offline_settings)
    results = await retriever.retrieve(Query(text="zzzzz nonexistentterm"), top_k=5)
    assert results == [] or all(r.score == 0.0 for r in results)


async def test_bm25_scores_are_descending(
    bm25_corpus: list[Chunk], offline_settings: Settings
) -> None:
    from ragorc.core.models import Query

    retriever = InMemoryBM25Retriever(bm25_corpus, settings=offline_settings)
    results = await retriever.retrieve(Query(text="dog fox"), top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert [r.rank for r in results] == list(range(len(results)))


# ---------------------------------------------------------------------------
# DBSF: the absent-entry fill (regression)
# ---------------------------------------------------------------------------
def test_dbsf_counts_a_one_result_list_as_a_vote() -> None:
    """A list with exactly one hit must still change the ranking.

    A single entry has no standard deviation, so it z-scores to 0 — and when the
    absent entries of that row were filled with the row's *minimum* (also 0) the
    whole row became zero and the leg voted for nothing. Here two symmetric legs
    deadlock ``p`` against ``q``, so the graph leg's one confident hit is the only
    thing that can break the tie: if its vote is erased, the tie survives.
    """
    dense = [sc("p", 0.9), sc("q", 0.1)]
    sparse = [sc("q", 0.9), sc("p", 0.1)]

    deadlock = distribution_based_score_fusion({"dense": dense, "sparse": sparse})
    tied = {f.chunk.id: f.score for f in deadlock}
    assert tied["p"] == pytest.approx(tied["q"]), "the two symmetric legs must cancel"

    graph = [sc("p", 5.0, source=RetrievalSource.GRAPH_LOCAL)]
    voted = distribution_based_score_fusion({"dense": dense, "sparse": sparse, "graph": graph})
    scores = {f.chunk.id: f.score for f in voted}
    assert scores["p"] > scores["q"], "a one-result list contributed nothing"
    assert voted[0].chunk.id == "p"


def test_dbsf_components_reconstruct_the_score() -> None:
    """``component_scores`` must add up to ``score`` even for an absent list.

    The fill is a real term in the sum, so leaving it out made the audit trail
    unable to explain its own ranking: a chunk only the graph leg found reported
    ``{'graph': 0.0}`` against a negative score.
    """
    dense = [sc("a", 0.9), sc("b", 0.5), sc("c", 0.1)]
    graph = [sc("z", 100.0, source=RetrievalSource.GRAPH_LOCAL)]
    fused = distribution_based_score_fusion({"dense": dense, "graph": graph})

    for item in fused:
        per_list = {k: v for k, v in item.component_scores.items() if not k.startswith("raw_")}
        assert set(per_list) == {"dense", "graph"}, f"{item.chunk.id} lost a list"
        assert sum(per_list.values()) == pytest.approx(item.score)

    only_graph = next(f for f in fused if f.chunk.id == "z")
    # Distinguishable from a real vote: no raw_ companion, not a fusion source.
    assert only_graph.explain["fusion_absent_fill"] == {"dense": pytest.approx(-2.2247, abs=1e-3)}
    assert only_graph.explain["fusion_sources"] == ["graph"]
    assert "raw_dense" not in only_graph.component_scores


def test_rrf_does_not_report_absent_lists() -> None:
    """RRF fills with 0, so there is nothing to explain and nothing must appear.

    Guards the DBSF fix against leaking into the default method, where a wall of
    zero-valued keys would bury the legs that actually voted.
    """
    fused = reciprocal_rank_fusion({"dense": ranked("a"), "sparse": ranked("b")})
    by_id = {f.chunk.id: f for f in fused}
    assert set(by_id["a"].component_scores) == {"dense", "raw_dense"}
    assert set(by_id["b"].component_scores) == {"sparse", "raw_sparse"}
    assert all("fusion_absent_fill" not in f.explain for f in fused)


# ---------------------------------------------------------------------------
# CRAG under the call ceiling (regression)
# ---------------------------------------------------------------------------
class _StaticRetriever:
    """Returns a fixed list, so the test measures CRAG and nothing else."""

    name = "static"

    def __init__(self, chunks: list[ScoredChunk]) -> None:
        self.chunks = chunks

    async def retrieve(self, query, *, top_k: int | None = None, **kwargs):  # noqa: ANN001,ANN003
        return list(self.chunks[: top_k or len(self.chunks)])


class _LedgerLLM(StubLLM):
    """A stub that spends the ambient ledger the way the real client does.

    ``StubLLM`` never touches the ledger, so a budget test against it cannot fail
    the way production fails. These are the two lines ``OpenRouterLLM`` runs around
    every call (``_precheck`` before, ``_record`` after) and nothing else — without
    them ``BudgetExceeded`` would never surface here, which is the whole symptom.
    """

    async def structured(self, prompt, schema, **kwargs):  # noqa: ANN001,ANN003
        ledger = current_ledger()
        if ledger is not None:
            ledger.check()
        out, usage = await super().structured(prompt, schema, **kwargs)
        if ledger is not None:
            ledger.record(usage, stage=kwargs.get("stage", "unknown"))
        return out, usage


def _crag_settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={
            "crag_enabled": True,
            "crag_grade_top_k": 5,
            "crag_web_fallback": False,
            "top_k": 5,
        },
    )


def _gradable(count: int = 5) -> list[ScoredChunk]:
    """Documents long enough that refinement has strips to grade.

    ``_strips`` merges sentences until each strip clears ``min_chunk_size`` (64),
    so the sentences have to be genuinely long or the whole document collapses to
    one strip and the refine fan-out — the widest one — never happens.
    """
    body = (
        "Refund requests from the {name} region are reviewed by the billing team within two weeks. "
        "Enterprise customers in the {name} region keep a dedicated account manager for issues. "
        "Shipping is free on every {name} order above fifty US dollars in all supported areas. "
    )
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    return [
        ScoredChunk(
            chunk=Chunk(id=f"d{i}", content=body.format(name=names[i % len(names)])),
            score=0.9 - 0.05 * i,
        )
        for i in range(count)
    ]


async def test_crag_caps_the_grade_fanout_to_the_remaining_budget() -> None:
    """CRAG must size its fan-out to the budget instead of walking into it.

    With ``cost.max_llm_calls_per_query = 40`` and a Self-RAG retry re-entering
    this stage, grading five documents and then refining their ~15 strips crossed
    the ceiling mid-fan-out and ``BudgetExceeded`` left ``query()`` — no answer, no
    abstention, HTTP 500. Six calls remain here, four of them reserved for
    synthesis and verification, so exactly two grades are affordable and
    refinement is not.
    """
    llm = _LedgerLLM(responses={"RelevanceGrade": {"relevant": True, "score": 0.9}})
    crag = CorrectiveRAG(_StaticRetriever(_gradable()), llm, _crag_settings())

    with new_request_context(max_calls=40) as (_trace, ledger):
        ledger.record(Usage(model="stub/model", calls=34), stage="earlier_stages")
        result, usage = await crag.run(Query(text="How long do refunds take?"), top_k=5)
        spent = ledger.total.calls

    assert len(llm.calls_for("crag_grade")) == 2, llm.stages()
    assert llm.calls_for("crag_refine") == [], "refinement had no budget and must be skipped"
    assert usage.calls == 2
    assert spent < 40, "the ceiling must not be reached"
    assert result.chunks, "degrading must still return the retrieved evidence"
    assert "crag_budget" in result.errors


async def test_crag_passes_the_ranking_through_when_the_budget_is_spent() -> None:
    """No budget at all: no claim, no calls, and the ranking survives.

    Deliberately *not* the INCORRECT branch. A zero-length grade would label the
    corpus irrelevant and route the query to the web — one more call, on the
    strength of no verdict at all — which is the same mistake the grader-outage
    path exists to avoid.
    """
    llm = _LedgerLLM(responses={"RelevanceGrade": {"relevant": True, "score": 0.9}})
    crag = CorrectiveRAG(_StaticRetriever(_gradable()), llm, _crag_settings())

    with new_request_context(max_calls=40) as (_trace, ledger):
        ledger.record(Usage(model="stub/model", calls=38), stage="earlier_stages")
        result, usage = await crag.run(Query(text="How long do refunds take?"), top_k=3)

    assert llm.call_count == 0, llm.stages()
    assert usage.calls == 0
    assert result.grade is None, "an exhausted budget is not a verdict about the documents"
    assert [c.chunk.id for c in result.chunks] == ["d0", "d1", "d2"]
    assert "crag_budget" in result.errors


async def test_crag_fans_out_fully_when_no_ceiling_is_configured() -> None:
    """An unbounded ledger is a deliberate configuration, not a budget of zero."""
    llm = _LedgerLLM(responses={"RelevanceGrade": {"relevant": True, "score": 0.9}})
    crag = CorrectiveRAG(_StaticRetriever(_gradable()), llm, _crag_settings())

    with new_request_context(max_calls=None):
        result, _usage = await crag.run(Query(text="How long do refunds take?"), top_k=5)

    assert len(llm.calls_for("crag_grade")) == 5
    assert llm.calls_for("crag_refine"), "refinement must still run when nothing caps it"
    assert result.grade is not None
    assert "crag_budget" not in result.errors


class _StaticWebRetriever:
    """Stands in for the web leg: enabled, cheap, and records the query it got."""

    name = "web"
    enabled = True

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query, *, top_k: int | None = None, **kwargs):  # noqa: ANN001,ANN003
        self.queries.append(query.text)
        return [
            ScoredChunk(chunk=Chunk(id="w0", content="An answer from the open web."), score=0.5)
        ]


async def test_crag_skips_the_web_rewrite_rather_than_the_web_search() -> None:
    """The last calls in the budget belong to the answer, not to rephrasing.

    Six documents to grade with one call affordable, then the INCORRECT branch
    wants a rewrite it can no longer pay for. Before the fan-out was sized to the
    ledger this raised ``BudgetExceeded`` inside grading and the request produced
    nothing; now the web search still runs, on the user's own words.
    """
    llm = _LedgerLLM(responses={"RelevanceGrade": {"relevant": False, "score": 0.9}})
    settings = _crag_settings()
    settings.retrieval.crag_grade_top_k = 6
    settings.retrieval.crag_web_fallback = True
    web = _StaticWebRetriever()
    crag = CorrectiveRAG(_StaticRetriever(_gradable(6)), llm, settings, web=web)

    with new_request_context(max_calls=40) as (_trace, ledger):
        ledger.record(Usage(model="stub/model", calls=35), stage="earlier_stages")
        result, _usage = await crag.run(Query(text="Where does the SLA live?"), top_k=3)
        spent = ledger.total.calls

    assert len(llm.calls_for("crag_grade")) == 1, llm.stages()
    assert llm.calls_for("crag_rewrite") == [], "the rewrite is an optimization, the search is not"
    assert web.queries == ["Where does the SLA live?"], "unrewritten query must reach the web leg"
    assert [c.chunk.id for c in result.chunks] == ["w0"]
    assert spent < 40


# ---------------------------------------------------------------------------
# Parent-document retrieval: the query-side half of the pattern
# ---------------------------------------------------------------------------
class _ChildRetriever:
    """Returns child chunks and records the top_k it was asked for."""

    name = "children"

    def __init__(self, children: list[ScoredChunk]) -> None:
        self.children = children
        self.asked: list[int | None] = []

    async def retrieve(self, query, *, top_k=None, **kwargs):  # noqa: ANN001, ANN003
        self.asked.append(top_k)
        return list(self.children[: top_k or len(self.children)])


def _child(cid: str, parent: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=cid, content=f"child {cid}", parent_id=parent, document_id="d1"),
        score=score,
    )


async def test_parent_retriever_overfetches_children() -> None:
    """Several children of one parent collapse into one result, so asking for
    exactly ``top_k`` children reliably returns fewer than ``top_k`` parents."""
    from ragorc.core.models import Query
    from ragorc.retrieve.parent import ParentDocumentRetriever

    inner = _ChildRetriever([_child(f"c{i}", "p1", 0.9 - i * 0.01) for i in range(12)])
    retriever = ParentDocumentRetriever(
        inner, store=None, settings=Settings(security={"enforce_tenant_isolation": False})
    )
    await retriever.retrieve(Query(text="q"), top_k=4)
    assert inner.asked == [12], f"asked for {inner.asked}, expected an over-fetch of 4x3"


async def test_parent_retriever_degrades_without_a_store() -> None:
    """No parent store is a narrower answer, not a failed query: the children are
    still relevant text and still correctly ranked."""
    from ragorc.core.models import Query, RetrievalSource
    from ragorc.retrieve.parent import ParentDocumentRetriever

    inner = _ChildRetriever([_child("c1", "p1", 0.9), _child("c2", "p2", 0.8)])
    retriever = ParentDocumentRetriever(
        inner, store=None, settings=Settings(security={"enforce_tenant_isolation": False})
    )
    out = await retriever.retrieve(Query(text="q"), top_k=2)
    assert [c.chunk.id for c in out] == ["c1", "c2"]
    assert all(c.source is RetrievalSource.PARENT for c in out)
    assert [c.rank for c in out] == [0, 1]


async def test_parent_retriever_survives_a_broken_store() -> None:
    """A lookup failure must not lose a query that already found the passage."""
    from ragorc.core.models import Query
    from ragorc.retrieve.parent import ParentDocumentRetriever

    class Broken:
        async def get_chunks(self, ids):  # noqa: ANN001, ANN201
            raise RuntimeError("parent store down")

    inner = _ChildRetriever([_child("c1", "p1", 0.9)])
    retriever = ParentDocumentRetriever(
        inner, store=Broken(), settings=Settings(security={"enforce_tenant_isolation": False})
    )
    out = await retriever.retrieve(Query(text="q"), top_k=1)
    assert [c.chunk.id for c in out] == ["c1"]


async def test_parent_retriever_returns_nothing_for_nothing() -> None:
    from ragorc.core.models import Query
    from ragorc.retrieve.parent import ParentDocumentRetriever

    retriever = ParentDocumentRetriever(
        _ChildRetriever([]),
        store=None,
        settings=Settings(security={"enforce_tenant_isolation": False}),
    )
    assert await retriever.retrieve(Query(text="q"), top_k=5) == []
