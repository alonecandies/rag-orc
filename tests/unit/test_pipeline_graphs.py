"""LangGraph pipeline graphs, driven end to end with stubs.

Two claims are tested. First, that each graph **compiles** — a LangGraph mistake
(a missing reducer on a field two concurrent nodes write, a conditional edge
naming a node that does not exist) surfaces at compile time, so compiling is a
real check rather than a formality. Second, that a graph **runs to an Answer**
against stub components, with no network and no services.
"""

from __future__ import annotations

import asyncio

import pytest

from ragorc.core.models import (
    Answer,
    Chunk,
    DataStore,
    GradeLabel,
    Query,
    RetrievalResult,
    RouteDecision,
    ScoredChunk,
    Usage,
)
from ragorc.core.schemas import GroundednessGrade, RelevanceGrade, RouteOutput, UtilityGrade
from ragorc.core.settings import Settings
from ragorc.generate.answer import AnswerGenerator
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import initial_state
from ragorc.retrieve.multi_store import MultiStoreRetriever
from tests.fakes import StubLLM

POLICY = "Refunds are processed within 14 days of the request."

GRAPH_MODULES = [
    "naive",
    "adaptive",
    "crag",
    "self_rag",
    "graphrag",
    "multihop",
    "agentic",
]


class StubRetriever:
    """Returns a fixed hit and records what it was asked."""

    name = "stub"

    def __init__(self, chunks: list[ScoredChunk] | None = None) -> None:
        self.chunks = (
            chunks
            if chunks is not None
            else [
                ScoredChunk(
                    chunk=Chunk(id="c1", content=POLICY, document_id="d1"), score=0.92, rank=0
                )
            ]
        )
        self.calls: list[str] = []

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs
    ) -> list[ScoredChunk]:
        self.calls.append(query.text)
        return list(self.chunks)

    async def retrieve_detailed(self, query: Query, **kwargs) -> RetrievalResult:
        self.calls.append(query.text)
        return RetrievalResult(chunks=list(self.chunks), per_store={"vector": list(self.chunks)})


@pytest.fixture
def settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k", "context_window": 8000},
        generation={"check_groundedness": True, "cite_sources": True},
    )


@pytest.fixture
def llm() -> StubLLM:
    return StubLLM(
        text="Refunds are processed within 14 days [1].",
        responses={
            "GroundednessGrade": GroundednessGrade(grounded=True, score=0.95),
            "UtilityGrade": UtilityGrade(useful=True, score=0.92),
            "RelevanceGrade": RelevanceGrade(relevant=True, score=0.9),
            "RouteOutput": RouteOutput(datastores=["vector"], confidence=0.9),
        },
    )


@pytest.fixture
def nodes(llm: StubLLM, settings: Settings) -> PipelineNodes:
    """A fully-wired node set.

    ``store_retrievers`` matters: the adaptive graph fans out to the stores the
    route chose, so a node set with only the default ``retriever`` correctly
    reports "no retriever configured for this store" and abstains — graceful, but
    not what these tests are exercising.
    """
    stub = StubRetriever()
    return PipelineNodes(
        llm=llm,
        generator=AnswerGenerator(llm, settings),
        retriever=stub,
        store_retrievers={
            DataStore.VECTOR: stub,
            DataStore.RELATIONAL: stub,
            DataStore.GRAPH: stub,
        },
        graph_retrievers={"local": stub, "global": stub, "drift": stub},
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GRAPH_MODULES)
def test_graph_compiles(name: str, nodes: PipelineNodes, settings: Settings) -> None:
    """Compilation catches the LangGraph mistakes that matter: a concurrent write
    to a field with no reducer, and a conditional edge naming a missing node."""
    import importlib

    module = importlib.import_module(f"ragorc.pipeline.graphs.{name}")
    compiled = module.build(nodes, settings=settings)
    assert compiled is not None


@pytest.mark.parametrize("name", GRAPH_MODULES)
def test_graph_is_drawable(name: str, nodes: PipelineNodes, settings: Settings) -> None:
    """A compiled graph must be printable, so the control flow is a diagram rather
    than something reconstructed by reading call sites."""
    import importlib

    module = importlib.import_module(f"ragorc.pipeline.graphs.{name}")
    compiled = module.build(nodes, settings=settings)
    mermaid = compiled.get_graph().draw_mermaid()
    assert "graph" in mermaid.lower() or "flowchart" in mermaid.lower()
    assert "generate" in mermaid


@pytest.mark.parametrize("name", GRAPH_MODULES)
def test_graph_has_a_generate_node(name: str, nodes: PipelineNodes, settings: Settings) -> None:
    import importlib

    module = importlib.import_module(f"ragorc.pipeline.graphs.{name}")
    compiled = module.build(nodes, settings=settings)
    node_names = set(compiled.get_graph().nodes)
    assert any("generate" in n for n in node_names), node_names


@pytest.mark.parametrize("name", GRAPH_MODULES)
def test_graph_declares_a_recursion_limit(name: str) -> None:
    """Cyclic graphs need a bound. The limit is a safety net that raises; the
    graphs also carry an explicit iteration counter for a graceful exit."""
    import importlib

    module = importlib.import_module(f"ragorc.pipeline.graphs.{name}")
    limit = module.recursion_limit(Settings(security={"enforce_tenant_isolation": False}))
    assert isinstance(limit, int) and limit > 0


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
async def test_naive_graph_produces_an_answer(nodes: PipelineNodes, settings: Settings) -> None:
    from ragorc.pipeline.graphs import naive

    compiled = naive.build(nodes, settings=settings)
    state = await compiled.ainvoke(
        initial_state("How long do refunds take?"),
        config={"recursion_limit": naive.recursion_limit(settings)},
    )
    answer = state.get("answer")
    assert isinstance(answer, Answer)
    assert not answer.abstained, answer.abstain_reason
    assert "14 days" in answer.text
    assert answer.citations, "the answer must carry resolvable citations"


async def test_adaptive_graph_routes_and_answers(
    nodes: PipelineNodes, settings: Settings, llm: StubLLM
) -> None:
    from ragorc.pipeline.graphs import adaptive

    compiled = adaptive.build(nodes, settings=settings)
    state = await compiled.ainvoke(
        initial_state("How long do refunds take?"),
        config={"recursion_limit": adaptive.recursion_limit(settings)},
    )
    answer = state.get("answer")
    assert isinstance(answer, Answer)
    assert not answer.abstained, answer.abstain_reason


async def test_graph_abstains_when_retrieval_is_empty(llm: StubLLM, settings: Settings) -> None:
    """The end-to-end proof of the central claim: no evidence yields an explicit
    abstention, not a confident answer from the model's parameters."""
    from ragorc.pipeline.graphs import naive

    nothing = StubRetriever(chunks=[])
    empty = PipelineNodes(
        llm=llm,
        generator=AnswerGenerator(llm, settings),
        retriever=nothing,
        store_retrievers={DataStore.VECTOR: nothing},
        settings=settings,
    )
    compiled = naive.build(empty, settings=settings)
    state = await compiled.ainvoke(
        initial_state("something the corpus cannot answer"),
        config={"recursion_limit": naive.recursion_limit(settings)},
    )
    answer = state.get("answer")
    assert isinstance(answer, Answer)
    assert answer.abstained
    assert answer.abstain_reason


async def test_graph_records_usage_for_costing(nodes: PipelineNodes, settings: Settings) -> None:
    from ragorc.pipeline.graphs import naive

    compiled = naive.build(nodes, settings=settings)
    state = await compiled.ainvoke(
        initial_state("How long do refunds take?"),
        config={"recursion_limit": naive.recursion_limit(settings)},
    )
    answer = state["answer"]
    assert answer.usage.calls >= 1, "every request must be costed"


async def test_state_accumulates_across_nodes(settings: Settings) -> None:
    """Fields written by concurrent nodes need a reducer, or LangGraph raises on
    the second write. This checks the reducers exist and compose."""
    from ragorc.core.models import Usage
    from ragorc.pipeline.state import merge_store_lists, total_usage

    merged = merge_store_lists({"vector": [1]}, {"graph": [2]})
    assert set(merged) == {"vector", "graph"}
    # Both keys survive: without a reducer, LangGraph raises when a second
    # concurrent node writes the same channel.
    overlapping = merge_store_lists({"vector": [1]}, {"vector": [2]})
    assert len(overlapping["vector"]) == 2

    state = {"usages": [Usage(calls=1, cost_usd=0.1), Usage(calls=2, cost_usd=0.2)]}
    assert total_usage(state).calls == 3
    assert total_usage({}).calls == 0


# ---------------------------------------------------------------------------
# Failure policy at the retrieval boundary
# ---------------------------------------------------------------------------
class SelfBoundingRetriever:
    """Shaped like :class:`~ragorc.retrieve.hybrid.HybridRetriever`.

    It bounds each of its own legs at ``retrieval.per_store_timeout_s`` and
    returns what the healthy legs produced, recording the leg it lost in
    ``RetrievalResult.errors`` — the degradation the docs promise. Reproducing
    *that* shape is the whole point: the bug the two tests below cover is a
    second deadline wrapped around a component which already has one.
    """

    name = "hybrid"

    def __init__(self, settings: Settings, chunks: list[ScoredChunk]) -> None:
        self.settings = settings
        self.chunks = chunks

    async def retrieve_detailed(self, query: Query, *, top_k=None, **kw) -> RetrievalResult:
        budget = float(self.settings.retrieval.per_store_timeout_s)
        errors: dict[str, str] = {}
        try:
            await asyncio.wait_for(asyncio.sleep(60), budget)
        except TimeoutError:
            errors["fulltext"] = f"timed out after {budget}s"
        return RetrievalResult(
            chunks=list(self.chunks),
            per_store={"dense": list(self.chunks)},
            errors=errors,
            total_candidates=len(self.chunks),
        )

    async def retrieve(self, query: Query, *, top_k=None, **kw) -> list[ScoredChunk]:
        return (await self.retrieve_detailed(query, top_k=top_k, **kw)).chunks


def _bounded_nodes(llm: StubLLM, settings: Settings) -> PipelineNodes:
    hit = ScoredChunk(chunk=Chunk(id="c1", content=POLICY, document_id="d1"), score=0.9, rank=0)
    retriever = SelfBoundingRetriever(settings, [hit])
    return PipelineNodes(
        llm=llm,
        generator=AnswerGenerator(llm, settings),
        retriever=retriever,
        store_retrievers={DataStore.VECTOR: retriever},
        settings=settings,
    )


@pytest.fixture
def tight_deadline() -> Settings:
    """The shipped configuration, with the deadline scaled down for a test.

    ``per_store_timeout_s`` is the *only* value changed: the defect is that the
    node's deadline and the retriever's leg deadline read this one setting, so a
    test that gave the outer timer a different budget would not be testing it.
    """
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k", "context_window": 8000},
        retrieval={"per_store_timeout_s": 0.25},
    )


async def test_store_node_keeps_the_legs_that_answered_when_one_leg_hangs(
    llm: StubLLM, tight_deadline: Settings
) -> None:
    """One slow optional leg must not delete the healthy legs' results.

    A retriever that bounds its own legs comes back *with* chunks and an error for
    the leg it lost. Wrapping it in a second deadline reading the same setting made
    the node throw that away — the outer timer is armed first, so it always fires
    first — and the query abstained on "insufficient context" while Qdrant had
    answered in milliseconds.
    """
    nodes = _bounded_nodes(llm, tight_deadline)
    out = await nodes.store_node(DataStore.VECTOR)({"query": Query(text="q"), "top_k": 3})

    assert len(out["candidates"]) == 1, "the healthy leg's chunk must survive the slow leg"
    assert out["per_store"]["vector"], "the store contributed, so it must not read as empty"
    # And the leg that was lost is still reported: surviving the outage silently
    # would trade one bug for another.
    assert out["errors"] == ["vector/fulltext: timed out after 0.25s"]


async def test_retrieve_node_puts_failed_legs_on_the_state(
    llm: StubLLM, tight_deadline: Settings
) -> None:
    """``RetrievalResult.errors`` has to reach the ``errors`` channel.

    That channel is what ``answer.metadata["errors"]``, the ``query_answered`` log
    line and the semantic cache's degraded-answer refusal are built from. While the
    node returned the result alone, a retrieval outage reported ``errors=0`` and the
    degraded answer was cacheable for the full TTL.
    """
    nodes = _bounded_nodes(llm, tight_deadline)
    out = await nodes.retrieve({"query": Query(text="q"), "top_k": 3})

    assert out["retrieval"].errors == {"fulltext": "timed out after 0.25s"}
    assert out["errors"] == ["fulltext: timed out after 0.25s"]


async def test_multi_store_keeps_the_legs_that_answered_when_one_leg_hangs(
    tight_deadline: Settings,
) -> None:
    """The same second deadline, in the fan-out the adaptive graph retrieves through.

    ``MultiStoreRetriever`` is wired with the hybrid retriever as its vector store,
    so its per-store deadline wrapped a component holding the same budget — with one
    consequence the graph node does not have: this layer owns the circuit breakers,
    so every such request also recorded an outage against a store that had answered,
    and four of them cut it out for the whole cooldown.
    """
    hit = ScoredChunk(chunk=Chunk(id="c1", content=POLICY, document_id="d1"), score=0.9, rank=0)
    retriever = SelfBoundingRetriever(tight_deadline, [hit])
    fan_out = MultiStoreRetriever(retrievers={DataStore.VECTOR: retriever}, settings=tight_deadline)
    breaker = fan_out.breakers[DataStore.VECTOR]
    # Tripped by one failure instead of four, so a single request settles whether a
    # partial answer is counted as an outage — the loop that would otherwise be
    # needed here only spends the deadline four times to learn the same thing.
    breaker.failure_threshold = 1
    route = RouteDecision(stores=(DataStore.VECTOR,), method="test", reasoning="test")

    result = await fan_out.retrieve_detailed(Query(text="q"), route=route)

    assert len(result.chunks) == 1, "the healthy leg's chunk must survive the slow leg"
    assert result.per_store["vector"], "the store contributed, so it must not read as empty"
    # Named by store *and* leg, because ``fulltext`` is one leg of the vector store
    # rather than a store of its own.
    assert result.errors == {"vector/fulltext": "timed out after 0.25s"}
    assert not breaker.is_open, "a store that answered must not be recorded as an outage"


async def test_fused_result_carries_the_fan_outs_errors(
    nodes: PipelineNodes, settings: Settings
) -> None:
    """The fused result is the authority, so it has to admit what failed.

    The parallel legs can only report through the ``errors`` channel, so the
    :class:`RetrievalResult` ``fuse`` assembles was built with ``errors={}`` — the
    one field ``docs/operations.md`` tells operators to alert on. Stage failures
    that are not legs stay out of it.
    """
    hit = ScoredChunk(chunk=Chunk(id="c1", content=POLICY, document_id="d1"), score=0.9, rank=0)
    out = await nodes.fuse(
        {
            "query": Query(text="q"),
            "per_store": {"vector": [hit], "relational": []},
            "errors": [
                "relational: StoreUnavailable: postgres is down",
                "translate: LLMError: rate limited",
            ],
        }
    )

    assert out["retrieval"].errors == {"relational": "StoreUnavailable: postgres is down"}
    assert out["retrieval"].chunks, "a failed leg must not cost the healthy leg its chunks"


# ---------------------------------------------------------------------------
# Rewrite-Retrieve-Read on the graph path
# ---------------------------------------------------------------------------
async def test_rrr_rewrites_before_retrieving_when_enabled(
    llm: StubLLM, settings: Settings
) -> None:
    """`generation.rrr_enabled` did nothing on every graph path.

    RRR was constructed in exactly one place — the HTTP service's *linear
    fallback* engine, reached only when the orchestrator is absent — while
    `describe()` reported RRR as an enabled feature. So the flag documented as
    "rewrite the question for search before retrieving" was, on the supported
    path, a no-op that still advertised itself.
    """
    asked: list[str] = []

    class _Recording(StubRetriever):
        async def retrieve_detailed(self, query: Query, **kwargs: object) -> RetrievalResult:
            asked.append(query.text)
            return await super().retrieve_detailed(query, **kwargs)

    on = Settings(**{**settings.model_dump(), "generation": {"rrr_enabled": True}})
    nodes = PipelineNodes(
        llm=llm, generator=AnswerGenerator(llm, on), retriever=_Recording(), settings=on
    )
    result, usages = await nodes._retrieve_with(
        nodes.retriever,
        Query(text="i've been waiting ages, when do i get my money back?"),
        route=None,
        top_k=3,
    )
    assert result.chunks, "retrieval must still return its hits"
    assert asked, "the retriever must have been called"
    assert usages and usages[0].calls, "the rewrite's cost must reach the ledger"


async def test_rrr_costs_nothing_when_disabled(llm: StubLLM, settings: Settings) -> None:
    """Off by default, and off means no model call and the original question."""
    question = "when do i get my money back?"
    asked: list[str] = []

    class _Recording(StubRetriever):
        async def retrieve_detailed(self, query: Query, **kwargs: object) -> RetrievalResult:
            asked.append(query.text)
            return await super().retrieve_detailed(query, **kwargs)

    off = Settings(**{**settings.model_dump(), "generation": {"rrr_enabled": False}})
    nodes = PipelineNodes(
        llm=llm, generator=AnswerGenerator(llm, off), retriever=_Recording(), settings=off
    )
    _, usages = await nodes._retrieve_with(
        nodes.retriever, Query(text=question), route=None, top_k=3
    )
    assert asked == [question], "the question must reach the retriever unrewritten"
    assert usages == [], "a disabled feature must not bill anything"


# ---------------------------------------------------------------------------
# The agentic graph has to keep the verdict it paid for
# ---------------------------------------------------------------------------
GOOD = ScoredChunk(chunk=Chunk(id="good", content=POLICY, document_id="d1"), score=0.9, rank=0)
JUNK = ScoredChunk(
    chunk=Chunk(id="junk", content="Our office is in Berlin.", document_id="d2"),
    score=0.88,
    rank=1,
)


class StubCrag:
    """A CRAG stage that grades everything irrelevant and returns nothing.

    Shaped like the real one in the detail that matters: ``per_store`` carries
    the *pre-grading* candidates, because that is what per-leg diagnostics are
    (``crag.py``: ``result.per_store[self.base.name] = list(initial)``), while
    ``chunks`` carries the verdict. Any collect step that rebuilds the ranking
    from ``per_store`` resurrects exactly what the grader threw out.
    """

    def __init__(self, kept: list[ScoredChunk], candidates: list[ScoredChunk]) -> None:
        self.kept = kept
        self.candidates = candidates

    async def run(self, query: Query, **kwargs: object) -> tuple[RetrievalResult, Usage]:
        del query, kwargs
        return (
            RetrievalResult(
                chunks=list(self.kept),
                per_store={"vector": list(self.candidates)},
                total_candidates=len(self.candidates),
                grade=GradeLabel.INCORRECT if not self.kept else GradeLabel.CORRECT,
            ),
            Usage(),
        )


async def test_collect_keeps_crags_verdict_instead_of_the_raw_candidates(
    nodes: PipelineNodes, settings: Settings
) -> None:
    """The finding, at the node that had it.

    ``fuse`` was wired in as the agentic graph's collect step. It rebuilds the
    ranking from ``per_store``, so grading five documents and discarding four
    produced a context containing all five.
    """
    nodes.crag = StubCrag(kept=[], candidates=[GOOD, JUNK])
    state = initial_state("what is the refund window?")
    state.update(await nodes.validate(state))
    graded = await nodes.grade(state)
    state.update({k: v for k, v in graded.items() if k != "per_store"})
    state["per_store"] = graded["per_store"]

    collected = await nodes.collect(state)

    assert [c.chunk.id for c in collected["retrieval"].chunks] == [], (
        "the grader discarded every document; the generator must receive none"
    )
    assert collected["retrieval"].total_candidates == 2, (
        "the per-leg diagnostics must still report what the store returned"
    )


async def test_collect_merges_hop_results_into_the_endorsed_set(
    nodes: PipelineNodes, settings: Settings
) -> None:
    """Hops are additions, not replacements.

    ``hop`` writes only ``candidates`` and its own ``per_store`` leg, leaving
    ``retrieval`` alone by design — so collect has to merge them in, or the
    evidence a hop went and fetched never reaches the answer.
    """
    hopped = ScoredChunk(
        chunk=Chunk(
            id="hopped", content="Refunds are issued to the original card.", document_id="d3"
        ),
        score=0.8,
    )
    state = initial_state("q")
    state.update(await nodes.validate(state))
    state["retrieval"] = RetrievalResult(chunks=[GOOD], per_store={"vector": [GOOD, JUNK]})
    state["per_store"] = {"vector": [GOOD, JUNK], "hop_1": [hopped]}

    collected = await nodes.collect(state)

    ids = {c.chunk.id for c in collected["retrieval"].chunks}
    assert ids == {"good", "hopped"}, f"expected the endorsed chunk plus the hop, got {ids}"


async def test_collect_ignores_evidence_from_a_rejected_attempt(
    nodes: PipelineNodes, settings: Settings
) -> None:
    """``per_store`` is a reducer; ``retrieval`` is not.

    When Self-RAG rejects an answer the graph re-enters ``grade``, and
    ``merge_store_lists`` concatenates the second grading's legs onto the
    first's. Fusing from ``per_store`` therefore mixed the evidence of the
    rejected attempt into the retry. Reading the last-written ``retrieval``
    cannot, and this pins that.
    """
    state = initial_state("q")
    state.update(await nodes.validate(state))
    # What the channel looks like after two passes through `grade`.
    state["per_store"] = {"vector": [JUNK, GOOD]}
    state["retrieval"] = RetrievalResult(chunks=[GOOD], per_store={"vector": [JUNK, GOOD]})

    collected = await nodes.collect(state)

    assert [c.chunk.id for c in collected["retrieval"].chunks] == ["good"]


async def test_agentic_abstains_when_crag_rejects_the_whole_corpus(
    nodes: PipelineNodes, llm: StubLLM
) -> None:
    """End to end through the compiled graph.

    The abstention signal is the entire reason CRAG is in this pipeline. With
    the retrieval budget at zero the graph goes ``grade -> gate -> collect``,
    so this isolates the collect step from the corrective loops.
    """
    from ragorc.pipeline.graphs import agentic

    tuned = Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k", "context_window": 8000},
        generation={"rrr_max_rewrites": 0, "self_rag_max_retries": 0},
        graph={"multihop_max_iterations": 0},
    )
    nodes.crag = StubCrag(kept=[], candidates=[GOOD, JUNK])
    compiled = agentic.build(nodes, settings=tuned)

    final = await compiled.ainvoke(
        initial_state("what is the refund window?", pipeline="agentic"),
        {"recursion_limit": agentic.recursion_limit(tuned)},
    )

    retrieval = final.get("retrieval")
    assert retrieval is not None and list(retrieval.chunks) == [], (
        f"the generator was handed {[c.chunk.id for c in retrieval.chunks]} after CRAG "
        "rejected every document"
    )


# ---------------------------------------------------------------------------
# The multi-hop early exit has to be able to see the hop
# ---------------------------------------------------------------------------
HOP_FINDING = "The founder of Acme studied at Cambridge."


async def test_sufficiency_check_reads_what_the_last_hop_retrieved(
    nodes: PipelineNodes, llm: StubLLM
) -> None:
    """``multihop_stop_on_sufficient`` was dead after hop 0.

    ``hop`` writes ``candidates`` and its own ``per_store`` leg and leaves
    ``retrieval`` for ``collect``. The check read ``evidence``, which is built
    from ``retrieval`` — so on every iteration after the first it was handed the
    *first* retrieval, judged passages it had already called insufficient, and
    returned the same verdict. The loop could only ever end on the budget, and
    every multi-hop query paid for all its hops however early the answer arrived.
    """
    state = initial_state("which university did Acme's founder attend?")
    state.update(await nodes.validate(state))
    # The state as it stands one hop in: `retrieval` is still hop 0's, and what
    # hop 1 found is in `candidates` and nowhere else.
    state["retrieval"] = RetrievalResult(chunks=[GOOD])
    state["candidates"] = [
        GOOD,
        ScoredChunk(chunk=Chunk(id="hop1", content=HOP_FINDING, document_id="d9"), score=0.87),
    ]

    await nodes.check_sufficiency(state)

    prompt = llm.calls_for("multihop_reason")[0]["prompt"]
    assert HOP_FINDING in prompt, (
        "the sufficiency call cannot decide the loop is done without seeing what "
        "the last hop retrieved"
    )


async def test_sufficiency_check_does_not_repeat_a_passage_it_already_has(
    nodes: PipelineNodes, llm: StubLLM
) -> None:
    """Consecutive hops on one topic overlap heavily, and this is a call about
    *confidence* — the same fact asserted three times makes a model more sure
    rather than better informed."""
    state = initial_state("q")
    state.update(await nodes.validate(state))
    state["retrieval"] = RetrievalResult(chunks=[GOOD])
    state["candidates"] = [GOOD, GOOD]

    await nodes.check_sufficiency(state)

    prompt = llm.calls_for("multihop_reason")[0]["prompt"]
    assert prompt.count(POLICY) == 1


# ---------------------------------------------------------------------------
# Streaming has to stream the pipeline that was asked for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", GRAPH_MODULES)
async def test_streaming_variant_runs_retrieval_and_stops_before_generation(
    name: str, nodes: PipelineNodes, llm: StubLLM, settings: Settings
) -> None:
    """``interrupt_before=["generate"]`` is what lets a stream run the real graph.

    Both halves matter: the retrieval side must actually execute, and generation
    must not — otherwise the answer would be produced before the first token was
    emitted, which is a slower non-streaming request rather than a stream.
    """
    from ragorc.pipeline.graphs import GRAPHS, build_graph

    # `crag` and `agentic` retrieve *through* their CRAG stage, so without one
    # wired they skip grading and never retrieve at all — which would make this
    # assertion pass for the wrong reason on the graphs it most needs to cover.
    nodes.crag = StubCrag(kept=[GOOD], candidates=[GOOD, JUNK])
    compiled = build_graph(name, nodes, settings=settings, interrupt_before=["generate"])
    final = await compiled.ainvoke(
        initial_state("what is the refund window?", pipeline=name),
        {"recursion_limit": GRAPHS[name].recursion_limit(settings)},
    )

    assert final.get("answer") is None, "generation ran despite the interrupt"
    assert "answer" not in llm.stages(), f"the generator was called: {llm.stages()}"
    assert final.get("retrieval") is not None or final.get("candidates"), (
        "the retrieval side did not run"
    )


@pytest.mark.parametrize("name", GRAPH_MODULES)
def test_the_streaming_variant_is_the_same_graph(
    name: str, nodes: PipelineNodes, settings: Settings
) -> None:
    """No second, hand-written approximation of the pipeline.

    The bug this replaces was exactly that: a fixed five-node list stood in for
    every graph, so it could not drift out of step with the definitions because
    it was never in step with them.
    """
    from ragorc.pipeline.graphs import build_graph

    streaming = build_graph(name, nodes, settings=settings, interrupt_before=["generate"])
    plain = build_graph(name, nodes, settings=settings)
    assert set(streaming.get_graph().nodes) == set(plain.get_graph().nodes)


async def test_streaming_graphrag_searches_the_graph(
    nodes: PipelineNodes, settings: Settings
) -> None:
    """Streaming ``graphrag`` used to perform no traversal whatsoever.

    The hand-written sequence was ``validate, translate, route,
    grade|retrieve, rerank`` — no ``classify``, and none of the local, global or
    DRIFT nodes. A caller asking to stream GraphRAG got hybrid search under the
    name and no way to tell.
    """
    from ragorc.pipeline.graphs import GRAPHS, build_graph

    compiled = build_graph("graphrag", nodes, settings=settings, interrupt_before=["generate"])
    final = await compiled.ainvoke(
        initial_state("how is Acme related to Beta?", pipeline="graphrag"),
        {"recursion_limit": GRAPHS["graphrag"].recursion_limit(settings)},
    )

    assert final.get("search_mode") in {"local", "global", "drift"}, (
        f"no graph search mode was taken: {final.get('search_mode')!r}"
    )
    assert any(leg.startswith("graph_") for leg in (final.get("per_store") or {})), (
        f"no graph leg contributed: {sorted(final.get('per_store') or {})}"
    )


async def test_streaming_multihop_consults_the_hop_loop(
    nodes: PipelineNodes, llm: StubLLM, settings: Settings
) -> None:
    """And ``multihop`` used to take no hops and never touch the bridge."""
    from ragorc.pipeline.graphs import GRAPHS, build_graph

    tuned = Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k", "context_window": 8000},
        graph={"multihop_max_iterations": 3},
    )
    compiled = build_graph("multihop", nodes, settings=tuned, interrupt_before=["generate"])
    await compiled.ainvoke(
        initial_state("which university did Acme's founder attend?", pipeline="multihop"),
        {"recursion_limit": GRAPHS["multihop"].recursion_limit(tuned)},
    )

    assert llm.calls_for("multihop_reason"), (
        f"the sufficiency check never ran; stages were {llm.stages()}"
    )


async def test_a_rewrite_drops_everything_derived_from_the_failed_question() -> None:
    """`_respell`'s docstring gives the rule — the vectors go "because they
    belong to the old text" — and two things survived it anyway.

    The variants were produced *from the question that failed*, and every store
    retriever expands `all_texts` into one ranked list per text and fuses with
    RRF: with three stale variants the rewritten question gets one slot in four,
    outvoted by the thing it was rewriting. The HyDE document is worse now that
    the retriever actually embeds it — the rewrite would search using a
    hypothetical answer to the previous question.
    """
    from ragorc.core.models import Query
    from ragorc.pipeline.nodes import PipelineNodes

    original = Query(
        text="how do I rotate the signing key?",
        variants=("what is key rotation", "signing key docs"),
        hypothetical="a hypothetical answer to the OLD question",
        metadata={"hyde_documents": ["stale document"], "hyde": True, "keep_me": 1},
    )

    rewritten = PipelineNodes._respell(original, "how do I rotate the signing secret?", reason="rewrite")

    assert rewritten.variants == (), f"stale variants survived: {rewritten.variants}"
    assert not rewritten.metadata.get("hyde_documents"), "a stale hypothetical survived"
    assert rewritten.hypothetical is None
    assert rewritten.metadata["keep_me"] == 1, "unrelated metadata must survive"
    assert rewritten.metadata["rewrite_of"] == original.text
    assert original.variants, "the original must not be mutated"


async def test_graphrag_pays_for_no_variants_it_cannot_read() -> None:
    """`classify` sends this graph to local, global or DRIFT search — all Neo4j
    traversals reading `query.text`. No store retriever is reached, so the
    variants every other graph expands into extra ranked lists were generated,
    billed and dropped on the floor."""
    from ragorc.core.models import Query
    from ragorc.core.settings import Settings
    from ragorc.pipeline.nodes import PipelineNodes
    from ragorc.pipeline.state import initial_state

    class Expanding:
        async def translate(self, query: Query) -> tuple[Query, object]:
            from ragorc.core.models import Usage

            return (
                Query(text=query.text, variants=("a", "b", "c"), original=query.original),
                Usage(model="stub", calls=1),
            )

    settings = Settings(llm={"api_key": "k"}, cache={"enabled": False},
                        embedding={"dense_dimension": 32})
    from ragorc.generate.answer import AnswerGenerator
    from tests.fakes import StubLLM

    llm = StubLLM()
    nodes = PipelineNodes(
        settings=settings, llm=llm, retriever=None, generator=AnswerGenerator(llm, settings)
    )
    nodes.translator = Expanding()
    state = initial_state("what themes appear across the corpus?")
    state["query"] = Query(text="what themes appear across the corpus?")

    kept = await nodes.translate(state, variants=True)
    assert kept["query"].variants == ("a", "b", "c"), "the translator produced none to drop"

    # Through the graph's own node, not `nodes.translate` directly. Asserting on
    # the parameter proves the mechanism; only invoking what `graphrag.build`
    # bound proves the graph uses it — a mutation reverting the binding survived
    # the direct call, which is the call-site gap this codebase is named for.
    from ragorc.pipeline.graphs import build_graph

    compiled = build_graph("graphrag", nodes, settings=settings)
    node = compiled.nodes["translate"].bound
    dropped = await node.ainvoke(dict(state))

    assert dropped["query"].variants == (), (
        f"graphrag carried variants no retriever on its path reads: {dropped['query'].variants}"
    )
