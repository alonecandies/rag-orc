"""The agentic pipeline: every mechanism in the library, with the model steering.

    validate -> translate -> construct -> route -> grade -+-> refine   -> gate
                                                          +-> web      -> gate | grade
                                                          +-> rewrite   -> web -> grade
                                          gate -+-> check -+-> hop -> gate
                                                |          +-> collect
                                                +-> collect
                              collect -> rerank -> compress -> generate
                                       generate -> verify(x2) -> judge -+-> END
                                                                        +-> rewrite -> grade
                                                                        +-> abstain -> END

Three feedback loops run in one graph: CRAG on the documents, multi-hop on the
evidence chain, Self-RAG on the answer. Every translation strategy runs, self-query
extracts metadata filters, the router picks the backends, and the multi-store fan-out
consults them concurrently.

What "the LLM decides which tools to use" means here
---------------------------------------------------
There is deliberately **no bespoke "pick a tool" call**. Every branch in this graph is
already driven by a schema-constrained model decision that exists for its own reasons:

=====================================  ==========================================
 decision (schema)                      tool it selects
=====================================  ==========================================
 ``RouteOutput`` (route)                which datastores to query, or none at all
 ``MetadataFilterOutput`` (construct)   which metadata filters to apply
 ``RelevanceGrade`` (grade / CRAG)      refine, add web results, or rewrite+retry
 ``SufficiencyCheck`` (check)           take another hop, or stop
 ``GroundednessGrade`` + ``UtilityGrade`` accept the answer, retry, or abstain
=====================================  ==========================================

Adding a planner call on top would buy nothing and cost something real: a model asked
"which tools should I use" answers from the question text alone, which is exactly the
information the router already has and strictly less than the grader has — the grader
has *seen the documents*. So the agency here is distributed across the decisions that
each have the evidence to make them, and the "agent loop" is the graph.

It also means every branch is already cost-tiered. All five of those calls route to
``fast_model`` through :class:`~ragorc.llm.router.ModelRouter`; only synthesis reaches
the balanced tier. A planner call would have been the one uncosted addition.

What it costs, relative to ``adaptive``
--------------------------------------
Order of magnitude, one query, default settings, no cache hits:

===================  =====================  ==========================
 stage                adaptive               agentic
===================  =====================  ==========================
 translation          1 (or 0)               3-4 (all strategies)
 self-query           0                      1
 routing              1                      1
 document grading     0                      5 (``crag_grade_top_k``)
 refinement           0                      ~10-20 strips
 rewrites             0                      0-2
 web search           0                      0-1 (plus a rewrite each)
 sufficiency          0                      0-3
 answer synthesis     1                      1-3 (Self-RAG retries)
 answer verification  1-2                    2-6
===================  =====================  ==========================
 **total calls**      **4-6**                **20-40**
 **cheap-tier share** ~70%                   ~95%
===================  =====================  ==========================

So roughly **5-8x the calls** but only about **2-3x the dollars**, because almost
everything added is on the cheap tier — that ratio is the entire point of the cascade
(ADR-0005). Latency is the harsher cost: the loops are sequential by nature, so a
worst-case agentic query can be several seconds slower than an adaptive one even
though most of the added calls are individually fast.

Choose it when answers are consumed by people who will act on them and a wrong answer
is expensive; choose ``adaptive`` when volume matters and the corpus reliably contains
the answer. Measure first — :mod:`ragorc.pipeline.graphs.naive` exists to be the
control, and on a clean corpus the loops routinely arrive at the same evidence the
baseline found.

Two ceilings, and why the graph anticipates the one it cannot catch
------------------------------------------------------------------
:class:`~ragorc.core.telemetry.CostLedger` enforces ``cost.max_llm_calls_per_query``,
``max_cost_per_query_usd`` and ``max_tokens_per_query`` by raising
:class:`~ragorc.core.errors.BudgetExceeded` *before* each call — that is the hard stop,
and it is what makes a graph with three loops safe to run at all.

But a raise mid-loop discards the answer, the trace and the evidence. So this graph
also *anticipates* the ceiling: :func:`budget_available` asks the live ledger whether
enough call budget remains to finish (synthesis plus verification), and every loop
predicate declines to iterate again when it does not. The loops therefore leave through
``generate`` and produce a real, cited, possibly-abstaining answer instead of a
traceback. The ledger stays as the backstop for the case anticipation cannot cover —
one node making more calls than expected.

One retrieval-side budget, not two
----------------------------------
Corrective rewrites and multi-hop hops share ``retrieve_iterations``. They are both
"retrieve again", and a graph that granted each loop its full budget independently
would retrieve up to ``rrr_max_rewrites + multihop_max_iterations`` times for one
question — five by default, which is not what either setting is asking for. The shared
ceiling is the larger of the two, spent on whichever mechanism the question turns out
to need.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.models import GradeLabel
from ragorc.core.settings import Settings
from ragorc.core.telemetry import current_ledger
from ragorc.pipeline.graphs.self_rag import judge, verdict
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

log = structlog.get_logger(__name__)

__all__ = [
    "NAME",
    "budget_available",
    "build",
    "decide_after_grade",
    "decide_after_judge",
    "recursion_limit",
    "retrieval_budget",
]

NAME = "agentic"

_GRADE = "grade"
_REFINE = "refine"
_WEB = "web_search"
_REWRITE = "rewrite"
_GATE = "gate"
_CHECK = "check_sufficiency"
_HOP = "hop"
_COLLECT = "collect"
_GENERATE = "generate"
_ABSTAIN = "abstain"
_ACCEPT = "accept"
_RETRY = "retry"

_RESERVED_CALLS = 4
"""Calls held back for the epilogue: synthesis, groundedness, utility, and one spare.

Reserved rather than measured because the exact epilogue cost depends on settings
(claim decomposition, self-consistency samples) and the point is to be *conservative*:
under-reserving means the loop spends the budget the answer needed and the run ends in
a raise, which is the outcome the reservation exists to prevent."""


def retrieval_budget(settings: Settings) -> int:
    """The shared corrective/hop ceiling: the larger of the two configured loops."""
    return max(
        int(settings.generation.rrr_max_rewrites),
        int(settings.graph.multihop_max_iterations),
        0,
    )


def budget_available(*, reserve: int = _RESERVED_CALLS) -> bool:
    """Is there room in the ledger for another loop iteration *and* the epilogue?

    Reads the request-scoped ledger through the contextvar rather than taking it as an
    argument: it is installed by :func:`~ragorc.core.telemetry.new_request_context` and
    threading it through every node signature would be a second, redundant channel that
    can disagree with the first.

    Returns ``True`` when no ceiling is configured. An unbounded run is a deliberate
    configuration, and inventing a limit here would silently override it.
    """
    ledger = current_ledger()
    if ledger is None or ledger.max_calls is None:
        return True
    remaining = ledger.max_calls - ledger.total.calls
    if remaining <= reserve:
        log.info("agentic_budget_reserved", remaining=remaining, reserve=reserve)
        return False
    return True


def decide_after_grade(state: RAGState, *, settings: Settings) -> str:
    """The CRAG branch, with the shared retrieval budget and the ledger both consulted.

    Identical in shape to :mod:`ragorc.pipeline.graphs.crag`, with one addition: a
    corrective retry is also declined when the ledger is nearly spent. In the dedicated
    CRAG graph the loop is the only spender and the counter is enough; here it competes
    with two other loops for one budget, and the first loop to notice should be the one
    that yields.
    """
    grade = state.get("grade")
    used = int(state.get("retrieve_iterations") or 0)

    if grade is GradeLabel.INCORRECT:
        if used >= retrieval_budget(settings) or not budget_available():
            log.info("agentic_correction_exhausted", iterations=used)
            return _GATE
        return _REWRITE
    if grade is GradeLabel.AMBIGUOUS:
        return _WEB
    return _REFINE


def _decide_after_web(state: RAGState) -> str:
    """AMBIGUOUS completes the corpus evidence; INCORRECT goes back for a retry."""
    return _GRADE if state.get("grade") is GradeLabel.INCORRECT else _GATE


def _decide_gate(state: RAGState, *, settings: Settings) -> str:
    """Is another hop possible? Budget first, so no verdict is bought that cannot be used."""
    done = int(state.get("retrieve_iterations") or 0)
    if done >= retrieval_budget(settings) or not budget_available():
        return _COLLECT
    return _CHECK


def _decide_after_check(state: RAGState, *, settings: Settings) -> str:
    if state.get("sufficient") and settings.graph.multihop_stop_on_sufficient:
        return _COLLECT
    if not (state.get("follow_up") or "").strip():
        # "Not sufficient" with nothing to ask for is a dead end, not licence to guess.
        return _COLLECT
    return _HOP


def decide_after_judge(state: RAGState, *, settings: Settings) -> str:
    """Self-RAG's verdict, with the ledger able to end the loop early.

    When the budget is spent and the answer failed its grades, the branch is
    ``abstain`` rather than ``accept``: shipping an answer that was graded and found
    wanting, because there was no money left to improve it, is exactly the outcome the
    grading exists to prevent.

    As in :mod:`ragorc.pipeline.graphs.self_rag`, an abstention the generator produced is
    a failed attempt rather than the final word — otherwise the generator's own
    groundedness gate would pre-empt the retry loop entirely.
    """
    answer = state.get("answer")
    label = verdict(state)
    declined = answer is not None and answer.abstained
    if not declined and label == "accepted":
        return _ACCEPT

    attempts = int(state.get("generate_iterations") or 0)
    if attempts > max(int(settings.generation.self_rag_max_retries), 0):
        return _ABSTAIN
    if not budget_available():
        log.info("agentic_retry_declined", attempts=attempts, verdict=label, reason="budget")
        return _ABSTAIN
    return _RETRY


def build(
    nodes: PipelineNodes, *, settings: Settings | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the agentic graph.

    The multi-store fan-out is *not* unrolled into graph nodes here, unlike in
    :mod:`ragorc.pipeline.graphs.adaptive`. CRAG owns the first-stage retrieval — it has
    to, because grading only means something applied to the wide candidate set it
    fetched itself — so a second graph-level fan-out would retrieve everything twice.
    The concurrency is not lost: :class:`~ragorc.retrieve.multi_store.MultiStoreRetriever`
    fans out under ``bounded_gather`` with a per-store deadline and a per-store circuit
    breaker, which is the same parallelism one superstep down.
    """
    resolved = settings or nodes.settings

    async def gate(state: RAGState) -> dict[str, Any]:
        """Join point between the CRAG branches and the evidence loop.

        A no-op barrier. It exists because three different CRAG outcomes need to
        converge before the hop decision is made, and LangGraph expresses "several
        edges in, one decision out" with a node.
        """
        del state
        return {}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)
    graph.add_node("validate", nodes.validate)
    graph.add_node("translate", nodes.translate)
    graph.add_node("construct", nodes.construct)
    graph.add_node("route", nodes.route)
    graph.add_node(_GRADE, nodes.grade)
    graph.add_node(_REFINE, nodes.compress)
    graph.add_node(_WEB, nodes.web_search)
    graph.add_node(_REWRITE, nodes.rewrite)
    graph.add_node(_GATE, gate)
    graph.add_node(_CHECK, nodes.check_sufficiency)
    graph.add_node(_HOP, nodes.hop)
    graph.add_node(_COLLECT, nodes.fuse)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("compress", nodes.compress)
    graph.add_node(_GENERATE, nodes.generate)
    graph.add_node("verify_groundedness", nodes.verify_groundedness)
    graph.add_node("verify_utility", nodes.verify_utility)
    graph.add_node("judge", judge)
    graph.add_node(_RETRY, nodes.rewrite)
    graph.add_node(_ABSTAIN, nodes.abstain)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "translate")
    graph.add_edge("translate", "construct")
    graph.add_edge("construct", "route")
    graph.add_edge("route", _GRADE)

    graph.add_conditional_edges(
        _GRADE,
        lambda state: decide_after_grade(state, settings=resolved),
        [_REFINE, _WEB, _REWRITE, _GATE],
    )
    graph.add_edge(_REWRITE, _WEB)
    graph.add_conditional_edges(_WEB, _decide_after_web, [_GRADE, _GATE])
    graph.add_edge(_REFINE, _GATE)

    graph.add_conditional_edges(
        _GATE, lambda state: _decide_gate(state, settings=resolved), [_CHECK, _COLLECT]
    )
    graph.add_conditional_edges(
        _CHECK, lambda state: _decide_after_check(state, settings=resolved), [_HOP, _COLLECT]
    )
    graph.add_edge(_HOP, _GATE)

    graph.add_edge(_COLLECT, "rerank")
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", _GENERATE)
    graph.add_edge(_GENERATE, "verify_groundedness")
    graph.add_edge(_GENERATE, "verify_utility")
    graph.add_edge("verify_groundedness", "judge")
    graph.add_edge("verify_utility", "judge")
    graph.add_conditional_edges(
        "judge",
        lambda state: decide_after_judge(state, settings=resolved),
        {_ACCEPT: END, _RETRY: _RETRY, _ABSTAIN: _ABSTAIN},
    )
    # An answer-side retry re-enters *graded* retrieval, not generation: regenerating
    # from the context that produced the rejected answer grades the same overreach
    # again at full price.
    graph.add_edge(_RETRY, _GRADE)
    graph.add_edge(_ABSTAIN, END)
    return graph.compile(name=NAME)


def recursion_limit(settings: Settings) -> int:
    """Derive the safety net from three nested loops.

    Per answer attempt: the CRAG branch is up to three supersteps, the evidence loop two
    per hop, and the epilogue (collect, rerank, compress, generate, verify, judge,
    rewrite) is seven. Multiplied by the answer-side attempts and offset by the
    four-superstep prologue.

    The number comes out large — around 60 with default settings — and that is correct
    rather than alarming: it is a *net*, not a target, and the ledger plus
    :func:`budget_available` are what actually stop this graph. A recursion limit tuned
    tight enough to be the primary bound would convert legitimate deep runs into
    ``GraphRecursionError`` instead of answers.
    """
    hops = retrieval_budget(settings)
    attempts = max(int(settings.generation.self_rag_max_retries), 0) + 1
    return 6 + attempts * (10 + 2 * hops)
