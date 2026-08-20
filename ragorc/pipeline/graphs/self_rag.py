"""Self-RAG: let the answer's own quality decide whether to retrieve again.

    validate -> translate -> route -> retrieve -> rerank -> generate
                                                              |
                                            +-----------------+-----------------+
                                            v  (one superstep)                  v
                                   verify_groundedness                  verify_utility
                                            +-----------------+-----------------+
                                                              v
                                                            judge -+-> accept  -> END
                                                                   +-> rewrite -> retrieve
                                                                   +-> abstain -> END

Where CRAG grades the *documents*, this grades the *answer*. Both loops exist because
they catch different failures: retrieval can succeed and synthesis still overreach,
and no amount of document grading notices a model asserting a causal claim its
evidence never made.

Why the two grades run concurrently
----------------------------------
Groundedness (Self-RAG's ISSUP) and utility (ISUSE) are independent judgements about
the same text — neither reads the other's output — so serializing them doubles the
latency of the verification step for nothing. Two static edges out of ``generate``
put them in one superstep, and their results land in different state keys, so no
reducer is needed: they are concurrent *writers* but not to the same channel.

That is also why this is a graph rather than a call to
:meth:`~ragorc.generate.self_rag.SelfRAG.run`. The grading itself is not
reimplemented — groundedness goes through the very
:class:`~ragorc.generate.groundedness.GroundednessChecker` that ``SelfRAG`` owns, and
utility uses the same prompt, schema and model tier — but a single ``run`` call
cannot express "these two steps are one superstep, and the branch after them is an
edge". ``SelfRAG.run`` remains the right entry point for a caller who wants the loop
*inside* a function; this module is the same loop with the control flow visible, and
therefore checkpointable, streamable and printable as a diagram.

Why the two failures get different rewrites
-------------------------------------------
An **ungrounded** answer means the model outran its evidence; retrying against the
same context will usually reproduce the same overreach, so the query is rewritten to
find documents that state the facts directly. An answer that is grounded but **not
useful** means the evidence was about the wrong thing, which also calls for
re-retrieval, but aimed at coverage rather than support. Treating both as "try again"
wastes the retry. The rewrite node sees the previous answer and the evidence it was
built from, which is what makes the distinction actionable.

Why the loop must end in an abstention
--------------------------------------
Returning the least-bad ungrounded answer after three graded attempts defeats the
entire mechanism: the point of grading is to be *able* to decline. So the exhausted
branch goes to ``abstain``, which keeps the rejected text under
``metadata["rejected_answer"]`` for diagnosis and returns the configured refusal.

Bounds, both of them
--------------------
``generation.self_rag_max_retries`` is the graceful bound, counted in
``generate_iterations`` (incremented by the generate node, so it counts *attempts*
rather than trips through any one edge). The recursion limit derived below is the
safety net that raises. See :mod:`ragorc.pipeline.graphs.crag` for why a system with
only one of the two is either unprotected or unable to answer.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.settings import Settings
from ragorc.core.telemetry import trace_step
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

log = structlog.get_logger(__name__)

__all__ = ["NAME", "build", "decide_after_judge", "recursion_limit", "verdict"]

NAME = "self_rag"

_ACCEPT = "accept"
_RETRY = "rewrite"
_ABSTAIN = "abstain"
_RETRIEVE = "retrieve"
_JUDGE = "judge"


def _budget(settings: Settings) -> int:
    return max(int(settings.generation.self_rag_max_retries), 0)


def verdict(state: RAGState) -> str:
    """Collapse the two grades into one label: accepted, ungrounded, not_useful.

    Order matters. Groundedness is checked first because it is the more serious
    failure — an unsupported answer is wrong in a way a merely off-target answer is
    not — and because the two rewrites it implies are different, so the label has to
    say *which* failure happened rather than just that one did.
    """
    if not bool(state.get("grounded", True)):
        return "ungrounded"
    if not bool(state.get("useful", True)):
        return "not_useful"
    return "accepted"


async def judge(state: RAGState) -> dict[str, Any]:
    """Join the two graders and record the verdict.

    A barrier node: LangGraph will not run it until both verification nodes have
    finished, which is what turns two concurrent grades into one decision point. It
    is also the only place the per-iteration verdict is written to the trace, so a
    Self-RAG run reads as a sequence of graded attempts rather than as a pile of
    grader calls.
    """
    label = verdict(state)
    attempt = int(state.get("generate_iterations") or 0)
    trace_step(
        "self_rag_iteration",
        iteration=attempt,
        verdict=label,
        grounded=bool(state.get("grounded", True)),
        groundedness=round(float(state.get("groundedness", 0.0)), 3),
        useful=bool(state.get("useful", True)),
        utility=round(float(state.get("utility", 0.0)), 3),
    )
    answer = state.get("answer")
    if answer is None:
        return {}
    # Stamped on the answer so an accepted result carries the evidence that it was
    # graded, not merely that it was produced.
    answer.metadata = {
        **answer.metadata,
        "self_rag": {
            "attempts": attempt,
            "verdict": label,
            "groundedness": round(float(state.get("groundedness", 0.0)), 3),
            "utility": round(float(state.get("utility", 0.0)), 3),
        },
    }
    return {"answer": answer}


def decide_after_judge(state: RAGState, *, settings: Settings) -> str:
    """Accept, retry, or abstain.

    An answer the generator already declined counts as a **failed attempt**, not as a
    final answer, and this is the subtle part of the whole loop. With
    ``generation.check_groundedness`` on — the default — the generator runs its own
    groundedness gate and abstains on an unsupported answer *before* this graph's
    verification nodes ever see it. Treating that refusal as terminal would make the
    Self-RAG loop unreachable in the default configuration: the mechanism that exists to
    retry an ungrounded answer would never fire, because the layer below had already
    given up. The two are the same judgement made at different levels, so the abstention
    is read as the failure signal it is, and the retry happens.

    Once the retries are exhausted the abstention stands — and goes through the
    ``abstain`` node anyway, so the attempt count and the gate are recorded the same way
    on every path that ends in a refusal.
    """
    label = verdict(state)
    answer = state.get("answer")
    declined = answer is not None and answer.abstained
    if not declined and label == "accepted":
        return _ACCEPT

    attempts = int(state.get("generate_iterations") or 0)
    if attempts > _budget(settings):
        log.info("self_rag_exhausted", attempts=attempts, verdict=label, declined=declined)
        return _ABSTAIN
    log.info("self_rag_retry", attempt=attempts, verdict=label, declined=declined)
    return _RETRY


def build(
    nodes: PipelineNodes, *, settings: Settings | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the Self-RAG graph."""
    resolved = settings or nodes.settings
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)

    graph.add_node("validate", nodes.validate)
    graph.add_node("translate", nodes.translate)
    graph.add_node("route", nodes.route)
    graph.add_node(_RETRIEVE, nodes.retrieve)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("generate", nodes.generate)
    graph.add_node("verify_groundedness", nodes.verify_groundedness)
    graph.add_node("verify_utility", nodes.verify_utility)
    graph.add_node(_JUDGE, judge)
    graph.add_node(_RETRY, nodes.rewrite)
    graph.add_node(_ABSTAIN, nodes.abstain)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "translate")
    graph.add_edge("translate", "route")
    graph.add_edge("route", _RETRIEVE)
    graph.add_edge(_RETRIEVE, "rerank")
    graph.add_edge("rerank", "generate")
    # Two static out-edges, not a conditional: both grades always run, and LangGraph
    # schedules them in the same superstep.
    graph.add_edge("generate", "verify_groundedness")
    graph.add_edge("generate", "verify_utility")
    graph.add_edge("verify_groundedness", _JUDGE)
    graph.add_edge("verify_utility", _JUDGE)
    graph.add_conditional_edges(
        _JUDGE,
        lambda state: decide_after_judge(state, settings=resolved),
        {_ACCEPT: END, _RETRY: _RETRY, _ABSTAIN: _ABSTAIN},
    )
    # The retry closes the cycle: a rewritten query re-enters retrieval, not
    # generation. Regenerating from the same context is what produced the rejected
    # answer, so the loop would grade the same overreach again at full price.
    graph.add_edge(_RETRY, _RETRIEVE)
    graph.add_edge(_ABSTAIN, END)
    return graph.compile(name=NAME)


def recursion_limit(settings: Settings) -> int:
    """Derive the safety net from the loop's shape.

    One attempt is six supersteps: retrieve, rerank, generate, the parallel
    verification, judge, and (on a retry) rewrite. Three for the prologue, one for the
    terminal abstention. Scaled by the configured retry budget so raising the budget
    cannot silently turn the net into the primary bound — which would replace every
    deep abstention with a ``GraphRecursionError``.
    """
    return 6 + 6 * (_budget(settings) + 1)
