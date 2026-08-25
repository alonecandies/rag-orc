"""Multi-hop: retrieve, reason about what is still missing, retrieve again.

    validate -> translate -> bridge -+-> paths found -> generate
                                     |
                                     +-> no paths -> retrieve -+-> check -+-> hop -+
                                                               |          |        |
                                                               |          +-> collect
                                                               +-> collect (budget spent)
                                                     collect -> rerank -> generate
                                                                     ^-- hop loops back

Two question shapes, two mechanisms
-----------------------------------
**"How is A related to B"** is the question class no amount of vector search answers.
The join between A and B is not written in any chunk — it is distributed across the
documents that asserted each edge along the path — so the only retrievable form of the
answer is *the path itself*. That is what the ``bridge`` node returns: the verbalized
paths plus the chunks that asserted each edge, so every hop is citable.

**"What is X's Y"** where Y is only reachable through an intermediate fact needs
iteration (IRCoT): retrieve, ask what is still missing, search for that, repeat. The
loop is what crosses a join the corpus never states in one sentence.

The branch between them needs no model call, because the shapes are distinguishable
from the entity index: a question naming two or more graph entities is asking about
their connection. The bridge node runs first and its emptiness *is* the signal — one
entity, or none, means iteration. When a bridge question has no path (the entities
really are unconnected in this graph, or the connection is longer than
``multihop_max_path_length``) the fall-through to iteration is not a consolation prize:
the corpus may state the relationship in prose that was never extracted into an edge,
and iteration is how that gets found.

Bridge results do not get reranked
----------------------------------
They go straight to generation. A cross-encoder over verbalized paths would re-score
them for lexical similarity to the question, which is not what makes a path the answer,
and trimming to ``top_k`` would cut the connections that *are* the answer. The
iterative branch does rerank, because there the results are ordinary passages.

The sufficiency check is the point, not an optimization
------------------------------------------------------
Most questions need one hop. Each additional hop costs a full retrieval plus a model
call, and the gains flatten after about three — so asking "does the evidence already
answer this" costs one cheap call and usually saves two expensive rounds. Hence
``multihop_stop_on_sufficient``, and hence the check being a node with an edge rather
than an ``if`` inside a retriever.

Two things the loop refuses to do, both learned the hard way:

* It does not pay for a judgement it cannot act on. The conditional edge after each
  retrieval checks the iteration budget *first*, so the final hop goes straight to
  collection instead of buying a sufficiency verdict that no branch can use.
* It does not continue on "not sufficient" with no follow-up. A model that cannot say
  what is missing has given a dead end, not an instruction to guess, and re-running the
  same query would return the same rows at the same price.

Bounds, both of them
--------------------
``graph.multihop_max_iterations`` bounds the hops gracefully — the loop leaves through
``collect`` with everything it gathered. The derived recursion limit is the safety net
that raises. See :mod:`ragorc.pipeline.graphs.crag` for why both.

Deduplication across hops is not optional and is handled by ``collect``: consecutive
queries on one topic overlap heavily, and the same passage arriving three times would
occupy three context slots and assert one fact three times — which makes the model more
confident in it rather than better informed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.settings import Settings
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

log = structlog.get_logger(__name__)

__all__ = [
    "NAME",
    "build",
    "decide_after_bridge",
    "decide_after_check",
    "decide_budget",
    "recursion_limit",
]

NAME = "multihop"

_CHECK = "check_sufficiency"
_HOP = "hop"
_COLLECT = "collect"
_RETRIEVE = "retrieve"
_GENERATE = "generate"


def _max_iterations(settings: Settings) -> int:
    """At least one: a multi-hop graph that takes zero retrievals is not a pipeline."""
    return max(1, int(settings.graph.multihop_max_iterations))


def _iterations_done(state: RAGState) -> int:
    """Retrievals performed so far.

    The first retrieval is not counted by any node — only ``hop`` increments — so it is
    added back here. Counting it inside ``retrieve`` instead would make that node
    unusable by every other graph in this package.
    """
    return 1 + int(state.get("retrieve_iterations") or 0)


def decide_after_bridge(state: RAGState) -> str:
    """Path search answered it, or iteration takes over.

    ``search_mode == "bridge"`` is set by the bridge node only when it actually
    returned paths, so an empty graph, a missing bridge retriever and a single-entity
    question all land on the iterative branch — which is the correct branch for all
    three, not a fallback.
    """
    if state.get("search_mode") == "bridge":
        return _GENERATE
    return _RETRIEVE


def decide_budget(state: RAGState, *, settings: Settings) -> str:
    """Is another hop even possible? Asked before paying for the verdict.

    Checking the budget before the sufficiency call is what stops the loop buying a
    judgement it cannot act on. On the last permitted iteration the answer is already
    fixed — collect and answer — so the call would be pure cost.
    """
    done = _iterations_done(state)
    limit = _max_iterations(settings)
    if done >= limit:
        log.info("multihop_budget_spent", iterations=done, limit=limit)
        return _COLLECT
    return _CHECK


def decide_after_check(state: RAGState) -> str:
    """Hop again, or collect what we have.

    Three ways to stop, and they are different facts worth distinguishing in the log:
    the evidence is sufficient (the good exit), the model had no follow-up (a dead
    end), or early exit is disabled and only the budget bounds the loop.
    """
    if state.get("sufficient") and _stop_on_sufficient(state):
        log.info("multihop_stop", reason="sufficient", iterations=_iterations_done(state))
        return _COLLECT
    if not (state.get("follow_up") or "").strip():
        log.info("multihop_stop", reason="no_follow_up", iterations=_iterations_done(state))
        return _COLLECT
    return _HOP


def _stop_on_sufficient(state: RAGState) -> bool:
    """``multihop_stop_on_sufficient``, read from the state's own settings snapshot.

    Kept as a helper so :func:`decide_after_check` stays a pure function of the state
    for tests; the flag is injected by the closure the graph builds.
    """
    return bool(state.get("metadata", {}).get("stop_on_sufficient", True))


def build(
    nodes: PipelineNodes,
    *,
    settings: Settings | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the multi-hop graph."""
    resolved = settings or nodes.settings
    stop_early = bool(resolved.graph.multihop_stop_on_sufficient)

    async def seed(state: RAGState) -> dict[str, Any]:
        """Publish the early-exit flag into the state.

        A one-line node rather than a closure over the predicate, so
        :func:`decide_after_check` remains testable against a plain dict and the
        configuration that governed a run is visible in the final state instead of
        being captured invisibly in a lambda.
        """
        return {"metadata": {**(state.get("metadata") or {}), "stop_on_sufficient": stop_early}}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)
    graph.add_node("validate", nodes.validate)
    graph.add_node("translate", nodes.translate)
    graph.add_node("seed", seed)
    graph.add_node("bridge", nodes.bridge)
    graph.add_node(_RETRIEVE, nodes.retrieve)
    graph.add_node(_CHECK, nodes.check_sufficiency)
    graph.add_node(_HOP, nodes.hop)
    graph.add_node(_COLLECT, nodes.fuse)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node(_GENERATE, nodes.generate)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "translate")
    graph.add_edge("translate", "seed")
    graph.add_edge("seed", "bridge")
    graph.add_conditional_edges("bridge", decide_after_bridge, [_RETRIEVE, _GENERATE])

    def budget(state: RAGState) -> str:
        return decide_budget(state, settings=resolved)

    # The same predicate on both retrieval nodes: "may I hop again" does not depend on
    # which retrieval just ran, and duplicating it would let the two drift apart.
    graph.add_conditional_edges(_RETRIEVE, budget, [_CHECK, _COLLECT])
    graph.add_conditional_edges(_HOP, budget, [_CHECK, _COLLECT])
    graph.add_conditional_edges(_CHECK, decide_after_check, [_HOP, _COLLECT])

    graph.add_edge(_COLLECT, "rerank")
    graph.add_edge("rerank", _GENERATE)
    graph.add_edge(_GENERATE, END)
    return graph.compile(name=NAME, interrupt_before=list(interrupt_before or ()))


def recursion_limit(settings: Settings) -> int:
    """Derive the safety net from the loop's shape.

    Four supersteps of prologue (validate, translate, seed, bridge), two per iteration
    (a retrieval and its sufficiency check), three of epilogue (collect, rerank,
    generate). Scaled by ``multihop_max_iterations`` so raising the hop budget cannot
    turn the net into the primary bound.
    """
    return 9 + 2 * _max_iterations(settings)
