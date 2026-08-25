"""The adaptive pipeline: route the question, then query only what it needs.

    validate -> translate -> route -> [ fan out to the routed stores ] -> fuse
             -> rerank -> generate

This is the default for a multi-store deployment, and the two things it adds over
:mod:`~ragorc.pipeline.graphs.naive` are both about *not* doing work.

Routing is the cheap decision that saves the expensive ones
----------------------------------------------------------
Vector, relational and graph are not three interchangeable indexes — they answer
three different kinds of question. "How many orders shipped late in March" is a SQL
question and a vector search over prose will answer it with something that merely
mentions shipping delays. One cheap classification call decides which backends the
question needs, and the ones it does not need are never touched: no generated SQL,
no graph traversal, no latency, no bill.

The ``none`` route is part of that saving and is easy to leave out. "Hi", "thanks",
"what can you do" need no corpus, and running retrieval for them costs an embedding
call, a search and a context window to answer a greeting from documents that have
nothing to do with it. It is handled here by branching straight to generation, which
answers with the ``answer_no_context`` prompt and reports ``grounded=False`` —
because a parametric answer may be right but is not *supported*, and saying
otherwise would make the groundedness score meaningless.

The fan-out is a real parallel superstep
----------------------------------------
The conditional edge after ``route`` returns a **list** of node names, one per
selected store, and LangGraph runs them all in the next superstep. That is not a
micro-optimization: serial fan-out makes a query's latency the *sum* of its
backends — a 120 ms Qdrant search plus a 400 ms generated SQL statement plus a
250 ms graph traversal is 770 ms of waiting — where concurrent it is 400 ms, the
slowest one, and a fourth store costs nothing unless it is slower than the current
worst. That difference is what makes a three-store architecture a feature instead of
a tax.

It is also why the fields those nodes write carry reducers (see
:mod:`ragorc.pipeline.state`). Two nodes writing one reducer-less channel in one
superstep raises ``InvalidUpdateError``, and this graph is where a reader first
meets that rule.

Every routed store gets a node whether or not it is wired up
------------------------------------------------------------
All four store nodes are added to the graph unconditionally, and a node whose
retriever is missing records ``"no retriever configured for this store"`` rather
than being skipped. A route that asks for the graph on a deployment with no graph
retriever is a configuration error, and answering from two stores instead of three
hides it indefinitely — the query still succeeds, slightly worse, forever.

Fusion then reranking, in that order
------------------------------------
``fuse`` merges the legs by rank (RRF) because the lists have no common scale, and
denoises *after* merging — fusion is what creates the duplicates and what gives the
relative score cutoff one scale to be relative to. ``rerank`` then buys precision
over the wide candidate window that fusion produced: the retrieval stages fetched
``fetch_k`` because recall cannot be recovered later, and the cross-encoder chooses
``top_k`` out of that rather than reordering ten documents.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.models import DataStore
from ragorc.core.settings import Settings
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

log = structlog.get_logger(__name__)

__all__ = ["NAME", "STORE_NODES", "build", "recursion_limit", "select_stores"]

NAME = "adaptive"

STORE_NODES: dict[DataStore, str] = {
    DataStore.VECTOR: "retrieve_vector",
    DataStore.RELATIONAL: "retrieve_relational",
    DataStore.GRAPH: "retrieve_graph",
    DataStore.WEB: "retrieve_web",
}
"""Datastore to node name. The conditional edge returns these strings, so this
mapping is the whole translation from a :class:`RouteDecision` to a fan-out."""

_DIRECT = "generate"
"""Where the ``none`` route goes: no retrieval, straight to synthesis."""


def select_stores(state: RAGState) -> list[str]:
    """Turn the route into the set of nodes to run concurrently.

    Returning a list is what makes this a fan-out; returning one name would make it
    an ordinary branch. Three cases are folded in here rather than pushed into the
    nodes:

    * **No route, or a route with no stores.** Defaults to the vector store, the one
      backend every deployment has. Fanning out to everything on an absent decision
      would make a failed router *more* expensive than a working one.
    * **``none`` alone.** Skips retrieval entirely.
    * **``none`` alongside real stores.** The real stores win. A router that says
      "maybe nothing, maybe the graph" is expressing uncertainty, and the safe
      reading of uncertainty is to look before answering.
    """
    route = state.get("route")
    stores = tuple(route.stores) if route is not None else ()
    real = [s for s in stores if s is not DataStore.NONE and s in STORE_NODES]

    if not real:
        if DataStore.NONE in stores:
            log.info("adaptive_route_none", reasoning=route.reasoning if route else None)
            return [_DIRECT]
        log.info("adaptive_route_default", stores=[s.value for s in stores])
        return [STORE_NODES[DataStore.VECTOR]]

    # dict.fromkeys, not set(): the fan-out order does not affect the result, but a
    # stable order makes the trace and the mermaid diagram reproducible.
    return [STORE_NODES[store] for store in dict.fromkeys(real)]


def build(
    nodes: PipelineNodes,
    *,
    settings: Settings | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the adaptive graph."""
    del settings  # topology is static; only the recursion limit reads settings
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)

    graph.add_node("validate", nodes.validate)
    graph.add_node("translate", nodes.translate)
    graph.add_node("route", nodes.route)
    for store, node_name in STORE_NODES.items():
        graph.add_node(node_name, nodes.store_node(store))
    graph.add_node("fuse", nodes.fuse)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "translate")
    graph.add_edge("translate", "route")
    graph.add_conditional_edges("route", select_stores, [*STORE_NODES.values(), _DIRECT])
    for node_name in STORE_NODES.values():
        # Every leg converges on ``fuse``. LangGraph waits for all of them before
        # running it, which is exactly the barrier fusion needs: merging a partial
        # set of legs would rank against a distribution that is still arriving.
        graph.add_edge(node_name, "fuse")
    graph.add_edge("fuse", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    return graph.compile(name=NAME, interrupt_before=list(interrupt_before or ()))


def recursion_limit(settings: Settings) -> int:
    """Supersteps this graph can ever need.

    Acyclic: validate, translate, route, one fan-out step, fuse, rerank, generate.
    The margin covers LangGraph counting the start and end sentinels.
    """
    del settings
    return 12
