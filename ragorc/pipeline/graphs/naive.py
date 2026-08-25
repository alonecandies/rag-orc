"""The naive pipeline: retrieve, then generate.

This graph exists to be the **control**. Every other graph in this package claims
to be better than "just retrieve and answer", and that claim is only measurable
against something. So this is the baseline the evaluation harness compares to: one
retrieval, one synthesis, no translation, no routing, no grading, no reranking, no
loops.

It is also the right pipeline for real deployments more often than the feature list
suggests. On a small, homogeneous corpus answering direct questions, the adaptive
graph spends four extra model calls to arrive at the same evidence. Reach for the
loops when the measurements say the baseline is losing, not before — an
architecture diagram is not a reason to pay for one.

Why ``validate`` is here anyway
-------------------------------
It is not a retrieval technique, so it does not compromise the control: it makes no
model call and changes no ranking. It is here because it is what *constructs* the
:class:`~ragorc.core.models.Query` every retriever consumes, and because tenant
scoping is not something a benchmark gets to switch off — an experiment that reads
across tenants is not measuring your production system.

What this graph deliberately does not do
----------------------------------------
No reranking, so the generator sees the retriever's own ordering. No abstention
tuning beyond what :class:`~ragorc.generate.answer.AnswerGenerator` always
enforces — the guarantees that live in the generator (citations, groundedness, the
two abstention gates) apply here too, because they are properties of answering
rather than of orchestration. This graph is a *cheaper* baseline, not a less honest
one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.settings import Settings
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

__all__ = ["NAME", "build", "recursion_limit"]

NAME = "naive"


def build(
    nodes: PipelineNodes,
    *,
    settings: Settings | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the baseline graph.

    ``settings`` is accepted and unused so every builder in this package has one
    signature; the facade in :mod:`ragorc.pipeline.builder` selects among them by
    name and should not need a special case for the simple one.
    """
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)
    graph.add_node("validate", nodes.validate)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(name=NAME, interrupt_before=list(interrupt_before or ()))


def recursion_limit(settings: Settings) -> int:
    """Supersteps this graph can ever need.

    Acyclic and three nodes long, so the limit is a constant. It is still stated
    rather than left to LangGraph's default of 25: a limit that is not derived from
    the graph is a limit nobody notices is wrong.
    """
    del settings  # no loop, nothing to scale with
    return 8
