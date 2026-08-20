"""Orchestration: the LangGraph state machines that compose everything below them.

Four things live here and the layering between them is strict:

===========================  =================================================
:mod:`~ragorc.pipeline.state`     the shared state object and its reducers
:mod:`~ragorc.pipeline.nodes`     reusable async nodes, one per stage
:mod:`~ragorc.pipeline.graphs`    seven compiled graphs wiring those nodes
:mod:`~ragorc.pipeline.builder`   the facade that builds components and runs a graph
===========================  =================================================

``state`` depends on ``core`` only, ``nodes`` on the stage packages, ``graphs`` on
``nodes``, and ``builder`` on all of them. Nothing points back up.

This package is imported eagerly, unlike :mod:`ragorc.retrieve`: it pulls LangGraph and
the vector store, which any pipeline needs anyway. The lazy boundary is one level up —
:mod:`ragorc` resolves ``RAGPipeline`` through ``__getattr__`` so ``import ragorc`` for a
``Chunk`` or a ``Settings`` costs milliseconds instead of hundreds of them.
"""

from ragorc.pipeline.builder import RAGPipeline, build_pipeline
from ragorc.pipeline.graphs import GRAPHS, GraphSpec, build_graph, graph_names
from ragorc.pipeline.nodes import Node, PipelineNodes, resolve_prompt_name
from ragorc.pipeline.state import (
    RAGState,
    evidence,
    failure,
    initial_state,
    merge_store_lists,
    total_usage,
)

__all__ = [
    "GRAPHS",
    "GraphSpec",
    "Node",
    "PipelineNodes",
    "RAGPipeline",
    "RAGState",
    "build_graph",
    "build_pipeline",
    "evidence",
    "failure",
    "graph_names",
    "initial_state",
    "merge_store_lists",
    "resolve_prompt_name",
    "total_usage",
]
