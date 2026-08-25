"""The seven pipelines, and the registry the facade selects from.

Each module here is one compiled LangGraph state machine over the shared nodes in
:mod:`ragorc.pipeline.nodes`, and each exposes the same three names — ``NAME``,
``build`` and ``recursion_limit`` — so :class:`~ragorc.pipeline.builder.RAGPipeline`
can choose among them by string without a special case per graph.

======================  ===================================================
 name                    what it adds
======================  ===================================================
``naive``               nothing. The control in benchmarks.
``adaptive``            routing + a parallel fan-out over the routed stores
``crag``                document grading, refinement, web fallback, retries
``self_rag``            answer grading (groundedness + utility), retries
``graphrag``            local / global / DRIFT search over the graph
``multihop``            bridge paths, and a bounded retrieve-reason loop
``agentic``             all of the above, with the model steering
======================  ===================================================

Why ``recursion_limit`` is per-graph and derived
-----------------------------------------------
LangGraph's recursion limit is a *safety net* that raises ``GraphRecursionError``;
every cyclic graph here also carries an explicit iteration counter in its state,
which is the graceful exit. The net has to sit strictly above the graceful bound or
it becomes the primary one, and a deep-but-legitimate run then raises instead of
answering. Since the graceful bound is computed from settings, so is the net — a
hard-coded 25 is correct until somebody raises a retry budget, and then it is silently
wrong.

Why the modules are imported eagerly here
-----------------------------------------
Unlike :mod:`ragorc.retrieve`, this package resolves nothing lazily. These modules
import only LangGraph and the shared node module — no database driver, no ONNX
session, no optional extra — so there is nothing to defer, and a registry that can
only be populated by importing modules is exactly the thing that should not be lazy:
``"crag"`` failing to resolve because nobody touched the module is indistinguishable
from a typo.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ragorc.core.errors import ConfigError
from ragorc.core.settings import Settings
from ragorc.pipeline.graphs import (
    adaptive,
    agentic,
    crag,
    graphrag,
    multihop,
    naive,
    self_rag,
)
from ragorc.pipeline.nodes import PipelineNodes

__all__ = [
    "GRAPHS",
    "GraphSpec",
    "adaptive",
    "agentic",
    "build_graph",
    "crag",
    "graph_names",
    "graphrag",
    "multihop",
    "naive",
    "self_rag",
]


@dataclass(frozen=True, slots=True)
class GraphSpec:
    """One pipeline: how to build it, how deep it may go, and what it is for."""

    name: str
    build: Callable[..., Any]
    recursion_limit: Callable[[Settings], int]
    summary: str

    def compile(
        self,
        nodes: PipelineNodes,
        *,
        settings: Settings | None = None,
        interrupt_before: Sequence[str] | None = None,
    ) -> Any:
        """Compile this graph against a component bundle.

        ``interrupt_before`` names nodes the run must stop *before* reaching.
        Streaming uses it to run a graph's retrieval side and hand the state back
        without entering generation — see
        :meth:`~ragorc.pipeline.builder.RAGPipeline.stream`. LangGraph accepts it
        with no checkpointer as long as the run is never resumed, which is exactly
        the usage here.
        """
        return self.build(nodes, settings=settings, interrupt_before=interrupt_before)


GRAPHS: dict[str, GraphSpec] = {
    naive.NAME: GraphSpec(
        naive.NAME, naive.build, naive.recursion_limit, "retrieve then generate; the control"
    ),
    adaptive.NAME: GraphSpec(
        adaptive.NAME,
        adaptive.build,
        adaptive.recursion_limit,
        "route, fan out over the chosen stores in parallel, fuse, rerank",
    ),
    crag.NAME: GraphSpec(
        crag.NAME,
        crag.build,
        crag.recursion_limit,
        "grade the documents; refine, add web results, or rewrite and retry",
    ),
    self_rag.NAME: GraphSpec(
        self_rag.NAME,
        self_rag.build,
        self_rag.recursion_limit,
        "grade the answer for groundedness and utility; retry or abstain",
    ),
    graphrag.NAME: GraphSpec(
        graphrag.NAME,
        graphrag.build,
        graphrag.recursion_limit,
        "local, global map-reduce, or DRIFT search, chosen by question shape",
    ),
    multihop.NAME: GraphSpec(
        multihop.NAME,
        multihop.build,
        multihop.recursion_limit,
        "bridge-entity paths, or a bounded retrieve-reason loop",
    ),
    agentic.NAME: GraphSpec(
        agentic.NAME,
        agentic.build,
        agentic.recursion_limit,
        "every mechanism, with the model choosing at each decision point",
    ),
}


def graph_names() -> list[str]:
    """The selectable pipeline names, ordered cheapest-first."""
    return list(GRAPHS)


def build_graph(
    name: str,
    nodes: PipelineNodes,
    *,
    settings: Settings | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> Any:
    """Compile one graph by name.

    Raises :class:`~ragorc.core.errors.ConfigError` listing the valid names, because
    a mistyped pipeline name is a configuration error and should read like one rather
    than like a ``KeyError`` from inside the facade.
    """
    spec = GRAPHS.get(name)
    if spec is None:
        raise ConfigError(
            f"unknown pipeline {name!r}", available=graph_names(), hint="pipeline='auto' picks one"
        )
    return spec.compile(nodes, settings=settings, interrupt_before=interrupt_before)
