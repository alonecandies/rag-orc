"""Corrective RAG: grade what came back, and act on the grade.

    validate -> translate -> route -> grade -+-> CORRECT   -> refine -> generate
                                             +-> AMBIGUOUS -> web    -> generate
                                             +-> INCORRECT -> rewrite -> web -> grade
                                                              (bounded)

The failure this graph exists to fix
------------------------------------
A retriever always returns its top-k, whether or not the corpus contains the answer.
Similarity is *relative*: on a question the corpus cannot answer, the ten nearest
neighbours still come back with respectable scores, and a generator instructed to
answer from the context duly answers from documents about something adjacent. A
score threshold does not catch it, because the scores are not the problem — they are
perfectly good scores relative to a corpus that has nothing to say.

CRAG (Yan et al., 2024) inserts a decision between retrieval and generation, and
this graph is that decision expressed as edges. The grading, the strip-level
knowledge refinement and the web fallback all live in
:class:`~ragorc.retrieve.crag.CorrectiveRAG`; what the graph adds is the branch —
and, crucially, the **loop**, which a single CRAG pass cannot express.

Why the three labels get three different treatments
---------------------------------------------------
``CORRECT`` (every graded document relevant) goes to refinement and then to the
answer. ``AMBIGUOUS`` (some relevant, some not) adds web results to the corpus
evidence rather than replacing it — half the documents being right is not the same
situation as all of them being right, and treating it as CORRECT is what makes a
half-answered question look answered. ``INCORRECT`` (none relevant) discards, asks
for different words, searches the open web, and tries corpus retrieval again with
the rewritten query.

``None`` — no verdict was obtainable because every grader call failed — is its own
branch and behaves like CORRECT: use the retrieval as it came back. Coercing it to
INCORRECT would route an entire corpus to the web because a provider blipped, and
coercing it to a verdict at all would be claiming a judgement nobody made.

Two independent bounds on the loop, and why one is not enough
------------------------------------------------------------
**The iteration counter** (``retrieve_iterations`` in the state, incremented by the
rewrite node) is the *graceful* exit. When it runs out the conditional edge routes to
``generate``, which produces a real :class:`~ragorc.core.models.Answer` — normally an
abstention, since the evidence was judged irrelevant — with the trace, the citations
and the cost ledger intact. That is the exit a user should ever see.

**LangGraph's recursion limit** is the *safety net*, and it raises
``GraphRecursionError``. It exists to catch the loop the counter cannot: an edge
function that returns the wrong branch, a node that fails before it increments, a
cycle nobody intended. It must never be the normal exit, because raising discards the
partial answer, the trace and the accumulated evidence — everything that makes the
failure diagnosable.

Keeping both is the point. A counter alone trusts the code that computes the exit,
which is exactly the code most likely to be wrong; a recursion limit alone turns
"the corpus could not answer this" into a stack-trace instead of an abstention.

The loop is bounded by ``generation.rrr_max_rewrites`` rather than a CRAG-specific
setting: that value *is* the rewrite budget, this loop is rewrite-driven, and
inventing a second knob for the same quantity guarantees the two eventually disagree.

Where the web results live
--------------------------
In ``web_chunks``, not in ``retrieval``. A corpus retry rebuilds ``retrieval`` from
scratch, and web evidence already paid for must survive that;
:func:`~ragorc.pipeline.state.evidence` unions the two by provenance when the
generator reads them — corpus first, then web, never interleaved by score, because a
cosine similarity and a search engine's rank position are not on one scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.models import GradeLabel
from ragorc.core.settings import Settings
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

log = structlog.get_logger(__name__)

__all__ = ["NAME", "build", "decide_after_grade", "decide_after_web", "recursion_limit"]

NAME = "crag"

_REFINE = "refine"
_WEB = "web_search"
_REWRITE = "rewrite"
_GENERATE = "generate"
_GRADE = "grade"


def _budget(settings: Settings) -> int:
    """How many corrective retries the loop may take."""
    return max(int(settings.generation.rrr_max_rewrites), 0)


def decide_after_grade(state: RAGState, *, settings: Settings) -> str:
    """The CRAG branch, as a function of the grade and the iteration counter.

    Pure, and separated from the graph so it can be unit tested against a
    hand-built state — which is the only cheap way to be sure a loop terminates.
    """
    grade = state.get("grade")
    used = int(state.get("retrieve_iterations") or 0)
    budget = _budget(settings)

    if grade is GradeLabel.INCORRECT:
        if used >= budget:
            # Out of retries. Leaving through ``generate`` rather than raising is
            # what turns "the corpus cannot answer this" into an abstention the
            # caller can read, with the trail of what was tried attached.
            log.info("crag_loop_exhausted", iterations=used, budget=budget)
            return _GENERATE
        return _REWRITE
    if grade is GradeLabel.AMBIGUOUS:
        return _WEB
    # CORRECT, or no verdict at all: use what retrieval produced.
    return _REFINE


def decide_after_web(state: RAGState) -> str:
    """Where a web search leads, which depends on why it ran.

    From ``AMBIGUOUS`` the corpus documents are still the primary evidence and the
    web results complete them, so the next stop is the answer. From ``INCORRECT``
    the corpus produced nothing usable, so the loop goes back for another graded
    retrieval with the rewritten query — the web results stay in ``web_chunks`` and
    are still there if that retry also fails.
    """
    return _GRADE if state.get("grade") is GradeLabel.INCORRECT else _GENERATE


def build(
    nodes: PipelineNodes,
    *,
    settings: Settings | None = None,
    interrupt_before: Sequence[str] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the CRAG graph.

    ``settings`` is read at *build* time by the conditional edge closure, so the
    iteration budget is fixed when the graph is compiled rather than re-read per
    superstep. A budget that can change mid-loop is a budget that cannot be reasoned
    about.
    """
    resolved = settings or nodes.settings
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)

    graph.add_node("validate", nodes.validate)
    graph.add_node("translate", nodes.translate)
    graph.add_node("route", nodes.route)
    graph.add_node(_GRADE, nodes.grade)
    # Refinement at graph level is the configured compressor; CRAG's own
    # strip-level refinement already ran inside the grade node. The node no-ops
    # when compression is disabled, so CORRECT still reaches the answer.
    graph.add_node(_REFINE, nodes.compress)
    graph.add_node(_REWRITE, nodes.rewrite)
    graph.add_node(_WEB, nodes.web_search)
    graph.add_node(_GENERATE, nodes.generate)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "translate")
    graph.add_edge("translate", "route")
    graph.add_edge("route", _GRADE)
    graph.add_conditional_edges(
        _GRADE,
        lambda state: decide_after_grade(state, settings=resolved),
        [_REFINE, _WEB, _REWRITE, _GENERATE],
    )
    graph.add_edge(_REWRITE, _WEB)
    graph.add_conditional_edges(_WEB, decide_after_web, [_GRADE, _GENERATE])
    graph.add_edge(_REFINE, _GENERATE)
    graph.add_edge(_GENERATE, END)
    return graph.compile(name=NAME, interrupt_before=list(interrupt_before or ()))


def recursion_limit(settings: Settings) -> int:
    """Derive the safety net from the graph's own shape.

    One corrective iteration is three supersteps (``grade``, ``rewrite``, ``web``);
    the prologue is three (``validate``, ``translate``, ``route``) and the epilogue is
    two (``refine``/``web``, ``generate``). Scaling with the configured budget rather
    than hard-coding a number is what keeps the net *above* the graceful exit: a fixed
    limit becomes the primary bound the moment somebody raises the retry budget, and
    then every deep correction raises instead of abstaining.
    """
    return 8 + 3 * (_budget(settings) + 1)
