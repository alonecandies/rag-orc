"""The graph state, and why several of its fields need reducers.

A RAG query is a state machine, not a chain (ADR-0001), and LangGraph's state
object is what makes that tractable: every node receives the whole state and
returns only the keys it changed, so control flow lives in the edges instead of
being threaded by hand through a dozen call signatures.

Reducers are the one part of this file that is not obvious
---------------------------------------------------------
LangGraph executes a fan-out as a **superstep**: every node reachable in the same
step runs concurrently, and their returned partial states are applied to the
channels together at the end of the step. A channel with no reducer is a
``LastValue`` channel, and ``LastValue`` *raises* ``InvalidUpdateError`` when two
nodes in one superstep write it — because there is no defensible answer to "which
of these two values is the state now?". Silently keeping one would be worse: the
retrieval results of an entire store would vanish with no error anywhere.

That is the single most common LangGraph mistake, and it only appears once you add
the second parallel branch — which in this library is the *default* path, since
:mod:`ragorc.pipeline.graphs.adaptive` fans out to every store the router chose.
So every field a parallel branch can write is annotated with a binary reducer:

===================  ==================  =====================================
 field                reducer             written concurrently by
===================  ==================  =====================================
``candidates``        ``operator.add``    one node per routed datastore
``per_store``         :func:`merge_store_lists`  same
``web_chunks``        ``operator.add``    the web branch, alongside the corpus
``usages``            ``operator.add``    every node that spends a model call
``errors``            ``operator.add``    any degrading node, in any branch
``warnings``          ``operator.add``    validation and the guards
``rewrites``          ``operator.add``    the CRAG and Self-RAG rewrite loops
``tools_used``        ``operator.add``    the agentic graph's branches
===================  ==================  =====================================

Everything else is deliberately ``LastValue``, and that is a *feature*: if two
nodes ever race to set ``route`` or ``grade``, the graph is wrong and the raised
``InvalidUpdateError`` says so at the first test run rather than producing a
plausible answer from whichever branch happened to finish second.

Why ``candidates`` and ``retrieval`` both exist
----------------------------------------------
``candidates`` is the *accumulator*: unordered, unfused, one append per branch.
``retrieval`` is the *authority*: the single fused, denoised, reranked
:class:`~ragorc.core.models.RetrievalResult` that the generator will read, written
by exactly one node per step. Collapsing them into one field would mean either
losing the per-branch contributions (which is what ``per_store`` diagnostics are
made of) or making the authoritative list a reducer target, i.e. making "what the
generator sees" depend on node completion order.

Usage is accumulated as a *list* rather than summed into one
:class:`~ragorc.core.models.Usage`. ``Usage.__add__`` keeps the first non-empty
model name, so summing eagerly across concurrent branches would discard the
per-call attribution that the cost report is built from; the ledger in
:mod:`ragorc.core.telemetry` already has the per-model and per-stage breakdown,
and :func:`total_usage` produces the single number when one is wanted.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from ragorc.core.models import (
    Answer,
    GradeLabel,
    Query,
    RetrievalResult,
    RouteDecision,
    ScoredChunk,
    Usage,
    dedupe_scored,
)

__all__ = [
    "RAGState",
    "evidence",
    "failure",
    "initial_state",
    "merge_store_lists",
    "total_usage",
]


def merge_store_lists(
    left: dict[str, list[ScoredChunk]] | None,
    right: dict[str, list[ScoredChunk]] | None,
) -> dict[str, list[ScoredChunk]]:
    """Merge two ``store -> results`` maps, concatenating on key collision.

    Keys are store names, so a collision means the same store contributed twice in
    one superstep — a retry, or a graph that queries one store from two branches.
    Concatenating preserves both contributions; the fuse step deduplicates by chunk
    id afterwards, which it has to do anyway because two *different* stores
    routinely return the same passage.
    """
    merged: dict[str, list[ScoredChunk]] = dict(left or {})
    for name, chunks in (right or {}).items():
        if name in merged:
            merged[name] = [*merged[name], *chunks]
        else:
            merged[name] = list(chunks)
    return merged


class RAGState(TypedDict, total=False):
    """State shared by every node of every graph in this package.

    ``total=False`` because a node returns a *partial* state and LangGraph only
    materializes channels that have been written; nodes therefore read with
    ``state.get(...)`` and never index a key they did not set themselves.

    One state type for all seven graphs rather than one per graph: the nodes in
    :mod:`ragorc.pipeline.nodes` are shared, so a per-graph state would mean a
    per-graph copy of every node. The cost is a few fields that only one graph
    populates, which is cheaper than the alternative in every dimension.
    """

    # --- request ---------------------------------------------------------
    question: str
    """The user's untouched question. Never rewritten — ``Query.text`` carries
    the rewrites, and the final answer is graded against this."""
    tenant_id: str | None
    top_k: int | None
    pipeline: str
    """Which graph was selected, for the trace and the answer metadata."""
    prompt_name: str | None
    """Overrides the generator's prompt. GraphRAG global search sets it so the
    reduce step uses the ``global_reduce`` wording."""

    # --- stage outputs ---------------------------------------------------
    query: Query | None
    route: RouteDecision | None
    retrieval: RetrievalResult | None
    answer: Answer | None

    # --- accumulators (see the module docstring) -------------------------
    candidates: Annotated[list[ScoredChunk], operator.add]
    per_store: Annotated[dict[str, list[ScoredChunk]], merge_store_lists]
    web_chunks: Annotated[list[ScoredChunk], operator.add]
    usages: Annotated[list[Usage], operator.add]
    errors: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    rewrites: Annotated[list[str], operator.add]
    tools_used: Annotated[list[str], operator.add]

    # --- grades ----------------------------------------------------------
    grade: GradeLabel | None
    """CRAG's document verdict. ``None`` means no verdict was obtained, which is
    not the same as INCORRECT and must not be treated as one."""
    grounded: bool
    groundedness: float
    useful: bool
    utility: float
    sufficient: bool
    """Multi-hop early exit: does the evidence already answer the question?"""
    follow_up: str
    """The next search the sufficiency check asked for. Empty means the model had
    no follow-up, which is a dead end rather than licence to guess."""

    # --- loop counters ---------------------------------------------------
    retrieve_iterations: int
    """Retrieval-side loops: CRAG corrections and multi-hop hops."""
    generate_iterations: int
    """Answer-side loops: Self-RAG regenerations."""

    # --- mode selection --------------------------------------------------
    search_mode: str
    """GraphRAG: ``local`` / ``global`` / ``drift``. Agentic: the branch taken."""

    metadata: dict[str, Any]


def initial_state(
    question: str,
    *,
    tenant_id: str | None = None,
    top_k: int | None = None,
    pipeline: str = "auto",
    prompt_name: str | None = None,
) -> RAGState:
    """Seed a run.

    Only the counters and the request fields are seeded. The accumulators are left
    absent on purpose: LangGraph initializes a reducer channel from its annotated
    type on first write, so seeding ``[]`` here would be redundant, and seeding a
    *shared* list would alias it across concurrent runs.
    """
    return RAGState(
        question=question,
        tenant_id=tenant_id,
        top_k=top_k,
        pipeline=pipeline,
        prompt_name=prompt_name,
        retrieve_iterations=0,
        generate_iterations=0,
        metadata={},
    )


def total_usage(state: RAGState) -> Usage:
    """Sum the per-call usages into one bill."""
    return Usage.sum(state.get("usages") or ())


def evidence(state: RAGState) -> list[ScoredChunk]:
    """Everything the generator should see, corpus first then web.

    Ordering is by *provenance*, matching :class:`CorrectiveRAG`'s reasoning: a
    cosine similarity and a search engine's rank position are not on one scale, so
    interleaving them by score would let an arbitrary scale decide the ranking. The
    packer's lost-in-the-middle reordering runs afterwards regardless.

    Deduplicated by chunk id keeping the higher score, because the same passage can
    arrive from the corpus and from a web result that quotes it.
    """
    retrieval = state.get("retrieval")
    corpus = list(retrieval.chunks) if retrieval is not None else []
    web = list(state.get("web_chunks") or ())
    if not web:
        return corpus
    merged = dedupe_scored([*corpus, *web])
    for rank, scored in enumerate(merged):
        scored.rank = rank
    return merged


def failure(stage: str, exc: BaseException) -> dict[str, Any]:
    """The partial state a degrading node returns.

    Truncated because these strings end up on the answer's trace and in logs, and
    a driver traceback pasted whole into a JSON log line is how a log pipeline
    starts dropping events.
    """
    return {"errors": [f"{stage}: {type(exc).__name__}: {str(exc)[:300]}"]}
