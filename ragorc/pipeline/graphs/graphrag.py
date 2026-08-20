"""GraphRAG: three searches over one graph, chosen by the shape of the question.

    validate -> translate -> classify -+-> graph_local  -> fuse -> rerank -> generate
                                       +-> graph_global -> fuse -> reduce -> generate
                                       +-> graph_drift  -> fuse -> rerank -> generate

The three modes are not variations on a theme, they answer structurally different
questions, and picking the wrong one produces a confident irrelevance rather than a
worse ranking:

**local** — entity-anchored. Matches the entities the question names against the
graph, expands outward, and collects the chunks that asserted those edges. This is
the mode for "what did Acme announce about the Q3 outage": there is a named thing,
and the answer is in its neighbourhood.

**global** — map-reduce over community summaries. There is no entity to anchor on, so
there is nothing to expand *from*; instead every community summary is asked what it
contributes, and the surviving partials are synthesized. This is the mode for "what
are the main themes across these documents", which no amount of nearest-neighbour
search answers because the answer is a property of the corpus rather than of any
passage in it.

**drift** — vector seeds, then graph expansion around whatever they were about. The
two halves cover each other's blind spots: vector search finds the right passage
without the question naming anything, but cannot see the edge that joins two
passages; the graph has the edges but must be entered at a named node.

The choice rule, and why it costs nothing
-----------------------------------------
Entity-specific goes local, thematic goes global, everything else goes drift. That
decision is made from the question's *shape* by :func:`classify` — no model call — for
the same reason :class:`~ragorc.retrieve.multihop.MultiHopRetriever` decides its own
branch without one: the signals are lexical and unambiguous. A capitalized proper
noun, a quoted phrase or an identifier means something is named; "overall", "across
the documents", "main themes", "trends" mean the question is about the corpus. Paying
a classification call to learn what a regex already knows would add latency and a
failure mode to the cheapest decision in the pipeline.

Two of the four cases collapse into drift on purpose. **Mixed** (a named entity *and*
a thematic frame — "how has Acme's strategy shifted overall") needs both halves, which
is what drift is. **Neither** (a descriptive question that names nothing and asks for
no synthesis) is the case local search cannot serve at all, because it has no entry
node — and it is exactly the case DRIFT was designed for. Falling back to local there
would return nothing and look like an empty corpus.

Why global search reduces through the generator
-----------------------------------------------
:class:`~ragorc.retrieve.graph.GraphGlobalRetriever` implements only the **map** half,
and that seam is deliberate: map is retrieval (N independent cheap parallel calls,
each producing a scored partial that is exactly ``ScoredChunk``-shaped), while reduce
is generation — one call that synthesizes partials into prose, which already needs
citation formatting, groundedness checking and abstention. Doing the reduce inside the
retriever would fork all of that.

So the reduce is the ordinary ``generate`` node with the ``global_reduce`` wording. The
prompt is registered under a second name because the generator renders every prompt
with ``(context, question)`` while ``global_reduce`` names its evidence slot
``partials``: same system prompt, same words, one slot renamed so the generator's
signature fits. Deriving it from the registered prompt rather than retyping it means
the two cannot drift apart.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ragorc.core.settings import Settings
from ragorc.llm.prompts import Prompt, get_prompt, register_prompt
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import RAGState

log = structlog.get_logger(__name__)

__all__ = [
    "GLOBAL_REDUCE_PROMPT",
    "MODE_NODES",
    "NAME",
    "build",
    "classify",
    "recursion_limit",
]

NAME = "graphrag"

MODE_NODES: dict[str, str] = {
    "local": "graph_local",
    "global": "graph_global",
    "drift": "graph_drift",
}

GLOBAL_REDUCE_PROMPT = "global_reduce_answer"
"""``global_reduce`` with its evidence slot renamed from ``partials`` to ``context``.

Registered at import so the generator — which renders every prompt as
``render(context=..., question=...)`` — can use the graph-specific reduce wording
without a special case, and without a second copy of the text to keep in sync.
"""


def _register_reduce_prompt() -> Prompt:
    source = get_prompt("global_reduce")
    return register_prompt(
        Prompt(
            name=GLOBAL_REDUCE_PROMPT,
            template=source.template.replace("{partials}", "{context}"),
            system=source.system,
            description=f"{source.description} Slot renamed for the answer generator.",
            tags=(*source.tags, "generate"),
        )
    )


_register_reduce_prompt()


# ---------------------------------------------------------------------------
# The choice rule
# ---------------------------------------------------------------------------
# Curly quotes are written as escapes, not as literals: they are the characters the
# injection scanner normalizes away, and a linter cannot tell a deliberate homoglyph
# in a pattern from an accidental one in prose.
_OPEN_QUOTES = "\"'\u201c\u2018"
_CLOSE_QUOTES = "\"'\u201d\u2019"
_QUOTED = re.compile(f"[{_OPEN_QUOTES}]([^{_CLOSE_QUOTES}]{{2,}})[{_CLOSE_QUOTES}]")
"""A quoted phrase is a named thing even when it is not capitalized."""

_SENTENCE_OPENERS = frozenset(
    [
        "a",
        "an",
        "the",
        "what",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "which",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "list",
        "give",
        "tell",
        "show",
        "summarize",
        "summarise",
        "compare",
        "explain",
        "describe",
        "in",
        "on",
        "of",
        "and",
        "or",
        "for",
        "to",
        "at",
        "by",
        "with",
    ]
)
"""Words that are capitalized because they start a question, not because they name
anything. Without this, every question would look entity-specific."""

_CAPITALIZED = re.compile("\\b[A-Z][A-Za-z0-9]*(?:[-'\u2019][A-Za-z0-9]+)*\\b")
_IDENTIFIER = re.compile(r"\b(?=[\w.-]*\d)(?=[\w.-]*[A-Za-z])[\w.-]{3,}\b")
"""Mixed letters and digits: error codes, part numbers, versions, ticket ids. These
are the tokens dense retrieval is worst at and the graph is often best at."""

_THEMATIC = (
    "overall",
    "in general",
    "main theme",
    "main topic",
    "key theme",
    "key topic",
    "major theme",
    "high level",
    "high-level",
    "across the",
    "across all",
    "across these",
    "common thread",
    "big picture",
    "summarize the",
    "summarise the",
    "summary of the",
    "what are the trends",
    "trends in",
    "patterns in",
    "landscape",
    "consensus",
)
"""Phrases that say the question is about the corpus rather than about a passage in
it. Matched as substrings on the casefolded question, which is deliberately blunt:
a false positive costs a global search instead of a local one, and a global search
still answers an entity question — expensively, but correctly."""


def _names_something(question: str) -> bool:
    if _QUOTED.search(question):
        return True
    if _IDENTIFIER.search(question):
        return True
    return any(word.casefold() not in _SENTENCE_OPENERS for word in _CAPITALIZED.findall(question))


def classify(question: str) -> str:
    """Pick the search mode from the question's shape.

    ==============================  ========  ===============================
     names an entity?                thematic?  mode
    ==============================  ========  ===============================
     yes                             no        ``local``
     no                              yes       ``global``
     yes                             yes       ``drift`` (needs both halves)
     no                              no        ``drift`` (nothing to anchor on)
    ==============================  ========  ===============================
    """
    thematic = any(marker in question.casefold() for marker in _THEMATIC)
    named = _names_something(question)
    if named and not thematic:
        return "local"
    if thematic and not named:
        return "global"
    return "drift"


async def classify_node(state: RAGState) -> dict[str, Any]:
    """Record the chosen mode so the conditional edge and the trace agree.

    The mode is written to state rather than recomputed in the edge function: an edge
    that re-derives its own decision can disagree with what was logged, and a
    disagreement between the trace and the execution is the worst kind of bug to
    debug in a graph.
    """
    query = state.get("query")
    question = (query.original or query.text) if query is not None else state["question"]
    mode = classify(question)
    log.info("graphrag_mode", mode=mode, question=question[:120])
    return {"search_mode": mode}


def _route_mode(state: RAGState) -> str:
    return MODE_NODES.get(state.get("search_mode") or "drift", MODE_NODES["drift"])


def _after_fuse(state: RAGState) -> str:
    """Global search reduces; the other two rerank.

    Reranking global partials would be a category error: they are generated text
    scored 0-10 for how much each community *contributes*, and a cross-encoder would
    re-score them for lexical relevance to the question, which is not the ranking that
    matters and discards the map step's judgement.
    """
    return "reduce" if state.get("search_mode") == "global" else "rerank"


async def reduce_node(state: RAGState) -> dict[str, Any]:
    """Switch the generator to the ``global_reduce`` wording.

    A node rather than a settings flag because the choice belongs to this *query* —
    the same deployment answers entity questions with the default prompt in the same
    process, on the same generator instance.
    """
    del state
    return {"prompt_name": GLOBAL_REDUCE_PROMPT}


def build(
    nodes: PipelineNodes, *, settings: Settings | None = None
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compile the GraphRAG graph."""
    del settings
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(RAGState)

    graph.add_node("validate", nodes.validate)
    graph.add_node("translate", nodes.translate)
    graph.add_node("classify", classify_node)
    for mode, node_name in MODE_NODES.items():
        graph.add_node(node_name, nodes.graph_node(mode))
    graph.add_node("fuse", nodes.fuse)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("reduce", reduce_node)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "validate")
    graph.add_edge("validate", "translate")
    graph.add_edge("translate", "classify")
    graph.add_conditional_edges("classify", _route_mode, list(MODE_NODES.values()))
    for node_name in MODE_NODES.values():
        # All three modes converge on ``fuse``, which is what turns the accumulated
        # ``per_store`` contributions into the one authoritative RetrievalResult. On
        # this path its relative score cutoff doubles as a helpfulness floor for the
        # global partials, which is the behaviour we want: a community that scored
        # 1/10 next to one that scored 9 has nothing to add.
        graph.add_edge(node_name, "fuse")
    graph.add_conditional_edges("fuse", _after_fuse, ["rerank", "reduce"])
    graph.add_edge("rerank", "generate")
    graph.add_edge("reduce", "generate")
    graph.add_edge("generate", END)
    return graph.compile(name=NAME)


def recursion_limit(settings: Settings) -> int:
    """Acyclic: validate, translate, classify, one search, fuse, rerank/reduce,
    generate. Stated rather than defaulted so the number is derived from the graph."""
    del settings
    return 12
