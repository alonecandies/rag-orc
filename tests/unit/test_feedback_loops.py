"""What the loops decide, and what the settings that govern them actually do.

Round eight audited the graph *structure* and found the agentic graph computing
CRAG's verdict and discarding it. This is the decisions: widths, flags, double
work, and whose bill reaches the answer.

Two of the four here are the same shape — a setting honoured on one path and not
the other, while `describe()` reports the feature as configured. A flag that is
read in three places and ignored in the fourth is worse than one that was never
implemented, because the operator has no reason to check.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Chunk, Query, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings
from ragorc.generate.answer import AnswerGenerator
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import initial_state
from tests.fakes import StubLLM


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
    }
    base.update(over)
    return Settings(**base)


class _Recording:
    name = "recording"

    def __init__(self) -> None:
        self.asked: list[int | None] = []

    async def retrieve(self, query: Query, *, top_k: int | None = None, **kw: Any) -> list[Any]:
        self.asked.append(top_k)
        return [
            ScoredChunk(
                chunk=Chunk(id=f"c{i}", content=f"body {i}", document_id="d"),
                score=1.0 - i / 100,
                source=RetrievalSource.DENSE,
                rank=i,
            )
            for i in range(top_k or 0)
        ]


def _nodes(settings: Settings, **kw: Any) -> PipelineNodes:
    llm = StubLLM()
    return PipelineNodes(
        settings=settings, llm=llm, generator=AnswerGenerator(llm, settings), **kw
    )


# ---------------------------------------------------------------------------
# How wide a later hop fetches
# ---------------------------------------------------------------------------
async def test_a_later_hop_fetches_as_wide_as_the_first() -> None:
    """The third site to read `state["top_k"]` where its siblings call
    `_fetch_k`, after `nodes.retrieve` (round twelve) and `naive` (round
    thirteen). Two of the three are in this file.

    It matters more here than anywhere else: the first hop is an ordinary
    search, and the *later* hops are what multi-hop exists for — so the narrow
    leg was on exactly the queries that needed the recall, feeding the same
    reranker a fifth as many candidates.
    """
    settings = _settings(retrieval={"top_k": 10, "fetch_k": 50})
    retriever = _Recording()
    nodes = _nodes(settings, retriever=retriever)

    state = initial_state("who founded the company?")
    state["query"] = Query(text="who founded the company?")
    await nodes.retrieve(state)
    state["follow_up"] = "when was it founded"
    await nodes.hop(state)

    assert retriever.asked == [50, 50], f"hop 0 and hop 1 disagree: {retriever.asked}"


@pytest.mark.parametrize(
    "node", ["retrieve", "hop", "bridge", "store_node", "multihop_retrieve"]
)
def test_every_retrieval_leg_uses_the_shared_width(node: str) -> None:
    """Asserted per node, because four separate legs have now made this mistake
    and a fifth would be found the same way — by hand, rounds later.

    Only the legs that *fetch*. `validate` sets the answer's `top_k` and `rerank`
    narrows *to* it; both read the state's value correctly and a blanket ban on
    the pattern would flag them. The distinction is whether the number bounds a
    search or bounds the answer.
    """
    import inspect

    source = inspect.getsource(getattr(PipelineNodes, node))
    assert 'top_k=state.get("top_k")' not in source, (
        f"{node} fetches at the state's top_k instead of _fetch_k"
    )
    assert "_fetch_k(state)" in source, f"{node} does not use the shared width"


# ---------------------------------------------------------------------------
# crag_web_fallback
# ---------------------------------------------------------------------------
class _Web:
    enabled = True
    name = "web"

    def __init__(self) -> None:
        self.searched: list[str] = []

    async def retrieve(self, query: Query, *, top_k: int | None = None, **kw: Any) -> list[Any]:
        self.searched.append(query.text)
        return [
            ScoredChunk(
                chunk=Chunk(id="w0", content="from the web", document_id="web"),
                score=0.5,
                source=RetrievalSource.WEB,
                rank=0,
            )
        ]


@pytest.mark.parametrize("enabled", [True, False])
async def test_the_web_node_honours_crag_web_fallback(enabled: bool) -> None:
    """Read by CRAG's own fallback, by the linear engine's retriever construction
    and by `describe()` — and not by this node, so on the `crag` and `agentic`
    graphs an operator who switched it off still had every AMBIGUOUS and
    INCORRECT query sent to a search engine while `/health` reported the feature
    disabled.

    A web search sends the user's question to a third party, so "configured off
    and still running" is the one direction this flag must not fail in.
    """
    settings = _settings(retrieval={"crag_web_fallback": enabled})
    web = _Web()
    nodes = _nodes(settings, retriever=_Recording(), web=web)

    state = initial_state("how long do refunds take?")
    state["query"] = Query(text="how long do refunds take?")
    await nodes.web_search(state)

    assert bool(web.searched) is enabled, f"searched={web.searched} with flag={enabled}"


async def test_the_graph_owns_the_web_step_so_crag_does_not_repeat_it() -> None:
    """`decide_after_grade` routes AMBIGUOUS to a `web_search` node, and CRAG's
    internal fallback fired as well — two rewrite calls, two provider requests,
    and both sets of results fused as though they were independent evidence.
    """
    import inspect

    from ragorc.retrieve.crag import CorrectiveRAG

    source = inspect.getsource(PipelineNodes.grade)
    assert "web=False" in source, "the grade node let CRAG run its own web leg too"

    # And the parameter exists with the default the linear engine relies on.
    assert inspect.signature(CorrectiveRAG.run).parameters["web"].default is True

    # Behaviour, not just the argument. A mutation that accepted `web` and
    # ignored it survived a source-only assertion — the same weakness that let a
    # docstring containing the word "rerank" satisfy a grep in round thirteen.
    from ragorc.core.models import GradeLabel

    crag = CorrectiveRAG(
        _Recording(), StubLLM(), _settings(retrieval={"crag_web_fallback": True}), web=_Web()
    )
    assert crag._web_wanted(GradeLabel.AMBIGUOUS, allowed=True) is True
    assert crag._web_wanted(GradeLabel.AMBIGUOUS, allowed=False) is False, (
        "the caller said it owns the web step and CRAG searched anyway"
    )


# ---------------------------------------------------------------------------
# allow_abstention
# ---------------------------------------------------------------------------
async def test_the_abstain_node_honours_allow_abstention() -> None:
    """`AbstentionPolicy` returns "do not abstain" whenever this is off, and the
    node's fallback branch overwrote the answer with the refusal regardless — so
    the policy's decision was made and then ignored, with
    `gate="loop_exhausted"`."""
    from ragorc.core.models import Answer

    settings = _settings(generation={"allow_abstention": False})
    nodes = _nodes(settings, retriever=_Recording())

    state = initial_state("q")
    state["answer"] = Answer(text="the model's best attempt", grounded=False, groundedness=0.1)
    state["grounded"] = False
    state["useful"] = False
    state["generate_iterations"] = 2

    out = await nodes.abstain(state)

    assert out == {}, f"the answer was replaced despite the setting: {out}"
    assert state["answer"].text == "the model's best attempt"
    assert not state["answer"].abstained


async def test_the_abstain_node_still_abstains_when_allowed() -> None:
    """The default, and the behaviour the loops exist to produce."""
    from ragorc.core.models import Answer

    settings = _settings(generation={"allow_abstention": True})
    nodes = _nodes(settings, retriever=_Recording())

    state = initial_state("q")
    state["answer"] = Answer(text="ungrounded guess", grounded=False, groundedness=0.1)
    state["grounded"] = False
    state["useful"] = False
    state["generate_iterations"] = 2

    out = await nodes.abstain(state)

    assert out, "an ungrounded exhausted loop must abstain by default"
    assert out["answer"].abstained


async def test_self_rag_returns_its_best_attempt_when_abstention_is_off() -> None:
    """The same flag, the same omission, one module over."""
    from ragorc.core.models import Answer
    from ragorc.generate.self_rag import SelfRAG

    settings = _settings(generation={"allow_abstention": False})
    loop = SelfRAG(StubLLM(), settings)
    best = Answer(text="the best of three attempts", grounded=False)

    out = loop._abstain(best, Query(text="q"), [])

    assert out.text == "the best of three attempts"
    assert not out.abstained
    assert out.grounded is False, "it is still not grounded, and must say so"
    assert "abstention_suppressed" in out.metadata


# ---------------------------------------------------------------------------
# Whose bill reaches the answer
# ---------------------------------------------------------------------------
async def test_the_loops_bill_reaches_the_answer() -> None:
    """`SelfRAGResult.usage` and `RRRResult.usage` were computed by their owners
    and had no reader in `_LinearEngine`, so a successful Self-RAG run — two
    answers, two groundedness grades, two utility grades, one rewrite — reported
    the single generation call the answer happened to carry.

    Everything downstream reads that number: the cost ceiling, the `$` column in
    `ragorc eval`, and the Prometheus histogram.
    """
    import inspect

    from ragorc.server.app import _LinearEngine

    source = inspect.getsource(_LinearEngine)
    assert "loop_usage" in source, "the loops' usage has no reader again"
    assert "Usage.sum([answer.usage, *loop_usage])" in source, (
        "the loops' usage must be added to the generation's, not replace it"
    )


async def test_usage_sum_adds_rather_than_replaces() -> None:
    """The half a source check cannot see: assigning instead of summing would
    lose the generation call that produced the answer."""
    from ragorc.core.models import Usage

    generation = Usage(model="m", calls=1, prompt_tokens=100, completion_tokens=20, cost_usd=0.01)
    loop = Usage(model="m", calls=5, prompt_tokens=500, completion_tokens=50, cost_usd=0.05)

    total = Usage.sum([generation, loop])

    assert total.calls == 6
    assert total.cost_usd == pytest.approx(0.06)
