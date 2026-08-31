"""What a prompt asks for, and what the code does with the answer.

Never audited. Three ways a prompt and its consumer disagreed:

* `citation_style="json"` sends the `answer_with_citations` system block, which
  says "Split your answer into statements and, for each, list the numbers of the
  passages that support it" — attribution deliberately *not* inline. `statements`
  had no reader anywhere in the package, and the branch then ran the inline-`[n]`
  regex over an answer that by construction contains none: zero citations,
  `citation_coverage` 0.0, `report.valid` still True.
* The same branch hardcoded that system block, discarding the routed prompt — so
  `answer_technical` and `answer_concise` had no effect under this style while
  `answer.metadata["prompt"]` still reported the routed name.
* `resolve_prompt_name` accepts the documented shorthand, was defined in
  `pipeline.nodes`, and so reached one of the two wirings — and the startup check
  rejected the shorthand outright, so the setting it was written for could not be
  set.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.settings import Settings


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
        "security": {"enforce_tenant_isolation": False},
        # Merged, not replaced. A caller overriding `generation` used to drop
        # `check_groundedness`, and the abstention gate then replaced the answer
        # under test with a refusal — four tests failing for a reason that had
        # nothing to do with what they assert.
        "generation": {"check_groundedness": False, "allow_abstention": False},
    }
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Settings(**base)


# ---------------------------------------------------------------------------
# The shorthand
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("default", "answer_default"),
        ("concise", "answer_concise"),
        ("technical", "answer_technical"),
        ("answer_default", "answer_default"),
        ("multi_query", "multi_query"),
    ],
)
def test_get_prompt_accepts_the_shorthand(given: str, expected: str) -> None:
    """Resolved inside `get_prompt`, so there is nothing to remember at a call
    site. It used to live in `pipeline.nodes`, which is why it reached the
    RAGPipeline node and not `AnswerGenerator` — the path the HTTP engine takes."""
    from ragorc.llm.prompts import get_prompt

    assert get_prompt(given).name == expected


def test_an_unknown_name_still_raises() -> None:
    from ragorc.llm.prompts import get_prompt

    with pytest.raises(KeyError, match="unknown prompt"):
        get_prompt("no_such_prompt")


@pytest.mark.parametrize("name", ["default", "concise", "technical", "answer_default"])
def test_the_shorthand_is_accepted_at_startup(name: str) -> None:
    """The resolver's whole purpose is that this value is legitimate, and the
    startup check rejected it — so the setting could not be set."""
    assert _settings(generation={"prompt_name": name}).generation.prompt_name == name


def test_a_genuinely_unknown_name_is_still_refused_at_startup() -> None:
    from ragorc.core.errors import ConfigError

    with pytest.raises(ConfigError, match=r"unknown generation\.prompt_name"):
        _settings(generation={"prompt_name": "not_a_prompt"})


def test_there_is_one_definition_of_the_resolver() -> None:
    """Two spellings is the shape this fixes."""
    import inspect

    from ragorc.pipeline import nodes

    source = inspect.getsource(nodes)
    assert "def resolve_prompt_name" not in source, "nodes defines a second copy again"
    assert "resolve_prompt_name" in source, "nodes no longer uses it at all"


# ---------------------------------------------------------------------------
# citation_style="json"
# ---------------------------------------------------------------------------
class _StatementLLM:
    """A model that obeys `answer_with_citations`: statements, no inline markers."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def structured(self, prompt: str, schema: Any, **kw: Any) -> Any:
        from ragorc.core.models import Usage

        self.calls.append({"prompt": prompt, **kw})
        return schema(
            answer="Refunds are processed within 14 days of the request.",
            statements=[
                {
                    "text": "Refunds are processed within 14 days of the request.",
                    "source_ids": [1],
                }
            ],
            sufficient=True,
        ), Usage(model="m", calls=1)

    async def complete(self, prompt: str, **kw: Any) -> Any:
        from ragorc.core.models import Usage

        self.calls.append({"prompt": prompt, **kw})
        return "unused", Usage(model="m", calls=1)


async def _answer(**over: Any) -> tuple[Any, _StatementLLM]:
    from ragorc.core.models import Chunk, Query, RetrievalResult, ScoredChunk
    from ragorc.generate.answer import AnswerGenerator

    settings = _settings(generation={"citation_style": "json", **over})
    llm = _StatementLLM()
    generator = AnswerGenerator(llm, settings)
    retrieval = RetrievalResult(
        chunks=[
            ScoredChunk(
                chunk=Chunk(
                    id="c1",
                    content="Refunds are processed within 14 days of the request.",
                    document_id="d1",
                ),
                score=1.0,
            )
        ]
    )
    answer = await generator.generate(Query(text="how long do refunds take?"), retrieval)
    return answer, llm


async def test_the_json_style_produces_citations() -> None:
    """Its entire purpose. The branch parsed `statements` and discarded it, then
    regexed for `[n]` markers its own system prompt had relocated."""
    answer, _llm = await _answer()

    assert answer.citations, "the style whose purpose is attribution produced none"
    citation = answer.citations[0]
    assert citation.chunk_id == "c1"
    assert citation.claim.startswith("Refunds are processed")


async def test_an_out_of_range_source_id_is_dropped_not_clamped() -> None:
    """A model naming passage 9 of 1 is guessing, and a citation pointing at the
    wrong passage is worse than one that is absent."""
    from ragorc.core.models import Chunk, Query, RetrievalResult, ScoredChunk
    from ragorc.generate.answer import AnswerGenerator

    class _Wild(_StatementLLM):
        async def structured(self, prompt: str, schema: Any, **kw: Any) -> Any:
            from ragorc.core.models import Usage

            return schema(
                answer="x",
                statements=[{"text": "x", "source_ids": [9]}],
                sufficient=True,
            ), Usage(model="m", calls=1)

    generator = AnswerGenerator(_Wild(), _settings(generation={"citation_style": "json"}))
    retrieval = RetrievalResult(
        chunks=[ScoredChunk(chunk=Chunk(id="c1", content="body", document_id="d"), score=1.0)]
    )
    answer = await generator.generate(Query(text="q"), retrieval)

    assert answer.citations == []


async def test_the_routed_prompt_survives_the_json_style() -> None:
    """The branch hardcoded `answer_with_citations.system`, so the router's choice
    had no effect while `metadata["prompt"]` still reported it."""
    from ragorc.llm.prompts import get_prompt

    _answer_obj, llm = await _answer(prompt_name="technical")

    sent = str(llm.calls[0].get("system") or "")
    assert get_prompt("answer_technical").system in sent, "the routed prompt was discarded"
    assert get_prompt("answer_with_citations").system in sent, "the attribution contract was lost"


async def test_the_inline_style_is_unchanged() -> None:
    """The default path must keep using the marker regex."""
    from ragorc.core.models import Chunk, Query, RetrievalResult, ScoredChunk
    from ragorc.generate.answer import AnswerGenerator
    from tests.fakes import StubLLM

    llm = StubLLM(text="Refunds take 14 days [1].")
    generator = AnswerGenerator(llm, _settings(generation={"citation_style": "inline"}))
    retrieval = RetrievalResult(
        chunks=[
            ScoredChunk(
                chunk=Chunk(id="c1", content="Refunds take 14 days.", document_id="d"), score=1.0
            )
        ]
    )
    answer = await generator.generate(Query(text="q"), retrieval)

    assert [c.chunk_id for c in answer.citations] == ["c1"]


# ---------------------------------------------------------------------------
# The scaffold stripper
# ---------------------------------------------------------------------------
def test_only_our_own_fence_counts_as_a_leak() -> None:
    """`system`, `instruction` and `context` were borrowed from the *inbound*
    injection pattern, where they are defensive. Here they deleted legitimate
    answer content, and `\\b` matches an XML namespace prefix — so an answer from
    `answer_technical`, whose job is to reproduce code exactly, came back mangled.
    """
    from ragorc.validate.output import _SCAFFOLD

    code = 'Add `<context:component-scan base-package="com.acme"/>` and not `<system>`.'
    assert _SCAFFOLD.sub("", code) == code, "the validator deleted the answer's code"

    leaked = "Here is the answer <untrusted_document index='1'> oops"
    assert "<untrusted_document" not in _SCAFFOLD.sub("", leaked)
