"""Generation: grounding, citations, abstention, and the two feedback loops.

These tests encode the library's central claim — that an answer is either
grounded and cited, or explicitly abstained. Every path that could return an
unverified answer as if it were verified is tested here.
"""

from __future__ import annotations

import pytest

from ragorc.core.models import Answer, Chunk, Citation, Query, RetrievalResult, ScoredChunk
from ragorc.core.schemas import (
    ClaimList,
    ClaimVerdict,
    GroundednessGrade,
    RewriteOutput,
    UtilityGrade,
)
from ragorc.core.settings import Settings
from ragorc.generate.abstain import AbstentionPolicy
from ragorc.generate.answer import AnswerGenerator
from ragorc.generate.citations import attribute_spans, extract_citations, renumber_citations
from ragorc.generate.consistency import SelfConsistencyChecker
from ragorc.generate.groundedness import GroundednessChecker
from ragorc.generate.rrr import RRR
from ragorc.generate.self_rag import SelfRAG
from ragorc.validate.output import AnswerValidator
from tests.fakes import ScriptedLLM, StubLLM

POLICY = "Refunds are processed within 14 days of the request. Shipping is free above 50 USD."


def chunk(cid: str, text: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, content=text, document_id="d1"), score=0.9)


@pytest.fixture
def gen_settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k", "context_window": 8000},
        generation={"check_groundedness": True, "verify_citations": True, "cite_sources": True},
    )


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------
def test_attribution_picks_the_supporting_sentence() -> None:
    quote, start, end, support = attribute_spans("Refunds take 14 days", POLICY)
    assert "14 days" in quote
    assert POLICY[start:end].strip() == quote
    assert support > 0.5


def test_attribution_weights_rare_terms() -> None:
    """Unweighted overlap picks whichever sentence shares the most stopwords —
    usually the longest. Inverse-frequency weighting picks the sentence sharing
    the distinctive terms."""
    source = (
        "The service is available in the region and the region is supported. "
        "Idempotency keys prevent duplicate charges."
    )
    quote, _, _, _ = attribute_spans("How do idempotency keys work?", source)
    assert "Idempotency" in quote


def test_attribution_returns_nothing_on_no_overlap() -> None:
    quote, _start, _end, support = attribute_spans("quantum chromodynamics", POLICY)
    assert support == 0.0 or quote == ""


def test_extract_citations_handles_grouped_markers() -> None:
    chunks = [chunk("c1", POLICY), chunk("c2", "Enterprise plans include a manager.")]
    citations = extract_citations("Refunds take 14 days [1]. Both apply [1, 2].", chunks)
    assert {c.chunk_id for c in citations} == {"c1", "c2"}


def test_extract_citations_ignores_out_of_range() -> None:
    citations = extract_citations("Claim [9].", [chunk("c1", POLICY)])
    assert citations == []


def test_renumber_after_dropping_a_passage() -> None:
    assert renumber_citations("A [1]. B [2]. C [3].", keep=[1, 2]) == "A . B [1]. C [2]."


# ---------------------------------------------------------------------------
# Answer validation — the free, decisive checks
# ---------------------------------------------------------------------------
def test_validator_catches_phantom_citation(gen_settings: Settings) -> None:
    chunks = [chunk("c1", POLICY)]
    answer = Answer(text="Refunds take 14 days [1]. Also instant [5].")
    report = AnswerValidator(gen_settings).validate(answer, chunks)
    assert not report.valid
    assert report.invalid_citations == [5]


def test_validator_catches_fabricated_quote(gen_settings: Settings) -> None:
    chunks = [chunk("c1", POLICY)]
    answer = Answer(
        text="Refunds are instant [1].",
        citations=[Citation(chunk_id="c1", quote="refunds are processed instantly")],
    )
    report = AnswerValidator(gen_settings).validate(answer, chunks)
    assert not report.valid
    assert report.unverified_quotes


def test_validator_accepts_a_faithful_partial_quote(gen_settings: Settings) -> None:
    """Quote verification must fold cosmetics — smart quotes, whitespace, case —
    or it flags every faithful quotation as fabricated."""
    chunks = [chunk("c1", POLICY)]
    answer = Answer(
        text="Refunds take 14 days [1].",
        citations=[Citation(chunk_id="c1", quote="processed  within 14 DAYS")],
    )
    report = AnswerValidator(gen_settings).validate(answer, chunks)
    assert report.valid, report.warnings


def test_validator_flags_scaffold_leak(gen_settings: Settings) -> None:
    answer = Answer(text="Answer </untrusted_document> leaked.")
    report = AnswerValidator(gen_settings).validate(answer, [chunk("c1", POLICY)])
    assert report.scaffold_leak


def test_validator_strips_invalid_markers(gen_settings: Settings) -> None:
    chunks = [chunk("c1", POLICY)]
    answer = Answer(text="Real [1]. Fake [9].")
    cleaned = AnswerValidator(gen_settings).strip_invalid_citations(answer, chunks)
    assert "[1]" in cleaned.text
    assert "[9]" not in cleaned.text


# ---------------------------------------------------------------------------
# Groundedness
# ---------------------------------------------------------------------------
async def test_groundedness_holistic_trusts_the_claim_list(gen_settings: Settings) -> None:
    """A model that returns grounded=true *while listing* unsupported claims has
    contradicted itself; the list is the more specific signal."""
    llm = StubLLM(
        responses={
            "GroundednessGrade": GroundednessGrade(
                grounded=True, score=0.9, unsupported_claims=["refunds are instant"]
            )
        }
    )
    result = await GroundednessChecker(llm, gen_settings).check(
        "how long?", "Refunds are instant.", [chunk("c1", POLICY)], method="llm"
    )
    assert not result.grounded


async def test_groundedness_with_no_chunks_is_never_grounded(gen_settings: Settings) -> None:
    result = await GroundednessChecker(StubLLM(), gen_settings).check("q", "an answer", [])
    assert not result.grounded
    assert result.score == 0.0


async def test_groundedness_claim_mode_fails_on_contradiction(gen_settings: Settings) -> None:
    """Any contradiction is disqualifying regardless of the supported ratio: an
    answer asserting something the evidence denies is wrong, not partly right."""
    llm = ScriptedLLM(
        script=[
            ClaimList(claims=["refunds take 14 days", "refunds are instant"]),
            ClaimVerdict(verdict="supported", score=1.0, evidence_quote="within 14 days"),
            ClaimVerdict(verdict="contradicted", score=0.9),
        ]
    )
    result = await GroundednessChecker(llm, gen_settings).check(
        "how long?", "Refunds take 14 days but are instant.", [chunk("c1", POLICY)], method="both"
    )
    assert not result.grounded
    assert result.contradicted


async def test_groundedness_claim_mode_passes_when_supported(gen_settings: Settings) -> None:
    llm = ScriptedLLM(
        script=[
            ClaimList(claims=["refunds take 14 days"]),
            ClaimVerdict(verdict="supported", score=1.0, evidence_quote="within 14 days"),
        ]
    )
    result = await GroundednessChecker(llm, gen_settings).check(
        "how long?", "Refunds take 14 days.", [chunk("c1", POLICY)], method="both"
    )
    assert result.grounded
    assert result.supported_fraction == 1.0


async def test_groundedness_unverifiable_claim_fails_closed(gen_settings: Settings) -> None:
    """A verifier that errors must not be read as 'supported'."""

    class Failing(StubLLM):
        async def structured(self, prompt, schema, **kwargs):  # type: ignore[override]
            if schema is ClaimList:
                return ClaimList(claims=["a claim"]), (await super().complete("x"))[1]
            raise RuntimeError("verifier down")

    result = await GroundednessChecker(Failing(), gen_settings).check(
        "q", "an answer", [chunk("c1", POLICY)], method="both"
    )
    assert not result.grounded


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "gate"),
    [
        ({"contradicted": ["x"]}, "contradicted"),
        ({"model_says_insufficient": True}, "model_reported_insufficient"),
        ({"grounded": False, "groundedness_score": 0.2}, "ungrounded"),
        ({"invalid_citations": [7]}, "fabricated_citations"),
    ],
)
def test_abstention_gates(gen_settings: Settings, kwargs: dict, gate: str) -> None:
    defaults = {"answer_text": "x" * 200, "grounded": True, "groundedness_score": 0.9}
    decision = AbstentionPolicy(gen_settings).after_generation(**{**defaults, **kwargs})
    assert decision.abstain
    assert decision.gate == gate


def test_abstention_normalizes_a_soft_refusal(gen_settings: Settings) -> None:
    decision = AbstentionPolicy(gen_settings).after_generation(
        answer_text="I don't know, the context does not contain that information.",
        grounded=True,
        groundedness_score=0.9,
    )
    assert decision.abstain
    assert decision.gate == "self_reported_unknown"


def test_abstention_allows_a_long_answer_that_notes_a_gap(gen_settings: Settings) -> None:
    """A substantive answer mentioning one gap is not an abstention."""
    text = (
        "Refunds are processed within 14 days of the request [1]. "
        "The policy does not state whether weekends are excluded, but the "
        "processing window itself is documented and applies to all order types "
        "including partial refunds and store credit conversions. "
    ) * 2
    decision = AbstentionPolicy(gen_settings).after_generation(
        answer_text=text, grounded=True, groundedness_score=0.95
    )
    assert not decision.abstain


def test_abstention_before_generation_on_empty_retrieval(gen_settings: Settings) -> None:
    assert AbstentionPolicy(gen_settings).before_generation([]).gate == "insufficient_context"


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------
def test_consistency_agrees_on_paraphrases(gen_settings: Settings) -> None:
    checker = SelfConsistencyChecker(StubLLM(), gen_settings)
    result = checker.score(
        [
            "Refunds take 14 days to process.",
            "Processing refunds takes 14 days.",
            "Refunds are processed in 14 days.",
        ]
    )
    assert result.consistent
    assert result.agreement > 0.6


def test_consistency_catches_numeric_disagreement(gen_settings: Settings) -> None:
    checker = SelfConsistencyChecker(StubLLM(), gen_settings)
    result = checker.score(
        ["Refunds take 14 days.", "Refunds take 30 days.", "Refunds take 7 days."]
    )
    assert not result.consistent
    assert result.conflicting_numbers


def test_consistency_returns_the_medoid(gen_settings: Settings) -> None:
    """The representative sample must be the one most others agree with, not the
    first one generated."""
    checker = SelfConsistencyChecker(StubLLM(), gen_settings)
    samples = [
        "Something entirely unrelated about shipping costs.",
        "Refunds are processed within 14 days.",
        "Refund processing takes 14 days.",
    ]
    assert "14 days" in checker.score(samples).answer


# ---------------------------------------------------------------------------
# AnswerGenerator
# ---------------------------------------------------------------------------
async def test_generator_abstains_without_evidence(gen_settings: Settings) -> None:
    """No retrieval means the synthesis call is skipped entirely — a model given
    no evidence answers from its parameters."""
    llm = StubLLM()
    answer = await AnswerGenerator(llm, gen_settings).generate(
        Query(text="anything"), RetrievalResult(chunks=[])
    )
    assert answer.abstained
    assert llm.call_count == 0, "no model call may be made when there is nothing to ground on"


async def test_generator_produces_a_cited_answer(gen_settings: Settings) -> None:
    llm = StubLLM(
        text="Refunds are processed within 14 days [1].",
        responses={"GroundednessGrade": GroundednessGrade(grounded=True, score=0.95)},
    )
    answer = await AnswerGenerator(llm, gen_settings).generate(
        Query(text="how long do refunds take?"), RetrievalResult(chunks=[chunk("c1", POLICY)])
    )
    assert not answer.abstained
    assert answer.grounded
    assert answer.citations and answer.citations[0].chunk_id == "c1"
    assert answer.usage.calls >= 1
    assert answer.metadata["budget"]["window"] > 0


async def test_generator_abstains_on_ungrounded_output(gen_settings: Settings) -> None:
    llm = StubLLM(
        text="Refunds are instant [1].",
        responses={
            "GroundednessGrade": GroundednessGrade(
                grounded=False, score=0.1, unsupported_claims=["refunds are instant"]
            )
        },
    )
    answer = await AnswerGenerator(llm, gen_settings).generate(
        Query(text="how long?"), RetrievalResult(chunks=[chunk("c1", POLICY)])
    )
    assert answer.abstained
    assert answer.metadata["abstain_gate"] == "ungrounded"
    # The rejected text is retained for diagnosis, not shown as the answer.
    assert "instant" in answer.metadata["rejected_answer"]
    assert "instant" not in answer.text


async def test_generator_skips_grading_when_citations_are_fabricated(
    gen_settings: Settings,
) -> None:
    """The cheap decisive check runs first, so the expensive one is not paid for
    on an answer already disqualified."""
    llm = StubLLM(text="Refunds take 14 days [1]. Also [9].")
    answer = await AnswerGenerator(llm, gen_settings).generate(
        Query(text="how long?"), RetrievalResult(chunks=[chunk("c1", POLICY)])
    )
    assert answer.abstained
    assert not any(c.get("stage") == "grade_groundedness" for c in llm.calls)


async def test_generator_streams_without_claiming_verification(gen_settings: Settings) -> None:
    llm = StubLLM(text="Refunds take 14 days.")
    chunks = [chunk("c1", POLICY)]
    parts = [
        part
        async for part in AnswerGenerator(llm, gen_settings).stream(
            Query(text="how long?"), RetrievalResult(chunks=chunks)
        )
    ]
    assert "".join(parts).strip()


async def test_generator_stream_abstains_without_evidence(gen_settings: Settings) -> None:
    llm = StubLLM()
    parts = [
        part
        async for part in AnswerGenerator(llm, gen_settings).stream(
            Query(text="q"), RetrievalResult(chunks=[])
        )
    ]
    assert parts and gen_settings.generation.abstain_message in "".join(parts)


# ---------------------------------------------------------------------------
# Self-RAG and RRR
# ---------------------------------------------------------------------------
async def test_self_rag_accepts_a_good_first_answer(gen_settings: Settings) -> None:
    llm = StubLLM(
        responses={
            "GroundednessGrade": GroundednessGrade(grounded=True, score=0.95),
            "UtilityGrade": UtilityGrade(useful=True, score=0.9),
        }
    )
    chunks = [chunk("c1", POLICY)]

    async def retrieve(_query):
        return RetrievalResult(chunks=chunks)

    async def generate(_query, _retrieval):
        return Answer(text="Refunds take 14 days [1].", chunks=chunks)

    result = await SelfRAG(llm, gen_settings).run(Query(text="how long?"), retrieve, generate)
    assert not result.answer.abstained
    assert result.accepted_iteration == 0
    assert len(result.attempts) == 1


async def test_self_rag_abstains_after_exhausting_retries(gen_settings: Settings) -> None:
    """The loop must terminate in an abstention, not in the least-bad failure."""
    settings = gen_settings.model_copy(deep=True)
    settings.generation.self_rag_max_retries = 1
    llm = StubLLM(
        responses={
            "GroundednessGrade": GroundednessGrade(grounded=False, score=0.1),
            "UtilityGrade": UtilityGrade(useful=True, score=0.8),
            "RewriteOutput": RewriteOutput(rewritten_query="refund processing window"),
        }
    )
    chunks = [chunk("c1", POLICY)]

    async def retrieve(_query):
        return RetrievalResult(chunks=chunks)

    async def generate(_query, _retrieval):
        return Answer(text="Refunds are instant [1].", chunks=chunks)

    result = await SelfRAG(llm, settings).run(Query(text="how long?"), retrieve, generate)
    assert result.answer.abstained
    assert result.accepted_iteration == -1
    assert len(result.attempts) == 2
    assert "instant" not in result.answer.text


async def test_rrr_rewrites_before_retrieving(gen_settings: Settings) -> None:
    """RRR pays one cheap rewrite up front rather than a full generation to
    discover the query was bad."""
    llm = StubLLM(
        responses={"RewriteOutput": RewriteOutput(rewritten_query="refund processing window")}
    )
    seen: list[str] = []

    async def retrieve(query):
        seen.append(query.text)
        return RetrievalResult(chunks=[chunk("c1", POLICY)])

    result = await RRR(llm, gen_settings).run(
        Query(text="ugh so when do I actually get my money back??"), retrieve
    )
    assert seen[0] == "refund processing window"
    assert result.succeeded
    assert result.rewrites


async def test_rrr_retries_on_empty_retrieval(gen_settings: Settings) -> None:
    settings = gen_settings.model_copy(deep=True)
    settings.generation.rrr_max_rewrites = 2
    llm = StubLLM(responses={"RewriteOutput": RewriteOutput(rewritten_query="better query")})
    attempts = 0

    async def retrieve(_query):
        nonlocal attempts
        attempts += 1
        return RetrievalResult(chunks=[])

    result = await RRR(llm, settings).run(Query(text="obscure question"), retrieve)
    assert attempts == 3, "one initial attempt plus two rewrites"
    assert not result.succeeded


async def test_rrr_degrades_when_the_rewriter_fails(gen_settings: Settings) -> None:
    class Failing(StubLLM):
        async def structured(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("rewriter down")

    async def retrieve(query):
        return RetrievalResult(chunks=[chunk("c1", POLICY)])

    result = await RRR(Failing(), gen_settings).run(Query(text="original question"), retrieve)
    assert result.query.text == "original question", "must fall back to the original query"
    assert result.succeeded


# ---------------------------------------------------------------------------
# Every answer path honours the answer-length budget
# ---------------------------------------------------------------------------
async def test_every_answer_path_applies_max_answer_tokens(gen_settings: Settings) -> None:
    """Only the plain-text path passed `generation.max_answer_tokens`.

    The JSON-citation path and self-consistency ran at the global
    `llm.max_tokens`, and self-consistency is the path that *multiplies* — N
    samples per question — so the most expensive one was the uncapped one. It also
    breaks a coupling the settings maintain: `Settings.model_post_init` sizes
    `reserved_output_tokens` from `max_answer_tokens`, so the packer builds a
    prompt against a budget the generation then ignores.
    """
    cap = 321
    settings = gen_settings.model_copy(deep=True)
    settings.generation.max_answer_tokens = cap
    settings.llm.max_tokens = 8192
    evidence = RetrievalResult(chunks=[chunk("c1", POLICY)])

    async def caps_for(**generation: object) -> list[object]:
        scoped = settings.model_copy(deep=True)
        for key, value in generation.items():
            setattr(scoped.generation, key, value)
        llm = StubLLM(
            text="Refunds are processed within 14 days [1].",
            responses={"GroundednessGrade": GroundednessGrade(grounded=True, score=0.95)},
        )
        await AnswerGenerator(llm, scoped).generate(Query(text="how long?"), evidence)
        return [
            call.get("max_tokens")
            for call in llm.calls
            if call.get("stage") in {"answer", "answer_sample"}
        ]

    plain = await caps_for(citation_style="inline")
    json_cited = await caps_for(citation_style="json")
    consistent = await caps_for(self_consistency_samples=3)

    assert plain and all(c == cap for c in plain), plain
    assert json_cited and all(c == cap for c in json_cited), (
        f"the JSON-citation path must carry the cap too, saw {json_cited}"
    )
    assert consistent and all(c == cap for c in consistent), (
        f"every self-consistency sample must carry the cap, saw {consistent}"
    )


async def test_the_answer_path_hands_the_packer_its_per_source_floors(
    gen_settings: Settings,
) -> None:
    """`BudgetPlan.per_source` was computed on every request and read by nothing.

    Implementing the floor in `ContextPacker` is only half of it: the packer takes
    `shares` as an optional argument, so the generator forgetting to pass them
    would restore the original bug with every packer test still green. That is
    how the field went dead the first time, so the wiring gets its own assertion.
    """
    from ragorc.context.pack import ContextPacker

    seen: dict[str, object] = {}

    class _Recording(ContextPacker):
        def build(self, chunks, **kwargs):  # noqa: ANN001, ANN003, ANN201
            seen["shares"] = kwargs.get("shares")
            return super().build(chunks, **kwargs)

    llm = StubLLM(
        text="Refunds are processed within 14 days [1].",
        responses={"GroundednessGrade": GroundednessGrade(grounded=True, score=0.95)},
    )
    generator = AnswerGenerator(llm, gen_settings, packer=_Recording(gen_settings))

    await generator.generate(
        Query(text="how long do refunds take?"),
        RetrievalResult(chunks=[chunk("c1", POLICY)]),
    )

    shares = seen.get("shares")
    assert shares, f"the packer was given no per-source floors: {shares!r}"
    assert all(isinstance(v, int) and v >= 0 for v in shares.values()), shares  # type: ignore[union-attr]
