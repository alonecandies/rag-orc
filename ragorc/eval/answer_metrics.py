"""Answer-quality metrics: RAGAS-style judgements, plus a cheap lexical floor.

Retrieval metrics stop at "was the evidence found". These start where the
generator does, and they exist because the two failure modes of a RAG answer are
independent: an answer can be perfectly faithful to the context and answer the
wrong question, or address the question precisely and invent half its facts. One
score cannot see both, so there are five, and each one is designed so its *inputs*
are available in production — not just on a labelled test set.

| metric | asks | needs a reference answer? |
|---|---|---|
`faithfulness` | is every claim supported by the retrieved context? | no |
`answer_relevance` | does the answer address the question that was asked? | no |
`context_precision` | how much of the retrieved context was actually used? | no |
`context_recall` | how much of the true answer does the context contain? | yes |
`answer_correctness` | is the answer right? | yes |

The first three are computable on live traffic, which is the point: they are the
ones you can alert on. The last two need ground truth and belong to the offline
harness.

Every metric returns its **reasoning** alongside its score. A number without a
reason cannot be acted on — "faithfulness 0.6" tells you nothing, "these two
claims are unsupported" tells you whether the retrieval missed a passage or the
model overreached, which are different fixes.

Reuse, not reimplementation
---------------------------
``faithfulness`` is claim decomposition plus per-claim entailment against the
context, which is exactly what :class:`~ragorc.generate.groundedness.GroundednessChecker`
already does — including the concurrency, the fail-closed handling of an
unverifiable claim, and the local NLI path that needs no API call. This module
calls it instead of writing a second claim verifier: two implementations of the
same judgement would drift, and then the offline metric would stop predicting the
production gate that actually blocks answers.

``context_recall`` is the same computation with its arguments swapped. Faithfulness
asks "is the answer contained in the context"; context recall asks "is the
*reference* contained in the context". Both are entailment of a text against the
retrieved passages, so the checker is reused verbatim with the reference answer in
the answer position. Naming that symmetry is more useful than hiding it: if
context recall is high and faithfulness is low, the evidence was there and the
model failed to use it; if both are low, retrieval failed and the generator never
had a chance.

On the choice of judge model
---------------------------
The cost cascade (ADR-0005) sends graders to the cheap tier, and for the *request
path* that is right: a grader runs on every query and a wrong grade costs one
degraded answer. Here the trade inverts. Evaluation is offline, runs over a few
hundred cases, and its output is the number a configuration decision is made from.
A judge with 15% error injects more variance than the differences between two
configurations usually are, so the default judge is the balanced tier and
``model=`` pins it explicitly. Pin it in any experiment you intend to compare
across weeks: changing the judge silently re-scales every metric, and the diff
looks like a regression in the pipeline.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog
from pydantic import BaseModel, Field

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError
from ragorc.core.models import FloatArray, ScoredChunk, Usage
from ragorc.core.protocols import LLM, DenseEmbedder
from ragorc.core.settings import Settings, get_settings
from ragorc.generate.groundedness import GroundednessChecker
from ragorc.llm.prompts import Prompt, get_prompt, register_prompt
from ragorc.llm.router import ModelRouter, ModelTier

log = structlog.get_logger(__name__)

__all__ = [
    "ALL_METRICS",
    "AnswerMetrics",
    "MetricScore",
    "Scorecard",
    "cheap_baseline",
    "lexical_overlap",
    "rouge_l_f1",
    "token_f1",
]

ALL_METRICS: tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "lexical_overlap",
)

REVERSE_QUESTIONS = 3
"""How many questions to reconstruct from the answer for ``answer_relevance``.

One reconstruction is a coin flip on phrasing; three averages that away. Beyond
about five the marginal variance reduction is smaller than the added cost, and the
model starts producing near-identical questions whose mean is the same number."""

CORRECTNESS_FACTUALITY_WEIGHT = 0.75
CORRECTNESS_SIMILARITY_WEIGHT = 0.25
"""``answer_correctness`` is mostly statement-level F1 with a minority weight on
embedding similarity. Factuality has to dominate: an answer that is fluent,
on-topic and wrong is the single most dangerous output a RAG system produces, and
similarity alone scores it highly. Similarity is kept as a minority term because
pure F1 is brittle — it punishes a correct answer that decomposes into differently
shaped statements than the reference did."""

_MAX_LEXICAL_TOKENS = 800
"""Cap for the ROUGE-L longest-common-subsequence DP, which is O(n·m). Answers
longer than this are truncated: the marginal ranking signal in the tail is far
smaller than the quadratic cost of measuring it."""

_WORD = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
r"""Word runs in any script. ``[^\W_]`` is ``\w`` without the underscore.

It was ``[a-z0-9]+``, which made every non-ASCII character a separator: two
byte-identical Russian answers scored 0.0 — indistinguishable from a completely
wrong answer — ``café`` tokenized to ``caf`` and ``Müller`` to ``m`` + ``ller``.
These metrics decide whether a change helped, so mis-scoring most of Europe and
all of CJK is not a cosmetic problem."""

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
r"""Kana, Hangul and Han. These scripts do not put spaces between words, so a
whole clause matches ``_WORD`` as one token and lexical overlap collapses to
exact match. Splitting them per character is the usual cheap substitute for a
segmenter, and it is what gives partial credit at all here."""


# ---------------------------------------------------------------------------
# Structured output for the judges (local to the harness — see dataset.py)
# ---------------------------------------------------------------------------
class _ReverseQuestions(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Questions that this answer answers, written as if the answer were a "
            "document you had found. Each must be self-contained."
        ),
    )
    noncommittal: bool = Field(
        default=False,
        description=(
            "True if the answer declines to answer, says it does not know, or "
            "hedges without committing to any content."
        ),
    )


class _ContextUsage(BaseModel):
    used: bool = Field(
        description=(
            "True if this passage contributed information that appears in the "
            "answer. False if the answer would be unchanged without it."
        )
    )
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the judgement.")
    reason: str = Field(default="", max_length=300)


class _CorrectnessBreakdown(BaseModel):
    correct_statements: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Statements in the answer that the reference confirms.",
    )
    incorrect_statements: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Statements in the answer that the reference contradicts or does not contain.",
    )
    missing_statements: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Statements in the reference that the answer omits.",
    )
    reasoning: str = Field(default="", max_length=600)


register_prompt(
    Prompt(
        name="eval_reverse_questions",
        tags=("eval",),
        description="Reconstruct the questions an answer answers, for answer relevance.",
        system=(
            "You read an answer and write the questions it answers.\n\n"
            "Write each question as though you had found the answer as a document and "
            "were reconstructing what someone must have asked. Base them strictly on "
            "what the answer says — do not use outside knowledge and do not improve on "
            "it. If the answer covers several things, write a question for each.\n"
            "Set noncommittal=true when the answer declines, says it cannot find the "
            "information, or hedges without asserting anything."
        ),
        template="Answer:\n{answer}\n\nWrite {n} questions this answer answers.",
    )
)

register_prompt(
    Prompt(
        name="eval_context_usage",
        tags=("eval",),
        description="Judge whether a retrieved passage was actually used by the answer.",
        system=(
            "You judge whether a retrieved passage was USED to produce an answer.\n\n"
            "This is not a relevance judgement. A passage can be on-topic and still "
            "unused. Ask only: does information from this passage appear in the "
            "answer? If removing this passage would leave the answer unchanged, it was "
            "not used.\n"
            "A passage that merely repeats what another passage already supplied still "
            "counts as used only if the answer's wording or figures came from it."
        ),
        template=(
            "Question: {question}\n\nAnswer:\n{answer}\n\nPassage:\n{passage}\n\n"
            "Was this passage used?"
        ),
    )
)

register_prompt(
    Prompt(
        name="eval_answer_correctness",
        tags=("eval",),
        description="Compare an answer against a reference answer, statement by statement.",
        system=(
            "You compare an answer against a reference answer that is known to be "
            "correct, and classify statements rather than giving an overall grade — a "
            "holistic score is not reproducible and cannot be acted on.\n\n"
            "Split both texts into statements, then sort them:\n"
            "- correct_statements: in the answer AND confirmed by the reference.\n"
            "- incorrect_statements: in the answer but contradicted by the reference, "
            "or absent from it. A number, name or date that differs from the "
            "reference's belongs here, however close it is.\n"
            "- missing_statements: in the reference but absent from the answer.\n\n"
            "A different wording of the same fact is correct, not incorrect: judge "
            "meaning, not phrasing. Ignore differences in style, order and length."
        ),
        template=(
            "Question: {question}\n\nReference answer:\n{reference}\n\n"
            "Answer to grade:\n{answer}\n\nClassify the statements."
        ),
    )
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class MetricScore:
    """One metric's value, why it has that value, and what it cost.

    ``score`` is ``nan`` when the metric was not applicable (no reference answer,
    no retrieved chunks, no embedder) or when the judge failed. Not zero: a metric
    that could not be computed is not a metric that scored badly, and averaging
    those together is how a missing label becomes an apparent regression.
    """

    name: str
    score: float = float("nan")
    reasoning: str = ""
    usage: Usage = field(default_factory=Usage)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def computed(self) -> bool:
        return not np.isnan(self.score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": None if not self.computed else round(self.score, 4),
            "reasoning": self.reasoning,
            "cost_usd": round(self.usage.cost_usd, 6),
            "llm_calls": self.usage.calls,
            "detail": self.detail,
        }

    @classmethod
    def skipped(cls, name: str, reason: str) -> MetricScore:
        return cls(name=name, reasoning=f"not computed: {reason}")


@dataclass(slots=True)
class Scorecard:
    """Every metric computed for one answer."""

    scores: dict[str, MetricScore] = field(default_factory=dict)

    def __getitem__(self, name: str) -> MetricScore:
        return self.scores[name]

    def add(self, score: MetricScore) -> None:
        self.scores[score.name] = score

    @property
    def usage(self) -> Usage:
        return Usage.sum(score.usage for score in self.scores.values())

    def values(self) -> dict[str, float]:
        """Metric name -> score, computed metrics only."""
        return {name: s.score for name, s in self.scores.items() if s.computed}

    def reasons(self) -> dict[str, str]:
        return {name: s.reasoning for name, s in self.scores.items() if s.reasoning}

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {name: score.to_dict() for name, score in self.scores.items()},
            "cost_usd": round(self.usage.cost_usd, 6),
            "llm_calls": self.usage.calls,
        }


# ---------------------------------------------------------------------------
# The cheap baseline — no model, no API, no cost
# ---------------------------------------------------------------------------
def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for word in _WORD.findall(text.casefold()):
        # A token holding un-spaced script is split per character; anything else
        # stays whole, so "5日" gives ["5", "日"] and "café" stays "café".
        out.extend(list(word) if _CJK.search(word) else [word])
        if len(out) >= _MAX_LEXICAL_TOKENS:
            break
    return out[:_MAX_LEXICAL_TOKENS]


def token_f1(answer: str, reference: str) -> float:
    """Unigram-overlap F1 over a bag of tokens (the SQuAD-style measure).

    Multiset intersection, so a word repeated three times in the reference and once
    in the answer contributes one match rather than three — otherwise a padded
    answer that repeats a keyword scores arbitrarily high.
    """
    a, b = Counter(_tokens(answer)), Counter(_tokens(reference))
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length.

    The token-equality matrix is one vectorized comparison; the dynamic program
    itself is sequential in both axes, so exactly one axis is looped and the inner
    step reads a numpy row instead of comparing strings.
    """
    if not a or not b:
        return 0
    eq = np.asarray(a, dtype=object)[:, None] == np.asarray(b, dtype=object)[None, :]
    prev = np.zeros(len(b) + 1, dtype=np.int32)
    cur = np.zeros(len(b) + 1, dtype=np.int32)
    for row in eq:
        cur[0] = 0
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] + 1 if row[j - 1] else max(cur[j - 1], prev[j])
        prev, cur = cur, prev
    return int(prev[-1])


def rouge_l_f1(answer: str, reference: str) -> float:
    """ROUGE-L: F1 over the longest common subsequence.

    Order-sensitive where :func:`token_f1` is not, so it separates "the same facts
    in a different order" from "the same words in a scrambled sentence" — which is
    the difference between a paraphrase and a degenerate generation.
    """
    a, b = _tokens(answer), _tokens(reference)
    if not a or not b:
        return 0.0
    lcs = _lcs_length(a, b)
    if not lcs:
        return 0.0
    precision, recall = lcs / len(a), lcs / len(b)
    return 2 * precision * recall / (precision + recall)


def lexical_overlap(answer: str, reference: str) -> MetricScore:
    """Token-F1 and ROUGE-L against a reference. No model, no cost, no variance.

    **When to trust it: regression detection.** Held constant — same dataset, same
    reference answers, same prompt — this number moves only when the generated text
    moves. It is deterministic, so it has no judge variance at all, and it can run
    on every commit in CI where an LLM-judged suite cannot. A five-point drop across
    a hundred cases is a real change in behaviour and worth bisecting.

    **When not to trust it: absolute quality.** It measures word overlap, and the
    two ways that diverges from quality both matter. A correct answer phrased
    differently from the reference scores low ("the window is 30 days" vs "refunds
    are accepted within a month"). A wrong answer that reuses the reference's
    vocabulary scores high — and reusing retrieved vocabulary is exactly what a
    hallucinating RAG model does. So never report it as a quality figure, never
    compare it across datasets, and never let it settle an argument about whether
    an answer is right. That is what ``answer_correctness`` is for.
    """
    if not answer.strip() or not reference.strip():
        return MetricScore.skipped("lexical_overlap", "empty answer or reference")
    unigram = token_f1(answer, reference)
    rouge = rouge_l_f1(answer, reference)
    return MetricScore(
        name="lexical_overlap",
        score=(unigram + rouge) / 2.0,
        reasoning=(
            f"token-F1 {unigram:.3f}, ROUGE-L {rouge:.3f} — lexical only; "
            "use for regression detection, not for absolute quality"
        ),
        detail={"token_f1": round(unigram, 4), "rouge_l": round(rouge, 4)},
    )


# ---------------------------------------------------------------------------
# The judged metrics
# ---------------------------------------------------------------------------
class AnswerMetrics:
    """RAGAS-style judged metrics over one answer at a time.

    Collaborators are injected: the same LLM, embedder and groundedness checker the
    pipeline uses can be handed straight in, which keeps the offline metric
    measuring the same thing the production gate measures.
    """

    name = "answer_metrics"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        embedder: DenseEmbedder | None = None,
        router: ModelRouter | None = None,
        grounding: GroundednessChecker | None = None,
        model: str | None = None,
        reverse_questions: int = REVERSE_QUESTIONS,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self.embedder = embedder
        self.grounding = grounding or GroundednessChecker(llm, self.settings, router=self.router)
        self.model = model or self.router.model_for_tier(ModelTier.BALANCED)
        self.reverse_questions = max(int(reverse_questions), 1)

    # -- faithfulness ------------------------------------------------------
    async def faithfulness(
        self, question: str, answer: str, chunks: Sequence[ScoredChunk]
    ) -> MetricScore:
        """Fraction of the answer's claims that the retrieved context supports.

        Delegates entirely to :class:`GroundednessChecker` — see the module
        docstring for why this must not be a second implementation. ``method="both"``
        forces the claim-decomposition path rather than a single holistic grade: a
        whole-answer verdict anchors on plausibility and misses the two failures that
        matter most, a composed causal claim and a number that is close but wrong.
        """
        if not chunks:
            return MetricScore.skipped("faithfulness", "no retrieved context")
        if not answer.strip():
            return MetricScore.skipped("faithfulness", "empty answer")
        result = await self.grounding.check(question, answer, chunks, method="both")
        unsupported = result.unsupported + result.contradicted
        reasoning = (
            "every claim is supported by the context"
            if not unsupported
            else f"{len(unsupported)} claim(s) unsupported: " + "; ".join(unsupported[:3])
        )
        return MetricScore(
            name="faithfulness",
            score=float(result.score),
            reasoning=reasoning,
            usage=result.usage,
            detail={
                "claims": len(result.claims),
                "supported": sum(1 for c in result.claims if c.supported),
                "contradicted": result.contradicted,
                "unsupported": result.unsupported,
                "method": result.method,
                "grounded": result.grounded,
            },
        )

    # -- answer relevance --------------------------------------------------
    async def answer_relevance(self, question: str, answer: str) -> MetricScore:
        """Does the answer address the question that was asked?

        The trick: ask the model to reconstruct the questions this answer answers,
        embed those reconstructions, and compare them to the real question. If the
        answer addresses the question, the question is recoverable from it and the
        reconstructions land in the same neighbourhood; if the answer drifted onto an
        adjacent topic, padded with background, or answered a question the user did
        not ask, the reconstructions land somewhere else.

        **Why no ground truth is needed.** Nothing here is compared against a correct
        answer — the comparison is between the *question* and a round trip through
        the answer. Both ends are already in hand at request time, which is what
        makes this the one quality metric that can run on live traffic and be
        alerted on. It is genuinely question-answer alignment, not correctness: a
        confidently wrong answer to the right question scores high here, and that is
        the division of labour with ``faithfulness``.

        An answer that declines to answer is scored 0.0 and flagged noncommittal.
        That is not a claim that abstaining is bad — abstention is a success state in
        this library — but it *is* zero relevance, and the flag is there so the
        harness reports abstentions separately instead of blending them into the mean.
        """
        if self.embedder is None:
            raise ConfigError(
                "answer_relevance compares embeddings and needs embedder=...",
                hint="pass the pipeline's DenseEmbedder to AnswerMetrics",
            )
        if not answer.strip():
            return MetricScore.skipped("answer_relevance", "empty answer")

        prompt = get_prompt("eval_reverse_questions")
        result, usage = await self.llm.structured(
            prompt.render(answer=answer, n=self.reverse_questions),
            _ReverseQuestions,
            system=prompt.system,
            model=self.model,
            stage="eval_answer_relevance",
        )
        if result.noncommittal:
            return MetricScore(
                name="answer_relevance",
                score=0.0,
                reasoning="answer is noncommittal (declined or hedged without content)",
                usage=usage,
                detail={"noncommittal": True},
            )

        generated = [q.strip() for q in result.questions if q.strip()][: self.reverse_questions]
        if not generated:
            return MetricScore(
                name="answer_relevance",
                score=float("nan"),
                reasoning="judge returned no reconstructed questions",
                usage=usage,
            )

        # One batched embed call for the question and every reconstruction: N+1
        # round trips here would dominate the metric's wall time on a 500-case run.
        vectors = await self.embedder.embed_queries([question, *generated])
        matrix = np.asarray(vectors, dtype=np.float32)
        similarities = _cosine_against_first(matrix)
        score = float(np.clip(similarities.mean(), 0.0, 1.0))
        return MetricScore(
            name="answer_relevance",
            score=score,
            reasoning=(
                f"{len(generated)} reconstructed question(s), mean cosine to the "
                f"original {score:.3f}; closest: {generated[int(np.argmax(similarities))][:120]!r}"
            ),
            usage=usage,
            detail={
                "questions": generated,
                "similarities": [round(float(s), 4) for s in similarities],
                "noncommittal": False,
            },
        )

    # -- context precision -------------------------------------------------
    async def context_precision(
        self, question: str, answer: str, chunks: Sequence[ScoredChunk]
    ) -> MetricScore:
        """Fraction of the retrieved passages that the answer actually used.

        Unused context is not free. Every passage in the prompt costs tokens, adds
        latency, and — because attention over long contexts is U-shaped — pushes the
        passages that *are* load-bearing toward the middle where the model attends to
        them least. A pipeline at 0.3 context precision is paying for ten passages to
        use three, and lowering ``top_k`` will usually make its answers better *and*
        cheaper.

        The ``rank_weighted`` figure in ``detail`` is the rank-sensitive variant
        (mean precision at each used position, the same computation as average
        precision with the judge's verdicts standing in for labels). Compare the two:
        equal values mean the used passages were spread through the window, while a
        much higher rank-weighted value means the reranker put the useful ones first
        and ``top_k`` is simply too wide.
        """
        if not chunks:
            return MetricScore.skipped("context_precision", "no retrieved context")
        if not answer.strip():
            return MetricScore.skipped("context_precision", "empty answer")

        prompt = get_prompt("eval_context_usage")

        async def judge(scored: ScoredChunk) -> tuple[_ContextUsage, Usage]:
            return await self.llm.structured(
                prompt.render(question=question, answer=answer, passage=scored.chunk.content),
                _ContextUsage,
                system=prompt.system,
                model=self.model,
                stage="eval_context_precision",
            )

        outcomes = await bounded_gather(
            (judge(chunk) for chunk in chunks),
            limit=self.settings.llm.max_concurrency,
            return_exceptions=True,
        )

        used: list[bool] = []
        usages: list[Usage] = []
        reasons: list[str] = []
        failures = 0
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                # A failed judgement is not a verdict. Counting it as "unused" would
                # make an unreliable judge look like a wasteful retriever.
                failures += 1
                used.append(False)
                reasons.append("(judge failed)")
                continue
            verdict, usage = outcome
            usages.append(usage)
            used.append(bool(verdict.used))
            reasons.append(verdict.reason[:120])

        flags = np.asarray(used, dtype=bool)
        judged = len(flags) - failures
        if judged <= 0:
            return MetricScore(
                name="context_precision",
                score=float("nan"),
                reasoning="every context-usage judgement failed",
                usage=Usage.sum(usages),
                detail={"failures": failures},
            )

        utilization = float(flags.sum()) / float(judged)
        positions = np.arange(1, len(flags) + 1, dtype=np.float64)
        precision_at_i = np.cumsum(flags, dtype=np.float64) / positions
        hits = int(flags.sum())
        rank_weighted = float((precision_at_i * flags).sum() / hits) if hits else 0.0
        return MetricScore(
            name="context_precision",
            score=utilization,
            reasoning=(
                f"{hits}/{judged} passage(s) used"
                + (f", {failures} judgement(s) failed" if failures else "")
                + f"; rank-weighted {rank_weighted:.3f}"
            ),
            usage=Usage.sum(usages),
            detail={
                "used": hits,
                "judged": judged,
                "failures": failures,
                "rank_weighted": round(rank_weighted, 4),
                "per_chunk": [
                    {"chunk_id": c.chunk.id, "used": bool(u), "reason": r}
                    for c, u, r in zip(chunks, used, reasons, strict=True)
                ],
            },
        )

    # -- context recall ----------------------------------------------------
    async def context_recall(
        self, reference: str, chunks: Sequence[ScoredChunk], *, question: str = ""
    ) -> MetricScore:
        """Fraction of the reference answer that the retrieved context covers.

        The mirror image of faithfulness, and computed by the same verifier with the
        reference in the answer position: decompose the known-correct answer into
        claims, then check each claim against the retrieved passages. What comes back
        is how much of the answer retrieval actually made available.

        This is the metric that assigns blame. A low answer score with high context
        recall is a *generation* failure — the evidence was in the prompt and the
        model did not use it, so look at the prompt, the packing order and the
        context budget. A low answer score with low context recall is a *retrieval*
        failure, and no amount of prompt work will fix it.
        """
        if not chunks:
            return MetricScore.skipped("context_recall", "no retrieved context")
        if not reference.strip():
            return MetricScore.skipped("context_recall", "no reference answer")
        result = await self.grounding.check(
            question or "(context recall)", reference, chunks, method="both"
        )
        missing = result.unsupported + result.contradicted
        reasoning = (
            "the context covers the whole reference answer"
            if not missing
            else f"{len(missing)} reference claim(s) absent from the context: "
            + "; ".join(missing[:3])
        )
        return MetricScore(
            name="context_recall",
            score=float(result.score),
            reasoning=reasoning,
            usage=result.usage,
            detail={
                "reference_claims": len(result.claims),
                "covered": sum(1 for c in result.claims if c.supported),
                "missing": result.unsupported,
                "contradicted": result.contradicted,
            },
        )

    # -- answer correctness ------------------------------------------------
    async def answer_correctness(self, question: str, answer: str, reference: str) -> MetricScore:
        """Is the answer right, judged against a reference answer?

        Statement-level F1 over the judge's classification — correct statements are
        true positives, statements the reference does not support are false
        positives, and statements the reference has that the answer lacks are false
        negatives — blended with embedding similarity at
        :data:`CORRECTNESS_SIMILARITY_WEIGHT`.

        F1 rather than a 1-5 score because a holistic rating is neither reproducible
        across judge versions nor decomposable: F1 separates *wrong* (false
        positives, the dangerous failure) from *incomplete* (false negatives, usually
        a retrieval gap), and those need different fixes. Both counts are in
        ``detail`` so the aggregate can be re-derived without re-running the judge.
        """
        if not reference.strip():
            return MetricScore.skipped("answer_correctness", "no reference answer")
        if not answer.strip():
            return MetricScore.skipped("answer_correctness", "empty answer")

        prompt = get_prompt("eval_answer_correctness")
        breakdown, usage = await self.llm.structured(
            prompt.render(question=question, reference=reference, answer=answer),
            _CorrectnessBreakdown,
            system=prompt.system,
            model=self.model,
            stage="eval_answer_correctness",
        )
        tp = len(breakdown.correct_statements)
        fp = len(breakdown.incorrect_statements)
        fn = len(breakdown.missing_statements)
        denominator = 2 * tp + fp + fn
        factuality = (2 * tp / denominator) if denominator else 0.0

        similarity = float("nan")
        if self.embedder is not None:
            try:
                vectors = await self.embedder.embed_queries([reference, answer])
                similarity = float(np.clip(_cosine_against_first(np.asarray(vectors))[0], 0.0, 1.0))
            except Exception as exc:  # noqa: BLE001 - similarity is the minority term
                log.warning("eval_correctness_similarity_failed", error=str(exc)[:160])

        if np.isnan(similarity):
            score = factuality
            blend = "factuality only (no embedder)"
        else:
            score = (
                CORRECTNESS_FACTUALITY_WEIGHT * factuality
                + CORRECTNESS_SIMILARITY_WEIGHT * similarity
            )
            blend = (
                f"{CORRECTNESS_FACTUALITY_WEIGHT:.2f}·factuality "
                f"{factuality:.3f} + {CORRECTNESS_SIMILARITY_WEIGHT:.2f}·similarity "
                f"{similarity:.3f}"
            )
        reasoning = f"TP {tp} / FP {fp} / FN {fn} — {blend}"
        if breakdown.reasoning:
            reasoning = f"{reasoning}. {breakdown.reasoning[:300]}"
        return MetricScore(
            name="answer_correctness",
            score=float(score),
            reasoning=reasoning,
            usage=usage,
            detail={
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "factuality_f1": round(factuality, 4),
                "similarity": None if np.isnan(similarity) else round(similarity, 4),
                "incorrect": breakdown.incorrect_statements[:5],
                "missing": breakdown.missing_statements[:5],
            },
        )

    # -- everything at once ------------------------------------------------
    async def evaluate(
        self,
        question: str,
        answer: str,
        chunks: Sequence[ScoredChunk],
        *,
        reference: str = "",
        metrics: Sequence[str] = ALL_METRICS,
    ) -> Scorecard:
        """Run the requested metrics concurrently and collect them.

        Concurrently because they are independent judgements over the same text, so
        serializing them multiplies the wall time of a run by the number of metrics
        for no benefit. Bounded by ``llm.max_concurrency`` like every other fan-out
        in this library — note that ``faithfulness`` and ``context_precision`` each
        fan out internally as well, so the real ceiling is nested and the semaphore
        is what keeps a 500-case run from opening 5,000 sockets.
        """
        wanted = [m for m in metrics if m in ALL_METRICS]
        card = Scorecard()

        jobs: dict[str, Any] = {}
        if "faithfulness" in wanted:
            jobs["faithfulness"] = self.faithfulness(question, answer, chunks)
        if "answer_relevance" in wanted and self.embedder is not None:
            jobs["answer_relevance"] = self.answer_relevance(question, answer)
        elif "answer_relevance" in wanted:
            card.add(MetricScore.skipped("answer_relevance", "no embedder configured"))
        if "context_precision" in wanted:
            jobs["context_precision"] = self.context_precision(question, answer, chunks)
        if "context_recall" in wanted:
            jobs["context_recall"] = self.context_recall(reference, chunks, question=question)
        if "answer_correctness" in wanted:
            jobs["answer_correctness"] = self.answer_correctness(question, answer, reference)

        if "lexical_overlap" in wanted:
            # Deterministic and free, so it gets no slot in the LLM fan-out — but it
            # still goes to a thread, because its LCS is quadratic CPU and a
            # coroutine that computes rather than awaits stalls every case in flight.
            card.add(await cheap_baseline(answer, reference))

        if jobs:
            names = list(jobs)
            outcomes = await bounded_gather(
                (jobs[name] for name in names),
                limit=max(self.settings.llm.max_concurrency // 2, 1),
                return_exceptions=True,
            )
            for name, outcome in zip(names, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    log.warning("eval_metric_failed", metric=name, error=str(outcome)[:200])
                    card.add(
                        MetricScore(
                            name=name,
                            reasoning=f"judge failed: {type(outcome).__name__}: {outcome}"[:300],
                        )
                    )
                else:
                    card.add(outcome)
        return card


def _cosine_against_first(matrix: FloatArray) -> FloatArray:
    """Cosine of row 0 against every other row, in one matvec.

    Rows are normalized here rather than assumed normalized: ``normalize`` is a
    per-provider setting, and a silently unnormalized dot product returns a
    magnitude, not a similarity, which would push this metric above 1.0.
    """
    data = np.asarray(matrix, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] < 2:
        return np.zeros(0, dtype=np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / np.maximum(norms, 1e-9)
    return (normalized[1:] @ normalized[0]).astype(np.float32)


async def cheap_baseline(answer: str, reference: str) -> MetricScore:
    """Async wrapper for :func:`lexical_overlap`, off the event loop.

    The LCS is quadratic in token count, so on a long answer this is milliseconds of
    pure CPU — small, but the runner calls it once per case while other cases are in
    flight, and a coroutine that computes rather than awaits stalls all of them.
    """
    return await asyncio.to_thread(lexical_overlap, answer, reference)
