"""Corrective RAG — grade what was retrieved, then act on the grade.

The failure this exists to fix
------------------------------
A retriever always returns its top-k, whether or not the corpus contains the
answer. Similarity is *relative*: on a question the corpus cannot answer, the ten
nearest neighbours still come back with respectable scores, and a generator that
has been told to answer from the context duly answers from documents about
something adjacent. Score thresholds do not catch this, because the scores are
not the problem — they are perfectly good scores relative to a corpus that has
nothing to say.

CRAG (Yan et al., 2024) inserts a decision between retrieval and generation:
grade the documents with a cheap model, then branch.

===========  ==================================  =============================
 label        condition                           action
===========  ==================================  =============================
 CORRECT      every graded document is relevant   refine them, use them
 AMBIGUOUS    some are, some are not              use both corpus *and* web
 INCORRECT    none are                            discard, rewrite, go to web
===========  ==================================  =============================

The paper uses two thresholds (an upper one for CORRECT, a lower one for
INCORRECT, ambiguous in between); ``crag_relevance_threshold`` is a single knob,
so unanimity stands in for the upper threshold. That keeps the interesting case
honest: "some documents are relevant" is *not* the same situation as "all of
them are", and treating it as CORRECT is what makes a half-answered question
look answered.

The strip step, which most implementations skip
-----------------------------------------------
"Relevant document" is a coarse judgement. A 500-token chunk usually earns its
relevance with two sentences and spends the rest on something else, and that
remainder is not free: it occupies context the answer needed, it dilutes the
attention over the part that mattered, and — worst — it is *topical* noise, the
kind a generator is most likely to weave into an answer because it reads like
support. So the kept documents are cut into strips, the strips are graded with
the same evaluator, and only the surviving strips are reassembled. Knowledge
refinement is the difference between "we found the right document" and "we sent
the model the right sentences", and it is the step that makes CORRECT worth
distinguishing from plain retrieval.

Reassembly keeps the surviving text verbatim and joins non-adjacent survivors
with a blank line. Verbatim because citation verification checks quoted spans
against chunk content, and a paraphrase there turns every citation into a
mismatch; the blank line because it marks the elision without inserting a token
that could appear inside a quote.

The split itself is not reimplemented: strips are cut with
:func:`ragorc.index.split.base.split_sentences`, the same segmenter
:mod:`ragorc.retrieve.compress` uses, so "where does a sentence end" has exactly
one answer in this codebase. What the strip step adds on top of it is the
*grading*, and that is also why the sentence-level compressor is offered as an
alternative rather than used as the default: pass
``compressor=SentenceLevelCompressor(...)`` and refinement costs one embedding
batch instead of a model call per strip, but it then selects sentences by
similarity to the query — the same signal that retrieved the chunk in the first
place, so a cheaper opinion rather than an independent one — and keeps a fixed
``compression_ratio`` rather than keeping what a judge called relevant.

What it costs, and why the cheap tier is mandatory
-------------------------------------------------
Grading N documents is N model calls, and refining them is another call per
strip. That is 5 documents plus ~20 strips = 25 calls on top of retrieval for a
single query. Three things keep it viable, and all three are load-bearing rather
than decorative:

* **The cheap tier.** ``Task.GRADE_RELEVANCE`` routes to ``fast_model``, which is
  20-50x cheaper than the synthesis model. Grading on a frontier model costs more
  than the answer it is protecting; this is the single most expensive mistake
  available in this file.
* **Hard caps.** Only ``crag_grade_top_k`` documents are graded, and only the
  documents that survived grading are stripped. The fan-out is bounded by the
  LLM's own concurrency semaphore, and the cost ledger's ceiling is checked
  before every call — a runaway loop raises ``BudgetExceeded`` instead of quietly
  spending money.
* **A fan-out sized to the budget that is left.** The ledger's ceiling is a
  *raise*, and a raise in the middle of a fan-out discards the request: with the
  shipped ``cost.max_llm_calls_per_query = 40``, grading plus refinement plus a
  rewrite is enough on its own to reach it once a Self-RAG retry re-enters this
  stage, and ``BudgetExceeded`` then leaves ``query()`` with no answer and no
  abstention. So every fan-out here first asks the ambient ledger how much room
  remains (:func:`_call_allowance`) and shrinks — grade fewer documents, refine
  fewer of them, skip the rewrite, in that order of preference — rather than
  walking into the ceiling. This is the same anticipate-the-hard-stop contract
  :mod:`ragorc.pipeline.graphs.agentic` applies to its loop predicates, which
  cannot see inside this stage.
* **Gates.** ``crag_enabled`` turns the whole stage off; ``crag_web_fallback``
  turns off just the external search, in which case the pipeline degrades to
  "grade and refine what the corpus gave us".

Every failure here degrades rather than raises. A dead corpus reads as a corpus
with nothing relevant (which CRAG already knows how to handle), a dead web
provider leaves the corpus results in place, and a grader outage produces *no*
verdict at all — the ranking passes through untouched with ``grade`` unset,
because an outage is not evidence about the documents and pretending otherwise
would send every query to the web the moment the grader wobbles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import LLMError, StoreUnavailable, TransientError
from ragorc.core.models import GradeLabel, Query, RetrievalResult, ScoredChunk, Usage
from ragorc.core.protocols import LLM, BatchStructuredLLM, Compressor, Retriever
from ragorc.core.registry import register
from ragorc.core.schemas import RelevanceGrade, RewriteOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import current_ledger, timed, trace_step
from ragorc.core.tokens import count_tokens, truncate_to_tokens
from ragorc.index.split.base import split_sentences
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.retrieve.web import NullWebRetriever, make_web_retriever
from ragorc.security.injection import render_untrusted_passages

log = structlog.get_logger(__name__)

__all__ = ["CorrectiveRAG", "CragDecision"]

_REWRITE_PREVIEW_TOKENS = 400
"""How much of the rejected evidence the rewriter gets to see.

The rewriter needs to know *why* the search missed — wrong vocabulary, wrong
entity, too narrow — and that is visible in the first few hundred tokens of what
came back. Sending all of it would add a multiple of the grading cost to every
corrective query and change the rewrite not at all.
"""


_RESERVED_CALLS = 4
"""Calls held back for everything that runs *after* this stage: synthesis, answer
verification, and one spare.

Duplicated from :mod:`ragorc.pipeline.graphs.agentic` rather than imported —
``retrieve`` must not depend on ``pipeline`` — and reserved rather than measured
for the reason given there: the epilogue's exact cost depends on settings (claim
decomposition, self-consistency samples), and the failure this guards against is
precisely the one under-reserving causes, a stage that spends the calls the answer
needed and ends the request in a traceback.
"""


def _call_allowance(*, reserve: int = _RESERVED_CALLS) -> int | None:
    """How many LLM calls this stage may still make. ``None`` means unbounded.

    Read from the request-scoped ledger through the contextvar rather than taken
    as an argument: the ledger is installed per request by
    :func:`~ragorc.core.telemetry.new_request_context`, and threading it through
    the :class:`~ragorc.core.protocols.Retriever` signature would be a second
    channel that can disagree with the one the LLM client already consults.

    No configured ceiling means no limit. An unbounded run is a deliberate
    configuration and inventing a cap here would silently override it.
    """
    ledger = current_ledger()
    if ledger is None or ledger.max_calls is None:
        return None
    return max(ledger.max_calls - ledger.total.calls - reserve, 0)


def _relevance(grade: RelevanceGrade) -> float:
    """Collapse the grader's two fields into one relevance number in [0, 1].

    ``relevant`` is the required field and ``score`` is optional, so a model that
    answers ``{"relevant": true}`` and omits the score inherits pydantic's
    default of 0.0. Thresholding on the score alone would then read "definitely
    relevant" as "not relevant at all" and push every query into the web branch —
    a silent, total misroute caused by an absent optional field.

    ``score`` also measures confidence *in the judgement*, not relevance, so it
    only modulates a positive verdict. High confidence in "not relevant" means
    less relevant, not more.
    """
    if not grade.relevant:
        return 0.0
    return min(float(grade.score), 1.0) if grade.score > 0.0 else 1.0


@dataclass(slots=True)
class _Graded:
    """A retrieved document plus the verdict on it."""

    scored: ScoredChunk
    relevance: float = 0.0
    relevant: bool = False
    reason: str = ""
    graded: bool = True
    """False when the grader call failed. Distinct from ``relevant=False``: one is
    a judgement, the other is the absence of one, and conflating them is how a
    provider blip becomes a corpus-wide verdict."""


@dataclass(slots=True)
class CragDecision:
    """Why CRAG did what it did — the record that goes on the request trace."""

    action: GradeLabel | None = None
    """``None`` means no decision was reachable (no grade was obtained)."""
    threshold: float = 0.0
    graded: int = 0
    relevant: int = 0
    irrelevant: int = 0
    ungraded: int = 0
    confidences: list[float] = field(default_factory=list)
    refined_documents: int = 0
    strips_kept: int = 0
    strips_total: int = 0
    rewritten_query: str | None = None
    web_results: int = 0
    calls: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def max_confidence(self) -> float:
        return max(self.confidences, default=0.0)

    @property
    def strip_retention(self) -> float:
        """Fraction of graded strips kept. Near 1.0 means refinement is not
        earning its calls on this corpus; near 0.2 means the chunks are far too
        big for the questions being asked."""
        return self.strips_kept / self.strips_total if self.strips_total else 1.0

    def report(self) -> dict[str, Any]:
        """Flat, log-safe summary. No document text: this ends up in the trace."""
        return {
            "action": self.action.value if self.action else "undecided",
            "threshold": self.threshold,
            "graded": self.graded,
            "relevant": self.relevant,
            "irrelevant": self.irrelevant,
            "ungraded": self.ungraded,
            "max_confidence": round(self.max_confidence, 3),
            "refined_documents": self.refined_documents,
            "strips_kept": self.strips_kept,
            "strips_total": self.strips_total,
            "strip_retention": round(self.strip_retention, 3),
            "rewritten": bool(self.rewritten_query),
            "web_results": self.web_results,
            "grader_calls": self.calls,
            "errors": dict(self.errors),
        }


@register("retriever", "crag")
class CorrectiveRAG:
    """Wraps another retriever with grading, knowledge refinement and web fallback."""

    name = "crag"

    def __init__(
        self,
        base: Retriever,
        llm: LLM,
        settings: Settings | None = None,
        *,
        web: Retriever | None = None,
        router: ModelRouter | None = None,
        compressor: Compressor | None = None,
        refine: bool = True,
    ) -> None:
        self.base = base
        self.llm = llm
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self.refine = refine
        # Optional replacement for strip grading — see the module docstring for
        # the trade-off. ``None`` keeps the paper's behaviour.
        self.compressor = compressor
        # Built here so a missing ``[web]`` extra fails at startup with a pip
        # hint instead of mid-query on the one question the corpus could not
        # answer. With the fallback switched off the null retriever is used and
        # the extra is never needed.
        if web is not None:
            self.web: Retriever = web
        elif self.settings.retrieval.crag_web_fallback:
            self.web = make_web_retriever(self.settings)
        else:
            self.web = NullWebRetriever(self.settings)

    # -- Retriever protocol -------------------------------------------------
    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """:class:`~ragorc.core.protocols.Retriever` entry point.

        The protocol has nowhere to put a grade or a bill, so both are recorded
        as side effects: the grade on the trace and on ``query.metadata["crag"]``,
        the cost on the request ledger. Call :meth:`run` when the caller needs
        them as values — the pipeline does, because it branches on the grade.
        """
        result, _usage = await self.run(query, top_k=top_k, **kwargs)
        return result.chunks

    async def run(
        self, query: Query, *, top_k: int | None = None, web: bool = True, **kwargs: Any
    ) -> tuple[RetrievalResult, Usage]:
        """Retrieve, grade, refine, optionally search the web, and report.

        ``web=False`` leaves the fallback to the caller. The ``crag`` and
        ``agentic`` graphs have their own ``web_search`` node — ``decide_after_grade``
        routes AMBIGUOUS to it — so with the internal leg also firing, every
        AMBIGUOUS query searched the web twice and every INCORRECT iteration did
        it again: two rewrite calls, two provider requests, two sets of results
        fused as though they were independent evidence.

        Defaulting to ``True`` keeps the linear engine — which has no graph nodes
        and genuinely owns this decision itself — working unchanged.
        """
        rs = self.settings.retrieval
        limit = int(top_k or query.top_k or rs.top_k)
        result = RetrievalResult()
        decision = CragDecision(threshold=rs.crag_relevance_threshold)
        usages: list[Usage] = []

        with timed("crag_retrieve") as clock:
            initial = await self._initial(query, limit, result, decision, kwargs)
        result.timings_ms["retrieve"] = clock.elapsed_ms
        result.per_store[self.base.name] = list(initial)
        result.total_candidates = len(initial)

        head = initial[: max(rs.crag_grade_top_k, 0)]
        allowance = _call_allowance()
        starved = allowance is not None and allowance < len(head)
        if starved:
            # Grade what the budget affords rather than raising in the middle of
            # the fan-out. The documents that fall out of the head are not lost:
            # they join the ungraded tail below and are still assembled into the
            # result, marked as never judged.
            log.info("crag_grade_budget_capped", requested=len(head), allowed=allowance)
            decision.errors["crag_budget"] = (
                f"call budget allowed grading {allowance} of {len(head)} documents"
            )
            head = head[:allowance]
        tail = initial[len(head) :]
        for scored in tail:
            # Ranked below the grading cap, so never judged. Marked rather than
            # silently mixed in with documents that were.
            scored.explain.setdefault("crag_graded", False)

        if starved and not head:
            # Not one call left. Make no claim: leave the ranking alone and leave
            # ``result.grade`` unset, exactly as a total grader outage does below.
            # An exhausted budget is not evidence about the documents, and the
            # INCORRECT label a zero-length grade otherwise produces routes to the
            # web — one *more* call, on the strength of no verdict at all. Gated on
            # ``starved`` rather than on ``not head`` so a deliberate
            # ``crag_grade_top_k = 0`` keeps behaving as it always has.
            log.warning("crag_skipped_no_budget", candidates=len(initial))
            result.chunks = self._rank(initial[:limit])
            return self._finish(query, result, decision, usages)

        with timed("crag_grade", documents=len(head)) as clock:
            graded, grade_usage = await self._grade_documents(query, head)
        result.timings_ms["grade"] = clock.elapsed_ms
        usages.append(grade_usage)

        decision.graded = sum(1 for g in graded if g.graded)
        decision.ungraded = len(initial) - decision.graded
        decision.confidences = [round(g.relevance, 3) for g in graded if g.graded]

        if head and decision.graded == 0:
            # Every grader call failed. Make no claim: leave the ranking alone,
            # leave ``result.grade`` unset, and record why. Calling this INCORRECT
            # would route the whole corpus to the web because a provider blipped.
            message = "grader returned no usable verdict"
            decision.errors["crag_grader"] = message
            result.chunks = self._rank(initial[:limit])
            return self._finish(query, result, decision, usages)

        relevant = [g for g in graded if g.relevant]
        irrelevant = [g for g in graded if g.graded and not g.relevant]
        decision.relevant = len(relevant)
        decision.irrelevant = len(irrelevant)
        decision.action = self._label(len(relevant), decision.graded)
        result.grade = decision.action

        refined: list[ScoredChunk] = []
        if decision.action is not GradeLabel.INCORRECT:
            with timed("crag_refine", documents=len(relevant)) as clock:
                refined, refine_usage = await self._refine(query, relevant, decision)
            result.timings_ms["refine"] = clock.elapsed_ms
            usages.append(refine_usage)

        web_chunks: list[ScoredChunk] = []
        if self._web_wanted(decision.action, allowed=web):
            with timed("crag_web") as clock:
                web_chunks, web_usage = await self._web(query, irrelevant, result, decision)
            result.timings_ms["web"] = clock.elapsed_ms
            usages.append(web_usage)
        if web_chunks:
            result.per_store["web"] = list(web_chunks)

        result.chunks = self._assemble(decision.action, refined, tail, web_chunks, limit)
        result.total_candidates += len(web_chunks)
        return self._finish(query, result, decision, usages)

    # -- stages -------------------------------------------------------------
    async def _initial(
        self,
        query: Query,
        limit: int,
        result: RetrievalResult,
        decision: CragDecision,
        kwargs: dict[str, Any],
    ) -> list[ScoredChunk]:
        """First-stage retrieval, fetched wide.

        ``fetch_k`` rather than ``top_k``: grading can only reject documents, so
        anything the first stage misses is lost for good, and CRAG's whole job is
        to notice that the top of the ranking is wrong.
        """
        rs = self.settings.retrieval
        fetch = max(rs.fetch_k, limit, rs.crag_grade_top_k)
        try:
            return list(await self.base.retrieve(query, top_k=fetch, **kwargs))
        except StoreUnavailable as exc:
            # Degrade, do not fail. A dead store is indistinguishable from a
            # store with nothing relevant, and CRAG already has a branch for
            # that: rewrite and go outside. Recorded on the decision; _finish
            # copies it onto the result's diagnostics.
            decision.errors[self.base.name] = str(exc)
            log.warning("crag_base_unavailable", retriever=self.base.name, error=str(exc)[:200])
            return []

    async def _grade_documents(
        self, query: Query, head: Sequence[ScoredChunk]
    ) -> tuple[list[_Graded], Usage]:
        """Grade the capped head of the ranking, concurrently.

        Documents are sent whole. Chunks are already bounded by the splitter's
        ``max_chunk_size``, so a grading prompt cannot blow up, and truncating
        here would risk cutting away the very passage that makes the document
        relevant — the grader would then be right about the wrong text.

        Grading is against ``query.original``, not ``query.text``: by the time
        retrieval runs, the text may be a HyDE pseudo-document or a step-back
        generalization, and relevance to a generated variant is not the question
        anyone asked.
        """
        if not head:
            return [], Usage()

        question = query.original or query.text
        grades, usage = await self._grade_texts(
            question, [s.chunk.content for s in head], stage="crag_grade"
        )
        threshold = self.settings.retrieval.crag_relevance_threshold

        out: list[_Graded] = []
        for scored, grade in zip(head, grades, strict=True):
            if grade is None:
                out.append(_Graded(scored=scored, graded=False))
                continue
            confidence = _relevance(grade)
            scored.component_scores["crag_relevance"] = confidence
            scored.explain["crag_graded"] = True
            out.append(
                _Graded(
                    scored=scored,
                    relevance=confidence,
                    # Both conditions: the boolean is the verdict, the threshold
                    # only filters *weak* positives. A threshold of 0.0 must not
                    # promote an explicit "not relevant" into a relevant one.
                    relevant=grade.relevant and confidence >= threshold,
                    reason=grade.reason,
                )
            )
        return out, usage

    async def _grade_texts(
        self, question: str, texts: Sequence[str], *, stage: str
    ) -> tuple[list[RelevanceGrade | None], Usage]:
        """One relevance grade per text. ``None`` where the call failed.

        Prefers the LLM's own ``batch_structured`` when it has one: that keeps the
        concurrency ceiling and the per-item error handling in a single place
        instead of reimplementing both here. The fallback path exists because the
        :class:`~ragorc.core.protocols.LLM` protocol does not require the method.
        """
        prompt = get_prompt("grade_relevance")
        model = self.router.model_for(Task.GRADE_RELEVANCE)
        # Fenced individually: a passage that instructs this grader to answer
        # CORRECT keeps itself in the candidate set, which is the whole point of
        # the grade. One passage per prompt, so each is numbered [1].
        prompts = [
            prompt.render(question=question, document=render_untrusted_passages([text]))
            for text in texts
        ]

        if isinstance(self.llm, BatchStructuredLLM):
            rows = await self.llm.batch_structured(
                prompts, RelevanceGrade, system=prompt.system, model=model, stage=stage
            )
        else:
            results = await bounded_gather(
                (
                    self.llm.structured(
                        p, RelevanceGrade, system=prompt.system, model=model, stage=stage
                    )
                    for p in prompts
                ),
                limit=self.settings.llm.max_concurrency,
                return_exceptions=True,
            )
            rows = []
            for item in results:
                if isinstance(item, BaseException):
                    log.warning("crag_grade_item_failed", stage=stage, error=str(item)[:200])
                    rows.append((None, Usage()))
                else:
                    rows.append(item)

        grades = [g if isinstance(g, RelevanceGrade) else None for g, _ in rows]
        return grades, Usage.sum(u for _, u in rows)

    async def _refine(
        self, query: Query, relevant: Sequence[_Graded], decision: CragDecision
    ) -> tuple[list[ScoredChunk], Usage]:
        """Knowledge refinement: strip, grade, reassemble.

        Only documents with more than one strip are stripped at all — a
        single-strip document has nothing to remove, and re-grading it would pay
        for the answer we already have.
        """
        keep = [g.scored for g in relevant]
        if not self.refine or not keep:
            return keep, Usage()
        if self.compressor is not None:
            return await self._refine_with_compressor(self.compressor, query, keep, decision)

        plans: list[tuple[int, list[tuple[int, int]]]] = []
        strips: list[str] = []
        allowance = _call_allowance()
        for index, scored in enumerate(keep):
            spans = self._strips(scored.chunk.content)
            if len(spans) < 2:
                continue
            if allowance is not None and len(strips) + len(spans) > allowance:
                # A call per strip makes this the widest fan-out in the file, so
                # it is what actually exhausts a 40-call ceiling. Stop between
                # documents, never mid-document: a document graded on a partial
                # set of verdicts would be reassembled from strips nobody judged,
                # which deletes evidence — worse than the noise refinement
                # exists to remove. Everything already planned still gets refined,
                # and the rest is kept whole.
                log.info(
                    "crag_refine_budget_capped",
                    documents=len(plans),
                    of=len(keep),
                    strips=len(strips),
                    allowed=allowance,
                )
                decision.errors.setdefault(
                    "crag_budget",
                    f"call budget allowed refining {len(plans)} of {len(keep)} documents",
                )
                break
            plans.append((index, spans))
            strips.extend(scored.chunk.content[start:end] for start, end in spans)

        decision.strips_total = len(strips)
        if not strips:
            return keep, Usage()

        question = query.original or query.text
        grades, usage = await self._grade_texts(question, strips, stage="crag_refine")
        threshold = self.settings.retrieval.crag_relevance_threshold

        out = list(keep)
        cursor = 0
        for index, spans in plans:
            verdicts = grades[cursor : cursor + len(spans)]
            cursor += len(spans)
            # Fail *open* at strip level: the document has already been judged
            # relevant, so an ungraded strip is kept. Deleting evidence because a
            # grader call dropped would be a worse error than keeping noise.
            keeps = [g is None or (g.relevant and _relevance(g) >= threshold) for g in verdicts]
            kept = sum(keeps)
            decision.strips_kept += kept

            if kept == len(spans):
                continue  # nothing to remove
            if kept == 0:
                # The document was relevant but no strip is: the two graders
                # contradict each other. Trust the document-level verdict, which
                # saw the whole thing in context.
                log.debug("crag_refine_kept_nothing", chunk_id=keep[index].id)
                decision.strips_kept += len(spans)
                continue

            text = self._reassemble(keep[index].chunk.content, spans, keeps)
            if not text.strip():
                continue
            out[index] = self._as_refined(keep[index], text, kept=kept, total=len(spans))
            decision.refined_documents += 1

        return out, usage

    async def _refine_with_compressor(
        self,
        compressor: Compressor,
        query: Query,
        keep: Sequence[ScoredChunk],
        decision: CragDecision,
    ) -> tuple[list[ScoredChunk], Usage]:
        """Delegate refinement to an injected :class:`Compressor`.

        Kept as a first-class alternative because the cost profiles are so
        different: the strip grader spends a model call per strip, while
        :class:`ragorc.retrieve.compress.SentenceLevelCompressor` spends one
        embedding batch for the whole set. Chunks are paired back up by id rather
        than by position, since a compressor is allowed to drop a chunk entirely.
        """
        before = {scored.id: scored.chunk.content for scored in keep}
        try:
            out, usage = await compressor.compress(query, keep)
        except Exception as exc:
            # Refinement is an improvement on evidence we have already decided to
            # use, so a failure here costs precision, not the answer.
            log.warning("crag_compressor_failed", error=str(exc)[:200])
            decision.errors["refine"] = str(exc)
            return list(keep), Usage()

        for scored in out:
            decision.strips_kept += int(scored.explain.get("sentences_kept", 0))
            decision.strips_total += int(scored.explain.get("sentences_total", 0))
            original = before.get(scored.id)
            if original is not None and original != scored.chunk.content:
                scored.explain["crag_refined"] = True
                decision.refined_documents += 1
        return list(out), usage

    def _web_wanted(self, action: GradeLabel | None, *, allowed: bool = True) -> bool:
        if not allowed:
            # The caller runs its own web step; see `run`.
            return False
        if action is GradeLabel.CORRECT or action is None:
            return False
        if not self.settings.retrieval.crag_web_fallback:
            return False
        # A retriever that reports itself disabled (the null one, or a
        # third-party stand-in) makes the rewrite call pure waste: nothing will
        # consume its output.
        return bool(getattr(self.web, "enabled", True))

    async def _web(
        self,
        query: Query,
        irrelevant: Sequence[_Graded],
        result: RetrievalResult,
        decision: CragDecision,
    ) -> tuple[list[ScoredChunk], Usage]:
        rewritten, usage = await self._rewrite(query, irrelevant)
        decision.rewritten_query = rewritten

        # Deliberately no ``filters``: corpus metadata filters mean nothing to a
        # search engine, and carrying them would suggest a scoping that is not
        # actually being applied.
        web_query = Query(
            text=rewritten,
            original=query.original,
            top_k=self.settings.retrieval.web_search_results,
            tenant_id=query.tenant_id,
            metadata={**query.metadata, "crag_rewrite_of": query.text},
        )
        try:
            chunks = list(
                await self.web.retrieve(web_query, top_k=self.settings.retrieval.web_search_results)
            )
        except (StoreUnavailable, ImportError) as exc:
            # ``ImportError`` belongs here with the store failures: a missing
            # optional extra is a deployment fact, not a property of this query,
            # and in the AMBIGUOUS branch there are still corpus documents worth
            # answering from.
            decision.errors["web"] = str(exc)
            log.warning("crag_web_unavailable", error=str(exc)[:200])
            return [], usage

        for scored in chunks:
            scored.explain["crag"] = "web_fallback"
        decision.web_results = len(chunks)
        return chunks, usage

    async def _rewrite(self, query: Query, irrelevant: Sequence[_Graded]) -> tuple[str, Usage]:
        """Rewrite the question for a web search engine.

        The corpus query and a web query are not the same artifact: one is aimed
        at an embedding of your documents, the other at an index of the open web
        that responds to different vocabulary. The rejected documents are shown
        to the rewriter as evidence of *how* the search missed.
        """
        allowance = _call_allowance()
        if allowance is not None and allowance < 1:
            # Same exit as a rewriter outage below, for the same reason: the
            # rewrite is an optimization and the search is the point. Spending the
            # last call in the budget on rephrasing the question would buy a
            # slightly better web query and cost the answer that has to be written
            # from it.
            log.info("crag_rewrite_budget_skipped")
            return query.text, Usage()

        prompt = get_prompt("rewrite_query")
        if irrelevant:
            # Fenced. This preview is retrieved text shown to the rewriter, and a
            # rewritten query is what the next retrieval runs — so an instruction
            # here steers the search itself.
            joined = render_untrusted_passages([g.scored.chunk.content for g in irrelevant])
            preview = truncate_to_tokens(joined, _REWRITE_PREVIEW_TOKENS)
        else:
            preview = "(retrieval returned nothing)"

        try:
            out, usage = await self.llm.structured(
                prompt.render(
                    question=query.original or query.text,
                    previous=query.text,
                    retrieved=preview,
                ),
                RewriteOutput,
                system=prompt.system,
                model=self.router.model_for(Task.REWRITE),
                stage="crag_rewrite",
            )
        except (LLMError, TransientError) as exc:
            # The rewrite is an optimization; the search is the point. Fall back
            # to the user's own words rather than skipping the fallback entirely.
            log.warning("crag_rewrite_failed", error=str(exc)[:200])
            return query.text, Usage()

        rewritten = (out.rewritten_query or "").strip()
        if rewritten and rewritten != query.text:
            log.debug("crag_rewrote_query", reason=out.reasoning[:160])
        return rewritten or query.text, usage

    # -- assembly -----------------------------------------------------------
    @staticmethod
    def _label(relevant_count: int, graded_count: int) -> GradeLabel:
        """Map grade counts onto the paper's three actions.

        ``graded_count == 0`` (nothing retrieved, or retrieval failed) is
        INCORRECT: no relevant document exists, which is exactly what the label
        means, and it routes to the web without spending a grader call.
        """
        if relevant_count == 0:
            return GradeLabel.INCORRECT
        if relevant_count == graded_count:
            return GradeLabel.CORRECT
        return GradeLabel.AMBIGUOUS

    def _assemble(
        self,
        action: GradeLabel | None,
        refined: Sequence[ScoredChunk],
        ungraded: Sequence[ScoredChunk],
        web_chunks: Sequence[ScoredChunk],
        limit: int,
    ) -> list[ScoredChunk]:
        """Order the surviving evidence and trim to ``top_k``.

        Ordering is by *provenance*, not by score, because the scores are not
        comparable: a cosine similarity from the vector store and a search
        engine's rank position measure different things, and interleaving them
        would let an arbitrary scale decide the ranking. So: documents graded
        relevant first, then web results (judged relevant by an engine, for a
        query we rewrote), then the ungraded tail that nobody looked at. The
        context packer's lost-in-the-middle reordering runs after this and moves
        the strongest evidence to both ends regardless.
        """
        if action is GradeLabel.INCORRECT:
            # The paper discards everything, and the ungraded tail goes with it:
            # the graded head was the retriever's *best*, so a tail that ranked
            # below an irrelevant best is not going to be better.
            merged = list(web_chunks)
        else:
            merged = [*refined, *web_chunks, *ungraded]
        return self._rank(merged[:limit])

    @staticmethod
    def _rank(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        for rank, scored in enumerate(chunks):
            scored.rank = rank
        return chunks

    def _finish(
        self,
        query: Query,
        result: RetrievalResult,
        decision: CragDecision,
        usages: Sequence[Usage],
    ) -> tuple[RetrievalResult, Usage]:
        """Emit the decision once, in the three places that need it."""
        usage = Usage.sum(usages)
        decision.calls = usage.calls
        # Every degradation recorded on the decision is also a retrieval
        # diagnostic. Copying them here — rather than writing both at each
        # failure site — means the two can never drift apart.
        for key, message in decision.errors.items():
            result.errors.setdefault(key, message)
        report = decision.report()
        # The trace is contextvar-scoped per request, so this is concurrency-safe
        # in a way instance state on a shared retriever would not be.
        trace_step("crag", duration_ms=sum(result.timings_ms.values()), usage=usage, **report)
        query.metadata["crag"] = report
        log.info("crag_decision", chunks=len(result.chunks), **report)
        return result, usage

    # -- strips -------------------------------------------------------------
    def _strips(self, text: str) -> list[tuple[int, int]]:
        """Cut a document into gradable strips, as contiguous character spans.

        Sentences are the natural unit, but a bare sentence list makes the grader
        judge fragments — a heading, "See figure 3.", a one-word line — and a
        fragment has too little content to be judged and too little cost to be
        worth a model call. Adjacent sentences are therefore merged until a strip
        reaches ``indexing.min_chunk_size``, the setting that already encodes
        "below this it is noise, not content" for the splitter. Reusing it keeps
        one definition of "too small to mean anything" in the codebase.

        Spans stay contiguous (``split_sentences`` guarantees it) so that
        :meth:`_reassemble` can merge surviving neighbours back into exact slices
        of the original text.
        """
        floor = max(self.settings.indexing.min_chunk_size, 1)
        spans = split_sentences(text)
        if len(spans) < 2:
            return spans

        merged: list[list[int]] = []
        for start, end in spans:
            if merged and merged[-1][1] - merged[-1][0] < floor:
                merged[-1][1] = end
            else:
                merged.append([start, end])
        # The final strip can still be a runt; fold it backwards so a trailing
        # fragment is never graded on its own.
        if len(merged) > 1 and merged[-1][1] - merged[-1][0] < floor:
            merged[-2][1] = merged.pop()[1]
        return [(start, end) for start, end in merged]

    @staticmethod
    def _reassemble(text: str, spans: Sequence[tuple[int, int]], keeps: Sequence[bool]) -> str:
        """Rebuild the document from the surviving strips, verbatim.

        Adjacent survivors are merged into one slice of the original text — which
        preserves the exact whitespace, and therefore preserves quoted spans for
        citation verification. A gap where a strip was dropped becomes a blank
        line: visible as an elision to the model, and not a token that could turn
        up inside a quote.
        """
        runs: list[list[int]] = []
        for (start, end), wanted in zip(spans, keeps, strict=True):
            if not wanted:
                continue
            if runs and runs[-1][1] == start:
                runs[-1][1] = end
            else:
                runs.append([start, end])
        return "\n\n".join(text[start:end].strip() for start, end in runs)

    @staticmethod
    def _as_refined(scored: ScoredChunk, text: str, *, kept: int, total: int) -> ScoredChunk:
        """Wrap refined text in a new object, keeping the chunk id.

        The id is the join key across Qdrant, Postgres and Neo4j and the handle
        the citation validator resolves against the store, so a refined chunk
        keeps it: this is a *view* of the stored chunk, not a new one. A new
        object rather than a mutation because the retrieved chunk may be shared
        with a cache entry, and refining it in place would poison that entry for
        every later query.
        """
        chunk = replace(
            scored.chunk,
            content=text,
            token_count=count_tokens(text),
            metadata={
                **scored.chunk.metadata,
                "crag_strips_kept": kept,
                "crag_strips_total": total,
            },
        )
        return ScoredChunk(
            chunk=chunk,
            score=scored.score,
            source=scored.source,
            rank=scored.rank,
            component_scores=dict(scored.component_scores),
            explain={**scored.explain, "crag_refined": True, "strips": f"{kept}/{total}"},
        )
