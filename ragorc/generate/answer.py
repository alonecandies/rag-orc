"""The answer generator.

This is where every guarantee the library makes is actually enforced, in a fixed
order chosen so that each step's cost is only paid when the previous step allowed
it:

    abstain (pre)  ->  budget  ->  pack (+compress on overflow)  ->  generate
                   ->  extract citations  ->  validate  ->  ground  ->  abstain (post)

Two ordering decisions are load-bearing:

* **Abstain before generating.** If retrieval returned nothing usable, the
  synthesis call is pure waste — and worse, a model given no evidence will answer
  from its parameters, producing the most confident hallucination in the system.
* **Ground after validating citations.** Citation validation is free (string
  matching) and catches fabricated attribution outright; groundedness costs model
  calls. Running the cheap, decisive check first means the expensive one is
  skipped on answers that were already disqualified.

The result is that an :class:`Answer` from this class always carries a
groundedness score, resolvable citations, an honest ``abstained`` flag, and a
complete cost ledger — none of which are optional extras a caller has to
remember to ask for.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import structlog

from ragorc.context.budget import ContextBudgeter
from ragorc.context.pack import ContextPacker
from ragorc.context.summarize import ContextSummarizer
from ragorc.core.models import (
    Answer,
    Citation,
    Query,
    RetrievalResult,
    RouteDecision,
    ScoredChunk,
    Usage,
)
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import AnswerWithCitations
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import current_trace, timed, trace_step
from ragorc.generate.abstain import AbstentionDecision, AbstentionPolicy
from ragorc.generate.citations import _base_offset, attribute_spans, extract_citations
from ragorc.generate.consistency import SelfConsistencyChecker
from ragorc.generate.groundedness import GroundednessChecker, GroundednessResult
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.validate.output import AnswerValidator

log = structlog.get_logger(__name__)

__all__ = ["AnswerGenerator"]


def _citations_from_statements(
    structured: AnswerWithCitations, packed: Sequence[ScoredChunk]
) -> list[Citation]:
    """Turn the model's ``statements`` into citations.

    This style's whole purpose is attribution, and its system prompt relocates
    attribution out of inline ``[n]`` markers and into ``statements`` — which had
    no reader anywhere in the package. So the branch parsed the field and threw it
    away, then ran the inline-marker regex over an answer that by construction
    contains no markers.

    ``source_ids`` are 1-based passage numbers, the same numbering
    :func:`~ragorc.generate.citations.extract_citations` resolves and the packer
    renders. Out-of-range ids are dropped rather than clamped: a model naming
    passage 9 of 3 is guessing, and a citation pointing at the wrong passage is
    worse than one that is absent.
    """
    out: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for statement in structured.statements:
        claim = statement.text.strip()
        if not claim:
            continue
        for number in statement.source_ids:
            index = int(number) - 1
            if not 0 <= index < len(packed):
                continue
            chunk = packed[index].chunk
            key = (chunk.id, claim[:80])
            if key in seen:
                continue
            seen.add(key)
            quote, start, end, support = attribute_spans(claim, chunk.content)
            base = _base_offset(packed[index])
            out.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    quote=quote,
                    claim=claim,
                    support=support,
                    source=chunk.metadata.get("source") or chunk.metadata.get("title"),
                    start_char=None if base is None or start is None else base + start,
                    end_char=None if base is None or end is None else base + end,
                )
            )
    return out


@register("generator", "answer", "default")
class AnswerGenerator:
    """Produces a grounded, cited, cost-accounted answer."""

    name = "answer"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        packer: ContextPacker | None = None,
        budgeter: ContextBudgeter | None = None,
        summarizer: ContextSummarizer | None = None,
        grounding: GroundednessChecker | None = None,
        validator: AnswerValidator | None = None,
        abstention: AbstentionPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.settings: Settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self.packer = packer or ContextPacker(self.settings)
        self.budgeter = budgeter or ContextBudgeter(self.settings)
        self.summarizer = summarizer or ContextSummarizer(llm, self.settings, router=self.router)
        self.grounding = grounding or GroundednessChecker(llm, self.settings, router=self.router)
        self.validator = validator or AnswerValidator(self.settings)
        self.abstention = abstention or AbstentionPolicy(self.settings)

    # -- public ------------------------------------------------------------
    async def generate(
        self,
        query: Query,
        retrieval: RetrievalResult,
        *,
        route: RouteDecision | None = None,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> Answer:
        gen = self.settings.generation
        chunks = list(retrieval.chunks)
        usages: list[Usage] = []

        # --- gate 1: is there anything to answer from? ---------------------
        pre = self.abstention.before_generation(chunks)
        if pre.abstain:
            return self._abstained(query, chunks, pre, usages, route)

        prompt = get_prompt(
            prompt_name or (route.prompt_name if route else None) or gen.prompt_name
        )

        # --- budget, pack, compress ----------------------------------------
        plan = self.budgeter.plan(system_prompt=prompt.system, question=query.text)
        # The packer's own measurement, so both agree on what has to fit.
        strategy = self.budgeter.decide_strategy(
            chunks, plan, overhead=self.packer.overhead(chunks)
        )
        if strategy == "summarize":
            with timed("compress_context"):
                compressed = await self.summarizer.fit(
                    query.text, chunks, budget=plan.budget.available_context, strategy="map_reduce"
                )
            chunks = compressed.chunks
            usages.append(compressed.usage)

        with timed("pack_context"):
            # `plan.per_source` is a floor per contributing store, not a cap: it
            # was computed on every request and read by nothing, so one leg could
            # take the whole window and the decisive row from another never
            # shipped. See `ContextPacker._allowances`.
            pack = self.packer.build(
                chunks, budget=plan.budget.available_context, shares=plan.per_source
            )
        packed = pack.chunks
        plan.context_tokens = pack.tokens
        plan.dropped_chunks = pack.dropped
        plan.overflow = strategy != "fit"
        plan.strategy = strategy

        if not packed:
            decision = self.abstention.before_generation([])
            return self._abstained(query, chunks, decision, usages, route)

        # --- generate ------------------------------------------------------
        rendered = prompt.render(context=pack.text, question=query.text)
        model = self.router.model_for(Task.ANSWER)

        json_citations: list[Citation] | None = None
        if gen.self_consistency_samples > 1:
            checker = SelfConsistencyChecker(self.llm, self.settings)
            with timed("generate_self_consistent", samples=gen.self_consistency_samples):
                consistency = await checker.generate(rendered, system=prompt.system, model=model)
            text = consistency.answer
            usages.append(consistency.usage)
            confidence = consistency.agreement
            model_insufficient = False
            trace_step("self_consistency", **consistency.report())
        elif gen.citation_style == "json":
            structured, usage = await self.llm.structured(
                rendered,
                AnswerWithCitations,
                # The routed prompt's system block *plus* the attribution
                # contract, not instead of it. Hardcoding
                # `answer_with_citations.system` discarded the router's choice, so
                # `answer_technical` and `answer_concise` had no effect under this
                # style while `answer.metadata["prompt"]` still reported the routed
                # name.
                system="\n\n".join(
                    filter(None, [prompt.system, get_prompt("answer_with_citations").system])
                ),
                model=model,
                # The same cap the plain path applies. Without it this path ran at
                # the global `llm.max_tokens`, and the context packer had already
                # reserved output room from `max_answer_tokens`
                # (`Settings.model_post_init`) — so the budget the prompt was built
                # against and the budget the answer was generated under disagreed.
                max_tokens=gen.max_answer_tokens,
                stage="answer",
            )
            usages.append(usage)
            text = structured.answer
            model_insufficient = not structured.sufficient
            confidence = 1.0
            # The attribution this style exists for. The system block above tells
            # the model to put it in `statements` and *not* to write inline `[n]`,
            # and `statements` had no reader anywhere in the package — so
            # `extract_citations` below regexed for markers the prompt had
            # explicitly relocated, `answer.citations` came back empty and
            # `citation_coverage` was 0.0 with `report.valid` still True.
            json_citations = _citations_from_statements(structured, packed)
        else:
            with timed("generate_answer", model=model):
                text, usage = await self.llm.complete(
                    rendered,
                    system=prompt.system,
                    model=model,
                    max_tokens=gen.max_answer_tokens,
                    stage="answer",
                )
            usages.append(usage)
            model_insufficient = False
            confidence = 1.0

        text = text.strip()

        # --- citations + validation (cheap, decisive) -----------------------
        if json_citations is not None:
            citations = json_citations if gen.cite_sources else []
        else:
            citations = extract_citations(text, packed) if gen.cite_sources else []
        answer = Answer(
            text=text,
            citations=citations,
            chunks=packed,
            usage=Usage.sum(usages),
            route=route,
        )
        report = self.validator.validate(answer, packed)

        # --- groundedness (expensive; skipped if already disqualified) ------
        grounded, score, unsupported, contradicted = True, 1.0, [], []
        if gen.check_groundedness and report.valid:
            with timed("check_groundedness", method=gen.groundedness_method):
                result = await self.grounding.check(query.text, text, packed)
            usages.append(result.usage)
            grounded, score = result.grounded, result.score
            unsupported, contradicted = result.unsupported, result.contradicted
            trace_step("groundedness", **result.report())
            self._attach_verified_quotes(answer, result)
        elif not report.valid:
            grounded, score = False, 0.0
            log.info("groundedness_skipped", reason="citation_validation_failed")

        # --- gate 2: is the answer fit to return? --------------------------
        post = self.abstention.after_generation(
            answer_text=text,
            grounded=grounded,
            groundedness_score=score,
            contradicted=contradicted,
            model_says_insufficient=model_insufficient,
            invalid_citations=report.invalid_citations,
        )

        answer.usage = Usage.sum(usages)
        answer.grounded = grounded
        answer.groundedness = score
        answer.confidence = min(confidence, score if gen.check_groundedness else confidence)
        answer.trace = list(current_trace())
        answer.metadata = {
            "prompt": prompt.name,
            "model": model,
            "budget": plan.report(),
            "validation": {
                "valid": report.valid,
                "invalid_citations": report.invalid_citations,
                "unverified_quotes": report.unverified_quotes,
                "citation_coverage": round(report.citation_coverage, 3),
                # Both were computed and reported nowhere. A caller cannot tell
                # that an answer was rewritten unless the rewrite is declared,
                # and "was any of this redacted?" is a question a compliance
                # reader asks of every answer, not only the ones that warned.
                "scaffold_leak": report.scaffold_leak,
                "pii_redacted": report.pii_entities,
                "warnings": report.warnings,
            },
            "unsupported_claims": unsupported,
            "contradicted_claims": contradicted,
        }

        if post.abstain:
            answer.abstained = True
            answer.abstain_reason = post.reason
            answer.metadata["abstain_gate"] = post.gate
            # The generated text is retained under `rejected_answer` rather than
            # discarded: it is the most useful artifact for diagnosing why the
            # pipeline declined, and callers must opt in to see it.
            answer.metadata["rejected_answer"] = answer.text
            answer.text = post.message
            answer.confidence = post.confidence
            answer.citations = []
        elif report.invalid_citations:
            # Valid answer, phantom markers: strip them rather than showing the
            # reader a citation they cannot follow.
            answer = self.validator.strip_invalid_citations(answer, packed)

        return answer

    async def stream(
        self,
        query: Query,
        retrieval: RetrievalResult,
        *,
        route: RouteDecision | None = None,
        prompt_name: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream the answer.

        Streaming and verification are fundamentally in tension: groundedness can
        only be judged once the answer is complete, but the point of streaming is
        to emit before then. The honest resolution is to stream the text and let
        the caller fetch verification separately — never to stream an answer while
        claiming it has been verified. The pre-generation gate still applies, so a
        stream is not started when there is no evidence at all.

        That tension is real for groundedness and citation checking, and it is not
        an argument for skipping *everything*, which is what this used to do. The
        leakage checks are regexes over emitted text: they need no completed
        answer, only a small held-back tail so a pattern split across two deltas is
        still seen whole. So PII redaction and scaffold stripping run here too, and
        an operator who switched on ``enable_pii_redaction`` gets it on both paths
        rather than on the non-streaming one alone.
        """
        gen = self.settings.generation
        chunks = list(retrieval.chunks)
        pre = self.abstention.before_generation(chunks)
        if pre.abstain:
            yield pre.message
            return

        prompt = get_prompt(
            prompt_name or (route.prompt_name if route else None) or gen.prompt_name
        )
        plan = self.budgeter.plan(system_prompt=prompt.system, question=query.text)
        pack = self.packer.build(chunks, budget=plan.budget.available_context)
        if not pack.chunks:
            yield gen.abstain_message
            return

        rendered = prompt.render(context=pack.text, question=query.text)
        # `LLM.stream` is declared `async def ... -> AsyncIterator[str]`, which
        # types the *call* as a coroutine yielding the iterator. Every real
        # implementation is an async generator (see `OpenRouterLLM.stream`), so
        # the call already is the iterator and awaiting it would raise at runtime.
        # The declaration in ragorc/core/protocols.py should drop the `async`
        # (`def stream(...) -> AsyncIterator[str]: ...`, the correct way to type
        # an async-generator method); this cast states what the object really is
        # until it does.
        deltas = cast(
            "AsyncIterator[str]",
            self.llm.stream(
                rendered,
                system=prompt.system,
                model=self.router.model_for(Task.ANSWER),
                max_tokens=gen.max_answer_tokens,
                stage="answer_stream",
            ),
        )
        leaks = self.validator.stream_filter()
        async for delta in deltas:
            emit = leaks.feed(delta)
            if emit:
                yield emit
        tail = leaks.flush()
        if tail:
            yield tail
        if leaks.redacted:
            # No Answer object exists on this path to hang metadata off, so the log
            # line is the only record that the stream was rewritten.
            log.info(
                "stream_answer_redacted",
                scaffold_leak=leaks.scaffold_leak,
                pii_entities=leaks.entities,
            )

    # -- helpers -----------------------------------------------------------
    def _abstained(
        self,
        query: Query,
        chunks: Sequence[ScoredChunk],
        decision: AbstentionDecision,
        usages: list[Usage],
        route: RouteDecision | None,
    ) -> Answer:
        return Answer(
            text=decision.message or self.settings.generation.abstain_message,
            chunks=list(chunks),
            usage=Usage.sum(usages),
            grounded=False,
            groundedness=0.0,
            confidence=decision.confidence,
            abstained=True,
            abstain_reason=decision.reason,
            trace=list(current_trace()),
            route=route,
            metadata={"abstain_gate": decision.gate, "question": query.text},
        )

    @staticmethod
    def _attach_verified_quotes(answer: Answer, result: GroundednessResult) -> None:
        """Upgrade lexically-attributed quotes with the verifier's own evidence.

        The claim-level verifier quotes the span it relied on, which is a stronger
        attribution than our lexical guess — it is what the judgement was actually
        based on. Where the verifier supplied one, it replaces the guess.
        """
        by_chunk: dict[str, tuple[str, float]] = {}
        for check in getattr(result, "claims", []) or []:
            if check.supported and check.evidence_quote and check.chunk_id:
                by_chunk.setdefault(check.chunk_id, (check.evidence_quote, check.score))
        for citation in answer.citations:
            upgrade = by_chunk.get(citation.chunk_id)
            if upgrade:
                citation.quote, citation.support = upgrade[0], max(citation.support, upgrade[1])
