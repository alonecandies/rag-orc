"""Self-RAG: let the answer's own quality drive re-retrieval.

Self-RAG (Asai et al., 2023) adds reflection tokens to the generation loop. Three
of the four map onto decisions we already need to make, and this module implements
them as explicit graded steps rather than as special tokens:

* **ISREL** — is each retrieved document relevant? (implemented in CRAG's grader)
* **ISSUP** — is the generated answer supported by the retrieved evidence?
* **ISUSE** — does the answer actually address the question?

The loop: generate, grade ISSUP and ISUSE **concurrently** (they are independent
judgements about the same text, so serializing them doubles the latency of the
verification step for nothing), and on failure either rewrite the query and
retrieve again or abstain.

Two design points that matter more than the algorithm:

**The two failures need different responses.** An *ungrounded* answer means the
model outran its evidence — retrying with the same context will usually produce
the same overreach, so the query gets rewritten and retrieval is re-run. An answer
that is grounded but *not useful* means the evidence was about the wrong thing —
that also calls for re-retrieval, but with a rewrite aimed at coverage rather than
support. Treating both as "just try again" wastes the retry.

**The loop must terminate in an abstention, not in the best failed attempt.**
Returning the least-bad ungrounded answer defeats the entire mechanism: the point
of grading is to be able to decline.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import Answer, Query, RetrievalResult, Usage
from ragorc.core.protocols import LLM
from ragorc.core.schemas import RewriteOutput, UtilityGrade
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import trace_step
from ragorc.generate.groundedness import GroundednessChecker, GroundednessResult
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["SelfRAG", "SelfRAGAttempt", "SelfRAGResult"]

RetrieveFn = Callable[[Query], Awaitable[RetrievalResult]]
GenerateFn = Callable[[Query, RetrievalResult], Awaitable[Answer]]


@dataclass(slots=True)
class SelfRAGAttempt:
    iteration: int
    query: str
    grounded: bool
    groundedness: float
    useful: bool
    utility: float
    verdict: str
    answer_preview: str = ""


@dataclass(slots=True)
class SelfRAGResult:
    answer: Answer
    attempts: list[SelfRAGAttempt] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    accepted_iteration: int = 0

    def report(self) -> dict[str, object]:
        return {
            "iterations": len(self.attempts),
            "accepted_at": self.accepted_iteration,
            "abstained": self.answer.abstained,
            "trail": [
                {
                    "i": a.iteration,
                    "grounded": a.grounded,
                    "useful": a.useful,
                    "verdict": a.verdict,
                }
                for a in self.attempts
            ],
        }


class SelfRAG:
    """Wraps a retrieve/generate pair in a graded reflection loop."""

    name = "self_rag"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        grounding: GroundednessChecker | None = None,
    ) -> None:
        self.llm = llm
        self.settings: Settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self.grounding = grounding or GroundednessChecker(llm, self.settings, router=self.router)

    async def run(
        self,
        query: Query,
        retrieve: RetrieveFn,
        generate: GenerateFn,
    ) -> SelfRAGResult:
        gen = self.settings.generation
        max_retries = max(gen.self_rag_max_retries, 0)
        usages: list[Usage] = []
        attempts: list[SelfRAGAttempt] = []
        current = query
        best: Answer | None = None

        for iteration in range(max_retries + 1):
            retrieval = await retrieve(current)
            answer = await generate(current, retrieval)
            usages.append(answer.usage)

            if answer.abstained:
                # The generator's own gates already declined. Grading a refusal
                # would spend calls to confirm what it just told us.
                attempts.append(
                    SelfRAGAttempt(
                        iteration=iteration,
                        query=current.text,
                        grounded=False,
                        groundedness=0.0,
                        useful=False,
                        utility=0.0,
                        verdict="abstained_by_generator",
                    )
                )
                best = answer
                break

            grounded, groundedness, useful, utility, grade_usage = await self._grade(
                current, answer, retrieval
            )
            usages.append(grade_usage)

            verdict = (
                "accepted"
                if grounded and useful
                else "ungrounded"
                if not grounded
                else "not_useful"
            )
            attempts.append(
                SelfRAGAttempt(
                    iteration=iteration,
                    query=current.text,
                    grounded=grounded,
                    groundedness=groundedness,
                    useful=useful,
                    utility=utility,
                    verdict=verdict,
                    answer_preview=answer.text[:120],
                )
            )
            trace_step(
                "self_rag_iteration",
                iteration=iteration,
                grounded=grounded,
                useful=useful,
                verdict=verdict,
            )

            if grounded and useful:
                answer.grounded = True
                answer.groundedness = groundedness
                answer.confidence = min(groundedness, utility)
                answer.metadata["self_rag"] = {
                    "iterations": iteration + 1,
                    "utility": round(utility, 3),
                }
                return SelfRAGResult(
                    answer=answer,
                    attempts=attempts,
                    usage=Usage.sum(usages),
                    accepted_iteration=iteration,
                )

            best = answer
            if iteration >= max_retries:
                break

            current, rewrite_usage = await self._rewrite(current, answer, verdict)
            usages.append(rewrite_usage)

        # Every iteration failed. Abstain rather than returning the least-bad
        # ungrounded answer — that is the whole point of having graded them.
        final = self._abstain(best, query, attempts)
        final.usage = Usage.sum(usages)
        return SelfRAGResult(
            answer=final,
            attempts=attempts,
            usage=final.usage,
            accepted_iteration=-1,
        )

    # -- grading -----------------------------------------------------------
    async def _grade(
        self, query: Query, answer: Answer, retrieval: RetrievalResult
    ) -> tuple[bool, float, bool, float, Usage]:
        """ISSUP and ISUSE, concurrently."""
        utility_prompt = get_prompt("grade_utility")

        async def utility() -> tuple[UtilityGrade | None, Usage]:
            try:
                return await self.llm.structured(
                    utility_prompt.render(
                        question=query.original or query.text, answer=answer.text
                    ),
                    UtilityGrade,
                    system=utility_prompt.system,
                    model=self.router.model_for(Task.GRADE_UTILITY),
                    stage="grade_utility",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("utility_grade_failed", error=str(exc)[:160])
                return None, Usage()

        results = await bounded_gather(
            [
                self.grounding.check(query.original or query.text, answer.text, answer.chunks),
                utility(),
            ],
            limit=2,
            return_exceptions=True,
        )

        # `bounded_gather` is generic in a single result type, so a heterogeneous
        # fan-out collapses its element type to `object`, and `return_exceptions=True`
        # additionally widens every slot with the exception its coroutine raised.
        # The casts restate the per-slot types the call site above already fixed
        # (results are in input order); they suppress nothing — the `isinstance`
        # guards below still have to prove which arm each value is in.
        ground_result = cast("GroundednessResult | BaseException", results[0])
        utility_result = cast("tuple[UtilityGrade | None, Usage] | BaseException", results[1])
        usages: list[Usage] = []

        if isinstance(ground_result, BaseException):
            log.warning("groundedness_failed", error=str(ground_result)[:160])
            grounded, groundedness = answer.grounded, answer.groundedness
        else:
            grounded = ground_result.grounded
            groundedness = ground_result.score
            usages.append(ground_result.usage)

        # Grading failure must not be read as failure of the answer: on either a
        # raised exception or a missing grade, default to useful so a broken
        # grader cannot force an abstention.
        if isinstance(utility_result, BaseException):
            useful, utility_score = True, 1.0
        else:
            grade, usage = utility_result
            if grade is None:
                useful, utility_score = True, 1.0
            else:
                useful = bool(grade.useful)
                utility_score = float(grade.score) or (1.0 if grade.useful else 0.0)
                usages.append(usage)

        return grounded, groundedness, useful, utility_score, Usage.sum(usages)

    # -- rewriting ---------------------------------------------------------
    async def _rewrite(self, query: Query, answer: Answer, verdict: str) -> tuple[Query, Usage]:
        """Rewrite the query, aimed at the specific failure that occurred."""
        prompt = get_prompt("rewrite_query")
        if verdict == "ungrounded":
            hint = (
                "The previous answer made claims the retrieved documents did not support. "
                "Rewrite the query to find documents that directly state these facts."
            )
            retrieved = "; ".join(c.chunk.content[:120] for c in answer.chunks[:3])
        else:
            hint = (
                "The previous answer was supported but did not address the question. "
                "Rewrite the query to target what was actually asked."
            )
            retrieved = answer.text[:300]

        try:
            result, usage = await self.llm.structured(
                prompt.render(
                    question=query.original or query.text,
                    previous=query.text,
                    retrieved=f"{hint}\n\n{retrieved}",
                ),
                RewriteOutput,
                system=prompt.system,
                model=self.router.model_for(Task.REWRITE),
                stage="self_rag_rewrite",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("rewrite_failed", error=str(exc)[:160])
            return query, Usage()

        rewritten = result.rewritten_query.strip() or query.text
        log.info("self_rag_rewrite", verdict=verdict, rewritten=rewritten[:80])
        return (
            Query(
                text=rewritten,
                original=query.original,
                variants=query.variants,
                filters=dict(query.filters),
                top_k=query.top_k,
                tenant_id=query.tenant_id,
                metadata={
                    **query.metadata,
                    "self_rag_rewrite_of": query.text,
                    "self_rag_verdict": verdict,
                },
            ),
            usage,
        )

    def _abstain(self, best: Answer | None, query: Query, attempts: list[SelfRAGAttempt]) -> Answer:
        gen = self.settings.generation
        reason = f"self-RAG exhausted {len(attempts)} attempt(s) without a grounded, useful answer"
        if not gen.allow_abstention and best is not None:
            # `generation.allow_abstention` is read by `AbstentionPolicy` and was
            # not read here, so a deployment that switched abstention off still
            # got the refusal from this loop. The best attempt is returned instead,
            # marked ungrounded and carrying the verdicts, which is what the
            # setting asks for: the model's best answer plus what is known about
            # it, rather than a refusal.
            log.info("self_rag_abstention_suppressed", attempts=len(attempts))
            best.metadata = {
                **best.metadata,
                "self_rag": {
                    "iterations": len(attempts),
                    "verdicts": [a.verdict for a in attempts],
                },
                "abstention_suppressed": reason,
            }
            best.grounded = False
            return best
        log.info("self_rag_abstained", attempts=len(attempts), reason=reason)
        answer = best or Answer(text="")
        answer.metadata = {
            **answer.metadata,
            "self_rag": {"iterations": len(attempts), "verdicts": [a.verdict for a in attempts]},
            "rejected_answer": answer.text,
        }
        answer.text = gen.abstain_message
        answer.abstained = True
        answer.abstain_reason = reason
        answer.grounded = False
        answer.confidence = 0.0
        answer.citations = []
        return answer
