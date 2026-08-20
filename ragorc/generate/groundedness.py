"""Groundedness checking: does the evidence actually support the answer?

Hallucination in RAG is rarely a wild fabrication. It is almost always one of
three quieter failures:

1. **Extrapolation** — the context supports 90% of a sentence and the model
   supplies the last 10% from its parameters.
2. **Composition** — two facts appear separately in the context and the answer
   asserts a causal or comparative relationship between them that no passage
   makes.
3. **Detail drift** — a number, date or name that is *close* to the source but
   not the source's. This is the most dangerous kind, because it is the most
   plausible.

A single "is this grounded?" call over the whole answer catches (1) and misses
(2) and (3): asked to judge a paragraph holistically, models anchor on overall
plausibility. So the strong mode decomposes the answer into atomic claims and
verifies each one independently against the evidence — the verifier sees one
claim at a time and has nothing to be holistically impressed by.

Three modes, by cost:

* ``llm`` — one whole-answer groundedness call. Cheap, catches gross failures.
* ``nli`` — a local cross-encoder entailment model. No API call, fast, and
  entirely local; needs the ``[nli]`` extra.
* ``both`` — claim decomposition plus per-claim verification, run concurrently.
  Most expensive, and the only mode that reliably catches composition and drift.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import ScoredChunk, Usage
from ragorc.core.protocols import LLM
from ragorc.core.schemas import ClaimList, ClaimVerdict, GroundednessGrade
from ragorc.core.settings import Settings, get_settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["ClaimCheck", "GroundednessChecker", "GroundednessResult"]


@dataclass(slots=True)
class ClaimCheck:
    claim: str
    verdict: str = "not_enough_info"
    score: float = 0.0
    evidence_quote: str = ""
    chunk_id: str | None = None

    @property
    def supported(self) -> bool:
        return self.verdict == "supported"

    @property
    def contradicted(self) -> bool:
        return self.verdict == "contradicted"


@dataclass(slots=True)
class GroundednessResult:
    grounded: bool
    score: float
    usage: Usage = field(default_factory=Usage)
    method: str = "llm"
    unsupported: list[str] = field(default_factory=list)
    contradicted: list[str] = field(default_factory=list)
    claims: list[ClaimCheck] = field(default_factory=list)

    @property
    def supported_fraction(self) -> float:
        if not self.claims:
            return 1.0 if self.grounded else 0.0
        return sum(1 for c in self.claims if c.supported) / len(self.claims)

    def report(self) -> dict[str, Any]:
        """Telemetry payload. ``Any`` rather than ``object`` because this is
        splatted into ``trace_step(**...)``, whose named parameters are typed."""
        return {
            "grounded": self.grounded,
            "score": round(self.score, 3),
            "method": self.method,
            "claims": len(self.claims),
            "supported": sum(1 for c in self.claims if c.supported),
            "contradicted": len(self.contradicted),
            "unsupported": len(self.unsupported),
        }


class GroundednessChecker:
    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        self.llm = llm
        self.settings: Settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        # `Any`, not the real class: `sentence_transformers` is the optional
        # `[nli]` extra, so naming `CrossEncoder` here would make the module
        # unimportable without it. Left bare, mypy infers this attribute as
        # `None` from this line and `_load_nli`'s assignment becomes an error
        # the moment anyone actually installs the extra.
        self._nli: Any = None

    async def check(
        self,
        question: str,
        answer: str,
        chunks: Sequence[ScoredChunk],
        *,
        method: str | None = None,
    ) -> GroundednessResult:
        gen = self.settings.generation
        method = method or gen.groundedness_method

        if not answer.strip():
            return GroundednessResult(grounded=False, score=0.0, method=method)
        if not chunks:
            # Nothing was retrieved, so nothing can ground the answer. Reporting
            # "grounded" here would be a lie of omission.
            return GroundednessResult(
                grounded=False, score=0.0, method=method, unsupported=[answer[:200]]
            )

        evidence = "\n\n".join(f"[{i}] {c.chunk.content}" for i, c in enumerate(chunks, 1))

        if method == "nli":
            return await self._check_nli(answer, chunks)
        if method == "both" or gen.decompose_claims:
            return await self._check_claims(answer, evidence, chunks)
        return await self._check_holistic(answer, evidence)

    # -- modes -------------------------------------------------------------
    async def _check_holistic(self, answer: str, evidence: str) -> GroundednessResult:
        prompt = get_prompt("grade_groundedness")
        grade, usage = await self.llm.structured(
            prompt.render(context=evidence, answer=answer),
            GroundednessGrade,
            system=prompt.system,
            model=self.router.model_for(Task.GRADE_GROUNDEDNESS),
            stage="grade_groundedness",
        )
        threshold = self.settings.generation.groundedness_threshold
        # Trust the boolean over the score: a model that lists unsupported claims
        # while returning grounded=true has contradicted itself, and the list is
        # the more specific signal.
        score = float(grade.score)
        grounded = bool(grade.grounded) and not grade.unsupported_claims and score >= threshold
        return GroundednessResult(
            grounded=grounded,
            score=score,
            usage=usage,
            method="llm",
            unsupported=list(grade.unsupported_claims),
        )

    async def _check_claims(
        self, answer: str, evidence: str, chunks: Sequence[ScoredChunk]
    ) -> GroundednessResult:
        """Decompose, then verify each claim independently and concurrently."""
        decompose = get_prompt("decompose_claims")
        claim_list, decompose_usage = await self.llm.structured(
            decompose.render(answer=answer),
            ClaimList,
            system=decompose.system,
            model=self.router.model_for(Task.DECOMPOSE_CLAIMS),
            stage="decompose_claims",
        )
        claims = [c.strip() for c in claim_list.claims if c.strip()]
        if not claims:
            # No factual claims to verify (a pure abstention, or a question
            # restatement) is vacuously grounded.
            return GroundednessResult(
                grounded=True, score=1.0, usage=decompose_usage, method="claims"
            )

        verify = get_prompt("verify_claim")
        model = self.router.model_for(Task.VERIFY_CLAIM)

        async def one(claim: str) -> tuple[ClaimCheck, Usage]:
            verdict, usage = await self.llm.structured(
                verify.render(evidence=evidence, claim=claim),
                ClaimVerdict,
                system=verify.system,
                model=model,
                stage="verify_claim",
            )
            check = ClaimCheck(
                claim=claim,
                verdict=verdict.verdict,
                score=float(verdict.score),
                evidence_quote=verdict.evidence_quote,
                chunk_id=self._locate_quote(verdict.evidence_quote, chunks),
            )
            return check, usage

        results = await bounded_gather(
            (one(c) for c in claims),
            limit=self.settings.llm.max_concurrency,
            return_exceptions=True,
        )

        checks: list[ClaimCheck] = []
        usages: list[Usage] = [decompose_usage]
        for claim, result in zip(claims, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("claim_verify_failed", error=str(result)[:160])
                # Fail closed: an unverifiable claim counts as unsupported, not
                # as supported. The whole point is to be conservative.
                checks.append(ClaimCheck(claim=claim, verdict="not_enough_info"))
                continue
            check, usage = result
            checks.append(check)
            usages.append(usage)

        supported = sum(1 for c in checks if c.supported)
        contradicted = [c.claim for c in checks if c.contradicted]
        unsupported = [c.claim for c in checks if not c.supported and not c.contradicted]
        score = supported / len(checks)
        threshold = self.settings.generation.groundedness_threshold
        # Any contradiction is disqualifying regardless of the ratio: an answer
        # that states something the evidence denies is wrong, not partly right.
        grounded = score >= threshold and not contradicted

        return GroundednessResult(
            grounded=grounded,
            score=score,
            usage=Usage.sum(usages),
            method="claims",
            unsupported=unsupported,
            contradicted=contradicted,
            claims=checks,
        )

    async def _check_nli(self, answer: str, chunks: Sequence[ScoredChunk]) -> GroundednessResult:
        """Local cross-encoder entailment. No API call, no cost, no network.

        Each answer sentence is scored against every chunk and takes its best
        score — a sentence only needs *one* passage to support it. Runs in a
        thread because the model is CPU-bound and would otherwise stall the loop.
        """
        model = self._load_nli()
        if model is None:
            log.info("nli_unavailable", fallback="llm")
            evidence = "\n\n".join(c.chunk.content for c in chunks)
            return await self._check_holistic(answer, evidence)

        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 15]
        if not sentences:
            return GroundednessResult(grounded=True, score=1.0, method="nli")

        premises = [c.chunk.content for c in chunks]
        pairs = [(p, s) for s in sentences for p in premises]
        scores = await asyncio.to_thread(model.predict, pairs)

        import numpy as np

        matrix = np.asarray(scores, dtype=np.float32).reshape(len(sentences), len(premises))
        best = matrix.max(axis=1)
        threshold = self.settings.generation.groundedness_threshold
        unsupported = [s for s, v in zip(sentences, best, strict=True) if float(v) < threshold]
        mean = float(best.mean())
        return GroundednessResult(
            grounded=not unsupported,
            score=mean,
            method="nli",
            unsupported=unsupported,
            claims=[
                ClaimCheck(
                    claim=s,
                    verdict="supported" if float(v) >= threshold else "not_enough_info",
                    score=float(v),
                    chunk_id=chunks[int(np.argmax(matrix[i]))].chunk.id,
                )
                for i, (s, v) in enumerate(zip(sentences, best, strict=True))
            ],
        )

    def _load_nli(self) -> Any:
        """The cached ``CrossEncoder``, or ``None`` when the extra is absent.

        Typed ``Any`` for the same reason as ``self._nli``: the concrete class
        lives behind an optional dependency and cannot be named unconditionally.
        """
        if self._nli is not None:
            return self._nli
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            return None
        try:
            self._nli = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
        except Exception as exc:  # pragma: no cover - model download failure
            log.warning("nli_load_failed", error=str(exc)[:200])
            return None
        return self._nli

    @staticmethod
    def _locate_quote(quote: str, chunks: Sequence[ScoredChunk]) -> str | None:
        """Attribute a verifier's quoted span back to a specific chunk, so a
        verified claim becomes a *citable* claim rather than just a passing grade."""
        if not quote:
            return None
        needle = " ".join(quote.lower().split())[:120]
        if not needle:
            return None
        for chunk in chunks:
            if needle in " ".join(chunk.chunk.content.lower().split()):
                return chunk.chunk.id
        return None
