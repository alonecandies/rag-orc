"""Abstention policy: deciding when "I don't know" is the correct answer.

An abstention is a **success**, not a failure. A RAG system that always produces
an answer has no way to signal that its evidence was inadequate, so its worst
outputs are indistinguishable from its best — which makes every output
untrustworthy. The ability to decline is what makes the other answers credible.

The policy is a set of independent gates, each of which can force an abstention.
They are ordered cheapest-first, so a query with no retrieval at all never
reaches the groundedness check.

Deliberately *not* a gate: low similarity scores alone. Absolute similarity is
not calibrated across corpora or embedding models, and thresholding on it
abstains on easy questions in hard corpora. Evidence sufficiency is judged by
the reranker's relative signal and by groundedness, which are calibrated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import structlog

from ragorc.core.models import ScoredChunk
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["AbstentionDecision", "AbstentionPolicy"]


@dataclass(slots=True)
class AbstentionDecision:
    abstain: bool = False
    reason: str = ""
    gate: str = ""
    message: str = ""
    confidence: float = 1.0


class AbstentionPolicy:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()

    def before_generation(self, chunks: Sequence[ScoredChunk]) -> AbstentionDecision:
        """Pre-generation gates. Abstaining here saves the synthesis call."""
        gen = self.settings.generation
        if not gen.allow_abstention:
            return AbstentionDecision()

        if len(chunks) < gen.min_context_chunks:
            return self._abstain(
                "insufficient_context",
                f"retrieval returned {len(chunks)} chunks, minimum is {gen.min_context_chunks}",
            )

        if chunks and all(not c.chunk.content.strip() for c in chunks):
            return self._abstain("empty_context", "all retrieved chunks are empty")

        return AbstentionDecision()

    def after_generation(
        self,
        *,
        answer_text: str,
        grounded: bool,
        groundedness_score: float,
        contradicted: Sequence[str] = (),
        model_says_insufficient: bool = False,
        invalid_citations: Sequence[int] = (),
    ) -> AbstentionDecision:
        """Post-generation gates, in order of how decisive the signal is."""
        gen = self.settings.generation
        if not gen.allow_abstention:
            return AbstentionDecision()

        if contradicted:
            # The evidence actively denies something the answer asserts. This is
            # the strongest possible reason not to return it.
            return self._abstain(
                "contradicted",
                f"{len(contradicted)} claim(s) contradicted by the evidence",
                confidence=0.0,
            )

        if model_says_insufficient:
            # The generator itself reported that the context could not answer the
            # question. Overriding that judgement to ship an answer anyway is how
            # confident nonsense gets produced.
            return self._abstain(
                "model_reported_insufficient",
                "the generator reported that the context did not contain the answer",
            )

        if gen.check_groundedness and not grounded:
            return self._abstain(
                "ungrounded",
                f"groundedness {groundedness_score:.2f} below threshold "
                f"{gen.groundedness_threshold:.2f}",
                confidence=groundedness_score,
            )

        if invalid_citations:
            # Citing passages that do not exist means the attribution is
            # fabricated, whatever the prose says.
            return self._abstain(
                "fabricated_citations",
                f"answer cites non-existent passages {list(invalid_citations)}",
                confidence=0.0,
            )

        if self._is_evasive(answer_text):
            # Normalize the model's own soft refusal into an explicit abstention,
            # so callers get one signal instead of having to parse prose.
            return self._abstain(
                "self_reported_unknown",
                "the answer states that the information is unavailable",
                confidence=0.3,
            )

        return AbstentionDecision(confidence=groundedness_score)

    # -- helpers -----------------------------------------------------------
    def _abstain(self, gate: str, reason: str, *, confidence: float = 0.0) -> AbstentionDecision:
        log.info("abstained", gate=gate, reason=reason)
        return AbstentionDecision(
            abstain=True,
            gate=gate,
            reason=reason,
            message=self.settings.generation.abstain_message,
            confidence=confidence,
        )

    @staticmethod
    def _is_evasive(text: str) -> bool:
        lowered = text.lower().strip()
        if len(lowered) < 24:
            return True
        markers = (
            "i don't know",
            "i do not know",
            "cannot be determined",
            "not enough information",
            "insufficient information",
            "does not contain",
            "no information about",
            "unable to answer",
            "the context does not",
            "not provided in the context",
            "not mentioned in the",
        )
        # Only count it as evasive when a marker dominates a short answer. A long,
        # substantive answer that also notes one gap is not an abstention.
        return any(m in lowered for m in markers) and len(lowered) < 400
