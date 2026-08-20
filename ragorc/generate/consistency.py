"""Self-consistency: sampling the same question N times and measuring agreement.

The insight (Wang et al., 2022) is that a *correct* answer is reachable by many
reasoning paths while a *hallucinated* one is idiosyncratic. Sample the same
question several times at non-zero temperature: convergent answers indicate the
evidence determines the conclusion, divergent answers indicate the model is
filling gaps differently each time — which is precisely the signature of a
fabricated detail.

This buys a calibrated confidence number that no single greedy generation can
provide, and it costs N times the synthesis call, so it is opt-in
(``generation.self_consistency_samples``) and belongs on high-stakes answers.

Agreement is measured on **claims**, not strings. Two answers can be verbally
different and factually identical; comparing text similarity would score that as
disagreement and comparing embeddings would score two answers with opposite
numbers as agreement. Numbers and entities are extracted and compared directly
for exactly that reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import Usage
from ragorc.core.protocols import LLM
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["ConsistencyResult", "SelfConsistencyChecker"]

_NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")
_ENTITY = re.compile(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*){0,3}\b")
_WORD = re.compile(r"\w+")


@dataclass(slots=True)
class ConsistencyResult:
    answer: str
    """The representative sample — the one closest to the consensus, not simply
    the first. Picking the medoid means the returned text is the one most other
    samples agree with."""
    agreement: float = 1.0
    samples: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    conflicting_numbers: list[str] = field(default_factory=list)
    consistent: bool = True

    def report(self) -> dict[str, Any]:
        """Telemetry payload. ``Any`` rather than ``object`` because this is
        splatted into ``trace_step(**...)``, whose named parameters are typed."""
        return {
            "samples": len(self.samples),
            "agreement": round(self.agreement, 3),
            "consistent": self.consistent,
            "conflicting_numbers": self.conflicting_numbers[:8],
        }


class SelfConsistencyChecker:
    def __init__(
        self, llm: LLM, settings: Settings | None = None, *, temperature: float = 0.7
    ) -> None:
        self.llm = llm
        self.settings: Settings = settings or get_settings()
        self.temperature = temperature

    async def generate(
        self, prompt: str, *, system: str | None = None, model: str | None = None
    ) -> ConsistencyResult:
        n = max(self.settings.generation.self_consistency_samples, 1)
        if n == 1:
            text, usage = await self.llm.complete(
                prompt, system=system, model=model, stage="answer"
            )
            return ConsistencyResult(answer=text, samples=[text], usage=usage)

        # Temperature must be > 0 or every sample is identical and the agreement
        # score is meaningless. This also bypasses the response cache by design.
        results = await bounded_gather(
            (
                self.llm.complete(
                    prompt,
                    system=system,
                    model=model,
                    temperature=self.temperature,
                    stage="answer_sample",
                )
                for _ in range(n)
            ),
            limit=min(n, self.settings.llm.max_concurrency),
            return_exceptions=True,
        )

        samples: list[str] = []
        usages: list[Usage] = []
        for result in results:
            if isinstance(result, BaseException):
                log.warning("consistency_sample_failed", error=str(result)[:160])
                continue
            text, usage = result
            if text.strip():
                samples.append(text)
                usages.append(usage)

        if not samples:
            return ConsistencyResult(answer="", agreement=0.0, consistent=False)
        if len(samples) == 1:
            return ConsistencyResult(
                answer=samples[0], samples=samples, usage=Usage.sum(usages), agreement=1.0
            )

        return self.score(samples, Usage.sum(usages))

    def score(self, samples: Sequence[str], usage: Usage | None = None) -> ConsistencyResult:
        """Compute pairwise agreement and pick the medoid sample."""
        similarity = self._agreement_matrix(samples)
        # Mean agreement with the others, excluding self-similarity.
        n = len(samples)
        off_diagonal = (similarity.sum(axis=1) - np.diag(similarity)) / max(n - 1, 1)
        medoid = int(np.argmax(off_diagonal))
        agreement = float(off_diagonal.mean())

        conflicts = self._number_conflicts(samples)
        threshold = self.settings.generation.self_consistency_threshold
        # A numeric conflict is disqualifying on its own: if three samples give
        # three different figures, at most one of them is right.
        consistent = agreement >= threshold and not conflicts

        if not consistent:
            log.info(
                "low_self_consistency",
                agreement=round(agreement, 3),
                threshold=threshold,
                conflicts=conflicts[:5],
            )
        return ConsistencyResult(
            answer=samples[medoid],
            agreement=agreement,
            samples=list(samples),
            usage=usage or Usage(),
            conflicting_numbers=conflicts,
            consistent=consistent,
        )

    # -- similarity --------------------------------------------------------
    @staticmethod
    def _stem(word: str) -> str:
        """Crude suffix stripping.

        Necessary, not cosmetic: "Processing refunds takes 14 days" and "Refunds
        take 14 days to process" are the *same* answer, but they share almost no
        exact word forms. Without normalization the agreement metric reports two
        paraphrases as a disagreement, which is exactly backwards. A real stemmer
        would be better; this handles the English inflections that actually differ
        between paraphrases, at no dependency cost.
        """
        for suffix in ("ing", "ed", "es", "s", "ly"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    @classmethod
    def _features(cls, text: str) -> tuple[set[str], set[str]]:
        """Split a sample into (lexical stems, normalized numbers).

        Numbers are kept separate and weighted separately, because they are where
        answers disagree substantively while the surrounding prose stays identical.
        """
        numbers: set[str] = set()
        for raw in _NUMBER.findall(text):
            try:
                value = float(raw.replace(",", "").rstrip("%"))
            except ValueError:
                continue
            suffix = "%" if raw.endswith("%") else ""
            numbers.add(f"{value:g}{suffix}")

        stripped = _NUMBER.sub(" ", text)
        lexical = {cls._stem(w.lower()) for w in _WORD.findall(stripped) if len(w) > 3}
        lexical |= {e.lower() for e in _ENTITY.findall(stripped) if len(e) > 3}
        return lexical, numbers

    @classmethod
    def _agreement_matrix(cls, samples: Sequence[str]) -> np.ndarray:
        """Pairwise agreement, as a weighted blend of two signals.

        * **Lexical** — overlap coefficient (intersection over the *smaller* set)
          rather than Jaccard. A terse answer and a verbose one that agree should
          not be penalized for their length difference, which is what Jaccard does.
        * **Numeric** — Jaccard over the figures. Weighted equally with the whole
          of the prose, because a single differing number makes two otherwise
          identical answers incompatible.
        """
        features = [cls._features(text) for text in samples]
        vocab: dict[str, int] = {}
        num_vocab: dict[str, int] = {}
        for lexical, numbers in features:
            for token in lexical:
                vocab.setdefault(token, len(vocab))
            for token in numbers:
                num_vocab.setdefault(token, len(num_vocab))

        n = len(samples)
        lex = np.zeros((n, max(len(vocab), 1)), dtype=np.float32)
        num = np.zeros((n, max(len(num_vocab), 1)), dtype=np.float32)
        for i, (lexical, numbers) in enumerate(features):
            for token in lexical:
                lex[i, vocab[token]] = 1.0
            for token in numbers:
                num[i, num_vocab[token]] = 1.0

        def overlap(matrix: np.ndarray, *, mode: str) -> np.ndarray:
            inter = matrix @ matrix.T
            sizes = matrix.sum(axis=1)
            if mode == "min":
                denom = np.minimum(sizes[:, None], sizes[None, :])
            else:
                denom = sizes[:, None] + sizes[None, :] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                # Two samples that both contain nothing of this kind agree
                # vacuously; scoring them 0 would punish answers with no figures.
                return np.where(denom > 0, inter / denom, 1.0).astype(np.float32)

        lexical_sim = overlap(lex, mode="min")
        numeric_sim = overlap(num, mode="jaccard")
        return (0.5 * lexical_sim + 0.5 * numeric_sim).astype(np.float32)

    @staticmethod
    def _number_conflicts(samples: Sequence[str]) -> list[str]:
        """Figures that disagree across samples within the same magnitude.

        The test is: at least two samples commit to a value in the same magnitude
        bucket, and those values differ. Requiring the value to appear in *every*
        sample would be wrong — "14 days" vs "30 days" vs "7 days" is three
        mutually exclusive answers, and no single value appears throughout.

        Bucketing by magnitude keeps "14 days" vs "3 items" from registering as a
        conflict while catching "14" vs "40".
        """
        per_sample: list[dict[int, set[float]]] = []
        for text in samples:
            buckets: dict[int, set[float]] = {}
            for raw in _NUMBER.findall(text):
                try:
                    value = float(raw.replace(",", "").rstrip("%"))
                except ValueError:
                    continue
                if value == 0:
                    continue
                magnitude = int(np.floor(np.log10(abs(value))))
                buckets.setdefault(magnitude, set()).add(value)
            per_sample.append(buckets)

        magnitudes = {m for buckets in per_sample for m in buckets}
        conflicts: list[str] = []
        for magnitude in sorted(magnitudes):
            committed = [b[magnitude] for b in per_sample if magnitude in b]
            if len(committed) < 2:
                continue
            union: set[float] = set().union(*committed)
            if len(union) > 1:
                conflicts.append(f"~1e{magnitude}: " + ", ".join(f"{v:g}" for v in sorted(union)))
        return conflicts
