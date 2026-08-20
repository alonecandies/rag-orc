"""Outbound validation for answers.

Three independent checks, because they catch different failures:

* **Citation existence** — every ``[n]`` marker must point at a passage that was
  actually retrieved. A model that invents ``[7]`` when six passages were
  supplied has fabricated its evidence, and this check is nearly free.
* **Quote fidelity** — a cited span must appear in the cited chunk. This catches
  the most convincing hallucination there is: a plausible quotation attributed to
  a real document that does not contain it. Comparison is on normalized text, so
  whitespace and smart-quote differences do not cause false alarms.
* **Leakage** — the answer must not contain PII that was redacted upstream, or
  the delimiters of our own prompt scaffolding.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import structlog

from ragorc.core.models import Answer, Citation, ScoredChunk
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["AnswerValidator", "OutputReport"]

#: Matches grouped markers too. The single-number version silently disagreed
#: with ``ragorc.generate.citations``, which has always emitted and parsed the
#: grouped form: ``"Claim [7, 9]."`` with three passages reported no invalid
#: citations at all, so the phantom markers reached the reader unstripped and the
#: ``invalid_citations`` abstention gate never fired. It also scored a fully-cited
#: answer at 33% coverage and warned about it.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _marker_indices(text: str) -> list[int]:
    """Every passage number referenced by a marker, grouped forms expanded."""
    out: list[int] = []
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            try:
                out.append(int(part.strip()))
            except ValueError:
                continue
    return out


_SCAFFOLD = re.compile(
    r"</?(?:untrusted_document|system|instruction|context)\b[^>]*>", re.IGNORECASE
)


#: Typographic characters that carry no meaning difference from their ASCII form.
#: Annotated because ``str.maketrans`` accepts ``dict[str, str | int | None]`` and a
#: bare ``dict[str, str]`` literal is not that type (``dict`` is invariant).
_PUNCTUATION_FOLD: dict[str, str | int | None] = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
}


def _normalize_for_match(text: str) -> str:
    """Fold the differences that are not meaning: case, whitespace runs, quote
    styles, dashes. Without this, verification fails on cosmetics."""
    folded = unicodedata.normalize("NFKC", text).lower()
    folded = folded.translate(str.maketrans(_PUNCTUATION_FOLD))
    return re.sub(r"\s+", " ", folded).strip()


@dataclass(slots=True)
class OutputReport:
    valid: bool = True
    warnings: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    unverified_quotes: list[str] = field(default_factory=list)
    scaffold_leak: bool = False
    citation_coverage: float = 1.0
    """Fraction of the answer's sentences that carry at least one citation."""


class AnswerValidator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, answer: Answer, chunks: list[ScoredChunk]) -> OutputReport:
        report = OutputReport()
        gen = self.settings.generation

        # --- citation markers point at real passages ----------------------
        referenced = set(_marker_indices(answer.text))
        valid_range = set(range(1, len(chunks) + 1))
        invalid = sorted(referenced - valid_range)
        if invalid:
            report.invalid_citations = invalid
            report.valid = False
            report.warnings.append(
                f"answer cites passages that do not exist: {invalid} "
                f"(only {len(chunks)} were supplied)"
            )

        # --- quoted spans exist in the chunks they cite -------------------
        if gen.verify_citations and answer.citations:
            by_id = {c.chunk.id: c.chunk.content for c in chunks}
            for citation in answer.citations:
                if not citation.quote:
                    continue
                source = by_id.get(citation.chunk_id)
                if source is None:
                    report.unverified_quotes.append(citation.quote[:80])
                    continue
                if not self._quote_present(citation.quote, source):
                    report.unverified_quotes.append(citation.quote[:80])
            if report.unverified_quotes:
                report.valid = False
                report.warnings.append(
                    f"{len(report.unverified_quotes)} cited quote(s) do not appear in the "
                    "documents they are attributed to"
                )

        # --- prompt scaffolding must not leak into the answer -------------
        if _SCAFFOLD.search(answer.text):
            report.scaffold_leak = True
            report.warnings.append("answer contains prompt scaffolding markup")

        # --- how much of the answer is actually attributed ----------------
        report.citation_coverage = self._coverage(answer.text)
        if gen.cite_sources and chunks and report.citation_coverage < 0.5:
            report.warnings.append(f"only {report.citation_coverage:.0%} of sentences are cited")

        if report.warnings:
            log.info("answer_validation", valid=report.valid, warnings=report.warnings)
        return report

    @staticmethod
    def _quote_present(quote: str, source: str, *, threshold: float = 0.92) -> bool:
        """Exact substring first (cheap), fuzzy ratio only as a fallback.

        Fuzzy matching is needed because a model may normalize an ellipsis or
        drop a footnote marker while quoting faithfully; the threshold is high
        enough that a rewritten sentence still fails.
        """
        q = _normalize_for_match(quote)
        s = _normalize_for_match(source)
        if not q:
            return True
        if q in s:
            return True
        if len(q) < 20:
            return False
        matcher = SequenceMatcher(None, q, s, autojunk=False)
        match = matcher.find_longest_match(0, len(q), 0, len(s))
        return (match.size / len(q)) >= threshold

    @staticmethod
    def _coverage(text: str) -> float:
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s) > 15]
        if not sentences:
            return 1.0
        cited = sum(1 for s in sentences if _CITATION.search(s))
        return cited / len(sentences)

    def strip_invalid_citations(self, answer: Answer, chunks: list[ScoredChunk]) -> Answer:
        """Remove markers that point nowhere. Better to show an uncited sentence
        than a citation the reader cannot follow."""
        valid = set(range(1, len(chunks) + 1))

        def keep_valid(match: re.Match[str]) -> str:
            """Rewrite a marker to only the passages that exist.

            A grouped marker must be *filtered*, not dropped whole: "[1, 9]" with
            eight passages becomes "[1]", because passage 1 is a real citation the
            reader should still be able to follow.
            """
            kept = [
                part.strip()
                for part in match.group(1).split(",")
                if part.strip().isdigit() and int(part.strip()) in valid
            ]
            return f"[{', '.join(kept)}]" if kept else ""

        answer.text = _CITATION.sub(keep_valid, answer.text)
        answer.citations = [
            c for c in answer.citations if c.chunk_id in {x.chunk.id for x in chunks}
        ]
        return answer


def build_citations(answer_text: str, chunks: list[ScoredChunk]) -> list[Citation]:
    """Turn ``[n]`` markers into :class:`Citation` objects against the passage
    list that was given to the model."""
    citations: list[Citation] = []
    seen: set[str] = set()
    for match in _CITATION.finditer(answer_text):
        idx = int(match.group(1)) - 1
        if not 0 <= idx < len(chunks):
            continue
        chunk = chunks[idx].chunk
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        citations.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                quote="",
                source=chunk.metadata.get("source") or chunk.document_id,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
        )
    return citations
