"""Outbound validation for answers.

Three independent checks, because they catch different failures:

* **Citation existence** — every ``[n]`` marker must point at a passage that was
  actually retrieved. A model that invents ``[7]`` when six passages were
  supplied has fabricated its evidence, and this check is nearly free.
* **Quote fidelity** — a cited span must appear in the cited chunk. This catches
  the most convincing hallucination there is: a plausible quotation attributed to
  a real document that does not contain it. Comparison is on normalized text, so
  whitespace and smart-quote differences do not cause false alarms.
* **Leakage** — the answer must not carry personal data out, nor the delimiters
  of our own prompt scaffolding. Both are *removed*, not merely reported: this
  section promised "must not contain" for nine rounds while implementing only
  half of it, and the half it did implement set a flag no code read.

The asymmetry the leakage check exists for
------------------------------------------
``enable_pii_redaction`` used to scrub the inbound *question* and nothing else.
The question is written by the caller, who already knows what is in it; the
answer is assembled from retrieved documents, which is where the corpus's
personal data actually lives. So the setting protected the least likely source
and missed the most likely one, and an operator reading "PII redaction: on" had
no way to tell.

Scaffolding is stripped rather than abstained on. A model echoing
``</untrusted_document>`` is usually quoting a document that contains it, not
mounting an attack, and refusing to answer would turn a formatting artifact into
an outage; removing the delimiter costs nothing and closes the case where the
echo *is* an attempt to forge a fence in the next turn's context.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import structlog

from ragorc.core.models import Answer, Citation, ScoredChunk
from ragorc.core.settings import Settings, get_settings
from ragorc.security.pii import PIIRedactor

log = structlog.get_logger(__name__)

__all__ = ["AnswerValidator", "OutputReport", "StreamLeakFilter"]

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


_SCAFFOLD = re.compile(r"</?untrusted_document\b[^>]*>", re.IGNORECASE)
"""Scaffolding *this library* emits, and nothing else.

``system``, ``instruction`` and ``context`` were borrowed from the *inbound*
injection-detection pattern (:mod:`ragorc.security.injection`), where they are
defensive. Here they delete legitimate answer content, and the ``\b`` after the
tag name matches an XML namespace prefix — so an answer written by
``answer_technical``, whose whole job is to reproduce code exactly, came back
mangled::

    model wrote : Add `<context:component-scan base-package="com.acme"/>` to the
                  beans file [1]. Do not use the `<system>` element [1].
    reader gets : Add `` to the beans file [1]. Do not use the `` element [1].

and the answer was flagged ``scaffold_leak`` for containing Spring configuration.
The only tag the packer produces is ``<untrusted_document>``
(:func:`~ragorc.security.injection.wrap_untrusted`), so that is the only one whose
appearance in an answer means the fence leaked.
"""


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
    """Prompt scaffolding appeared in the answer and was stripped from it."""
    pii_entities: list[str] = field(default_factory=list)
    """Entity kinds redacted out of the answer, in the order first seen."""
    citation_coverage: float = 1.0
    """Fraction of the answer's sentences that carry at least one citation."""

    @property
    def redacted(self) -> bool:
        """Whether :meth:`AnswerValidator.validate` rewrote the answer text."""
        return self.scaffold_leak or bool(self.pii_entities)


class AnswerValidator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.redactor = PIIRedactor(self.settings.security)

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

        # --- nothing leaks out of the answer ------------------------------
        # Rewrites `answer.text` in place, which is why it runs before coverage
        # is measured below: coverage counts sentences, and a redaction can
        # remove one.
        self._strip_leaks(answer, report)

        # --- how much of the answer is actually attributed ----------------
        # Measured from the citations when the style does not use inline markers.
        # `citation_style="json"` sends a system block that relocates attribution
        # into `statements` and tells the model *not* to write `[n]`, so counting
        # markers reported 0% on a fully attributed answer and warned about it —
        # a metric that measures the other style's output.
        if gen.citation_style == "json":
            report.citation_coverage = self._coverage_from_citations(answer)
        else:
            report.citation_coverage = self._coverage(answer.text)
        if gen.cite_sources and chunks and report.citation_coverage < 0.5:
            report.warnings.append(f"only {report.citation_coverage:.0%} of sentences are cited")

        if report.warnings:
            log.info("answer_validation", valid=report.valid, warnings=report.warnings)
        return report

    @staticmethod
    def _coverage_from_citations(answer: Answer) -> float:
        """Attributed share, for a style whose attribution is not inline.

        The same quantity `_coverage` computes — what fraction of the answer's
        sentences carry a source — read from `Answer.citations` instead of from
        `[n]` markers, because the JSON style's prompt forbids the markers.

        Each citation carries the claim it attributes, and the generator builds
        those from the model's own statements, so the ratio is over statements
        rather than re-split sentences: splitting again would disagree with the
        split the model was asked to perform.
        """
        if not answer.citations:
            return 0.0
        claims = {c.claim.strip() for c in answer.citations if c.claim.strip()}
        # The same split `_coverage` uses, so the two styles report a comparable
        # number rather than two differently-defined ratios under one name.
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.text.strip()) if len(s) > 15]
        if not sentences:
            return 1.0 if claims else 0.0
        cited = sum(
            1
            for sentence in sentences
            if any(claim in sentence or sentence in claim for claim in claims)
        )
        return round(cited / len(sentences), 4)

    def stream_filter(self) -> StreamLeakFilter:
        """A leak filter for a token stream.

        Groundedness cannot run on a stream — it needs the whole answer, and you
        cannot un-emit a token. The leakage checks are regexes over emitted text
        and have no such constraint, so they were being skipped for a reason that
        only covered the other two checks. ``stream()`` had no outbound validation
        of any kind, which meant ``enable_pii_redaction`` silently applied to
        ``/query`` and not to ``/query/stream``.
        """
        return StreamLeakFilter(
            self.redactor if self.settings.security.enable_pii_redaction else None
        )

    def _strip_leaks(self, answer: Answer, report: OutputReport) -> None:
        """Remove scaffolding and personal data from the answer, and say so.

        Removal rather than detection. The previous version set
        ``report.scaffold_leak`` and appended a warning, and nothing read either:
        the flag had no consumer anywhere in the package, it did not set
        ``report.valid`` to ``False``, it was absent from
        ``answer.metadata["validation"]``, and the markup went to the reader
        unchanged. "Must not contain" describes prevention; that was a log line.

        Neither finding invalidates the answer. ``valid`` gates the groundedness
        check and the abstention path, and an answer that quoted a delimiter or
        mentioned an email address is not therefore ungrounded — failing it would
        spend a retry on a text problem that has already been fixed by the time
        anything reads the flag.
        """
        text = answer.text
        if _SCAFFOLD.search(text):
            report.scaffold_leak = True
            text = _SCAFFOLD.sub("", text)
            report.warnings.append("prompt scaffolding was stripped from the answer")

        if self.settings.security.enable_pii_redaction:
            result = self.redactor.redact(text)
            if result.found:
                report.pii_entities = list(result.entities)
                text = result.text
                report.warnings.append(
                    f"PII redacted from answer: {', '.join(report.pii_entities)}"
                )

        if text != answer.text:
            answer.text = text.strip()

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


#: How much of the stream is held back so a pattern split across two deltas is
#: still seen whole. A provider emits a few characters at a time, so an email
#: address or a `</untrusted_document>` tag routinely straddles a boundary and a
#: filter that scanned each delta alone would miss most of them.
#:
#: 96 characters comfortably exceeds every pattern in `security.pii` and a normal
#: scaffold tag. The bound is real and worth stating: a *longer* construction —
#: `<untrusted_document ` followed by 200 characters of attributes — can still
#: straddle the window and reach the reader. The complete answer is re-checked by
#: `validate()` on the non-streaming path, which is the one to use when this
#: matters.
_STREAM_TAIL = 96

#: Characters no pattern can contain, and which therefore end any partial match.
#: Whitespace is the useful one: a trailing run of non-space characters is the
#: only thing an email, a JWT or an AWS key could still be growing into, so
#: holding back to the last space covers them at a cost of a few characters
#: rather than ninety-six.
_BOUNDARY = frozenset(" \t\n\r")

#: Characters that mean a *whitespace-spanning* pattern may be in progress, where
#: holding to the last space is not enough. Phone numbers, credit cards, IBANs and
#: PEM headers all interleave separators — and all of them contain a digit or a
#: dash. `@` and `<` are here so an email or a fence in flight is never emitted in
#: halves. Deliberately not upper-case letters: they appear at the start of every
#: sentence, and including them held the whole window on ordinary prose, which is
#: how the first version of this function saved nothing at all.
_SPANNING = frozenset("0123456789@<-")


def _hold_back(text: str) -> int:
    """How many trailing characters must be withheld, given what they are.

    A fixed window was correct and expensive. Holding the last 96 characters
    unconditionally cost ~310 ms of time-to-first-token against a provider
    emitting 4-character deltas — on the one path whose docstring says to use it
    "when latency to first token" matters, and by default, because PII redaction
    is off by default while the scaffold check is not.

    The window is only needed while a pattern *could* still be extending. Ordinary
    prose contains none of :data:`_RISKY`, so nothing is held and the stream flows
    at the provider's rate. Once a risky character appears, everything from it
    onward is held until the window is exhausted — which is the same guarantee the
    fixed window gave, applied only where it can matter.
    """
    if _STREAM_TAIL <= 0:
        # `text[-0:]` is `text[0:]`, i.e. the whole string — so a window of zero
        # held *everything* rather than nothing, and the answer arrived only at
        # flush. Harmless at the shipped value and a trap for anyone tuning it
        # down, which is the only reason the constant is worth a branch.
        return 0
    window = text[-_STREAM_TAIL:]
    if any(char in _SPANNING for char in window):
        # Conservative and rare: something in the window could be a pattern that
        # keeps going across a space, so the whole window waits.
        return len(window)
    for offset in range(len(window) - 1, -1, -1):
        if window[offset] in _BOUNDARY:
            return len(window) - offset - 1
    return len(window)



class StreamLeakFilter:
    """Redacts a token stream as it passes, holding back a small tail.

    Stateful and single-use, like the stream it wraps. Feed each delta and emit
    what comes back; call :meth:`flush` once the stream ends to release the tail.
    """

    __slots__ = ("_buffer", "_redactor", "entities", "scaffold_leak")

    def __init__(self, redactor: PIIRedactor | None) -> None:
        self._redactor = redactor
        self._buffer = ""
        self.scaffold_leak = False
        self.entities: list[str] = []

    def feed(self, delta: str) -> str:
        """Absorb one delta and return the part that is safe to emit."""
        self._buffer += delta
        cleaned = self._clean(self._buffer)
        # Measured after cleaning: a redaction changes the length, so an offset
        # computed before it would cut mid-token.
        hold = _hold_back(cleaned)
        if hold >= len(cleaned):
            self._buffer = cleaned
            return ""
        emit, self._buffer = cleaned[: len(cleaned) - hold], cleaned[len(cleaned) - hold :]
        return emit

    def flush(self) -> str:
        """Release what is held back. Called once, when the stream is done."""
        out, self._buffer = self._clean(self._buffer), ""
        return out

    def _clean(self, text: str) -> str:
        if _SCAFFOLD.search(text):
            self.scaffold_leak = True
            text = _SCAFFOLD.sub("", text)
        if self._redactor is not None:
            result = self._redactor.redact(text)
            if result.found:
                for entity in result.entities:
                    if entity not in self.entities:
                        self.entities.append(entity)
                text = result.text
        return text

    @property
    def redacted(self) -> bool:
        return self.scaffold_leak or bool(self.entities)


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
