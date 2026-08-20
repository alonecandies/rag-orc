"""Citation extraction and span-level attribution.

Two levels of attribution, because they answer different questions.

**Marker level** — the ``[3]`` in the answer text. Cheap, and enough to let a
reader follow a claim to a source. Built by mapping markers back to the numbered
passages that were actually packed into the prompt.

**Span level** — *which sentence* of passage 3 supports the claim. This is what
makes a citation verifiable rather than decorative, and it is what the answer
validator checks. Computed lexically rather than with an LLM call: for each cited
sentence, find the source sentence with the highest weighted token overlap, where
rare tokens count for more. That is cheap enough to run on every answer, which an
LLM attribution pass would not be.

Rare-token weighting matters. Unweighted overlap picks whichever source sentence
shares the most stopwords, which is usually the longest one; weighting by inverse
frequency picks the sentence that shares the *distinctive* terms — names, figures,
identifiers — which is almost always the right one.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import structlog

from ragorc.core.models import Chunk, Citation, ScoredChunk

log = structlog.get_logger(__name__)

__all__ = ["attribute_spans", "extract_citations", "renumber_citations"]

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[\w'/-]+")
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "its",
        "their",
        "there",
        "here",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "when",
        "where",
        "how",
        "why",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "can",
        "could",
        "may",
        "might",
        "will",
        "would",
        "shall",
        "should",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "also",
        "more",
        "most",
        "other",
        "some",
        "any",
        "each",
    ]
)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP and len(w) > 1]


def _base_offset(chunk: Chunk) -> int:
    """Document offset of the text the generator actually saw.

    ``chunk.start_char`` describes the chunk as it was *indexed*. The context
    packer may have replaced ``content`` with a wider span — the sentence window,
    or the parent document — and after that swap the chunk's own offsets no longer
    describe what was in the prompt. Adding a within-quote offset to them lands on
    unrelated text: measured a citation for "The fee is 3 percent." reported
    (90, 111), which sliced to ``'unts apply in Q4 only'``.

    Both splitters that widen a chunk record the correct base for exactly this
    purpose — ``window_start`` and ``parent_start_char`` — so this prefers them
    and falls back to the chunk's own offset when no expansion happened.
    """
    metadata = getattr(chunk, "metadata", None) or {}
    for key in ("parent_start_char", "window_start"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
    return int(getattr(chunk, "start_char", 0) or 0)


def extract_citations(
    answer_text: str, chunks: Sequence[ScoredChunk], *, attribute: bool = True
) -> list[Citation]:
    """Build :class:`Citation` objects from ``[n]`` markers in the answer.

    Handles the grouped form ``[1, 3]`` that models produce despite being asked
    for single markers, and ignores out-of-range markers — those are handled (and
    reported) by the answer validator, not silently resolved to something else.
    """
    if not chunks:
        return []

    sentences = _split_sentences(answer_text)
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()

    for sentence in sentences:
        indices: list[int] = []
        for match in _CITATION.finditer(sentence):
            for part in match.group(1).split(","):
                try:
                    indices.append(int(part.strip()) - 1)
                except ValueError:
                    continue
        if not indices:
            continue
        claim = _CITATION.sub("", sentence).strip()
        for index in indices:
            if not 0 <= index < len(chunks):
                continue
            chunk = chunks[index].chunk
            key = (chunk.id, claim[:80])
            if key in seen:
                continue
            seen.add(key)
            quote, start, end, support = ("", None, None, 1.0)
            if attribute and claim:
                quote, start, end, support = attribute_spans(claim, chunk.content)
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    quote=quote,
                    claim=claim,
                    support=support,
                    source=chunk.metadata.get("source") or chunk.metadata.get("title"),
                    start_char=(_base_offset(chunk) + start) if start is not None else None,
                    end_char=(_base_offset(chunk) + end) if end is not None else None,
                )
            )
    return citations


def attribute_spans(claim: str, source: str) -> tuple[str, int | None, int | None, float]:
    """Find the sentence in ``source`` that best supports ``claim``.

    Returns ``(quote, start_offset, end_offset, support)`` where ``support`` is
    the weighted overlap in [0, 1]. Offsets are relative to ``source``, so the
    caller can turn them into absolute document offsets.
    """
    claim_tokens = set(_tokens(claim))
    if not claim_tokens:
        return "", None, None, 0.0
    spans = _sentence_spans(source)
    if not spans:
        return "", None, None, 0.0

    # Inverse-frequency weights over the source's own sentences: a token that
    # appears in every sentence discriminates nothing.
    frequency: Counter[str] = Counter()
    per_span: list[set[str]] = []
    for start, end in spans:
        tokens = set(_tokens(source[start:end]))
        per_span.append(tokens)
        frequency.update(tokens)
    total = len(spans)
    weights = {token: math.log(1 + total / count) for token, count in frequency.items()}

    best_index = -1
    best_score = 0.0
    claim_weight = sum(weights.get(t, math.log(1 + total)) for t in claim_tokens) or 1.0
    for i, tokens in enumerate(per_span):
        shared = claim_tokens & tokens
        if not shared:
            continue
        score = sum(weights.get(t, 0.0) for t in shared) / claim_weight
        if score > best_score:
            best_score, best_index = score, i

    if best_index < 0:
        return "", None, None, 0.0
    start, end = spans[best_index]
    return source[start:end].strip(), start, end, min(best_score, 1.0)


def renumber_citations(answer_text: str, keep: Sequence[int]) -> str:
    """Rewrite markers after chunks are dropped, so numbering stays contiguous.

    Needed because compression and noise filtering can remove a passage *after*
    the answer was generated. Leaving a gap ("[1] … [4]") or, worse, letting the
    old numbers point at the new list, breaks attribution silently.
    """
    mapping = {old + 1: new + 1 for new, old in enumerate(keep)}

    def rewrite(match: re.Match[str]) -> str:
        out: list[str] = []
        for part in match.group(1).split(","):
            try:
                old = int(part.strip())
            except ValueError:
                continue
            if old in mapping:
                out.append(str(mapping[old]))
        return f"[{', '.join(out)}]" if out else ""

    return _CITATION.sub(rewrite, answer_text)


# ---------------------------------------------------------------------------
def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence boundaries as exact offsets into ``text``."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in _SENTENCE.split(text):
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start == -1:
            continue
        end = start + len(piece)
        spans.append((start, end))
        cursor = end
    return spans
