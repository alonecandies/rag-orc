"""Shared splitter machinery.

Why a splitter returns spans and nothing else
---------------------------------------------
Chunking sets the ceiling on retrieval quality: nothing downstream can recover
information destroyed at split time. ADR-0002 adds a second constraint that
shapes this whole package — **a splitter must not embed**. Late chunking embeds
the *document* once and then mean-pools the token vectors belonging to each
chunk, so it needs the boundaries before any vector exists. A splitter that
embedded as it split would foreclose the default strategy of the library (and
cost one forward pass per chunk instead of one per document).

So the output of every splitter is a :class:`~ragorc.core.models.Chunk` carrying
``content``, ``index``, ``start_char``, ``end_char`` and **no vectors**.

``start_char``/``end_char`` are load-bearing, not decoration
------------------------------------------------------------
The late-chunking pooler converts a chunk's character span into a token span in
the document's single forward pass. An offset that is off by one character
silently pools the wrong tokens, which is the worst class of bug in this
codebase: no exception, no log line, a few points of recall gone. Two rules
follow, and both are enforced here rather than trusted to each strategy:

* Offsets are only ever *moved*, never recomputed from a rebuilt string. Content
  is always ``document.content[start:end]`` — including whitespace trimming,
  which advances the offsets instead of stripping the slice.
* :meth:`BaseSplitter._make_chunks` re-derives every slice and compares it to the
  chunk content before returning. One string compare per chunk is nothing next
  to tokenization, and it turns a silent pooling bug into a startup failure.

Why the size enforcement lives in the base class
------------------------------------------------
Every strategy has the same three failure modes at the edges — a runt chunk, an
oversized chunk, and a boundary that cuts an answer in half — so they are solved
once:

* :func:`merge_small_spans` folds adjacent runts together. A 20-character chunk
  is noise in the index that can still win a similarity contest: it has almost no
  content to dilute the query match, so it scores high, displaces a real answer
  and tells the generator nothing.
* :func:`split_oversized_spans` cuts anything over ``max_chunk_size`` at the best
  nearby whitespace boundary — except spans marked ``atomic`` (a fenced code
  block, a table), which are carried whole because half a table is not half as
  useful, it is useless.
* :func:`apply_overlap` extends each span backwards into its predecessor.
  Overlap costs storage and buys the boundary case where the answer straddles a
  cut, which no retriever can recover from afterwards.

Token counts are computed with one batched ``tiktoken`` call per document
(``encode_batch`` releases the GIL and parallelizes in Rust). Counting per chunk
in a Python loop is several times slower for the identical result.
"""

from __future__ import annotations

import asyncio
import itertools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

import structlog

from ragorc.core.concurrency import map_concurrent
from ragorc.core.errors import ValidationFailed
from ragorc.core.ids import chunk_id
from ragorc.core.models import Chunk, Document, Modality
from ragorc.core.settings import IndexingSettings, Settings, get_settings
from ragorc.core.tokens import count_tokens_batch

R = TypeVar("R")

log = structlog.get_logger(__name__)

__all__ = [
    "UNBOUNDED_OVERLAP",
    "BaseSplitter",
    "Span",
    "apply_overlap",
    "merge_small_spans",
    "normalize_spans",
    "split_oversized_spans",
    "split_sentences",
]

UNBOUNDED_OVERLAP = -1
"""Sentinel for :attr:`BaseSplitter.allowed_overlap_chars`.

A splitter whose overlap is specified in *tokens* cannot state a character bound
for it, so it opts out of the character-space check and keeps only the structural
one (spans strictly increasing, never nested)."""

_OFFLOAD_MIN_CHARS = 20_000
"""Above this document size, span planning and tokenization move to a worker
thread. Below it the work is tens of microseconds and ``asyncio.to_thread``'s own
overhead would dominate — offloading everything makes small documents slower."""


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Span:
    """A candidate chunk, expressed purely as offsets into the document.

    Everything a strategy knows but the base class cannot infer travels here:
    ``atomic`` protects code fences and tables from being cut, ``group`` stops
    :func:`merge_small_spans` from welding two different markdown sections (or
    two different classes) into one chunk, and ``prefix`` becomes
    :attr:`Chunk.contextual_prefix` so a heading path is embedded with the chunk
    without being spliced into ``content`` — which would break the invariant
    ``content == document.content[start:end]``.
    """

    start: int
    end: int
    atomic: bool = False
    group: str = ""
    prefix: str | None = None
    modality: Modality | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start

    def merged_with(self, other: Span) -> Span:
        """Fold a following span into this one.

        This span's metadata wins on conflict — it is the earlier, more general
        context (the section a runt belongs to, the class a tiny method sits in).
        ``merged_spans`` records how many units the result covers so a downstream
        consumer can tell a single definition from four packed together.
        """
        merged = dict(other.metadata)
        merged.update(self.metadata)
        merged["merged_spans"] = int(self.metadata.get("merged_spans", 1)) + int(
            other.metadata.get("merged_spans", 1)
        )
        return Span(
            start=self.start,
            end=other.end,
            atomic=self.atomic or other.atomic,
            group=self.group,
            prefix=self.prefix,
            modality=self.modality if self.modality == other.modality else None,
            metadata=merged,
        )


def _as_span(raw: Span | tuple[int, int]) -> Span:
    """Accept ``(start, end)`` tuples so a strategy can stay offset-only."""
    if isinstance(raw, Span):
        return replace(raw, metadata=dict(raw.metadata))
    start, end = raw
    return Span(int(start), int(end))


def normalize_spans(spans: Iterable[Span | tuple[int, int]], text: str) -> list[Span]:
    """Clamp, whitespace-trim, order and de-nest spans. Idempotent.

    Trimming moves the offsets inwards rather than stripping the slice, so
    ``text[start:end]`` still reproduces the content exactly. Spans fully
    contained in their predecessor are dropped: they are duplicates in the index
    and they break the strictly-increasing invariant the validator relies on.
    """
    limit = len(text)
    cleaned: list[Span] = []
    for raw in spans:
        span = _as_span(raw)
        start = max(0, min(span.start, limit))
        end = max(0, min(span.end, limit))
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            continue
        span.start, span.end = start, end
        cleaned.append(span)

    cleaned.sort(key=lambda s: (s.start, s.end))
    out: list[Span] = []
    for span in cleaned:
        if not out:
            out.append(span)
            continue
        previous = out[-1]
        if span.end <= previous.end:
            continue  # nested inside its predecessor: a duplicate in the index
        if span.start <= previous.start:
            out[-1] = span  # same start, wider: keep the one that loses no text
            continue
        out.append(span)
    return out


def merge_small_spans(
    spans: Sequence[Span], *, min_size: int, max_size: int | None = None
) -> list[Span]:
    """Fold adjacent spans that fall under ``min_size`` characters into one.

    A 20-character chunk is noise in the index that can still win a similarity
    contest — there is nothing in it to dilute the match, so it outscores the
    paragraph that actually answers the question and then tells the generator
    nothing. Merging is the cheap fix and it is strictly better than dropping the
    runt, which would lose text.

    Two guards keep merging honest:

    * ``max_size`` — merging must not manufacture an oversized chunk that the
      next stage then has to cut again;
    * ``group`` equality and ``atomic`` — a runt paragraph never absorbs into a
      different markdown section, and never into a code fence.

    Setting ``min_size == max_size == chunk_size`` turns this into greedy packing
    to a target size, which is exactly what the recursive strategy needs after it
    has cut a document into sentence-sized pieces.
    """
    if min_size <= 0:
        return list(spans)
    out: list[Span] = []
    for span in spans:
        if out:
            prev = out[-1]
            adjacent = span.start >= prev.end
            small = prev.length < min_size or span.length < min_size
            compatible = not prev.atomic and not span.atomic and prev.group == span.group
            if adjacent and small and compatible:
                merged_length = span.end - prev.start
                if max_size is None or merged_length <= max_size:
                    out[-1] = prev.merged_with(span)
                    continue
        out.append(span)
    return out


def split_oversized_spans(
    spans: Sequence[Span], text: str, *, max_size: int, target: int | None = None
) -> list[Span]:
    """Cut any span longer than ``max_size`` at the best nearby text boundary.

    ``atomic`` spans are exempt and instead get an ``oversized`` marker in their
    metadata: a fenced code block or a markdown table split in two produces two
    chunks that are each syntactically meaningless, so the right answer is to
    carry it whole and let the context packer decide whether it fits.
    """
    if max_size <= 0:
        return list(spans)
    goal = min(target or max_size, max_size)
    out: list[Span] = []
    for span in spans:
        if span.length <= max_size:
            out.append(span)
        elif span.atomic:
            out.append(replace(span, metadata={**span.metadata, "oversized": True}))
        else:
            out.extend(_cut_to_size(span, text, max_size=max_size, goal=max(goal, 1)))
    return out


def _cut_to_size(span: Span, text: str, *, max_size: int, goal: int) -> list[Span]:
    pieces: list[Span] = []
    cursor = span.start
    while span.end - cursor > max_size:
        floor = cursor + max(goal // 2, 1)
        ceiling = min(cursor + max_size, span.end)
        cut = _best_cut(text, floor, min(cursor + goal, ceiling), ceiling)
        pieces.append(replace(span, start=cursor, end=cut, metadata=dict(span.metadata)))
        cursor = cut
    pieces.append(replace(span, start=cursor, end=span.end, metadata=dict(span.metadata)))
    return pieces


def _best_cut(text: str, floor: int, ideal: int, ceiling: int) -> int:
    """Last natural boundary at or before ``ideal``, widening to ``ceiling``.

    Preference order is paragraph, line, sentence, word. The separator stays with
    the piece that precedes it, which is what keeps a sentence's terminator
    attached to the sentence.
    """
    for sep in ("\n\n", "\n", ". ", " "):
        found = text.rfind(sep, floor, max(ideal, floor + 1))
        if found == -1:
            found = text.rfind(sep, floor, ceiling)
        if found > floor:
            return min(found + len(sep), ceiling)
    return ceiling


def apply_overlap(spans: Sequence[Span], text: str, *, overlap: int) -> list[Span]:
    """Extend every span backwards into its predecessor by up to ``overlap``.

    Overlap is insurance against the one failure retrieval cannot recover from:
    the sentence that answers the question straddling a boundary, so neither
    chunk contains the whole answer and neither one ranks. It is applied
    backwards only (the tail of chunk *i-1* is repeated at the head of chunk *i*)
    and snapped forward to a word boundary, so a chunk never begins mid-word.

    The extension stops one character short of the predecessor's start, so no
    chunk can ever contain another — nesting would put two near-identical
    vectors in the index and break the ordering invariant the validator checks.

    Two boundaries are never crossed: an ``atomic`` block (borrowing the tail of
    a code fence adds a syntax fragment, not context) and a change of ``group``.
    The second matters for structured input: a chunk that declares
    ``heading_path = "Guide > Install"`` must not contain text from the previous
    section, or the context travelling with it is a lie.
    """
    if overlap <= 0 or len(spans) < 2:
        return list(spans)
    out: list[Span] = [spans[0]]
    for prev, span in itertools.pairwise(spans):
        if span.atomic or prev.atomic or prev.group != span.group:
            out.append(span)
            continue
        start = max(prev.start + 1, span.start - overlap)
        if start < span.start:
            start = _snap_forward(text, start, span.start)
        out.append(replace(span, start=start) if start < span.start else span)
    return out


def _snap_forward(text: str, start: int, limit: int) -> int:
    """Move ``start`` to just after the first whitespace before ``limit``."""
    for index in range(start, limit):
        if text[index].isspace():
            return index + 1
    return start


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------
#: Abbreviations whose trailing period is not a sentence end. Kept case-folded
#: and dot-stripped, so "Dr", "dr." and "DR." all hit the same entry.
_ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "mx", "prof", "rev", "hon", "pres", "gov", "sen", "rep",
        "sr", "jr", "st", "mt", "ft", "gen", "col", "lt", "sgt", "capt", "cmdr", "adm",
        "inc", "ltd", "llc", "llp", "plc", "co", "corp", "dept", "div", "univ", "assn",
        "e.g", "i.e", "etc", "vs", "viz", "cf", "al", "ibid", "op", "seq",
        "fig", "figs", "eq", "eqs", "ref", "refs", "vol", "vols", "no", "nos", "pp",
        "ed", "eds", "trans", "approx", "est", "min", "max", "avg", "std", "var",
        "a.m", "p.m", "u.s", "u.k", "u.n", "e.u", "ph.d", "m.d", "b.a", "m.a", "b.s", "m.s",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
        "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    }
)  # fmt: skip

#: Closing punctuation that belongs to the sentence it terminates: straight and
#: curly quotes plus brackets. Written as escapes because a literal curly quote
#: is an ambiguous-unicode lint violation, and rightly so.
_CLOSERS = "[\"'\u2019\u201d\u00bb\\)\\]\\}]*"

#: A terminator run (so an ellipsis is one boundary, not three) followed by its
#: closers and whitespace. The lookahead keeps the whitespace out of the match,
#: which is what makes the boundary offset exact.
_TERMINATOR = re.compile("[.!?\\u2026]+" + _CLOSERS + r"(?=\s)")

#: A blank line ends a sentence even with no terminator — headings, list items
#: and table rows never end in a period, and treating a whole section as one
#: sentence defeats semantic chunking entirely.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")

#: The token immediately before a candidate period: a word (possibly internally
#: dotted, as in "e.g" or "U.S") or a run of digits.
_TRAILING_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*|\d+)$")

_LOOKBEHIND = 48
_LOOKAHEAD = 32
"""Bounded windows for the boundary tests. Slicing ``text[:start]`` and
``text[end:]`` instead — the obvious implementation — copies a prefix and a
suffix of the *document* per candidate boundary, which turns sentence
segmentation into O(n²): a 500KB document has ~12k candidates and would copy
gigabytes. No abbreviation is 48 characters long, so the window costs nothing in
accuracy."""


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Segment text into contiguous sentence spans with exact offsets.

    Contiguous on purpose: span *i+1* starts where span *i* ends, so any group of
    consecutive sentences is itself one exact span. The semantic splitter groups
    sentences and the sentence-window splitter emits them one at a time; both
    need to hand the result straight to :meth:`BaseSplitter._make_chunks`.

    Segmentation is regex-based, and deliberately biased towards *missing* a
    boundary rather than inventing one. A missed boundary makes one chunk
    slightly longer; a false boundary cuts "Dr. Smith" or "3.14" in half, and
    that damage is permanent — the index never sees the original again. So the
    candidate periods rejected here are: known abbreviations, single-letter
    initials (``J. Smith``, ``U.S.``, ``e.g.``), list markers (``1. Overview``)
    and anything followed by a lowercase word, which is a continuation.

    A dedicated sentence tokenizer (pysbd, spaCy) is better at this. It is also a
    heavyweight dependency for a component in the ingest hot path, and the
    failure it prevents — an occasional over-long chunk — is one the size
    enforcement already bounds.
    """
    if not text:
        return []

    boundaries = {match.end() for match in _PARAGRAPH_BREAK.finditer(text)}
    boundaries.update(
        match.end()
        for match in _TERMINATOR.finditer(text)
        if not _is_false_boundary(text, match.start(), match.end())
    )

    spans: list[tuple[int, int]] = []
    cursor = 0
    for boundary in sorted(boundaries):
        if boundary <= cursor or boundary >= len(text):
            continue
        # A whitespace-only piece is not a sentence; leave the cursor where it is
        # so the whitespace attaches to the next real one and coverage stays
        # contiguous.
        if text[cursor:boundary].strip():
            spans.append((cursor, boundary))
            cursor = boundary
    if text[cursor:].strip():
        spans.append((cursor, len(text)))
    return spans


def _is_false_boundary(text: str, run_start: int, run_end: int) -> bool:
    """True when a terminator run does not actually end a sentence."""
    following = text[run_end : run_end + _LOOKAHEAD].lstrip()
    if following and following[0].islower():
        return True  # continuation: "... 5 p.m. and then we left"
    if text[run_start] != ".":
        return False  # "!" and "?" are unambiguous; an ellipsis run ends here
    token = _TRAILING_TOKEN.search(text, max(run_start - _LOOKBEHIND, 0), run_start)
    if token is None:
        return False
    word = token.group(0)
    if word.isdigit():
        return True  # "1. Overview" is a list marker, "Section 2." a reference
    folded = word.strip(".").casefold()
    tail = folded.rsplit(".", 1)[-1]
    if len(tail) <= 1:
        return True  # initials: "J. Smith", "U.S.", "e.g."
    return folded in _ABBREVIATIONS or tail in _ABBREVIATIONS


# ---------------------------------------------------------------------------
# Base splitter
# ---------------------------------------------------------------------------
class BaseSplitter:
    """Template method shared by every strategy.

    A subclass implements exactly one thing — :meth:`_spans_sync`, which proposes
    candidate boundaries — and inherits offset normalization, size enforcement,
    overlap, chunk-id assignment, batched token counting, validation and bounded
    fan-out over documents. Strategies that need I/O to plan (the semantic
    splitter awaits an embedder) override the async :meth:`_spans` instead.

    The size knobs are properties rather than constructor arguments so a strategy
    can opt out of one with a documented reason: the token splitter has no
    character ceiling because its ceiling is in tokens, and the sentence-window
    splitter has no minimum because a one-sentence chunk *is* the design.
    """

    name: str = "base"
    requires_embedder: bool = False
    """Read by :func:`ragorc.index.split.build_splitter` to decide whether the
    factory can construct this splitter from settings alone."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config: IndexingSettings = self.settings.indexing

    # -- size policy, overridable per strategy ------------------------------
    @property
    def target_chars(self) -> int:
        return self.config.chunk_size

    @property
    def min_chars(self) -> int:
        return self.config.min_chunk_size

    @property
    def max_chars(self) -> int:
        return self.config.max_chunk_size

    @property
    def overlap_chars(self) -> int:
        return self.config.chunk_overlap

    @property
    def allowed_overlap_chars(self) -> int:
        """Overlap the validator will tolerate. ``UNBOUNDED_OVERLAP`` disables
        the character-space check for splitters that overlap in tokens."""
        return self.overlap_chars

    # -- the seam a strategy implements -------------------------------------
    def _spans_sync(self, document: Document) -> list[Span]:
        raise NotImplementedError(f"{type(self).__name__} must implement _spans_sync(document)")

    async def _spans(self, document: Document) -> list[Span]:
        return await self._offload(len(document.content), self._spans_sync, document)

    # -- public API ---------------------------------------------------------
    async def split(self, document: Document) -> list[Chunk]:
        """Boundaries only: the returned chunks carry no vectors (ADR-0002)."""
        text = document.content
        if not text or not text.strip():
            return []
        spans = await self._spans(document)
        chunks = await self._offload(len(text), self._assemble, document, spans)
        log.debug(
            "document_split",
            splitter=self.name,
            document_id=document.id,
            chars=len(text),
            chunks=len(chunks),
        )
        return chunks

    async def split_many(self, documents: Sequence[Document]) -> list[Chunk]:
        """Split documents concurrently, bounded by ``max_concurrent_documents``.

        Bounded because the fan-out is over a whole corpus: an unbounded gather
        over 100k documents holds every chunk list of every document in memory at
        once, and the semantic strategy would also fire 100k embedding batches.

        Failures propagate rather than being swallowed per document. Splitting is
        deterministic local computation, so an exception here is a bug in this
        package — and a document silently missing from the index is the kind of
        defect nobody notices for months.
        """
        if not documents:
            return []
        batches = await map_concurrent(
            self.split, list(documents), limit=max(1, self.config.max_concurrent_documents)
        )
        chunks = [chunk for batch in batches for chunk in batch]
        log.info(
            "documents_split", splitter=self.name, documents=len(documents), chunks=len(chunks)
        )
        return chunks

    # -- assembly -----------------------------------------------------------
    def _assemble(self, document: Document, spans: Sequence[Span]) -> list[Chunk]:
        text = document.content
        refined = self._refine(normalize_spans(spans, text), text)
        return self._make_chunks(document, refined)

    def _refine(self, spans: Sequence[Span], text: str) -> list[Span]:
        """Apply the size policy: merge runts, cut giants, then overlap."""
        refined = merge_small_spans(spans, min_size=self.min_chars, max_size=self.max_chars)
        if self.max_chars > 0:
            refined = split_oversized_spans(
                refined, text, max_size=self.max_chars, target=self.target_chars
            )
        if self.overlap_chars > 0:
            refined = apply_overlap(refined, text, overlap=self.overlap_chars)
        return normalize_spans(refined, text)

    def _make_chunks(
        self, document: Document, spans: Sequence[Span | tuple[int, int]]
    ) -> list[Chunk]:
        """Build validated :class:`Chunk` objects from ``(start, end)`` offsets.

        Accepts bare tuples as well as :class:`Span`, so a strategy can stay
        purely offset-based. Token counts come from one batched encode over all
        chunk contents; ids come from
        :func:`ragorc.core.ids.chunk_id`, which folds the content in so an edited
        chunk gets a new id and the stale vector is replaced on re-ingest.
        """
        text = document.content
        prepared = normalize_spans(spans, text)
        if not prepared:
            return []
        self._validate(prepared, document)

        contents = [text[span.start : span.end] for span in prepared]
        token_counts = count_tokens_batch(contents)
        chunks: list[Chunk] = []
        for index, (span, content, tokens) in enumerate(
            zip(prepared, contents, token_counts, strict=True)
        ):
            metadata = dict(span.metadata)
            metadata["splitter"] = self.name
            # Carry the document's provenance onto every chunk. Three things read
            # it and all of them are downstream of here, so deriving it later is
            # not possible: the context packer prints it as the source line beside
            # each numbered passage, citations resolve to it, and document-level
            # eval grading matches against it. A chunk that cannot say where it
            # came from produces an answer whose citations a reader cannot follow.
            # Span metadata wins — a markdown or code splitter that already set a
            # more specific source knows better than the document does.
            if document.source and "source" not in metadata:
                metadata["source"] = document.source
            if document.title and "title" not in metadata:
                metadata["title"] = document.title
            chunks.append(
                Chunk(
                    id=chunk_id(document.id, index, content),
                    content=content,
                    document_id=document.id,
                    index=index,
                    start_char=span.start,
                    end_char=span.end,
                    metadata=metadata,
                    modality=span.modality or document.modality,
                    contextual_prefix=span.prefix,
                    token_count=tokens,
                    tenant_id=document.tenant_id,
                )
            )

        # The invariant late chunking depends on, checked once per chunk. Cheap
        # next to tokenization, and it converts a silent mis-pooling into a loud
        # failure at ingest time.
        broken = next(
            (c for c in chunks if text[c.start_char : c.end_char] != c.content),
            None,
        )
        if broken is not None:
            raise ValidationFailed(
                "chunk content does not match its document offsets",
                splitter=self.name,
                document_id=document.id,
                chunk_index=broken.index,
                start_char=broken.start_char,
                end_char=broken.end_char,
            )
        return chunks

    def _validate(self, spans: Sequence[Span], document: Document) -> None:
        """Check offsets are in bounds, ordered, non-nested and only overlapping
        as far as the configured overlap allows."""
        limit = len(document.content)
        allowed = self.allowed_overlap_chars
        previous: Span | None = None
        for span in spans:
            if not (0 <= span.start < span.end <= limit):
                raise ValidationFailed(
                    "chunk offsets outside the document",
                    splitter=self.name,
                    document_id=document.id,
                    start_char=span.start,
                    end_char=span.end,
                    doc_chars=limit,
                )
            if previous is not None:
                if span.start <= previous.start or span.end <= previous.end:
                    raise ValidationFailed(
                        "chunk spans are not strictly increasing",
                        splitter=self.name,
                        document_id=document.id,
                        previous=(previous.start, previous.end),
                        current=(span.start, span.end),
                    )
                overlap = previous.end - span.start
                if allowed != UNBOUNDED_OVERLAP and overlap > allowed:
                    raise ValidationFailed(
                        "chunk overlap exceeds the configured overlap",
                        splitter=self.name,
                        document_id=document.id,
                        overlap_chars=overlap,
                        allowed_chars=allowed,
                    )
            previous = span

    # -- plumbing -----------------------------------------------------------
    async def _offload(self, size: int, fn: Callable[..., R], /, *args: Any) -> R:
        """Run CPU-bound planning off the event loop for large documents only.

        Regex scanning and BPE encoding over a megabyte of text are tens of
        milliseconds of pure CPU — long enough to stall every in-flight store
        request if left on the loop, and short enough that a thread hop is pure
        overhead on a short document.
        """
        if size >= _OFFLOAD_MIN_CHARS:
            return await asyncio.to_thread(fn, *args)
        return fn(*args)
