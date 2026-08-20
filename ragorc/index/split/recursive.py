"""Recursive character splitting — the universal fallback.

Why this strategy exists when semantic chunking is the default
-------------------------------------------------------------
It needs no model, no network and no tokenizer, so it is the only strategy that
cannot fail for an environmental reason. Three call sites depend on that: the
semantic splitter falls back here when a document has too few sentences to
cluster, the markdown and code splitters use :func:`recursive_spans` to break up
an oversized prose block or definition body, and
:func:`ragorc.index.split.build_splitter` degrades to it when a semantic splitter
is requested without an embedder.

The algorithm, and the part everyone gets wrong
-----------------------------------------------
Separators are tried from coarsest to finest — paragraph, line, sentence, clause,
word, character. At each level the splitter picks the *largest* separator whose
pieces all fit inside ``chunk_size``; if none does, it takes the coarsest
separator that is actually present and recurses into the pieces that are still
too large. Pieces are then packed back up greedily to ``chunk_size``, because a
document split on ``". "`` without repacking yields one chunk per sentence, which
is a different (and much worse) strategy.

The part that gets implemented wrong is offsets. The obvious implementation is
``text.split(sep)`` then re-join, which loses the mapping back to the original
document: separators are consumed, whitespace is stripped, and the reconstructed
chunk no longer sits at a known position. Late chunking then pools the wrong
tokens (ADR-0002) and nothing raises. So this module never builds a string. It
only ever computes index pairs with ``str.find``, and the separator stays with
the piece it terminates so a sentence keeps its full stop.

Cost: each recursion level scans its segment once per candidate separator, so the
work is ``O(len(separators) * n)`` per level with depth bounded by the number of
separators — linear in practice, and it never allocates a copy of the document.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import structlog

from ragorc.core.models import Document
from ragorc.core.registry import register
from ragorc.core.settings import Settings
from ragorc.index.split.base import BaseSplitter, Span, merge_small_spans

log = structlog.get_logger(__name__)

__all__ = ["DEFAULT_SEPARATORS", "RecursiveSplitter", "recursive_spans"]

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", "")
"""Coarsest to finest. The trailing ``""`` is the last resort: a fixed-width
character cut for text with no separators at all (minified JSON, CJK prose
without spaces, a base64 blob)."""


def recursive_spans(
    text: str,
    start: int,
    end: int,
    *,
    chunk_size: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
    template: Span | None = None,
) -> list[Span]:
    """Split ``text[start:end]`` into packed spans of at most ``chunk_size``.

    ``template`` supplies the metadata, group and modality for every produced
    span, which is how the markdown and code splitters reuse this machinery
    without losing the structural context they discovered (heading path,
    enclosing class).
    """
    if end <= start:
        return []
    size = max(chunk_size, 1)
    cuts: list[tuple[int, int]] = []
    _cut(text, start, end, 0, size, tuple(separators), cuts)
    base = template or Span(start, end)
    spans = [
        replace(base, start=piece_start, end=piece_end, metadata=dict(base.metadata))
        for piece_start, piece_end in cuts
    ]
    # min == max == chunk_size turns the merge into greedy packing to target.
    return merge_small_spans(spans, min_size=size, max_size=size)


def _cut(
    text: str,
    start: int,
    end: int,
    depth: int,
    chunk_size: int,
    separators: Sequence[str],
    out: list[tuple[int, int]],
) -> None:
    if end - start <= chunk_size:
        out.append((start, end))
        return

    fallback: list[tuple[int, int]] | None = None
    fallback_depth = depth
    for index in range(depth, len(separators)):
        separator = separators[index]
        if not separator:
            break
        pieces = _pieces_on(text, start, end, separator)
        if len(pieces) < 2:
            continue
        if max(piece_end - piece_start for piece_start, piece_end in pieces) <= chunk_size:
            out.extend(pieces)
            return
        if fallback is None:
            fallback, fallback_depth = pieces, index

    if fallback is not None:
        for piece_start, piece_end in fallback:
            _cut(text, piece_start, piece_end, fallback_depth + 1, chunk_size, separators, out)
        return

    # No separator occurs in this range: cut on width. This is the only place a
    # word can be broken, and it is unavoidable for text that has no words.
    cursor = start
    while cursor < end:
        out.append((cursor, min(cursor + chunk_size, end)))
        cursor += chunk_size


def _pieces_on(text: str, start: int, end: int, separator: str) -> list[tuple[int, int]]:
    """Tile ``[start, end)`` on ``separator``, keeping it with the left piece.

    Tiling (rather than splitting) is what preserves exact offsets: the pieces are
    contiguous and cover the range completely, so any run of consecutive pieces is
    itself an exact span of the original document.
    """
    pieces: list[tuple[int, int]] = []
    cursor = start
    found = text.find(separator, cursor, end)
    while found != -1:
        cut = found + len(separator)
        pieces.append((cursor, cut))
        cursor = cut
        found = text.find(separator, cursor, end)
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


@register("splitter", "recursive")
class RecursiveSplitter(BaseSplitter):
    """Separator-driven character splitting with exact offsets."""

    name = "recursive"

    def __init__(
        self,
        *,
        separators: Sequence[str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self.separators: tuple[str, ...] = tuple(separators or DEFAULT_SEPARATORS)

    def _spans_sync(self, document: Document) -> list[Span]:
        text = document.content
        return recursive_spans(
            text,
            0,
            len(text),
            chunk_size=self.target_chars,
            separators=self.separators,
        )
