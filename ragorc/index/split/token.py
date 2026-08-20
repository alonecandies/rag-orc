"""Token-window splitting.

Why a token splitter at all
---------------------------
Every downstream budget is denominated in tokens: the embedder truncates at
``max_length`` tokens, the context packer fits chunks into a token window, and
the bill is per token. A character-sized chunk is a proxy for that, and the proxy
drifts badly by content type — 512 characters is ~110 tokens of English prose,
~180 tokens of Python, and ~500 tokens of CJK text. When you need the guarantee
"no chunk ever exceeds the embedder's context", this is the strategy that gives
it, because it is measured in the same unit as the limit.

So here, and only here, ``indexing.chunk_size`` and ``indexing.chunk_overlap``
are read as **tokens** rather than characters.

Mapping token boundaries back to characters
-------------------------------------------
The hard part is that the chunk contract is in character offsets (ADR-0002: the
late-chunking pooler indexes the document by them) while the cut points are in
token space. Decoding each window back to text and searching for it is both slow
and wrong — the same substring can appear twice, and BPE round-tripping is not
guaranteed to be byte-identical.

Instead this module builds the exact map once per document:

1. ``encode`` the document, then ``decode_tokens_bytes`` once for the whole id
   list — one FFI call, not one per token — giving each token's byte length.
2. ``cumsum`` those lengths: token *i* ends at byte ``offsets[i]``.
3. Build a byte->character index from the UTF-8 continuation-byte pattern
   (``byte & 0xC0 != 0x80`` marks a character start) and ``cumsum`` that too.

Both steps are single numpy passes over arrays, no Python loop over tokens.

**Multi-byte safety.** A BPE token boundary is a *byte* boundary and can fall
inside a multi-byte character (tiktoken falls back to byte-level tokens for rare
codepoints and emoji). The byte->character map counts character *starts*, so such
a boundary resolves to the index just after the containing character: the
character lands wholly in the left chunk, the right chunk begins at the next one.
No mojibake, no dropped or duplicated character, and the two chunks stay
contiguous because both use the same map.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import structlog

from ragorc.core.models import Document, IntArray
from ragorc.core.registry import register
from ragorc.core.settings import Settings
from ragorc.index.split.base import UNBOUNDED_OVERLAP, BaseSplitter, Span

log = structlog.get_logger(__name__)

__all__ = ["TokenSplitter", "token_spans"]

_DEFAULT_ENCODING = "o200k_base"
"""Matches :mod:`ragorc.core.tokens`: within a few percent of the tokenizers used
by the Llama/Mistral/Claude-family models served through OpenRouter, and the
counts reported by every other stage of the pipeline come from the same encoder,
so the numbers agree."""


@functools.lru_cache(maxsize=4)
def _encoder(encoding: str = _DEFAULT_ENCODING) -> Any:
    """Cached tiktoken encoder. Construction reads a merge table from disk, so
    doing it per document would dominate the split cost."""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - tiktoken is a base dependency
        raise ImportError("the token splitter needs tiktoken: pip install 'ragorc'") from exc
    return tiktoken.get_encoding(encoding)


def _byte_to_char_index(text: str) -> IntArray:
    """Map every UTF-8 byte offset in ``text`` to a character offset.

    Returns an array of ``len(utf8) + 1`` entries so a token's end offset can be
    looked up directly. ASCII text short-circuits to ``arange``, which is the
    common case and avoids materializing the mask.
    """
    data = text.encode("utf-8")
    if len(data) == len(text):
        return np.arange(len(text) + 1, dtype=np.int64)
    raw = np.frombuffer(data, dtype=np.uint8)
    # 0b10xxxxxx is a UTF-8 continuation byte; anything else starts a character.
    starts = (raw & 0xC0) != 0x80
    out = np.empty(len(data) + 1, dtype=np.int64)
    out[0] = 0
    out[1:] = np.cumsum(starts, dtype=np.int64)
    return out


def token_spans(
    text: str,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
    encoding: str = _DEFAULT_ENCODING,
) -> list[tuple[int, int]]:
    """Character spans for fixed-size token windows with a stride for overlap."""
    if not text:
        return []
    encoder = _encoder(encoding)
    ids = encoder.encode(text, disallowed_special=())
    total = len(ids)
    if total == 0:
        return []

    size = max(chunk_tokens, 1)
    stride = max(size - max(overlap_tokens, 0), 1)

    lengths = np.fromiter(
        (len(piece) for piece in encoder.decode_tokens_bytes(ids)),
        dtype=np.int64,
        count=total,
    )
    token_end_bytes = np.cumsum(lengths)
    byte_to_char = _byte_to_char_index(text)

    starts = np.arange(0, total, stride, dtype=np.int64)
    ends = np.minimum(starts + size, total)
    start_bytes = np.where(starts > 0, token_end_bytes[np.maximum(starts - 1, 0)], 0)
    end_bytes = token_end_bytes[ends - 1]
    char_starts = byte_to_char[start_bytes]
    char_ends = byte_to_char[end_bytes]

    # Drop windows that add nothing: once a window reaches the end of the
    # document the remaining strides only re-emit its tail. This is also what
    # makes an undersized final chunk impossible — a surviving tail window is at
    # least ``size - overlap`` tokens long — so no minimum-size merge is needed.
    keep = (np.diff(char_starts, prepend=-1) > 0) & (np.diff(char_ends, prepend=-1) > 0)
    return [
        (int(span_start), int(span_end))
        for span_start, span_end in zip(char_starts[keep], char_ends[keep], strict=True)
    ]


@register("splitter", "token")
class TokenSplitter(BaseSplitter):
    """Fixed token windows with a token stride, mapped to character offsets."""

    name = "token"

    def __init__(
        self,
        *,
        encoding: str = _DEFAULT_ENCODING,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self.encoding = encoding

    # The token ceiling *is* the ceiling. Applying the character limits on top
    # would cut inside a token window, which defeats the only guarantee this
    # strategy exists to provide.
    @property
    def min_chars(self) -> int:
        return 0

    @property
    def max_chars(self) -> int:
        return 0

    @property
    def overlap_chars(self) -> int:
        return 0  # the stride already overlaps; a second pass would double it

    @property
    def allowed_overlap_chars(self) -> int:
        return UNBOUNDED_OVERLAP

    def _spans_sync(self, document: Document) -> list[Span]:
        spans = token_spans(
            document.content,
            chunk_tokens=self.config.chunk_size,
            overlap_tokens=self.config.chunk_overlap,
            encoding=self.encoding,
        )
        return [Span(start, end, metadata={"encoding": self.encoding}) for start, end in spans]
