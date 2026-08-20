"""Sentence-window splitting: search a precise unit, generate from a wide one.

The pattern
-----------
The retrieval unit and the generation unit want opposite things. Retrieval wants
the smallest text that still contains the answer, because a short chunk's
embedding is dominated by its one idea and therefore scores sharply against a
question about that idea. Generation wants the *surrounding* text, because a lone
sentence rarely carries the antecedents, units and qualifiers the model needs to
answer without inventing them.

Fixed-size chunking has to pick one and lose the other. This strategy refuses the
trade: index one chunk **per sentence**, and store the ``sentence_window_size``
sentences on either side in ``metadata["window_text"]``. Retrieval scores the
sentence; at generation time the pipeline substitutes the window, so the model
reads a paragraph while the index searched a sentence.

Consequences that are deliberate
--------------------------------
* **No minimum size.** ``min_chunk_size`` is disabled here, because merging runts
  would recreate the multi-sentence chunk this strategy exists to avoid. A short
  sentence is a legitimate chunk — it is also the case the window exists for.
* **No character overlap.** ``chunk_overlap`` is disabled too: the window *is* the
  overlap, and it is stored once per chunk instead of duplicated into the indexed
  text where it would blur the very vector we are trying to keep sharp.
* **Storage cost is real.** Each chunk's payload carries roughly
  ``2 * window_size + 1`` sentences of text, so the payload is several times the
  indexed content. That is the price of the pattern; ``window_start`` and
  ``window_end`` are stored alongside so a caller that keeps its documents can
  re-slice from the source instead and pay nothing.
* **A pathological "sentence"** — a minified JavaScript line, a base64 blob with
  no terminator — is still bounded by ``max_chunk_size`` through the base class,
  because one chunk of unbounded size would blow the embedder's context.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragorc.core.models import Document
from ragorc.core.registry import register
from ragorc.index.split.base import BaseSplitter, Span, split_sentences

log = structlog.get_logger(__name__)

__all__ = ["SentenceWindowSplitter"]


@register("splitter", "sentence_window")
class SentenceWindowSplitter(BaseSplitter):
    """One chunk per sentence, with its neighbourhood stored for generation."""

    name = "sentence_window"

    @property
    def min_chars(self) -> int:
        return 0  # a one-sentence chunk is the point, not a defect

    @property
    def overlap_chars(self) -> int:
        return 0  # metadata["window_text"] carries the context instead

    @property
    def window_size(self) -> int:
        return max(self.settings.retrieval.sentence_window_size, 0)

    def _spans_sync(self, document: Document) -> list[Span]:
        text = document.content
        sentences = split_sentences(text)
        if not sentences:
            return []
        window = self.window_size
        return [self._span_for(text, sentences, index, window) for index in range(len(sentences))]

    def _span_for(
        self,
        text: str,
        sentences: list[tuple[int, int]],
        index: int,
        window: int,
    ) -> Span:
        count = len(sentences)
        first = max(0, index - window)
        last = min(count - 1, index + window)
        # Sentence spans are contiguous, so the window is a single slice of the
        # original text — no joining, and the stored offsets stay meaningful.
        window_start, window_end = sentences[first][0], sentences[last][1]
        start, end = sentences[index]
        metadata: dict[str, Any] = {
            "window_text": text[window_start:window_end],
            "window_start": window_start,
            "window_end": window_end,
            "window_size": window,
            "sentence_index": index,
            "sentence_count": count,
        }
        return Span(start, end, metadata=metadata)
