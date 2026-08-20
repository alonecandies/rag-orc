"""Semantic splitting — the default strategy.

The idea
--------
Fixed-size chunking cuts where the character counter runs out, which is
uncorrelated with where the meaning changes. Semantic chunking cuts where the
*topic* changes: embed a sliding window over the sentences, measure the cosine
distance between consecutive windows, and treat the peaks as boundaries. A chunk
then contains one idea, which is exactly what a dense retriever scores well.

Why windows instead of single sentences
--------------------------------------
A lone sentence is a noisy thing to embed — "It grew 40% year over year." carries
almost no topical signal, so its distance to its neighbours is dominated by
syntax rather than subject. Embedding sentence *i* together with
``semantic_buffer_size`` sentences on each side smooths that out: the signal
becomes "how much does the local context change here", which is the question we
actually want answered. The windows are built by slicing the original text
between two sentence offsets, so no string is rebuilt and offsets stay exact.

Why one embedding batch
-----------------------
All N windows go to the embedder in a **single** ``embed_documents`` call. This
matters more than it looks: hosted providers bill per input but charge latency per
*request*, and local ONNX inference is throughput-bound on batch size. One
request for a 400-sentence document instead of 400 is the difference between a
document taking 200ms and taking a minute. It is also why the splitter takes a
``DenseEmbedder`` and not an LLM — this is the only strategy in the package that
needs a model, and it needs the cheapest kind.

Note that these window vectors are *not* the chunk vectors. They are a boundary
detector that is thrown away; the chunks come out of here with no vectors at all,
so late chunking can still embed the document once and pool per span (ADR-0002).

Breakpoint selection
--------------------
All four methods from ``indexing.semantic_breakpoint`` are implemented because
they fail on different corpora, and the right one is a property of the text:

* ``percentile`` — threshold at the Nth percentile of the distances. Scale-free
  and predictable: it produces a boundary rate, so ``semantic_threshold=95``
  means "cut at the top 5% of transitions" regardless of how similar the corpus
  is overall. The safe default.
* ``stddev`` — ``mean + k * std``. Assumes a roughly normal distance
  distribution; sharper than percentile when there genuinely are a few strong
  topic shifts and a flat background.
* ``interquartile`` — ``mean + k * (q3 - q1)``. The robust cousin: the IQR is not
  moved by a handful of extreme distances, so one wild outlier (a boilerplate
  footer, a table dumped into the prose) does not raise the bar for the whole
  document the way ``std`` does.
* ``gradient`` — percentile over ``np.gradient`` of the distance curve. This is
  the one that saves dense technical prose, where *every* consecutive pair is
  far apart and an absolute threshold either cuts everywhere or nowhere. The
  derivative asks a different question — where is the distance *changing
  fastest* — and that still has structure when the level does not.

All of the distance mathematics is vectorized: one ``einsum`` over the normalized
matrix gives every consecutive-pair similarity at once. A Python loop over N
sentence pairs would cost more than the embedding call it follows.

Fallback
--------
A document with fewer than ``semantic_min_sentences`` sentences has nothing to
cluster — there are zero or one distances to threshold — so it degrades to the
recursive splitter and logs that it did. Silently returning one chunk per
document would be a worse failure, because it looks like success.
"""

from __future__ import annotations

import numpy as np
import structlog

from ragorc.core.models import Document, FloatArray
from ragorc.core.protocols import DenseEmbedder
from ragorc.core.registry import register
from ragorc.core.settings import Settings
from ragorc.embed.base import l2_normalize
from ragorc.index.split.base import BaseSplitter, Span, split_sentences
from ragorc.index.split.recursive import RecursiveSplitter

log = structlog.get_logger(__name__)

__all__ = ["SemanticSplitter"]


@register("splitter", "semantic")
class SemanticSplitter(BaseSplitter):
    """Distance-peak boundary detection over sentence windows."""

    name = "semantic"
    requires_embedder = True

    def __init__(self, embedder: DenseEmbedder, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.embedder = embedder
        # Constructed eagerly: it is a settings wrapper with no I/O, and building
        # it lazily inside the fallback path would put the allocation on the
        # first document that needs it, under a thread offload.
        self._fallback = RecursiveSplitter(settings=self.settings)

    async def _spans(self, document: Document) -> list[Span]:
        text = document.content
        sentences = await self._offload(len(text), split_sentences, text)
        minimum = max(self.config.semantic_min_sentences, 2)
        if len(sentences) < minimum:
            log.info(
                "semantic_split_fallback",
                document_id=document.id,
                sentences=len(sentences),
                min_sentences=minimum,
                fallback=self._fallback.name,
                reason="too few sentences to cluster",
            )
            return await self._fallback._spans(document)

        windows = _windows(text, sentences, self.config.semantic_buffer_size)
        vectors = await self.embedder.embed_documents(windows)
        if len(vectors) != len(sentences):
            # A provider that silently drops inputs would misalign every
            # boundary. Degrade rather than emit spans built from a shifted
            # distance curve.
            log.warning(
                "semantic_split_embedding_mismatch",
                document_id=document.id,
                sentences=len(sentences),
                vectors=len(vectors),
                fallback=self._fallback.name,
            )
            return await self._fallback._spans(document)

        return await self._offload(len(text), self._group_sentences, sentences, vectors)

    # -- boundary detection -------------------------------------------------
    def _group_sentences(
        self,
        sentences: list[tuple[int, int]],
        vectors: list[FloatArray],
    ) -> list[Span]:
        matrix = l2_normalize(np.asarray(np.stack(vectors), dtype=np.float32))
        # Cosine distance between every consecutive window pair in one pass.
        # einsum over the two shifted views computes N-1 dot products without
        # materializing an (N, N) similarity matrix we would only read a diagonal
        # of.
        distances = 1.0 - np.einsum("ij,ij->i", matrix[:-1], matrix[1:])
        metric, threshold = self._threshold(distances)
        # distances[i] sits between sentence i and i+1, so index i means "cut
        # after sentence i".
        #
        # Strictly greater, deliberately. With ``>=``, a document whose distances
        # are all equal — two sentences, or boilerplate repeated verbatim — breaks
        # at *every* position, which is the worst possible answer. Producing no
        # breakpoint at all is the safe direction: the group is then the whole
        # document and ``_emit`` still cuts it at its weakest seams to satisfy
        # ``max_chunk_size``, so the output is size-bounded either way.
        cuts = np.flatnonzero(metric > threshold)

        spans: list[Span] = []
        group_start = 0
        for cut in (*cuts.tolist(), len(sentences) - 1):
            if cut < group_start:
                continue
            self._emit(sentences, distances, group_start, cut, spans)
            group_start = cut + 1

        log.debug(
            "semantic_split_boundaries",
            sentences=len(sentences),
            breakpoints=int(cuts.size),
            method=self.config.semantic_breakpoint,
            threshold=round(float(threshold), 6),
            spans=len(spans),
        )
        return spans

    def _threshold(self, distances: FloatArray) -> tuple[FloatArray, float]:
        """Return ``(metric, threshold)`` for the configured breakpoint method."""
        method = self.config.semantic_breakpoint
        setting = float(self.config.semantic_threshold)
        if method == "gradient":
            # np.gradient needs at least two samples for a central difference.
            metric = np.gradient(distances) if distances.size > 1 else distances
            return metric, float(np.percentile(metric, float(np.clip(setting, 0.0, 100.0))))
        if method == "stddev":
            return distances, float(distances.mean() + setting * distances.std())
        if method == "interquartile":
            q1, q3 = np.percentile(distances, (25.0, 75.0))
            return distances, float(distances.mean() + setting * float(q3 - q1))
        return distances, float(np.percentile(distances, float(np.clip(setting, 0.0, 100.0))))

    def _emit(
        self,
        sentences: list[tuple[int, int]],
        distances: FloatArray,
        first: int,
        last: int,
        out: list[Span],
    ) -> None:
        """Turn sentences ``[first, last]`` into spans, honouring ``max_chars``.

        An oversized group is cut at its weakest interior seam — the largest
        remaining distance — rather than at an arbitrary character offset. The
        distance curve is already computed, so using it costs one ``argmax`` and
        keeps the guarantee that every boundary in the output is a semantic one.
        The work list is explicit rather than recursive: a single 200KB paragraph
        would otherwise recurse deep enough to matter.
        """
        ceiling = self.max_chars
        pending: list[tuple[int, int]] = [(first, last)]
        produced: list[tuple[int, int]] = []
        while pending:
            lo, hi = pending.pop()
            start, end = sentences[lo][0], sentences[hi][1]
            if hi <= lo or ceiling <= 0 or end - start <= ceiling:
                produced.append((lo, hi))
                continue
            seam = lo + int(np.argmax(distances[lo:hi]))
            pending.append((seam + 1, hi))
            pending.append((lo, seam))
        produced.sort()
        out.extend(
            Span(
                sentences[lo][0],
                sentences[hi][1],
                metadata={"sentences": hi - lo + 1, "breakpoint": self.config.semantic_breakpoint},
            )
            for lo, hi in produced
        )


def _windows(text: str, sentences: list[tuple[int, int]], buffer: int) -> list[str]:
    """Buffered window text for each sentence, sliced from the original.

    Sentence spans are contiguous, so a window is one slice between two offsets —
    no joining, no copies beyond the slices themselves.
    """
    count = len(sentences)
    if buffer <= 0:
        return [text[start:end] for start, end in sentences]
    return [
        text[sentences[max(0, index - buffer)][0] : sentences[min(count - 1, index + buffer)][1]]
        for index in range(count)
    ]
