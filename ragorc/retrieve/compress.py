"""Contextual compression: raising the evidence density of the context.

Retrieval returns whole chunks, but a 512-token chunk usually earns its place on
one or two sentences. The rest is not free:

* it is paid for, in tokens, on every call that carries it;
* it is a *distractor*. Irrelevant text in context measurably lowers answer
  accuracy — the model attends to it, and a plausible-but-unrelated sentence is
  exactly the raw material for a confident wrong answer;
* it occupies budget. Filler in chunk 3 is what evicts chunk 9, which held the
  other half of the answer.

So compression is not a cost optimization that happens to help quality; it is a
quality step that happens to be cheaper. It runs *after* reranking, on the
passages that survived, because compressing something that is about to be
discarded is pure waste.

Three implementations on a deliberate cost ladder:

============================  ===========  =========================================
compressor                    LLM calls    what it costs
============================  ===========  =========================================
``embedding_filter``          0            one matmul over vectors already in hand
``sentence``                  0            one embedding batch over all sentences
``extract``                   1 per chunk  the most precise, and the most expensive
============================  ===========  =========================================

``embedding_filter`` is the default for the reason its row shows: it is the only
one whose cost is indistinguishable from zero, it cannot invent text, and it
attacks the dominant failure of a recall-tuned hybrid retriever — the handful of
candidates that fusion dragged in because two retrievers each ranked them
mid-pack. It does not compress *within* a chunk, which is what the other two buy.

Two rules every compressor here obeys.

**Never return nothing.** Compression is lossy and irreversible; the chunks are
gone by the time the generator runs. A filter that empties the context turns a
mediocre answer into an abstention, so the best candidate always survives — the
same reasoning that admits the top chunk unconditionally in
:mod:`ragorc.context.pack`.

**Never paraphrase.** Only the LLM extractor *can* paraphrase, and a paraphrase
is not a smaller version of the evidence — it is new text that no document
contains. Downstream, :mod:`ragorc.validate.output` verifies each cited span
against the chunk it is attributed to, so a paraphrased excerpt converts a
correct, well-sourced answer into one whose citations cannot be verified. That is
why the prompt demands verbatim text and why this module checks rather than
trusts, falling back to the original chunk when the check fails.

Every stage reports the reduction it achieved, per chunk and in aggregate,
because compression that saves 5% is not worth its latency and there is no way to
know which case you are in without the numbers.
"""

from __future__ import annotations

import importlib
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError
from ragorc.core.models import FloatArray, Query, ScoredChunk, Usage
from ragorc.core.protocols import LLM, DenseEmbedder
from ragorc.core.registry import available, register
from ragorc.core.schemas import CompressedExcerpt
from ragorc.core.settings import Settings, get_settings
from ragorc.core.tokens import count_tokens, count_tokens_batch
from ragorc.index.split.base import split_sentences
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.security.injection import render_untrusted_passages

log = structlog.get_logger(__name__)

__all__ = [
    "BaseCompressor",
    "CompressionReport",
    "EmbeddingFilterCompressor",
    "LLMExtractCompressor",
    "PipelineCompressor",
    "SentenceLevelCompressor",
    "build_compressor",
    "verbatim_excerpt",
]

_DENSE_PROVIDERS = {
    "fastembed": "ragorc.embed.fastembed_provider",
    "openai": "ragorc.embed.openai_provider",
    "voyage": "ragorc.embed.voyage_provider",
    "cohere": "ragorc.embed.cohere_provider",
    "sentence_transformers": "ragorc.embed.sentence_transformers_provider",
}

#: Typographic characters folded to their ASCII equivalent before comparison.
#: Written as escapes rather than literals because the literal forms are exactly
#: the confusable characters the linter flags — a model that "quotes verbatim"
#: while smartening a quote or lengthening a dash has not paraphrased anything,
#: and must not be treated as if it had.
_QUOTE_FOLD = str.maketrans(
    {
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
    }
)
_WHITESPACE = re.compile(r"\s+")


def _default_dense_embedder(settings: Settings) -> DenseEmbedder:
    """Instantiate the configured dense embedder through the registry.

    The provider module is imported here rather than at module scope: importing
    all five would drag every optional SDK (openai, voyageai, cohere, torch) into
    a process that only needs the default ONNX one. The import is what runs the
    provider's ``@register``, so the resolve immediately after it cannot fail for
    a supported provider.
    """
    provider = settings.embedding.provider
    module = _DENSE_PROVIDERS.get(provider)
    if module is None:  # pragma: no cover - Settings constrains the literal
        raise ConfigError(f"no dense embedder for provider {provider!r}")
    importlib.import_module(module)
    from ragorc.core.registry import resolve

    return resolve("dense_embedder", provider)(settings=settings)


def _normalize(text: str) -> str:
    """Fold everything that is not meaning: unicode form, case, whitespace runs,
    smart quotes, dash styles.

    This must stay equivalent to the folding
    :mod:`ragorc.validate.output` applies before verifying a cited quote. If the
    two drift apart, an excerpt accepted as verbatim here can still be rejected
    as an unverifiable citation there — the exact failure the verbatim check
    exists to prevent.
    """
    folded = unicodedata.normalize("NFKC", text).lower().translate(_QUOTE_FOLD)
    return _WHITESPACE.sub(" ", folded).strip()


def verbatim_excerpt(excerpt: str, source: str) -> str | None:
    """Return the part of ``excerpt`` that genuinely occurs in ``source``.

    Whole-excerpt containment is only the fast path. The prompt asks for *the
    relevant sentences*, which are frequently non-adjacent in the document, so a
    faithful multi-sentence extract is legitimately not a contiguous substring.
    Verification therefore falls back to per-sentence containment, which accepts
    the honest case and still rejects a rewritten sentence — a paraphrase changes
    words, and no changed sentence appears in the source.

    ``None`` means nothing in the excerpt could be found, i.e. the model
    paraphrased or invented, and the caller must fall back to the original text.
    """
    haystack = _normalize(source)
    needle = _normalize(excerpt)
    if not needle or not haystack:
        return None
    if needle in haystack:
        return excerpt.strip()
    kept = [
        excerpt[start:end].strip()
        for start, end in split_sentences(excerpt)
        if _normalize(excerpt[start:end]) and _normalize(excerpt[start:end]) in haystack
    ]
    if not kept:
        return None
    return " ".join(kept)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CompressionReport:
    """What one compression stage actually achieved."""

    compressor: str
    chunks_in: int = 0
    chunks_out: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def ratio(self) -> float:
        """Surviving fraction of the tokens. 1.0 means the stage did nothing."""
        return self.tokens_out / self.tokens_in if self.tokens_in else 1.0

    @property
    def tokens_saved(self) -> int:
        return max(self.tokens_in - self.tokens_out, 0)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form: this lands in ``explain``, which is returned by the
        API and written to traces."""
        return {
            "compressor": self.compressor,
            "chunks_in": self.chunks_in,
            "chunks_out": self.chunks_out,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_saved": self.tokens_saved,
            "ratio": round(self.ratio, 4),
        }


def _token_counts(chunks: Sequence[ScoredChunk]) -> list[int]:
    """Token cost of each chunk, in one batched ``tiktoken`` call.

    ``encode_batch`` releases the GIL and parallelizes in Rust, so counting N
    chunks at once is several times faster than N calls for the identical
    result. The counts are cached back onto the chunks, which is where the
    packer looks for them next.
    """
    missing = [i for i, c in enumerate(chunks) if c.chunk.token_count is None]
    if missing:
        counted = count_tokens_batch([chunks[i].chunk.content for i in missing])
        for index, value in zip(missing, counted, strict=True):
            chunks[index].chunk.token_count = value
    return [c.chunk.token_count or 0 for c in chunks]


def _carry(scored: ScoredChunk) -> ScoredChunk:
    """A copy of ``scored`` with its text intact.

    Survivors are always copies, even the ones nothing happened to.
    :meth:`BaseCompressor._finish` writes ``rank`` and ``explain`` on whatever it
    is handed, and a compressor that stamped those onto the caller's objects
    would be editing the retrieval result it was asked to read.
    """
    return scored.with_score(scored.score)


def _rewrite(scored: ScoredChunk, content: str) -> ScoredChunk:
    """A copy of ``scored`` whose chunk carries ``content``.

    The :class:`~ragorc.core.models.Chunk` is *replaced*, not mutated:
    ``ScoredChunk.with_score`` shares the underlying chunk object, so editing
    ``content`` in place would rewrite the caller's data — and the retrieval
    result, the cache and anything else holding that chunk.

    The id is deliberately preserved even though it no longer content-hashes
    (``ids.chunk_id`` mixes the text in). Citations, the retrieval trace and the
    parent-expansion lookup all reference chunks by id, and the untruncated text
    is still in the store under that id.
    """
    clone = scored.with_score(scored.score)
    clone.chunk = replace(scored.chunk, content=content, token_count=count_tokens(content))
    return clone


class BaseCompressor:
    """Shared bookkeeping: token accounting, ranks, reporting, logging.

    Subclasses implement :meth:`compress` and call :meth:`_finish` on the
    survivors, so no implementation can forget to renumber ranks or to say what
    it saved.
    """

    name = "compressor"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def compress(
        self, query: Query, chunks: Sequence[ScoredChunk], **kwargs: Any
    ) -> tuple[list[ScoredChunk], Usage]:
        raise NotImplementedError(f"{type(self).__name__} must implement compress()")

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _question(query: Query | str) -> str:
        return query.text if isinstance(query, Query) else str(query)

    def _finish(
        self, kept: list[ScoredChunk], *, chunks_in: int, tokens_in: int
    ) -> list[ScoredChunk]:
        report = CompressionReport(
            compressor=self.name,
            chunks_in=chunks_in,
            chunks_out=len(kept),
            tokens_in=tokens_in,
            tokens_out=sum(_token_counts(kept)),
        )
        payload = report.to_dict()
        for rank, scored in enumerate(kept):
            scored.rank = rank
            # A list, so a PipelineCompressor's stages accumulate a history
            # instead of the last one erasing the evidence of the first.
            history = scored.explain.get("compression")
            scored.explain["compression"] = [*history, payload] if history else [payload]
        log.info(
            "post_retrieval_compressed",
            compressor=self.name,
            chunks_in=report.chunks_in,
            chunks_out=report.chunks_out,
            tokens_in=report.tokens_in,
            tokens_out=report.tokens_out,
            ratio=round(report.ratio, 3),
        )
        return kept


# ---------------------------------------------------------------------------
# Embedding filter — the default
# ---------------------------------------------------------------------------
@register("compressor", "embedding_filter", "embedding")
class EmbeddingFilterCompressor(BaseCompressor):
    """Drop whole chunks whose embedding is too far from the query.

    Zero LLM calls and one matmul: the query vector against an ``(n, dim)``
    matrix of chunk vectors, most of which the vector store already returned. On
    a 50-candidate set that is microseconds, which is what makes this the default
    — the other compressors have to justify their latency, this one does not.

    Two selection modes. With an explicit ``threshold`` it is an absolute cosine
    floor. Without one it keeps the top ``retrieval.compression_ratio`` fraction,
    which is the mode to prefer for the reason spelled out in
    :mod:`ragorc.retrieve.noise`: an absolute similarity floor is wrong per corpus
    and per embedding model, and it fails in both directions — on an easy query
    everything clears it, on a hard one nothing does.

    Selection is by similarity; *order* is the incoming rank order. The score is
    left alone, because this stage is a filter and not a second reranker: the
    similarity it computes is a weaker signal than the reranked score it would
    otherwise overwrite. It is recorded in ``component_scores`` instead.
    """

    name = "embedding_filter"

    def __init__(
        self,
        embedder: DenseEmbedder | None = None,
        *,
        threshold: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self._embedder = embedder
        self.threshold = threshold

    @property
    def embedder(self) -> DenseEmbedder:
        """Resolved on first use so constructing the compressor never loads a
        model — the pipeline builds every configured stage, including the ones a
        given request will not reach."""
        if self._embedder is None:
            self._embedder = _default_dense_embedder(self.settings)
        return self._embedder

    async def compress(
        self, query: Query, chunks: Sequence[ScoredChunk], **kwargs: Any
    ) -> tuple[list[ScoredChunk], Usage]:
        items = list(chunks)
        if len(items) < 2:
            # Nothing to filter against, and the "never return nothing" rule
            # would keep this chunk anyway.
            return items, Usage()
        tokens_in = sum(_token_counts(items))
        try:
            similarity = await self._similarity(query, items)
        except Exception as exc:  # noqa: BLE001 - degrade: an unfiltered context still answers
            log.warning(
                "embedding_filter_skipped",
                candidates=len(items),
                error=str(exc)[:200],
                error_type=type(exc).__name__,
            )
            return items, Usage()

        keep = self._select(similarity)
        kept: list[ScoredChunk] = []
        for index in keep.tolist():
            scored = items[index]
            clone = _carry(scored)
            clone.component_scores = {
                **scored.component_scores,
                "query_similarity": float(similarity[index]),
            }
            kept.append(clone)
        return self._finish(kept, chunks_in=len(items), tokens_in=tokens_in), Usage()

    async def _similarity(self, query: Query | str, items: Sequence[ScoredChunk]) -> FloatArray:
        """Cosine similarity of every chunk to the query, in one matmul.

        Vectors that came back from the store are reused; only the gaps are
        embedded. If anything has to be embedded here, the query is embedded with
        *this* embedder too: mixing a query vector produced by one model with
        document vectors produced by another yields a number that is numerically
        valid and semantically meaningless, and nothing downstream can detect it.
        """
        question = self._question(query)
        vectors: list[FloatArray | None] = [c.chunk.dense for c in items]
        missing = [i for i, vector in enumerate(vectors) if vector is None]
        if missing:
            fresh = await self.embedder.embed_documents([items[i].chunk.content for i in missing])
            for index, vector in zip(missing, fresh, strict=True):
                vectors[index] = vector
            query_vector = await self.embedder.embed_query(question)
        elif isinstance(query, Query) and query.dense is not None:
            query_vector = query.dense  # already computed by the vector retriever
        else:
            query_vector = await self.embedder.embed_query(question)

        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        # Unit vectors, so the dot product *is* cosine similarity: in [-1, 1],
        # higher is better, no distance to invert.
        return matrix @ q

    def _select(self, similarity: FloatArray) -> np.ndarray:
        """Indices to keep, ascending so the incoming rank order survives."""
        total = int(similarity.shape[0])
        if self.threshold is not None:
            keep = np.flatnonzero(similarity >= self.threshold)
        else:
            ratio = min(max(self.settings.retrieval.compression_ratio, 0.0), 1.0)
            wanted = max(math.ceil(total * ratio), 1)
            if wanted >= total:
                return np.arange(total)
            keep = np.argpartition(-similarity, wanted - 1)[:wanted]
        if keep.size == 0:
            # Never return nothing: the single best candidate always survives.
            # ``intp`` is the index dtype the two branches above already produce.
            keep = np.asarray([int(np.argmax(similarity))], dtype=np.intp)
        return np.sort(keep)


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------
@register("compressor", "extract", "llm_extract")
class LLMExtractCompressor(BaseCompressor):
    """Ask the model for the relevant spans of each chunk, verbatim.

    The most precise compressor available — it is the only one that understands
    the question rather than measuring proximity to it — and the most expensive:
    one structured call per chunk, fanned out under
    :func:`~ragorc.core.concurrency.bounded_gather` so the latency is roughly one
    call rather than N, and charged to the cheap tier via
    :attr:`~ragorc.llm.router.Task.COMPRESS`.

    Three outcomes per chunk, and the third is the one that matters:

    * an empty excerpt means the model found nothing relevant, and the chunk is
      dropped. That is the point of the stage: the reranker had to return
      ``rerank_top_k`` candidates whether or not that many were any good;
    * a verbatim excerpt replaces the chunk's text;
    * a *paraphrased* excerpt is discarded and the original chunk kept, with a
      warning. Accepting it would produce a context whose sentences appear in no
      document, so every citation drawn from it would fail verification in
      :mod:`ragorc.validate.output`. Keeping the original text costs tokens;
      keeping a paraphrase costs the answer's provenance.
    """

    name = "extract"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        super().__init__(settings)
        self.llm = llm
        self.router = router or ModelRouter(self.settings.llm)

    async def compress(
        self, query: Query, chunks: Sequence[ScoredChunk], **kwargs: Any
    ) -> tuple[list[ScoredChunk], Usage]:
        items = list(chunks)
        if not items:
            return [], Usage()
        tokens_in = sum(_token_counts(items))
        question = self._question(query)
        prompt = get_prompt("compress_extract")
        model = self.router.model_for(Task.COMPRESS)

        results = await bounded_gather(
            (self._extract(prompt, model, question, scored) for scored in items),
            limit=max(self.settings.retrieval.max_concurrent_retrievers, 1),
            return_exceptions=True,
        )

        kept: list[ScoredChunk] = []
        usages: list[Usage] = []
        dropped = 0
        for scored, result in zip(items, results, strict=True):
            if isinstance(result, BaseException):
                # A failed extraction must not lose the passage: an uncompressed
                # chunk is worse than a compressed one and far better than none.
                log.warning(
                    "compress_extract_failed",
                    chunk_id=scored.chunk.id,
                    error=str(result)[:200],
                    error_type=type(result).__name__,
                )
                kept.append(_carry(scored))
                continue
            replacement, usage = result
            usages.append(usage)
            if replacement is None:
                dropped += 1
                continue
            kept.append(replacement)

        if not kept and items:
            # The model rejected everything. Trust the reranker's top hit over a
            # unanimous "nothing is relevant", which is also what a mis-specified
            # question looks like.
            log.warning("compress_extract_kept_nothing", candidates=len(items))
            kept = [_carry(items[0])]
        log.debug("compress_extract", candidates=len(items), dropped=dropped, kept=len(kept))
        return (
            self._finish(kept, chunks_in=len(items), tokens_in=tokens_in),
            Usage.sum(usages),
        )

    async def _extract(
        self, prompt: Any, model: str, question: str, scored: ScoredChunk
    ) -> tuple[ScoredChunk | None, Usage]:
        source = scored.chunk.content
        result, usage = await self.llm.structured(
            # Fenced. `verbatim_excerpt` below is not a substitute: it only checks
            # the excerpt is a substring of the source, which an instruction
            # embedded in that source trivially is — and the verified excerpt then
            # replaces the chunk body and is packed into the answer prompt as
            # retrieved evidence.
            prompt.render(question=question, document=render_untrusted_passages([source])),
            CompressedExcerpt,
            system=prompt.system,
            model=model,
            stage="compress_extract",
        )
        excerpt = (result.excerpt or "").strip()
        if not result.relevant or not excerpt:
            return None, usage

        verified = verbatim_excerpt(excerpt, source)
        if verified is None:
            log.warning(
                "compress_paraphrase_rejected",
                chunk_id=scored.chunk.id,
                excerpt_chars=len(excerpt),
                source_chars=len(source),
                model=model,
            )
            fallback = _carry(scored)
            fallback.explain["compress_paraphrased"] = True
            return fallback, usage

        clone = _rewrite(scored, verified)
        clone.explain["extracted"] = True
        return clone, usage


# ---------------------------------------------------------------------------
# Sentence-level filtering
# ---------------------------------------------------------------------------
@register("compressor", "sentence", "sentence_level")
class SentenceLevelCompressor(BaseCompressor):
    """Keep the sentences of each chunk that match the query, in original order.

    Finer-grained than the chunk-level filter and free of the extractor's LLM
    cost. Every sentence of every chunk is embedded in **one** batch: the
    alternative — a call per chunk — pays per-request overhead N times, and for a
    hosted provider that is N round trips for the same tokens and the same money.

    **Why original order is preserved.** Sentences are not independent evidence.
    Pronouns, "however", "the latter", "in that case" and every comparative
    resolve against the sentence before them, so a set of sentences reordered by
    relevance reads fluently and states things the document does not. It would
    also break span attribution: the citation verifier locates a quote inside the
    chunk, and a reordered chunk no longer contains the sequences it is quoted
    for. So relevance decides *which* sentences survive and position decides the
    order they are emitted in.
    """

    name = "sentence"

    def __init__(
        self,
        embedder: DenseEmbedder | None = None,
        *,
        keep_ratio: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self._embedder = embedder
        self.keep_ratio = keep_ratio

    @property
    def embedder(self) -> DenseEmbedder:
        if self._embedder is None:
            self._embedder = _default_dense_embedder(self.settings)
        return self._embedder

    async def compress(
        self, query: Query, chunks: Sequence[ScoredChunk], **kwargs: Any
    ) -> tuple[list[ScoredChunk], Usage]:
        items = list(chunks)
        if not items:
            return [], Usage()
        tokens_in = sum(_token_counts(items))

        owners: list[int] = []
        sentences: list[str] = []
        for index, scored in enumerate(items):
            content = scored.chunk.content
            for start, end in split_sentences(content):
                piece = content[start:end].strip()
                if piece:
                    owners.append(index)
                    sentences.append(piece)
        if len(sentences) <= len(items):
            # One sentence per chunk at best: nothing to select within a chunk.
            return items, Usage()

        try:
            scores = await self._score(query, sentences)
        except Exception as exc:  # noqa: BLE001 - degrade to the uncompressed chunks
            log.warning(
                "sentence_filter_skipped",
                sentences=len(sentences),
                error=str(exc)[:200],
                error_type=type(exc).__name__,
            )
            return items, Usage()

        ratio = self.keep_ratio
        if ratio is None:
            ratio = self.settings.retrieval.compression_ratio
        ratio = min(max(ratio, 0.0), 1.0)

        owner_array = np.asarray(owners, dtype=np.int64)
        kept: list[ScoredChunk] = []
        for index, scored in enumerate(items):
            positions = np.flatnonzero(owner_array == index)
            if positions.size == 0:
                kept.append(_carry(scored))
                continue
            selected = self._select(scores[positions], ratio)
            # np.sort restores document order after selection by score.
            text = " ".join(sentences[int(positions[i])] for i in np.sort(selected).tolist())
            clone = _rewrite(scored, text)
            clone.component_scores = {
                **scored.component_scores,
                "sentence_max_similarity": float(scores[positions].max()),
            }
            clone.explain["sentences_kept"] = int(selected.size)
            clone.explain["sentences_total"] = int(positions.size)
            kept.append(clone)
        return self._finish(kept, chunks_in=len(items), tokens_in=tokens_in), Usage()

    async def _score(self, query: Query | str, sentences: Sequence[str]) -> FloatArray:
        """One embedding call for every sentence of every chunk."""
        question = self._question(query)
        vectors = await self.embedder.embed_documents(list(sentences))
        query_vector = await self.embedder.embed_query(question)
        matrix = np.asarray(vectors, dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        return matrix @ q  # cosine over unit vectors: higher is better

    @staticmethod
    def _select(scores: FloatArray, ratio: float) -> np.ndarray:
        total = int(scores.shape[0])
        wanted = max(math.ceil(total * ratio), 1)
        if wanted >= total:
            return np.arange(total)
        return np.argpartition(-scores, wanted - 1)[:wanted]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@register("compressor", "pipeline", "both", "none")
class PipelineCompressor(BaseCompressor):
    """Chain compressors, cheapest first.

    Order is the whole design. The embedding filter decides how many chunks reach
    the extractor, and the extractor's cost *is* the pipeline's cost — one LLM
    call per surviving chunk. Filtering first therefore halves the bill of the
    stage behind it at no quality cost, because a chunk the filter drops is one
    the extractor would have found nothing in.

    An empty chain is a working no-op, which is what ``compressor: none``
    resolves to: the pipeline always has a compressor object and never has to
    branch on ``None``.
    """

    name = "pipeline"

    def __init__(
        self,
        stages: Sequence[BaseCompressor] = (),
        *,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self.stages = list(stages)

    async def compress(
        self, query: Query, chunks: Sequence[ScoredChunk], **kwargs: Any
    ) -> tuple[list[ScoredChunk], Usage]:
        current = list(chunks)
        if not self.stages or not current:
            return current, Usage()
        tokens_in = sum(_token_counts(current))
        usages: list[Usage] = []
        for stage in self.stages:
            current, usage = await stage.compress(query, current, **kwargs)
            usages.append(usage)
            if not current:
                break
        # Stamp the end-to-end reduction on top of the per-stage entries, so the
        # trace answers "what did compression cost me overall" without summing.
        return self._finish(current, chunks_in=len(chunks), tokens_in=tokens_in), Usage.sum(usages)

    @property
    def names(self) -> list[str]:
        return [stage.name for stage in self.stages]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_compressor(
    name: str | None = None,
    *,
    llm: LLM | None = None,
    embedder: DenseEmbedder | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseCompressor:
    """Resolve ``settings.retrieval.compressor`` (or an explicit name) to a stage.

    ``"none"`` returns an empty :class:`PipelineCompressor` rather than ``None``:
    a caller that has to test for ``None`` before every compression step will
    eventually forget to.
    """
    resolved = settings or get_settings()
    key = (name or resolved.retrieval.compressor or "none").strip().lower().replace("-", "_")

    if key in {"embedding_filter", "embedding", "filter"}:
        return EmbeddingFilterCompressor(embedder, settings=resolved, **kwargs)
    if key in {"sentence", "sentence_level", "sentences"}:
        return SentenceLevelCompressor(embedder, settings=resolved, **kwargs)
    if key in {"extract", "llm_extract"}:
        return LLMExtractCompressor(_require_llm(llm, key), resolved, **kwargs)
    if key in {"both", "pipeline"}:
        return PipelineCompressor(
            [
                EmbeddingFilterCompressor(embedder, settings=resolved),
                LLMExtractCompressor(_require_llm(llm, key), resolved),
            ],
            settings=resolved,
        )
    if key in {"none", "noop", ""}:
        return PipelineCompressor((), settings=resolved)

    raise ConfigError(
        f"unknown compressor {key!r}",
        available=sorted({"embedding_filter", "sentence", "extract", "both", "none"}),
        registered=available("compressor")["compressor"],
    )


def _require_llm(llm: LLM | None, key: str) -> LLM:
    if llm is None:
        raise ConfigError(
            f"the {key!r} compressor makes LLM calls and needs llm=...",
            hint="pass the LLM instance, or select compressor='embedding_filter'",
        )
    return llm
