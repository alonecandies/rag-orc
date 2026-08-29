"""Reranking: buying precision on exactly the candidates that can still change
the answer.

First-stage retrieval is a *recall* device. It fetches ``fetch_k`` candidates per
retriever per query variant, fuses them by rank, and is deliberately generous
because a document the first stage misses can never be recovered downstream. The
price of that generosity is that the ordering of the top 50 is only roughly
right — and the generator sees ``top_k`` of them, so the ordering is the whole
game.

Why a cross-encoder beats bi-encoder similarity
-----------------------------------------------
A bi-encoder (everything that can be stored in a vector index) embeds the query
and the document *independently*. The document's vector is fixed at index time,
before any query exists, so it is a lossy summary of the passage compressed in
ignorance of what will be asked; the score is a dot product between two such
summaries. Term-level correspondence, negation, and which of three entities the
question is actually about are all things that vector has already thrown away.

A cross-encoder concatenates the query and the document into one sequence and
runs full self-attention over the pair. Every query token can attend to every
document token, so the model scores the *interaction* rather than comparing two
independent digests. That is why it is markedly more accurate — and, in the same
breath, why it cannot be precomputed or indexed: there is no query-independent
document representation to store. The model must run once per ``(query,
document)`` pair, and the cost is linear in candidates with nothing amortizable.

The operational consequence is the whole design of this module: **rerank the top
~50, never the corpus**. Fifty pairs on an ONNX CPU session is tens of
milliseconds; a million pairs is hours, per query. Recall is set by ``fetch_k``,
precision by ``rerank_top_k``, and the reranker is only ever handed what the
first stage already shortlisted.

ColBERT sits deliberately between the two. Late interaction keeps a vector *per
token*, precomputed at index time, and scores with MaxSim — so it keeps some of
the term-level resolution a pooled vector loses while remaining indexable. It
costs ~100x the storage of a single dense vector and, having no cross-attention,
lands below a cross-encoder on accuracy; it is the right choice when reranking
hundreds of candidates rather than tens, or when the scores must come from
vectors the store already holds.

Two invariants hold across every class here. Scores are higher-is-better (raw
cross-encoder logits and MaxSim similarities already are; nothing here is a
distance, so nothing is inverted). And provenance survives: the first-stage
score is preserved in ``component_scores`` next to the rerank score, because
"the cross-encoder disagreed with the retriever" is the single most useful thing
to see when a result looks wrong.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError
from ragorc.core.ids import cache_key, content_hash
from ragorc.core.models import FloatArray, IntArray, Query, ScoredChunk, Usage
from ragorc.core.protocols import LLM, Cache, LateInteractionEmbedder, Reranker
from ragorc.core.registry import available, register
from ragorc.core.settings import _COLBERT_RERANKER_NAMES, Settings, get_settings
from ragorc.embed.base import cached_batch, l2_normalize, l2_normalize_list
from ragorc.embed.fastembed_provider import FastEmbedLateInteraction, FastEmbedReranker

log = structlog.get_logger(__name__)

__all__ = [
    "BaseReranker",
    "ColBERTReranker",
    "CrossEncoderReranker",
    "IdentityReranker",
    "build_reranker",
    "maxsim",
]

_SCORE_FRAME = struct.Struct("<f")
"""Cache payload for one score: 4 bytes, exactly the precision the model
produced. The same float as a JSON number is ~20 bytes and costs a parse that
allocates a Python object; at 50 candidates per query that difference is the
whole point of having a cache."""


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------
def _top_pairs(scores: FloatArray, limit: int | None) -> list[tuple[int, float]]:
    """``(index, score)`` pairs against the input order, best first.

    ``argpartition`` is O(n) where ``argsort`` is O(n log n), and only the
    retained slice is then sorted — the same trade the embedding-layer reranker
    makes, for the same reason: at n=200, k=20 it is an order of magnitude less
    comparison work for an identical answer.
    """
    total = int(scores.shape[0])
    if total == 0:
        return []
    keep = total if limit is None else max(min(limit, total), 1)
    if keep < total:
        top = np.argpartition(-scores, keep - 1)[:keep]
        top = top[np.argsort(-scores[top], kind="stable")]
    else:
        top = np.argsort(-scores, kind="stable")
    return [(int(i), float(scores[i])) for i in top]


class BaseReranker:
    """Everything a rerank stage needs except the ranking itself.

    Subclasses implement one method, :meth:`_order`, which returns ``(index,
    score)`` pairs against the input order. The rest is inherited and therefore
    cannot be got wrong per-implementation:

    * :meth:`rerank` — the :class:`ragorc.core.protocols.Reranker` surface over
      plain strings, for benchmarks and for callers that hold no chunks;
    * :meth:`rerank_chunks` — the pipeline surface, which rebuilds
      :class:`ScoredChunk` objects with ``rank`` filled from 0 and
      ``component_scores`` populated;
    * :meth:`rerank_with_usage` — the same, plus the cost, because a listwise
      reranker spends money and the protocol has nowhere to report it;
    * degradation. A reranker is an *enhancement*: if the model, the ONNX
      session or the provider fails, the query must still be answered from the
      first-stage ordering rather than not at all.
    """

    name = "reranker"
    model_name = "reranker"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # -- the one method subclasses implement -------------------------------
    async def _order(
        self,
        question: str,
        texts: list[str],
        ids: list[str] | None,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float]], Usage]:
        """Rank ``texts`` against ``question``.

        ``ids`` carries stable per-document identity when the caller has it
        (chunk ids), so cache keys can be built without hashing the text again.
        Implementations must return every index they were given at most once and
        must never drop a document silently.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _order()")

    async def _order_chunks(
        self, question: str, chunks: Sequence[ScoredChunk], top_k: int | None
    ) -> tuple[list[tuple[int, float]], Usage]:
        """Seam for rerankers that can use a representation the chunk already
        carries (ColBERT multivectors) instead of its text. The default reads
        the text, which is what a cross-encoder needs anyway."""
        return await self._order(
            question,
            [c.chunk.content for c in chunks],
            [c.chunk.id for c in chunks],
            top_k,
        )

    # -- public API --------------------------------------------------------
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        """Protocol entry point. ``top_k=None`` means *score everything* here,
        deliberately unlike :meth:`rerank_chunks`: this is the model-level API
        and it should do what it is told, not what the config prefers."""
        docs = list(documents)
        if not docs:
            return []
        pairs, _ = await self._order(query, docs, None, top_k)
        return pairs

    async def rerank_chunks(
        self,
        query: Query | str,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """Pipeline entry point: reranked chunks, ranks renumbered from 0."""
        result, _ = await self.rerank_with_usage(query, chunks, top_k=top_k)
        return result

    async def rerank_with_usage(
        self,
        query: Query | str,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
    ) -> tuple[list[ScoredChunk], Usage]:
        """As :meth:`rerank_chunks`, plus what it cost.

        Local rerankers report an empty :class:`Usage`; the listwise LLM one
        reports its calls, so the caller aggregates cost identically either way
        and never has to know which reranker is configured.
        """
        items = list(chunks)
        if not items:
            return [], Usage()
        question = query.text if isinstance(query, Query) else str(query)
        limit = self._limit(top_k, len(items))
        try:
            pairs, usage = await self._order_chunks(question, items, limit)
        except Exception as exc:  # noqa: BLE001 - degrade: an unranked answer beats none
            log.warning(
                "rerank_failed",
                reranker=self.name,
                model=self.model_name,
                candidates=len(items),
                error=str(exc)[:200],
                error_type=type(exc).__name__,
            )
            return self._passthrough(items, limit), Usage()
        return self._rebuild(items, pairs), usage

    # -- internals ---------------------------------------------------------
    def _limit(self, top_k: int | None, available_count: int) -> int:
        """How many results the *pipeline* wants. Input width is ``fetch_k`` and
        belongs to the retriever; output width is ``rerank_top_k`` and belongs
        here, with the noise filter trimming to ``top_k`` afterwards."""
        requested = top_k if top_k is not None else self.settings.retrieval.rerank_top_k
        return max(min(requested, available_count), 1)

    def _rebuild(
        self, items: Sequence[ScoredChunk], pairs: Sequence[tuple[int, float]]
    ) -> list[ScoredChunk]:
        out: list[ScoredChunk] = []
        for rank, (index, score) in enumerate(pairs):
            scored = items[index]
            clone = scored.with_score(float(score))
            clone.rank = rank
            # Keep the first-stage score beside the new one. Fusion already
            # wrote its own components in here; overwriting the dict instead of
            # extending it would erase why the candidate was a candidate.
            clone.component_scores = {
                **scored.component_scores,
                f"pre_{self.name}": scored.score,
                self.name: float(score),
            }
            clone.explain = {
                **scored.explain,
                "reranked_by": self.name,
                "rerank_model": self.model_name,
                "rank_before": scored.rank,
            }
            out.append(clone)
        return out

    def _passthrough(self, items: Sequence[ScoredChunk], limit: int) -> list[ScoredChunk]:
        """First-stage order, untouched scores, marked as un-reranked so the
        trace does not claim a precision it did not deliver."""
        out: list[ScoredChunk] = []
        for rank, scored in enumerate(items[:limit]):
            clone = scored.with_score(scored.score)
            clone.rank = rank
            clone.explain = {**scored.explain, "rerank_degraded": self.name}
            out.append(clone)
        return out


# ---------------------------------------------------------------------------
# Cross-encoder (the default)
# ---------------------------------------------------------------------------
# NOTE on the registry name: ``cross_encoder`` and ``fastembed`` under the
# ``reranker`` kind already belong to the ONNX *model wrapper* in
# ``ragorc.embed``. This class is the pipeline *stage* that drives that model, so
# it takes its own key and ``build_reranker`` maps the user-facing
# ``settings.retrieval.reranker`` spelling onto it.
@register("reranker", "cross_encoder_stage")
class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranking with batching and a score cache.

    The model itself is :class:`ragorc.embed.fastembed_provider.FastEmbedReranker`
    — one ONNX session, one tokenizer, cached per ``(model, threads)`` in that
    module. Constructing a second loader here would duplicate the heaviest
    object in the process for no benefit, so this class wraps rather than
    reimplements. Any object satisfying :class:`Reranker` can be injected
    instead, which is how a hosted reranker (Cohere) is swapped in.

    Caching is worth it because the score is a deterministic function of the
    ``(query, chunk)`` pair, and that pair recurs constantly in a real service:
    Self-RAG and CRAG loops re-rank after a rewrite, multi-query variants fetch
    overlapping candidate sets, and repeat questions are the norm. Chunk ids are
    the cache identity because :func:`ragorc.core.ids.chunk_id` is content
    derived — an edited passage gets a new id, so a stale score can never be
    served for text that changed.
    """

    name = "cross_encoder"

    def __init__(
        self,
        model: Reranker | None = None,
        *,
        cache: Cache | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self.model = model or FastEmbedReranker(settings=self.settings)
        self.model_name = self.model.model_name
        self.cache = cache

    @property
    def cache_enabled(self) -> bool:
        """Both switches plus a backend: the global cache tier and the rerank
        opt-in must agree, or the backend is not touched at all."""
        cfg = self.settings.cache
        return self.cache is not None and cfg.enabled and cfg.cache_rerank

    async def _order(
        self,
        question: str,
        texts: list[str],
        ids: list[str] | None,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float]], Usage]:
        scores = await self._scores(question, texts, ids)
        # Raw logits: unbounded, comparable within a query, meaningless across
        # queries — and already higher-is-better, so nothing is converted.
        return _top_pairs(scores, top_k), Usage()

    async def _scores(
        self, question: str, texts: Sequence[str], ids: Sequence[str] | None
    ) -> FloatArray:
        if not self.cache_enabled:
            return np.asarray(await self._score_batches(question, list(texts)), dtype=np.float32)

        # Fall back to a content hash when the caller passed bare strings, so
        # the string-level API caches too rather than silently missing.
        identity = list(ids) if ids is not None else [content_hash(t) for t in texts]
        keys = [cache_key("rerank", self.model_name, question, key) for key in identity]
        values = await cached_batch(
            keys,
            list(texts),
            reader=self._cache_read,
            writer=self._cache_write,
            compute=lambda batch: self._score_batches(question, batch),
        )
        return np.asarray(values, dtype=np.float32)

    async def _score_batches(self, question: str, texts: list[str]) -> list[float]:
        """Score every pair, ``rerank_batch_size`` at a time.

        Batches are issued *sequentially*, not fanned out. One ONNX session
        already saturates the configured thread count, so concurrent batches
        would contend for the same cores while multiplying peak memory — the
        tokenized input of a batch is a ``(batch, max_length)`` integer tensor.
        The ``await`` between batches is what matters: the forward pass runs in
        a worker thread inside the wrapped model (``asyncio.to_thread``), so the
        event loop keeps servicing the other stores while this runs, and it gets
        a scheduling point after every batch instead of one after 200 pairs.
        """
        size = max(self.settings.retrieval.rerank_batch_size, 1)
        out = np.zeros(len(texts), dtype=np.float32)
        for start in range(0, len(texts), size):
            batch = texts[start : start + size]
            pairs = await self.model.rerank(question, batch)
            # The wrapped model returns pairs sorted by score; scatter them back
            # onto input positions in one vectorized assignment.
            index = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
            value = np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs))
            if int(index.size) < len(batch):
                # A hosted reranker configured with its own ``top_n`` returns
                # fewer pairs than it was given. Those documents were not scored,
                # not scored zero: park them below the batch minimum so they sort
                # last, instead of leaving them on an arbitrary 0.0 that outranks
                # every genuinely negative logit.
                floor = float(value.min()) - 1.0 if value.size else 0.0
                out[start : start + len(batch)] = floor
                log.warning(
                    "rerank_partial_scores",
                    model=self.model_name,
                    requested=len(batch),
                    returned=int(index.size),
                )
            out[start + index] = value
        return out.tolist()

    async def _cache_read(self, keys: Sequence[str]) -> list[float | None]:
        assert self.cache is not None
        raws = await bounded_gather(
            (self.cache.get(key) for key in keys),
            limit=max(self.settings.retrieval.max_concurrent_retrievers, 1),
        )
        return [
            _SCORE_FRAME.unpack(raw)[0]
            if raw is not None and len(raw) == _SCORE_FRAME.size
            else None
            for raw in raws
        ]

    async def _cache_write(self, values: Mapping[str, float]) -> None:
        assert self.cache is not None
        await bounded_gather(
            (
                self.cache.set(key, _SCORE_FRAME.pack(value), ttl=self.settings.cache.redis_ttl_s)
                for key, value in values.items()
            ),
            limit=max(self.settings.retrieval.max_concurrent_retrievers, 1),
        )


# ---------------------------------------------------------------------------
# ColBERT / late interaction
# ---------------------------------------------------------------------------
def _stack_ragged(matrices: Sequence[FloatArray], dim: int) -> tuple[FloatArray, IntArray]:
    """Pack ragged ``(n_tokens, dim)`` matrices into one ``(n, max_tokens, dim)``
    buffer, without a Python loop over documents.

    Every matrix is concatenated into a single ``(total_tokens, dim)`` block (one
    C-level copy) and scattered into the padded buffer by fancy indexing: ``rows``
    is each token's document, ``cols`` is its position within that document, both
    derived from the length vector with ``repeat``/``cumsum``. The alternative —
    assigning ``padded[i, :len] = m`` in a loop — pays Python dispatch and a
    fresh slice object per document, which at 200 candidates is the same order of
    magnitude as the matmul it is preparing for.
    """
    prepared = [np.ascontiguousarray(m, dtype=np.float32).reshape(-1, dim) for m in matrices]
    lengths = np.fromiter((m.shape[0] for m in prepared), dtype=np.int64, count=len(prepared))
    width = int(lengths.max())
    padded = np.zeros((len(prepared), width, dim), dtype=np.float32)
    if width == 0:
        return padded, lengths
    flat = np.concatenate(prepared)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)[:-1]))
    rows = np.repeat(np.arange(len(prepared), dtype=np.int64), lengths)
    cols = np.arange(flat.shape[0], dtype=np.int64) - np.repeat(starts, lengths)
    padded[rows, cols] = flat
    return padded, lengths


def maxsim(query_matrix: FloatArray, doc_matrices: Sequence[FloatArray]) -> FloatArray:
    """ColBERT MaxSim over ragged documents, as one ``einsum``.

    ``score(Q, D) = sum over query tokens q of max over document tokens t of
    Q[q] . D[t]`` — for a query matrix ``(nq, dim)`` and a document matrix
    ``(nd, dim)``, that is every entry of ``Q @ D.T`` reduced by a max along the
    document axis and a sum along the query axis. The point of writing it as a
    single ``einsum`` over a 3-D stack is that all N documents go through one
    batched GEMM: N separate ``Q @ D.T`` calls compute identical arithmetic while
    paying N dispatches into BLAS and forfeiting the batched kernel.

    **Why the mask is ``-inf`` and not zero.** Documents have different token
    counts, so the stack has to be padded; the padding rows are zeros, and a zero
    row's similarity to any query token is exactly ``0.0``. Since these vectors
    are unit-norm, real similarities live in ``[-1, 1]`` — so a padded row *wins*
    the max for any query token whose best genuine match is negative. Short
    documents would silently score higher than they should, in a way that shows
    up as mysteriously bad ranking rather than as an error. Masking the padded
    columns to ``-inf`` makes them unable to win any max, which is the only
    correct floor.

    A document with no tokens at all has nothing to max over and would reduce to
    ``-inf`` for every query token; it is given the theoretical MaxSim floor
    (``-nq``) instead so the result stays finite and it simply ranks last.

    (The alternative to padding is a flat ``(total_tokens, dim)`` matmul plus
    ``np.maximum.reduceat`` over segment offsets. It saves the padded memory, but
    ``reduceat`` has no defined answer for an empty segment and needs a separate
    correction pass for one, where the padded form keeps the reduction a plain
    ``max``/``sum`` over a rectangular array.)
    """
    if len(doc_matrices) == 0:
        return np.zeros(0, dtype=np.float32)
    q = np.ascontiguousarray(query_matrix, dtype=np.float32)
    if q.ndim != 2:
        raise ValueError(f"query matrix must be 2-D (n_tokens, dim), got shape {q.shape}")
    n_query, dim = int(q.shape[0]), int(q.shape[1])
    floor = -float(n_query)

    padded, lengths = _stack_ragged(doc_matrices, dim)
    if int(padded.shape[1]) == 0:
        return np.full(len(doc_matrices), floor, dtype=np.float32)

    # (n_docs, n_query_tokens, n_doc_tokens) in one pass.
    similarity = np.einsum("qd,ntd->nqt", q, padded, optimize=True)
    valid = np.arange(padded.shape[1], dtype=np.int64)[None, :] < lengths[:, None]
    np.copyto(similarity, -np.inf, where=~valid[:, None, :])
    scores = similarity.max(axis=2).sum(axis=1)
    return np.where(lengths > 0, scores, floor).astype(np.float32)


@register("reranker", "colbert", "late_interaction")
class ColBERTReranker(BaseReranker):
    """Late-interaction reranking by MaxSim over token-level matrices.

    Cheaper than a cross-encoder per candidate and more expressive than a pooled
    dot product, which makes it the right stage when the candidate set is in the
    hundreds. It also has a property no cross-encoder has: the document side is
    precomputable, so a chunk that came back from Qdrant with its multivector
    attached is scored for *free*.
    """

    name = "colbert"

    def __init__(
        self,
        embedder: LateInteractionEmbedder | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(settings)
        self.embedder = embedder or FastEmbedLateInteraction(settings=self.settings)
        self.model_name = self.embedder.model_name

    async def _order(
        self,
        question: str,
        texts: list[str],
        ids: list[str] | None,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float]], Usage]:
        matrices = await self.embedder.embed_documents(texts)
        return await self._score(question, matrices, top_k), Usage()

    async def _order_chunks(
        self, question: str, chunks: Sequence[ScoredChunk], top_k: int | None
    ) -> tuple[list[tuple[int, float]], Usage]:
        """Embed only the chunks that arrived without a multivector.

        Re-embedding 50 passages is 50 forward passes; reusing what the store
        returned is a memory read. Mixed candidate sets are normal — dense and
        sparse hits carry no ColBERT vector — so the two paths coexist.
        """
        matrices: list[FloatArray | None] = [c.chunk.multi for c in chunks]
        missing = [i for i, matrix in enumerate(matrices) if matrix is None]
        if missing:
            fresh = await self.embedder.embed_documents([chunks[i].chunk.content for i in missing])
            for index, matrix in zip(missing, fresh, strict=True):
                matrices[index] = matrix
        resolved = [m for m in matrices if m is not None]
        return await self._score(question, resolved, top_k), Usage()

    async def _score(
        self, question: str, matrices: Sequence[FloatArray], top_k: int | None
    ) -> list[tuple[int, float]]:
        query_matrix = await self.embedder.embed_query(question)
        scores = await asyncio.to_thread(self._maxsim_sync, query_matrix, list(matrices))
        # MaxSim is a similarity: higher is better already, no conversion.
        return _top_pairs(scores, top_k)

    @staticmethod
    def _maxsim_sync(query_matrix: FloatArray, matrices: list[FloatArray]) -> FloatArray:
        """The CPU-bound half, on a worker thread.

        Normalization is repeated here on purpose. The embedder emits unit-norm
        token vectors, but a multivector read back out of the store carries
        whatever was written to it, and MaxSim is only cosine if both sides are
        unit-norm. It is one pass over data we are about to matmul anyway.
        """
        return maxsim(l2_normalize(query_matrix), l2_normalize_list(matrices))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
@register("reranker", "identity", "none", "noop")
class IdentityReranker(BaseReranker):
    """No-op reranker, for benchmarking and for turning the stage off.

    ``rerank_chunks`` leaves the scores exactly as the first stage produced them
    — which is the point. An identity baseline that rewrote scores would change
    everything downstream that reads them (the relative cutoff, MMR's relevance
    term, the abstention thresholds), and the measurement would no longer isolate
    the reranker's contribution.
    """

    name = "identity"
    model_name = "identity"

    async def _order(
        self,
        question: str,
        texts: list[str],
        ids: list[str] | None,
        top_k: int | None,
    ) -> tuple[list[tuple[int, float]], Usage]:
        # The string-level API has no incoming scores, so it synthesizes ones
        # that are strictly decreasing in input position: still higher-is-better,
        # still a faithful reproduction of the input order.
        total = len(texts)
        keep = total if top_k is None else max(min(top_k, total), 1)
        return [(i, 1.0 - i / total) for i in range(keep)], Usage()

    async def _order_chunks(
        self, question: str, chunks: Sequence[ScoredChunk], top_k: int | None
    ) -> tuple[list[tuple[int, float]], Usage]:
        keep = top_k if top_k is not None else len(chunks)
        return [(i, chunks[i].score) for i in range(min(keep, len(chunks)))], Usage()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_reranker(
    name: str | None = None,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
    late_embedder: LateInteractionEmbedder | None = None,
    **kwargs: Any,
) -> BaseReranker:
    """Resolve ``settings.retrieval.reranker`` (or an explicit name) to a stage.

    ``late_embedder`` is the deployment's ColBERT embedder, which carries the
    shared embedding cache. Only the ColBERT stage receives it; passing it through
    ``**kwargs`` would hand it to a cross-encoder that cannot use it.

    The aliases accepted here are the *user-facing* spellings from
    :class:`~ragorc.core.settings.RetrievalSettings`, which are not identical to
    the registry keys: the embedding layer already owns ``cross_encoder`` under
    the ``reranker`` kind for its model wrapper (see the note above
    :class:`CrossEncoderReranker`).
    """
    resolved = settings or get_settings()
    key = (name or resolved.retrieval.reranker or "none").strip().lower().replace("-", "_")

    if key in {"cross_encoder", "cross_encoder_stage", "crossencoder", "ce", "fastembed"}:
        return CrossEncoderReranker(settings=resolved, **kwargs)
    if key in _COLBERT_RERANKER_NAMES:
        # Routed rather than passed through `**kwargs`: a cross-encoder has no
        # use for a late-interaction embedder and would reject it.
        return ColBERTReranker(late_embedder, settings=resolved, **kwargs)
    if key == "rankgpt":
        # Imported here rather than at module scope: the listwise reranker pulls
        # in the prompt library and the model router, and it subclasses
        # BaseReranker from this module, so a top-level import would be a cycle.
        from ragorc.retrieve.rankgpt import RankGPTReranker

        if llm is None:
            raise ConfigError(
                "the rankgpt reranker is an LLM component and needs llm=...",
                hint="pass the LLM instance, or select reranker='cross_encoder'",
            )
        return RankGPTReranker(llm, resolved, **kwargs)
    if key in {"none", "identity", "noop", ""}:
        return IdentityReranker(resolved)

    raise ConfigError(
        f"unknown reranker {key!r}",
        available=sorted({"cross_encoder", "colbert", "rankgpt", "none"}),
        registered=available("reranker")["reranker"],
    )
