"""FastEmbed providers — the default embedding stack (ADR-0004).

Why FastEmbed is the default
----------------------------
It runs ONNX Runtime, not torch: ~90 MB of wheels instead of ~2.5 GB, no CUDA
toolchain, cold start in hundreds of milliseconds, and quantized int8 models
that are competitive with the torch originals on CPU. One dependency covers all
four representations hybrid retrieval needs — dense, sparse/BM25, ColBERT
multivectors and cross-encoder reranking — which means one model cache, one
threading story and one place where inference happens.

Three design decisions in this module are load-bearing:

**Model instances are cached per ``(model, threads)`` in a module-level dict.**
Constructing a ``TextEmbedding`` downloads (first time), builds an ONNX
InferenceSession and loads a tokenizer: 0.5-3s and tens to hundreds of MB of
arena. Two components that both want ``bge-small`` must share one session, or a
pipeline with a splitter, an indexer and a semantic cache pays for three. The
dict is guarded by a ``threading.Lock`` rather than an asyncio lock because
loading happens *inside* a worker thread, where an asyncio primitive would be
the wrong kind of lock entirely.

**Every FastEmbed call is a synchronous generator, so it runs in a thread.**
``embed()`` yields batches as ONNX finishes them. Iterating it on the event loop
would stall every other coroutine for the full batch. ``asyncio.to_thread``
plus ONNX releasing the GIL turns that into real overlap with network I/O to
Qdrant and OpenRouter.

**Dimensions come from FastEmbed's model registry, not from a probe.**
``get_embedding_size(model)`` is a static lookup: no download, no forward pass,
available at construction time — which is what lets the vector store create a
collection before the first document arrives. ``warmup()`` still verifies it
against a real forward pass when you want the check.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.errors import EmbeddingError
from ragorc.core.models import FloatArray, SparseVector
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.embed._runtime import configure_onnx_runtime, register_shutdown_hook
from ragorc.embed.base import (
    BaseEmbedder,
    cached_batch,
    l2_normalize,
    l2_normalize_list,
    run_in_thread,
)
from ragorc.embed.cache import EmbeddingCache

log = structlog.get_logger(__name__)

__all__ = [
    "FastEmbedDense",
    "FastEmbedLateInteraction",
    "FastEmbedReranker",
    "FastEmbedSparse",
    "clear_model_cache",
]

# Disable ONNX telemetry before the runtime loads. See ragorc/embed/_runtime.py:
# its teardown race aborts the process on exit, which reads as a crash even
# though the work completed.
configure_onnx_runtime()

_MODELS: dict[tuple[str, str, int | None], Any] = {}
_LOAD_LOCK = threading.Lock()

_TOKEN_LIMIT_RE = re.compile(r"(\d[\d_]*)\s*(?:input\s+)?tokens?\s+truncation", re.IGNORECASE)


def clear_model_cache() -> int:
    """Drop every cached ONNX session. Returns how many were released.

    Used by tests and by long-lived workers that switch models at runtime; ONNX
    arenas are not returned to the OS until the session is collected.
    """
    with _LOAD_LOCK:
        count = len(_MODELS)
        _MODELS.clear()
    return count


# Drop sessions while the interpreter is still alive, so ONNX tears its
# threads down in a defined order instead of racing static destruction.
register_shutdown_hook(clear_model_cache)


def _model_class(kind: str) -> Any:
    """Resolve the FastEmbed class for a kind.

    Imported here, not at module scope: pulling in ``fastembed`` also pulls
    ``onnxruntime`` and ``huggingface_hub`` (~1s and ~200 MB of RSS), which
    nothing that merely imports ``ragorc.embed`` should have to pay for.
    """
    try:
        if kind == "dense":
            from fastembed import TextEmbedding

            return TextEmbedding
        if kind == "sparse":
            from fastembed import SparseTextEmbedding

            return SparseTextEmbedding
        if kind == "late":
            from fastembed import LateInteractionTextEmbedding

            return LateInteractionTextEmbedding
        if kind == "cross":
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            return TextCrossEncoder
    except ImportError as exc:  # pragma: no cover - fastembed is a base dep
        raise ImportError(
            "fastembed is required for the default embedding provider: "
            "pip install 'ragorc' (or 'ragorc[all]')"
        ) from exc
    raise EmbeddingError(f"unknown fastembed model kind {kind!r}")


def _load(kind: str, model_name: str, threads: int | None) -> Any:
    """Get or build a FastEmbed model, one instance per (kind, model, threads).

    Double-checked locking: the fast path is a dict hit with no lock at all, and
    the slow path serializes concurrent first loads so two threads cannot
    download the same model twice or build two ONNX sessions for it.
    """
    key = (kind, model_name, threads)
    model = _MODELS.get(key)
    if model is not None:
        return model
    with _LOAD_LOCK:
        model = _MODELS.get(key)
        if model is not None:
            return model
        cls = _model_class(kind)
        try:
            model = cls(model_name=model_name, threads=threads)
        except (ValueError, OSError) as exc:
            raise EmbeddingError(
                f"could not load fastembed {kind} model {model_name!r}",
                hint="check the name against <Class>.list_supported_models()",
                error=str(exc)[:300],
            ) from exc
        log.info("fastembed_model_loaded", kind=kind, model=model_name, threads=threads)
        _MODELS[key] = model
        return model


def _describe(kind: str, model_name: str) -> dict[str, Any]:
    """Static registry metadata for a model — no download, no ONNX session."""
    try:
        entries = _model_class(kind).list_supported_models()
    except (ImportError, EmbeddingError):  # pragma: no cover - fastembed is a base dep
        return {}
    lowered = model_name.lower()
    for entry in entries:
        if str(entry.get("model", "")).lower() == lowered:
            return dict(entry)
    return {}


def _max_tokens_from(description: str, fallback: int) -> int:
    """Read the truncation limit out of FastEmbed's model description.

    The registry states it in prose ("8192 input tokens truncation") and nowhere
    as a field. Parsing it is worth the regex: late chunking needs the real
    window of the model, and guessing 512 for jina-v2 would throw away 15/16ths
    of its context.
    """
    match = _TOKEN_LIMIT_RE.search(description or "")
    if not match:
        return fallback
    try:
        return int(match.group(1).replace("_", ""))
    except ValueError:  # pragma: no cover - regex only matches digits
        return fallback


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------
@register("dense_embedder", "fastembed")
class FastEmbedDense(BaseEmbedder):
    """Pooled dense embeddings via ``fastembed.TextEmbedding``."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        config = resolved.embedding
        name = model_name or config.dense_model
        meta = _describe("dense", name)
        dimension = config.dense_dimension or int(meta.get("dim") or 0)
        super().__init__(
            model_name=name,
            dimension=dimension,
            max_tokens=_max_tokens_from(str(meta.get("description", "")), config.max_length),
            cache=cache,
            settings=resolved,
        )
        if not dimension:
            log.warning(
                "embedding_dimension_unknown",
                model=name,
                hint="set RAGORC_EMBEDDING__DENSE_DIMENSION or call warmup()",
            )

    # -- inference ---------------------------------------------------------
    async def _embed_batch(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        return await run_in_thread(self._embed_sync, list(texts), is_query)

    def _embed_sync(self, texts: list[str], is_query: bool) -> list[FloatArray]:
        model = _load("dense", self.model_name, self.config.threads)
        if is_query and not self.config.query_prefix:
            # No configured prefix: let FastEmbed apply the model's own query
            # convention (a no-op for symmetric models, the right instruction
            # for the ones that need it). With a configured prefix we have
            # already applied it and must not have a second one added.
            raw = model.query_embed(texts, batch_size=self.config.batch_size)
        else:
            raw = model.embed(texts, batch_size=self.config.batch_size)
        vectors = list(raw)
        if not vectors:
            return []
        # FastEmbed already L2-normalizes most dense models; re-normalizing is
        # idempotent and makes the guarantee ours rather than the model's.
        return self._finalize(vectors)


# ---------------------------------------------------------------------------
# Sparse (BM25 / SPLADE)
# ---------------------------------------------------------------------------
@register("sparse_embedder", "fastembed")
class FastEmbedSparse:
    """Sparse vectors via ``fastembed.SparseTextEmbedding``.

    Two model families with genuinely different semantics behind one interface:

    * **BM25** (``Qdrant/bm25``, ``is_lexical=True``) — a hashed bag of stemmed
      terms. Document weights carry the BM25 term-frequency saturation and
      length normalization; the **IDF factor is deliberately absent** because it
      depends on corpus statistics the client does not have. Qdrant applies it
      server-side through the ``idf`` modifier on the sparse vector, which is
      what makes this true BM25 rather than an approximation.
    * **SPLADE** (``prithivida/Splade_PP_en_v1``, ``is_lexical=False``) — a
      learned expansion: a transformer emits weights over the whole vocabulary,
      so "car" retrieves "automobile". Roughly 4x the index size of BM25 and
      better on paraphrase.

    Queries **must** go through ``embed_query`` and not ``embed_documents``. For
    BM25, FastEmbed's ``query_embed`` emits weight 1.0 per query term and skips
    the length normalization that only makes sense for documents; scoring a
    query as if it were a document silently reweights every term.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.embedding
        self.model_name = model_name or (
            self.config.splade_model if self.config.use_splade else self.config.sparse_model
        )
        self.cache = cache
        meta = _describe("sparse", self.model_name)
        # `requires_idf` is exactly the "needs server-side IDF" marker, i.e. the
        # lexical families (BM25, BM42). Fall back to the name for registries
        # that predate the field.
        lowered = self.model_name.lower()
        self.is_lexical = bool(
            meta.get(
                "requires_idf", "bm25" in lowered or "bm42" in lowered or "minicoil" in lowered
            )
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._embed(list(texts), is_query=False)

    async def embed_query(self, text: str) -> SparseVector:
        return (await self._embed([text], is_query=True))[0]

    async def embed_queries(self, texts: Sequence[str]) -> list[SparseVector]:
        return await self._embed(list(texts), is_query=True)

    async def _embed(self, texts: list[str], *, is_query: bool) -> list[SparseVector]:
        if not texts:
            return []
        cache = self.cache
        if cache is None or not cache.enabled:
            return await run_in_thread(self._embed_sync, texts, is_query)
        kind = "sparse_q" if is_query else "sparse_d"
        keys = [cache.key(self.model_name, text, kind=kind) for text in texts]
        return await cached_batch(
            keys,
            texts,
            reader=cache.get_sparse_many,
            writer=cache.set_sparse_many,
            compute=lambda items: run_in_thread(self._embed_sync, items, is_query),
        )

    def _embed_sync(self, texts: list[str], is_query: bool) -> list[SparseVector]:
        model = _load("sparse", self.model_name, self.config.threads)
        raw = (
            model.query_embed(texts, batch_size=self.config.batch_size)
            if is_query
            else model.embed(texts, batch_size=self.config.batch_size)
        )
        return [_to_sparse_vector(item) for item in raw]

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model": self.model_name, "is_lexical": self.is_lexical}
        if self.cache is not None:
            out["cache"] = self.cache.stats()
        return out


def _to_sparse_vector(embedding: Any) -> SparseVector:
    """Convert FastEmbed's ``SparseEmbedding`` to our wire form.

    Both dtypes are forced: FastEmbed hands back int32 indices for BM25 and — in
    the query path — *integer* values (``np.ones_like`` over an int index array).
    Qdrant wants int64 indices and float32 weights, and a silent int/float mix
    here surfaces much later as a type error inside the client.
    """
    return SparseVector(
        np.ascontiguousarray(embedding.indices, dtype=np.int64),
        np.ascontiguousarray(embedding.values, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Late interaction (ColBERT)
# ---------------------------------------------------------------------------
@register("late_interaction_embedder", "fastembed")
class FastEmbedLateInteraction:
    """Token-level embeddings via ``fastembed.LateInteractionTextEmbedding``.

    Returns ``(n_tokens, dim)`` float32 matrices for MaxSim scoring. Storage is
    ~100x a single dense vector, which is why the default configuration uses
    this as a reranking stage over a few hundred candidates rather than as a
    first-stage index.

    It also backs late chunking: ``token_vectors`` exposes the per-token output
    *with character offsets*, which is the one thing pooled embedders cannot
    give you.
    """

    supports_token_offsets = True

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.embedding
        self.model_name = model_name or self.config.late_interaction_model
        self.cache = cache
        meta = _describe("late", self.model_name)
        self.dimension = int(meta.get("dim") or 0)
        self.max_tokens = _max_tokens_from(str(meta.get("description", "")), self.config.max_length)

    async def embed_documents(self, texts: Sequence[str]) -> list[FloatArray]:
        items = list(texts)
        if not items:
            return []
        cache = self.cache
        if cache is None or not cache.enabled:
            return await run_in_thread(self._embed_sync, items, False)
        keys = [cache.key(self.model_name, text, kind="multi_d") for text in items]
        return await cached_batch(
            keys,
            items,
            reader=cache.get_multi_many,
            writer=cache.set_multi_many,
            compute=lambda batch: run_in_thread(self._embed_sync, batch, False),
        )

    async def embed_query(self, text: str) -> FloatArray:
        """Queries use the model's query path: ColBERT pads short queries to a
        minimum length with ``[MASK]`` tokens, and that padding is part of how
        query expansion works — skipping it measurably hurts MaxSim."""
        vectors = await run_in_thread(self._embed_sync, [text], True)
        return vectors[0]

    def _embed_sync(self, texts: list[str], is_query: bool) -> list[FloatArray]:
        model = _load("late", self.model_name, self.config.threads)
        raw = (
            model.query_embed(texts, batch_size=self.config.batch_size)
            if is_query
            else model.embed(texts, batch_size=self.config.batch_size)
        )
        matrices = [np.ascontiguousarray(item, dtype=np.float32) for item in raw]
        if matrices and not self.dimension:
            self.dimension = int(matrices[0].shape[-1])
        # ColBERT vectors are unit-norm per token by construction; normalize
        # anyway so MaxSim is a plain matmul regardless of the model.
        return l2_normalize_list(matrices) if self.config.normalize else matrices

    # -- late-chunking support --------------------------------------------
    async def token_vectors(self, text: str) -> tuple[FloatArray, list[tuple[int, int]]]:
        """Per-token vectors for ``text`` plus each token's character span.

        Returns only *content* tokens: ``[CLS]``, ``[SEP]`` and ColBERT's
        document-marker token are dropped because they have no character span to
        pool over, while every returned row has already attended to the whole
        input and is therefore context-conditioned.
        """
        return await run_in_thread(self.token_vectors_sync, text)

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Character spans of every token in ``text``, *untruncated*.

        The tokenizer FastEmbed configures truncates at the model limit, which
        would hide everything past token 512 — exactly the region late chunking
        needs to plan windows over. A cloned tokenizer with truncation disabled
        answers the planning question without touching the shared one.
        """
        model = _load("late", self.model_name, self.config.threads)
        tokenizer = _untruncated_tokenizer(model)
        encoding = tokenizer.encode(text, add_special_tokens=False)
        return [tuple(pair) for pair in encoding.offsets]

    def token_vectors_sync(self, text: str) -> tuple[FloatArray, list[tuple[int, int]]]:
        """Blocking form of :meth:`token_vectors`.

        Public because it *is* the interface late chunking consumes: the pooling
        core already runs inside a worker thread, so it needs the synchronous
        method directly — awaiting from there would mean a second thread hop.
        ``_TokenBackend`` in :mod:`ragorc.embed.late_chunking` probes for this
        name together with ``token_offsets`` to decide whether an embedder is
        token-capable, so keeping it private made this class fail that probe and
        took the whole no-torch late-chunking path down with it.
        """
        model = _load("late", self.model_name, self.config.threads)
        worker = model.model  # the Colbert/JinaColbert implementation
        if getattr(worker, "model", None) is None:
            worker.load_onnx_model()
        tokenizer = worker.tokenizer
        if tokenizer is None:  # pragma: no cover - load_onnx_model sets it
            raise EmbeddingError(
                "fastembed model exposes no tokenizer; late chunking needs character offsets",
                model=self.model_name,
            )

        encoding = tokenizer.encode(text)
        # `onnx_embed` returns the raw per-token output, aligned 1:1 with the
        # input ids. The public `embed()` prunes punctuation and padding rows
        # *after* inference, which destroys that alignment and with it any hope
        # of mapping a character span onto a row.
        output = worker.onnx_embed([text])
        matrix = np.ascontiguousarray(output.model_output[0], dtype=np.float32)

        offsets: list[tuple[int, int]] = [tuple(pair) for pair in encoding.offsets]
        special = list(encoding.special_tokens_mask)
        if matrix.shape[0] == len(offsets) + 1:
            # ColBERT inserts its document/query marker token at position 1.
            offsets.insert(1, (0, 0))
            special.insert(1, 1)
        if matrix.shape[0] != len(offsets):
            raise EmbeddingError(
                "fastembed token output does not align with the tokenizer",
                model=self.model_name,
                rows=int(matrix.shape[0]),
                tokens=len(offsets),
                hint="fall back to EARLY or CONTEXTUAL chunking",
            )

        keep = ~np.asarray(special, dtype=bool)
        # Per-token L2 is part of ColBERT's contract (the public `embed()` does
        # it too) and it matters for pooling: the mean of unnormalized token
        # vectors is dominated by whichever tokens happen to have large norms.
        content = l2_normalize(matrix[keep])
        spans = [offsets[i] for i in np.flatnonzero(keep).tolist()]
        return content, spans

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"model": self.model_name, "dimension": self.dimension}
        if self.cache is not None:
            out["cache"] = self.cache.stats()
        return out


_CLONE_ATTR = "_ragorc_untruncated_tokenizer"


def _untruncated_tokenizer(model: Any) -> Any:
    """A truncation-free clone of a loaded model's tokenizer.

    Cloned through ``to_str``/``from_str`` rather than calling
    ``no_truncation()`` on the original: the original is shared by every caller
    of that cached model, and flipping its truncation policy would silently
    change inference for all of them. The clone is memoized on the worker
    object, so it lives and dies with the model it belongs to and the JSON round
    trip happens once.
    """
    worker = getattr(model, "model", model)
    if getattr(worker, "model", None) is None:
        worker.load_onnx_model()
    existing = getattr(worker, _CLONE_ATTR, None)
    if existing is not None:
        return existing
    tokenizer = worker.tokenizer
    if tokenizer is None:  # pragma: no cover - load_onnx_model sets it
        raise EmbeddingError("fastembed model exposes no tokenizer")
    from tokenizers import Tokenizer

    clone = Tokenizer.from_str(tokenizer.to_str())
    clone.no_truncation()
    clone.no_padding()
    setattr(worker, _CLONE_ATTR, clone)
    return clone


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
@register("reranker", "fastembed", "cross_encoder")
class FastEmbedReranker:
    """ONNX cross-encoder reranker via ``fastembed.rerank.TextCrossEncoder``.

    A cross-encoder reads the query and the document *together*, so it can score
    relevance a bi-encoder cannot represent — at the cost of one forward pass per
    pair, which is why it runs over ``rerank_top_k`` candidates and not the
    corpus.

    Scores are raw logits: unbounded, comparable within one query, meaningless
    across queries. Higher is better, which is the convention everywhere in this
    library, so they are returned unmodified rather than squashed through a
    sigmoid that would only lose resolution.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.embedding
        self.model_name = model_name or self.config.reranker_model

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        docs = list(documents)
        if not docs:
            return []
        scores = await run_in_thread(self._score_sync, query, docs)
        limit = min(top_k, len(docs)) if top_k else len(docs)
        # argpartition is O(n) against argsort's O(n log n); only the retained
        # slice is then sorted. At n=200, k=20 that is ~10x less comparison work.
        if limit < len(docs):
            top = np.argpartition(-scores, limit - 1)[:limit]
            top = top[np.argsort(-scores[top], kind="stable")]
        else:
            top = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in top]

    def _score_sync(self, query: str, documents: list[str]) -> FloatArray:
        model = _load("cross", self.model_name, self.config.threads)
        return np.fromiter(
            model.rerank(query, documents, batch_size=self.settings.retrieval.rerank_batch_size),
            dtype=np.float32,
            count=len(documents),
        )
