"""Shared embedding machinery.

Five concerns every provider in this package has, factored out so each one is
solved once and identically:

**L2 normalization, vectorized.** Cosine similarity reduces to a dot product
only on unit vectors, and every store in this library scores with a dot
product. A batch of N 1024-dim vectors normalizes in one ``np.linalg.norm``
call over an ``(N, 1024)`` matrix; the per-vector version does the same
arithmetic ~30x slower because each element pays Python dispatch overhead.

**Batching.** Hosted providers charge per input but *bill latency* per request:
one 2048-input call is one round trip, 2048 single-input calls are 2048 round
trips for the same money. The per-provider input ceiling is a module constant
in that provider — it is an API fact, not a tunable — while
``settings.embedding.batch_size`` lowers it for memory-constrained local
inference.

**Thread offload.** ONNX inference and sentence-transformers ``encode`` are
CPU-bound C calls that release the GIL. Left on the event loop they block every
other coroutine for the length of a batch — hundreds of milliseconds, which is
the entire latency budget of a retrieval stage — so they go through
``run_in_thread``.

**Asymmetric prefixes.** E5/BGE/GTE were trained with different instructions on
the two sides (``"query: "`` / ``"passage: "``). Embedding a query with the
document prefix costs several points of recall and raises no error anywhere, so
the prefix is applied in exactly one place: here.

**Cache handshake.** Content-hash keyed, checked once per batch, with in-batch
deduplication — a re-ingest of an unchanged corpus then performs zero forward
passes, which is the single largest cost saving available at index time.

One deliberate asymmetry in ``BaseEmbedder``: ``embed_queries`` is the batch
primitive and ``embed_query`` delegates to it, never the reverse. Deriving the
batch method from the single-item method is the most common performance mistake
in embedding wrappers — it silently converts one provider round trip into N.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from itertools import islice
from typing import Any, TypeVar

import numpy as np
import structlog

from ragorc.core.models import FloatArray
from ragorc.core.settings import EmbeddingSettings, Settings, get_settings
from ragorc.embed.cache import EmbeddingCache

T = TypeVar("T")
R = TypeVar("R")

log = structlog.get_logger(__name__)

__all__ = [
    "BaseEmbedder",
    "aclose_client",
    "apply_prefix",
    "batched",
    "cached_batch",
    "l2_normalize",
    "l2_normalize_list",
    "prefix_for",
    "provider_concurrency",
    "run_in_thread",
    "to_float32_matrix",
]

_EPS = np.float32(1e-12)
"""Norm floor. A zero vector (empty string, all-stopword query) would otherwise
divide by zero and poison every downstream dot product with NaN."""


def l2_normalize(arrays: FloatArray) -> FloatArray:
    """Normalize along the last axis. Accepts ``(dim,)``, ``(n, dim)`` or
    ``(n, tokens, dim)`` and does the whole batch in one pass."""
    arr = np.asarray(arrays, dtype=np.float32)
    norms = np.linalg.norm(arr, ord=2, axis=-1, keepdims=True)
    np.maximum(norms, _EPS, out=norms)
    return arr / norms


def l2_normalize_list(items: Sequence[FloatArray]) -> list[FloatArray]:
    """Normalize a list of vectors.

    When every entry has the same shape — the common case for dense output —
    they are stacked and normalized as one matrix, so the cost is one BLAS-level
    pass instead of N. Ragged input (late-interaction matrices differ in token
    count) falls back to per-array normalization; that loop is over *arrays*,
    never over dimensions.
    """
    if not items:
        return []
    first = items[0].shape
    if all(item.shape == first for item in items):
        return list(l2_normalize(np.stack(items)))
    return [l2_normalize(item) for item in items]


def to_float32_matrix(vectors: Sequence[Any]) -> FloatArray:
    """Stack provider output into one contiguous ``(n, dim)`` float32 matrix.

    float32 is not a rounding of convenience: it is what Qdrant and pgvector
    store, so emitting float64 doubles memory and adds a conversion on every
    upsert.
    """
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items.

    ``itertools.batched`` is 3.12+; this package supports 3.11, and ``islice``
    over a single iterator costs the same.
    """
    if size <= 0:
        raise ValueError(f"batch size must be positive, got {size}")
    iterator = iter(iterable)
    while chunk := list(islice(iterator, size)):
        yield chunk


async def run_in_thread(fn: Callable[..., R], /, *args: Any, **kwargs: Any) -> R:
    """Run blocking, CPU-bound work on the default executor.

    ONNX Runtime, tokenizers and torch all release the GIL inside their C
    kernels, so this is real parallelism rather than a politeness gesture — and
    it keeps the event loop free to keep other stores' requests in flight.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


def prefix_for(config: EmbeddingSettings, *, is_query: bool) -> str:
    return config.query_prefix if is_query else config.document_prefix


def apply_prefix(texts: Sequence[str], prefix: str) -> list[str]:
    """Prepend an instruction prefix. Concatenation is literal — E5 expects
    ``"query: text"``, so the trailing space belongs to the configured prefix,
    not to us."""
    if not prefix:
        return list(texts)
    return [f"{prefix}{text}" for text in texts]


def provider_concurrency(settings: Settings) -> int:
    """In-flight request ceiling for hosted embedding providers.

    Embedding fan-out happens during ingest, so it is bounded by the same knob
    that bounds ingest concurrency. Sharing the dial keeps total in-flight work
    proportional to the pipeline's width instead of multiplying two independent
    ceilings together.
    """
    return max(1, settings.indexing.max_concurrent_documents)


async def cached_batch(
    keys: Sequence[str],
    payloads: Sequence[T],
    *,
    reader: Callable[[Sequence[str]], Awaitable[list[R | None]]],
    writer: Callable[[Mapping[str, R]], Awaitable[None]],
    compute: Callable[[list[T]], Awaitable[list[R]]],
) -> list[R]:
    """Read-through cache for a batch, with in-batch de-duplication.

    One read pass, one compute pass over the misses only, one write pass. Two
    properties matter and are easy to get wrong:

    * repeated payloads inside the same batch collapse to one key and therefore
      one forward pass — boilerplate text is a large share of any real corpus;
    * output order matches input order exactly, so callers can zip results back
      onto their chunks without carrying an index.

    Dense, sparse and late-interaction embedders all share this, which is why it
    is parameterized by reader/writer instead of duplicated three times.
    """
    if not keys:
        return []
    first_index: dict[str, int] = {}
    for index, key in enumerate(keys):
        first_index.setdefault(key, index)
    unique = list(first_index)

    found = await reader(unique)
    by_key: dict[str, R] = {
        key: value for key, value in zip(unique, found, strict=True) if value is not None
    }
    missing = [key for key in unique if key not in by_key]
    if missing:
        fresh = await compute([payloads[first_index[key]] for key in missing])
        by_key.update(zip(missing, fresh, strict=True))
        await writer({key: by_key[key] for key in missing})
    return [by_key[key] for key in keys]


async def aclose_client(client: Any) -> None:
    """Release a vendor SDK client, whatever it calls the method.

    The hosted providers each wrap an httpx client inside their own SDK and expose
    a different name for closing it — ``aclose``, ``close``, or nothing at all.
    Guessing once here beats three provider-specific branches, and suppressing is
    right because a caller running this is on their way out: a socket that will not
    close cleanly must not stop the next provider from releasing its own.
    """
    if client is None:
        return
    for name in ("aclose", "close"):
        closer = getattr(client, name, None)
        if closer is None:
            continue
        with contextlib.suppress(Exception):
            result = closer()
            if inspect.isawaitable(result):
                await result
        return


class BaseEmbedder:
    """Mixin supplying the :class:`ragorc.core.protocols.DenseEmbedder` surface.

    Subclasses implement one method — ``_embed_batch`` — and inherit prefixing,
    caching, de-duplication and the query/document split.
    """

    def __init__(
        self,
        *,
        model_name: str,
        dimension: int,
        max_tokens: int,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config: EmbeddingSettings = self.settings.embedding
        self.model_name = model_name
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.cache = cache

    # -- the one method subclasses implement -------------------------------
    async def _embed_batch(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        """Embed already-prefixed, already-deduplicated texts."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _embed_batch(texts, is_query=...)"
        )

    # -- public API --------------------------------------------------------
    async def embed_documents(self, texts: Sequence[str]) -> list[FloatArray]:
        return await self._embed(texts, is_query=False)

    async def embed_queries(self, texts: Sequence[str]) -> list[FloatArray]:
        return await self._embed(texts, is_query=True)

    async def embed_query(self, text: str) -> FloatArray:
        return (await self.embed_queries([text]))[0]

    async def warmup(self) -> None:
        """Load the model and pin ``dimension`` from a real forward pass.

        Called once at startup so the first user-facing request does not pay
        model load (ONNX session + tokenizer, 0.5-3s) or discover a dimension
        mismatch against an already-created collection.
        """
        vector = (await self._embed_batch(["warmup"], is_query=False))[0]
        detected = int(vector.shape[-1])
        if detected and detected != self.dimension:
            log.info(
                "embedding_dimension_detected",
                model=self.model_name,
                declared=self.dimension,
                detected=detected,
            )
            self.dimension = detected

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model_name,
            "dimension": self.dimension,
            "max_tokens": self.max_tokens,
        }
        if self.cache is not None:
            out["cache"] = self.cache.stats()
        return out

    # -- internals ---------------------------------------------------------
    def _finalize(self, vectors: Sequence[Any]) -> list[FloatArray]:
        """Cast to float32 and normalize the whole batch in one pass."""
        matrix = to_float32_matrix(vectors)
        if self.config.normalize:
            matrix = l2_normalize(matrix)
        return list(matrix)

    async def _embed(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        prepared = apply_prefix(texts, prefix_for(self.config, is_query=is_query))
        if not prepared:
            return []
        cache = self.cache
        if cache is None or not cache.enabled:
            return await self._embed_batch(prepared, is_query=is_query)

        # Cache identity is (model, side, dimension override, normalization,
        # text). The dimension *override* is used rather than the detected
        # dimension so keys stay stable in a process that has not warmed up
        # yet — otherwise every restart would miss its own entries.
        kind = "dense_q" if is_query else "dense_d"
        extra = (self.config.dense_dimension, self.config.normalize)
        keys = [cache.key(self.model_name, text, kind=kind, extra=extra) for text in prepared]
        return await cached_batch(
            keys,
            prepared,
            reader=cache.get_dense_many,
            writer=cache.set_dense_many,
            compute=lambda items: self._embed_batch(items, is_query=is_query),
        )
