"""Embedding cache — raw float32 frames, not JSON.

Why the binary framing exists
-----------------------------
An embedding cache entry is a numeric array, and JSON is the wrong container for
one in every dimension that matters. A 1024-dim float32 vector is 4096 bytes as
``tobytes``; the same vector as a JSON array of decimal floats is ~20 KB and
costs a parse that allocates 1024 Python floats. On a 100k-chunk re-ingest that
is the difference between ~0.4 GB of cache traffic decoded in C and ~2 GB
decoded one PyObject at a time. ``numpy.frombuffer`` turns the payload back into
an array with a single memcpy.

Frame layout (little-endian, 16-byte header so the payload stays 8-byte aligned
and ``frombuffer`` needs no fixup):

    magic  4s   b"RGE1"     format tag; a stale frame is dropped, not decoded
    kind   B    1|2|3       dense | multi-vector | sparse
    pad    3x   -           alignment
    rows   I    n           1 for dense, token count for multi, nnz for sparse
    cols   I    dim         0 for sparse

Sparse vectors are two arrays in one frame: ``int64`` indices followed by
``float32`` values. Storing them as a dict-of-str-to-float — the shape Qdrant's
JSON API uses — would be ~8x larger and would need a Python-level rebuild of
both arrays on read.

Why ``get_many``/``set_many``
----------------------------
The backing :class:`~ragorc.core.protocols.Cache` is one-key-at-a-time, and the
Redis tier costs a network round trip per key. A 128-text batch therefore does
one bounded fan-out rather than 128 sequential awaits, which collapses the
latency of a cache check from 128 x RTT to roughly one RTT.

Keys are content hashes (``ragorc.core.ids.cache_key``), so identical text
embeds once per model regardless of which document it appeared in — deduplicated
boilerplate (headers, disclaimers, licence blocks) is a large fraction of a real
corpus.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.ids import cache_key
from ragorc.core.models import FloatArray, SparseVector
from ragorc.core.protocols import Cache
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["EmbeddingCache", "decode_frame", "encode_dense", "encode_multi", "encode_sparse"]

_MAGIC = b"RGE1"
_HEADER = struct.Struct("<4sB3xII")
_HEADER_SIZE = _HEADER.size  # 16

_KIND_DENSE = 1
_KIND_MULTI = 2
_KIND_SPARSE = 3


def encode_dense(vector: FloatArray) -> bytes:
    flat = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
    return _HEADER.pack(_MAGIC, _KIND_DENSE, 1, flat.size) + flat.tobytes()


def encode_multi(matrix: FloatArray) -> bytes:
    """Late-interaction ``(n_tokens, dim)`` matrix."""
    mat = np.ascontiguousarray(matrix, dtype=np.float32)
    if mat.ndim != 2:
        raise ValueError(f"multi-vector must be 2-D, got shape {mat.shape}")
    return _HEADER.pack(_MAGIC, _KIND_MULTI, mat.shape[0], mat.shape[1]) + mat.tobytes()


def encode_sparse(vector: SparseVector) -> bytes:
    indices = np.ascontiguousarray(vector.indices, dtype=np.int64)
    values = np.ascontiguousarray(vector.values, dtype=np.float32)
    header = _HEADER.pack(_MAGIC, _KIND_SPARSE, indices.size, 0)
    return header + indices.tobytes() + values.tobytes()


def decode_frame(raw: bytes) -> FloatArray | SparseVector | None:
    """Decode a frame, or ``None`` if it is truncated or from another format.

    A corrupt or stale entry must degrade to a cache miss. Raising here would
    turn a cache problem into an ingest failure.
    """
    if raw is None or len(raw) < _HEADER_SIZE:
        return None
    magic, kind, rows, cols = _HEADER.unpack_from(raw, 0)
    if magic != _MAGIC:
        return None
    body = memoryview(raw)[_HEADER_SIZE:]
    if kind == _KIND_DENSE:
        if body.nbytes != rows * cols * 4:
            return None
        # .copy() because frombuffer views immutable bytes and hands back a
        # read-only array; downstream code (quantization, in-place scaling)
        # expects to own its vectors.
        return np.frombuffer(body, dtype=np.float32, count=cols).copy()
    if kind == _KIND_MULTI:
        if body.nbytes != rows * cols * 4:
            return None
        return np.frombuffer(body, dtype=np.float32, count=rows * cols).reshape(rows, cols).copy()
    if kind == _KIND_SPARSE:
        if body.nbytes != rows * 12:  # int64 index + float32 value per entry
            return None
        indices = np.frombuffer(body, dtype=np.int64, count=rows).copy()
        values = np.frombuffer(body, dtype=np.float32, count=rows, offset=rows * 8).copy()
        return SparseVector(indices, values)
    return None


class EmbeddingCache:
    """Content-hash keyed embedding cache over any :class:`Cache` backend."""

    def __init__(
        self,
        backend: Cache,
        settings: Settings | None = None,
        *,
        concurrency: int | None = None,
    ) -> None:
        self.backend = backend
        self.settings = settings or get_settings()
        self.concurrency = concurrency or max(
            1, self.settings.indexing.max_concurrent_documents * 4
        )
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def enabled(self) -> bool:
        """Both switches must be on: the global cache tier and the embedding
        opt-in. Either one off means "do not touch the backend at all"."""
        return (
            self.settings.cache.enabled
            and self.settings.cache.cache_embeddings
            and self.settings.embedding.cache_embeddings
        )

    def key(self, model: str, text: str, *, kind: str = "dense", extra: Any = None) -> str:
        return cache_key("emb", model, kind, extra, text)

    # -- reads -------------------------------------------------------------
    async def _get_many(
        self, keys: Sequence[str], *, expect: type | tuple[type, ...]
    ) -> list[Any | None]:
        if not keys or not self.enabled:
            self.misses += len(keys)
            return [None] * len(keys)
        raws = await bounded_gather((self.backend.get(key) for key in keys), limit=self.concurrency)
        out: list[Any | None] = []
        for raw in raws:
            value = decode_frame(raw) if raw is not None else None
            if value is not None and not isinstance(value, expect):
                value = None  # frame of the wrong kind under this key
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            out.append(value)
        return out

    async def get_dense_many(self, keys: Sequence[str]) -> list[FloatArray | None]:
        values = await self._get_many(keys, expect=np.ndarray)
        return [v if v is None or v.ndim == 1 else None for v in values]

    async def get_multi_many(self, keys: Sequence[str]) -> list[FloatArray | None]:
        values = await self._get_many(keys, expect=np.ndarray)
        return [v if v is None or v.ndim == 2 else None for v in values]

    async def get_sparse_many(self, keys: Sequence[str]) -> list[SparseVector | None]:
        return await self._get_many(keys, expect=SparseVector)

    # -- writes ------------------------------------------------------------
    async def _set_many(
        self,
        items: Mapping[str, Any],
        *,
        encoder: Callable[[Any], bytes],
        ttl: float | None = None,
    ) -> None:
        if not items or not self.enabled:
            return
        ttl = ttl if ttl is not None else self.settings.cache.redis_ttl_s
        await bounded_gather(
            (self.backend.set(key, encoder(value), ttl=ttl) for key, value in items.items()),
            limit=self.concurrency,
        )
        self.writes += len(items)

    async def set_dense_many(
        self, items: Mapping[str, FloatArray], *, ttl: float | None = None
    ) -> None:
        await self._set_many(items, encoder=encode_dense, ttl=ttl)

    async def set_multi_many(
        self, items: Mapping[str, FloatArray], *, ttl: float | None = None
    ) -> None:
        await self._set_many(items, encoder=encode_multi, ttl=ttl)

    async def set_sparse_many(
        self, items: Mapping[str, SparseVector], *, ttl: float | None = None
    ) -> None:
        await self._set_many(items, encoder=encode_sparse, ttl=ttl)

    # -- single-item convenience ------------------------------------------
    async def get_dense(self, key: str) -> FloatArray | None:
        return (await self.get_dense_many([key]))[0]

    async def set_dense(self, key: str, vector: FloatArray, *, ttl: float | None = None) -> None:
        await self.set_dense_many({key: vector}, ttl=ttl)

    async def get_sparse(self, key: str) -> SparseVector | None:
        return (await self.get_sparse_many([key]))[0]

    async def set_sparse(self, key: str, vector: SparseVector, *, ttl: float | None = None) -> None:
        await self.set_sparse_many({key: vector}, ttl=ttl)

    # -- diagnostics -------------------------------------------------------
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "hit_rate": round(self.hit_rate, 3),
        }
