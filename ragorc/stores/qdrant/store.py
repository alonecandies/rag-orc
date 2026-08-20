"""The Qdrant vector store: one round trip for hybrid retrieval.

The design in one paragraph
---------------------------
Points carry up to three named vectors (``dense``, ``sparse``, ``colbert``).
A hybrid search is a *single* ``query_points`` call whose ``prefetch`` list runs
the dense and sparse branches inside the engine, whose ``query`` fuses them with
RRF or DBSF, and — when ColBERT reranking is on — whose *outer* query is the
late-interaction matrix so MaxSim rescoring also happens server-side. Nothing
about that is convenience: the client-side alternative is two round trips, two
JSON/protobuf decodes of ``fetch_k`` payloads each, a Python merge, and then a
third round trip to fetch the multivectors it needs to rerank. Same recall,
three times the latency and an order of magnitude more bytes on the wire.

The nesting is the whole point::

    prefetch = [ Prefetch(                       # candidate generation
                     prefetch=[dense, sparse],   #   two cheap indexed branches
                     query=FusionQuery(RRF),     #   fused server-side
                     limit=rerank_top_k) ]       #   bounded candidate window
    query    = colbert_matrix, using="colbert"   # precision: MaxSim rescoring

Read it inside-out: cheap indexes propose, fusion agrees, MaxSim decides. The
``limit`` on the middle layer is the cost control — MaxSim is O(candidates x
query_tokens x doc_tokens), so the candidate window, not the collection size,
determines what reranking costs. That is also why the ``colbert`` named vector
is created with ``m=0`` (no HNSW): it is only ever scored for candidates handed
to it, never searched. See :mod:`ragorc.stores.qdrant.collections`.

Other decisions worth knowing
-----------------------------
* **Named vectors are sent present-only.** A chunk without a ColBERT matrix must
  omit the key; an empty multivector is not "no vector", it is an invalid one and
  Qdrant rejects the whole batch.
* **``PointStruct`` construction runs in a thread.** ``ndarray.tolist()`` over a
  256-chunk batch of multivectors is millions of Python floats — hundreds of
  milliseconds of pure CPU that would otherwise stall every other coroutine on
  the loop.
* **Scores are similarities, never distances.** The collection uses Cosine and
  the embedders L2-normalize, so what Qdrant returns is already "higher is
  better" for dense, sparse (BM25 via the IDF modifier), fusion and MaxSim
  alike. No conversion anywhere.
* **``score_threshold`` is only applied to single-vector queries.** An RRF score
  is ~1/60 and a MaxSim score is a sum over query tokens; feeding a cosine
  threshold to either silently empties the result set.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager
from functools import partial
from typing import Any, TypeVar

import numpy as np
import structlog
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ragorc.core.concurrency import CircuitBreaker, bounded_gather, retry_async
from ragorc.core.errors import (
    ConfigError,
    RagOrcError,
    RateLimited,
    RetrievalError,
    StoreUnavailable,
    ValidationFailed,
)
from ragorc.core.ids import stable_uuid
from ragorc.core.models import (
    Chunk,
    FloatArray,
    FusionMethod,
    Query,
    RetrievalSource,
    ScoredChunk,
    SparseVector,
)
from ragorc.core.protocols import DenseEmbedder, LateInteractionEmbedder, SparseEmbedder
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.stores.qdrant.client import build_client, release_client
from ragorc.stores.qdrant.collections import (
    COLBERT_VECTOR,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    bulk_load_mode,
    ensure_payload_indexes,
)
from ragorc.stores.qdrant.collections import ensure_collection as ensure_collection_schema
from ragorc.stores.qdrant.filters import to_qdrant_filter, with_tenant

try:  # grpcio ships with qdrant-client; guard anyway so an unusual install works
    import grpc
except ImportError:  # pragma: no cover
    # Justification: mypy is right that this rebinds a module name to None, and
    # Python offers no annotation that says "module or None" -- pre-declaring
    # `grpc: ModuleType | None` before the import makes the import itself a
    # [no-redef] error, so there is no annotation-level fix. This suppression is
    # the idiom mypy's own docs prescribe for an optional import. It is inert
    # today (grpcio ships no py.typed, so `grpc` is Any here) and load-bearing
    # the moment types-grpcio is installed; the guarded use is `grpc is not
    # None and isinstance(exc, grpc.RpcError)` in _map_error, which is exactly
    # the None case mypy is warning about and which is handled.
    grpc = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)

__all__ = ["QdrantStore"]

T = TypeVar("T")

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 0.25
_RETRY_MAX_DELAY_S = 4.0
"""Retry budget, deliberately small. The whole store call has to finish inside
``retrieval.per_store_timeout_s`` (10s by default) or the retrying *is* the
outage: the multi-store retriever would rather drop this store and answer from
the others than wait. Long backoff belongs on the ingest path, where the
circuit breaker handles a genuinely dead server."""

_DEFAULT_COLBERT_DIM = 128
"""ColBERTv2's projection width, used only when no late embedder can report it."""

_SERVER_FUSION: dict[FusionMethod, models.Fusion] = {
    FusionMethod.RRF: models.Fusion.RRF,
    FusionMethod.DBSF: models.Fusion.DBSF,
}
"""The two fusions Qdrant can do itself. ``weighted``/``relative``/``max`` exist
only client-side (see the ensemble retriever), so they degrade to RRF here."""

_MAX_UINT64 = 2**64 - 1

_RETRYABLE_GRPC_CODES = frozenset(
    {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "RESOURCE_EXHAUSTED",
        "ABORTED",
        "INTERNAL",
        "UNKNOWN",
    }
)


def _point_id(chunk_id: str) -> str | int:
    """Coerce a chunk id into something Qdrant accepts as a point id.

    Qdrant point ids are an unsigned 64-bit integer or a UUID. Everything
    produced by :func:`ragorc.core.ids.chunk_id` is already a UUIDv5, so this is
    a safety net for hand-built chunks — and it has to be *deterministic*,
    because an upsert of the same logical chunk must replace rather than
    duplicate. UUID is tried first: a 32-digit numeric string is a valid hex
    UUID but not a valid uint64.
    """
    try:
        return str(uuid.UUID(chunk_id))
    except (ValueError, AttributeError, TypeError):
        pass
    if chunk_id.isdigit() and int(chunk_id) <= _MAX_UINT64:
        return int(chunk_id)
    return stable_uuid("chunk-id", chunk_id)


_ORIGINAL_ID_FIELD = "_ragorc_chunk_id"
"""Payload key holding the caller's chunk id when it differs from the point id.

Underscore-prefixed and namespaced so it cannot collide with a user metadata key,
and stripped by :func:`_chunk_from_payload` so it never reaches a prompt."""


def _chunk_from_payload(point_id: Any, payload: dict[str, Any] | None) -> Chunk:
    """Rebuild a chunk, restoring the caller's id.

    Every read path goes through here rather than calling
    ``Chunk.from_payload(str(point.id), ...)`` directly. That was the bug this
    replaces: ``get()`` restored the original id while ``search()`` and
    ``scroll()`` returned Qdrant's converted UUID, so the same chunk had two
    different ids depending on how it was found.
    """
    data = dict(payload or {})
    original = data.pop(_ORIGINAL_ID_FIELD, None)
    return Chunk.from_payload(str(original or point_id), data)


def _sparse_to_qdrant(vector: SparseVector) -> models.SparseVector:
    return models.SparseVector(
        indices=vector.indices.astype(np.int64, copy=False).tolist(),
        values=vector.values.astype(np.float32, copy=False).tolist(),
    )


def _sparse_from_qdrant(raw: Any) -> SparseVector | None:
    """Read a sparse vector back out of a point, from either transport shape."""
    if raw is None:
        return None
    indices = getattr(raw, "indices", None)
    values = getattr(raw, "values", None)
    if indices is None and isinstance(raw, dict):
        indices, values = raw.get("indices"), raw.get("values")
    if indices is None or values is None:
        return None
    return SparseVector(np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.float32))


def _translate(exc: BaseException, *, op: str) -> RagOrcError:
    """Map a driver exception onto our hierarchy.

    Only transport-level failures and server-side 5xx/429 become retryable.
    A 4xx is our own malformed request and a pydantic error is our own bad data:
    retrying either wastes the budget and, worse, keeps the circuit breaker
    tripped over a bug that no amount of waiting fixes.
    """
    if isinstance(exc, RagOrcError):
        return exc
    if isinstance(exc, UnexpectedResponse):
        # The driver types `status_code` as `int | None`: a reply whose status
        # line could not be read arrives here as None. That is a broken
        # connection rather than a rejected request, so it has to stay
        # retryable — and `int(None)` would raise a TypeError *out of this
        # translator*, replacing the real outage with a bug report about the
        # error handler and denying the circuit breaker the failure it needs.
        raw_status = exc.status_code
        if raw_status is None:
            return StoreUnavailable(
                "qdrant", f"malformed qdrant response during {op}", detail=str(exc)[:200]
            )
        status = int(raw_status)
        if status == 429:
            return RateLimited("qdrant rate limited", op=op, status=status)
        if status >= 500:
            return StoreUnavailable("qdrant", f"qdrant {status} on {op}", status=status)
        return RetrievalError(f"qdrant rejected {op}", status=status, detail=str(exc)[:300])
    if grpc is not None and isinstance(exc, grpc.RpcError):
        code = getattr(exc, "code", lambda: None)()
        name = getattr(code, "name", str(code))
        if name in _RETRYABLE_GRPC_CODES:
            return StoreUnavailable("qdrant", f"grpc {name} on {op}", grpc_code=name)
        return RetrievalError(f"qdrant grpc {name} on {op}", grpc_code=name)
    if isinstance(exc, ResponseHandlingException | ConnectionError | TimeoutError | OSError):
        return StoreUnavailable("qdrant", f"cannot reach qdrant during {op}", cause=str(exc)[:200])
    return RetrievalError(
        f"qdrant {op} failed", cause=str(exc)[:300], cause_type=type(exc).__name__
    )


@register("vector_store", "qdrant")
class QdrantStore:
    """Qdrant-backed :class:`ragorc.core.protocols.VectorStore`.

    Embedders are injected rather than constructed so that the ingest pipeline
    and the retriever share one loaded ONNX session; when they are omitted the
    store still works, it just requires the caller to bring pre-computed vectors
    on ``Chunk`` / ``Query``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncQdrantClient | None = None,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        late_embedder: LateInteractionEmbedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant.collection
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.late_embedder = late_embedder
        # Connections are built lazily and shared through the module cache in
        # client.py. Lazily, because build_client keys its cache on the running
        # event loop and a store is often constructed before the loop exists.
        self._client = client
        self._owns_client = client is None
        self._breaker = CircuitBreaker(name="qdrant")

    # -- plumbing ---------------------------------------------------------
    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = build_client(self.settings)
        return self._client

    @property
    def _timeout(self) -> int:
        return max(1, round(self.settings.qdrant.timeout_s))

    @property
    def _has_sparse(self) -> bool:
        return self.sparse_embedder is not None or self.settings.retrieval.use_sparse

    @property
    def _has_colbert(self) -> bool:
        return self.late_embedder is not None or self.settings.embedding.enable_late_interaction

    def _dense_dim(self) -> int:
        if self.dense_embedder is not None:
            return int(self.dense_embedder.dimension)
        if self.settings.embedding.dense_dimension:
            return int(self.settings.embedding.dense_dimension)
        raise ConfigError(
            "cannot determine the dense vector dimension",
            hint="pass dense_embedder=... or set embedding.dense_dimension",
        )

    def _colbert_dim(self) -> int:
        if self.late_embedder is not None:
            return int(self.late_embedder.dimension)
        return _DEFAULT_COLBERT_DIM

    async def _guard(self, op: str, call: Callable[[], Awaitable[T]]) -> T:
        """Run one server call behind the circuit breaker and the retry policy.

        ``call`` is a factory, not a coroutine: a retry needs a *fresh*
        coroutine, and awaiting a spent one raises ``RuntimeError`` instead of
        retrying.
        """
        self._breaker.check()

        @retry_async(
            max_attempts=_RETRY_ATTEMPTS,
            base_delay=_RETRY_BASE_DELAY_S,
            max_delay=_RETRY_MAX_DELAY_S,
        )
        async def _attempt() -> T:
            try:
                return await call()
            except Exception as exc:
                raise _translate(exc, op=op) from exc

        try:
            result = await _attempt()
        except StoreUnavailable:
            # Only unreachability counts against the breaker. A 429 is the
            # server working (tripping on it would take out a healthy cluster
            # under load) and a 4xx is our own malformed request, which no
            # amount of cooling off repairs.
            self._breaker.record_failure()
            raise
        self._breaker.record_success()
        return result

    def _query_filter(
        self, filters: dict[str, Any] | None, tenant_id: str | None
    ) -> models.Filter | None:
        return with_tenant(
            to_qdrant_filter(filters),
            tenant_id or self.settings.tenant_id,
            settings=self.settings,
        )

    def _search_params(self) -> models.SearchParams:
        """Search-time knobs, applied to every branch of every query.

        ``hnsw_ef`` is the recall/latency dial. The quantization block matters
        just as much: without ``rescore`` the int8 approximation *is* the final
        ranking, and oversampling exists so the rescoring pass has more than
        ``limit`` candidates to choose from.
        """
        qs = self.settings.qdrant
        quantization = None
        if qs.quantization != "none":
            quantization = models.QuantizationSearchParams(
                rescore=qs.rescore, oversampling=qs.oversampling
            )
        return models.SearchParams(hnsw_ef=qs.hnsw_ef_search, quantization=quantization)

    # -- schema -----------------------------------------------------------
    async def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection and its payload indexes if they are missing."""
        dim = self._dense_dim()
        created = await self._guard(
            "ensure_collection",
            lambda: ensure_collection_schema(
                self.client,
                self.settings,
                dim,
                has_sparse=self._has_sparse,
                has_colbert=self._has_colbert,
                colbert_dim=self._colbert_dim(),
                sparse_is_lexical=getattr(self.sparse_embedder, "is_lexical", None),
                collection=self.collection,
                recreate=recreate,
            ),
        )
        await self._guard(
            "ensure_payload_indexes",
            lambda: ensure_payload_indexes(self.client, self.settings, collection=self.collection),
        )
        log.info("qdrant_ready", collection=self.collection, created=created, dense_dim=dim)

    def bulk_load(self) -> AbstractAsyncContextManager[None]:
        """Context manager that turns indexing off for a bulk ingest.

        ``async with store.bulk_load(): await store.upsert(...)`` — see
        :func:`ragorc.stores.qdrant.collections.bulk_load_mode` for why this is
        the largest single ingest speedup available.
        """
        return bulk_load_mode(self.client, self.collection)

    # -- writes -----------------------------------------------------------
    def _build_points(self, chunks: Sequence[Chunk]) -> list[models.PointStruct]:
        """Convert chunks to points. Pure CPU — always called via ``to_thread``."""
        points: list[models.PointStruct] = []
        for chunk in chunks:
            vectors: dict[str, Any] = {}
            if chunk.dense is not None and chunk.dense.size:
                vectors[DENSE_VECTOR] = np.asarray(chunk.dense, dtype=np.float32).ravel().tolist()
            if chunk.sparse is not None and len(chunk.sparse):
                vectors[SPARSE_VECTOR] = _sparse_to_qdrant(chunk.sparse)
            if chunk.multi is not None and chunk.multi.size:
                matrix = np.asarray(chunk.multi, dtype=np.float32)
                if matrix.ndim != 2:
                    raise ValidationFailed(
                        "late-interaction vector must be 2-D (n_tokens, dim)",
                        chunk_id=chunk.id,
                        shape=tuple(matrix.shape),
                    )
                vectors[COLBERT_VECTOR] = matrix.tolist()
            if not vectors:
                # Writing a point with no vectors produces a chunk that can
                # never be retrieved — a silent data loss that only shows up as
                # unexplained recall gaps months later.
                raise ValidationFailed("chunk has no vectors to index", chunk_id=chunk.id)
            point_id = _point_id(chunk.id)
            payload = chunk.payload()
            if str(point_id) != chunk.id:
                # `_point_id` had to rewrite the id to satisfy Qdrant's uint64-or-
                # UUID constraint. Record the original so the read paths can hand
                # it back, because a chunk whose id changes in transit breaks four
                # things at once: near-duplicate collapse sees one chunk as two,
                # the citation validator cannot match `citation.chunk_id` against
                # the retrieved chunk, parent-document expansion cannot resolve
                # `parent_id`, and the cross-store join documented in
                # ragorc/core/ids.py stops holding against Postgres and Neo4j.
                #
                # Only written when it actually differs: production ids come from
                # `chunk_id()` and are already UUIDs, so the common case costs
                # nothing.
                payload[_ORIGINAL_ID_FIELD] = chunk.id
            points.append(models.PointStruct(id=point_id, vector=vectors, payload=payload))
        return points

    async def _upsert_batch(self, chunks: Sequence[Chunk]) -> int:
        points = await asyncio.to_thread(self._build_points, chunks)
        await self._guard(
            "upsert",
            lambda: self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=self.settings.qdrant.wait_on_upsert,
            ),
        )
        return len(points)

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        """Upsert chunks in parallel batches. Returns the number written.

        Batches are sized by ``qdrant.upsert_batch_size`` and run
        ``qdrant.parallel_upserts`` at a time: one giant request serializes on a
        single shard connection, and one request per chunk pays the round trip
        per point. ``wait_on_upsert`` is ``False`` in production so batches
        return as soon as the write is queued — the ingest pipeline does one
        final waiting flush rather than blocking on every batch.
        """
        if not chunks:
            return 0
        qs = self.settings.qdrant
        size = max(1, qs.upsert_batch_size)
        batches = [chunks[i : i + size] for i in range(0, len(chunks), size)]
        with timed("qdrant.upsert", chunks=len(chunks), batches=len(batches)):
            counts = await bounded_gather(
                (self._upsert_batch(batch) for batch in batches),
                limit=max(1, qs.parallel_upserts),
            )
        total = int(sum(counts))
        log.info(
            "qdrant_upsert",
            collection=self.collection,
            chunks=total,
            batches=len(batches),
            waited=qs.wait_on_upsert,
        )
        return total

    # -- query vectors ----------------------------------------------------
    async def _embed_dense(self, query: Query) -> None:
        assert self.dense_embedder is not None
        query.dense = await self.dense_embedder.embed_query(query.text)

    async def _embed_sparse(self, query: Query) -> None:
        assert self.sparse_embedder is not None
        query.sparse = await self.sparse_embedder.embed_query(query.text)

    async def _embed_multi(self, query: Query) -> None:
        assert self.late_embedder is not None
        query.multi = await self.late_embedder.embed_query(query.text)

    async def _ensure_vectors(
        self, query: Query, *, dense: bool, sparse: bool, multi: bool
    ) -> None:
        """Fill in whichever query representations are missing, concurrently.

        The results are written back onto the :class:`Query`, so a pipeline that
        hits two stores or retries a step embeds once. Three ONNX sessions
        overlap here; running them in sequence would add their latencies.
        """
        jobs: list[Coroutine[Any, Any, None]] = []
        if dense and query.dense is None and self.dense_embedder is not None:
            jobs.append(self._embed_dense(query))
        if sparse and query.sparse is None and self.sparse_embedder is not None:
            jobs.append(self._embed_sparse(query))
        if multi and query.multi is None and self.late_embedder is not None:
            jobs.append(self._embed_multi(query))
        if jobs:
            await bounded_gather(jobs, limit=len(jobs))

    @staticmethod
    def _dense_payload(vector: FloatArray) -> list[float]:
        return np.asarray(vector, dtype=np.float32).ravel().tolist()

    @staticmethod
    def _multi_payload(matrix: FloatArray) -> list[list[float]]:
        arr = np.asarray(matrix, dtype=np.float32)
        if arr.ndim != 2:
            raise ValidationFailed(
                "late-interaction query must be 2-D (n_tokens, dim)", shape=tuple(arr.shape)
            )
        return arr.tolist()

    # -- search -----------------------------------------------------------
    def _to_scored(
        self,
        points: Sequence[models.ScoredPoint],
        *,
        source: RetrievalSource,
        explain: dict[str, Any],
    ) -> list[ScoredChunk]:
        """``ScoredPoint`` -> ``ScoredChunk``, ranks filled from 0.

        ``component_scores`` carries the contribution under the name of the
        stage that produced the number. Server-side fusion is the one place
        where per-branch scores are genuinely unavailable — Qdrant returns the
        fused value only. That is the price of the single round trip; use the
        client-side ensemble retriever when the breakdown matters more than the
        latency.
        """
        out: list[ScoredChunk] = []
        for rank, point in enumerate(points):
            chunk = _chunk_from_payload(point.id, point.payload)
            score = float(point.score)
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    source=source,
                    rank=rank,
                    component_scores={source.value: score},
                    explain=dict(explain),
                )
            )
        return out

    def _fusion(self) -> models.Fusion:
        configured = self.settings.retrieval.fusion
        fusion = _SERVER_FUSION.get(configured)
        if fusion is None:
            log.warning(
                "qdrant_fusion_unsupported_server_side",
                configured=configured.value,
                using="rrf",
                hint="use the ensemble retriever for weighted/relative fusion",
            )
            return models.Fusion.RRF
        return fusion

    async def search(
        self,
        query: Query,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Hybrid search in one request.

        Accepted keyword arguments: ``fetch_k`` (candidates per branch),
        ``use_dense`` / ``use_sparse`` / ``colbert_rerank`` (override the
        corresponding settings), ``tenant_id``.
        """
        rs = self.settings.retrieval
        limit = int(top_k or query.top_k or rs.top_k)
        fetch_k = max(int(kwargs.get("fetch_k") or rs.fetch_k), limit)

        want_dense = bool(kwargs.get("use_dense", rs.use_dense))
        want_sparse = bool(kwargs.get("use_sparse", rs.use_sparse)) and self._has_sparse
        want_colbert = bool(kwargs.get("colbert_rerank", rs.colbert_rerank)) and self._has_colbert

        await self._ensure_vectors(query, dense=want_dense, sparse=want_sparse, multi=want_colbert)

        query_filter = self._query_filter(filters or query.filters, kwargs.get("tenant_id"))
        params = self._search_params()

        # Candidate-generation branches: only the indexed named vectors belong
        # here. `colbert` is intentionally excluded — its field has no HNSW
        # graph, so using it as a prefetch branch is a full collection scan.
        branches: list[models.Prefetch] = []
        reps: list[str] = []
        if want_dense and query.dense is not None:
            branches.append(
                models.Prefetch(
                    query=self._dense_payload(query.dense),
                    using=DENSE_VECTOR,
                    limit=fetch_k,
                    params=params,
                    filter=query_filter,
                )
            )
            reps.append(DENSE_VECTOR)
        if want_sparse and query.sparse is not None and len(query.sparse):
            branches.append(
                models.Prefetch(
                    query=_sparse_to_qdrant(query.sparse),
                    using=SPARSE_VECTOR,
                    limit=fetch_k,
                    params=params,
                    filter=query_filter,
                )
            )
            reps.append(SPARSE_VECTOR)

        # Bind the reranking decision to the matrix itself rather than to a
        # bool: every branch below that reranks needs the matrix, and a bool
        # carries no evidence that it is present.
        colbert: FloatArray | None = (
            query.multi
            if want_colbert and query.multi is not None and query.multi.size > 0
            else None
        )
        if not branches and colbert is None:
            raise RetrievalError(
                "no query representation available for search",
                hint="embed the query or inject embedders into QdrantStore",
            )

        prefetch: list[models.Prefetch] | None = None
        threshold: float | None = None
        source = RetrievalSource.DENSE
        mode = "single"

        if len(branches) > 1 and rs.server_side_fusion:
            fusion = self._fusion()
            if colbert is not None:
                # Three stages, one request. The inner prefetch generates
                # candidates from both cheap indexes; the middle layer fuses
                # them and *caps* the candidate set at rerank_top_k, which is
                # what bounds MaxSim cost; the outer query rescores exactly
                # those survivors with late interaction.
                candidates = max(rs.rerank_top_k, limit)
                prefetch = [
                    models.Prefetch(
                        prefetch=branches,
                        query=models.FusionQuery(fusion=fusion),
                        limit=candidates,
                        # No filter here: every branch below already applied it,
                        # so re-evaluating it on the fused list is pure work.
                    )
                ]
                query_obj: Any = self._multi_payload(colbert)
                using: str | None = COLBERT_VECTOR
                source = RetrievalSource.COLBERT
                mode = f"fusion_{fusion.value}+colbert"
            else:
                prefetch = branches
                query_obj = models.FusionQuery(fusion=fusion)
                using = None
                source = RetrievalSource.FUSED
                mode = f"fusion_{fusion.value}"
        elif branches and colbert is not None:
            # One branch plus reranking: same two-stage shape, no fusion layer
            # to insert between them.
            prefetch = branches
            query_obj = self._multi_payload(colbert)
            using = COLBERT_VECTOR
            source = RetrievalSource.COLBERT
            mode = f"{reps[0]}+colbert"
        elif branches:
            if len(branches) > 1:
                # Server-side fusion is off but two representations exist. Use
                # the first (dense before sparse) rather than silently returning
                # some mix nobody configured; client-side fusion is the ensemble
                # retriever's job, and it calls the single-mode methods below.
                log.debug("qdrant_server_side_fusion_disabled", reps=reps, using=reps[0])
            prefetch = None
            query_obj = branches[0].query
            using = branches[0].using
            source = RetrievalSource.DENSE if reps[0] == DENSE_VECTOR else RetrievalSource.SPARSE
            # A single-vector query returns cosine similarity, which is the only
            # scale score_threshold is calibrated for.
            threshold = rs.score_threshold
            mode = f"single_{reps[0]}"
        else:
            # ColBERT with nothing to prefetch from: correct, but every point's
            # matrix gets scored. Loud, because on a real corpus this is a
            # multi-second query.
            log.warning(
                "qdrant_colbert_full_scan",
                collection=self.collection,
                hint="enable dense or sparse retrieval so ColBERT only reranks",
            )
            # Reaching here means `branches` is empty, and the guard above
            # already rejected "no branches and no ColBERT matrix".
            assert colbert is not None
            prefetch = None
            query_obj = self._multi_payload(colbert)
            using = COLBERT_VECTOR
            source = RetrievalSource.COLBERT
            mode = "colbert_only"

        explain = {
            "store": "qdrant",
            "mode": mode,
            "reps": reps,
            "fetch_k": fetch_k,
            "top_k": limit,
        }
        with timed("qdrant.search", mode=mode, top_k=limit):
            response = await self._guard(
                "search",
                lambda: self.client.query_points(
                    collection_name=self.collection,
                    prefetch=prefetch,
                    query=query_obj,
                    using=using,
                    query_filter=query_filter,
                    search_params=params,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=threshold,
                    timeout=self._timeout,
                ),
            )
        results = self._to_scored(response.points, source=source, explain=explain)
        log.debug(
            "qdrant_search",
            collection=self.collection,
            mode=mode,
            returned=len(results),
            filtered=query_filter is not None,
        )
        return results

    async def _single(
        self,
        *,
        using: str,
        query_obj: Any,
        source: RetrievalSource,
        top_k: int,
        query_filter: models.Filter | None,
        prefetch: list[models.Prefetch] | None = None,
        apply_threshold: bool = True,
    ) -> list[ScoredChunk]:
        params = self._search_params()
        response = await self._guard(
            f"search_{using}",
            lambda: self.client.query_points(
                collection_name=self.collection,
                prefetch=prefetch,
                query=query_obj,
                using=using,
                query_filter=query_filter,
                search_params=params,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=(
                    self.settings.retrieval.score_threshold if apply_threshold else None
                ),
                timeout=self._timeout,
            ),
        )
        return self._to_scored(
            response.points,
            source=source,
            explain={"store": "qdrant", "mode": f"single_{using}", "top_k": top_k},
        )

    async def search_dense(
        self,
        query: Query,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> list[ScoredChunk]:
        """Dense-only search. Used by the client-side ensemble retriever, which
        needs each modality's own ranking to fuse them itself."""
        limit = int(top_k or query.top_k or self.settings.retrieval.top_k)
        await self._ensure_vectors(query, dense=True, sparse=False, multi=False)
        if query.dense is None:
            raise RetrievalError("no dense query vector", hint="inject a dense embedder")
        return await self._single(
            using=DENSE_VECTOR,
            query_obj=self._dense_payload(query.dense),
            source=RetrievalSource.DENSE,
            top_k=limit,
            query_filter=self._query_filter(filters or query.filters, tenant_id),
        )

    async def search_sparse(
        self,
        query: Query,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> list[ScoredChunk]:
        """Sparse-only search. With ``Modifier.IDF`` on the collection this is
        real BM25 scoring computed by the engine, not client-side term weights."""
        limit = int(top_k or query.top_k or self.settings.retrieval.top_k)
        await self._ensure_vectors(query, dense=False, sparse=True, multi=False)
        if query.sparse is None or not len(query.sparse):
            raise RetrievalError("no sparse query vector", hint="inject a sparse embedder")
        source = (
            RetrievalSource.BM25
            if getattr(self.sparse_embedder, "is_lexical", not self.settings.embedding.use_splade)
            else RetrievalSource.SPARSE
        )
        return await self._single(
            using=SPARSE_VECTOR,
            query_obj=_sparse_to_qdrant(query.sparse),
            source=source,
            top_k=limit,
            query_filter=self._query_filter(filters or query.filters, tenant_id),
            # BM25 scores are unbounded and corpus-dependent; a cosine-scale
            # threshold does not mean anything here.
            apply_threshold=False,
        )

    async def search_colbert(
        self,
        query: Query,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        candidates: int | None = None,
    ) -> list[ScoredChunk]:
        """Late-interaction ranking, bounded by a dense prefetch when possible.

        MaxSim over the whole collection is a full scan by construction (the
        ``colbert`` field has no HNSW graph), so when a dense vector is available
        this narrows to ``candidates`` dense hits first and rescores those. That
        keeps a per-modality ColBERT ranking affordable for the ensemble
        retriever, which needs it separately from the fused query.
        """
        rs = self.settings.retrieval
        limit = int(top_k or query.top_k or rs.top_k)
        window = max(int(candidates or rs.fetch_k), limit)
        await self._ensure_vectors(query, dense=True, sparse=False, multi=True)
        if query.multi is None or not query.multi.size:
            raise RetrievalError("no late-interaction query vectors", hint="inject a late_embedder")
        query_filter = self._query_filter(filters or query.filters, tenant_id)
        prefetch = None
        if query.dense is not None:
            prefetch = [
                models.Prefetch(
                    query=self._dense_payload(query.dense),
                    using=DENSE_VECTOR,
                    limit=window,
                    params=self._search_params(),
                    filter=query_filter,
                )
            ]
        else:
            log.warning("qdrant_colbert_full_scan", collection=self.collection)
        return await self._single(
            using=COLBERT_VECTOR,
            query_obj=self._multi_payload(query.multi),
            source=RetrievalSource.COLBERT,
            top_k=limit,
            query_filter=query_filter,
            prefetch=prefetch,
            # MaxSim sums over query tokens: the scale is "number of query
            # terms", not [0, 1].
            apply_threshold=False,
        )

    # -- reads ------------------------------------------------------------
    def _attach_vectors(self, chunk: Chunk, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        dense = raw.get(DENSE_VECTOR)
        if dense is not None:
            chunk.dense = np.asarray(dense, dtype=np.float32)
        sparse = _sparse_from_qdrant(raw.get(SPARSE_VECTOR))
        if sparse is not None:
            chunk.sparse = sparse
        multi = raw.get(COLBERT_VECTOR)
        if multi:
            chunk.multi = np.asarray(multi, dtype=np.float32)

    async def get(self, ids: Sequence[str], *, with_vectors: bool = False) -> list[Chunk]:
        """Fetch chunks by id, in the order requested.

        Missing ids are skipped rather than returned as ``None``: the parent
        retriever's expansion set routinely names chunks that a later ingest
        replaced, and a hole in the list is not useful to any caller.
        """
        if not ids:
            return []
        point_ids = [_point_id(i) for i in ids]
        records = await self._guard(
            "get",
            lambda: self.client.retrieve(
                collection_name=self.collection,
                ids=point_ids,
                with_payload=True,
                with_vectors=with_vectors,
            ),
        )
        by_id = {str(record.id): record for record in records}
        out: list[Chunk] = []
        for original, point_id in zip(ids, point_ids, strict=True):
            record = by_id.get(str(point_id))
            if record is None:
                continue
            # Reconstruct with the id the caller asked for, which is the id the
            # rest of the pipeline (and Postgres, and Neo4j) knows.
            chunk = _chunk_from_payload(original, record.payload)
            if with_vectors:
                self._attach_vectors(chunk, record.vector)
            out.append(chunk)
        return out

    async def count(
        self,
        *,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        exact: bool = True,
        **kwargs: Any,
    ) -> int:
        """Count points, optionally filtered. ``exact=False`` is an estimate
        from segment statistics — much cheaper on very large collections."""
        result = await self._guard(
            "count",
            lambda: self.client.count(
                collection_name=self.collection,
                count_filter=self._query_filter(filters, tenant_id),
                exact=exact,
                timeout=self._timeout,
            ),
        )
        return int(result.count)

    async def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        wait: bool = True,
        **kwargs: Any,
    ) -> int:
        """Delete by id or by filter. Returns the number of points removed.

        Qdrant's update result carries no affected-row count, so the filter path
        counts first: one cheap extra request in exchange for a return value
        that is true.

        Deleting by id still goes through a filter when tenant isolation is on —
        ``HasIdCondition`` ANDed with the tenant condition — because a bare id
        list would let one tenant delete another tenant's point by guessing a
        deterministic id.
        """
        if ids:
            point_ids = [_point_id(i) for i in ids]
            scoped = self._query_filter(None, tenant_id)
            if scoped is None:
                selector: Any = models.PointIdsList(points=point_ids)
                removed = len(point_ids)
            else:
                id_filter = models.Filter(
                    must=[models.HasIdCondition(has_id=point_ids), *(scoped.must or [])]
                )
                selector = models.FilterSelector(filter=id_filter)
                removed = int(
                    (
                        await self._guard(
                            "count",
                            lambda: self.client.count(
                                collection_name=self.collection,
                                count_filter=id_filter,
                                exact=True,
                            ),
                        )
                    ).count
                )
        elif filters:
            query_filter = self._query_filter(filters, tenant_id)
            if query_filter is None:
                raise ValidationFailed("delete filter translated to no condition")
            removed = await self.count(filters=filters, tenant_id=tenant_id)
            selector = models.FilterSelector(filter=query_filter)
        else:
            raise ValidationFailed("delete requires either ids or filters")

        await self._guard(
            "delete",
            lambda: self.client.delete(
                collection_name=self.collection, points_selector=selector, wait=wait
            ),
        )
        log.info(
            "qdrant_delete",
            collection=self.collection,
            removed=removed,
            by="ids" if ids else "filter",
        )
        return removed

    async def scroll(
        self,
        *,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        batch_size: int | None = None,
        limit: int | None = None,
        with_vectors: bool = False,
    ) -> AsyncIterator[Chunk]:
        """Iterate the collection page by page.

        Cursor-based, not offset-based: Qdrant returns the next page's start id,
        so paging stays O(page) instead of degrading as the offset grows. This
        is the primitive behind re-embedding, migration and export — all jobs
        that must not materialize the whole corpus in memory, hence an async
        generator rather than a list.
        """
        page = max(1, batch_size or self.settings.indexing.batch_size)
        query_filter = self._query_filter(filters, tenant_id)
        offset: Any = None
        emitted = 0
        while True:
            take = page if limit is None else min(page, limit - emitted)
            if take <= 0:
                return
            # The cursor and the page size are bound now, not captured: a
            # closure over the loop variables would read the *next* iteration's
            # values on a retry. `partial` binds them at construction the way a
            # default argument would, and keeps the call's type inferable.
            records, offset = await self._guard(
                "scroll",
                partial(
                    self.client.scroll,
                    collection_name=self.collection,
                    scroll_filter=query_filter,
                    limit=take,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                ),
            )
            for record in records:
                chunk = _chunk_from_payload(record.id, record.payload)
                if with_vectors:
                    self._attach_vectors(chunk, record.vector)
                yield chunk
                emitted += 1
            if offset is None or not records:
                return

    async def close(self) -> None:
        """Release the connection if this store built it.

        An injected client belongs to the caller: closing it here would break
        every other store sharing it.
        """
        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await release_client(client)
