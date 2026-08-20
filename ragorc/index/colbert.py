"""ColBERT late interaction: one vector per token, and what that costs.

Why a matrix instead of a vector
--------------------------------
A pooled dense vector is a lossy summary of a passage: every token is averaged
into one point, so a rare identifier — an error code, a version string, a drug
name — contributes a few percent of the direction and then disappears under the
topical mass of the rest of the chunk. That is precisely the query class dense
retrieval is worst at, and the reason hybrid search exists at all.

Late interaction (Khattab & Zaharia, 2020) keeps the per-token vectors and
defers the comparison to query time. Scoring is MaxSim: for each *query* token,
take its best-matching *document* token, and sum those maxima. The comparison
therefore happens at token granularity, which recovers exact-term matching
without giving up the semantic space — a query token "CVE-2021-44228" can find
its counterpart in a 400-token chunk even though the pooled vector of that chunk
says nothing about it.

Why this is a reranking field and never an index
------------------------------------------------
:mod:`ragorc.stores.qdrant.collections` creates the ``colbert`` named vector
with ``MultiVectorConfig(comparator=MAX_SIM)`` **and** ``HnswConfigDiff(m=0)``.
The ``m=0`` is the load-bearing part: a multivector field is not a search index,
it is a scoring field. Building HNSW over it would link every token vector of
every chunk into a graph — roughly two orders of magnitude more edges than the
dense graph, for a field whose only job is to rescore the few hundred candidates
a cheap index already proposed. With ``m=0``, Qdrant stores the matrices and
evaluates MaxSim only for the points a ``prefetch`` handed it.

So the access pattern this module feeds is::

    prefetch: dense + sparse, fused, limited to rerank_top_k   (cheap, indexed)
    query:    the ColBERT matrix, using="colbert"              (precise, scanned)

MaxSim costs ``O(candidates x query_tokens x doc_tokens)``. Two of those three
factors are bounded by configuration (``retrieval.rerank_top_k`` and the query
length); the third is what this module bounds.

Storage, and why pruning is not optional
----------------------------------------
One float32 token vector at ColBERTv2's 128 dimensions is 512 bytes. An
unpruned 512-token chunk is therefore 256 KiB, against 1.5 KiB for a 384-dim
dense vector — about 170x. Multiply by ten million chunks and the ColBERT field
alone is 2.4 TiB, which is how a promising reranking stage becomes an
unaffordable one.

:func:`prune_tokens` caps each chunk at ``max_tokens_per_doc`` vectors and drops
the *lowest-norm* rows first. The heuristic works because MaxSim only reads a
document token when that token is the argmax for some query token: removing a
row changes the score if and only if it was winning something. Token norm is the
encoder's own confidence signal — subword fragments, punctuation and function
words come out short, content words come out long — so norm-ordered pruning
removes rows that were unlikely to win any argmax before it removes rows that
were. The trade-off is real and one-directional: what pruning eventually costs
is the rare-term match that motivated using ColBERT in the first place, because
a rare token appears exactly once and pruning it deletes the only row that could
have matched it. Treat ``max_tokens_per_doc`` as a recall dial with a linear
price tag, not as a free optimization.

There is one wrinkle the heuristic has to handle: every embedder in this package
L2-normalizes ColBERT output per token (it is part of ColBERT's contract, and
:mod:`ragorc.embed.fastembed_provider` does it explicitly), which makes all
norms exactly 1.0 and the ranking degenerate. When that is detected, pruning
falls back to *centroid distance*: keep the tokens furthest from the chunk's own
mean token. Same objective — a token that is nearly the average of its
neighbours is a row that some other row can already answer for — expressed in
the only signal a unit-norm matrix still has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError, EmbeddingError
from ragorc.core.models import Chunk, FloatArray
from ragorc.core.protocols import LateInteractionEmbedder, VectorStore
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.embed.base import provider_concurrency, run_in_thread

log = structlog.get_logger(__name__)

__all__ = [
    "ColBERTIndexer",
    "StorageEstimate",
    "estimate_storage",
    "maxsim",
    "prune_tokens",
]

_DEFAULT_MAX_TOKENS_PER_DOC = 192
"""Token ceiling per chunk after pruning.

Chosen against the default chunk sizes rather than pulled from the air: at
``indexing.chunk_size = 512`` characters a chunk is ~128 tokens of English, so
the common case is untouched and pruning is a no-op, while the
``max_chunk_size = 2048`` outliers (~512 tokens) are cut ~2.7x — and those
outliers are exactly the chunks that would otherwise dominate the field's
footprint. Not a setting because it is a per-corpus storage/recall trade the
caller should make explicitly at construction time."""

_DEFAULT_COLBERT_DIM = 128
"""ColBERTv2's projection width. Used only to price storage before an embedder
has been loaded; the real number always comes from the embedder."""

_BYTES_PER_VALUE = 4
"""float32. Qdrant stores multivectors at full precision — quantization is
attached to the ``dense`` vector only, because lossily compressing the precision
stage defeats the point of having one."""

_NORM_SPREAD_FLOOR = 1e-3
"""Below this max-min spread the norms carry no ordering information (the matrix
is per-token L2-normalized) and pruning switches to centroid distance."""


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------
def prune_tokens(matrix: FloatArray, limit: int) -> FloatArray:
    """Keep at most ``limit`` token vectors, dropping the least useful first.

    Row order is preserved among the survivors. MaxSim is order-invariant, so
    that is not a correctness requirement — it keeps the stored matrix
    interpretable against the chunk's text when a retrieval result has to be
    explained.

    Selection uses ``argpartition``, not a sort: only the identity of the top
    ``limit`` rows matters, never their relative order, and partition is O(n)
    where a sort is O(n log n) on the hottest loop of the ingest path.
    """
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise EmbeddingError(
            "late-interaction matrix must be 2-D (n_tokens, dim)",
            shape=tuple(arr.shape),
        )
    rows = arr.shape[0]
    if limit <= 0 or rows <= limit:
        return np.ascontiguousarray(arr)

    norms = np.linalg.norm(arr, ord=2, axis=1)
    # Annotated because the two branches differ in width: the norms are float32,
    # the centroid distance promotes to float64. Only the ranking matters here,
    # never the dtype, and narrowing either branch would cost precision for a
    # cosmetic win.
    keep_score: npt.NDArray[np.floating[Any]]
    if float(norms.max() - norms.min()) > _NORM_SPREAD_FLOOR:
        keep_score = norms
        criterion = "norm"
    else:
        # Unit-norm rows: norms rank nothing. Distance from the chunk's own mean
        # token is the surviving signal — the closer a row sits to the centroid,
        # the more completely its neighbours already cover whatever it would
        # have matched.
        centroid = arr.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= float(_NORM_SPREAD_FLOOR):
            # Degenerate matrix (rows cancel out); norms are as good as anything.
            keep_score = norms
            criterion = "norm"
        else:
            keep_score = 1.0 - (arr @ (centroid / centroid_norm))
            criterion = "centroid_distance"

    keep = np.argpartition(keep_score, rows - limit)[rows - limit :]
    keep.sort()
    log.debug("colbert_pruned", tokens=rows, kept=limit, criterion=criterion)
    return np.ascontiguousarray(arr[keep])


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def maxsim(query_matrix: FloatArray, doc_matrices: Sequence[FloatArray]) -> FloatArray:
    r"""MaxSim score of one query matrix against many document matrices.

    ``score(d) = sum_q max_t <query[q], doc_d[t]>`` — for every query token, its
    best-matching document token; summed over query tokens. Higher is better, and
    on L2-normalized input every inner product is a cosine similarity, so nothing
    downstream converts a distance.

    The whole batch is **one** ``einsum`` over a padded 3-D array. Document
    matrices are ragged (a 40-token chunk next to a 190-token one), and the two
    obvious alternatives both lose: a Python loop of matmuls pays interpreter
    dispatch per candidate on a per-query hot path, and a concatenate-plus-
    segment-max needs an index gather that is slower than the padding it avoids.
    Padding wastes ``max_len - len(d)`` columns of arithmetic per document and
    buys a single BLAS-backed call.

    Padding rows are zeros, and a zero row's inner product is 0.0 — which would
    *win* the max for any query token whose real best match is negative. Hence
    the ``-inf`` mask over the ragged tail: it is not defensive tidiness, it is
    what makes the padded computation equal the ragged one.

    Worked example — 3 candidates, a 32-token query, ColBERTv2's 128 dims::

        query_matrix                     (32, 128)
        doc_matrices    [(190, 128), (40, 128), (137, 128)]
        lengths                          [190, 40, 137]
        padded                           (3, 190, 128)   longest wins
        einsum "qd,ntd->nqt"             (3, 32, 190)    every pair scored
        -inf where t >= lengths[n]       (3, 32, 190)    tail neutralized
        .max(axis=2)                     (3, 32)         best doc token per query token
        .sum(axis=1)                     (3,)            one score per candidate

    An empty document matrix scores 0.0 rather than ``-inf``: a chunk with no
    tokens is unrankable, not infinitely bad, and ``-inf`` would poison any
    downstream normalization it landed in.
    """
    if not doc_matrices:
        return np.empty(0, dtype=np.float32)

    query = np.ascontiguousarray(np.asarray(query_matrix, dtype=np.float32))
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if query.ndim != 2:
        raise EmbeddingError(
            "query matrix must be 2-D (n_query_tokens, dim)", shape=tuple(query.shape)
        )
    dim = int(query.shape[1])

    docs = [np.asarray(matrix, dtype=np.float32) for matrix in doc_matrices]
    bad = next((d for d in docs if d.ndim != 2 and d.size), None)
    if bad is not None:
        raise EmbeddingError(
            "document matrices must be 2-D (n_tokens, dim)", shape=tuple(bad.shape)
        )
    widths = {int(d.shape[1]) for d in docs if d.size}
    if widths and widths != {dim}:
        raise EmbeddingError(
            "query and document matrices disagree on dimension",
            query_dim=dim,
            document_dims=sorted(widths),
            hint="both sides must come from the same late-interaction model",
        )

    lengths = np.fromiter(
        (d.shape[0] if d.size else 0 for d in docs), dtype=np.int64, count=len(docs)
    )
    longest = int(lengths.max())
    if longest == 0:
        return np.zeros(len(docs), dtype=np.float32)

    padded = np.zeros((len(docs), longest, dim), dtype=np.float32)
    # A loop over *arrays*, not over elements: ragged input cannot be stacked in
    # one call, and every row copy inside it is a memcpy.
    for index, doc in enumerate(docs):
        rows = int(lengths[index])
        if rows:
            padded[index, :rows] = doc

    sims = np.einsum("qd,ntd->nqt", query, padded, optimize=True)
    pad_mask = np.arange(longest)[None, :] >= lengths[:, None]
    # Broadcast the (n, t) mask across the query axis without materializing an
    # (n, q, t) boolean copy of it.
    np.copyto(sims, -np.inf, where=pad_mask[:, None, :])

    scores = sims.max(axis=2).sum(axis=1)
    scores[lengths == 0] = 0.0
    return np.ascontiguousarray(scores, dtype=np.float32)


# ---------------------------------------------------------------------------
# Storage estimation
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class StorageEstimate:
    """What enabling the ColBERT field costs, before it is enabled."""

    chunks: int
    tokens_per_chunk: float
    dimension: int
    dense_dimension: int
    bytes_per_chunk: int
    total_bytes: int
    dense_bytes_per_chunk: int

    @property
    def ratio_vs_dense(self) -> float:
        """Multiple of the dense field's footprint. This is the number that
        decides whether ColBERT is affordable for a given corpus."""
        return self.bytes_per_chunk / self.dense_bytes_per_chunk

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 1024**3

    def report(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "tokens_per_chunk": round(self.tokens_per_chunk, 1),
            "dimension": self.dimension,
            "bytes_per_chunk": self.bytes_per_chunk,
            "dense_bytes_per_chunk": self.dense_bytes_per_chunk,
            "ratio_vs_dense": round(self.ratio_vs_dense, 1),
            "total_bytes": self.total_bytes,
            "total_gib": round(self.total_gib, 3),
        }


def estimate_storage(
    chunks: int,
    *,
    tokens_per_chunk: float,
    dimension: int = _DEFAULT_COLBERT_DIM,
    dense_dimension: int = 384,
) -> StorageEstimate:
    """Price the ``colbert`` field: ``tokens x dim x 4`` bytes per chunk.

    Payload, ids and the dense/sparse fields are excluded on purpose — this
    answers "what does turning late interaction on add", which is the decision
    being made. The estimate is a lower bound on disk: Qdrant's segment
    bookkeeping adds a few percent, and the field is created ``on_disk=True``,
    so this is disk pressure rather than RAM pressure.
    """
    per_chunk = round(max(tokens_per_chunk, 0.0) * max(dimension, 0) * _BYTES_PER_VALUE)
    dense_per_chunk = max(int(dense_dimension) * _BYTES_PER_VALUE, 1)
    return StorageEstimate(
        chunks=max(chunks, 0),
        tokens_per_chunk=float(tokens_per_chunk),
        dimension=int(dimension),
        dense_dimension=int(dense_dimension),
        bytes_per_chunk=per_chunk,
        total_bytes=per_chunk * max(chunks, 0),
        dense_bytes_per_chunk=dense_per_chunk,
    )


# ---------------------------------------------------------------------------
# The indexer
# ---------------------------------------------------------------------------
@register("indexer", "colbert")
class ColBERTIndexer:
    """Fills ``Chunk.multi`` with pruned ``(n_tokens, dim)`` matrices.

    The embedder is injected so the ingest path and the retriever share one ONNX
    session; when it is omitted it is built lazily from
    ``embedding.late_interaction_model`` on first use. Lazily, because importing
    ``fastembed`` pulls in ``onnxruntime`` and ``huggingface_hub`` (~1s, ~200 MB
    of RSS) and a caller who only wants :meth:`storage_estimate` — the "should I
    even turn this on" question — should not pay for it.
    """

    name = "colbert"

    def __init__(
        self,
        late_embedder: LateInteractionEmbedder | None = None,
        settings: Settings | None = None,
        *,
        store: VectorStore | None = None,
        max_tokens_per_doc: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.embedding
        self.store = store
        self.max_tokens_per_doc = int(
            max_tokens_per_doc if max_tokens_per_doc is not None else _DEFAULT_MAX_TOKENS_PER_DOC
        )
        self._embedder = late_embedder

    # -- plumbing ---------------------------------------------------------
    @property
    def embedder(self) -> LateInteractionEmbedder:
        if self._embedder is None:
            from ragorc.embed.fastembed_provider import FastEmbedLateInteraction

            self._embedder = FastEmbedLateInteraction(settings=self.settings)
            log.info(
                "colbert_embedder_built",
                model=self._embedder.model_name,
                reason="none injected",
            )
        return self._embedder

    @property
    def dimension(self) -> int:
        """Token-vector width, without forcing a model load.

        FastEmbed reports it from static registry metadata, so an injected
        embedder usually knows it before its first forward pass; when nobody
        knows, ColBERTv2's 128 is the right guess and only affects cost
        estimates, never stored data.
        """
        declared = int(getattr(self._embedder, "dimension", 0) or 0)
        return declared or _DEFAULT_COLBERT_DIM

    # -- indexing ---------------------------------------------------------
    async def index(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Populate ``chunk.multi`` in place; returns the same chunks.

        Chunks that already carry a matrix are skipped, which makes a retried or
        resumed ingest cheap rather than a full recompute — the same property
        ``skip_unchanged`` gives the document level. Whitespace-only chunks are
        skipped too: they would produce a matrix of special tokens that MaxSim can
        still score above zero, which is a false positive with no content behind it.

        Batches are sized by ``embedding.batch_size`` and run
        ``indexing.max_concurrent_documents`` at a time (via
        :func:`~ragorc.embed.base.provider_concurrency`, so ingest width is one
        dial rather than two multiplied together). Pruning runs in a worker
        thread with the batch it belongs to: it is numpy over a few hundred
        matrices, long enough to stall the loop and short enough that a thread
        hop per *matrix* would cost more than the work.
        """
        targets = [chunk for chunk in chunks if chunk.multi is None and chunk.content.strip()]
        skipped = len(chunks) - len(targets)
        if not targets:
            log.debug("colbert_index_noop", chunks=len(chunks), skipped=skipped)
            return list(chunks)

        size = max(1, self.config.batch_size)
        batches = [targets[i : i + size] for i in range(0, len(targets), size)]
        with timed("colbert.index", chunks=len(targets), batches=len(batches)):
            await bounded_gather(
                (self._index_batch(batch) for batch in batches),
                limit=provider_concurrency(self.settings),
            )

        tokens = np.asarray(
            [c.multi.shape[0] for c in targets if c.multi is not None], dtype=np.int64
        )
        mean_tokens = float(tokens.mean()) if tokens.size else 0.0
        log.info(
            "colbert_indexed",
            chunks=len(targets),
            skipped=skipped,
            mean_tokens=round(mean_tokens, 1),
            max_tokens=int(tokens.max()) if tokens.size else 0,
            pruned_at=self.max_tokens_per_doc,
            dimension=self.dimension,
            # Reported on every run, not only on request: the cost of this field
            # is the one thing an operator needs in front of them before deciding
            # to keep it enabled.
            storage=self.storage_estimate(len(targets), tokens_per_chunk=mean_tokens).report(),
        )
        return list(chunks)

    async def _index_batch(self, batch: Sequence[Chunk]) -> None:
        matrices = await self.embedder.embed_documents([chunk.embed_text for chunk in batch])
        if len(matrices) != len(batch):
            # A provider that silently drops inputs would attach every matrix to
            # the wrong chunk from that point on — unretrievable content that
            # still looks indexed. Never repair by position.
            raise EmbeddingError(
                "late-interaction embedder returned a different number of matrices",
                model=getattr(self.embedder, "model_name", "unknown"),
                expected=len(batch),
                received=len(matrices),
            )
        pruned = await run_in_thread(self._prune_batch, matrices)
        for chunk, matrix in zip(batch, pruned, strict=True):
            chunk.multi = matrix

    def _prune_batch(self, matrices: Sequence[FloatArray]) -> list[FloatArray]:
        """Prune a batch. Pure CPU — always called through a thread."""
        pruned = [prune_tokens(matrix, self.max_tokens_per_doc) for matrix in matrices]
        widths = {int(m.shape[1]) for m in pruned if m.size}
        if len(widths) > 1:
            raise EmbeddingError(
                "late-interaction matrices disagree on dimension within one batch",
                dimensions=sorted(widths),
            )
        declared = int(getattr(self._embedder, "dimension", 0) or 0)
        if widths and declared and widths != {declared}:
            # The collection's colbert field was sized from `dimension`; writing
            # a different width fails the whole upsert batch at the server.
            raise EmbeddingError(
                "late-interaction output does not match the embedder's declared dimension",
                declared=declared,
                observed=sorted(widths),
            )
        return pruned

    async def index_and_upsert(self, chunks: Sequence[Chunk]) -> int:
        """Index, then write the chunks through the injected vector store.

        The store is what turns ``chunk.multi`` into the ``colbert`` named vector
        (see :meth:`ragorc.stores.qdrant.store.QdrantStore._build_points`); this
        method exists so an ingest job can enable late interaction with one call
        instead of remembering the ordering.
        """
        if self.store is None:
            raise ConfigError(
                "ColBERTIndexer has no vector store",
                hint="pass store=... to index_and_upsert, or call index() and upsert yourself",
            )
        if not self.config.enable_late_interaction:
            # The collection only has a `colbert` field if it was created with
            # one; upserting matrices into a collection without it fails at the
            # server, and the fix is a setting, not a retry.
            log.warning(
                "colbert_upsert_without_flag",
                hint="set embedding.enable_late_interaction before creating the collection",
            )
        await self.index(chunks)
        return await self.store.upsert(chunks)

    # -- scoring / estimation --------------------------------------------
    async def embed_query(self, text: str) -> FloatArray:
        """Query-side matrix. Never pruned, deliberately.

        ColBERT pads short queries to a minimum length with ``[MASK]`` tokens and
        those tokens are how query expansion works — dropping them measurably
        hurts MaxSim. A query matrix is also transient: it costs one round trip
        of bytes, never storage, so there is nothing to buy by shrinking it.
        """
        return await self.embedder.embed_query(text)

    async def score(self, query_text: str, chunks: Sequence[Chunk]) -> FloatArray:
        """MaxSim ``query_text`` against chunks that already carry matrices.

        In-process scoring for the cases where the server cannot do it: a
        reranking pass over candidates that came from Postgres or Neo4j, or an
        offline evaluation. When the candidates live in Qdrant, prefer
        ``QdrantStore.search_colbert`` — the same arithmetic next to the data
        instead of after a full multivector download.
        """
        if not chunks:
            return np.empty(0, dtype=np.float32)
        missing = [chunk.id for chunk in chunks if chunk.multi is None]
        if missing:
            raise EmbeddingError(
                "chunks have no late-interaction matrices to score",
                missing=len(missing),
                first=missing[0],
                hint="call index() first, or fetch with with_vectors=True",
            )
        query = await self.embed_query(query_text)
        matrices = [chunk.multi for chunk in chunks if chunk.multi is not None]
        return await run_in_thread(maxsim, query, matrices)

    def storage_estimate(
        self, chunks: int, *, tokens_per_chunk: float | None = None
    ) -> StorageEstimate:
        """Cost of the ColBERT field for ``chunks`` chunks under this config.

        With no measurement to go on it assumes the pruning cap is hit, i.e. the
        worst case — which is the right bias for a number a user reads *before*
        enabling the feature.
        """
        return estimate_storage(
            chunks,
            tokens_per_chunk=(
                float(tokens_per_chunk)
                if tokens_per_chunk is not None
                else float(self.max_tokens_per_doc)
            ),
            dimension=self.dimension,
            dense_dimension=int(
                self.config.dense_dimension or self.settings.postgres.vector_dimension
            ),
        )

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": getattr(self._embedder, "model_name", self.config.late_interaction_model),
            "dimension": self.dimension,
            "max_tokens_per_doc": self.max_tokens_per_doc,
            "enabled": self.config.enable_late_interaction,
        }
