"""Collection schema, payload indexes, bulk-load mode and alias swaps.

The schema is the performance contract
--------------------------------------
Everything expensive about this store is decided here, at ``create_collection``
time, and cannot be renegotiated later without a reindex:

* **Named vectors, not three collections.** ``dense``, ``sparse`` and
  ``colbert`` live on the same points, which is what allows one
  ``query_points`` call to prefetch two of them, fuse the results and rerank
  with the third. Split across collections, hybrid search costs N round trips
  plus a client-side join on ids.

* **``colbert`` is deliberately un-indexed** (``HnswConfigDiff(m=0)``). This is
  the whole ColBERT-on-Qdrant insight: a multivector field is not a search
  index, it is a *scoring* field. Building HNSW over it would index every token
  vector of every chunk — ~100x the graph for a field whose only job is to
  MaxSim-rescore the few hundred candidates that a cheap index already found.
  With ``m=0`` Qdrant stores the matrices and evaluates MaxSim only for the
  points a ``prefetch`` handed it, which is exactly the access pattern.

* **IDF lives in the sparse config.** ``Modifier.IDF`` makes Qdrant compute the
  inverse document frequency term server-side from its own collection
  statistics, which turns a bag of term weights into real BM25 scoring. Without
  it a BM25 embedder's client-side weights are missing the corpus term and
  hybrid search quietly degrades to term-frequency matching. It is *wrong* for
  learned sparse (SPLADE) — those weights already encode importance, and
  applying IDF on top double-counts it.

* **Quantization is per-named-vector, not per collection.** It is attached to
  ``dense`` only: int8/binary rescoring interacts badly with MaxSim, and the
  multivector field is the last thing you want to lossily compress since it is
  the precision stage.

Payload indexes are not optional
--------------------------------
A filtered vector search without a payload index is a filtered *scan*: Qdrant
must evaluate the condition for every candidate the graph walk produces, and
with a selective filter the walk produces almost nothing useful. The tenant
index goes further — ``is_tenant=True`` tells Qdrant to physically group each
tenant's points, so a per-tenant query reads contiguous storage instead of
touching every segment.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient, models

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = [
    "COLBERT_VECTOR",
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "bulk_load_mode",
    "ensure_collection",
    "ensure_payload_indexes",
    "swap_alias",
    "wait_for_green",
]

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
COLBERT_VECTOR = "colbert"

_DEFAULT_COLBERT_DIM = 128
"""ColBERTv2's projection dimension. Only used when no late-interaction
embedder is available to report its own ``dimension``."""

_PQ_COMPRESSION = models.CompressionRatio.X16
"""Product quantization ratio. 16x is the point where PQ still beats int8 on
memory without the recall collapse that x32/x64 bring on <1k dimensions.
Not a setting because ``quantization: product`` is already the escape hatch for
"I know exactly what I am doing with memory"."""


def _quantization_config(settings: Settings) -> Any:
    """Translate ``qdrant.quantization`` into the server config object."""
    qs = settings.qdrant
    always_ram = qs.quantization_always_ram
    if qs.quantization == "scalar":
        # int8 with the default 0.99 quantile: 4x smaller, SIMD-friendly, ~1%
        # recall loss that `rescore` buys back.
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, always_ram=always_ram
            )
        )
    if qs.quantization == "binary":
        return models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=always_ram)
        )
    if qs.quantization == "product":
        return models.ProductQuantization(
            product=models.ProductQuantizationConfig(
                compression=_PQ_COMPRESSION, always_ram=always_ram
            )
        )
    return None


async def ensure_collection(
    client: AsyncQdrantClient,
    settings: Settings | None,
    dense_dim: int,
    *,
    has_sparse: bool = True,
    has_colbert: bool = False,
    colbert_dim: int = _DEFAULT_COLBERT_DIM,
    sparse_is_lexical: bool | None = None,
    collection: str | None = None,
    recreate: bool = False,
) -> bool:
    """Create the collection with named vectors if it is missing.

    Returns ``True`` when a collection was created, ``False`` when an existing
    one was left alone. ``recreate=True`` drops first — it is destructive and
    logged at warning level for that reason.

    ``sparse_is_lexical`` decides the IDF modifier. Left as ``None`` it is
    inferred from ``embedding.use_splade``; pass the embedder's own
    ``is_lexical`` flag when you have it, since the embedder is the authority on
    what its weights mean.
    """
    st = settings or get_settings()
    qs = st.qdrant
    name = collection or qs.collection

    if dense_dim <= 0:
        raise ConfigError(
            "dense vector dimension must be positive",
            dense_dim=dense_dim,
            hint="pass a dense embedder or set embedding.dense_dimension",
        )

    exists = await client.collection_exists(name)
    if exists and not recreate:
        log.debug("qdrant_collection_exists", collection=name)
        return False
    if exists:
        await client.delete_collection(name)
        log.warning("qdrant_collection_dropped", collection=name, reason="recreate")

    vectors: dict[str, models.VectorParams] = {
        DENSE_VECTOR: models.VectorParams(
            size=dense_dim,
            # Cosine, and the embedders L2-normalize, so the server's dot
            # product is the cosine similarity directly. Higher is better;
            # nothing downstream converts a distance.
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(
                m=qs.hnsw_m,
                ef_construct=qs.hnsw_ef_construct,
                full_scan_threshold=qs.full_scan_threshold,
            ),
            on_disk=qs.on_disk_vectors,
            quantization_config=_quantization_config(st),
        )
    }

    if has_colbert:
        vectors[COLBERT_VECTOR] = models.VectorParams(
            size=colbert_dim,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            # m=0 disables HNSW for this named vector on purpose. A multivector
            # field is a reranking field: it is only ever scored for the
            # candidates a `prefetch` selected, so an index over its token
            # vectors would cost ~100x the graph memory and buy nothing. Never
            # query "colbert" as a first stage — that is a full scan by
            # construction.
            hnsw_config=models.HnswConfigDiff(m=0),
            # ~100x the bytes of a dense vector, read for a few hundred points
            # per query: disk is the right place for it regardless of
            # on_disk_vectors.
            on_disk=True,
        )

    sparse_config: dict[str, models.SparseVectorParams] | None = None
    if has_sparse:
        lexical = (not st.embedding.use_splade) if sparse_is_lexical is None else sparse_is_lexical
        sparse_config = {
            SPARSE_VECTOR: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=qs.on_disk_vectors,
                    full_scan_threshold=qs.full_scan_threshold,
                ),
                # IDF turns client-side term weights into server-side BM25.
                # None for learned sparse: SPLADE weights already carry
                # importance and IDF would count it twice.
                modifier=models.Modifier.IDF if lexical else None,
            )
        }

    await client.create_collection(
        collection_name=name,
        vectors_config=vectors,
        sparse_vectors_config=sparse_config,
        shard_number=qs.shard_number,
        replication_factor=qs.replication_factor,
        write_consistency_factor=qs.write_consistency_factor,
        on_disk_payload=qs.on_disk_payload,
        optimizers_config=models.OptimizersConfigDiff(
            # Defer HNSW construction until a segment is worth indexing, and
            # let Qdrant pick the segment count (0 ~ CPU count) so search
            # parallelizes across segments.
            indexing_threshold=qs.indexing_threshold,
            default_segment_number=qs.default_segment_number,
        ),
    )
    log.info(
        "qdrant_collection_created",
        collection=name,
        dense_dim=dense_dim,
        sparse=has_sparse,
        colbert=has_colbert,
        colbert_dim=colbert_dim if has_colbert else None,
        quantization=qs.quantization,
        on_disk_vectors=qs.on_disk_vectors,
        shards=qs.shard_number,
    )
    return True


def _payload_index_schema(settings: Settings) -> dict[str, Any]:
    """The payload fields worth indexing, with the right index *type* per field.

    Types follow :meth:`ragorc.core.models.Chunk.payload`: ``level`` and
    ``index`` are written as integers, so they need integer indexes — a keyword
    index would simply never match an int, and the filter would silently fall
    back to a scan.
    """
    qs = settings.qdrant
    keyword = models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD)
    # lookup + range: `level == 0` (leaves only) and `level > 0` (RAPTOR
    # summaries) are both hot filters, and they need different index features.
    integer = models.IntegerIndexParams(
        type=models.IntegerIndexType.INTEGER, lookup=True, range=True
    )
    schema: dict[str, Any] = {
        "document_id": keyword,
        "modality": keyword,
        "parent_id": keyword,
        "level": integer,
        "index": integer,
    }
    schema["tenant_id"] = models.KeywordIndexParams(
        type=models.KeywordIndexType.KEYWORD,
        # is_tenant makes Qdrant co-locate one tenant's points on disk, so a
        # tenant-filtered search reads contiguous storage instead of scanning
        # every segment and discarding 99% of it.
        is_tenant=True if qs.use_multitenancy_index else None,
    )
    return schema


async def ensure_payload_indexes(
    client: AsyncQdrantClient,
    settings: Settings | None = None,
    *,
    collection: str | None = None,
) -> list[str]:
    """Create the payload indexes that are missing. Returns the fields created.

    Existing indexes are read from ``payload_schema`` and skipped rather than
    re-created: re-creating is a cluster-wide operation, and this function runs
    on every ``ensure_collection``.
    """
    st = settings or get_settings()
    name = collection or st.qdrant.collection
    schema = _payload_index_schema(st)

    info = await client.get_collection(name)
    present = set(info.payload_schema or {})
    missing = [field for field in schema if field not in present]
    if not missing:
        log.debug("qdrant_payload_indexes_present", collection=name, fields=sorted(present))
        return []

    await bounded_gather(
        (
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema[field], wait=True
            )
            for field in missing
        ),
        limit=4,
    )
    log.info(
        "qdrant_payload_indexes_created",
        collection=name,
        fields=missing,
        multitenancy=st.qdrant.use_multitenancy_index,
    )
    return missing


async def wait_for_green(
    client: AsyncQdrantClient,
    collection: str,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.0,
) -> bool:
    """Poll until the collection reports ``green``, i.e. indexing has settled.

    Returns ``False`` on timeout instead of raising: an unfinished optimizer is
    a latency problem, not a data-loss problem, and the caller (an ingest job)
    usually wants to log it and move on.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        info = await client.get_collection(collection)
        if info.status == models.CollectionStatus.GREEN:
            return True
        if time.monotonic() >= deadline:
            log.warning(
                "qdrant_index_not_green",
                collection=collection,
                status=str(info.status),
                waited_s=round(timeout_s, 1),
            )
            return False
        await asyncio.sleep(poll_interval_s)


@contextlib.asynccontextmanager
async def bulk_load_mode(
    client: AsyncQdrantClient,
    collection: str,
    *,
    restore_threshold: int | None = None,
    wait_for_green_s: float = 300.0,
    poll_interval_s: float = 1.0,
) -> AsyncIterator[None]:
    """Disable HNSW construction for the duration of a bulk ingest.

    This is the single biggest ingest speedup available. With a normal
    ``indexing_threshold`` the optimizer starts building HNSW graphs while data
    is still arriving, and because segments keep growing it rebuilds them
    repeatedly — the same vectors are linked into a graph several times over,
    and each rebuild competes with the writes for CPU. Setting the threshold to
    0 turns indexing off, so ingest is an append; restoring it afterwards builds
    each graph exactly once, over a segment whose final size is already known.
    On a multi-million-point load that is the difference between hours and
    minutes.

    The previous threshold is read from the live collection rather than from
    settings, so an operator's manual tuning survives an ingest. On exit the
    threshold is restored and — if the body succeeded — the function waits for
    ``green`` so callers can be sure that the search they run next is served by
    a finished index instead of a brute-force scan over unindexed segments.
    """
    previous = restore_threshold
    if previous is None:
        info = await client.get_collection(collection)
        previous = info.config.optimizer_config.indexing_threshold
    if previous is None:
        previous = get_settings().qdrant.indexing_threshold

    async def _restore() -> None:
        await client.update_collection(
            collection_name=collection,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=previous),
        )

    await client.update_collection(
        collection_name=collection,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    log.info("qdrant_bulk_load_begin", collection=collection, restore_threshold=previous)

    started = time.monotonic()
    try:
        yield
    except BaseException:
        # Best effort: whatever broke the ingest probably also breaks the
        # restore, and the ingest failure is the interesting exception.
        with contextlib.suppress(Exception):
            await _restore()
        log.warning(
            "qdrant_bulk_load_aborted",
            collection=collection,
            elapsed_s=round(time.monotonic() - started, 2),
        )
        raise

    # The body succeeded, so a failed restore is a real problem: leaving
    # indexing_threshold at 0 means every later search is a brute-force scan.
    await _restore()
    green = await wait_for_green(
        client, collection, timeout_s=wait_for_green_s, poll_interval_s=poll_interval_s
    )
    log.info(
        "qdrant_bulk_load_end",
        collection=collection,
        indexing_threshold=previous,
        green=green,
        elapsed_s=round(time.monotonic() - started, 2),
    )


async def swap_alias(client: AsyncQdrantClient, alias: str, new_collection: str) -> str | None:
    """Point ``alias`` at ``new_collection``. Returns the collection it left.

    Zero-downtime reindex: build ``ragorc_v2`` alongside the live
    ``ragorc_v1``, load and index it fully, then move the alias. The delete and
    the create are submitted as one alias operation batch, which Qdrant applies
    atomically — readers never observe a window where the alias resolves to
    nothing. The old collection is left in place so a rollback is one more
    ``swap_alias`` rather than a re-ingest.
    """
    aliases = (await client.get_aliases()).aliases
    current = next((a.collection_name for a in aliases if a.alias_name == alias), None)
    if current == new_collection:
        log.debug("qdrant_alias_unchanged", alias=alias, collection=new_collection)
        return current

    operations: list[Any] = []
    if current is not None:
        operations.append(
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=new_collection, alias_name=alias)
        )
    )
    await client.update_collection_aliases(change_aliases_operations=operations)
    log.info("qdrant_alias_swapped", alias=alias, previous=current, current=new_collection)
    return current
