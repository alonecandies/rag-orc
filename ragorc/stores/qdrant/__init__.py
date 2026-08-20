"""Qdrant vector store.

Named vectors (``dense`` / ``sparse`` / ``colbert``) on shared points, so one
``query_points`` call can prefetch two indexes, fuse them with RRF or DBSF, and
rerank the survivors with ColBERT MaxSim — all server-side, one round trip.
"""

from __future__ import annotations

from ragorc.stores.qdrant.client import (
    build_client,
    close_all_clients,
    qdrant_settings,
    release_client,
)
from ragorc.stores.qdrant.collections import (
    COLBERT_VECTOR,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    bulk_load_mode,
    ensure_collection,
    ensure_payload_indexes,
    swap_alias,
    wait_for_green,
)
from ragorc.stores.qdrant.filters import TENANT_FIELD, to_qdrant_filter, with_tenant
from ragorc.stores.qdrant.store import QdrantStore

__all__ = [
    "COLBERT_VECTOR",
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "TENANT_FIELD",
    "QdrantStore",
    "build_client",
    "bulk_load_mode",
    "close_all_clients",
    "ensure_collection",
    "ensure_payload_indexes",
    "qdrant_settings",
    "release_client",
    "swap_alias",
    "to_qdrant_filter",
    "wait_for_green",
    "with_tenant",
]
