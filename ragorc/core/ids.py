"""Deterministic identifiers.

Ids are content-derived, never random. Three consequences that matter:

* **Idempotent ingest** — re-running a load upserts in place instead of
  duplicating, so a crashed job is simply restarted.
* **Cross-store joins** — the same chunk has the same id in Qdrant, Postgres
  and Neo4j without a mapping table.
* **Cache keys** — a chunk's id *is* its embedding-cache key.

BLAKE2b is used rather than SHA-256: it is faster in CPython and lets us pick
the digest size directly, which keeps ids short.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import orjson

__all__ = ["cache_key", "chunk_id", "content_hash", "document_id", "entity_id", "stable_uuid"]

_NAMESPACE = uuid.UUID("6b2f4a5e-9c3d-4f1a-8b7e-2d5c9a1f3e40")


def content_hash(*parts: Any, size: int = 16) -> str:
    """Hex BLAKE2b digest over the parts. ``size`` is bytes, so 16 -> 32 chars."""
    h = hashlib.blake2b(digest_size=size)
    for part in parts:
        if part is None:
            continue
        if isinstance(part, bytes):
            h.update(part)
        elif isinstance(part, str):
            h.update(part.encode())
        else:
            h.update(orjson.dumps(part, option=orjson.OPT_SORT_KEYS))
        h.update(b"\x1f")  # unit separator: prevents ("ab","c") == ("a","bc")
    return h.hexdigest()


def stable_uuid(*parts: Any) -> str:
    """UUIDv5 over the parts.

    Qdrant point ids must be an unsigned int or a UUID, so every chunk id that
    reaches the vector store goes through here. Deterministic, so upserts
    replace rather than duplicate.
    """
    return str(uuid.uuid5(_NAMESPACE, content_hash(*parts, size=32)))


def document_id(source: str, content: str | None = None, tenant_id: str | None = None) -> str:
    return stable_uuid("doc", tenant_id or "", source, content or "")


def chunk_id(doc_id: str, index: int, content: str, level: int = 0) -> str:
    """Includes the content, so an edited chunk gets a new id and the stale
    vector is replaced on the next ingest rather than lingering."""
    return stable_uuid("chunk", doc_id, index, level, content)


def entity_id(name: str, tenant_id: str | None = None) -> str:
    return stable_uuid("entity", tenant_id or "", name.strip().casefold())


def cache_key(namespace: str, *parts: Any) -> str:
    return f"{namespace}:{content_hash(*parts, size=20)}"
