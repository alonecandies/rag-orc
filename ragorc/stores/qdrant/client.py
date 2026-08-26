"""Qdrant connection construction and reuse.

Why gRPC is the default transport
---------------------------------
A vector request *is* a float array and a vector response *is* a batch of float
arrays. Over REST both are JSON: a 1024-dim float32 vector serializes to ~20 KB
of ASCII that the server has to re-parse into 4 KB of floats, and a 256-point
upsert batch multiplies that by 256. Over gRPC the same payload is protobuf —
length-prefixed little-endian floats that land in memory with no numeric
parsing at all. That is the 2-3x figure in ``QdrantSettings.prefer_grpc``, and
it grows with dimensionality, batch size and (dramatically) with ColBERT
multivectors, where one point carries a ``(n_tokens, dim)`` matrix.

``prefer_grpc`` is a preference, not a switch: ``qdrant-client`` keeps the REST
client alive and uses it for the few endpoints gRPC does not cover, so nothing
is lost by leaving it on.

Why a module-level client cache
-------------------------------
``AsyncQdrantClient`` owns an HTTP/2 connection pool *and* a gRPC channel.
Building one per store, per request or per ingest task re-pays TCP (and TLS)
handshakes and leaks channels — a gRPC channel keeps sockets and background
threads alive until it is explicitly closed, so a hot loop that constructs
clients ends in file-descriptor exhaustion rather than in a slowdown. Caching
makes the second and later ``build_client`` calls a dict lookup.

The cache key includes the identity of the running event loop, which is the
non-obvious part. A ``grpc.aio`` channel binds to the loop that created it;
handing a client built on a finished loop to a new loop produces hangs that
look exactly like server timeouts. Per-test ``asyncio.run`` and per-invocation
CLI commands are precisely that situation, so they get separate entries and the
dead ones are evicted on lookup.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient

from ragorc.core.ids import content_hash
from ragorc.core.settings import QdrantSettings, Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = [
    "build_client",
    "close_all_clients",
    "qdrant_settings",
    "release_client",
]

# key -> (client, loop that owned it at construction time)
_CLIENTS: dict[tuple[Any, ...], tuple[AsyncQdrantClient, asyncio.AbstractEventLoop | None]] = {}


def qdrant_settings(settings: Settings | QdrantSettings | None = None) -> QdrantSettings:
    """Accept the root settings, the Qdrant section, or nothing.

    Every helper in this package takes ``settings`` in one of those three
    shapes; normalizing here keeps the ``settings.qdrant.qdrant`` mistake
    impossible.
    """
    if isinstance(settings, QdrantSettings):
        return settings
    return (settings or get_settings()).qdrant


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _is_dead(client: AsyncQdrantClient, loop: asyncio.AbstractEventLoop | None) -> bool:
    """True when the cached client can no longer serve requests.

    Two ways that happens: someone closed it, or the loop its gRPC channel is
    bound to has finished. ``_client.closed`` is the only accessor the library
    exposes for the former, hence the ``getattr`` chain.
    """
    if loop is not None and loop.is_closed():
        return True
    inner = getattr(client, "_client", None)
    return bool(getattr(inner, "closed", False))


def build_client(settings: Settings | QdrantSettings | None = None) -> AsyncQdrantClient:
    """Return a (cached) async Qdrant client for ``settings``.

    Call this from inside the running loop that will use the client — the cache
    key is loop-scoped, and a client built at import time lands in a separate
    bucket from one built inside ``asyncio.run``.
    """
    qs = qdrant_settings(settings)
    api_key = qs.api_key.get_secret_value() or None
    # The client wants whole seconds; anything under 1s is a misconfiguration
    # that would turn every request into a timeout.
    timeout = max(1, round(qs.timeout_s))
    loop = _running_loop()
    # The key hashes the credential instead of holding it: two deployments
    # against the same URL with different keys must not share a connection,
    # and a key must never be one repr() away from a log line.
    key = (
        id(loop),
        qs.url,
        qs.prefer_grpc,
        qs.grpc_port,
        timeout,
        content_hash(api_key or "", size=8),
    )

    # Sweep first. Eviction below only fires on a *lookup of the same key*, and
    # the key contains ``id(loop)`` — so a client whose loop has finished is
    # looked up under a different key forever and is never reached. The check the
    # docstring describes could not run for the case it names, and `_CLIENTS`
    # grew one entry per loop for the process lifetime. The map holds a handful
    # of entries, so scanning it is cheaper than the leak.
    for dead_key, (dead_client, dead_owner) in list(_CLIENTS.items()):
        if _is_dead(dead_client, dead_owner):
            del _CLIENTS[dead_key]
            log.info("qdrant_client_evicted", reason="closed_or_dead_loop")

    cached = _CLIENTS.get(key)
    if cached is not None:
        client, owner = cached
        if not _is_dead(client, owner):
            return client
        del _CLIENTS[key]
        log.info("qdrant_client_evicted", url=qs.url, reason="closed_or_dead_loop")

    client = AsyncQdrantClient(
        url=qs.url,
        api_key=api_key,
        prefer_grpc=qs.prefer_grpc,
        grpc_port=qs.grpc_port,
        timeout=timeout,
    )
    _CLIENTS[key] = (client, loop)
    log.info(
        "qdrant_client_built",
        url=qs.url,
        prefer_grpc=qs.prefer_grpc,
        grpc_port=qs.grpc_port,
        timeout_s=timeout,
        pooled=len(_CLIENTS),
    )
    return client


async def release_client(client: AsyncQdrantClient) -> None:
    """Evict ``client`` from the cache and close it.

    Eviction happens first: a closed client left in the cache would be handed
    out once more before ``_is_dead`` noticed, and that request would fail for
    no visible reason.
    """
    for key, (cached, _owner) in list(_CLIENTS.items()):
        if cached is client:
            del _CLIENTS[key]
    if _is_dead(client, None):
        return
    await client.close()
    log.info("qdrant_client_closed", pooled=len(_CLIENTS))


async def close_all_clients() -> None:
    """Close every cached client. For process shutdown and test teardown."""
    entries = list(_CLIENTS.values())
    _CLIENTS.clear()
    for client, _owner in entries:
        if _is_dead(client, None):
            continue
        try:
            await client.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.warning("qdrant_client_close_failed", error=str(exc))
    log.info("qdrant_clients_closed", count=len(entries))
