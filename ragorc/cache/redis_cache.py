"""Shared cache tier.

Needed as soon as more than one process serves traffic: without it each worker
warms its own memory cache and the effective hit rate divides by the worker
count. ``hiredis`` is used for parsing (C, several times faster than the pure
Python parser).

Optional dependency: ``pip install 'ragorc[redis]'``.
"""

from __future__ import annotations

from typing import Any, cast

import structlog

log = structlog.get_logger(__name__)

__all__ = ["RedisCache"]


class RedisCache:
    """Async Redis tier. Failures degrade to a miss — a cache outage must not
    become a service outage."""

    def __init__(self, url: str, *, prefix: str = "ragorc", ttl_s: float = 86_400.0) -> None:
        try:
            from redis.asyncio import Redis
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "RedisCache requires the redis extra: pip install 'ragorc[redis]'"
            ) from exc
        self._redis = Redis.from_url(url, decode_responses=False)
        self.prefix = prefix
        self.ttl_s = ttl_s
        self.hits = 0
        self.misses = 0
        self.errors = 0

    def _k(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    async def get(self, key: str) -> bytes | None:
        try:
            value = await self._redis.get(self._k(key))
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            self.errors += 1
            log.warning("redis_get_failed", error=str(exc)[:200])
            return None
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        # redis-py types ``get`` as ``bytes | str | None`` because the same method
        # serves both decode modes. The client above is built with
        # ``decode_responses=False``, so the ``str`` arm is unreachable here.
        return cast("bytes | None", value)

    async def set(self, key: str, value: bytes, *, ttl: float | None = None) -> None:
        try:
            await self._redis.set(self._k(key), value, ex=int(ttl or self.ttl_s))
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            log.warning("redis_set_failed", error=str(exc)[:200])

    async def delete(self, key: str) -> None:
        try:
            await self._redis.delete(self._k(key))
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_delete_failed", error=str(exc)[:200])

    async def clear(self, prefix: str | None = None) -> int:
        pattern = f"{self.prefix}:{prefix or ''}*"
        removed = 0
        try:
            # SCAN, never KEYS: KEYS blocks the server for the whole keyspace.
            async for key in self._redis.scan_iter(match=pattern, count=500):
                await self._redis.delete(key)
                removed += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("redis_clear_failed", error=str(exc)[:200])
        return removed

    async def close(self) -> None:
        await self._redis.aclose()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "tier": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
