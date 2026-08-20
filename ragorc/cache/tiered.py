"""Tiered cache: memory -> redis -> miss, with read-through promotion.

A hit in a lower tier is written back to the higher tiers, so a value fetched
once from Redis is served from memory for the rest of its TTL. Writes go to
every tier.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragorc.cache.memory import MemoryCache
from ragorc.core.protocols import Cache
from ragorc.core.settings import CacheSettings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["TieredCache", "build_cache"]


class TieredCache:
    """Ordered chain of caches, cheapest first."""

    def __init__(self, tiers: list[Cache]) -> None:
        if not tiers:
            raise ValueError("TieredCache needs at least one tier")
        self.tiers = tiers

    async def get(self, key: str) -> bytes | None:
        for depth, tier in enumerate(self.tiers):
            value = await tier.get(key)
            if value is not None:
                # Promote into every tier we missed on the way down.
                for higher in self.tiers[:depth]:
                    await higher.set(key, value)
                return value
        return None

    async def set(self, key: str, value: bytes, *, ttl: float | None = None) -> None:
        for tier in self.tiers:
            await tier.set(key, value, ttl=ttl)

    async def delete(self, key: str) -> None:
        for tier in self.tiers:
            await tier.delete(key)

    async def clear(self, prefix: str | None = None) -> int:
        return sum([await tier.clear(prefix) for tier in self.tiers])

    async def close(self) -> None:
        for tier in self.tiers:
            closer = getattr(tier, "close", None)
            if closer is not None:
                await closer()

    def stats(self) -> list[dict[str, Any]]:
        return [t.stats() for t in self.tiers if hasattr(t, "stats")]


class NullCache:
    """No-op cache, for benchmarks and tests that must not be memoized."""

    async def get(self, key: str) -> bytes | None:
        return None

    async def set(self, key: str, value: bytes, *, ttl: float | None = None) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def clear(self, prefix: str | None = None) -> int:
        return 0


def build_cache(settings: CacheSettings | None = None) -> Cache:
    """Assemble the cache chain from configuration."""
    settings = settings or get_settings().cache
    if not settings.enabled:
        return NullCache()
    tiers: list[Cache] = [
        MemoryCache(max_items=settings.memory_max_items, ttl_s=settings.memory_ttl_s)
    ]
    if settings.redis_url:
        try:
            from ragorc.cache.redis_cache import RedisCache

            tiers.append(
                RedisCache(
                    settings.redis_url, prefix=settings.redis_prefix, ttl_s=settings.redis_ttl_s
                )
            )
        except ImportError:
            log.warning("redis_tier_unavailable", hint="pip install 'ragorc[redis]'")
    return TieredCache(tiers)
