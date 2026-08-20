"""In-process cache tier — TTL + LRU, no network hop.

First tier because it is ~200ns per lookup against ~200us for Redis. On a
pipeline that makes 40 keyed lookups per request, that difference is the whole
latency budget of a stage.

``cachetools.TTLCache`` is used rather than a hand-rolled dict: eviction is
O(1) amortized and the expiry bookkeeping is already correct.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cachetools import TTLCache

__all__ = ["MemoryCache"]


class MemoryCache:
    """Bounded, TTL'd, async-safe in-process cache."""

    def __init__(self, max_items: int = 20_000, ttl_s: float = 900.0) -> None:
        self._cache: TTLCache[str, bytes] = TTLCache(maxsize=max_items, ttl=ttl_s)
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> bytes | None:
        async with self._lock:
            value = self._cache.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, key: str, value: bytes, *, ttl: float | None = None) -> None:
        # cachetools.TTLCache has a single global TTL; a per-key TTL shorter
        # than it is honoured by the caller storing an expiry alongside the
        # value where that matters. Here we accept the global TTL.
        async with self._lock:
            self._cache[key] = value

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self, prefix: str | None = None) -> int:
        async with self._lock:
            if prefix is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            keys = [k for k in self._cache if k.startswith(prefix)]
            for k in keys:
                self._cache.pop(k, None)
            return len(keys)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "tier": "memory",
            "size": len(self._cache),
            "max": self._cache.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }
