"""Cache tiers: in-process, Redis, semantic. See docs/adr/0007-cache-tiers.md."""

from ragorc.cache.memory import MemoryCache
from ragorc.cache.tiered import NullCache, TieredCache, build_cache

__all__ = ["MemoryCache", "NullCache", "TieredCache", "build_cache"]


def __getattr__(name: str):  # noqa: ANN202
    # RedisCache and SemanticCache pull optional dependencies (redis, a live
    # Qdrant). Importing them lazily keeps `import ragorc.cache` free of both.
    if name == "RedisCache":
        from ragorc.cache.redis_cache import RedisCache

        return RedisCache
    if name == "SemanticCache":
        from ragorc.cache.semantic import SemanticCache

        return SemanticCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
