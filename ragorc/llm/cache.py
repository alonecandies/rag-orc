"""LLM response cache.

Two layers with different hit characteristics:

* **Exact** — hash of (model, system, prompt, temperature, schema). Cheap and
  safe. Hits whenever the same prompt recurs, which in RAG is constantly: the
  same document gets graded for the same query, the same schema summary gets
  embedded in every Text-to-SQL prompt, the same chunk gets summarized on every
  re-ingest.
* **Semantic** — nearest-neighbour lookup over question embeddings, so "what is
  X" can serve the cached answer for "explain X". This is the largest single
  cost lever in a production RAG service, and also the most dangerous: too low
  a threshold and you answer a question nobody asked. Default 0.97.

Only deterministic calls are cached. ``temperature > 0`` bypasses the cache,
because caching a sample defeats the purpose of sampling.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragorc.core.ids import cache_key
from ragorc.core.protocols import Cache
from ragorc.core.settings import CacheSettings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["LLMCache"]


class LLMCache:
    """Adapter between :class:`OpenRouterLLM` and a :class:`Cache` backend."""

    def __init__(self, backend: Cache, settings: CacheSettings | None = None) -> None:
        self.backend = backend
        self.settings = settings or get_settings().cache
        self.hits = 0
        self.misses = 0

    def _key(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str,
        temperature: float,
        extra: Any = None,
    ) -> str:
        return cache_key("llm", model, temperature, system or "", prompt, extra)

    async def get_completion(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str,
        temperature: float,
        extra: Any = None,
    ) -> str | None:
        if not (self.settings.enabled and self.settings.cache_llm):
            return None
        if temperature and temperature > 0.0:
            return None  # sampling must not be memoized
        raw = await self.backend.get(
            self._key(
                prompt=prompt, system=system, model=model, temperature=temperature, extra=extra
            )
        )
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return raw.decode()

    async def set_completion(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str,
        temperature: float,
        value: str,
        extra: Any = None,
        ttl: float | None = None,
    ) -> None:
        if not (self.settings.enabled and self.settings.cache_llm):
            return
        if temperature and temperature > 0.0:
            return
        await self.backend.set(
            self._key(
                prompt=prompt, system=system, model=model, temperature=temperature, extra=extra
            ),
            value.encode(),
            ttl=ttl,
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 3)}
