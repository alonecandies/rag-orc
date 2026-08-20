"""Per-caller rate limiting for the HTTP surface.

The library-internal limiter in :mod:`ragorc.core.concurrency` protects the
*provider* from us. This one protects *us* from a caller: a single client
looping on an expensive pipeline can exhaust the cost budget for everyone.

A sliding-window counter is used rather than a fixed window, because a fixed
window allows a 2x burst across the boundary — 60 requests at 11:59:59 plus 60
at 12:00:00. Buckets are pruned lazily so memory stays proportional to *active*
callers, not to every caller ever seen.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

from ragorc.core.errors import RateLimited
from ragorc.core.settings import SecuritySettings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["KeyedRateLimiter"]


@dataclass(slots=True)
class KeyedRateLimiter:
    """Sliding-window limiter keyed by API key, tenant or IP."""

    per_minute: int = 60
    burst: int = 20
    window_s: float = 60.0
    _hits: dict[str, deque[float]] = field(default_factory=dict, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _last_prune: float = field(default=0.0, init=False)

    @classmethod
    def from_settings(cls, settings: SecuritySettings | None = None) -> KeyedRateLimiter:
        s = settings or get_settings().security
        return cls(per_minute=s.rate_limit_per_minute, burst=s.rate_limit_burst)

    async def check(self, key: str) -> None:
        """Raise :class:`RateLimited` when ``key`` is over budget."""
        now = time.monotonic()
        async with self._lock:
            window = self._hits.setdefault(key, deque())
            cutoff = now - self.window_s
            while window and window[0] < cutoff:
                window.popleft()

            # Burst check: allow a short spike, but not a sustained one.
            recent = sum(1 for t in window if t > now - 1.0)
            if recent >= self.burst:
                raise RateLimited("burst limit exceeded", retry_after=1.0, key_hint=key[:8])
            if len(window) >= self.per_minute:
                retry_after = max(window[0] + self.window_s - now, 0.1)
                raise RateLimited(
                    "rate limit exceeded", retry_after=round(retry_after, 2), key_hint=key[:8]
                )
            window.append(now)

            if now - self._last_prune > self.window_s:
                self._last_prune = now
                for k in [k for k, w in self._hits.items() if not w or w[-1] < cutoff]:
                    self._hits.pop(k, None)

    def remaining(self, key: str) -> int:
        window = self._hits.get(key)
        if not window:
            return self.per_minute
        cutoff = time.monotonic() - self.window_s
        return max(self.per_minute - sum(1 for t in window if t >= cutoff), 0)
