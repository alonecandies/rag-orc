"""Concurrency, resilience and backpressure primitives.

The single most important performance property of a RAG pipeline is *overlap*:
three stores, N query variants and M grader calls should all be in flight at
once. The single most important reliability property is that this fan-out is
**bounded** — an unbounded ``asyncio.gather`` over 50k chunks will exhaust
memory and get you rate-limited before it gets you an answer.

Everything here exists to make fan-out both wide and safe.
"""

from __future__ import annotations

import asyncio
import functools
import random
import time
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import structlog

from ragorc.core.errors import RateLimited, StoreUnavailable, TransientError

T = TypeVar("T")
R = TypeVar("R")

log = structlog.get_logger(__name__)

__all__ = [
    "CircuitBreaker",
    "RateLimiter",
    "bounded_gather",
    "gather_dict",
    "install_uvloop",
    "map_concurrent",
    "retry_async",
    "run_with_timeout",
    "safe_gather",
]


_uvloop_warned = False
"""Whether the "you are not on uvloop" advisory has already been logged.

Once per process, not once per call: a library that repeats the same advisory on
every pipeline construction is a library whose logs get filtered, and this one is
only actionable at the point where the caller starts its loop."""


def install_uvloop() -> bool:
    """Make uvloop the loop policy, and report whether that actually took effect.

    Returns ``True`` only when the caller really ends up on uvloop: either the
    running loop already is one, or no loop exists yet and the policy is now
    installed, so the next ``asyncio.run()`` builds a uvloop loop.

    The honesty matters because this call is a *policy* change, and a policy is
    only consulted when a loop is **created**. Called from inside a coroutine —
    which is where a library is almost always called from — it cannot upgrade the
    loop that is already running, so the documented 2-4x never materializes. That
    was the bug: the old version set the policy, returned ``True`` and let both the
    caller and ``docs/performance.md`` believe the throughput claim, while every
    user of the ``asyncio.run(main())`` quickstart stayed on the selector loop.

    So there are exactly two working recipes, and both start the loop themselves::

        uvloop.run(main())                 # or
        install_uvloop(); asyncio.run(main())

    With a loop already running the policy is deliberately *not* touched. It could
    not help this loop, and the only thing it would do is mutate process-global
    state on behalf of loops this call has nothing to do with — a library reaching
    into its host application to no effect. A one-time warning is more useful.
    """
    global _uvloop_warned
    try:
        import uvloop
    except ImportError:
        return False

    try:
        running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None:
        if type(running).__module__.split(".")[0] == "uvloop":
            return True
        if not _uvloop_warned:
            _uvloop_warned = True
            log.warning(
                "uvloop_not_in_use",
                loop=type(running).__qualname__,
                reason="a loop policy cannot replace a loop that is already running",
                hint="start the loop with uvloop.run(main()), or call "
                "install_uvloop() before asyncio.run()",
            )
        return False

    try:
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:  # pragma: no cover
        return False
    return True


async def bounded_gather(
    coros: Iterable[Coroutine[Any, Any, T]],
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list[T]:
    """``asyncio.gather`` with a concurrency ceiling, preserving input order.

    Coroutines are wrapped so the semaphore is acquired *inside* the task —
    acquiring before scheduling would serialize creation and defeat the point.

    **A failure cancels its siblings and waits for them.** Plain ``gather`` with
    ``return_exceptions=False`` re-raises the first exception and leaves the other
    tasks running, unawaited and unowned: the caller has already unwound while
    work it no longer knows about carries on. Where the siblings are writes to
    different stores that is not merely untidy, it is the wrong outcome — the
    ingest's two chunk writes go to Qdrant and Postgres, and the Postgres rows are
    the marker that says a document is ingested. A Qdrant failure that let the
    Postgres write commit anyway produced chunk rows with no vectors, which the
    next run then *skips* as already done: the corpus is missing, the ingest says
    it is fine, and no query can trace the gap back to here. Cancelling instead
    rolls that transaction back.

    The first exception is re-raised unchanged. ``TaskGroup`` would give the same
    cancellation and wrap it in an ``ExceptionGroup``, which every caller here
    would have to unwrap to find the store error it already handles by type.
    """
    coros = list(coros)
    if not coros:
        return []

    sem = asyncio.Semaphore(limit) if 0 < limit < len(coros) else None

    async def _run(coro: Coroutine[Any, Any, T]) -> T:
        if sem is None:
            return await coro
        try:
            await sem.acquire()
        except BaseException:
            # Cancelled while still queued behind the semaphore, so the wrapped
            # coroutine never started. Closing it is what keeps the cancellation
            # quiet: a coroutine that is never awaited warns when it is collected,
            # at a point in some later test with no connection to this one.
            coro.close()
            raise
        try:
            return await coro
        finally:
            sem.release()

    tasks = [asyncio.ensure_future(_run(c)) for c in coros]
    try:
        # `gather` is typed by a fixed set of positional overloads, up to five
        # awaitables; a list splat falls off the end of them and degrades to
        # `list[Any]`. The element type is `T` by construction — the ignore stands
        # in for an overload typeshed cannot write, not for an unchecked cast.
        return await asyncio.gather(*tasks, return_exceptions=return_exceptions)  # type: ignore[return-value]
    except BaseException:
        # Includes the cancellation of *this* coroutine: a caller giving up must
        # not leave the fan-out running either.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def safe_gather(
    coros: Iterable[Coroutine[Any, Any, T]],
    *,
    limit: int = 8,
    label: str = "task",
) -> tuple[list[T], list[BaseException]]:
    """Run everything; return successes and failures separately.

    This is the shape the multi-store retriever wants: one dead store must not
    take down a query that three other sources could have answered.
    """
    results = await bounded_gather(coros, limit=limit, return_exceptions=True)
    ok: list[T] = []
    errors: list[BaseException] = []
    for item in results:
        if isinstance(item, BaseException):
            errors.append(item)
            log.warning("task_failed", label=label, error=str(item), error_type=type(item).__name__)
        else:
            ok.append(item)
    return ok, errors


async def gather_dict(
    mapping: Mapping[str, Coroutine[Any, Any, Any]],
    *,
    limit: int = 8,
    return_exceptions: bool = True,
) -> dict[str, Any]:
    """Gather a name->coroutine mapping into a name->result mapping.

    Keeps per-store results labelled, which is what makes the retrieval
    diagnostics in :class:`RetrievalResult` possible.
    """
    if not mapping:
        return {}
    keys = list(mapping)
    values = await bounded_gather(
        (mapping[k] for k in keys), limit=limit, return_exceptions=return_exceptions
    )
    return dict(zip(keys, values, strict=True))


async def map_concurrent(
    fn: Callable[[T], Coroutine[Any, Any, R]],
    items: Sequence[T],
    *,
    limit: int = 8,
    return_exceptions: bool = False,
) -> list[R]:
    """Async ``map`` with bounded concurrency.

    ``fn`` must return a *coroutine* rather than any awaitable, because
    :func:`bounded_gather` only bounds work it gets to schedule itself: an
    already-running ``Task`` would sail past the semaphore.
    """
    return await bounded_gather(
        (fn(item) for item in items), limit=limit, return_exceptions=return_exceptions
    )


async def run_with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout: float | None,  # noqa: ASYNC109 - this IS the timeout wrapper
    *,
    default: T | None = None,
    label: str = "op",
) -> T | None:
    """Await with a deadline, returning ``default`` instead of raising.

    Used for optional enrichment (a slow reranker, a slow store) where a
    partial answer now beats a complete answer too late.
    """
    if timeout is None:
        return await coro
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError:
        log.warning("timeout", label=label, timeout_s=timeout)
        return default


def retry_async(
    *,
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    retry_on: tuple[type[BaseException], ...] = (TransientError,),
    jitter: bool = True,
):
    """Exponential backoff with full jitter.

    Jitter is not decoration: without it, a fleet of workers that all hit a
    429 retries in lockstep and reproduces the burst that caused it.

    ``RateLimited.retry_after`` is honoured when the provider supplies it —
    guessing when the server has told you is wasteful.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except retry_on as exc:
                    last = exc
                    if attempt >= max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if isinstance(exc, RateLimited) and exc.retry_after:
                        delay = max(delay, float(exc.retry_after))
                    if jitter:
                        delay = random.uniform(0, delay)  # noqa: S311 - not crypto
                    log.info(
                        "retrying",
                        fn=getattr(fn, "__qualname__", str(fn)),
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay_s=round(delay, 3),
                        error=str(exc)[:200],
                    )
                    await asyncio.sleep(delay)
            assert last is not None
            raise last

        return wrapper

    return decorator


@dataclass(slots=True)
class CircuitBreaker:
    """Trips after N consecutive failures and stops calling a dead dependency.

    Without one, every request pays the full timeout for a store that is down;
    with one, the first few requests pay it and the rest fail instantly and
    degrade. ``half_open`` lets a single probe through after the cooldown so
    recovery is automatic.
    """

    name: str
    failure_threshold: int = 5
    reset_timeout_s: float = 30.0
    _failures: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _half_open: bool = field(default=False, init=False)

    @property
    def is_open(self) -> bool:
        if self._failures < self.failure_threshold:
            return False
        if time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._half_open = True
            return False
        return True

    def record_success(self) -> None:
        if self._failures:
            log.info("circuit_closed", name=self.name, after_failures=self._failures)
        self._failures = 0
        self._half_open = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures == self.failure_threshold:
            self._opened_at = time.monotonic()
            log.warning("circuit_opened", name=self.name, failures=self._failures)
        elif self._half_open:
            self._opened_at = time.monotonic()
            self._half_open = False

    def check(self) -> None:
        """Raise if the circuit is open. Call before touching the dependency."""
        if self.is_open:
            remaining = self.reset_timeout_s - (time.monotonic() - self._opened_at)
            raise StoreUnavailable(
                self.name,
                f"circuit breaker open for {self.name}",
                retry_in_s=round(max(remaining, 0.0), 1),
            )

    async def call(self, coro: Coroutine[Any, Any, T]) -> T:
        self.check()
        try:
            result = await coro
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


@dataclass(slots=True)
class RateLimiter:
    """Token bucket over requests and (optionally) tokens per minute.

    Client-side limiting is strictly better than absorbing 429s: no wasted
    round trip, no retry storm, and the delay lands where it can be scheduled.
    """

    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    _req_allowance: float = field(default=0.0, init=False)
    _tok_allowance: float = field(default=0.0, init=False)
    _last: float = field(default_factory=time.monotonic, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        self._req_allowance = float(self.requests_per_minute or 0)
        self._tok_allowance = float(self.tokens_per_minute or 0)

    async def acquire(self, tokens: int = 0) -> None:
        if not self.requests_per_minute and not self.tokens_per_minute:
            return
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                if self.requests_per_minute:
                    self._req_allowance = min(
                        float(self.requests_per_minute),
                        self._req_allowance + elapsed * self.requests_per_minute / 60.0,
                    )
                if self.tokens_per_minute:
                    self._tok_allowance = min(
                        float(self.tokens_per_minute),
                        self._tok_allowance + elapsed * self.tokens_per_minute / 60.0,
                    )
                need_req = bool(self.requests_per_minute) and self._req_allowance < 1.0
                need_tok = bool(self.tokens_per_minute) and self._tok_allowance < tokens
                if not need_req and not need_tok:
                    if self.requests_per_minute:
                        self._req_allowance -= 1.0
                    if self.tokens_per_minute:
                        self._tok_allowance -= tokens
                    return
                waits = []
                if need_req and self.requests_per_minute:
                    waits.append((1.0 - self._req_allowance) * 60.0 / self.requests_per_minute)
                if need_tok and self.tokens_per_minute:
                    waits.append((tokens - self._tok_allowance) * 60.0 / self.tokens_per_minute)
                await asyncio.sleep(max(min(waits), 0.01))
