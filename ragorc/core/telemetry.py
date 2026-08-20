"""Logging, tracing, timing and cost accounting.

Three things are measured on every request, because in RAG they are the only
three that matter operationally:

* **latency, per stage** — "the query took 4s" is useless; "reranking took 3.6s
  of the 4s" is actionable.
* **cost, per call** — a pipeline with 20 LLM calls needs the bill itemized.
* **cache hits** — the difference between a viable and an unviable unit cost.

The tracer is a plain context-local list, not an OTel dependency. OTel export
is optional and additive (``ragorc[otel]``).
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.models import StepTrace, Usage

__all__ = [
    "CostLedger",
    "Timer",
    "configure_logging",
    "current_ledger",
    "current_trace",
    "get_logger",
    "new_request_context",
    "timed",
    "trace_step",
]

_trace_var: contextvars.ContextVar[list[StepTrace] | None] = contextvars.ContextVar(
    "ragorc_trace", default=None
)
_ledger_var: contextvars.ContextVar[CostLedger | None] = contextvars.ContextVar(
    "ragorc_ledger", default=None
)

_configured = False


def configure_logging(level: str = "INFO", json_logs: bool = True, redact: bool = True) -> None:
    """Configure structlog once. Idempotent — safe to call from library code."""
    global _configured
    if _configured:
        return
    _configured = True

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if redact:
        processors.append(_redact_secrets)
    processors.append(structlog.processors.format_exc_info)
    processors.append(
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_SECRET_HINTS = ("api_key", "apikey", "password", "token", "secret", "authorization", "dsn")


def _redact_secrets(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Never let a credential reach a log sink. Cheap insurance."""
    for key in list(event_dict):
        lowered = key.lower()
        if any(hint in lowered for hint in _SECRET_HINTS):
            value = event_dict[key]
            if isinstance(value, str) and value:
                event_dict[key] = f"{value[:4]}***" if len(value) > 8 else "***"
    return event_dict


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Timer:
    """``time.perf_counter`` based stopwatch. Monotonic, nanosecond resolution."""

    label: str = ""
    start: float = field(default_factory=time.perf_counter)
    elapsed_ms: float = 0.0

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0
        return self.elapsed_ms

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


@contextlib.contextmanager
def timed(label: str, **detail: Any) -> Iterator[Timer]:
    """Time a block and append it to the request trace automatically."""
    timer = Timer(label)
    try:
        yield timer
    finally:
        timer.stop()
        trace = _trace_var.get()
        if trace is not None:
            trace.append(StepTrace(name=label, duration_ms=timer.elapsed_ms, detail=dict(detail)))


def trace_step(
    name: str, duration_ms: float = 0.0, usage: Usage | None = None, **detail: Any
) -> None:
    trace = _trace_var.get()
    if trace is not None:
        trace.append(
            StepTrace(name=name, duration_ms=duration_ms, detail=dict(detail), usage=usage)
        )


def current_trace() -> list[StepTrace]:
    trace = _trace_var.get()
    return trace if trace is not None else []


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CostLedger:
    """Per-request cost and call accounting with hard ceilings.

    The ceilings are the important part. A pipeline with retries, loops and
    fan-out has no natural upper bound on spend; a runaway CRAG loop on a
    frontier model can cost dollars per query. ``check()`` is called before
    every LLM request so the ceiling is enforced *before* the money is spent.
    """

    max_cost_usd: float | None = None
    max_calls: int | None = None
    max_tokens: int | None = None
    by_model: dict[str, Usage] = field(default_factory=dict)
    by_stage: dict[str, Usage] = field(default_factory=dict)

    def record(self, usage: Usage, *, stage: str = "unknown") -> None:
        model = usage.model or "unknown"
        self.by_model[model] = self.by_model.get(model, Usage()) + usage
        self.by_stage[stage] = self.by_stage.get(stage, Usage()) + usage

    @property
    def total(self) -> Usage:
        return Usage.sum(self.by_model.values())

    def check(self, *, projected_cost: float = 0.0) -> None:
        from ragorc.core.errors import BudgetExceeded

        total = self.total
        if self.max_calls is not None and total.calls >= self.max_calls:
            raise BudgetExceeded(
                "LLM call budget exhausted", calls=total.calls, limit=self.max_calls
            )
        if self.max_cost_usd is not None and total.cost_usd + projected_cost > self.max_cost_usd:
            raise BudgetExceeded(
                "cost budget exhausted",
                spent_usd=round(total.cost_usd, 6),
                limit_usd=self.max_cost_usd,
            )
        if self.max_tokens is not None and total.total_tokens >= self.max_tokens:
            raise BudgetExceeded(
                "token budget exhausted", tokens=total.total_tokens, limit=self.max_tokens
            )

    def report(self) -> dict[str, Any]:
        total = self.total
        return {
            "total_cost_usd": round(total.cost_usd, 6),
            "total_tokens": total.total_tokens,
            "prompt_tokens": total.prompt_tokens,
            "completion_tokens": total.completion_tokens,
            "calls": total.calls,
            "cached_calls": total.cached,
            "cache_hit_rate": round(total.cached / total.calls, 3) if total.calls else 0.0,
            "by_model": {
                m: {"calls": u.calls, "cost_usd": round(u.cost_usd, 6), "tokens": u.total_tokens}
                for m, u in self.by_model.items()
            },
            "by_stage": {
                s: {"calls": u.calls, "cost_usd": round(u.cost_usd, 6), "tokens": u.total_tokens}
                for s, u in self.by_stage.items()
            },
        }


def current_ledger() -> CostLedger | None:
    return _ledger_var.get()


@contextlib.contextmanager
def new_request_context(
    *,
    request_id: str | None = None,
    max_cost_usd: float | None = None,
    max_calls: int | None = None,
    max_tokens: int | None = None,
) -> Iterator[tuple[list[StepTrace], CostLedger]]:
    """Install a fresh trace and ledger for one request.

    Uses contextvars, so concurrent requests in the same event loop keep
    separate traces and budgets without passing either through every signature.
    """
    trace: list[StepTrace] = []
    ledger = CostLedger(max_cost_usd=max_cost_usd, max_calls=max_calls, max_tokens=max_tokens)
    trace_token = _trace_var.set(trace)
    ledger_token = _ledger_var.set(ledger)
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        yield trace, ledger
    finally:
        _trace_var.reset(trace_token)
        _ledger_var.reset(ledger_token)
        if request_id:
            structlog.contextvars.unbind_contextvars("request_id")
