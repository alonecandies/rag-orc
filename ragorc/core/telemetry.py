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
import re
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
        # A factory, not `PrintLoggerFactory(file=sys.stderr)`. That binds the
        # stream object at configure time, and anything that later swaps
        # `sys.stderr` — pytest's capture, typer's CliRunner, a daemonizer — leaves
        # every subsequent log line writing into a closed handle and raising
        # `ValueError: I/O operation on closed file` from inside the logger. Looking
        # `sys.stderr` up per logger keeps logging pointed at whatever stderr is
        # now, which is the only answer that stays true.
        logger_factory=lambda *args: structlog.PrintLogger(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


_SECRET_HINTS = ("api_key", "apikey", "password", "token", "secret", "authorization", "dsn")


#: Identifiers that leak through a value rather than through a key name. Key
#: filtering cannot catch these: they arrive inside a provider's prose error body
#: under an innocuous key like ``body`` or ``message``, and that body is echoed
#: into an abstention reason, an HTTP error detail and the CLI. What they expose
#: is the *operator's* account — a key-management URL carrying the key id, an
#: account user id — to whoever called the API.
_IDENTIFIER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bsk-(?:or-v\d+-)?[A-Za-z0-9]{16,}"), "sk-***"),
    (re.compile(r"\b(?:gh[pousr]_|xox[baprs]-)[A-Za-z0-9-]{10,}"), "***"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{10,}", re.IGNORECASE), "Bearer ***"),
    (re.compile(r"https?://[^\s\"'<>]*/keys/[A-Za-z0-9._-]{8,}"), "<key-url-redacted>"),
    (re.compile(r"(user_id\\?\"?\s*[:=]\s*\\?\"?)[A-Za-z0-9_-]{6,}"), r"\1<redacted>"),
    # A credential inside a URL's authority. `_SECRET_HINTS` masks a value whose
    # *key* is dsn/password/token, which covers `log.info("x", dsn=...)` and not
    # `log.warning("connect_failed", target=...)` or an exception string that
    # happens to quote the URL. Measured: neither psycopg nor the neo4j driver
    # puts the password in a connection error, so this closes a gap rather than a
    # demonstrated leak — but there are 104 `error=str(exc)` sites and the cost of
    # one more pattern is a substring check that almost always misses.
    (re.compile(r"\b([a-z][a-z0-9+.-]*://[^\s:/@]*):[^\s/@]+@"), r"\1:***@"),
)

#: Cheap pre-filter. Almost every string that reaches the redactor contains none
#: of these, and a substring scan is an order of magnitude cheaper than five
#: regex passes — which matters because this runs on log lines.
_IDENTIFIER_MARKERS = ("sk-", "/keys/", "user_id", "earer ", "ghp_", "gho_", "xox", "://")


def redact_identifiers(text: str) -> str:
    """Blank operator-identifying values inside free text.

    Complements the key-name filtering below rather than replacing it: that
    catches ``{"api_key": ...}``, this catches the same secret quoted inside
    ``{"body": "...visit https://.../keys/abc123 ..."}``, which is how provider
    errors actually deliver it.
    """
    if not text or not any(marker in text for marker in _IDENTIFIER_MARKERS):
        return text
    for pattern, replacement in _IDENTIFIER_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _redact_secrets(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Never let a credential reach a log sink. Cheap insurance."""
    for key in list(event_dict):
        value = event_dict[key]
        if any(hint in key.lower() for hint in _SECRET_HINTS):
            if isinstance(value, str) and value:
                event_dict[key] = f"{value[:4]}***" if len(value) > 8 else "***"
        elif isinstance(value, str):
            event_dict[key] = redact_identifiers(value)
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
    frontier model can cost dollars per query.

    Why checking is not enough, and reservations exist
    -------------------------------------------------
    ``check()`` reads what has been *recorded*, and a call is recorded only after
    its round trip. Under a fan-out, every coroutine runs to its first ``await``
    before any of them completes, so N pre-checks all read a ledger that still
    says zero and all pass. Measured with 40 prompts against a five-call ceiling
    and a transport that yields, as any real provider does: 40 requests served.
    The ceiling was read forty times and enforced none.

    :meth:`reserve` closes that window by claiming the budget *synchronously* —
    ``check`` and the increment happen with no ``await`` between them, which on a
    single-threaded event loop is atomic — and releasing it when the call
    finishes, however it finishes. So the bound becomes the number of calls
    permitted rather than the width of the fan-out.

    An earlier measurement of this suggested the ceiling held; that was an
    artifact of a mock transport that returned without yielding, which serialized
    the very interleaving the bug needs.
    """

    max_cost_usd: float | None = None
    max_calls: int | None = None
    max_tokens: int | None = None
    by_model: dict[str, Usage] = field(default_factory=dict)
    by_stage: dict[str, Usage] = field(default_factory=dict)
    _reserved_calls: int = 0
    _reserved_cost: float = 0.0
    _reserved_tokens: int = 0

    def record(self, usage: Usage, *, stage: str = "unknown") -> None:
        model = usage.model or "unknown"
        self.by_model[model] = self.by_model.get(model, Usage()) + usage
        self.by_stage[stage] = self.by_stage.get(stage, Usage()) + usage

    @property
    def total(self) -> Usage:
        return Usage.sum(self.by_model.values())

    def check(self, *, projected_cost: float = 0.0, projected_tokens: int = 0) -> None:
        """Raise if this request cannot proceed. Counts in-flight reservations.

        Spend already claimed by :meth:`reserve` is counted as spent, because it
        is about to be: a call that has passed the gate and not yet returned is
        money committed, and treating it as zero is what let a fan-out through.
        """
        from ragorc.core.errors import BudgetExceeded

        total = self.total
        calls = total.calls + self._reserved_calls
        if self.max_calls is not None and calls >= self.max_calls:
            raise BudgetExceeded("LLM call budget exhausted", calls=calls, limit=self.max_calls)
        cost = total.cost_usd + self._reserved_cost + projected_cost
        if self.max_cost_usd is not None and cost > self.max_cost_usd:
            raise BudgetExceeded(
                "cost budget exhausted",
                spent_usd=round(total.cost_usd, 6),
                committed_usd=round(cost, 6),
                limit_usd=self.max_cost_usd,
            )
        tokens = total.total_tokens + self._reserved_tokens + projected_tokens
        if self.max_tokens is not None and tokens >= self.max_tokens:
            raise BudgetExceeded("token budget exhausted", tokens=tokens, limit=self.max_tokens)

    @contextlib.contextmanager
    def reserve(self, *, cost: float = 0.0, tokens: int = 0) -> Iterator[None]:
        """Claim one call's budget for the duration of that call.

        The check and the claim are adjacent with no ``await`` between them, which
        is what makes this atomic on an event loop and what a bare ``check()``
        could never be. Released in a ``finally``, so a failed or cancelled call
        gives its budget back instead of wedging the ceiling for the rest of the
        request.

        ``tokens`` is the caller's ``max_tokens`` — the most the call can spend,
        known before it is made. ``cost`` is an estimate when one is available;
        with a call ceiling configured the overshoot is bounded by that ceiling
        regardless, which is the common case.
        """
        self.check(projected_cost=cost, projected_tokens=tokens)
        self._reserved_calls += 1
        self._reserved_cost += cost
        self._reserved_tokens += tokens
        try:
            yield
        finally:
            self._reserved_calls -= 1
            self._reserved_cost -= cost
            self._reserved_tokens -= tokens

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
    trace: bool = True,
) -> Iterator[tuple[list[StepTrace], CostLedger]]:
    """Install a fresh trace and ledger for one request.

    Uses contextvars, so concurrent requests in the same event loop keep
    separate traces and budgets without passing either through every signature.

    ``trace=False`` collects no steps. That is a privacy control rather than a
    performance one: a step trace records what each stage did with the retrieved
    passages, it is attached to every ``Answer``, and ``observability.log_prompts``
    is off by default for exactly the same reason. The switch meant to turn it off
    — ``observability.trace_enabled`` — was read by nothing, so there was no way
    to decline.
    """
    steps: list[StepTrace] = []
    trace_list: list[StepTrace] | None = steps if trace else None
    ledger = CostLedger(max_cost_usd=max_cost_usd, max_calls=max_calls, max_tokens=max_tokens)
    trace_token = _trace_var.set(trace_list)
    ledger_token = _ledger_var.set(ledger)
    if request_id:
        structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        yield steps, ledger
    finally:
        _trace_var.reset(trace_token)
        _ledger_var.reset(ledger_token)
        if request_id:
            structlog.contextvars.unbind_contextvars("request_id")
