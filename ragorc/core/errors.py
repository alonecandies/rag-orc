"""Exception hierarchy.

Every error carries enough structure for the pipeline to *decide* rather than
just log: :class:`TransientError` is retryable, :class:`StoreUnavailable` trips
a circuit breaker and degrades the query, :class:`GuardrailViolation` is never
retried because retrying a blocked SQL statement is still blocked.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BudgetExceeded",
    "ConfigError",
    "ConstructionError",
    "EmbeddingError",
    "GuardrailViolation",
    "LLMError",
    "RagOrcError",
    "RateLimited",
    "RetrievalError",
    "StoreUnavailable",
    "StructuredOutputError",
    "TransientError",
    "ValidationFailed",
]


class RagOrcError(Exception):
    """Base class. ``detail`` is attached to API responses and traces."""

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        if not self.detail:
            return self.message
        extras = " ".join(f"{k}={v!r}" for k, v in self.detail.items())
        return f"{self.message} ({extras})"


class ConfigError(RagOrcError):
    """Bad or missing configuration. Raised at startup, never mid-request."""


class TransientError(RagOrcError):
    """Retryable. The retry decorator keys off this type."""


class RateLimited(TransientError):
    def __init__(self, message: str, retry_after: float | None = None, **detail: Any) -> None:
        super().__init__(message, retry_after=retry_after, **detail)
        self.retry_after = retry_after


class LLMError(RagOrcError):
    """Non-retryable model failure (bad request, content filter, no credits)."""


class StructuredOutputError(LLMError):
    """The model returned JSON that does not satisfy the schema, and the
    repair attempt also failed."""


class EmbeddingError(RagOrcError):
    pass


class StoreUnavailable(TransientError):
    """A datastore is unreachable or the circuit breaker is open. The
    multi-store retriever catches this and continues with the others."""

    def __init__(self, store: str, message: str = "", **detail: Any) -> None:
        super().__init__(message or f"{store} unavailable", store=store, **detail)
        self.store = store


class RetrievalError(RagOrcError):
    pass


class ConstructionError(RagOrcError):
    """Text-to-SQL / Text-to-Cypher / self-query produced nothing usable."""


class GuardrailViolation(RagOrcError):
    """Security guard rejected an input or a generated query. Never retried."""

    def __init__(
        self, message: str, *, rule: str = "", severity: str = "high", **detail: Any
    ) -> None:
        super().__init__(message, rule=rule, severity=severity, **detail)
        self.rule = rule
        self.severity = severity


class ValidationFailed(RagOrcError):
    pass


class BudgetExceeded(RagOrcError):
    """Token, cost or latency budget for the request is spent."""
