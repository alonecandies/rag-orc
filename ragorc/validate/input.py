"""Inbound validation for queries.

Runs before anything expensive happens. Ordering matters: reject cheaply first
(length, encoding) and only then spend a regex sweep or an LLM call. A query that
fails here has cost nothing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import structlog

from ragorc.core.errors import ValidationFailed
from ragorc.core.models import Query
from ragorc.core.settings import Settings, get_settings
from ragorc.security.injection import InjectionScanner
from ragorc.security.pii import PIIRedactor

log = structlog.get_logger(__name__)

__all__ = ["QueryValidator", "ValidatedQuery"]

_WHITESPACE = re.compile(r"\s+")
_REPEATED = re.compile(r"(.{3,}?)\1{8,}")


@dataclass(slots=True)
class ValidatedQuery:
    query: Query
    warnings: list[str] = field(default_factory=list)
    injection_risk: float = 0.0
    pii_entities: tuple[str, ...] = ()


class QueryValidator:
    """Normalizes and screens an incoming question."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.scanner = InjectionScanner(self.settings.security)
        self.redactor = PIIRedactor(self.settings.security)

    def validate(
        self, text: str, *, tenant_id: str | None = None, top_k: int | None = None
    ) -> ValidatedQuery:
        sec = self.settings.security
        warnings: list[str] = []

        if not isinstance(text, str):
            raise ValidationFailed("query must be a string", got=type(text).__name__)

        # NFC (not NFKC) for the *query*: we want canonical equivalence without
        # destroying meaningful formatting the user typed. The injection scanner
        # applies NFKC separately on its own copy.
        normalized = unicodedata.normalize("NFC", text)
        normalized = normalized.replace("\x00", "")
        collapsed = _WHITESPACE.sub(" ", normalized).strip()

        if len(collapsed) < sec.min_query_length:
            raise ValidationFailed(
                "query is too short", length=len(collapsed), minimum=sec.min_query_length
            )
        if len(collapsed) > sec.max_query_length:
            # Truncate rather than reject: an over-long query is usually a paste
            # accident, and the useful part is at the start.
            collapsed = collapsed[: sec.max_query_length]
            warnings.append(f"query truncated to {sec.max_query_length} characters")

        if _REPEATED.search(collapsed):
            warnings.append("query contains heavy repetition; possible token-flooding attempt")

        scan = self.scanner.scan_query(collapsed)
        text_out = scan.clean_text

        pii_entities: tuple[str, ...] = ()
        if sec.enable_pii_redaction:
            result = self.redactor.redact(text_out)
            if result.found:
                pii_entities = result.entities
                text_out = result.text
                warnings.append(f"PII redacted from query: {', '.join(pii_entities)}")

        query = Query(
            text=text_out,
            original=text,
            tenant_id=tenant_id,
            top_k=top_k or self.settings.retrieval.top_k,
        )
        return ValidatedQuery(
            query=query,
            warnings=warnings,
            injection_risk=scan.risk,
            pii_entities=pii_entities,
        )
