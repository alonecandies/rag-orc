"""Audit trail.

Records the decisions that a security review will ask about: what was asked,
what SQL/Cypher was generated and whether it was blocked, what injection signals
fired, whether PII was redacted, and what the answer cost. Written as one JSON
object per line so it can be shipped to any log pipeline without a parser.

Prompts and answers are *not* recorded by default — an audit log that copies
customer data becomes the thing it was meant to protect against. Set
``observability.log_prompts`` deliberately if you need them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import structlog

from ragorc.core.models import utcnow
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger("ragorc.audit")

__all__ = ["MAX_LOGGED_CHARS", "AuditEvent", "AuditLog"]

MAX_LOGGED_CHARS = 4000
"""Cap on recorded question and answer text.

An audit line is meant to be greppable and shippable to a log pipeline; an
unbounded answer would put a whole context window on one line. Generous enough
that the truncation is rare and the text is still useful when it happens."""


@dataclass(slots=True)
class AuditEvent:
    action: str
    outcome: str = "allowed"
    tenant_id: str | None = None
    principal: str | None = None
    rule: str | None = None
    detail: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "ts": utcnow().isoformat(),
            "action": self.action,
            "outcome": self.outcome,
            "tenant_id": self.tenant_id,
            "principal": self.principal,
            "rule": self.rule,
            **(self.detail or {}),
        }


class AuditLog:
    """Append-only audit sink: structlog always, plus an optional file."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.enabled = self.settings.security.audit_log_enabled
        #: Whether question and answer text is recorded alongside the metadata.
        #:
        #: This module's docstring already said "Set ``observability.log_prompts``
        #: deliberately if you need them" and docs/security.md documents it as a
        #: control — and nothing read it. `model_post_init` even forces it off in
        #: prod, so it had a writer and no reader: an operator who turned it on
        #: for an incident got the same metadata-only lines and no way to tell.
        #:
        #: Decided here rather than at the eight call sites, because a predicate
        #: each caller spells out is the shape half this library's defects take.
        self.log_prompts = self.settings.observability.log_prompts
        self._path: Path | None = None
        if self.settings.security.audit_log_path:
            self._path = Path(self.settings.security.audit_log_path)
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: AuditEvent) -> None:
        if not self.enabled:
            return
        record = event.to_record()
        log.info("audit", **record)
        if self._path is not None:
            line = orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
            # O_APPEND makes concurrent writes from multiple workers atomic for
            # writes below PIPE_BUF, so lines never interleave.
            fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)

    # -- convenience recorders --------------------------------------------
    def query(self, *, tenant_id: str | None, principal: str | None, question: str) -> None:
        """Record that a question was asked, and its text only if asked to.

        Takes the question rather than its length so the *log* decides what to
        keep. A caller that passed `len(question)` had already made the decision,
        which is why `observability.log_prompts` had nowhere to be read.
        """
        detail: dict[str, Any] = {"query_length": len(question)}
        if self.log_prompts:
            detail["question"] = question[:MAX_LOGGED_CHARS]
        self.record(
            AuditEvent("query", tenant_id=tenant_id, principal=principal, detail=detail)
        )

    def deleted(self, *, tenant_id: str | None, documents: int) -> None:
        """Record a deletion request before it runs.

        Before, not after, and unconditionally. A delete is the one operation
        whose *attempt* has to be in the log even when it fails: a partial delete
        is the outcome an auditor most needs to see, and recording only successes
        would leave the interesting case invisible.
        """
        self.record(AuditEvent("delete", tenant_id=tenant_id, detail={"documents": documents}))

    def blocked(self, action: str, rule: str, **detail: Any) -> None:
        self.record(AuditEvent(action, outcome="blocked", rule=rule, detail=detail))

    def generated_query(
        self, target: str, statement: str, *, allowed: bool, rule: str | None = None
    ) -> None:
        self.record(
            AuditEvent(
                f"generated_{target}",
                outcome="allowed" if allowed else "blocked",
                rule=rule,
                # The statement itself is structure, not customer data, and is
                # the single most useful field when reviewing an incident.
                detail={"statement": statement[:1000]},
            )
        )

    def answered(
        self,
        *,
        tenant_id: str | None,
        cost_usd: float,
        chunks: int,
        grounded: bool,
        streamed: bool = False,
        answer: str = "",
    ) -> None:
        """Record that a question was answered, and what it cost.

        ``streamed`` marks the answers that carry no groundedness verdict because
        the path they took cannot produce one, so a reader of the log can tell
        "not grounded" from "never checked" — two very different facts that a bare
        ``grounded=False`` conflates.
        """
        self.record(
            AuditEvent(
                "answer",
                tenant_id=tenant_id,
                detail={
                    "cost_usd": round(cost_usd, 6),
                    "chunks": chunks,
                    "grounded": grounded,
                    "streamed": streamed,
                    **({"answer": answer[:MAX_LOGGED_CHARS]} if self.log_prompts and answer else {}),
                },
            )
        )
