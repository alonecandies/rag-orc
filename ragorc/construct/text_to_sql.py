"""Text-to-SQL: natural language into a guarded, executed, chunk-shaped result.

Why a relational leg at all
---------------------------
A vector index cannot count. "How many enterprise accounts churned last
quarter", "top 10 customers by revenue", "orders placed between two dates" have
answers that are not written down in any document — they are *computed*. No
amount of embedding quality retrieves a number that does not exist yet, which is
the entire justification for routing some questions to Postgres instead of
Qdrant. And that means an LLM has to write SQL.

Which makes this the most dangerous module in the library: it turns untrusted
natural language into a statement we execute against a database. Three
structural decisions follow.

1. **Construction and execution are separate methods.** ``construct`` returns a
   :class:`~ragorc.security.sql_guard.SQLValidation`, not a row set. Nothing can
   reach the database without having passed through the guard first, because the
   only object that carries executable SQL is the guard's own output.
2. **The guard is not reimplemented here.** ``SQLGuard`` validates on a parsed
   AST, clamps or injects ``LIMIT``, and enforces the table allowlist. This
   module's job is to feed it and to react to its verdict.
3. **Exactly one repair attempt.** See below.

Why one retry and not zero, and not three
-----------------------------------------
Guard rejections fall into two populations. The first is a *schema
misunderstanding*: a column that does not exist, a join the model invented, a
missing ``LIMIT``, one join too many. The violation message names the problem
precisely, and a model that sees it fixes the query on the next try — this is
the majority case, so zero retries throws away cheap recall.

The second is a *blocked pattern*: the model wants ``pg_catalog``, a
``COPY``, a filesystem function. No amount of feedback changes that, because the
guard is not confused — it is refusing. Every further attempt re-sends the whole
schema (the largest part of the prompt) to be told the same thing again. So the
retry budget is one, and the failure after it is a
:class:`~ragorc.core.errors.ConstructionError` rather than a loop.

The repair reuses the *same* prompt rather than a bespoke one. That keeps the
hard rules and the DDL in the identical system block, so the provider's prompt
cache still hits on the second call; only the question slot carries the
rejection feedback.

Why rows become chunks
----------------------
``to_chunks`` exists so a SQL result enters the same fusion, reranking, packing,
citation and groundedness machinery as a vector hit. Without it, structured
answers would bypass every correctness check the generation side applies to
retrieved text — the one class of evidence most likely to be quoted verbatim
would be the one class nobody verified.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import orjson
import structlog

from ragorc.core.errors import ConstructionError, GuardrailViolation
from ragorc.core.ids import chunk_id, content_hash
from ragorc.core.models import (
    Chunk,
    Modality,
    Query,
    RetrievalSource,
    ScoredChunk,
    Usage,
)
from ragorc.core.protocols import LLM, RelationalStore
from ragorc.core.registry import register
from ragorc.core.schemas import SQLQuery
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.security.audit import AuditLog
from ragorc.security.sql_guard import SQLGuard, SQLValidation
from ragorc.security.tenancy import require_generated_query_isolation

log = structlog.get_logger(__name__)

__all__ = ["TextToSQLConstructor"]

#: Per-cell rendering ceiling. One runaway ``text`` column would otherwise eat
#: the whole context budget, and the context packer cannot trim *inside* a cell —
#: it can only drop the chunk entirely, losing the other columns with it.
_MAX_CELL_CHARS = 240

#: Rows are exact matches, not similarities, so there is no score gradient to
#: report. A flat 1.0 makes rank-based fusion (RRF) order them by row position,
#: which is the only ordering the query itself expressed (``ORDER BY``).
_EXACT_MATCH_SCORE = 1.0


@register("constructor", "text_to_sql")
class TextToSQLConstructor:
    """Builds, validates and (optionally) runs read-only SQL for a question."""

    name = "text_to_sql"
    target = "sql"

    def __init__(
        self,
        llm: LLM,
        store: RelationalStore | None = None,
        *,
        guard: SQLGuard | None = None,
        audit: AuditLog | None = None,
        router: ModelRouter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.store = store
        self.max_rows = self.settings.postgres.max_sql_rows
        self.guard = guard or SQLGuard(
            self.settings.security,
            allowed_tables=list(self.settings.postgres.allowed_tables),
            max_rows=self.max_rows,
        )
        self.audit = audit or AuditLog(self.settings)
        self.router = router or ModelRouter(self.settings.llm)
        self.prompt = get_prompt("text_to_sql")
        # The guard clamps or injects LIMIT regardless, but telling the model the
        # same ceiling up front means the common case validates without a rewrite
        # warning, and the query it wrote is the query that runs.
        self.system = self.prompt.system.format(max_rows=self.max_rows)

    # -- construction ------------------------------------------------------
    async def construct(
        self, query: Query, store: RelationalStore | None = None, **kwargs: Any
    ) -> tuple[SQLValidation, Usage]:
        """Return validated SQL plus the token cost of producing it.

        Raises :class:`ConstructionError` when the guard rejects both the first
        attempt and its repair, and propagates
        :class:`~ragorc.core.errors.StoreUnavailable` if the schema cannot be
        read — a stale schema would produce hallucinated column names, which is
        worse than failing.
        """
        target = self._require_store(store)
        # Cached by the store: re-reading information_schema per query is a
        # round trip that buys nothing, since DDL changes on deploy boundaries.
        schema = await target.schema_summary()
        question = query.text
        model = self.router.model_for(Task.SELF_QUERY)

        with timed("text_to_sql_construct"):
            candidate, usage = await self._ask(schema, question, model)
            try:
                validation = self.guard.validate(candidate.sql, max_rows=self.max_rows)
            except GuardrailViolation as first:
                self.audit.generated_query(
                    "sql", candidate.sql, allowed=False, rule=first.rule or "unknown"
                )
                log.warning(
                    "sql_guard_rejected",
                    rule=first.rule,
                    attempt=1,
                    error=str(first)[:300],
                )
                validation, repair_usage = await self._repair(
                    schema, question, candidate.sql, first
                )
                usage = usage + repair_usage

        self.audit.generated_query("sql", validation.sql, allowed=True)
        log.info(
            "sql_constructed",
            tables=validation.tables,
            joins=validation.joins,
            warnings=validation.warnings,
            model=usage.model,
        )
        return validation, usage

    async def _ask(self, schema: str, question: str, model: str) -> tuple[SQLQuery, Usage]:
        rendered = self.prompt.render(schema=schema, question=question)
        result, usage = await self.llm.structured(
            rendered,
            SQLQuery,
            system=self.system,
            model=model,
            stage="text_to_sql",
        )
        return result, usage

    async def _repair(
        self, schema: str, question: str, rejected: str, violation: GuardrailViolation
    ) -> tuple[SQLValidation, Usage]:
        """One corrective round trip, then give up.

        Escalated to the strong model on purpose: this call happens only on the
        failure path, at most once per query, and it is the last chance to answer
        a question the router already decided needs SQL. Paying frontier prices
        for a single call in the tail is a rounding error next to abandoning the
        query, and the escalation measurably raises the odds that a subtle schema
        mistake is actually fixed rather than restated.
        """
        model = self.router.model_for(Task.SELF_QUERY, escalate=True)
        feedback = _repair_question(question, rejected, str(violation))
        candidate, usage = await self._ask(schema, feedback, model)
        try:
            validation = self.guard.validate(candidate.sql, max_rows=self.max_rows)
        except GuardrailViolation as second:
            self.audit.generated_query(
                "sql", candidate.sql, allowed=False, rule=second.rule or "unknown"
            )
            log.warning("sql_guard_rejected", rule=second.rule, attempt=2, error=str(second)[:300])
            raise ConstructionError(
                "text-to-SQL could not produce a statement the SQL guard accepts",
                first_rule=violation.rule,
                second_rule=second.rule,
                attempts=2,
                sql=candidate.sql[:300],
            ) from second
        log.info("sql_repaired", rule=violation.rule, model=model)
        return validation, usage

    # -- execution ---------------------------------------------------------
    async def construct_and_execute(
        self, query: Query, store: RelationalStore | None = None
    ) -> tuple[list[dict[str, Any]], SQLValidation, Usage]:
        """Construct, validate, then run — in that order, always.

        Only ``validation.sql`` is executed, never the model's raw output: the
        guard's rewrite is what carries the injected/clamped ``LIMIT``.
        """
        # Checked before the model is even asked. This library cannot scope a
        # generated statement to a tenant — where a predicate belongs across
        # joins and CTEs is a schema question — so with tenant isolation on, this
        # leg refuses unless the operator has declared how isolation is actually
        # enforced. The gap this closes was silent: tenant_id reached here only to
        # stamp the resulting chunk, never to filter the query, so the vector leg
        # was scoped and the sql leg read every tenant's rows.
        require_generated_query_isolation("SQL", self.settings)
        target = self._require_store(store)
        validation, usage = await self.construct(query, target)
        with timed("text_to_sql_execute", tables=list(validation.tables)):
            rows = await target.execute_readonly(validation.sql, limit=self.max_rows)
        log.info("sql_executed", rows=len(rows), tables=validation.tables)
        return rows, validation, usage

    def _require_store(self, store: RelationalStore | None) -> RelationalStore:
        target = store or self.store
        if target is None:
            raise ConstructionError(
                "text-to-SQL needs a RelationalStore for the schema and for execution",
                hint="pass store= to the constructor or to construct()",
            )
        return target

    # -- result -> chunks ---------------------------------------------------
    def to_chunks(
        self,
        rows: Sequence[dict[str, Any]],
        sql: str,
        *,
        tenant_id: str | None = None,
        score: float = _EXACT_MATCH_SCORE,
    ) -> list[ScoredChunk]:
        """Render a result set as retrieval evidence.

        The whole result set becomes **one** chunk, not one chunk per row. Rows
        are only meaningful next to their header and next to each other: split
        them and reranking will happily drop row 4 of a ranking, fusion will make
        the rows compete with one another for the top-k budget, and the generator
        receives a mutilated table it cannot tell is incomplete.

        Format matters more than it looks. The generator is instructed to quote
        figures exactly and to cite the passage each claim came from, so the
        rendering has to be something a model reads reliably:

        * **Several columns -> a compact markdown table.** Column names appear
          once instead of once per row (a real token saving on a 50-row result),
          values line up so the model can read *down* a column when the question
          was "which is highest", and pipe-delimited tables are the single most
          common tabular format in instruction-tuning data. A ``repr`` of
          ``list[dict]`` instead forces the model to parse ``Decimal('12.50')``
          and ``datetime.datetime(2024, 1, 1, ...)`` noise, which is where
          transcription errors come from.
        * **One column -> one fact per line.** A single-column table is a header
          plus a separator row of pure overhead. ``label: value`` lines read as
          facts, survive being reordered by the context packer, and each line can
          be quoted on its own.

        An empty result returns an empty list rather than a chunk saying "0
        rows". An evidence chunk with no evidence in it invites a citation to a
        passage that supports nothing, and "consulted, found nothing" is already
        recorded by the empty per-store entry in
        :class:`~ragorc.core.models.RetrievalResult`.
        """
        if not rows:
            log.info("sql_no_rows", sql=sql[:200])
            return []

        columns = _columns(rows)
        content = (
            _render_facts(columns[0], rows) if len(columns) == 1 else _render_table(columns, rows)
        )
        # Names, not keys: the payload is serialized to the vector store and read
        # back by the citation layer, both of which expect plain strings.
        names = tuple(str(c) for c in columns)
        # Content-derived id: the same query over the same data yields the same
        # chunk id, so dedupe and the embedding cache both behave.
        document = f"sql:{content_hash(sql)}"
        chunk = Chunk(
            id=chunk_id(document, 0, content),
            content=content,
            document_id=document,
            modality=Modality.TABLE,
            tenant_id=tenant_id,
            metadata={
                "sql": sql,
                "row_count": len(rows),
                "columns": list(names),
                "source": "text_to_sql",
            },
        )
        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                source=RetrievalSource.SQL,
                rank=0,
                component_scores={RetrievalSource.SQL.value: score},
                explain={"sql": sql, "rows": len(rows), "columns": list(names)},
            )
        ]


def _repair_question(question: str, rejected: str, reason: str) -> str:
    """Fold the guard's verdict into the question slot of the same prompt.

    Deliberately not a second prompt template: the system block (rules) and the
    template (schema) stay byte-identical to the first call, so the provider
    prompt cache still applies to the expensive part.
    """
    return (
        f"{question}\n\n"
        "Your previous query was rejected by the read-only SQL guard before it ran.\n"
        f"Rejected query:\n{rejected[:1500]}\n"
        f"Rejection reason: {reason[:600]}\n"
        "Write a corrected query that obeys every rule above. If the schema cannot "
        "answer the question without the rejected construct, return the closest "
        "permitted query instead."
    )


def _columns(rows: Sequence[dict[str, Any]]) -> tuple[Any, ...]:
    """Column order from the first row, plus any keys later rows add.

    Driver rows preserve ``SELECT`` order, and the model chose that order for a
    reason (the entity first, the measure last); reordering it alphabetically
    would make the table read against the question.

    Keys are returned exactly as the row carried them, never coerced to ``str``.
    A stringified key is not a valid ``row.get`` argument, so coercing here and
    looking up there renders every value of a non-``str`` column as ``NULL`` —
    silent data loss in the one kind of evidence the generator is told to quote
    verbatim. Stringification belongs at the render and metadata boundary.
    """
    seen: dict[Any, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return tuple(seen)


def _cell(value: Any) -> str:
    """One database value as text the generator can quote verbatim."""
    if value is None:
        # Explicit, because a blank cell reads as "not applicable" while NULL in
        # an aggregate usually means "no matching rows", which is an answer.
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        # str() preserves the stored scale: NUMERIC(10,2) '12.50' must not print
        # as 12.5 when the generator is told to reproduce figures exactly.
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes | memoryview):
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, list | tuple | dict):
        # jsonb / array columns: JSON is compact and unambiguous, and str() of a
        # Python dict emits single quotes that read as SQL string literals.
        return orjson.dumps(value, default=str).decode()
    return str(value)


def _flatten(text: str) -> str:
    """Collapse to one line and neutralize the delimiter.

    A newline inside a value would end the markdown row early and silently
    shift every following column by one.
    """
    return " ".join(text.split()).replace("|", "\\|")


def _clip(text: str) -> str:
    return text if len(text) <= _MAX_CELL_CHARS else f"{text[:_MAX_CELL_CHARS]}..."


def _render_table(columns: Sequence[Any], rows: Sequence[dict[str, Any]]) -> str:
    header = "| " + " | ".join(_flatten(str(c)) for c in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_clip(_flatten(_cell(row.get(c)))) for c in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _render_facts(column: Any, rows: Sequence[dict[str, Any]]) -> str:
    label = str(column)
    return "\n".join(f"{label}: {_clip(_flatten(_cell(row.get(column))))}" for row in rows)
