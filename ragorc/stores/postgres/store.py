"""Postgres as three stores in one.

This object wears three hats, and the reason they live together is that they all
read the same rows:

* **pgvector store** — ANN search over ``embedding vector(d)``.
* **full-text index** — lexical search over a generated ``tsvector``.
* **Text-to-SQL target** — arbitrary generated ``SELECT`` under a read-only
  barrier.

Why RRF is computed in SQL, not in Python
-----------------------------------------
:meth:`PostgresStore.hybrid_search` runs the dense and lexical branches as two
CTEs in a single statement and fuses them with reciprocal rank fusion in the
query itself. The obvious alternative — two round trips plus a merge in Python —
is worse on four independent axes:

1. **Bytes on the wire.** Fusing client-side requires shipping both candidate
   lists *with their content*: ``2 x fetch_k`` rows at ~1 KB of text each, of
   which the merge then discards more than half. Fusing server-side ships
   ``top_k`` rows. At the defaults that is ~100 KB versus ~10 KB, and chunk text
   is by far the dominant payload of a retrieval response.
2. **Pool pressure.** Two statements need two connection checkouts. The pool has
   ``max_pool_size`` slots shared by every component; under load, doubling
   checkouts per query doubles queueing delay, which shows up in p99 long before
   it shows up in a benchmark of one query at a time.
3. **Work sharing.** Both branches scan the same table, so the second branch runs
   against a warm buffer cache, and the final projection is a single PK index
   scan for exactly the ids that survived fusion — the rows that lost are never
   materialized.
4. **Consistency.** One statement is one snapshot. Two round trips can straddle a
   commit, fusing a dense ranking over the old corpus with a lexical ranking over
   the new one. The fused ranks are then not a ranking of anything.

The cost is one non-trivial SQL statement, which is a maintenance concern, not a
runtime one.

Why COPY for bulk writes
------------------------
``COPY`` streams rows as one continuous protocol frame: no per-row parse, plan or
bind. ``executemany`` — even using psycopg's pipeline mode — sends a Bind/Execute
pair per row, and the server plans once but binds and executes N times. The
crossover is around a hundred rows; past a few thousand COPY is an order of
magnitude faster. Below the threshold the bulk path's extra staging-table DDL and
``INSERT ... SELECT`` round trips cost more than they save, so
:data:`COPY_MIN_ROWS` picks between them.

COPY has no ``ON CONFLICT``, and our ids are content-derived, so re-ingesting an
unchanged corpus is the *normal* case and must replace rather than raise. The
bulk path therefore COPYs into an ``ON COMMIT DROP`` temp table and then does one
set-based ``INSERT ... SELECT ... ON CONFLICT DO UPDATE``. That is still one
streamed transfer plus one statement, versus N binds.

Scores
------
Everything returned is higher-is-better, per the retrieval contract. pgvector's
``<=>`` is a cosine *distance* in ``[0, 2]``, so it is converted with
``1 - distance`` (which is exactly cosine similarity) and the conversion is done
in the projection, never in the ``ORDER BY`` — ordering must stay on the raw
distance operator or the HNSW index cannot be used.
"""

from __future__ import annotations

import contextlib
import enum
import math
import re
import uuid
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import orjson
import psycopg
import structlog
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.sql import SQL, Composable, Composed, Identifier, Literal, Placeholder
from psycopg.types.json import Jsonb
from psycopg_pool import PoolClosed, PoolTimeout

from ragorc.core.concurrency import CircuitBreaker, retry_async
from ragorc.core.errors import StoreUnavailable, TransientError, ValidationFailed
from ragorc.core.models import (
    Chunk,
    Document,
    FloatArray,
    Modality,
    RetrievalSource,
    ScoredChunk,
    utcnow,
)
from ragorc.core.protocols import Cache
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.security.tenancy import require_tenant
from ragorc.stores.postgres.ddl import (
    CHUNK_COLUMNS,
    CHUNK_READ_COLUMNS,
    DOCUMENT_COLUMNS,
    chunks_table,
    documents_table,
    ensure_schema,
)
from ragorc.stores.postgres.introspect import SchemaIntrospector
from ragorc.stores.postgres.pool import close_pool, open_pool

log = structlog.get_logger(__name__)

Statement = SQL | Composed
"""What ``execute()`` accepts. Narrower than ``Composable``, which also covers
fragments such as a bare ``Identifier``."""

__all__ = ["COPY_MIN_ROWS", "PostgresStore"]

COPY_MIN_ROWS = 128
"""Row count at which the COPY path starts winning. Deliberately equal to
``IndexingSettings.batch_size`` so a default ingest batch takes the fast path;
below it, staging-table DDL plus ``INSERT ... SELECT`` costs more round trips
than a pipelined ``executemany`` saves."""

MAX_CELL_CHARS = 500
"""Per-cell truncation for Text-to-SQL results. One 2 MB text column would
otherwise consume the whole generation context by itself."""

MAX_CELL_BYTES = 64
"""Bytes of a ``bytea`` column that get rendered. A prompt cannot use binary
data; the point is only to show that a value was present and how big it was."""

# Serialization-level failures are genuinely transient — two concurrent ingest
# workers upserting overlapping chunk ids can deadlock — so the write path
# retries. Values are fixed rather than settings-driven because they describe
# lock-contention timescales, not a deployment preference.
_WRITE_RETRY_ATTEMPTS = 3
_WRITE_RETRY_BASE_DELAY = 0.1
_WRITE_RETRY_MAX_DELAY = 2.0

# SQLSTATEs that psycopg models as ``OperationalError`` even though the server
# answered, so they must not be mistaken for a dead database:
#   57014 query_canceled       - our own statement_timeout fired
#   40001 serialization_failure- concurrent transaction, retry the statement
#   40P01 deadlock_detected    - ditto
_QUERY_LEVEL_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.errors.QueryCanceled,
    psycopg.errors.SerializationFailure,
    psycopg.errors.DeadlockDetected,
)
_WRITE_CONFLICT_ERRORS: tuple[type[BaseException], ...] = (
    psycopg.errors.SerializationFailure,
    psycopg.errors.DeadlockDetected,
)

# Filter keys that map onto real columns. Anything else is treated as a metadata
# containment predicate, which is what the ``jsonb_path_ops`` GIN index answers.
_FILTER_COLUMNS: dict[str, str] = {
    "id": "id",
    "ids": "id",
    "chunk_id": "id",
    "chunk_ids": "id",
    "document_id": "document_id",
    "document_ids": "document_id",
    "parent_id": "parent_id",
    "level": "level",
    "modality": "modality",
    "tenant_id": "tenant_id",
}

# Composable type names for the staging table. ``vector`` carries its dimension
# so the temp table rejects a mis-sized embedding before the real table has to.
_PG_TYPES: dict[str, Composable] = {
    "text": SQL("text"),
    "int4": SQL("integer"),
    "jsonb": SQL("jsonb"),
    "timestamptz": SQL("timestamptz"),
}


@register("store", "postgres")
class PostgresStore:
    """``RelationalStore`` over psycopg3 + pgvector.

    Reads and writes go through the primary pool; :meth:`execute_readonly` uses
    the read-only pool when ``postgres.readonly_dsn`` is configured.
    """

    name = "postgres"

    def __init__(self, settings: Settings | None = None, *, cache: Cache | None = None) -> None:
        self.settings = settings or get_settings()
        self.pg = self.settings.postgres
        self.cache = cache
        self._chunks = chunks_table(self.pg)
        self._documents = documents_table(self.pg)
        self._introspector: SchemaIntrospector | None = None
        # One breaker for the whole store: the failure being tracked is "the
        # database is unreachable", which is not per-query. Thresholds are the
        # concurrency defaults — the first few requests pay the timeout, the rest
        # fail instantly and let the multi-store retriever degrade.
        self._breaker = CircuitBreaker("postgres")

    # ------------------------------------------------------------------
    # Connections
    # ------------------------------------------------------------------
    @contextlib.asynccontextmanager
    async def _connection(self, *, readonly: bool = False) -> AsyncIterator[AsyncConnection[Any]]:
        """Check out a connection, translating transport failures.

        Only transport-level errors count against the circuit breaker. A query
        that fails with a syntax or constraint error proves the database is
        *alive*, so counting it would trip the breaker on a bad generated query
        and take out the healthy vector search alongside it.

        That distinction is not free, because psycopg maps several *query*-level
        SQLSTATEs onto ``OperationalError``: a statement timeout (57014), a
        serialization failure (40001) and a deadlock (40P01) are all
        ``OperationalError`` subclasses even though the server answered. Catching
        ``OperationalError`` alone would therefore trip the breaker on exactly the
        conditions the timeout exists to produce, so those are re-raised first and
        untouched.
        """
        self._breaker.check()
        pool = await open_pool(self.pg, readonly=readonly)
        try:
            async with pool.connection() as conn:
                yield conn
        except _QUERY_LEVEL_ERRORS:
            # The server responded, so the store is not down. Let the real error
            # reach the caller: a Text-to-SQL repair loop needs the database's own
            # message to fix the query, and the write path needs to tell a
            # deadlock apart from an outage.
            raise
        except (psycopg.OperationalError, PoolTimeout, PoolClosed, OSError) as exc:
            self._breaker.record_failure()
            raise StoreUnavailable("postgres", str(exc)) from exc
        else:
            self._breaker.record_success()

    async def ensure_schema(self, *, drop: bool = False) -> None:
        """Create extension, tables and indexes. Idempotent; safe at every boot."""
        pool = await open_pool(self.pg)
        try:
            await ensure_schema(pool, self.pg, drop=drop)
        except _QUERY_LEVEL_ERRORS:
            raise
        except (psycopg.OperationalError, PoolTimeout, PoolClosed, OSError) as exc:
            raise StoreUnavailable("postgres", str(exc)) from exc
        if self._introspector is not None:
            await self._introspector.invalidate()

    async def close(self) -> None:
        """Release both pools. A shutdown hook, not a per-request operation."""
        await close_pool(self.pg)
        if self.pg.readonly_dsn.get_secret_value():
            await close_pool(self.pg, readonly=True)

    # ------------------------------------------------------------------
    # Text-to-SQL
    # ------------------------------------------------------------------
    async def execute_readonly(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Run a guard-approved ``SELECT`` behind three more barriers.

        The statement string is **never** interpolated here: it arrives already
        parsed and allowlisted by the guard layer, and this layer's job is to add
        the barriers a parser cannot provide.

        * ``SET TRANSACTION READ ONLY`` — the server refuses any write node for
          the duration, including writes reached through a function.
        * ``SET LOCAL statement_timeout`` — a cartesian product that survived the
          join limit still cannot pin a backend past the deadline.
        * a row cap enforced *twice*: the statement is wrapped in
          ``SELECT * FROM (...) LIMIT n`` so the server stops producing rows, and
          the fetch itself is bounded, so a wrap that somehow did not apply still
          cannot stream an unbounded result into memory.

        Rows come back as plain JSON-safe dicts (``Decimal`` -> float, temporal
        types -> ISO strings, ``bytea`` -> a short repr, long text truncated) so
        the result can go straight into a prompt and through ``orjson`` without
        a second conversion pass.

        The cursor is text-format on purpose: a generated projection can return
        any type, and psycopg has binary loaders only for the types it knows.

        A statement the server rejects — bad syntax, unknown column, timeout —
        surfaces as the original ``psycopg.Error``, not as a ragorc exception. That
        is deliberate: the self-correcting Text-to-SQL loop feeds the database's
        own message back to the model, and wrapping it would throw away the only
        information that makes the repair possible.
        """
        cap = max(1, min(int(limit or self.pg.max_sql_rows), self.pg.max_sql_rows))
        statement = _capped(sql, cap)
        use_readonly_pool = bool(self.pg.readonly_dsn.get_secret_value())

        with timed("pg_execute_readonly", rows_cap=cap):
            async with self._connection(readonly=use_readonly_pool) as conn, conn.transaction():
                cur = conn.cursor(binary=False, row_factory=dict_row)
                await cur.execute("SET TRANSACTION READ ONLY")
                await cur.execute(
                    SQL("SET LOCAL statement_timeout = {ms}").format(
                        ms=Literal(int(self.pg.statement_timeout_ms))
                    )
                )
                await cur.execute(statement, list(params) if params else None)
                rows = await cur.fetchmany(cap)

        out = [{str(k): _json_safe(v) for k, v in row.items()} for row in rows]
        log.info("pg_readonly_query", rows=len(out), readonly_role=use_readonly_pool)
        return out

    async def schema_summary(self, *, refresh: bool = False) -> str:
        """Compact DDL summary for the Text-to-SQL prompt. See
        :class:`~ragorc.stores.postgres.introspect.SchemaIntrospector`."""
        introspector = await self._get_introspector()
        try:
            return await introspector.summary(refresh=refresh)
        except _QUERY_LEVEL_ERRORS:
            raise
        except (psycopg.OperationalError, PoolTimeout, PoolClosed, OSError) as exc:
            self._breaker.record_failure()
            raise StoreUnavailable("postgres", str(exc)) from exc

    async def _get_introspector(self) -> SchemaIntrospector:
        if self._introspector is None:
            pool = await open_pool(self.pg, readonly=bool(self.pg.readonly_dsn.get_secret_value()))
            self._introspector = SchemaIntrospector(pool, self.settings, cache=self.cache)
        return self._introspector

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    async def fulltext_search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Lexical search.

        Two engines, one interface:

        * ``ts_rank_cd`` (default, portable) is a **cover-density** heuristic: it
          rewards query terms appearing close together in the document. It has no
          IDF term and no document-length normalization, so on a corpus with
          mixed document lengths and a query containing one rare and one common
          term it ranks by proximity rather than by informativeness. Fine for
          re-ranking a small candidate set, weaker as a primary ranker.
        * ParadeDB's ``@@@`` (``postgres.use_pg_search``) is real BM25 over a
          Tantivy index: corpus-level IDF plus length normalization, i.e. the
          scoring function the lexical half of hybrid retrieval assumes. Costs an
          extra extension and a second index structure to maintain.

        Normalization flag ``32`` divides ``ts_rank_cd``'s raw rank by itself plus
        one, mapping it into ``(0, 1)``. Without it the rank is an unbounded
        float whose scale depends on document length, which makes it useless to
        fuse with a cosine similarity.
        """
        k = self._top_k(top_k)
        if not query.strip():
            return []
        params: dict[str, Any] = {"q": query, "k": k}
        clauses = self._filter_clauses(filters, tenant_id, alias="c", params=params)
        statement = (
            self._bm25_statement(clauses)
            if self.pg.use_pg_search
            else self._tsrank_statement(clauses)
        )

        with timed("pg_fulltext_search", top_k=k):
            rows = await self._fetch(statement, params)
        return _scored(rows, RetrievalSource.FULLTEXT, "fulltext")

    async def vector_search(
        self,
        query_vector: FloatArray,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """ANN search over the pgvector column.

        ``ORDER BY embedding <=> $1`` is the only shape pgvector's index
        recognizes, so the distance-to-similarity conversion happens one level up
        in a subquery projection. Writing ``ORDER BY 1 - (embedding <=> $1) DESC``
        instead silently degrades to a sequential scan plus a sort.
        """
        k = self._top_k(top_k)
        vector = _as_vector(query_vector, self.pg.vector_dimension)
        params: dict[str, Any] = {"vec": vector, "k": k}
        clauses = self._filter_clauses(filters, tenant_id, alias="c", params=params)

        with timed("pg_vector_search", top_k=k):
            rows = await self._fetch(
                self._vector_statement(clauses), params, tuning=self._ann_tuning(k)
            )
        return _scored(rows, RetrievalSource.DENSE, "dense")

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: FloatArray,
        *,
        top_k: int | None = None,
        candidates: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        """Dense + lexical + reciprocal rank fusion in **one** round trip.

        Each branch is ranked independently over ``candidates`` rows
        (``retrieval.fetch_k``), then fused as
        ``sum over branches of 1 / (rrf_k + rank)``. RRF is used rather than a
        weighted score sum because the two branches produce incomparable scales —
        a cosine similarity and a cover-density rank — and normalizing them
        requires knowing each branch's score distribution for *this* query, which
        a single statement cannot know. Ranks are scale-free, which is the whole
        argument for RRF.

        ``row_number()`` is applied *outside* each branch's ``LIMIT`` subquery: a
        window function is evaluated before ``ORDER BY``/``LIMIT`` at the same
        query level, so ranking inside the branch would number every matching row
        in the table and throw away the index's top-k shortcut.
        """
        k = self._top_k(top_k)
        cand = max(k, int(candidates or self.settings.retrieval.fetch_k))
        vector = _as_vector(query_vector, self.pg.vector_dimension)
        params: dict[str, Any] = {
            "vec": vector,
            "q": query_text,
            "k": k,
            "cand": cand,
            "rrf": int(self.settings.retrieval.rrf_k),
        }
        clauses = self._filter_clauses(filters, tenant_id, alias="c", params=params)

        with timed("pg_hybrid_search", top_k=k, candidates=cand):
            rows = await self._fetch(
                self._hybrid_statement(clauses), params, tuning=self._ann_tuning(cand)
            )

        out: list[ScoredChunk] = []
        for rank, row in enumerate(rows):
            dense_score = float(row.get("dense_score") or 0.0)
            lex_score = float(row.get("lex_score") or 0.0)
            out.append(
                ScoredChunk(
                    chunk=_row_to_chunk(row),
                    score=float(row["score"]),
                    source=RetrievalSource.FUSED,
                    rank=rank,
                    component_scores={"dense": dense_score, "fulltext": lex_score},
                    explain={
                        "dense_rank": row.get("dense_rank"),
                        "fulltext_rank": row.get("lex_rank"),
                        "fusion": "rrf",
                        "rrf_k": params["rrf"],
                    },
                )
            )
        return out

    # ------------------------------------------------------------------
    # Reads by key
    # ------------------------------------------------------------------
    async def get_chunks(self, ids: Sequence[str], *, tenant_id: str | None = None) -> list[Chunk]:
        """Fetch by id, preserving the caller's order.

        Scoped like every other read here. It was the exception: ``= ANY(array)``
        with no tenant predicate, so naming a chunk id fetched its body whoever
        owned it — and the ids on the GraphRAG path come out of Neo4j, which
        stores no tenant at all.

        ``= ANY(array)`` is one index scan over a passed-in array rather than N
        statements or an ``IN`` list that changes shape per call and therefore
        defeats the prepared-statement cache. ``array_position`` restores the
        requested order server-side, which matters because callers hand these
        straight to a reranker whose input order is part of its contract.
        """
        wanted = list(dict.fromkeys(ids))
        if not wanted:
            return []
        params: dict[str, Any] = {"ids": wanted}
        clauses = self._filter_clauses(None, tenant_id, alias="c", params=params)
        clauses.append(SQL("c.id = ANY({ids})").format(ids=Placeholder("ids")))
        statement = SQL(
            "SELECT {cols} FROM {tbl} AS c {where} ORDER BY array_position({ids}, c.id)"
        ).format(
            cols=_qualified_columns("c"),
            tbl=self._chunks,
            where=_where(clauses),
            ids=Placeholder("ids"),
        )
        rows = await self._fetch(statement, params)
        return [_row_to_chunk(row) for row in rows]

    async def get_children(self, parent_id: str, *, tenant_id: str | None = None) -> list[Chunk]:
        """Children of a parent chunk, in document order.

        This is the parent-document retriever's expansion step: search hits the
        small precise child, generation gets the large parent's full span.
        """
        params: dict[str, Any] = {"pid": parent_id}
        clauses = self._filter_clauses(None, tenant_id, alias="c", params=params)
        clauses.append(SQL("c.parent_id = {ph}").format(ph=Placeholder("pid")))
        statement = SQL("SELECT {cols} FROM {tbl} AS c {where} ORDER BY c.index").format(
            cols=_qualified_columns("c"), tbl=self._chunks, where=_where(clauses)
        )
        rows = await self._fetch(statement, params)
        return [_row_to_chunk(row) for row in rows]

    async def count(self, *, tenant_id: str | None = None) -> int:
        """Exact chunk count. Exact, not estimated: this answers "did the ingest
        land?", where the planner's ``reltuples`` guess is not good enough."""
        params: dict[str, Any] = {}
        clauses = self._filter_clauses(None, tenant_id, alias="c", params=params)
        statement = SQL("SELECT count(*) AS n FROM {tbl} AS c {where}").format(
            tbl=self._chunks, where=_where(clauses)
        )
        rows = await self._fetch(statement, params)
        return int(rows[0]["n"]) if rows else 0

    async def delete_document(self, document_id: str, *, tenant_id: str | None = None) -> int:
        """Delete a document and its chunks; return the number of rows removed.

        The chunk delete is explicit even though the FK cascades, because the
        cascade's row count is not reported back and "how many chunks did that
        remove?" is the number an operator actually wants.

        Scoped like every other statement in this class. It was the one that was
        not: ``count`` directly above threads ``tenant_id`` through
        :meth:`_filter_clauses` and this deleted by id alone, so with isolation on
        a caller who could name another tenant's document id could remove it. Not
        reachable through the shipped entry points — :func:`ragorc.core.ids.document_id`
        folds the tenant in and both the server route and the loaders derive it —
        but "no caller can currently construct the id" is a property of the callers,
        which is exactly the reasoning :meth:`_filter_clauses` rejects one docstring
        up.

        A delete that matches nothing returns 0 rather than raising, which is what
        makes it idempotent: re-running a purge after a partial failure must not
        turn "already gone" into an error.
        """
        params: dict[str, Any] = {"id": document_id}
        chunk_scope = self._filter_clauses(None, tenant_id, alias="c", params=params)
        # The documents table names its key `id`, the chunks table `document_id`,
        # and `_filter_clauses` writes predicates against a chunk alias — so the
        # document-row scope is built separately rather than reusing that output.
        doc_scope: list[Composable] = []
        tenant = require_tenant(tenant_id or self.settings.tenant_id, self.settings.security)
        if tenant:
            params["tenant"] = tenant
            doc_scope.append(SQL("d.tenant_id = {ph}").format(ph=Placeholder("tenant")))
        with timed("pg_delete_document"):
            async with self._connection() as conn, conn.transaction():
                cur = conn.cursor()
                await cur.execute(
                    SQL("DELETE FROM {tbl} AS c WHERE c.document_id = {ph} {extra}").format(
                        tbl=self._chunks,
                        ph=Placeholder("id"),
                        extra=_and(chunk_scope),
                    ),
                    params,
                )
                removed = max(cur.rowcount, 0)
                await cur.execute(
                    SQL("DELETE FROM {tbl} AS d WHERE d.id = {ph} {extra}").format(
                        tbl=self._documents,
                        ph=Placeholder("id"),
                        extra=_and(doc_scope),
                    ),
                    params,
                )
                removed += max(cur.rowcount, 0)
        log.info("pg_document_deleted", document_id=document_id, rows=removed, tenant_id=tenant)
        return removed

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    @retry_async(
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        base_delay=_WRITE_RETRY_BASE_DELAY,
        max_delay=_WRITE_RETRY_MAX_DELAY,
    )
    async def upsert_documents(self, docs: Sequence[Document]) -> int:
        """Insert or replace documents. Returns the number of rows written."""
        rows = _dedupe([_document_row(d, self.settings.tenant_id) for d in docs])
        if not rows:
            return 0
        with timed("pg_upsert_documents", rows=len(rows)):
            written = await self._write(self._documents, DOCUMENT_COLUMNS, rows, "documents")
        log.info("pg_documents_upserted", rows=written, copy=len(rows) >= COPY_MIN_ROWS)
        return written

    @retry_async(
        max_attempts=_WRITE_RETRY_ATTEMPTS,
        base_delay=_WRITE_RETRY_BASE_DELAY,
        max_delay=_WRITE_RETRY_MAX_DELAY,
    )
    async def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Insert or replace chunks, embeddings included.

        Chunks whose ``document_id`` is empty are stored with a NULL FK rather
        than rejected: RAPTOR summary nodes and synthetic proposition chunks have
        no owning document row, and pushing that special case into the ingest
        pipeline would duplicate it at every call site.
        """
        rows = _dedupe(
            [_chunk_row(c, self.settings.tenant_id, self.pg.vector_dimension) for c in chunks]
        )
        if not rows:
            return 0
        with timed("pg_upsert_chunks", rows=len(rows)):
            written = await self._write(self._chunks, CHUNK_COLUMNS, rows, "chunks")
        log.info("pg_chunks_upserted", rows=written, copy=len(rows) >= COPY_MIN_ROWS)
        return written

    async def _write(
        self,
        table: Identifier,
        columns: tuple[tuple[str, str], ...],
        rows: list[tuple[Any, ...]],
        label: str,
    ) -> int:
        names = [name for name, _ in columns]
        try:
            async with self._connection() as conn, conn.transaction():
                if len(rows) >= COPY_MIN_ROWS:
                    return await self._copy_upsert(conn, table, columns, rows, label)
                return await self._values_upsert(conn, table, names, rows)
        except _WRITE_CONFLICT_ERRORS as exc:
            # Concurrent ingest workers touching overlapping ids. Retryable by
            # construction: the same statement on a fresh snapshot succeeds, which
            # is why it is re-raised as a ``TransientError`` — the type the retry
            # decorator on the callers keys off.
            raise TransientError(f"postgres write conflict on {label}: {exc}") from exc

    async def _values_upsert(
        self,
        conn: AsyncConnection[Any],
        table: Identifier,
        names: list[str],
        rows: list[tuple[Any, ...]],
    ) -> int:
        statement = SQL(
            "INSERT INTO {tbl} ({cols}) VALUES ({vals}) ON CONFLICT (id) DO UPDATE SET {updates}"
        ).format(
            tbl=table,
            cols=SQL(", ").join(Identifier(n) for n in names),
            vals=SQL(", ").join(Placeholder() for _ in names),
            updates=_excluded_updates(names),
        )
        cur = conn.cursor()
        # psycopg pipelines executemany, so this is one network flush, not N.
        await cur.executemany(statement, rows)
        return len(rows)

    async def _copy_upsert(
        self,
        conn: AsyncConnection[Any],
        table: Identifier,
        columns: tuple[tuple[str, str], ...],
        rows: list[tuple[Any, ...]],
        label: str,
    ) -> int:
        """COPY into an unlogged temp table, then one set-based upsert.

        The staging table exists only because COPY cannot express a conflict
        action. It is created ``ON COMMIT DROP`` inside the caller's transaction,
        gets no indexes (nothing queries it by key) and no generated columns, so
        the tsvector is computed exactly once — in the real table, on the rows
        that actually land.
        """
        names = [name for name, _ in columns]
        # A per-call name keeps two concurrent writers in the same session (or a
        # retry after a rollback) from colliding on the temp relation.
        stage = Identifier(f"ragorc_stage_{label}_{uuid.uuid4().hex[:12]}")
        await conn.execute(
            SQL("CREATE TEMP TABLE {stage} ({defs}) ON COMMIT DROP").format(
                stage=stage, defs=_stage_columns(columns, self.pg.vector_dimension)
            )
        )
        cur = conn.cursor()
        copy_stmt = SQL("COPY {stage} ({cols}) FROM STDIN (FORMAT BINARY)").format(
            stage=stage, cols=SQL(", ").join(Identifier(n) for n in names)
        )
        async with cur.copy(copy_stmt) as copy:
            # Binary COPY sends no type metadata, so the types are declared once
            # here; PostgreSQL applies no cast rules to a binary stream, which is
            # exactly why it is fast.
            copy.set_types([pgtype for _, pgtype in columns])
            for row in rows:
                await copy.write_row(row)

        await conn.execute(
            SQL(
                "INSERT INTO {tbl} ({cols}) SELECT {cols} FROM {stage} "
                "ON CONFLICT (id) DO UPDATE SET {updates}"
            ).format(
                tbl=table,
                cols=SQL(", ").join(Identifier(n) for n in names),
                stage=stage,
                updates=_excluded_updates(names),
            )
        )
        return len(rows)

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------
    def _top_k(self, top_k: int | None) -> int:
        return max(1, int(top_k if top_k is not None else self.settings.retrieval.top_k))

    def _ann_tuning(self, depth: int) -> Statement | None:
        """Per-transaction ANN recall/latency dial.

        There is no dedicated ``ef_search`` setting for pgvector, and inventing
        one is not allowed, so it is derived: ``ef_search`` must be at least the
        ``LIMIT`` or HNSW cannot return that many candidates at all, and it is
        floored at ``ef_construction`` because searching a graph narrower than it
        was built for discards recall the build already paid for.
        """
        if self.pg.vector_index == "hnsw":
            ef = max(depth * 2, self.pg.hnsw_ef_construction)
            return SQL("SET LOCAL hnsw.ef_search = {v}").format(v=Literal(ef))
        if self.pg.vector_index == "ivfflat":
            # sqrt(lists) is pgvector's recommended probe count: recall improves
            # roughly with the square root of the work done.
            probes = max(1, math.isqrt(max(self.pg.ivf_lists, 1)))
            return SQL("SET LOCAL ivfflat.probes = {v}").format(v=Literal(probes))
        return None

    def _filter_clauses(
        self,
        filters: dict[str, Any] | None,
        tenant_id: str | None,
        *,
        alias: str,
        params: dict[str, Any],
    ) -> list[Composable]:
        """Turn a filter dict into predicates plus bound parameters.

        Known keys become column predicates; everything else is folded into a
        single ``metadata @> '{...}'`` containment test, which is the one
        operation the ``jsonb_path_ops`` index can answer and therefore the only
        metadata predicate worth generating.

        Tenant resolution falls back to ``Settings.tenant_id``, and then
        **refuses**. This used to log a warning and emit no predicate — deferring
        to the guard layer, "which sees the request, while this layer only sees a
        filter dict". That reasoning holds only for as long as every path reaches
        the guard, which is a property of the callers rather than of this code,
        and the graph path turned out not to. A store that answers unscoped when
        told to isolate is the one thing tenancy cannot afford to leave to a
        convention.

        The boot argument it replaced does not apply: ``enforce_tenant_isolation``
        is off by default, so nothing refuses until an operator asks for it.
        """
        clauses: list[Composable] = []
        tenant = require_tenant(
            tenant_id if tenant_id is not None else self.settings.tenant_id,
            self.settings.security,
        )
        if tenant:
            params["tenant"] = tenant
            clauses.append(
                SQL("{col} = {ph}").format(col=_col(alias, "tenant_id"), ph=Placeholder("tenant"))
            )

        metadata: dict[str, Any] = {}
        for key, value in (filters or {}).items():
            column = _FILTER_COLUMNS.get(key)
            if column is None:
                metadata[key] = value
                continue
            if value is None:
                clauses.append(SQL("{col} IS NULL").format(col=_col(alias, column)))
                continue
            name = f"f{len(params)}"
            if isinstance(value, (list, tuple, set, frozenset)):
                params[name] = [_scalar(v) for v in value]
                clauses.append(
                    SQL("{col} = ANY({ph})").format(col=_col(alias, column), ph=Placeholder(name))
                )
            else:
                params[name] = _scalar(value)
                clauses.append(
                    SQL("{col} = {ph}").format(col=_col(alias, column), ph=Placeholder(name))
                )

        if metadata:
            # Not run through the prompt-safe serializer: a filter value must
            # match the stored value byte for byte, and that serializer truncates.
            params["md"] = Jsonb(metadata, dumps=_dumps)
            clauses.append(
                SQL("{col} @> {ph}").format(col=_col(alias, "metadata"), ph=Placeholder("md"))
            )
        return clauses

    def _tsrank_statement(self, clauses: list[Composable]) -> Composed:
        # The tsquery is built once, as a one-row FROM item, rather than inlined
        # into both the WHERE clause and ts_rank_cd. Inlining it would parse the
        # query string once per *matching* row, because PostgreSQL performs no
        # common-subexpression elimination and the ranking expression is
        # evaluated before ORDER BY/LIMIT prunes anything. This is also the shape
        # the PostgreSQL manual uses for ranked full-text search, and the planner
        # turns it into a parameterized index scan over the GIN index.
        conditions = [SQL("c.content_tsv @@ q.tsq"), *clauses]
        return SQL(
            "SELECT {cols}, ts_rank_cd(c.content_tsv, q.tsq, 32) AS score "
            "FROM {tbl} AS c, websearch_to_tsquery({cfg}, {q}) AS q(tsq) "
            "{where} ORDER BY score DESC LIMIT {k}"
        ).format(
            cols=_qualified_columns("c"),
            tbl=self._chunks,
            # A literal regconfig, not a parameter: the two-argument
            # ``to_tsvector``/``websearch_to_tsquery`` form with a constant config
            # is what makes the generated column's expression match this query's,
            # which is what lets the planner use the GIN index.
            cfg=Literal(self.pg.fulltext_config),
            q=Placeholder("q"),
            where=_where(conditions),
            k=Placeholder("k"),
        )

    def _bm25_statement(self, clauses: list[Composable]) -> Composed:
        conditions = [SQL("c.content @@@ {q}").format(q=Placeholder("q")), *clauses]
        return SQL(
            "SELECT {cols}, paradedb.score(c.id) AS score "
            "FROM {tbl} AS c {where} ORDER BY score DESC LIMIT {k}"
        ).format(
            cols=_qualified_columns("c"),
            tbl=self._chunks,
            where=_where(conditions),
            k=Placeholder("k"),
        )

    def _vector_statement(self, clauses: list[Composable]) -> Composed:
        conditions = [SQL("c.embedding IS NOT NULL"), *clauses]
        return SQL(
            "SELECT {outer}, 1.0 - hit.dist AS score, hit.dist AS distance FROM ("
            "SELECT {inner}, c.embedding <=> {vec} AS dist FROM {tbl} AS c {where} "
            "ORDER BY dist LIMIT {k}) AS hit"
        ).format(
            outer=_qualified_columns("hit"),
            inner=_qualified_columns("c"),
            vec=Placeholder("vec"),
            tbl=self._chunks,
            where=_where(conditions),
            k=Placeholder("k"),
        )

    def _hybrid_statement(self, clauses: list[Composable]) -> Composed:
        dense_conditions = [SQL("c.embedding IS NOT NULL"), *clauses]
        lex_conditions = [SQL("c.content_tsv @@ q.tsq"), *clauses]
        return SQL(
            "WITH dense AS ("
            "  SELECT id, row_number() OVER (ORDER BY dist) AS rnk FROM ("
            "    SELECT c.id AS id, c.embedding <=> {vec} AS dist"
            "    FROM {tbl} AS c {dense_where} ORDER BY dist LIMIT {cand}"
            "  ) AS d"
            "), lex AS ("
            "  SELECT id, row_number() OVER (ORDER BY ts DESC) AS rnk FROM ("
            "    SELECT c.id AS id, ts_rank_cd(c.content_tsv, q.tsq, 32) AS ts"
            "    FROM {tbl} AS c, websearch_to_tsquery({cfg}, {q}) AS q(tsq)"
            "    {lex_where} ORDER BY ts DESC LIMIT {cand}"
            "  ) AS l"
            "), fused AS ("
            "  SELECT COALESCE(d.id, l.id) AS id,"
            "         COALESCE(1.0 / ({rrf} + d.rnk), 0.0) AS dense_score,"
            "         COALESCE(1.0 / ({rrf} + l.rnk), 0.0) AS lex_score,"
            "         d.rnk AS dense_rank, l.rnk AS lex_rank"
            "  FROM dense AS d FULL JOIN lex AS l ON l.id = d.id"
            ") "
            "SELECT {cols}, f.dense_score + f.lex_score AS score,"
            "       f.dense_score, f.lex_score, f.dense_rank, f.lex_rank "
            "FROM fused AS f JOIN {tbl} AS c ON c.id = f.id "
            "ORDER BY score DESC LIMIT {k}"
        ).format(
            cols=_qualified_columns("c"),
            tbl=self._chunks,
            vec=Placeholder("vec"),
            cfg=Literal(self.pg.fulltext_config),
            q=Placeholder("q"),
            dense_where=_where(dense_conditions),
            lex_where=_where(lex_conditions),
            cand=Placeholder("cand"),
            rrf=Placeholder("rrf"),
            k=Placeholder("k"),
        )

    async def _fetch(
        self,
        statement: Statement,
        params: dict[str, Any],
        *,
        tuning: Statement | None = None,
    ) -> list[dict[str, Any]]:
        """Run one read.

        Wrapped in an explicit transaction whenever ``tuning`` is present:
        ``SET LOCAL`` outside a transaction block is a no-op with a warning, so
        the ANN dial would silently do nothing.
        """
        async with self._connection() as conn:
            if tuning is None:
                cur = await conn.execute(statement, params)
                return await cur.fetchall()
            async with conn.transaction():
                cur = conn.cursor()
                await cur.execute(tuning)
                await cur.execute(statement, params)
                return await cur.fetchall()


# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------
def _col(alias: str | None, name: str) -> Identifier:
    return Identifier(alias, name) if alias else Identifier(name)


def _qualified_columns(alias: str) -> Composed:
    return SQL(", ").join(Identifier(alias, name) for name in CHUNK_READ_COLUMNS)


def _where(clauses: Sequence[Composable]) -> Composable:
    if not clauses:
        return SQL("")
    return SQL("WHERE ") + SQL(" AND ").join(clauses)


def _and(clauses: Sequence[Composable]) -> Composable:
    """The same clauses appended to a ``WHERE`` that already exists.

    Separate from :func:`_where` because the leading keyword is the whole
    difference, and a statement that already has a fixed predicate — a delete by
    id, say — needs the scope joined on rather than introduced.
    """
    if not clauses:
        return SQL("")
    return SQL(" AND ") + SQL(" AND ").join(clauses)


def _excluded_updates(names: Iterable[str]) -> Composed:
    """``SET col = EXCLUDED.col`` for every column but the key."""
    return SQL(", ").join(
        SQL("{c} = EXCLUDED.{c}").format(c=Identifier(name)) for name in names if name != "id"
    )


def _stage_columns(columns: tuple[tuple[str, str], ...], dimension: int) -> Composed:
    parts: list[Composable] = []
    for name, pgtype in columns:
        if pgtype == "vector":
            type_sql: Composable = SQL("vector({d})").format(d=Literal(int(dimension)))
        else:
            type_sql = _PG_TYPES[pgtype]
        parts.append(SQL("{n} {t}").format(n=Identifier(name), t=type_sql))
    return SQL(", ").join(parts)


# A statement terminator cannot appear inside the subquery the cap is built from,
# and ``rstrip(";")`` only reaches one that is literally last. Generated SQL
# routinely writes ``SELECT ...; -- how many rows``, which hides the semicolon
# behind a comment and leaves it mid-subquery — a syntax error. This matches a
# ``;`` only when nothing but whitespace, further semicolons and line comments
# follow it to the end of the string, so the terminator is removed while the
# trailing comment survives. A ``;`` inside a string literal is never matched:
# the character after it is neither whitespace, ``;`` nor the start of a comment,
# so the lookahead fails.
_TRAILING_TERMINATOR = re.compile(r";(?=(?:\s|;|--[^\n]*)*\Z)")


def _capped(statement: str, cap: int) -> Composed:
    """Wrap a statement so the row cap is enforced by the server.

    A subquery wrap is used rather than parsing for a top-level ``LIMIT``: it is
    deterministic, needs no SQL dialect knowledge, and an outer ``LIMIT n`` over
    an inner ``LIMIT m`` yields ``min(n, m)``, so a statement that already had a
    limit keeps the stricter one. Placeholders inside the statement survive —
    psycopg processes them after composition — and any brace characters in the
    statement are inert because it is substituted as an already-composed
    fragment, not as a format template.

    The closing parenthesis goes on its own line because generated SQL routinely
    ends in a ``--`` comment, and a line comment would otherwise swallow it.
    """
    inner = _TRAILING_TERMINATOR.sub("", statement.strip()).strip()
    if not inner:
        raise ValidationFailed("empty SQL statement")
    return SQL("SELECT * FROM (\n{inner}\n) AS ragorc_capped LIMIT {cap}").format(
        inner=SQL(inner),  # noqa: S608 - guard-validated statement, not interpolation
        cap=Literal(cap),
    )


# ---------------------------------------------------------------------------
# Row mapping
# ---------------------------------------------------------------------------
def _dumps(value: Any) -> bytes:
    """orjson for jsonb payloads: 3-10x stdlib ``json``, and it serializes numpy
    scalars and datetimes that would otherwise raise on a metadata dict."""
    return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY, default=str)


def _scalar(value: Any) -> Any:
    """Unwrap the wrappers that reach a filter dict: enum members (``Modality``)
    and numpy scalars, neither of which psycopg can adapt."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_safe(value: Any) -> Any:
    """Make a value both prompt-printable and ``orjson``-serializable."""
    if isinstance(value, enum.Enum):
        # Ahead of the scalar branches, not after them: the enums that reach a
        # row are mixin enums (``Modality`` is ``(str, Enum)``), so an
        # ``isinstance(value, str)`` test matches the *member* and would return
        # it unchanged — rendering ``Modality.CODE`` into a prompt instead of
        # ``code``. Unwrapping first is also what ``_scalar`` does.
        return _json_safe(value.value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_CELL_CHARS else value[:MAX_CELL_CHARS] + "..."
    if isinstance(value, Decimal):
        # Prompts and JSON have no exact-decimal type; a float is the honest
        # lossy rendering and is what a model can reason about.
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return f"<{len(raw)} bytes {raw[:MAX_CELL_BYTES]!r}>"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    to_list = getattr(value, "to_list", None)  # pgvector Vector, and lookalikes
    if callable(to_list):
        return to_list()
    return str(value)


def _as_vector(vector: FloatArray | Sequence[float], dimension: int) -> np.ndarray:
    """Coerce to the 1-D float32 layout pgvector's binary dumper expects.

    ``ravel`` rather than ``reshape``: a ``(1, d)`` array from a batched embedder
    is the common accident, and flattening it here beats an opaque
    "expected ndim to be 1" from deep inside the adapter.
    """
    arr = np.asarray(vector, dtype=np.float32).ravel()
    if arr.size != dimension:
        raise ValidationFailed(
            "query vector dimension mismatch", got=int(arr.size), expected=int(dimension)
        )
    return arr


def _modality(value: Any) -> Modality:
    try:
        return Modality(value)
    except ValueError:
        # Forward compatibility: a row written by a newer version must not make
        # an older reader crash on a whole result set.
        return Modality.TEXT


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        id=row["id"],
        content=row.get("content") or "",
        document_id=row.get("document_id") or "",
        index=int(row.get("index") or 0),
        start_char=int(row.get("start_char") or 0),
        end_char=int(row.get("end_char") or 0),
        metadata=dict(row.get("metadata") or {}),
        parent_id=row.get("parent_id"),
        level=int(row.get("level") or 0),
        modality=_modality(row.get("modality")),
        token_count=row.get("token_count"),
        tenant_id=row.get("tenant_id"),
        created_at=row.get("created_at") or utcnow(),
    )


def _scored(
    rows: list[dict[str, Any]], source: RetrievalSource, component: str
) -> list[ScoredChunk]:
    """Attach rank (from 0) and the per-retriever contribution, per the contract:
    without ``component_scores`` "why did this rank third?" is unanswerable after
    fusion has flattened everything into one number."""
    out: list[ScoredChunk] = []
    for rank, row in enumerate(rows):
        score = float(row["score"])
        out.append(
            ScoredChunk(
                chunk=_row_to_chunk(row),
                score=score,
                source=source,
                rank=rank,
                component_scores={component: score},
                explain=({"cosine_distance": float(row["distance"])} if "distance" in row else {}),
            )
        )
    return out


def _document_row(doc: Document, default_tenant: str | None) -> tuple[Any, ...]:
    return (
        doc.id,
        doc.source,
        doc.title,
        doc.checksum,
        doc.tenant_id or default_tenant,
        # Stored verbatim: ``_json_safe`` truncates for prompts and must never
        # touch the write path.
        Jsonb(doc.metadata, dumps=_dumps),
        doc.created_at,
    )


def _chunk_row(chunk: Chunk, default_tenant: str | None, dimension: int) -> tuple[Any, ...]:
    embedding = None if chunk.dense is None else _as_vector(chunk.dense, dimension)
    return (
        chunk.id,
        chunk.document_id or None,
        chunk.content,
        chunk.index,
        chunk.start_char,
        chunk.end_char,
        chunk.level,
        chunk.parent_id,
        chunk.modality.value,
        chunk.token_count,
        chunk.tenant_id or default_tenant,
        Jsonb(chunk.metadata, dumps=_dumps),
        embedding,
        chunk.created_at,
    )


def _dedupe(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Last write wins on a repeated id.

    ``ON CONFLICT DO UPDATE`` raises "cannot affect row a second time" if the
    same key appears twice in one statement, so the duplicate has to be removed
    before it reaches the server rather than after.
    """
    by_id: dict[Any, tuple[Any, ...]] = {}
    for row in rows:
        by_id[row[0]] = row
    return list(by_id.values())
