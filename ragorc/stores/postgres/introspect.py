"""Schema introspection for the Text-to-SQL prompt.

What the model actually needs
-----------------------------
A Text-to-SQL model fails in three characteristic ways, and each one maps to a
piece of context that has to be in the prompt:

1. **It invents columns.** Fixed by the column list with types.
2. **It invents join conditions.** Fixed by declaring primary and foreign keys —
   given ``chunks.document_id -> documents.id`` the model stops guessing
   ``chunks.doc = documents.name``.
3. **It invents literals.** This is the expensive one. Asked "how many code
   chunks are there", a model that has never seen the data writes
   ``WHERE modality = 'Code'`` or ``= 'source_code'`` and gets zero rows — a
   *silently* wrong answer, not an error. Handing it the actual distinct values
   of low-cardinality text columns removes the entire failure class, and is
   consistently the single highest-value addition to a Text-to-SQL prompt.

Row counts are included because they change the SQL the model writes (it will
add ``LIMIT`` and avoid cross joins on a big table) and because they tell a human
reading a trace whether an empty result was plausible.

Why ``reltuples`` and ``pg_stats`` instead of counting
------------------------------------------------------
This runs to build a *prompt*, not to answer a question, so no part of it may
scan a table:

* ``pg_class.reltuples`` is the planner's own estimate, maintained by
  ``ANALYZE``/autovacuum. Reading it is a single index lookup regardless of table
  size, where ``COUNT(*)`` is a full heap scan — seconds to minutes on a large
  table, per query, for a number the model only needs to an order of magnitude.
* ``pg_stats.most_common_vals`` is the statistics collector's existing sample of
  the most frequent values. It is free: the work was already done by ``ANALYZE``.
  ``SELECT DISTINCT col FROM t`` would be a full scan per candidate column.

Only when a table has never been analysed does this fall back to a probe, and
that probe reads a bounded prefix of the table (``LIMIT`` inside the subquery) so
its cost does not grow with the corpus.

Why cache the result
--------------------
A schema changes on deploys; a query arrives thousands of times between them.
Introspecting per query costs four catalog round trips plus the string assembly
to produce a byte-identical answer — pure latency added to every request. The
summary is cached under ``cache.cache_schema`` and invalidated by
``refresh=True``, which is what a migration hook should call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier, Literal
from psycopg_pool import AsyncConnectionPool

from ragorc.core.ids import cache_key
from ragorc.core.protocols import Cache
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["ColumnInfo", "SchemaIntrospector", "TableInfo"]

# These are prompt-shaping constants, not deployment knobs, so they live here
# rather than in Settings (which must not grow a field per module). They are
# constructor arguments as well, for the rare caller that needs to tune them.
#
# 12 sample values: enough to enumerate a real enum-like column, few enough that
# a wide table's samples do not crowd the schema out of the prompt.
MAX_SAMPLE_VALUES = 12
# Above ~40 distinct values a column is an identifier or free text, not an enum;
# listing its values teaches the model nothing and costs tokens.
MAX_DISTINCT_FOR_SAMPLES = 40
# Bounded prefix scanned when a table has no statistics yet.
SAMPLE_PROBE_ROWS = 5_000
# Sample values are truncated: a 4 KB value in a "low-cardinality" column would
# otherwise blow the prompt budget on its own.
MAX_SAMPLE_CHARS = 60

# Types worth sampling. Sampling a numeric or timestamp column tells the model
# nothing it cannot infer from the type.
_SAMPLEABLE_TYPES = frozenset({"text", "character varying", "character", "name", "citext"})


@dataclass(slots=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool = True
    samples: tuple[str, ...] = ()

    def render(self) -> str:
        parts = [f"  {self.name} {self.type}"]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.samples:
            parts.append("-- values: " + ", ".join(self.samples))
        return " ".join(parts)


@dataclass(slots=True)
class TableInfo:
    name: str
    kind: str = "r"
    """``pg_class.relkind``: ``r``/``p`` table, ``v`` view, ``m`` matview."""
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = ()
    approx_rows: int = 0

    def render(self) -> str:
        label = "VIEW" if self.kind in {"v", "m"} else "TABLE"
        header = f"{label} {self.name}  -- ~{self.approx_rows:,} rows"
        lines = [header, *(c.render() for c in self.columns)]
        if self.primary_key:
            lines.append(f"  PRIMARY KEY ({', '.join(self.primary_key)})")
        for cols, ref_table, ref_cols in self.foreign_keys:
            lines.append(f"  FOREIGN KEY ({', '.join(cols)}) -> {ref_table}({', '.join(ref_cols)})")
        return "\n".join(lines)


# One query per concern, four total. Splitting them is faster than one big join:
# each is an index scan over a system catalog, and joining constraint rows
# against attribute rows in SQL would multiply the result set for no benefit.
_COLUMNS_SQL = """
SELECT c.relname::text                            AS table_name,
       c.relkind::text                             AS relkind,
       a.attname::text                             AS column_name,
       format_type(a.atttypid, a.atttypmod)::text  AS data_type,
       a.attnotnull                                AS not_null
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid
WHERE n.nspname = %(schema)s
  AND c.relkind = ANY (ARRAY['r', 'p', 'v', 'm', 'f'])
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum
"""

_CONSTRAINTS_SQL = """
SELECT c.relname::text  AS table_name,
       con.contype::text AS kind,
       (SELECT array_agg(a.attname::text ORDER BY k.ord)
          FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum
       ) AS cols,
       fn.nspname::text AS ref_schema,
       fc.relname::text AS ref_table,
       (SELECT array_agg(a.attname::text ORDER BY k.ord)
          FROM unnest(con.confkey) WITH ORDINALITY AS k(attnum, ord)
          JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum
       ) AS ref_cols
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_class fc ON fc.oid = con.confrelid
LEFT JOIN pg_namespace fn ON fn.oid = fc.relnamespace
WHERE n.nspname = %(schema)s
  AND con.contype = ANY (ARRAY['p'::"char", 'f'::"char"])
"""

# reltuples is -1 on a relation that has never been analysed, and a float; the
# GREATEST(...) floor keeps the rendered summary from claiming "-1 rows".
_ROWCOUNT_SQL = """
SELECT c.relname::text AS table_name,
       GREATEST(c.reltuples, 0)::bigint AS approx_rows
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = %(schema)s
  AND c.relkind = ANY (ARRAY['r', 'p', 'm', 'f'])
"""

# ``most_common_vals`` is declared ``anyarray``; the double cast through text is
# the standard way to read it, because anyarray has no direct cast to text[].
# n_distinct < 0 means "a fraction of the row count" — the planner's way of
# saying the count scales with the table, i.e. not an enum.
_STATS_SQL = """
SELECT tablename::text AS table_name,
       attname::text   AS column_name,
       n_distinct,
       most_common_vals::text::text[] AS common_vals
FROM pg_stats
WHERE schemaname = %(schema)s
  AND most_common_vals IS NOT NULL
"""


class SchemaIntrospector:
    """Builds and caches the compact DDL summary handed to the SQL constructor."""

    def __init__(
        self,
        pool: AsyncConnectionPool,
        settings: Settings | None = None,
        *,
        cache: Cache | None = None,
        max_sample_values: int = MAX_SAMPLE_VALUES,
        max_distinct_for_samples: int = MAX_DISTINCT_FOR_SAMPLES,
        sample_probe_rows: int = SAMPLE_PROBE_ROWS,
    ) -> None:
        self.pool = pool
        self.settings = settings or get_settings()
        self.cache = cache
        self.max_sample_values = max_sample_values
        self.max_distinct_for_samples = max_distinct_for_samples
        self.sample_probe_rows = sample_probe_rows
        self._local: str | None = None
        """Process-local memo. Even with a Cache configured this saves the
        serialization round trip on the second query in the same process."""

    # -- public API --------------------------------------------------------
    async def summary(self, *, refresh: bool = False) -> str:
        """Return the DDL summary, from cache unless ``refresh``."""
        if not refresh:
            if self._local is not None:
                return self._local
            cached = await self._cache_get()
            if cached is not None:
                self._local = cached
                return cached

        tables = await self.tables()
        text = "\n\n".join(t.render() for t in tables)
        self._local = text
        await self._cache_set(text)
        log.info("pg_schema_introspected", tables=len(tables), chars=len(text))
        return text

    async def tables(self) -> list[TableInfo]:
        """The structured form, for callers that want more than a prompt string."""
        pg = self.settings.postgres
        params = {"schema": pg.schema_name}
        allowed = self._allowed()

        async with self.pool.connection() as conn:
            # Text format, deliberately: the catalogs return types psycopg has no
            # binary loader for (``"char"``, ``regtype``, ``anyarray``), which
            # would come back as raw bytes. This runs once per cache miss, so the
            # binary protocol's win is irrelevant here.
            cur = conn.cursor(binary=False, row_factory=dict_row)
            await cur.execute(_COLUMNS_SQL, params)
            column_rows = await cur.fetchall()
            await cur.execute(_CONSTRAINTS_SQL, params)
            constraint_rows = await cur.fetchall()
            await cur.execute(_ROWCOUNT_SQL, params)
            rowcount_rows = await cur.fetchall()
            await cur.execute(_STATS_SQL, params)
            stats_rows = await cur.fetchall()

            tables = self._assemble(
                column_rows, constraint_rows, rowcount_rows, stats_rows, allowed
            )
            await self._fill_missing_samples(conn, tables)
        return tables

    async def invalidate(self) -> None:
        """Forget the cached summary. Call this from a migration hook."""
        self._local = None
        if self.cache is not None:
            await self.cache.delete(self._key())

    # -- assembly ----------------------------------------------------------
    def _allowed(self) -> frozenset[str] | None:
        """``postgres.allowed_tables`` is the Text-to-SQL allowlist. An empty
        list means "everything the role can read"; anything else means the model
        must not even *learn* that the other tables exist — a table it cannot see
        is a table it cannot be talked into querying."""
        allowed = self.settings.postgres.allowed_tables
        if not allowed:
            return None
        # Accept both ``table`` and ``schema.table`` spellings.
        return frozenset(name.rsplit(".", 1)[-1] for name in allowed)

    def _is_internal(self, table: str) -> bool:
        """Whether ``table`` is this library's own storage rather than user data.

        Excluded from the Text-to-SQL schema because it is the *vector index*, not
        a business table, and describing it to the model actively misleads: the
        chunks table has a ``content`` column, so a model shown it will happily
        answer a prose question with ``SELECT ... FROM ragorc_chunks WHERE content
        ILIKE '%…%'``. Observed exactly that — a query about an approval policy
        generated SQL against a hallucinated column of the chunks table, when the
        answer was prose sitting in Qdrant. Substring matching over embedded text
        is what the vector store exists to do, and it does it better.

        An explicit ``allowed_tables`` entry still wins: someone who deliberately
        lists the chunks table wants it, and this should not override them.
        """
        pg = self.settings.postgres
        internal = {pg.chunks_table.lower(), pg.documents_table.lower()}
        name = table.lower()
        if name in internal:
            return True
        # Per-run tables from the integration suite share the configured prefixes.
        return any(
            name.startswith(f"{base}_") or name.startswith("ragorc_test_") for base in internal
        )

    def _assemble(
        self,
        column_rows: list[dict[str, Any]],
        constraint_rows: list[dict[str, Any]],
        rowcount_rows: list[dict[str, Any]],
        stats_rows: list[dict[str, Any]],
        allowed: frozenset[str] | None,
    ) -> list[TableInfo]:
        counts = {r["table_name"]: int(r["approx_rows"] or 0) for r in rowcount_rows}
        samples = self._sample_index(stats_rows)

        tables: dict[str, TableInfo] = {}
        for row in column_rows:
            name = row["table_name"]
            if allowed is not None:
                if name not in allowed:
                    continue
            elif self._is_internal(name):
                # No explicit allowlist: hide the library's own storage. An
                # explicit entry still wins, which is why this is the else branch.
                continue
            info = tables.get(name)
            if info is None:
                info = tables[name] = TableInfo(
                    name=name, kind=row["relkind"], approx_rows=counts.get(name, 0)
                )
            data_type = row["data_type"]
            info.columns.append(
                ColumnInfo(
                    name=row["column_name"],
                    type=data_type,
                    nullable=not row["not_null"],
                    samples=samples.get((name, row["column_name"]), ()),
                )
            )

        for row in constraint_rows:
            info = tables.get(row["table_name"])
            if info is None:
                continue
            cols = tuple(row["cols"] or ())
            if row["kind"] == "p":
                info.primary_key = cols
            elif row["kind"] == "f" and row["ref_table"]:
                ref = row["ref_table"]
                if allowed is not None and ref not in allowed:
                    # Do not leak the existence of an out-of-scope table.
                    continue
                info.foreign_keys = (*info.foreign_keys, (cols, ref, tuple(row["ref_cols"] or ())))

        return [tables[k] for k in sorted(tables)]

    def _sample_index(
        self, stats_rows: list[dict[str, Any]]
    ) -> dict[tuple[str, str], tuple[str, ...]]:
        out: dict[tuple[str, str], tuple[str, ...]] = {}
        for row in stats_rows:
            n_distinct = row["n_distinct"]
            # Negative n_distinct is a *fraction of the table*, i.e. cardinality
            # that grows with the corpus: an id or free-text column.
            if n_distinct is None or n_distinct < 0 or n_distinct > self.max_distinct_for_samples:
                continue
            values = [v for v in (row["common_vals"] or []) if v is not None]
            if not values:
                continue
            out[(row["table_name"], row["column_name"])] = tuple(
                _quote_sample(v) for v in values[: self.max_sample_values]
            )
        return out

    async def _fill_missing_samples(
        self, conn: AsyncConnection[Any], tables: list[TableInfo]
    ) -> None:
        """Probe text columns that ``ANALYZE`` has not covered yet.

        A freshly loaded table has no ``pg_stats`` rows at all, which is exactly
        the state a first-run Text-to-SQL prompt is built in. The probe reads a
        bounded prefix (``LIMIT sample_probe_rows``) *inside* a subquery, so the
        planner stops after that many rows and the cost is independent of table
        size. It is an approximation — the distinct values of the first N rows —
        and that is the right trade for a prompt hint.
        """
        schema = self.settings.postgres.schema_name
        cur = conn.cursor(binary=False, row_factory=dict_row)
        for table in tables:
            if table.approx_rows > 0 or table.kind not in {"r", "p"}:
                # Analysed already (pg_stats had its say), or not a heap we can
                # cheaply read a prefix of — a view's prefix may cost a full
                # aggregate.
                continue
            key = set(table.primary_key)
            for i, column in enumerate(table.columns):
                if column.samples or column.type not in _SAMPLEABLE_TYPES:
                    continue
                if column.name in key:
                    # A key column is unique by definition; probing it spends a
                    # query to learn nothing.
                    continue
                query = SQL(
                    "SELECT DISTINCT {col}::text AS v FROM "
                    "(SELECT {col} FROM {tbl} LIMIT {probe}) AS s "
                    "WHERE {col} IS NOT NULL LIMIT {cap}"
                ).format(
                    col=Identifier(column.name),
                    tbl=Identifier(schema, table.name),
                    probe=Literal(self.sample_probe_rows),
                    cap=Literal(self.max_distinct_for_samples + 1),
                )
                try:
                    await cur.execute(query)
                    rows = await cur.fetchall()
                except Exception as exc:  # noqa: BLE001 - a hint, never a failure
                    log.debug(
                        "pg_sample_probe_failed",
                        table=table.name,
                        column=column.name,
                        error=str(exc),
                    )
                    continue
                if not rows or len(rows) > self.max_distinct_for_samples:
                    continue
                table.columns[i] = ColumnInfo(
                    name=column.name,
                    type=column.type,
                    nullable=column.nullable,
                    samples=tuple(_quote_sample(r["v"]) for r in rows[: self.max_sample_values]),
                )

    # -- cache -------------------------------------------------------------
    def _key(self) -> str:
        pg = self.settings.postgres
        # The DSN is part of the key so two databases in one process cannot
        # serve each other's schema; only the host/db portion is used, never the
        # credentials.
        return cache_key(
            "pg_schema",
            pg.dsn.get_secret_value().rsplit("@", 1)[-1],
            pg.schema_name,
            tuple(sorted(pg.allowed_tables)),
        )

    async def _cache_get(self) -> str | None:
        if self.cache is None or not self.settings.cache.cache_schema:
            return None
        raw = await self.cache.get(self._key())
        return raw.decode() if raw else None

    async def _cache_set(self, text: str) -> None:
        if self.cache is None or not self.settings.cache.cache_schema:
            return
        # A schema outlives a query by orders of magnitude, so it gets the
        # long-lived TTL rather than a per-request one; ``invalidate()`` is the
        # correctness path for a migration.
        await self.cache.set(self._key(), text.encode(), ttl=self.settings.cache.redis_ttl_s)


def _quote_sample(value: Any) -> str:
    """Render one sample value the way it must appear in SQL.

    Quoted, because the model copies these verbatim into a ``WHERE`` clause and
    an unquoted ``draft`` becomes an identifier reference. Truncated, because a
    single pathological value must not dominate the prompt.
    """
    text = str(value)
    if len(text) > MAX_SAMPLE_CHARS:
        text = text[:MAX_SAMPLE_CHARS] + "..."
    return "'" + text.replace("'", "''") + "'"
