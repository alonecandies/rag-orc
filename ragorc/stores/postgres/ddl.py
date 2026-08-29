"""Schema definition for the Postgres store.

The table layout is driven by three access patterns that have to coexist in one
relation, because splitting them would mean keeping two copies of every chunk in
sync:

* **ANN search** over ``embedding vector(d)`` with an HNSW index,
* **lexical search** over a ``tsvector`` with a GIN index,
* **relational access** (fetch by id, fetch a document's chunks in order, fetch a
  parent's children) with plain btree indexes.

Keeping them in one table is what makes hybrid search a single round trip: the
fusion CTE in :mod:`ragorc.stores.postgres.store` joins the dense branch and the
lexical branch back to the *same* rows, so there is no cross-store join and no
possibility of the two indexes disagreeing about what a chunk contains.

Why the tsvector is a GENERATED column
--------------------------------------
``content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content))
STORED`` keeps the full-text index in lockstep with ``content`` with no trigger
to write, no trigger to maintain, and no window in which an updated chunk is
searchable under its old text. The two alternatives are both worse: a
``BEFORE INSERT OR UPDATE`` trigger is another object that can be dropped,
disabled or forgotten in a migration, and an expression index
(``ON t USING gin (to_tsvector(...))``) cannot be used by ``ts_rank_cd`` in the
projection without recomputing the vector per row at query time. The two-argument
``to_tsvector(regconfig, text)`` form is required: the one-argument form depends
on ``default_text_search_config`` and is therefore only ``STABLE``, which
PostgreSQL rejects in a generated column.

Why HNSW and not IVFFlat by default
-----------------------------------
IVFFlat needs a training step over representative data, so an index built on an
empty or small table is permanently mis-clustered and has to be rebuilt after
ingest; HNSW is incremental and has strictly better recall-per-latency at the
same memory budget. The IVFFlat variant is still emitted when
``postgres.vector_index == "ivfflat"``, for corpora big enough that HNSW's build
time or memory becomes the binding constraint.

Why ``jsonb_path_ops`` on metadata
----------------------------------
The only metadata query this store issues is containment (``metadata @> '{...}'``)
for filtered retrieval. ``jsonb_path_ops`` indexes key *paths* rather than every
key and every value separately, producing an index roughly a third the size of
the default ``jsonb_ops`` with faster containment lookups. It cannot answer
key-existence (``?``) queries — which we never ask.

Everything here is idempotent (``IF NOT EXISTS``) and applied inside one
transaction, so ``ensure_schema`` either produces the complete schema or changes
nothing. DDL is transactional in PostgreSQL; a half-created schema is the one
failure mode worth designing out.
"""

from __future__ import annotations

from typing import Any

import structlog
from pgvector.psycopg import register_vector_async
from psycopg.sql import SQL, Composed, Identifier, Literal
from psycopg_pool import AsyncConnectionPool

from ragorc.core.errors import ConfigError
from ragorc.core.settings import PostgresSettings, get_settings

log = structlog.get_logger(__name__)

Statement = SQL | Composed
"""What ``AsyncConnection.execute()`` accepts: a literal fragment or a composed
one. Deliberately narrower than ``Composable`` — an ``Identifier`` alone is a
fragment, never a statement."""

__all__ = [
    "CHUNK_COLUMNS",
    "CHUNK_READ_COLUMNS",
    "DOCUMENT_COLUMNS",
    "chunks_table",
    "documents_table",
    "drop_statements",
    "ensure_schema",
    "index_statements",
    "optional_statements",
    "schema_statements",
    "table_statements",
]

# (column, pg type) for the write path. This tuple is the single source of truth
# shared with the COPY path in ``store.py``: COPY is positional, so the column
# list and the value tuple must not be able to drift apart. Generated and
# defaulted columns are deliberately absent — ``content_tsv`` is computed by the
# server and must not appear in an INSERT target list at all.
CHUNK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "text"),
    ("document_id", "text"),
    ("content", "text"),
    ("index", "int4"),
    ("start_char", "int4"),
    ("end_char", "int4"),
    ("level", "int4"),
    ("parent_id", "text"),
    ("modality", "text"),
    ("token_count", "int4"),
    ("tenant_id", "text"),
    ("metadata", "jsonb"),
    ("embedding", "vector"),
    ("created_at", "timestamptz"),
)

DOCUMENT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("id", "text"),
    ("source", "text"),
    ("title", "text"),
    ("checksum", "text"),
    ("tenant_id", "text"),
    ("metadata", "jsonb"),
    ("created_at", "timestamptz"),
)

# Read path. ``embedding`` is excluded on purpose: a 384-float column is ~1.5 KB
# per row, so projecting it into a 50-row candidate list ships 75 KB that the
# caller never looks at — the score already encodes everything retrieval needs.
CHUNK_READ_COLUMNS: tuple[str, ...] = (
    "id",
    "document_id",
    "content",
    "index",
    "start_char",
    "end_char",
    "level",
    "parent_id",
    "modality",
    "token_count",
    "tenant_id",
    "metadata",
    "created_at",
)


def chunks_table(settings: PostgresSettings) -> Identifier:
    return Identifier(settings.schema_name, settings.chunks_table)


def documents_table(settings: PostgresSettings) -> Identifier:
    return Identifier(settings.schema_name, settings.documents_table)


def _index_name(table: str, suffix: str) -> Identifier:
    """Index names are schema-global in PostgreSQL, so they are derived from the
    table name — two ragorc schemas in one database must not collide."""
    return Identifier(f"ix_{table}_{suffix}")


def schema_statements(settings: PostgresSettings) -> list[Statement]:
    """The extension and schema everything else depends on. Must not fail."""
    return [
        SQL("CREATE EXTENSION IF NOT EXISTS vector"),
        SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(schema=Identifier(settings.schema_name)),
    ]


def optional_statements(settings: PostgresSettings) -> list[Statement]:
    """Objects whose absence degrades a feature instead of breaking the store.

    ``ensure_schema`` runs each of these inside its own savepoint: a server
    without ParadeDB installed should fall back to ``ts_rank_cd``, not abort the
    transaction that creates the tables. The BM25 index has to come *after* the
    tables, so these are applied last.
    """
    if not settings.use_pg_search:
        return []
    return [
        SQL("CREATE EXTENSION IF NOT EXISTS pg_search"),
        # ParadeDB requires the key field in the index definition; it is what
        # ``paradedb.score(<key>)`` scores against.
        SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON {chunks} "
            "USING bm25 (id, content) WITH (key_field = 'id')"
        ).format(
            name=_index_name(settings.chunks_table, "content_bm25"),
            chunks=chunks_table(settings),
        ),
    ]


def table_statements(settings: PostgresSettings) -> list[Statement]:
    docs = documents_table(settings)
    chunks = chunks_table(settings)
    return [
        SQL(
            """
            CREATE TABLE IF NOT EXISTS {docs} (
                id          text PRIMARY KEY,
                source      text,
                title       text,
                checksum    text,
                tenant_id   text,
                metadata    jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                created_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        ).format(docs=docs),
        SQL(
            """
            CREATE TABLE IF NOT EXISTS {chunks} (
                id           text PRIMARY KEY,
                document_id  text REFERENCES {docs} (id) ON DELETE CASCADE,
                content      text NOT NULL DEFAULT '',
                index        integer NOT NULL DEFAULT 0,
                start_char   integer NOT NULL DEFAULT 0,
                end_char     integer NOT NULL DEFAULT 0,
                level        integer NOT NULL DEFAULT 0,
                parent_id    text,
                modality     text NOT NULL DEFAULT 'text',
                token_count  integer,
                tenant_id    text,
                metadata     jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding    vector({dim}),
                content_tsv  tsvector GENERATED ALWAYS AS (to_tsvector({cfg}, content)) STORED,
                created_at   timestamptz NOT NULL DEFAULT now()
            )
            """
        ).format(
            chunks=chunks,
            docs=docs,
            dim=Literal(int(settings.vector_dimension)),
            cfg=Literal(settings.fulltext_config),
        ),
    ]


def index_statements(settings: PostgresSettings) -> list[Statement]:
    """Every index this store's queries can actually use, and no others.

    ``document_id`` is nullable and the FK is ``ON DELETE CASCADE``: a RAPTOR
    summary or a synthetic proposition chunk has no owning document row, and
    rejecting it at insert time would push the special case into the ingest
    pipeline. A nullable FK still cascades for the chunks that do have one.
    """
    chunks = chunks_table(settings)
    docs = documents_table(settings)
    ct, dt = settings.chunks_table, settings.documents_table
    stmts: list[Statement] = []

    # --- ANN index -------------------------------------------------------
    if settings.vector_index == "hnsw":
        stmts.append(
            SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {chunks} "
                "USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = {m}, ef_construction = {efc})"
            ).format(
                name=_index_name(ct, "embedding_hnsw"),
                chunks=chunks,
                m=Literal(int(settings.hnsw_m)),
                efc=Literal(int(settings.hnsw_ef_construction)),
            )
        )
    elif settings.vector_index == "ivfflat":
        # IVFFlat partitions the space into ``lists`` Voronoi cells at build
        # time; the rule of thumb is rows/1000 up to 1M rows. Building it before
        # the data exists produces useless centroids, so this index is meant to
        # be created *after* a bulk load, not before it.
        stmts.append(
            SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {chunks} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
            ).format(
                name=_index_name(ct, "embedding_ivfflat"),
                chunks=chunks,
                lists=Literal(int(settings.ivf_lists)),
            )
        )

    # --- lexical ---------------------------------------------------------
    stmts.append(
        SQL("CREATE INDEX IF NOT EXISTS {name} ON {chunks} USING gin (content_tsv)").format(
            name=_index_name(ct, "content_tsv"), chunks=chunks
        )
    )
    # --- relational ------------------------------------------------------
    stmts.extend(
        [
            SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {chunks} USING gin (metadata jsonb_path_ops)"
            ).format(name=_index_name(ct, "metadata"), chunks=chunks),
            # (document_id, index) not (document_id): parent expansion and the
            # sentence-window retriever read a document's chunks *in order*, so
            # the composite index turns that into an index-ordered range scan
            # with no sort node.
            SQL("CREATE INDEX IF NOT EXISTS {name} ON {chunks} (document_id, index)").format(
                name=_index_name(ct, "document_index"), chunks=chunks
            ),
            SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {chunks} (tenant_id) "
                "WHERE tenant_id IS NOT NULL"
            ).format(name=_index_name(ct, "tenant"), chunks=chunks),
            SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {chunks} (parent_id) "
                "WHERE parent_id IS NOT NULL"
            ).format(name=_index_name(ct, "parent"), chunks=chunks),
            SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {docs} (tenant_id) "
                "WHERE tenant_id IS NOT NULL"
            ).format(name=_index_name(dt, "tenant"), docs=docs),
            # ``IndexingSettings.skip_unchanged`` compares checksums before
            # embedding; that lookup is the hot path of every re-ingest.
            SQL("CREATE INDEX IF NOT EXISTS {name} ON {docs} (checksum)").format(
                name=_index_name(dt, "checksum"), docs=docs
            ),
        ]
    )
    return stmts


def drop_statements(settings: PostgresSettings) -> list[Statement]:
    """Both tables in one statement so the FK order cannot be wrong."""
    return [
        SQL("DROP TABLE IF EXISTS {chunks}, {docs} CASCADE").format(
            chunks=chunks_table(settings), docs=documents_table(settings)
        )
    ]


async def _assert_vector_width(conn: Any, settings: PostgresSettings) -> None:
    """Refuse to proceed when the live column is not the width we are configured for.

    ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already exists, so
    a corpus indexed at 384 dimensions and then re-pointed at a 768-dimensional
    model kept its ``vector(384)`` column and failed thousands of writes later with
    ``query vector dimension mismatch (got=768 expected=384)`` — a message that
    names neither the setting nor the model that disagree.

    That was reachable from the *documented* path: ``dense_dimension`` is
    "auto-detected from the model when left unset", and the lockstep assignment in
    ``Settings.model_post_init`` only runs when it is set, so changing
    ``dense_model`` alone — which its own docstring recommends, by name, twice —
    moved Qdrant and left pgvector behind.

    pgvector encodes the declared dimension directly in ``atttypmod``, so this is
    one catalog read inside the DDL transaction. An absent column means a fresh
    database mid-create, which is not a mismatch.
    """
    row = await (
        await conn.execute(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
              AND a.attname = 'embedding' AND NOT a.attisdropped
            """,
            (settings.schema_name, settings.chunks_table),
        )
    ).fetchone()
    if row is None:
        return
    actual = int(row[0] if not isinstance(row, dict) else row["atttypmod"])
    wanted = int(settings.vector_dimension)
    if actual > 0 and actual != wanted:
        raise ConfigError(
            "the chunks table's embedding column is not the configured width",
            table=f"{settings.schema_name}.{settings.chunks_table}",
            column_dimension=actual,
            configured_dimension=wanted,
            hint=(
                "set embedding.dense_dimension (and so postgres.vector_dimension) to "
                f"{actual}, or re-index into a new postgres.chunks_table at {wanted}"
            ),
        )


async def ensure_schema(
    pool: AsyncConnectionPool,
    settings: PostgresSettings | None = None,
    *,
    drop: bool = False,
) -> None:
    """Apply the schema idempotently.

    Everything required shares one transaction: DDL is transactional in
    PostgreSQL, so the outcome is the complete schema or no change at all — a
    half-created schema is the failure mode worth designing out. The optional
    ParadeDB objects each get their own savepoint so that a server without the
    extension degrades to ``ts_rank_cd`` rather than failing the migration.
    """
    settings = settings or get_settings().postgres

    # `build_pool` returns an *unopened* pool — psycopg_pool 3.2 deprecated
    # opening in the constructor — so a caller who does the obvious thing
    # (`ensure_schema(build_pool(settings))`) otherwise gets "the pool is not open
    # yet" from three frames down, which names neither the cause nor the fix.
    # Opening here costs a flag check on an already-open pool and removes the trap.
    if pool.closed:
        await pool.open(wait=False)

    async with pool.connection() as conn, conn.transaction():
        for stmt in schema_statements(settings):
            await conn.execute(stmt)

        # The pool's configure() callback ran before CREATE EXTENSION on a virgin
        # database, so this connection has no cached ``vector`` OID yet.
        await register_vector_async(conn)

        if drop:
            for stmt in drop_statements(settings):
                await conn.execute(stmt)
        for stmt in table_statements(settings):
            await conn.execute(stmt)
        for stmt in index_statements(settings):
            await conn.execute(stmt)

        await _assert_vector_width(conn, settings)

        for stmt in optional_statements(settings):
            try:
                async with conn.transaction():
                    await conn.execute(stmt)
            except Exception as exc:  # noqa: BLE001 - the point is to degrade
                log.warning(
                    "optional_object_unavailable",
                    error=str(exc),
                    hint="install ParadeDB or set postgres.use_pg_search=false",
                )

    log.info(
        "postgres_schema_ready",
        schema=settings.schema_name,
        chunks=settings.chunks_table,
        documents=settings.documents_table,
        dimension=settings.vector_dimension,
        vector_index=settings.vector_index,
        dropped=drop,
    )
