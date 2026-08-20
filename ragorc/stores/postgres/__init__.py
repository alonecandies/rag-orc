"""Postgres backend: pgvector store, full-text index and Text-to-SQL target.

The three roles share one table because they share one set of rows — see
:mod:`ragorc.stores.postgres.store` for why fusing them server-side beats
running them as separate services, and :mod:`ragorc.stores.postgres.ddl` for the
schema that makes it possible.

The submodules are layered so that each can be used on its own: ``pool`` knows
nothing about our schema, ``ddl`` knows nothing about queries, ``introspect``
knows nothing about chunks, and ``store`` composes all three.
"""

from __future__ import annotations

from ragorc.stores.postgres.ddl import (
    CHUNK_COLUMNS,
    CHUNK_READ_COLUMNS,
    DOCUMENT_COLUMNS,
    chunks_table,
    documents_table,
    ensure_schema,
    index_statements,
    table_statements,
)
from ragorc.stores.postgres.introspect import ColumnInfo, SchemaIntrospector, TableInfo
from ragorc.stores.postgres.pool import (
    BinaryAsyncConnection,
    build_pool,
    close_all,
    close_pool,
    open_pool,
)
from ragorc.stores.postgres.store import COPY_MIN_ROWS, PostgresStore

__all__ = [
    "CHUNK_COLUMNS",
    "CHUNK_READ_COLUMNS",
    "COPY_MIN_ROWS",
    "DOCUMENT_COLUMNS",
    "BinaryAsyncConnection",
    "ColumnInfo",
    "PostgresStore",
    "SchemaIntrospector",
    "TableInfo",
    "build_pool",
    "chunks_table",
    "close_all",
    "close_pool",
    "documents_table",
    "ensure_schema",
    "index_statements",
    "open_pool",
    "table_statements",
]
