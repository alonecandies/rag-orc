"""Async connection pooling for Postgres.

Why pool at all
---------------
A single RAG query issues three to five statements against Postgres (schema
introspection, a Text-to-SQL execution, a vector search, a full-text search).
Establishing a fresh connection costs a TCP handshake plus a TLS handshake plus
``authenticate + set up backend`` — on the order of 5-15 ms locally and 30-80 ms
across an availability zone. An HNSW probe over a million rows takes ~2 ms. Not
pooling would make connection setup 10-40x the cost of the work it enables, so
the pool is not an optimization here, it is the difference between a viable and
an unviable design.

Why the binary protocol matters
-------------------------------
This store moves float arrays and timestamps, which are exactly the two types
where PostgreSQL's text protocol is worst:

* A 384-dim ``float32`` vector is 1536 bytes on the wire in binary. Rendered as
  ``'[0.0123456,-0.98765,...]'`` it is 4-6 KB *and* both ends pay a
  float-to-decimal / decimal-to-float conversion per component — ~2300 `strtod`
  calls for one 200-row result set with an embedding column.
* ``timestamptz`` is 8 bytes binary versus a formatted string that has to be
  parsed back through a calendar routine.

So ``binary`` is on by default (``PostgresSettings.binary``). Since psycopg
selects the result format *per cursor*, this is implemented as a connection
subclass whose ``cursor()`` defaults to the binary format — ``conn.execute()``
goes through ``cursor()`` too, so one override covers every call site.

Why binary is *not* used for Text-to-SQL
----------------------------------------
psycopg ships binary loaders for the types it knows, and falls back to handing
back raw ``bytes`` for the ones it does not (``regtype``, ``money``, ``xid``,
``macaddr``, ``bit``, ``tsvector``, ``xml``, geometric types, …). Our own
queries have a fixed, known column list, so binary is always decodable. A
generated ``SELECT`` can project anything, so :meth:`PostgresStore.execute_readonly`
asks for a text-format cursor explicitly: the text protocol is the universally
decodable one, and one Text-to-SQL result per query is not where latency lives.

Why a second, read-only pool
----------------------------
Defence in depth. Text-to-SQL is a remote-code-execution primitive and the SQL
guard is a *parser* — it can only reject constructs it knows about. Three
independent barriers sit in front of a write:

1. the guard rejects anything that is not ``SELECT``/``WITH``,
2. this layer runs the statement in a ``READ ONLY`` transaction,
3. when ``readonly_dsn`` is set, the connection belongs to a role with no write
   grants at all — a novel bypass of (1) and (2) still cannot mutate a row.

If ``readonly_dsn`` is empty there is no second role to connect as, so the
read-only lookup returns the primary pool and barrier (2) carries the weight
alone; spinning up a duplicate pool against the same DSN would double the
connection budget and buy nothing.

Why ``configure`` and not the DSN
---------------------------------
``statement_timeout`` could be smuggled into the connection string via
``options=-c ...``, but pgvector's type OIDs cannot: they are assigned per
database at ``CREATE EXTENSION`` time and psycopg caches type adapters per
connection, so *every* connection has to look them up. Doing both in one
``configure`` callback keeps session setup in a single place and makes the pool
self-healing — a connection handed out by the pool always knows how to dump a
numpy array into a ``vector`` column and always has a server-side deadline.

Why the pool is module-cached
-----------------------------
A pool owns sockets and background maintenance tasks. ``PostgresStore``,
``SchemaIntrospector`` and the ingest writer are separate objects that all want
Postgres; if each built its own pool, ``max_pool_size`` would silently multiply
by the number of components and exhaust ``max_connections``. Pools are keyed by
DSN, created lazily, opened without blocking, and torn down by
:func:`close_all` at shutdown.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import psycopg
import structlog
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection, AsyncCursor, AsyncServerCursor
from psycopg.rows import AsyncRowFactory, dict_row
from psycopg.sql import SQL, Literal
from psycopg_pool import AsyncConnectionPool, PoolClosed, PoolTimeout

from ragorc.core.errors import StoreUnavailable
from ragorc.core.settings import PostgresSettings, get_settings

log = structlog.get_logger(__name__)

__all__ = [
    "BinaryAsyncConnection",
    "build_pool",
    "close_all",
    "close_pool",
    "open_pool",
    "resolve_dsn",
]

# Keyed by the DSN actually connected to, so a readonly lookup that falls back
# to the primary DSN reuses the primary pool instead of cloning it.
_POOLS: dict[str, AsyncConnectionPool] = {}


class BinaryAsyncConnection(AsyncConnection[Any]):
    """An :class:`AsyncConnection` whose cursors request binary results.

    psycopg decides the result format per cursor (``cur.format``), not per
    connection, and ``AsyncConnection.execute()`` builds its cursor through
    ``self.cursor()`` — so overriding that one method flips the whole connection
    to binary while leaving ``cursor(binary=False)`` available for the
    Text-to-SQL path, which needs the text protocol's universal decodability.
    """

    # The base method is a four-way overload over (named cursor?) x (row factory
    # given?); collapsing it into one signature is the point of the override, so
    # the variance complaint is expected rather than a defect.
    def cursor(  # type: ignore[override]
        self,
        name: str = "",
        *,
        binary: bool = True,
        row_factory: AsyncRowFactory[Any] | None = None,
        scrollable: bool | None = None,
        withhold: bool = False,
    ) -> AsyncCursor[Any] | AsyncServerCursor[Any]:
        return super().cursor(
            name,
            binary=binary,
            # Falling back to the connection's own factory is what the base
            # method does for a missing argument; doing it here keeps the
            # signature non-optional for the call below.
            row_factory=row_factory if row_factory is not None else self.row_factory,
            scrollable=scrollable,
            withhold=withhold,
        )


def resolve_dsn(settings: PostgresSettings, *, readonly: bool = False) -> str:
    """The DSN a pool should connect to.

    ``readonly`` degrades to the primary DSN when no restricted role has been
    configured; the caller still gets the per-transaction ``READ ONLY`` barrier.
    """
    if readonly and settings.readonly_dsn.get_secret_value():
        return settings.readonly_dsn.get_secret_value()
    return settings.dsn.get_secret_value()


def _make_configure(
    settings: PostgresSettings, *, readonly: bool
) -> Callable[[AsyncConnection[Any]], Coroutine[Any, Any, None]]:
    """Build the per-connection session-setup callback.

    Ordering is deliberate:

    1. ``CREATE EXTENSION IF NOT EXISTS vector`` — on an existing extension this
       is a catalog lookup and a NOTICE, needing no privileges, so it is cheap
       insurance against the race where the pool fills before
       :func:`ragorc.stores.postgres.ddl.ensure_schema` has run. Skipped on a
       read-only pool, where any DDL node is rejected before it executes.
    2. register pgvector's type OIDs, inside a savepoint so a database without
       the extension degrades to "no vector search" instead of "no Postgres".
    3. ``SET statement_timeout`` so a pathological plan cannot pin a backend.
    4. ``SET default_transaction_read_only`` on the read-only pool.

    The callback must leave the connection idle: psycopg_pool discards any
    connection that ``configure`` leaves inside a transaction, so it commits.
    """

    async def configure(conn: AsyncConnection[Any]) -> None:
        if not readonly:
            try:
                async with conn.transaction():
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.Error as exc:
                # Insufficient privilege, or no pgvector on the server. Vector
                # search will fail loudly later; everything else still works.
                log.warning(
                    "pgvector_extension_unavailable",
                    error=str(exc),
                    hint="run CREATE EXTENSION vector as a superuser",
                )
        try:
            async with conn.transaction():
                await register_vector_async(conn)
        except psycopg.Error as exc:
            log.warning(
                "pgvector_types_unregistered",
                error=str(exc),
                hint="ensure_schema() creates the extension; vector columns stay unusable until then",
            )

        await conn.execute(
            SQL("SET statement_timeout = {ms}").format(
                ms=Literal(int(settings.statement_timeout_ms))
            )
        )
        if readonly:
            await conn.execute("SET default_transaction_read_only = on")
        await conn.commit()

    return configure


def build_pool(
    settings: PostgresSettings | None = None, *, readonly: bool = False
) -> AsyncConnectionPool:
    """Return the process-wide pool for this DSN, creating it if needed.

    The pool is created **closed**: psycopg_pool deprecates connecting from the
    constructor because it hides connection errors in a background task and ties
    the pool to whichever event loop happened to be current. Call
    :func:`open_pool` (or ``await pool.open()``) to start it.
    """
    settings = settings or get_settings().postgres
    dsn = resolve_dsn(settings, readonly=readonly)
    existing = _POOLS.get(dsn)
    if existing is not None:
        return existing

    # ``prepare_threshold`` promotes a statement to a server-side prepared plan
    # after N executions: the retrieval queries are issued thousands of times
    # with different parameters, so parse+plan happens once per connection
    # instead of once per query. ``dict_row`` everywhere keeps row handling
    # uniform and makes ``execute_readonly`` a direct pass-through.
    kwargs: dict[str, Any] = {
        "prepare_threshold": settings.prepare_threshold,
        "row_factory": dict_row,
    }

    pool = AsyncConnectionPool(
        dsn,
        connection_class=BinaryAsyncConnection if settings.binary else AsyncConnection,
        kwargs=kwargs,
        min_size=settings.min_pool_size,
        max_size=settings.max_pool_size,
        timeout=settings.timeout_s,
        max_idle=settings.max_idle_s,
        configure=_make_configure(settings, readonly=readonly),
        name=f"ragorc-pg-ro-{len(_POOLS)}" if readonly else f"ragorc-pg-{len(_POOLS)}",
        open=False,
    )
    _POOLS[dsn] = pool
    log.info(
        "pg_pool_created",
        readonly=readonly,
        min_size=settings.min_pool_size,
        max_size=settings.max_pool_size,
        binary=settings.binary,
    )
    return pool


async def open_pool(
    settings: PostgresSettings | None = None, *, readonly: bool = False
) -> AsyncConnectionPool:
    """Get an *open* pool, opening it on first use.

    ``open(wait=False)`` returns immediately and lets the pool's workers fill it
    in the background: a store that is constructed but never queried should not
    pay for connections, and a store that is queried immediately blocks in
    ``pool.connection()`` for at most ``timeout_s`` anyway. Waiting here would
    turn a cold start into a serial round of handshakes.

    This is on the hot path — every query calls it — so an already-open pool
    short-circuits before ``open()``, which would otherwise take the pool's lock
    on every request just to discover there is nothing to do.
    """
    settings = settings or get_settings().postgres
    pool = build_pool(settings, readonly=readonly)
    if not pool.closed:
        return pool
    try:
        await pool.open(wait=False)
    except PoolClosed:
        # psycopg_pool cannot reopen a pool that was opened and then closed, and
        # ``pool.closed`` is also True for a pool that has merely never been
        # opened — so the distinction is made by attempting the open, not by
        # inspecting the flag. Replace it, so a process that shut its stores down
        # and started them again keeps working.
        _POOLS.pop(resolve_dsn(settings, readonly=readonly), None)
        pool = build_pool(settings, readonly=readonly)
        await pool.open(wait=False)
    except (psycopg.OperationalError, PoolTimeout, OSError) as exc:
        raise StoreUnavailable("postgres", f"cannot open connection pool: {exc}") from exc
    return pool


async def close_pool(settings: PostgresSettings | None = None, *, readonly: bool = False) -> None:
    """Close and forget one pool. Idempotent."""
    settings = settings or get_settings().postgres
    dsn = resolve_dsn(settings, readonly=readonly)
    pool = _POOLS.pop(dsn, None)
    if pool is None:
        return
    await pool.close()
    log.info("pg_pool_closed", readonly=readonly)


async def close_all() -> None:
    """Shutdown hook: close every cached pool.

    Called from application shutdown. Without it, the pool's maintenance tasks
    keep the event loop alive and PostgreSQL keeps the backends until they hit
    ``max_idle``.
    """
    pools = list(_POOLS.values())
    _POOLS.clear()
    for pool in pools:
        try:
            await pool.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            log.warning("pg_pool_close_failed", error=str(exc))
    if pools:
        log.info("pg_pools_closed", count=len(pools))
