"""Neo4j driver construction and the process-wide pool cache.

Why the driver is cached at module scope
---------------------------------------
``AsyncGraphDatabase.driver()`` does not open a connection — it builds a
*connection pool*, a routing table, a background liveness checker and a TLS
context. Creating one per request (or per store instance, or per FastAPI
dependency) is the most common and most expensive Neo4j mistake there is:
every request then pays a fresh TCP + TLS + Bolt handshake plus a ``ROUTE``
round trip, sockets accumulate until the server's Bolt thread pool is
exhausted, and the pool's entire reason for existing — handing back an already
warm, already authenticated connection — is defeated. A driver is designed to
be created once and shared by the whole process; it is task- and thread-safe.
So drivers are memoised here, keyed by the parameters that actually change the
pool's identity.

The cache is guarded by a ``threading.Lock``, not an ``asyncio.Lock``:
constructing a driver is pure CPU with no ``await``, so the critical section is
microseconds long, and a plain lock has no event-loop affinity — which matters
because workers and test suites routinely create and discard event loops around
a driver that outlives them.

Three deadlines, three different meanings
-----------------------------------------
* ``connection_timeout`` — the TCP/TLS connect budget for one socket.
* ``connection_acquisition_timeout`` — how long a task waits for a *free*
  connection from a saturated pool. Pinned to the connect budget: a request
  that cannot get a socket in the time it takes to open a new one is better off
  failing fast than queueing behind fifty others and blowing the request SLA.
* ``max_transaction_retry_time`` — the driver's own budget for retrying a
  managed transaction (leader switch, deadlock, transient routing failure).
  This is the one that turns a cluster failover into a hiccup instead of an
  error, and it is why every statement in this package goes through
  ``execute_query`` rather than a hand-rolled session.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import TYPE_CHECKING, Any

import structlog
from neo4j import AsyncGraphDatabase, basic_auth
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)

from ragorc.core.errors import ConfigError, StoreUnavailable
from ragorc.core.ids import cache_key
from ragorc.core.settings import Neo4jSettings, Settings, get_settings

if TYPE_CHECKING:
    from neo4j import AsyncDriver

log = structlog.get_logger(__name__)

__all__ = [
    "build_driver",
    "close_all_drivers",
    "close_driver",
    "driver_cache_key",
    "verify_connectivity",
]

_DRIVERS: dict[str, AsyncDriver] = {}
_LOCK = threading.Lock()


def _resolve(settings: Settings | Neo4jSettings | None) -> tuple[Neo4jSettings, str]:
    """Accept the root settings, the Neo4j subtree, or nothing at all."""
    if settings is None:
        settings = get_settings()
    if isinstance(settings, Neo4jSettings):
        return settings, "ragorc"
    return settings.neo4j, settings.app_name


def driver_cache_key(cfg: Neo4jSettings) -> str:
    """Identity of a pool: everything that would make two pools incompatible.

    The password is hashed into the key by :func:`cache_key`, never stored in
    it — the key ends up in log lines and cache dumps.
    """
    return cache_key(
        "neo4j-driver",
        cfg.uri,
        cfg.user,
        cfg.password.get_secret_value(),
        cfg.max_connection_pool_size,
        cfg.connection_timeout_s,
        cfg.max_transaction_retry_time_s,
        cfg.fetch_size,
    )


def _construct(cfg: Neo4jSettings, app_name: str) -> AsyncDriver:
    password = cfg.password.get_secret_value()
    # An empty password means the server runs with authentication disabled;
    # sending basic auth with an empty secret fails with a confusing message.
    auth = basic_auth(cfg.user, password) if password else None
    try:
        return AsyncGraphDatabase.driver(
            cfg.uri,
            auth=auth,
            max_connection_pool_size=cfg.max_connection_pool_size,
            connection_timeout=cfg.connection_timeout_s,
            connection_acquisition_timeout=cfg.connection_timeout_s,
            max_transaction_retry_time=cfg.max_transaction_retry_time_s,
            # Default page size for streamed results. The graph queries in this
            # package are all bounded, so this only matters for introspection.
            fetch_size=cfg.fetch_size,
            # Shows up in `SHOW TRANSACTIONS`, which is how you find out whose
            # query is pinning a core when the graph gets slow.
            user_agent=app_name,
        )
    except (ConfigurationError, ValueError) as exc:
        # A bad URI scheme or a nonsensical timeout is a startup problem, not a
        # request problem: fail loudly and immediately.
        raise ConfigError(f"invalid Neo4j configuration: {exc}", uri=cfg.uri) from exc


def build_driver(
    settings: Settings | Neo4jSettings | None = None, *, cached: bool = True
) -> AsyncDriver:
    """Return the shared :class:`neo4j.AsyncDriver` for these settings.

    Synchronous on purpose: no I/O happens here, so this is safe to call from
    ``__init__`` and there is no async constructor to await. Pass
    ``cached=False`` only when you genuinely want an isolated pool (a migration
    job, a test that closes the driver underneath itself).
    """
    cfg, app_name = _resolve(settings)
    if not cached:
        return _construct(cfg, app_name)

    key = driver_cache_key(cfg)
    # Construct *inside* the lock: it costs microseconds and removes the
    # otherwise unavoidable "two tasks both built a pool, one leaks" race.
    with _LOCK:
        existing = _DRIVERS.get(key)
        if existing is not None:
            return existing
        driver = _construct(cfg, app_name)
        _DRIVERS[key] = driver
    log.info(
        "neo4j_driver_created",
        uri=cfg.uri,
        database=cfg.database,
        pool_size=cfg.max_connection_pool_size,
        cached_drivers=len(_DRIVERS),
    )
    return driver


async def verify_connectivity(
    driver: AsyncDriver,
    *,
    database: str | None = None,
    # This function *is* the timeout wrapper: the deadline is applied below with
    # asyncio.wait_for, so ASYNC109 has nothing to hoist out to the caller.
    timeout: float | None = None,  # noqa: ASYNC109 - this IS the timeout wrapper
) -> dict[str, Any]:
    """Round-trip the server once and return its identity.

    Called from ``/health`` and from ingest startup. Failure is translated into
    :class:`StoreUnavailable` so the caller can degrade instead of parsing
    driver exceptions, except for bad credentials — that is a configuration
    error and retrying it will never help.
    """
    config: dict[str, Any] = {}
    if database:
        config["database"] = database
    try:
        info = await asyncio.wait_for(driver.get_server_info(**config), timeout)
    except TimeoutError as exc:
        raise StoreUnavailable("neo4j", "connectivity check timed out", timeout_s=timeout) from exc
    except AuthError as exc:
        raise ConfigError(f"Neo4j authentication failed: {exc}") from exc
    except (ServiceUnavailable, SessionExpired) as exc:
        raise StoreUnavailable("neo4j", str(exc)) from exc
    except Neo4jError as exc:
        raise StoreUnavailable("neo4j", str(exc), code=getattr(exc, "code", None)) from exc

    detail = {
        "address": str(info.address),
        "agent": info.agent,
        "protocol_version": ".".join(str(part) for part in info.protocol_version),
    }
    log.info("neo4j_connected", **detail)
    return detail


async def close_driver(driver: AsyncDriver) -> None:
    """Close a driver and evict it from the cache.

    Eviction comes first so a concurrent :func:`build_driver` cannot hand out
    the pool we are in the middle of tearing down.
    """
    with _LOCK:
        for key, cached in list(_DRIVERS.items()):
            if cached is driver:
                del _DRIVERS[key]
    await driver.close()


async def close_all_drivers() -> None:
    """Shutdown hook. Closing is best-effort: a pool that is already broken
    still has to be dropped, and raising here would mask the real error that
    triggered shutdown."""
    with _LOCK:
        drivers = list(_DRIVERS.values())
        _DRIVERS.clear()
    for driver in drivers:
        with contextlib.suppress(Exception):
            await driver.close()
    if drivers:
        log.info("neo4j_drivers_closed", count=len(drivers))
