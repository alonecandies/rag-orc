"""Neo4j graph store: Text-to-Cypher execution, GraphRAG search, path finding.

Everything here is shaped by four properties of Bolt and of Cypher.

**1. Round trips dominate.** A Bolt round trip is ~0.5-2 ms on a LAN; merging
one entity takes microseconds of server time. Writing 10 000 entities with one
statement per entity is therefore ~10 s of pure latency and ~0.05 s of work.
Every write in this module is a single ``UNWIND $rows`` statement: the batch
crosses the wire once, the server loops over it in-process, and the same 10 000
merges land in one round trip. This is the difference between an ingest that
takes minutes and one that takes seconds, and it is the reason the row-building
happens in a thread — with the network cost removed, building the payload
becomes the bottleneck.

**2. A relationship type is part of the query plan, not the data.** Cypher
cannot parameterize it: ``-[:$type]->`` does not exist. So the type has to be
interpolated, which is exactly the shape of an injection. Two safe routes are
implemented — ``apoc.merge.relationship`` when APOC is installed (the type
becomes a real argument), and otherwise one grouped ``UNWIND`` statement per
*distinct* type, which is a handful of round trips because our extraction
schema emits a bounded vocabulary. Either way the type is validated against
``^[A-Z][A-Z0-9_]*$`` first and a mismatch raises
:class:`~ragorc.core.errors.GuardrailViolation`.

**3. A driver value is not JSON.** ``Node``, ``Relationship``, ``Path``,
``neo4j.time.DateTime`` and ``neo4j.spatial.Point`` are all rich Python objects.
Handing one to ``orjson`` raises; handing one to an LLM prompt via ``str()``
leaks internal element ids and burns tokens on ``<Node element_id='4:...'>``.
Results from generated Cypher are therefore flattened into plain dicts before
they leave this module.

**4. ``READ`` routing is not a write guard.** It sends the query to a follower
in a cluster — on a single instance it happily executes ``DETACH DELETE``. So
:meth:`Neo4jStore.execute_readonly` also refuses the statement keywords in
``security.cypher_forbid_keywords`` and, when
``security.cypher_explain_dryrun`` is on, plans the query with ``EXPLAIN``
before running it. User text never reaches the query string: parameters only,
always.

Scores follow the house rule — higher is better. Lucene relevance and
weight-per-hop path scores are both already oriented that way, so no distance
conversion is needed anywhere in this file.
"""

from __future__ import annotations

import asyncio
import functools
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson
import structlog
from neo4j import Query, RoutingControl
from neo4j.exceptions import (
    ClientError,
    ConnectionAcquisitionTimeoutError,
    DatabaseUnavailable,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)
from neo4j.exceptions import TransientError as Neo4jTransientError
from neo4j.graph import Node, Path, Relationship
from neo4j.spatial import Point
from neo4j.time import Duration

from ragorc.core.concurrency import CircuitBreaker, bounded_gather
from ragorc.core.errors import (
    ConstructionError,
    GuardrailViolation,
    StoreUnavailable,
    TransientError,
)
from ragorc.core.models import Community, Entity, GraphPath, Relation
from ragorc.core.settings import Settings, get_settings
from ragorc.stores.neo4j.driver import build_driver, close_driver, verify_connectivity
from ragorc.stores.neo4j.schema import (
    GraphSchemaIntrospector,
    ensure_schema,
    fulltext_index_name,
    validate_label,
    validate_rel_type,
)

if TYPE_CHECKING:
    from neo4j import AsyncDriver, AsyncResult, Record

log = structlog.get_logger(__name__)

__all__ = ["Neo4jStore"]

# Fixed relationship types. Literals in the query text, never user-supplied.
_MENTIONS = "MENTIONS"
_IN_COMMUNITY = "IN_COMMUNITY"

_SEMANTIC_EDGE = f":!{_MENTIONS}&!{_IN_COMMUNITY}"
"""Every relationship type *except* the library's own two structural ones.

Entity-to-entity types are authored by the extraction model, so they cannot be
enumerated and the filter has to be a negation. The two excluded types are
scaffolding rather than extracted knowledge:

``MENTIONS``      ``(:Chunk)-[:MENTIONS]->(:Entity)``
``IN_COMMUNITY``  ``(:Entity)-[:IN_COMMUNITY]->(:Community)``

Written into quantified path patterns, and the **node labels in those patterns
are what actually stops the traversal walking through a chunk** — an intermediate
node has to be an ``Entity``, and neither structural type joins two entities. A
plain ``-[r*1..2]-`` constrains only the *endpoint*, which is why the untyped
version let a two-hop expansion return ``A -[:MENTIONS]- chunk -[:MENTIONS]- B``.
That is not a weaker answer but a different claim — "A and B appear in the same
passage" is co-occurrence, and it was being returned as an extracted
relationship, with ``startNode(r)`` on that edge being a ``Chunk`` that
``_entity_from_props`` then built an "entity" out of.

So the type expression is, against today's schema, redundant with the labels. It
is kept because the redundancy is free and the assumption behind it is one edge
type away from being wrong: a resolution edge between entities (``SAME_AS``,
``ALIAS_OF``) is a plausible addition, and it would be structural too. Stating
"only extracted relationships" here means a new scaffolding type cannot silently
start appearing in traversal results.

Quantified path patterns require **Neo4j 5.9+** (the compose stack pins 5.26).
The portable alternative, ``WHERE none(r IN rels ...)``, filters *after*
expanding: on an entity that ten thousand chunks mention, a two-hop expansion
materializes all of them before discarding them. ``MENTIONS`` is the densest
edge type in this schema, so that is the one place the difference is not
academic."""

# The store is down / unreachable. Trips the breaker and degrades the query.
_UNAVAILABLE = (
    ServiceUnavailable,
    SessionExpired,
    DatabaseUnavailable,
    ConnectionAcquisitionTimeoutError,
)

_CYPHER_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_CYPHER_LITERAL_RE = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", re.DOTALL)
_LEADING_PLAN_RE = re.compile(r"^\s*(?:EXPLAIN|PROFILE)\s+", re.IGNORECASE)

# Lucene's query language is not Cypher. An unescaped ``?``, ``(`` or ``:`` in
# a natural-language question is a parse error inside the full-text index, so
# question text is escaped down to plain terms before it is handed to Lucene.
_LUCENE_SPECIAL_RE = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')


@functools.lru_cache(maxsize=256)
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    """Word-boundary matcher for a forbidden keyword, multi-word aware.

    Compiled once per keyword: the guard runs on every generated query and
    ``re.compile`` is not free.
    """
    body = r"\s+".join(re.escape(part) for part in keyword.split())
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


def _lucene_escape(text: str) -> str:
    return _LUCENE_SPECIAL_RE.sub(r"\\\1", text.strip())


def _strip_noise(cypher: str) -> str:
    """Remove comments and string literals before keyword matching.

    Without this, a query that merely *mentions* a forbidden word inside a
    quoted value is rejected, and a comment can be used to hide one.
    """
    return _CYPHER_LITERAL_RE.sub("''", _CYPHER_COMMENT_RE.sub(" ", cypher))


def _dedupe(values: Iterable[Any]) -> list[str]:
    """Order-preserving de-duplication of a string list."""
    seen: dict[str, None] = {}
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Cypher fragment builders
#
# These exist so the accumulate-on-repeat-assertion semantics are written once
# and are provably identical for entities and relationships.
# ---------------------------------------------------------------------------
def _accumulate_text(target: str, field: str = "description") -> str:
    """Append ``row.<field>`` to ``<target>.<field>``, skipping a repeat.

    The ``CONTAINS`` arm is what stops a description growing without bound
    across re-ingests of the same corpus.
    """
    return (
        f"CASE WHEN row.{field} = '' THEN {target}.{field} "
        f"WHEN coalesce({target}.{field}, '') = '' THEN row.{field} "
        f"WHEN {target}.{field} CONTAINS row.{field} THEN {target}.{field} "
        f"ELSE {target}.{field} + '\\n' + row.{field} END"
    )


def _accumulate_list(target: str, field: str) -> str:
    """Union two lists, de-duplicated.

    Cypher has no set type and no list-level ``DISTINCT``, so the idiom is a
    ``reduce``. It is O(n*m), which is irrelevant for the handful of aliases or
    chunk ids an entity carries and would be the wrong tool for anything larger.
    """
    return (
        f"reduce(acc = [], x IN coalesce({target}.{field}, []) + row.{field} | "
        "CASE WHEN x IN acc THEN acc ELSE acc + x END)"
    )


def _accumulate_weight(target: str) -> str:
    """Add ``row.weight`` unless this is a re-assertion from known chunks.

    An edge asserted by many documents must outrank one asserted once, so
    weight accumulates. But re-ingesting the *same* chunk must not inflate it,
    and chunk ids are content-derived, so they are the idempotency key. This
    expression must be evaluated before ``source_chunk_ids`` is updated — SET
    items apply in order, so weight comes first in every SET clause below.
    """
    return (
        "CASE WHEN size(row.source_chunk_ids) > 0 AND all(x IN row.source_chunk_ids "
        f"WHERE x IN coalesce({target}.source_chunk_ids, [])) "
        f"THEN coalesce({target}.weight, 0.0) "
        f"ELSE coalesce({target}.weight, 0.0) + row.weight END"
    )


# ---------------------------------------------------------------------------
# Value serialization
# ---------------------------------------------------------------------------
def _serialize(value: Any) -> Any:
    """Flatten any driver value into something ``orjson`` can encode.

    Graph objects keep their identity under underscore-prefixed keys so a
    Text-to-Cypher answer can still cite a node, while the properties stay at
    the top level where a model will actually read them.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Node):
        return {
            "_element_id": value.element_id,
            "_labels": sorted(value.labels),
            **{key: _serialize(item) for key, item in value.items()},
        }
    if isinstance(value, Relationship):
        start = value.start_node
        end = value.end_node
        return {
            "_element_id": value.element_id,
            "_type": value.type,
            "_start": None if start is None else start.element_id,
            "_end": None if end is None else end.element_id,
            **{key: _serialize(item) for key, item in value.items()},
        }
    if isinstance(value, Path):
        return {
            "_nodes": [_serialize(node) for node in value.nodes],
            "_relationships": [_serialize(rel) for rel in value.relationships],
            "_length": len(value.relationships),
        }
    if isinstance(value, Point):
        # Point subclasses tuple, so it has to be handled before sequences.
        return {"srid": value.srid, "coordinates": list(value)}
    if isinstance(value, Duration):
        return str(value)  # ISO-8601 duration; no isoformat() on this type
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):  # date / time / datetime, native and neo4j.time
        return isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serialize(item) for item in value]
    return str(value)


def _serialize_record(record: Record) -> dict[str, Any]:
    return {key: _serialize(value) for key, value in record.items()}


# ---------------------------------------------------------------------------
# Row builders (CPU-bound; run in a thread)
# ---------------------------------------------------------------------------
def _entity_rows(entities: Sequence[Entity]) -> list[dict[str, Any]]:
    """Collapse entities to one row per exact name.

    Duplicates within a single ``UNWIND`` batch would each apply the
    accumulate-on-match branch, double-counting a description that arrived
    twice in the same call. Merging here keeps the statement idempotent per
    batch. Case-insensitive merging is *not* done: that is entity resolution
    (``graph.resolve_entities``) and it happens before the store sees anything.
    """
    merged: dict[str, dict[str, Any]] = {}
    for entity in entities:
        name = entity.name.strip()
        if not name:
            continue
        row = merged.get(name)
        if row is None:
            embedding = entity.embedding
            merged[name] = {
                "name": name,
                "type": entity.type or "Entity",
                "description": entity.description or "",
                "aliases": _dedupe(entity.aliases),
                "source_chunk_ids": _dedupe(entity.source_chunk_ids),
                "degree": int(entity.degree),
                "community_id": entity.community_id,
                # tolist() is a C-level conversion of the whole buffer, not a
                # Python loop over 384 floats.
                "embedding": (
                    None
                    if embedding is None
                    else np.asarray(embedding, dtype=np.float32).ravel().tolist()
                ),
                # Neo4j properties are scalars or arrays of scalars — a nested
                # map has to be serialized to survive the round trip.
                "metadata_json": (
                    orjson.dumps(entity.metadata).decode() if entity.metadata else None
                ),
            }
            continue
        if entity.description and entity.description not in row["description"]:
            row["description"] = (
                f"{row['description']}\n{entity.description}"
                if row["description"]
                else entity.description
            )
        row["aliases"] = _dedupe([*row["aliases"], *entity.aliases])
        row["source_chunk_ids"] = _dedupe([*row["source_chunk_ids"], *entity.source_chunk_ids])
        row["degree"] = max(row["degree"], int(entity.degree))
        if row["community_id"] is None:
            row["community_id"] = entity.community_id
        if row["embedding"] is None and entity.embedding is not None:
            row["embedding"] = np.asarray(entity.embedding, dtype=np.float32).ravel().tolist()
    return list(merged.values())


def _relation_rows(relations: Sequence[Relation]) -> dict[str, list[dict[str, Any]]]:
    """Group relations by validated type, merging duplicate edges.

    Grouping is the non-APOC execution plan: one statement per distinct type.
    Merging duplicates within a group means weights sum once, in Python, rather
    than through repeated ``SET`` round trips on the same edge.
    """
    grouped: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for relation in relations:
        source = relation.source.strip()
        target = relation.target.strip()
        if not source or not target:
            continue
        rel_type = validate_rel_type((relation.type or "RELATED_TO").strip().upper())
        bucket = grouped.setdefault(rel_type, {})
        key = (source, target)
        row = bucket.get(key)
        if row is None:
            bucket[key] = {
                "source": source,
                "target": target,
                "type": rel_type,
                "description": relation.description or "",
                "weight": float(relation.weight),
                "source_chunk_ids": _dedupe(relation.source_chunk_ids),
                "metadata_json": (
                    orjson.dumps(relation.metadata).decode() if relation.metadata else None
                ),
            }
            continue
        row["weight"] += float(relation.weight)
        if relation.description and relation.description not in row["description"]:
            row["description"] = (
                f"{row['description']}\n{relation.description}"
                if row["description"]
                else relation.description
            )
        row["source_chunk_ids"] = _dedupe([*row["source_chunk_ids"], *relation.source_chunk_ids])
    return {rel_type: list(rows.values()) for rel_type, rows in grouped.items()}


def _community_rows(communities: Sequence[Community]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for community in communities:
        embedding = community.embedding
        rows.append(
            {
                "id": int(community.id),
                "level": int(community.level),
                "title": community.title or "",
                "summary": community.summary or "",
                "rank": float(community.rank),
                "parent_id": community.parent_id,
                "entity_names": _dedupe(community.entity_names),
                # Relation keys are triples; a list of lists is not a legal
                # Neo4j property, so they travel as JSON.
                "relation_keys_json": (
                    orjson.dumps([list(key) for key in community.relation_keys]).decode()
                    if community.relation_keys
                    else None
                ),
                "embedding": (
                    None
                    if embedding is None
                    else np.asarray(embedding, dtype=np.float32).ravel().tolist()
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Row -> model
# ---------------------------------------------------------------------------
def _metadata_from_json(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        loaded = orjson.loads(raw)
    except orjson.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _entity_from_props(props: Mapping[str, Any]) -> Entity:
    """Build an :class:`Entity` from node properties.

    ``embedding`` is deliberately not read back. It is index payload; shipping
    384 floats per entity into a prompt path costs bandwidth and buys nothing.
    """
    return Entity(
        name=str(props.get("name") or ""),
        type=str(props.get("type") or "Entity"),
        description=str(props.get("description") or ""),
        aliases=tuple(str(a) for a in (props.get("aliases") or ())),
        source_chunk_ids=tuple(str(c) for c in (props.get("source_chunk_ids") or ())),
        degree=int(props.get("degree") or 0),
        community_id=props.get("community_id"),
        metadata=_metadata_from_json(props.get("metadata_json")),
    )


def _relation_from_props(props: Mapping[str, Any]) -> Relation:
    return Relation(
        source=str(props.get("source") or ""),
        target=str(props.get("target") or ""),
        type=str(props.get("type") or "RELATED_TO"),
        description=str(props.get("description") or ""),
        weight=float(props.get("weight") or 1.0),
        source_chunk_ids=tuple(str(c) for c in (props.get("source_chunk_ids") or ())),
        metadata=_metadata_from_json(props.get("metadata_json")),
    )


class Neo4jStore:
    """A :class:`~ragorc.core.protocols.GraphStore` over Neo4j 5.

    The driver is shared process-wide (see :mod:`ragorc.stores.neo4j.driver`);
    constructing a store is free and does no I/O, so it is safe to build one
    per request if that is what your dependency injection wants.
    """

    def __init__(
        self,
        driver: AsyncDriver | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.neo4j
        self._external_driver = driver is not None
        self.driver = driver or build_driver(self.settings)
        # Validated exactly once, here. Every label interpolated into Cypher
        # below comes from one of these three attributes, so the injection
        # surface of this module is closed at construction time.
        self.node_label = validate_label(self.cfg.node_label)
        self.chunk_label = validate_label(self.cfg.chunk_label)
        self.community_label = validate_label(self.cfg.community_label)
        self.breaker = CircuitBreaker("neo4j")
        self._introspector = GraphSchemaIntrospector(self.driver, settings=self.settings)
        self._has_apoc: bool | None = None
        self._apoc_lock: asyncio.Lock | None = None

    # -- plumbing ---------------------------------------------------------
    async def _run(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        write: bool = False,
        transformer: Callable[[AsyncResult], Any] | None = None,
        # Not a client-side cancel scope: this is the *server-side* transaction
        # timeout, carried in the Query object so Neo4j itself stops the
        # statement. asyncio.timeout would abandon the client while the server
        # kept burning CPU on the query, which is the failure mode this avoids.
        timeout: float | None = None,  # noqa: ASYNC109 - server-side query timeout
        label: str = "query",
    ) -> Any:
        """The single funnel for every statement this store issues.

        ``execute_query`` rather than a hand-rolled session, because it is the
        only entry point that gets the driver's managed-transaction retry: a
        leader switch mid-write becomes a retry instead of an error.

        Error mapping is the whole point of centralising this:

        * unreachable / expired / pool-starved -> :class:`StoreUnavailable`,
          and the breaker counts it, because those are the failures where
          continuing to call a dead database wastes the request budget;
        * Neo4j ``TransientError`` (deadlock, store overload) ->
          :class:`TransientError`, retryable by the caller;
        * ``ClientError`` propagates untouched and does **not** open the
          breaker — a syntax error or a constraint violation is our statement's
          fault, and tripping the breaker would take the store down over a bad
          generated query.
        """
        self.breaker.check()
        # Justification for the ignore below: the driver types Query.text as
        # LiteralString to discourage f-string-built Cypher, and a statement
        # assembled at runtime can never satisfy that annotation however safe it
        # is -- LiteralString cannot express "assembled from trusted parts".
        # The risk it stands in for is handled by other means on every path into
        # here: execute_readonly runs generated Cypher through CypherGuard
        # first, interpolated identifiers go through validate_label /
        # validate_rel_type / _bounded_hops, and all data is bound via $params.
        # The driver accepts a plain str at runtime.
        query = Query(  # type: ignore[arg-type]
            cypher, timeout=self.cfg.query_timeout_s if timeout is None else timeout
        )
        extra: dict[str, Any] = {}
        if transformer is not None:
            extra["result_transformer_"] = transformer
        try:
            result = await self.driver.execute_query(
                query,
                params or {},
                routing_=RoutingControl.WRITE if write else RoutingControl.READ,
                database_=self.cfg.database,
                **extra,
            )
        except _UNAVAILABLE as exc:
            self.breaker.record_failure()
            raise StoreUnavailable("neo4j", str(exc), statement=label) from exc
        except Neo4jTransientError as exc:
            self.breaker.record_failure()
            raise TransientError(
                f"neo4j transient failure: {exc}",
                statement=label,
                code=getattr(exc, "code", ""),
            ) from exc
        self.breaker.record_success()
        return result

    async def _records(
        self, cypher: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> list[Record]:
        result = await self._run(cypher, params, **kwargs)
        return list(result.records)

    async def _write_count(self, cypher: str, params: dict[str, Any], *, label: str) -> int:
        records = await self._records(cypher, params, write=True, label=label)
        return int(records[0]["written"]) if records else 0

    async def _apoc_available(self) -> bool:
        """Probe for APOC once per store, then remember the answer.

        Probing beats attempting-and-falling-back: an unknown procedure fails
        at plan time so nothing would be half-written, but a *different*
        ``ClientError`` from a graph that does have APOC would then be
        misdiagnosed as "APOC missing" on every subsequent call.
        """
        if self._has_apoc is not None:
            return self._has_apoc
        if self._apoc_lock is None:
            self._apoc_lock = asyncio.Lock()
        async with self._apoc_lock:
            if self._has_apoc is not None:
                return self._has_apoc
            try:
                records = await self._records(
                    "SHOW PROCEDURES YIELD name WHERE name = $name RETURN count(*) AS n",
                    {"name": "apoc.merge.relationship"},
                    label="apoc_probe",
                )
                self._has_apoc = bool(records and int(records[0]["n"]) > 0)
            except (Neo4jError, StoreUnavailable, TransientError) as exc:
                # No APOC and no privilege to ask are the same thing here: use
                # the grouped-per-type plan, which needs neither.
                log.debug("neo4j_apoc_probe_failed", error=str(exc)[:200])
                self._has_apoc = False
            log.info("neo4j_apoc", available=self._has_apoc)
            return self._has_apoc

    def _bounded_hops(self, hops: int, *, name: str = "hops") -> int:
        """Coerce a hop count to a small positive int.

        This value is interpolated into ``*1..N``, so it is validated as an
        ``int`` and never derived from user text. The ceiling is
        ``graph.multihop_max_path_length``: variable-length expansion is
        exponential in the branching factor, and an unbounded pattern on a hub
        node will exhaust the transaction's heap long before it returns.
        """
        try:
            value = int(hops)
        except (TypeError, ValueError) as exc:
            raise GuardrailViolation(
                f"{name} must be an integer", rule="cypher_hops", value=repr(hops)
            ) from exc
        if value < 1:
            raise GuardrailViolation(f"{name} must be >= 1", rule="cypher_hops", value=value)
        ceiling = max(1, int(self.settings.graph.multihop_max_path_length))
        if value > ceiling:
            log.warning("neo4j_hops_clamped", requested=value, ceiling=ceiling)
            return ceiling
        return value

    def _row_cap(self, limit: int | None) -> int:
        """Effective row budget: the caller's ask, capped by settings.

        ``max_cypher_rows`` is a hard ceiling and not a default — a generated
        query is allowed to ask for less and never for more, because whatever
        comes back is heading for a prompt.
        """
        requested = self.cfg.max_cypher_rows if limit is None else int(limit)
        return max(1, min(requested, self.cfg.max_cypher_rows))

    # -- generated Cypher -------------------------------------------------
    def _guard(self, cypher: str) -> str:
        """Second line of defence, delegating to the real guard.

        This used to be a private keyword scan, which is how it fell one list
        behind: backtick-quoted procedure names (``CALL `apoc.load.json` ``) passed
        it while :class:`~ragorc.security.cypher_guard.CypherGuard` refused them.
        A defence-in-depth layer that reimplements the check it is backing up is
        not a second layer, it is a second thing to keep in sync — and it was
        already out of sync. Delegating means both layers improve together.

        Still a real second layer: this runs at the store, so it also covers a
        caller that reached ``execute_readonly`` without going through the
        constructor.
        """
        from ragorc.security.cypher_guard import CypherGuard

        guard = CypherGuard(
            self.settings.security,
            max_rows=self.settings.neo4j.max_cypher_rows,
            max_hops=self.settings.graph.multihop_max_path_length,
        )
        return guard.validate(cypher).cypher

    async def execute_readonly(
        self, cypher: str, params: dict[str, Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Run a generated read query and return plain, JSON-safe rows.

        Rows are capped with ``AsyncResult.fetch`` rather than by appending a
        ``LIMIT`` to the statement: the query may already end in one (``LIMIT 5
        LIMIT 200`` is a syntax error), and wrapping it in a subquery changes
        its semantics. ``fetch(n)`` pulls exactly ``n`` records off the stream
        and ``consume()`` closes the transaction, discarding the rest — the tail
        of a runaway result never crosses the network.
        """
        cypher = self._guard(cypher)
        cap = self._row_cap(limit)

        async def _capped(result: AsyncResult) -> list[dict[str, Any]]:
            records = await result.fetch(cap)
            await result.consume()
            return [_serialize_record(record) for record in records]

        try:
            if self.settings.security.cypher_explain_dryrun:
                # EXPLAIN plans without executing: catches a syntax error, an
                # unbound parameter and an unbounded expansion for the price of
                # one round trip and zero data access.
                await self._run(f"EXPLAIN {cypher}", params, label="explain")
            rows: list[dict[str, Any]] = await self._run(
                cypher, params, transformer=_capped, label="readonly"
            )
        except ClientError as exc:
            # The statement is wrong, not the database. For generated Cypher
            # that is a construction failure the caller can repair and retry.
            raise ConstructionError(
                f"Cypher rejected by Neo4j: {exc}",
                code=getattr(exc, "code", ""),
                cypher=cypher[:500],
            ) from exc
        log.debug("neo4j_readonly", rows=len(rows), cap=cap)
        return rows

    async def schema_summary(self, *, refresh: bool = False) -> str:
        return await self._introspector.summary(refresh=refresh)

    async def ensure_schema(self) -> list[str]:
        """Create constraints and indexes, then invalidate the cached schema."""
        applied = await ensure_schema(self.driver, self.settings)
        self._introspector.invalidate()
        return applied

    async def health(self) -> dict[str, Any]:
        return await verify_connectivity(
            self.driver, database=self.cfg.database, timeout=self.cfg.connection_timeout_s
        )

    # -- writes -----------------------------------------------------------
    async def upsert_entities(self, entities: Sequence[Entity]) -> int:
        """Merge a batch of entities in one round trip.

        ``ON CREATE`` writes the properties; ``ON MATCH`` *accumulates* them,
        because the same entity is described differently by every chunk it
        appears in and the union of those descriptions is the node's value.
        """
        if not entities:
            return 0
        rows = await asyncio.to_thread(_entity_rows, entities)
        if not rows:
            return 0

        cypher = f"""
UNWIND $rows AS row
MERGE (n:`{self.node_label}` {{name: row.name}})
ON CREATE SET
    n.type = row.type,
    n.description = row.description,
    n.aliases = row.aliases,
    n.source_chunk_ids = row.source_chunk_ids,
    n.degree = row.degree,
    n.community_id = row.community_id,
    n.metadata_json = row.metadata_json,
    n.embedding = row.embedding,
    n.created_at = timestamp(),
    n.updated_at = timestamp()
ON MATCH SET
    n.type = CASE WHEN coalesce(n.type, 'Entity') = 'Entity' THEN row.type ELSE n.type END,
    n.description = {_accumulate_text("n")},
    n.aliases = {_accumulate_list("n", "aliases")},
    n.source_chunk_ids = {_accumulate_list("n", "source_chunk_ids")},
    n.degree = CASE WHEN row.degree > coalesce(n.degree, 0) THEN row.degree ELSE n.degree END,
    n.community_id = coalesce(row.community_id, n.community_id),
    n.metadata_json = coalesce(row.metadata_json, n.metadata_json),
    n.embedding = CASE WHEN row.embedding IS NULL THEN n.embedding ELSE row.embedding END,
    n.updated_at = timestamp()
RETURN count(DISTINCT n) AS written
"""
        written = await self._write_count(cypher, {"rows": rows}, label="upsert_entities")
        log.info("neo4j_entities_upserted", rows=len(rows), written=written)
        return written

    async def upsert_relations(self, relations: Sequence[Relation]) -> int:
        """Merge typed, weighted edges — one statement per distinct type.

        The endpoints are merged, not matched: a relation implies its endpoints,
        so an edge extracted before its entity row lands still creates a usable
        node, and the later :meth:`upsert_entities` fills in the properties.
        """
        if not relations:
            return 0
        grouped = await asyncio.to_thread(_relation_rows, relations)
        if not grouped:
            return 0

        if await self._apoc_available():
            flat = [row for rows in grouped.values() for row in rows]
            written = await self._write_count(
                self._apoc_relation_cypher(), {"rows": flat}, label="upsert_relations_apoc"
            )
        else:
            # One statement per type: a handful of round trips, because the
            # type vocabulary comes from our extraction schema and is bounded.
            counts = await bounded_gather(
                [
                    self._write_count(
                        self._typed_relation_cypher(rel_type),
                        {"rows": rows},
                        label=f"upsert_relations[{rel_type}]",
                    )
                    for rel_type, rows in grouped.items()
                ],
                limit=max(1, self.settings.indexing.max_concurrent_documents),
            )
            written = sum(counts)
        log.info(
            "neo4j_relations_upserted",
            types=len(grouped),
            rows=sum(len(rows) for rows in grouped.values()),
            written=written,
        )
        return written

    def _apoc_relation_cypher(self) -> str:
        """APOC path: the type becomes a procedure *argument*, not query text.

        ``apoc.merge.relationship`` merges on ``(start, type, end)`` when the
        identifying property map is empty, which is exactly our edge identity.
        Weight is accumulated after the YIELD instead of through
        ``onCreateProps``/``onMatchProps`` so the create and match branches
        share one expression: ``coalesce(null, 0.0) + w`` is ``w``.
        """
        return f"""
UNWIND $rows AS row
MERGE (a:`{self.node_label}` {{name: row.source}})
    ON CREATE SET a.created_at = timestamp()
MERGE (b:`{self.node_label}` {{name: row.target}})
    ON CREATE SET b.created_at = timestamp()
WITH a, b, row
CALL apoc.merge.relationship(a, row.type, {{}}, {{}}, b, {{}}) YIELD rel
SET rel.weight = {_accumulate_weight("rel")},
    rel.source_chunk_ids = {_accumulate_list("rel", "source_chunk_ids")},
    rel.description = {_accumulate_text("rel")},
    rel.metadata_json = coalesce(row.metadata_json, rel.metadata_json),
    rel.updated_at = timestamp()
RETURN count(DISTINCT rel) AS written
"""

    def _typed_relation_cypher(self, rel_type: str) -> str:
        rel_type = validate_rel_type(rel_type)
        return f"""
UNWIND $rows AS row
MERGE (a:`{self.node_label}` {{name: row.source}})
    ON CREATE SET a.created_at = timestamp()
MERGE (b:`{self.node_label}` {{name: row.target}})
    ON CREATE SET b.created_at = timestamp()
MERGE (a)-[r:`{rel_type}`]->(b)
ON CREATE SET
    r.description = row.description,
    r.weight = row.weight,
    r.source_chunk_ids = row.source_chunk_ids,
    r.metadata_json = row.metadata_json,
    r.created_at = timestamp(),
    r.updated_at = timestamp()
ON MATCH SET
    r.weight = {_accumulate_weight("r")},
    r.source_chunk_ids = {_accumulate_list("r", "source_chunk_ids")},
    r.description = {_accumulate_text("r")},
    r.metadata_json = coalesce(row.metadata_json, r.metadata_json),
    r.updated_at = timestamp()
RETURN count(DISTINCT r) AS written
"""

    async def upsert_chunk_links(self, chunk_ids_by_entity: Mapping[str, Sequence[str]]) -> int:
        """Link chunks to the entities they mention.

        ``(:Chunk)-[:MENTIONS]->(:Entity)`` is the edge that lets local search
        walk from a matched entity back to the text that asserted it, which is
        what turns a graph hit into a citable chunk. The edge carries no
        counter: chunk ids are content-derived, so re-ingest must be a no-op
        rather than an increment.
        """
        if not chunk_ids_by_entity:
            return 0
        rows = [
            {"name": name.strip(), "chunk_id": str(chunk_id)}
            for name, chunk_ids in chunk_ids_by_entity.items()
            if name and name.strip()
            for chunk_id in _dedupe(chunk_ids)
        ]
        if not rows:
            return 0

        cypher = f"""
UNWIND $rows AS row
MERGE (c:`{self.chunk_label}` {{id: row.chunk_id}})
    ON CREATE SET c.created_at = timestamp()
MERGE (e:`{self.node_label}` {{name: row.name}})
    ON CREATE SET e.created_at = timestamp()
MERGE (c)-[m:`{_MENTIONS}`]->(e)
    ON CREATE SET m.created_at = timestamp()
RETURN count(DISTINCT m) AS written
"""
        written = await self._write_count(cypher, {"rows": rows}, label="upsert_chunk_links")
        log.info("neo4j_chunk_links_upserted", rows=len(rows), written=written)
        return written

    async def upsert_communities(self, communities: Sequence[Community]) -> int:
        """Store community summaries and rebuild their membership edges.

        Membership is *replaced*, not accumulated: community detection is
        re-run over the whole graph, so an entity that moved between runs must
        not stay attached to its old community — a stale member silently
        poisons the summary that global search reads. The two unit subqueries
        keep the detach-then-attach on one round trip without letting the
        ``OPTIONAL MATCH`` fan out the outer row.
        """
        if not communities:
            return 0
        rows = await asyncio.to_thread(_community_rows, communities)
        if not rows:
            return 0

        cypher = f"""
UNWIND $rows AS row
MERGE (c:`{self.community_label}` {{id: row.id}})
SET c.level = row.level,
    c.title = row.title,
    c.summary = row.summary,
    c.rank = row.rank,
    c.parent_id = row.parent_id,
    c.entity_names = row.entity_names,
    c.relation_keys_json = row.relation_keys_json,
    c.embedding = CASE WHEN row.embedding IS NULL THEN c.embedding ELSE row.embedding END,
    c.updated_at = timestamp()
WITH c, row
CALL {{
    WITH c
    OPTIONAL MATCH (:`{self.node_label}`)-[stale:`{_IN_COMMUNITY}`]->(c)
    DELETE stale
}}
CALL {{
    WITH c, row
    UNWIND row.entity_names AS name
    MATCH (e:`{self.node_label}` {{name: name}})
    MERGE (e)-[:`{_IN_COMMUNITY}`]->(c)
}}
RETURN count(DISTINCT c) AS written
"""
        written = await self._write_count(cypher, {"rows": rows}, label="upsert_communities")
        log.info("neo4j_communities_upserted", rows=len(rows), written=written)
        return written

    # -- reads ------------------------------------------------------------
    async def fulltext_entities(
        self, query: str, *, limit: int | None = None
    ) -> list[tuple[Entity, float]]:
        """Lucene search over entity name + description — GraphRAG local search
        step one: turn a question into graph entry points.

        Returns ``(entity, score)`` pairs; the score is Lucene relevance, so
        higher is better and no conversion is needed. A missing index degrades
        to an empty result rather than failing the request — the query can
        still be answered from the other stores — but it is logged at error
        level because it means :func:`ensure_schema` was never run.
        """
        text = _lucene_escape(query)
        if not text:
            return []
        cap = self._row_cap(limit)
        index = fulltext_index_name(self.node_label)
        cypher = """
CALL db.index.fulltext.queryNodes($index, $q, {limit: $limit}) YIELD node, score
WHERE $label IN labels(node)
RETURN node.name AS name,
       node.type AS type,
       node.description AS description,
       node.aliases AS aliases,
       node.source_chunk_ids AS source_chunk_ids,
       node.degree AS degree,
       node.community_id AS community_id,
       node.metadata_json AS metadata_json,
       score
ORDER BY score DESC
"""
        try:
            records = await self._records(
                cypher,
                {"index": index, "q": text, "limit": cap, "label": self.node_label},
                label="fulltext_entities",
            )
        except ClientError as exc:
            log.error(
                "neo4j_fulltext_unavailable",
                index=index,
                code=getattr(exc, "code", ""),
                error=str(exc)[:200],
                hint="run ensure_schema() with neo4j.create_fulltext_index enabled",
            )
            return []
        return [(_entity_from_props(record), float(record["score"])) for record in records]

    async def neighbors(
        self, names: Sequence[str], *, hops: int = 1, limit: int = 50
    ) -> tuple[list[Entity], list[Relation]]:
        """Bounded ego-network expansion around the given entities.

        Edges are ordered by weight before truncation, so cutting the result at
        ``limit`` keeps the *strongest* assertions instead of whichever ones the
        planner happened to emit first — which is what makes the truncated
        subgraph still a useful prompt.

        Traversal is undirected (``-[*1..N]-``) because "what is connected to
        X" does not care which way the extractor happened to orient the edge;
        the true direction is recovered from ``startNode``/``endNode``.
        """
        seeds = _dedupe(names)
        if not seeds:
            return [], []
        depth = self._bounded_hops(hops)
        cap = self._row_cap(limit)

        edge_cypher = f"""
MATCH (a:`{self.node_label}`) WHERE a.name IN $names
MATCH (a)
      ((:`{self.node_label}`)-[rels{_SEMANTIC_EDGE}]-(:`{self.node_label}`)){{1,{depth}}}
      (:`{self.node_label}`)
UNWIND rels AS r
WITH DISTINCT r
ORDER BY coalesce(r.weight, 1.0) DESC
LIMIT $limit
RETURN startNode(r) AS a,
       endNode(r) AS b,
       type(r) AS type,
       coalesce(r.description, '') AS description,
       coalesce(r.weight, 1.0) AS weight,
       coalesce(r.source_chunk_ids, []) AS source_chunk_ids,
       r.metadata_json AS metadata_json
"""
        # The seed lookup is a second query rather than an OPTIONAL MATCH so a
        # seed with no edges still contributes its description to the prompt.
        seed_cypher = f"""
MATCH (n:`{self.node_label}`) WHERE n.name IN $names
RETURN n AS node
LIMIT $limit
"""
        edge_records, seed_records = await bounded_gather(
            [
                self._records(edge_cypher, {"names": seeds, "limit": cap}, label="neighbors_edges"),
                self._records(seed_cypher, {"names": seeds, "limit": cap}, label="neighbors_seeds"),
            ],
            limit=2,
        )

        entities: dict[str, Entity] = {}
        for record in seed_records:
            entity = _entity_from_props(record["node"])
            if entity.name:
                entities[entity.name] = entity

        relations: list[Relation] = []
        seen_edges: set[tuple[str, str, str]] = set()
        for record in edge_records:
            source = _entity_from_props(record["a"])
            target = _entity_from_props(record["b"])
            for entity in (source, target):
                if entity.name and entity.name not in entities:
                    entities[entity.name] = entity
            relation = _relation_from_props(
                {
                    "source": source.name,
                    "target": target.name,
                    "type": record["type"],
                    "description": record["description"],
                    "weight": record["weight"],
                    "source_chunk_ids": record["source_chunk_ids"],
                    "metadata_json": record["metadata_json"],
                }
            )
            if relation.key in seen_edges:
                continue
            seen_edges.add(relation.key)
            relations.append(relation)

        log.debug(
            "neo4j_neighbors",
            seeds=len(seeds),
            hops=depth,
            entities=len(entities),
            relations=len(relations),
        )
        return list(entities.values()), relations

    async def subgraph(
        self, names: Sequence[str], *, hops: int = 1
    ) -> tuple[list[Entity], list[Relation]]:
        """Prompt-sized ego network: :meth:`neighbors` at the row budget.

        ``max_cypher_rows`` is the limit because this result is context, and the
        row budget is the same one every other prompt-bound query answers to.
        """
        return await self.neighbors(names, hops=hops, limit=self.cfg.max_cypher_rows)

    async def paths(
        self,
        start: Sequence[str],
        end: Sequence[str],
        *,
        max_hops: int = 3,
        limit: int = 10,
    ) -> list[GraphPath]:
        """All shortest paths between two entity sets, ranked by evidence.

        ``allShortestPaths`` gives every equally short connection, which is what
        a multi-hop question wants: the shortest path is the explanation, and
        the alternatives are corroboration. Among paths of equal length the
        ranking is summed relationship weight divided by hop count — an edge
        asserted by twenty documents is stronger evidence than one asserted
        once, and a shorter chain of the same total weight is a better
        explanation. Ordering happens server-side so truncation at ``limit``
        keeps the best paths; the score arithmetic is then recomputed in one
        vectorized numpy expression rather than per row.
        """
        starts = _dedupe(start)
        ends = _dedupe(end)
        if not starts or not ends:
            return []
        depth = self._bounded_hops(max_hops, name="max_hops")
        cap = self._row_cap(limit)

        cypher = f"""
MATCH (a:`{self.node_label}`) WHERE a.name IN $start
MATCH (b:`{self.node_label}`) WHERE b.name IN $end
WITH a, b WHERE a <> b
MATCH p = ALL SHORTEST
      (a) ((:`{self.node_label}`)-[{_SEMANTIC_EDGE}]-(:`{self.node_label}`)){{1,{depth}}} (b)
WITH p,
     length(p) AS hops,
     reduce(w = 0.0, r IN relationships(p) | w + coalesce(r.weight, 1.0)) AS total_weight
WITH p, hops, total_weight,
     CASE WHEN hops = 0 THEN 0.0 ELSE total_weight / hops END AS score
ORDER BY score DESC, hops ASC
LIMIT $limit
RETURN [n IN nodes(p) | n.name] AS names,
       [r IN relationships(p) | {{
           source: startNode(r).name,
           target: endNode(r).name,
           type: type(r),
           description: coalesce(r.description, ''),
           weight: coalesce(r.weight, 1.0),
           source_chunk_ids: coalesce(r.source_chunk_ids, []),
           metadata_json: r.metadata_json
       }}] AS rels,
       hops,
       total_weight
"""
        records = await self._records(
            cypher, {"start": starts, "end": ends, "limit": cap}, label="paths"
        )
        if not records:
            return []

        count = len(records)
        weights = np.fromiter(
            (float(record["total_weight"] or 0.0) for record in records),
            dtype=np.float32,
            count=count,
        )
        hop_counts = np.fromiter(
            (int(record["hops"] or 0) for record in records), dtype=np.int64, count=count
        )
        scores = weights / np.maximum(hop_counts, 1).astype(np.float32)

        paths: list[GraphPath] = []
        for record, score in zip(records, scores.tolist(), strict=True):
            paths.append(
                GraphPath(
                    nodes=tuple(str(name) for name in record["names"]),
                    relations=tuple(_relation_from_props(rel) for rel in record["rels"]),
                    score=float(score),
                )
            )
        log.debug("neo4j_paths", found=len(paths), hops=depth)
        return paths

    async def communities(
        self, *, level: int | None = None, limit: int | None = None
    ) -> list[Community]:
        """Read community summaries — the unit of GraphRAG *global* search.

        Ordered by rank descending so the row cap keeps the communities that
        matter. Membership is read from the ``IN_COMMUNITY`` edges and falls
        back to the stored name list when the edges have not been built (the
        summary is still usable without them).
        """
        cap = self._row_cap(limit)
        cypher = f"""
MATCH (c:`{self.community_label}`)
WHERE $level IS NULL OR c.level = $level
OPTIONAL MATCH (e:`{self.node_label}`)-[:`{_IN_COMMUNITY}`]->(c)
WITH c, collect(e.name) AS members
RETURN c.id AS id,
       c.level AS level,
       c.title AS title,
       c.summary AS summary,
       c.rank AS rank,
       c.parent_id AS parent_id,
       CASE WHEN size(members) > 0 THEN members ELSE coalesce(c.entity_names, []) END AS names,
       c.relation_keys_json AS relation_keys_json,
       c.embedding AS embedding
ORDER BY c.rank DESC, c.id ASC
LIMIT $limit
"""
        records = await self._records(cypher, {"level": level, "limit": cap}, label="communities")
        out: list[Community] = []
        for record in records:
            raw_keys = record["relation_keys_json"]
            keys = orjson.loads(raw_keys) if raw_keys else []
            embedding = record["embedding"]
            out.append(
                Community(
                    id=int(record["id"]),
                    level=int(record["level"] or 0),
                    entity_names=tuple(str(name) for name in (record["names"] or ())),
                    relation_keys=tuple(
                        (str(k[0]), str(k[1]), str(k[2])) for k in keys if len(k) == 3
                    ),
                    title=str(record["title"] or ""),
                    summary=str(record["summary"] or ""),
                    rank=float(record["rank"] or 0.0),
                    parent_id=record["parent_id"],
                    embedding=(
                        None if embedding is None else np.asarray(embedding, dtype=np.float32)
                    ),
                )
            )
        log.debug("neo4j_communities", level=level, found=len(out))
        return out

    async def degree_centrality(self, names: Sequence[str]) -> dict[str, float]:
        """Degree centrality from a plain Cypher count — no GDS required.

        ``COUNT { (n)--() }`` is a count-subquery over the node's relationship
        chain, which is what GDS's degree centrality computes anyway; projecting
        a graph into GDS to get it would cost more than the answer is worth.
        Values are normalized against the largest degree in the requested set
        so they compose with other [0, 1] signals, and an unknown name scores
        0.0 rather than being absent.
        """
        wanted = _dedupe(names)
        if not wanted:
            return {}
        cypher = f"""
MATCH (n:`{self.node_label}`) WHERE n.name IN $names
RETURN n.name AS name,
       COUNT {{ (n)-[r]-(:`{self.node_label}`)
                WHERE NOT type(r) IN [$mentions, $in_community] }} AS degree
"""
        records = await self._records(
            cypher,
            {"names": wanted, "mentions": _MENTIONS, "in_community": _IN_COMMUNITY},
            label="degree_centrality",
        )
        centrality = dict.fromkeys(wanted, 0.0)
        if not records:
            return centrality
        found = [str(record["name"]) for record in records]
        degrees = np.fromiter(
            (float(record["degree"] or 0) for record in records),
            dtype=np.float64,
            count=len(records),
        )
        peak = float(degrees.max())
        normalized = degrees / peak if peak > 0.0 else np.zeros_like(degrees)
        centrality.update(zip(found, (float(v) for v in normalized), strict=True))
        return centrality

    # -- lifecycle --------------------------------------------------------
    async def close(self) -> None:
        """Release the pool.

        An injected driver is left alone — it belongs to whoever passed it in.
        A driver taken from the module cache is evicted as well as closed, so a
        later :func:`build_driver` builds a fresh pool instead of handing out a
        closed one.
        """
        if self._external_driver:
            return
        await close_driver(self.driver)
