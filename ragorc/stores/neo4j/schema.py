"""Neo4j constraints, indexes and schema introspection.

Why the indexes are not optional
--------------------------------
Every write path in :mod:`ragorc.stores.neo4j.store` is a ``MERGE`` on
``(:Entity {name})``, ``(:Chunk {id})`` or ``(:Community {id})``. Without a
uniqueness constraint on those properties, ``MERGE`` degrades from an index
seek to a **full label scan** — so entity upserts go quadratic in corpus size
and concurrent extraction workers happily create duplicate nodes, because
``MERGE`` is only atomic when an index can lock the key. The constraint is
therefore a correctness requirement first and a performance one second.

The full-text index is the entry point for GraphRAG *local* search: a question
is matched against entity names and descriptions to find where in the graph to
start traversing. Exact-match lookup cannot do that job — questions say
"Anthropic's safety team", the graph says "Anthropic" — so a Lucene index over
``name`` and ``description`` is what makes local search work at all.

Why introspection is cached
---------------------------
The Text-to-Cypher prompt needs a schema description on *every* query, and the
description changes only when someone ingests new node types. Re-deriving it
per request costs 3 + N round trips (labels, relationship types, one property
sample per label) — pure waste against a schema that is stable for hours.

Why property sampling and not a metadata catalog
------------------------------------------------
Neo4j is schema-optional: there is no ``information_schema`` to read. APOC has
``apoc.meta.schema`` but APOC is a plugin and is absent on Aura Free and on
most hardened deployments, so the base implementation samples: 50 nodes per
label, aggregate their ``keys(n)`` server-side. Counts come from
``apoc.meta.stats()`` when the plugin exists and otherwise from per-label
``count(n)``, which Neo4j answers from the **count store** in constant time
rather than by scanning.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

import structlog
from neo4j import Query, RoutingControl
from neo4j.exceptions import (
    ClientError,
    DatabaseError,
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
)

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError, GuardrailViolation, StoreUnavailable
from ragorc.core.settings import Settings, get_settings

if TYPE_CHECKING:
    from neo4j import AsyncDriver, Record

log = structlog.get_logger(__name__)

__all__ = [
    "GraphSchemaIntrospector",
    "ensure_schema",
    "fulltext_index_name",
    "validate_label",
    "validate_rel_type",
    "vector_index_name",
]

# Neither a label nor a relationship type can be a query parameter in Cypher —
# they are part of the query plan, not of the data — so anything interpolated
# into a pattern is validated against these first. Relationship types are
# additionally required to be SCREAMING_SNAKE_CASE, which is what our own
# extraction schema emits; anything else did not come from us.
_LABEL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

# Nodes sampled per label when deriving the property list. 50 is enough to see
# every property of a homogeneous label and cheap enough to run for every label
# in the database on a cold cache.
_PROPERTY_SAMPLE = 50

_TRANSPORT_ERRORS = (ServiceUnavailable, SessionExpired)


def validate_label(label: str) -> str:
    """Return ``label`` if it is safe to interpolate into Cypher, else raise."""
    if not _LABEL_RE.match(label or ""):
        raise GuardrailViolation(
            "illegal Neo4j label",
            rule="cypher_label",
            label=label,
            pattern=_LABEL_RE.pattern,
        )
    return label


def validate_rel_type(rel_type: str) -> str:
    """Return ``rel_type`` if it matches ``^[A-Z][A-Z0-9_]*$``, else raise.

    This is the guard that makes dynamic relationship types safe: the type
    arrives from our extraction schema, but "arrives from our own code" is an
    assumption, and an LLM-authored type string is one prompt injection away
    from being ``FOO]->() DETACH DELETE n //``.
    """
    if not _REL_TYPE_RE.match(rel_type or ""):
        raise GuardrailViolation(
            "illegal Neo4j relationship type",
            rule="cypher_relationship_type",
            relationship_type=rel_type,
            pattern=_REL_TYPE_RE.pattern,
        )
    return rel_type


def fulltext_index_name(label: str) -> str:
    return f"ragorc_{validate_label(label).lower()}_fulltext"


def vector_index_name(label: str) -> str:
    return f"ragorc_{validate_label(label).lower()}_embedding"


def _schema_statements(settings: Settings) -> list[tuple[str, str]]:
    """``(name, cypher)`` pairs, in dependency order."""
    cfg = settings.neo4j
    entity = validate_label(cfg.node_label)
    chunk = validate_label(cfg.chunk_label)
    community = validate_label(cfg.community_label)

    stmts: list[tuple[str, str]] = [
        # Uniqueness constraints also create the backing index, which is what
        # makes MERGE an index seek instead of a label scan.
        (
            f"constraint_{entity.lower()}_name",
            f"CREATE CONSTRAINT ragorc_{entity.lower()}_name IF NOT EXISTS "
            f"FOR (n:`{entity}`) REQUIRE n.name IS UNIQUE",
        ),
        (
            f"constraint_{chunk.lower()}_id",
            f"CREATE CONSTRAINT ragorc_{chunk.lower()}_id IF NOT EXISTS "
            f"FOR (n:`{chunk}`) REQUIRE n.id IS UNIQUE",
        ),
        (
            f"constraint_{community.lower()}_id",
            f"CREATE CONSTRAINT ragorc_{community.lower()}_id IF NOT EXISTS "
            f"FOR (n:`{community}`) REQUIRE n.id IS UNIQUE",
        ),
        # Plain CREATE INDEX is a RANGE index in Neo4j 5 — the successor to the
        # 4.x BTREE index, same access pattern, same use here: filtering an
        # entity traversal by type without scanning the label.
        (
            f"index_{entity.lower()}_type",
            f"CREATE INDEX ragorc_{entity.lower()}_type IF NOT EXISTS "
            f"FOR (n:`{entity}`) ON (n.type)",
        ),
        # Global search reads one level of the community hierarchy at a time.
        (
            f"index_{community.lower()}_level",
            f"CREATE INDEX ragorc_{community.lower()}_level IF NOT EXISTS "
            f"FOR (n:`{community}`) ON (n.level)",
        ),
    ]

    if cfg.create_fulltext_index:
        stmts.append(
            (
                fulltext_index_name(entity),
                f"CREATE FULLTEXT INDEX {fulltext_index_name(entity)} IF NOT EXISTS "
                f"FOR (n:`{entity}`) ON EACH [n.name, n.description]",
            )
        )

    if cfg.create_vector_index:
        dim = settings.embedding.dense_dimension or settings.postgres.vector_dimension
        if not isinstance(dim, int) or dim <= 0:
            raise ConfigError(
                "create_vector_index needs a positive embedding dimension",
                dimension=dim,
            )
        # Index OPTIONS must be literals — Neo4j resolves them at plan time, so
        # a parameter is rejected. Safe because the value is an int we own.
        stmts.append(
            (
                vector_index_name(entity),
                f"CREATE VECTOR INDEX {vector_index_name(entity)} IF NOT EXISTS "
                f"FOR (n:`{entity}`) ON (n.embedding) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {int(dim)}, "
                "`vector.similarity_function`: 'cosine'}}",
            )
        )

    return stmts


async def ensure_schema(driver: AsyncDriver, settings: Settings | None = None) -> list[str]:
    """Create every constraint and index this store needs. Idempotent.

    Statements run **sequentially, not concurrently**: schema changes take a
    schema lock, so firing them in parallel buys nothing and can deadlock two
    workers that start up at the same instant.

    ``IF NOT EXISTS`` handles the same-name case. It does *not* handle an
    equivalent index that already exists under a different name, and it does
    not handle a server too old for vector indexes — both surface as a
    ``ClientError``, both are logged and skipped rather than allowed to abort
    an ingest. A transport failure is different: the database is not there, and
    there is nothing to be gained from continuing.
    """
    settings = settings or get_settings()
    database = settings.neo4j.database
    applied: list[str] = []

    for name, cypher in _schema_statements(settings):
        try:
            await driver.execute_query(cypher, routing_=RoutingControl.WRITE, database_=database)
        except _TRANSPORT_ERRORS as exc:
            raise StoreUnavailable("neo4j", str(exc), stage="ensure_schema") from exc
        except (ClientError, DatabaseError) as exc:
            log.warning(
                "neo4j_schema_statement_skipped",
                index=name,
                code=getattr(exc, "code", ""),
                error=str(exc)[:300],
            )
        else:
            applied.append(name)

    log.info("neo4j_schema_ready", applied=applied, database=database)
    return applied


class GraphSchemaIntrospector:
    """Builds the compact schema description used by the Text-to-Cypher prompt.

    The output is prose, not JSON, because it is prompt input: a model writes
    better Cypher from ``(:Chunk)-[:MENTIONS]->(:Entity)`` than from a nested
    object describing the same edge, and the pattern form costs a third of the
    tokens.
    """

    def __init__(self, driver: AsyncDriver, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.cfg = self.settings.neo4j
        self.driver = driver
        self._summary: str | None = None
        self._lock: asyncio.Lock | None = None

    async def summary(self, *, refresh: bool = False) -> str:
        """Return the cached description, introspecting on first use.

        The lock collapses a cold-start stampede: N concurrent queries would
        otherwise each run the full introspection fan-out against a database
        that is already busy answering them.
        """
        use_cache = self.settings.cache.cache_schema and not refresh
        if use_cache and self._summary is not None:
            return self._summary
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if use_cache and self._summary is not None:
                return self._summary
            self._summary = await self._introspect()
            return self._summary

    def invalidate(self) -> None:
        """Drop the cache. Call after an ingest that introduced new labels."""
        self._summary = None

    # -- plumbing ---------------------------------------------------------
    async def _read(self, cypher: str, params: dict[str, Any] | None = None) -> list[Record]:
        try:
            result = await self.driver.execute_query(
                # Justification: the driver types Query.text as LiteralString to
                # discourage f-string-built Cypher. Every statement here is a
                # module-level constant composed at runtime, so it can never
                # satisfy LiteralString no matter how safe it is -- the
                # annotation cannot express "assembled from trusted parts". The
                # risk it stands in for is handled directly: identifiers go
                # through validate_label/validate_rel_type and all data is bound
                # as parameters. The driver itself accepts a plain str.
                Query(cypher, timeout=self.cfg.query_timeout_s),  # type: ignore[arg-type]
                params or {},
                routing_=RoutingControl.READ,
                database_=self.cfg.database,
            )
        except _TRANSPORT_ERRORS as exc:
            raise StoreUnavailable("neo4j", str(exc), stage="introspect") from exc
        return list(result.records)

    async def _try_read(self, cypher: str, params: dict[str, Any] | None = None) -> list[Record]:
        """Read something that may not exist on this server or for this user.

        ``db.schema.visualization``, ``SHOW INDEXES`` and every APOC procedure
        are all optional in some deployment, and a missing extra must degrade
        the description rather than fail the query that needed it.
        """
        try:
            return await self._read(cypher, params)
        except Neo4jError as exc:
            log.debug("neo4j_introspection_unavailable", code=getattr(exc, "code", ""))
            return []

    # -- introspection ----------------------------------------------------
    async def _introspect(self) -> str:
        label_rows, type_rows = await bounded_gather(
            [
                self._read("CALL db.labels() YIELD label RETURN label ORDER BY label"),
                self._read(
                    "CALL db.relationshipTypes() YIELD relationshipType AS t RETURN t ORDER BY t"
                ),
            ],
            limit=2,
        )
        # A pathological graph can have thousands of labels; the prompt gets the
        # same row budget as any other Cypher result.
        budget = max(1, self.cfg.max_cypher_rows)
        labels = [str(r["label"]) for r in label_rows if _LABEL_RE.match(str(r["label"]))][:budget]
        rel_types = [str(r["t"]) for r in type_rows if _REL_TYPE_RE.match(str(r["t"]))][:budget]

        stats = await self._apoc_stats()
        # These four calls return four different shapes, and ``bounded_gather``
        # is homogeneous (``list[T]``), so the element type collapses to their
        # common supertype and nothing downstream is checked. Bind each slot
        # back to the type its coroutine actually returns; the annotations are
        # what make the ``_render`` call below type-checked rather than erased.
        gathered: list[Any] = await bounded_gather(
            [
                self._sample_properties(labels),
                self._patterns(),
                self._indexes(),
                self._counts(labels, rel_types, stats),
            ],
            limit=4,
        )
        properties: dict[str, list[str]] = gathered[0]
        patterns: list[str] = gathered[1]
        indexes: list[str] = gathered[2]
        counts: dict[str, Any] = gathered[3]
        text = self._render(
            labels=labels,
            rel_types=rel_types,
            properties=properties,
            patterns=patterns,
            indexes=indexes,
            counts=counts,
        )
        log.info(
            "neo4j_schema_introspected",
            labels=len(labels),
            relationship_types=len(rel_types),
            chars=len(text),
            counts_source=counts.get("source"),
        )
        return text

    async def _apoc_stats(self) -> dict[str, Any]:
        """APOC gives every count in one round trip. Absent almost as often as
        present, so its absence is a normal path, not an error path."""
        rows = await self._try_read(
            "CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypesCount "
            "RETURN nodeCount, relCount, labels, relTypesCount"
        )
        if not rows:
            return {}
        row = rows[0]
        return {
            "nodeCount": int(row["nodeCount"] or 0),
            "relCount": int(row["relCount"] or 0),
            "labels": dict(row["labels"] or {}),
            "relTypesCount": dict(row["relTypesCount"] or {}),
        }

    async def _sample_properties(self, labels: list[str]) -> dict[str, list[str]]:
        """One sampled ``keys(n)`` aggregation per label, fanned out.

        Concurrency is bounded by the connection pool: more in-flight queries
        than there are connections just queues inside the driver.
        """
        if not labels:
            return {}

        async def one(label: str) -> tuple[str, list[str]]:
            rows = await self._try_read(
                f"MATCH (n:`{label}`) WITH n LIMIT $sample "
                "UNWIND keys(n) AS k RETURN collect(DISTINCT k) AS props",
                {"sample": _PROPERTY_SAMPLE},
            )
            props = sorted(str(p) for p in (rows[0]["props"] if rows else []))
            return label, props

        pairs = await bounded_gather(
            [one(label) for label in labels],
            limit=max(1, self.cfg.max_connection_pool_size),
        )
        return dict(pairs)

    async def _patterns(self) -> list[str]:
        """``(:A)-[:REL]->(:B)`` triples from ``db.schema.visualization``.

        The procedure returns *virtual* nodes whose ``name`` property is the
        label, which is the only place the driver hands back a graph object
        that never existed in the store.
        """
        rows = await self._try_read("CALL db.schema.visualization()")
        if not rows:
            return []
        patterns: set[str] = set()
        for rel in rows[0].get("relationships") or []:
            start = rel.start_node.get("name") if rel.start_node is not None else None
            end = rel.end_node.get("name") if rel.end_node is not None else None
            if not start or not end:
                continue
            patterns.add(f"(:{start})-[:{rel.type}]->(:{end})")
        return sorted(patterns)

    async def _indexes(self) -> list[str]:
        """Tell the model which indexes exist.

        Without this the generated Cypher never calls
        ``db.index.fulltext.queryNodes`` — it cannot know the index is there —
        and every entity lookup becomes an exact-match label scan.
        """
        rows = await self._try_read(
            "SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties "
            "RETURN name, type, entityType, labelsOrTypes, properties ORDER BY name"
        )
        out: list[str] = []
        for row in rows:
            kind = str(row["type"] or "")
            if kind == "LOOKUP":
                continue  # token lookup indexes carry no schema information
            targets = ", ".join(str(t) for t in (row["labelsOrTypes"] or []))
            props = ", ".join(str(p) for p in (row["properties"] or []))
            pattern = f"(:{targets})" if row["entityType"] == "NODE" else f"[:{targets}]"
            out.append(f"{kind} {row['name']} on {pattern} [{props}]")
        return out

    async def _counts(
        self, labels: list[str], rel_types: list[str], stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Node/relationship counts, from APOC if present and the count store
        otherwise.

        ``MATCH (n:L) RETURN count(n)`` and ``MATCH ()-[r:T]->() RETURN
        count(r)`` are answered from Neo4j's count store — O(1), no scan — so
        the fallback costs one cheap round trip per label rather than a full
        pass over the graph. The total it derives is an *estimate*: a node with
        two labels is counted twice.
        """
        if stats:
            return {
                "source": "apoc.meta.stats",
                "nodes": stats["nodeCount"],
                "relationships": stats["relCount"],
                "by_label": {k: int(v) for k, v in stats["labels"].items()},
                "by_type": {k: int(v) for k, v in stats["relTypesCount"].items()},
            }

        async def label_count(label: str) -> tuple[str, int]:
            rows = await self._try_read(f"MATCH (n:`{label}`) RETURN count(n) AS c")
            return label, (int(rows[0]["c"] or 0) if rows else 0)

        async def type_count(rel_type: str) -> tuple[str, int]:
            rows = await self._try_read(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS c")
            return rel_type, (int(rows[0]["c"] or 0) if rows else 0)

        limit = max(1, self.cfg.max_connection_pool_size)
        label_pairs = await bounded_gather([label_count(x) for x in labels], limit=limit)
        type_pairs = await bounded_gather([type_count(x) for x in rel_types], limit=limit)
        by_label = dict(label_pairs)
        by_type = dict(type_pairs)
        return {
            "source": "count store (estimated totals)",
            "nodes": sum(by_label.values()),
            "relationships": sum(by_type.values()),
            "by_label": by_label,
            "by_type": by_type,
        }

    def _render(
        self,
        *,
        labels: list[str],
        rel_types: list[str],
        properties: dict[str, list[str]],
        patterns: list[str],
        indexes: list[str],
        counts: dict[str, Any],
    ) -> str:
        by_label: dict[str, int] = counts.get("by_label") or {}
        by_type: dict[str, int] = counts.get("by_type") or {}
        lines: list[str] = [f"Neo4j graph schema (database: {self.cfg.database})", ""]

        lines.append("Node labels and their properties:")
        if not labels:
            lines.append("  (empty graph)")
        for label in labels:
            count = by_label.get(label)
            size = f" [{count} nodes]" if count is not None else ""
            props = ", ".join(properties.get(label) or []) or "(no properties sampled)"
            lines.append(f"  (:{label}){size}: {props}")

        lines += ["", "Relationship types:"]
        if not rel_types:
            lines.append("  (none)")
        for rel_type in rel_types:
            count = by_type.get(rel_type)
            size = f" [{count}]" if count is not None else ""
            lines.append(f"  [:{rel_type}]{size}")

        if patterns:
            lines += ["", "Connection patterns:"]
            lines += [f"  {p}" for p in patterns]

        if indexes:
            lines += ["", "Indexes:"]
            lines += [f"  {i}" for i in indexes]

        lines += [
            "",
            f"Totals: {counts.get('nodes', 0)} nodes, "
            f"{counts.get('relationships', 0)} relationships "
            f"(source: {counts.get('source', 'unknown')}).",
        ]
        return "\n".join(lines)
