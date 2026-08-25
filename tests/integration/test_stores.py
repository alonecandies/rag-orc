"""Integration tests against live Postgres, Neo4j and Qdrant.

Deselected by default (`-m "not integration"` in ``addopts``). Run with:

    make up && .venv/bin/python -m pytest tests -m integration

These exist because the unit suite's fakes cannot verify the things that only a
real server does: whether Qdrant's server-side fusion actually fuses, whether the
pgvector HNSW index is used, whether a Cypher `UNWIND` batch writes what we think
it writes. Those are exactly the claims worth checking against reality.

Every test creates and drops its own namespace, so the suite is idempotent and
does not depend on a seeded database.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Any

import numpy as np
import pytest

from ragorc.core.models import Chunk, Document, Entity, Query, Relation
from ragorc.core.settings import Settings

pytestmark = pytest.mark.integration


def _required(value: str, name: str) -> str:
    if not value:
        pytest.skip(f"{name} is not configured")
    return value


@pytest.fixture
def settings() -> Settings:
    """A settings object pointed at the compose stack, with an isolated namespace
    per run so concurrent runs and reruns cannot collide.

    Connection details come from ``Settings()`` — the same ``.env`` the CLI and the
    service read — rather than from literals here. They used to be hardcoded
    defaults, which meant these tests ignored the port in ``.env`` and connected to
    whatever answered on 5432. On a machine already running an unrelated Postgres
    there, the friendly outcome is an authentication failure; the unfriendly one is
    a test run that creates its tables in someone else's database.

    Only the fields that make a run *isolated* are overridden below.
    """
    suffix = uuid.uuid4().hex[:8]
    env = Settings()
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        embedding={"dense_dimension": 32},
        qdrant={
            "url": _required(env.qdrant.url, "qdrant.url"),
            "collection": f"ragorc_test_{suffix}",
            "prefer_grpc": True,
        },
        postgres={
            "dsn": _required(env.postgres.dsn, "postgres.dsn"),
            "chunks_table": f"ragorc_test_chunks_{suffix}",
            "documents_table": f"ragorc_test_docs_{suffix}",
            "vector_dimension": 32,
        },
        neo4j={
            "uri": _required(env.neo4j.uri, "neo4j.uri"),
            "user": env.neo4j.user,
            "password": env.neo4j.password,
            "node_label": f"TestEntity{suffix}",
        },
    )


def vector(seed: int, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


async def drop_collection(store: Any) -> None:
    """Remove the test collection outright.

    ``QdrantStore.delete()`` deliberately refuses a call with neither ids nor
    filters — an unbounded delete has to be asked for explicitly, which is the
    right default for a method a retrieval path can reach. Test cleanup genuinely
    does want the whole namespace gone, so it goes through the client.
    """
    with contextlib.suppress(Exception):
        await store.client.delete_collection(store.collection)


def make_chunks(n: int = 6) -> list[Chunk]:
    texts = [
        "Refunds are processed within 14 days of the request.",
        "Shipping is free on orders above 50 USD.",
        "Enterprise plans include a dedicated account manager.",
        "Support replies within one business day on weekdays.",
        "The API rate limit is 1000 requests per minute.",
        "Data is encrypted at rest using AES-256.",
    ]
    out = []
    for i, text in enumerate(texts[:n]):
        chunk = Chunk(id=f"itest-{i}", content=text, document_id="itest-doc", index=i)
        chunk.dense = vector(i)
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------
async def test_qdrant_roundtrip(settings: Settings) -> None:
    from ragorc.stores.qdrant.store import QdrantStore

    store = QdrantStore(settings=settings)
    try:
        await store.ensure_collection(recreate=True)
        chunks = make_chunks()
        written = await store.upsert(chunks)
        assert written == len(chunks)

        query = Query(text="how long do refunds take?", top_k=3)
        query.dense = chunks[0].dense  # nearest neighbour is chunk 0 by construction
        results = await store.search(query, top_k=3)
        assert results, "search returned nothing"
        assert results[0].chunk.id == "itest-0"
        assert results[0].score > 0.9, "cosine similarity with itself must be ~1"
        assert [r.rank for r in results] == list(range(len(results)))

        fetched = await store.get(["itest-1"])
        assert fetched and fetched[0].content.startswith("Shipping")
        assert await store.count() == len(chunks)
    finally:
        await drop_collection(store)
        await store.close()


async def test_qdrant_filters_are_applied(settings: Settings) -> None:
    from ragorc.stores.qdrant.store import QdrantStore

    store = QdrantStore(settings=settings)
    try:
        await store.ensure_collection(recreate=True)
        chunks = make_chunks(4)
        for i, chunk in enumerate(chunks):
            chunk.metadata["bucket"] = "a" if i < 2 else "b"
        await store.upsert(chunks)

        query = Query(text="anything", top_k=10)
        query.dense = vector(0)
        results = await store.search(query, top_k=10, filters={"bucket": "b"})
        assert results, "filtered search returned nothing"
        assert all(r.chunk.metadata.get("bucket") == "b" for r in results)
    finally:
        await drop_collection(store)
        await store.close()


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
async def test_postgres_schema_and_search(settings: Settings) -> None:
    from ragorc.stores.postgres.ddl import ensure_schema
    from ragorc.stores.postgres.pool import open_pool
    from ragorc.stores.postgres.store import PostgresStore

    pool = await open_pool(settings.postgres)
    store = PostgresStore(settings=settings)
    tables = (settings.postgres.chunks_table, settings.postgres.documents_table)
    try:
        await ensure_schema(pool, settings.postgres)

        # The chunks table has a foreign key to documents, and it is right to:
        # a chunk whose document row is missing can never be expanded to its
        # parent or attributed to a source. So the document goes in first, which
        # is the order real ingest uses.
        document = Document(
            id="itest-doc",
            content="the source document for the integration fixtures",
            source="itest.md",
            checksum="itest-checksum",
        )
        await store.upsert_documents([document])

        chunks = make_chunks()
        await store.upsert_chunks(chunks)

        # Full-text: the tsvector column is GENERATED, so it must be populated
        # without any trigger of ours.
        hits = await store.fulltext_search("refunds", top_k=3)
        assert hits, "full-text search found nothing — is content_tsv populated?"
        assert "Refund" in hits[0].chunk.content or "refund" in hits[0].chunk.content.lower()

        # Vector: distances must be converted to higher-is-better similarity.
        vector_hits = await store.vector_search(chunks[0].dense, top_k=3)
        assert vector_hits
        assert vector_hits[0].score >= vector_hits[-1].score, "scores must descend"
        assert 0.0 <= vector_hits[0].score <= 1.0001
    finally:
        # Drop what this run created. The fixture names tables per-run to keep
        # concurrent runs isolated, so without this every run leaks two tables
        # forever — and they are not merely clutter: they land in the schema
        # summary that Text-to-SQL is shown, so a few dozen accumulated
        # `ragorc_test_chunks_*` tables become prompt payload and a distraction
        # for the model on every query.
        async with pool.connection() as conn:
            for table in tables:
                await conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')  # noqa: S608
        await store.close()


async def test_postgres_readonly_guard_blocks_writes(settings: Settings) -> None:
    """The last layer of defence: even a statement that somehow passed the guard
    must fail against a read-only transaction."""
    import psycopg

    from ragorc.stores.postgres.store import PostgresStore

    store = PostgresStore(settings=settings)
    try:
        with pytest.raises((psycopg.errors.ReadOnlySqlTransaction, psycopg.Error)):
            await store.execute_readonly("CREATE TABLE should_not_exist (id int)")
    finally:
        await store.close()


async def test_postgres_schema_summary_is_compact(settings: Settings) -> None:
    from ragorc.stores.postgres.store import PostgresStore

    store = PostgresStore(settings=settings)
    try:
        summary = await store.schema_summary(refresh=True)
        assert summary, "an empty schema summary makes Text-to-SQL hallucinate columns"
        # It goes into every Text-to-SQL prompt, so size is a cost concern.
        assert len(summary) < 20_000, f"schema summary is {len(summary)} chars"
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------
async def test_neo4j_upsert_and_traverse(settings: Settings) -> None:
    from ragorc.stores.neo4j.store import Neo4jStore

    store = Neo4jStore(settings=settings)
    label = settings.neo4j.node_label
    try:
        entities = [
            Entity(name="Northwind", type="ORGANIZATION", description="a distributor"),
            Entity(name="Acme Supply", type="ORGANIZATION", description="a supplier"),
            Entity(name="Contoso", type="ORGANIZATION", description="a parent company"),
        ]
        relations = [
            Relation("Northwind", "Acme Supply", "SUPPLIED_BY", weight=3.0),
            Relation("Acme Supply", "Contoso", "OWNED_BY", weight=2.0),
        ]
        assert await store.upsert_entities(entities) == 3
        assert await store.upsert_relations(relations) == 2

        found, edges = await store.neighbors(["Northwind"], hops=2)
        names = {e.name for e in found}
        assert "Acme Supply" in names
        assert "Contoso" in names, "two-hop expansion did not reach the second hop"
        assert edges

        # The multi-hop question no vector search can answer.
        paths = await store.paths(["Northwind"], ["Contoso"], max_hops=3)
        assert paths, "no path found between entities that are connected"
        assert paths[0].hops == 2
        assert "Acme Supply" in paths[0].nodes
    finally:
        await store.execute_readonly(f"MATCH (n:{label}) RETURN count(n) AS n")
        await store.close()


async def test_neo4j_rejects_a_write_through_execute_readonly(settings: Settings) -> None:
    from ragorc.stores.neo4j.store import Neo4jStore

    store = Neo4jStore(settings=settings)
    try:
        with pytest.raises(Exception):  # noqa: B017 - driver raises a Neo4jError subclass
            await store.execute_readonly("CREATE (n:ShouldNotExist) RETURN n")
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# Cross-store
# ---------------------------------------------------------------------------
async def test_all_three_stores_are_reachable(settings: Settings) -> None:
    """A smoke check worth having on its own: it distinguishes 'the code is
    broken' from 'the stack is not up', which is the first question when an
    integration run fails."""
    from ragorc.stores.neo4j.store import Neo4jStore
    from ragorc.stores.postgres.store import PostgresStore
    from ragorc.stores.qdrant.store import QdrantStore

    qdrant = QdrantStore(settings=settings)
    postgres = PostgresStore(settings=settings)
    neo4j = Neo4jStore(settings=settings)
    try:
        await qdrant.ensure_collection(recreate=True)
        assert await qdrant.count() == 0
        assert await postgres.execute_readonly("SELECT 1 AS ok") == [{"ok": 1}]
        assert await neo4j.execute_readonly("RETURN 1 AS ok") == [{"ok": 1}]
    finally:
        await drop_collection(qdrant)
        await qdrant.close()
        await postgres.close()
        await neo4j.close()


async def test_neo4j_traversal_does_not_walk_through_chunks_or_communities(
    settings: Settings,
) -> None:
    """A graph traversal must return relationships a document asserted.

    `MENTIONS` and `IN_COMMUNITY` are the library's own scaffolding, not
    extracted knowledge. Untyped, a two-hop expansion walked straight through
    them, and the result was not a weaker answer but a different claim:

        A -[:MENTIONS]- chunk -[:MENTIONS]- B   "they appear in one passage"
        A -[:IN_COMMUNITY]-> c <-[:IN_COMMUNITY]- B   "Leiden grouped them"

    Both were returned as extracted graph relationships, and `startNode(r)` on a
    `MENTIONS` edge is a `Chunk`, so `_entity_from_props` built an "entity" out
    of one. `paths()` is the worse case: it is the whole answer to "how is A
    related to B", and it was answering with co-occurrence.

    Only reachable against a real server — the Cypher *is* the thing under test.
    """
    from ragorc.stores.neo4j.store import Neo4jStore

    store = Neo4jStore(settings=settings)
    tag = uuid.uuid4().hex[:8]
    linked, other, isolated = f"Linked{tag}", f"Other{tag}", f"Isolated{tag}"
    chunk_id = f"chunk-{tag}"
    try:
        await store.upsert_entities(
            [
                Entity(name=n, type="ORGANIZATION", description="d")
                for n in (linked, other, isolated)
            ]
        )
        await store.upsert_relations([Relation(linked, other, "PARTNERS_WITH", weight=2.0)])
        # `isolated` shares a chunk with `linked` and nothing else. Co-occurrence,
        # not a relationship. The mapping is `{entity: [chunk_ids]}` — inverted, it
        # silently creates an *Entity* named after the chunk and the traversal has
        # nothing to walk through, so the test would pass without the filter.
        await store.upsert_chunk_links({linked: [chunk_id], isolated: [chunk_id]})

        found, edges = await store.neighbors([linked], hops=2)
        names = {e.name for e in found}
        assert other in names, "the real relationship must survive the filter"
        assert isolated not in names, "co-mention in one chunk is not a graph edge"
        assert all(e.type not in {"MENTIONS", "IN_COMMUNITY"} for e in edges), (
            f"structural edges leaked into the relations: {[e.type for e in edges]}"
        )

        assert await store.paths([linked], [other], max_hops=3), (
            "the genuine path must still be found"
        )
        assert await store.paths([linked], [isolated], max_hops=3) == [], (
            "'how is A related to B' must not answer with 'the same chunk mentions both'"
        )

        centrality = await store.degree_centrality([linked, other, isolated])
        assert centrality[isolated] == 0.0, (
            "an entity with no relationships is not central because chunks mention it"
        )
    finally:
        with contextlib.suppress(Exception):
            await store._write_count(
                f"MATCH (n:`{settings.neo4j.node_label}`) WHERE n.name ENDS WITH $tag "
                "DETACH DELETE n RETURN count(*) AS written",
                {"tag": tag},
                label="cleanup",
            )
            await store._write_count(
                f"MATCH (c:`{settings.neo4j.chunk_label}` {{id: $id}}) "
                "DETACH DELETE c RETURN count(*) AS written",
                {"id": chunk_id},
                label="cleanup",
            )
        await store.close()


async def test_neo4j_prunes_superseded_communities_but_not_other_batches(
    settings: Settings,
) -> None:
    """A rebuilt community must replace the old one, not join it.

    Community ids are a hash of level plus membership — which is what makes a
    re-ingest converge instead of duplicating, and is also why a *changed*
    document produces a community with a new id rather than an updated node.
    Nothing deleted the old one, so `communities()` returned both and global
    search mapped over a report describing a partition that no longer existed.

    The second half is the one that makes the fix safe. Detection runs over the
    entities of the chunks *being ingested*, not the whole graph, so "delete
    every community not in this build" would drop the global-search index for
    every document already indexed.
    """
    from ragorc.core.models import Community
    from ragorc.stores.neo4j.store import Neo4jStore

    store = Neo4jStore(settings=settings)
    tag = uuid.uuid4().hex[:8]
    mine_a, mine_b, theirs = f"Mine{tag}A", f"Mine{tag}B", f"Theirs{tag}"
    superseded, untouched, rebuilt = 990_001, 990_002, 990_003
    try:
        await store.upsert_entities(
            [Entity(name=n, type="ORGANIZATION", description="d") for n in (mine_a, mine_b, theirs)]
        )
        await store.upsert_communities(
            [
                Community(
                    id=superseded,
                    level=0,
                    entity_names=(mine_a, mine_b),
                    title="old",
                    summary="a stale report",
                ),
                Community(
                    id=untouched,
                    level=0,
                    entity_names=(theirs,),
                    title="another batch",
                    summary="still live",
                ),
            ]
        )
        # The document behind {mine_a, mine_b} is re-ingested; membership shifted,
        # so detection emits a new id for the same entities.
        await store.upsert_communities(
            [
                Community(
                    id=rebuilt,
                    level=0,
                    entity_names=(mine_a, mine_b),
                    title="new",
                    summary="a fresh report",
                )
            ]
        )

        assert await store.prune_communities(keep_ids=[rebuilt], entity_names=[mine_a, mine_b]) == 1

        live = {c.id for c in await store.communities()}
        assert superseded not in live, "the superseded report is still being searched"
        assert rebuilt in live
        assert untouched in live, (
            "a community whose members this build never touched was deleted; that is "
            "the global-search index of every other document"
        )
    finally:
        with contextlib.suppress(Exception):
            await store._write_count(
                f"MATCH (c:`{settings.neo4j.community_label}`) WHERE c.id IN $ids "
                "DETACH DELETE c RETURN count(*) AS written",
                {"ids": [superseded, untouched, rebuilt]},
                label="cleanup",
            )
            await store._write_count(
                f"MATCH (n:`{settings.neo4j.node_label}`) WHERE n.name CONTAINS $tag "
                "DETACH DELETE n RETURN count(*) AS written",
                {"tag": tag},
                label="cleanup",
            )
        await store.close()
