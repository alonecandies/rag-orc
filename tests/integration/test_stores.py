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

import asyncio
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


from tests.integration.conftest import deployment_env_file  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    """A settings object pointed at the compose stack, with an isolated namespace
    per run so concurrent runs and reruns cannot collide.

    Connection details come from the deployment's own configuration — the same
    ``.env`` the CLI and the
    service read — rather than from literals here. They used to be hardcoded
    defaults, which meant these tests ignored the port in ``.env`` and connected to
    whatever answered on 5432. On a machine already running an unrelated Postgres
    there, the friendly outcome is an authentication failure; the unfriendly one is
    a test run that creates its tables in someone else's database.

    Only the fields that make a run *isolated* are overridden below.
    """
    suffix = uuid.uuid4().hex[:8]
    env = Settings(_env_file=deployment_env_file())
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


async def test_by_id_chunk_reads_are_scoped_in_both_stores(settings: Settings) -> None:
    """An id is not a filter, and both by-id reads used to prove it.

    `search` builds every filter through `with_tenant`/`_filter_clauses`, so the
    query paths were scoped. `QdrantStore.get` issued a bare `client.retrieve`
    and `PostgresStore.get_chunks` a bare `= ANY(array)`, so *naming* a chunk
    fetched its body whoever owned it. That is reachable because the ids on the
    GraphRAG path are resolved out of Neo4j, which stores no tenant at all.

    Both stores, against the live servers, because this is a filter-generation
    claim and the fakes cannot settle it.
    """
    from ragorc.stores.postgres.store import PostgresStore
    from ragorc.stores.qdrant.store import QdrantStore

    dim = settings.embedding.dense_dimension
    tag = uuid.uuid4().hex[:8]
    mine, theirs = f"mine-{tag}", f"theirs-{tag}"

    def chunk(cid: str, tenant: str, text: str) -> Chunk:
        c = Chunk(id=cid, content=text, document_id=f"doc-{tenant}-{tag}", tenant_id=tenant)
        c.dense = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        return c

    rows = [chunk(mine, "globex", "globex public note"), chunk(theirs, "acme", "ACME CONFIDENTIAL")]

    vector = QdrantStore(settings=settings)
    relational = PostgresStore(settings=settings)
    try:
        await vector.ensure_collection()
        await relational.ensure_schema()
        await vector.upsert(rows)
        await relational.upsert_documents(
            [Document(id=f"doc-{t}-{tag}", content="x", tenant_id=t) for t in ("globex", "acme")]
        )
        await relational.upsert_chunks(rows)

        for name, got in (
            ("qdrant", await vector.get([mine, theirs], tenant_id="globex")),
            ("postgres", await relational.get_chunks([mine, theirs], tenant_id="globex")),
        ):
            ids = [c.id for c in got]
            assert theirs not in ids, f"{name} returned another tenant's chunk by id: {ids}"
            assert ids == [mine], f"{name} lost the caller's own chunk: {ids}"

        # Unscoped, both are still reachable — the parameter is what scopes it,
        # so a caller that omits it (single-tenant, isolation off) is unaffected.
        assert len(await vector.get([mine, theirs])) == 2
    finally:
        with contextlib.suppress(Exception):
            await vector.drop_collection()
        with contextlib.suppress(Exception):
            for t in ("globex", "acme"):
                await relational.delete_document(f"doc-{t}-{tag}")
        await vector.close()
        await relational.close()

@pytest.mark.integration
async def test_correcting_a_document_drops_the_answers_built_from_it(
    settings: Settings,
) -> None:
    """The end-to-end shape of the staleness bug, against the real cache.

    Before the fix, this sequence — answer a question, correct the document,
    re-ingest — went on serving the answer computed from the old text for up to
    ``cache.semantic_ttl_s`` (one hour by default), with no citations attached,
    because a cache hit carries none.

    Exercised through ``IngestPipeline._purge`` rather than a full ingest: the
    purge is the step that knows which documents are being replaced, and driving
    a whole ingest here would make the test about embedding throughput.
    """
    from ragorc.cache.semantic import SemanticCache, scope_key
    from ragorc.index.pipeline import IngestPipeline, IngestReport

    cached = Settings(
        llm=settings.llm,
        qdrant=settings.qdrant,
        embedding={"dense_dimension": 32},
        cache={
            "enabled": True,
            "semantic_enabled": True,
            "semantic_collection": f"{settings.qdrant.collection}_semcache",
        },
    )
    cache = SemanticCache(_StubEmbedder32(), settings=cached)
    scope = scope_key(None, None)
    question = "what is the refund window?"

    try:
        await cache.set(
            question,
            {"text": "The refund window is 30 days."},
            tenant_id="acme",
            scope=scope,
            document_ids=["policy"],
        )
        await asyncio.sleep(0.4)
        hit = await cache.get(question, tenant_id="acme", scope=scope)
        assert hit is not None and "30 days" in hit.answer["text"]

        # The operator corrects the policy and re-ingests it.
        pipeline = IngestPipeline(settings=cached, answer_cache=cache)
        await pipeline._purge(
            [Document(id="policy", content="7 days", source="policy.md", tenant_id="acme")],
            IngestReport(),
        )
        await asyncio.sleep(0.4)

        assert await cache.get(question, tenant_id="acme", scope=scope) is None, (
            "the corrected document is still answered from the old one"
        )
    finally:
        await cache.clear()


class _StubEmbedder32:
    """Deterministic 32-d vectors. The semantic cache needs *an* embedder and the
    identity of the vectors is irrelevant here — what is under test is whether a
    stored point survives a filtered delete."""

    dimension = 32
    model_name = "stub"

    async def embed_query(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=32).astype(np.float32)
        return (v / np.linalg.norm(v)).astype(np.float32)

@pytest.mark.integration
async def test_deleting_a_document_leaves_entities_another_one_still_mentions(
    settings: Settings,
) -> None:
    """The rule the graph's shared-key model forces, against a real Neo4j.

    An ``Entity`` merges on ``name``, so two documents that both discuss "Acme"
    converge on one node. Deleting the first document must not take the entity
    with it, or the second document silently loses a node it still asserts. Only
    entities with no ``MENTIONS`` edge left may go.

    That rule has a second property worth pinning: in a graph that stores no
    tenant at all, it is tenant-safe by construction. Another tenant's surviving
    chunk keeps the edge, so the entity stays.
    """
    from ragorc.stores.neo4j.store import Neo4jStore

    store = Neo4jStore(settings=settings)
    try:
        await store.ensure_schema()
        await store.upsert_entities(
            [
                Entity(name="Acme", type="ORG", description="shared by both documents"),
                Entity(name="Solo", type="ORG", description="only in the doomed document"),
            ]
        )
        # doc-a owns chunk-a1; doc-b owns chunk-b1. Both mention Acme.
        await store.upsert_chunk_links({"Acme": ["chunk-a1", "chunk-b1"], "Solo": ["chunk-a1"]})

        counts = await store.delete_chunks(["chunk-a1"])

        assert counts["chunks"] == 1
        surviving = {e.name for e, _score in await store.fulltext_entities("Acme Solo", limit=10)}
        assert "Acme" in surviving, "an entity another document still mentions was deleted"
        assert "Solo" not in surviving, "an orphaned entity was left behind"
        assert counts["entities"] >= 1
    finally:
        async with store.driver.session() as session:
            await session.run(
                f"MATCH (n:`{settings.neo4j.node_label}`) DETACH DELETE n"
            )
            await session.run("MATCH (c:Chunk) WHERE c.id STARTS WITH 'chunk-' DETACH DELETE c")
        await store.close()


@pytest.mark.integration
async def test_search_returns_dense_vectors_only_when_a_stage_reads_them(
    settings: Settings,
) -> None:
    """Both halves against a real Qdrant: asked for, and attached on arrival.

    `with_vectors` was hardcoded `False`, so `chunk.dense` was `None` on every
    hit and MMR silently became `chunks[:k]`. Requesting them is only half a fix —
    `_to_scored` builds the chunk from the payload, so vectors could travel the
    wire and be dropped on receipt.
    """
    from ragorc.retrieve.noise import mmr_select
    from ragorc.stores.qdrant.store import QdrantStore

    def _store(mmr: bool) -> QdrantStore:
        return QdrantStore(
            Settings(
                llm=settings.llm,
                qdrant=settings.qdrant,
                embedding={"dense_dimension": 32},
                retrieval={"mmr_enabled": mmr, "compression_enabled": False},
                security={"enforce_tenant_isolation": False},
            )
        )

    store = _store(mmr=True)
    try:
        await store.ensure_collection()
        chunks = []
        for i in range(12):
            chunk = Chunk(id=f"v{i}", content=f"refund paragraph {i}", document_id="d")
            # Four directions among twelve chunks, so a diversity-aware pick and a
            # relevance-ordered one cannot coincide by luck.
            axis = np.zeros(32, dtype=np.float32)
            axis[i % 4] = 1.0
            chunk.dense = axis
            chunks.append(chunk)
        await store.upsert(chunks)
        await store.flush()

        query = Query(text="refund")
        query.dense = (np.ones(32, dtype=np.float32) / np.sqrt(32)).astype(np.float32)

        hits = await store.search(query, top_k=10)
        assert hits and all(h.chunk.dense is not None for h in hits), (
            "vectors were requested but not attached to the chunk"
        )

        # Asserted as a property of the selection, not as "differs from the
        # relevance order". The first version of this test used the latter and it
        # is ill-posed: with twelve chunks on four axes the relevance order can
        # already be diverse, and then a correctly-diversifying MMR reproduces it.
        # It passed on the ordering Qdrant happened to return and failed on the
        # next one, which is a flaky test, not a finding.
        picked = mmr_select(hits, k=4, lambda_mult=0.5)
        axes = {int(np.argmax(c.chunk.dense)) for c in picked}
        assert len(axes) == 4, (
            f"MMR picked {len(axes)} of the four available directions: "
            f"{[c.chunk.id for c in picked]}"
        )

        # And the inert path is still distinguishable: with the vectors stripped,
        # `mmr_select` falls back to truncation and cannot cover them all.
        for candidate in hits:
            candidate.chunk.dense = None
        blind = mmr_select(hits, k=4, lambda_mult=0.5)
        assert [c.chunk.id for c in blind] == [h.chunk.id for h in hits[:4]]

        # And the default configuration must not pay for what it will not read.
        off = _store(mmr=False)
        assert all(h.chunk.dense is None for h in await off.search(query, top_k=10))
        await off.close()
    finally:
        await drop_collection(store)
        await store.close()


@pytest.mark.integration
async def test_document_summaries_group_and_filter_in_postgres(settings: Settings) -> None:
    """Hand-written SQL, so only a real server can say whether it parses.

    `documents()` routes here rather than scrolling the vector store because a
    chunk count per document needs every chunk, and the scroll therefore could not
    stop at the limit — 2 000 points read to return three rows. A GROUP BY with a
    LIMIT answers the same question from an index, and its correctness is not
    something a fake relational store can attest to.
    """
    from ragorc.stores.postgres.store import PostgresStore

    store = PostgresStore(settings)
    try:
        await store.ensure_schema()
        await store.upsert_documents(
            [
                Document(id="policy", content="x", source="/corpus/policy.md"),
                Document(id="faq", content="y", source="/corpus/faq.md"),
            ]
        )
        await store.upsert_chunks(
            [
                Chunk(id="s1", content="a", document_id="policy"),
                Chunk(id="s2", content="b", document_id="policy"),
                Chunk(id="s3", content="c", document_id="faq"),
            ]
        )

        every = await store.document_summaries()
        assert [(r["document_id"], r["chunks"]) for r in every] == [("faq", 1), ("policy", 2)]
        assert every[1]["source"] == "/corpus/policy.md"

        assert [r["document_id"] for r in await store.document_summaries(source="POLICY")] == [
            "policy"
        ], "the source filter must be case-insensitive"
        assert len(await store.document_summaries(limit=1)) == 1
    finally:
        await store.close()


@pytest.mark.integration
async def test_a_join_keeps_every_column_and_every_decimal_digit(settings: Settings) -> None:
    """Both halves of the text-to-SQL evidence path, against a real server.

    `execute_readonly` used `dict_row`, and a dict cannot hold two columns of the
    same name — so `SELECT * FROM a JOIN b`, the commonest shape a model
    produces, returned three of six columns with the right-hand table's values
    under the left-hand table's question. And `_json_safe` converted NUMERIC to
    float, so `12.50` printed as `12.5` and `1234567890123456789.99` as
    `1.2345678901234568e+18` in a table the generator is instructed to reproduce
    verbatim.

    Neither is observable against a fake: one needs a real cursor description,
    the other a real NUMERIC column.
    """
    from ragorc.construct.text_to_sql import _cell, _columns
    from ragorc.stores.postgres.store import PostgresStore

    suffix = uuid.uuid4().hex[:8]
    store = PostgresStore(settings)
    orders, refunds = f"o_{suffix}", f"r_{suffix}"
    try:
        async with store._connection() as conn:
            await conn.execute(
                f"CREATE TABLE {orders} (id int, customer text, amount numeric(38,2))"
            )
            await conn.execute(
                f"CREATE TABLE {refunds} (id int, customer text, amount numeric(38,2))"
            )
            await conn.execute(
                f"INSERT INTO {orders} VALUES (1,'acme',12.50),"
                f"(2,'globex',1234567890123456789.99)"
            )
            await conn.execute(
                f"INSERT INTO {refunds} VALUES (1,'acme-refund',1.00),(2,'globex-refund',2.00)"
            )

        rows = await store.execute_readonly(
            f"SELECT * FROM {orders} o JOIN {refunds} r ON o.id = r.id ORDER BY o.id"
        )

        assert len(rows[0]) == 6, f"the join projects six columns; got {sorted(rows[0])}"
        assert list(_columns(rows)) == [
            "id",
            "customer",
            "amount",
            "id__2",
            "customer__2",
            "amount__2",
        ]
        assert rows[0]["customer"] == "acme", "the left table's value was overwritten"
        assert rows[0]["customer__2"] == "acme-refund"

        assert _cell(rows[0]["amount"]) == "12.50", "the stored scale was lost"
        assert _cell(rows[1]["amount"]) == "1234567890123456789.99", "digits were lost"
    finally:
        async with store._connection() as conn:
            for table in (orders, refunds):
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        await store.close()
