"""The structured retrieval legs: text-to-SQL, text-to-Cypher, pgvector, Postgres FTS.

Both modules open with the same promise about *how they fail*, and nothing else in
the suite holds them to it. A leaf retriever must let
:class:`~ragorc.core.errors.StoreUnavailable` through — only the fan-out layer
knows which store a coroutine belonged to, so only it can record the outage
against a name — while degrading a refused generated query to no evidence,
because a model writing SQL the guard rejects says nothing about the health of
Postgres.

Both halves are asserted here, from both directions, because they share a single
``except`` clause. Widen that tuple by one name and a dead database starts
returning ``[]``, which is indistinguishable from "we looked and there was
nothing": the pipeline then answers confidently about a store it never reached,
and no other test in the suite turns red.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from ragorc.core.errors import RetrievalError, StoreUnavailable
from ragorc.core.models import (
    Chunk,
    Entity,
    FloatArray,
    Query,
    RetrievalSource,
    ScoredChunk,
)
from ragorc.core.schemas import CypherQuery, SQLQuery
from ragorc.core.settings import Settings
from ragorc.retrieve.cypher import CypherRetriever
from ragorc.retrieve.sql import (
    PgFullTextRetriever,
    PgVectorRetriever,
    SQLRetriever,
    label_and_rank,
)
from tests.fakes import FakeGraphStore, FakeRelationalStore, FakeVectorStore, StubEmbedder, StubLLM

# ---------------------------------------------------------------------------
# Settings, stores and stubs
# ---------------------------------------------------------------------------
_BASE = {
    "environment": "dev",
    "cache": {"enabled": False},
    "llm": {"api_key": "test-key"},
    "embedding": {"dense_dimension": 32},
}


@pytest.fixture
def sql_settings() -> Settings:
    """Guards on (as in production) and tenant isolation off.

    Isolation on is not the neutral setting for a generated-query leg: it makes
    the leg refuse outright, which is its own assertion further down rather than
    a background condition for every other test here.
    """
    return Settings(**_BASE, security={"enforce_tenant_isolation": False})


@pytest.fixture
def cypher_settings() -> Settings:
    """As ``sql_settings``, minus the ``EXPLAIN`` dry run.

    The dry run sends the statement through the same store, so leaving it on
    would put two entries in ``executed`` and blur the only question these tests
    ask of it: what reached the database, and in what order.

    It also changes the failure policy, which is worth stating rather than
    stumbling over: ``CypherGuard.explain`` catches every exception the store
    raises and re-raises it as a ``GuardrailViolation``, so with the dry run on a
    dead Neo4j arrives here as a rejected query and is degraded to no evidence.
    The outage test below therefore pins the retriever's own ``except`` clause,
    not that whole path.
    """
    return Settings(
        **_BASE,
        security={"enforce_tenant_isolation": False, "cypher_explain_dryrun": False},
    )


def sql_llm(sql: str) -> StubLLM:
    return StubLLM(responses={"SQLQuery": SQLQuery(sql=sql)})


def cypher_llm(cypher: str) -> StubLLM:
    return StubLLM(responses={"CypherQuery": CypherQuery(cypher=cypher)})


def graph_store_with_two_companies() -> FakeGraphStore:
    store = FakeGraphStore()
    store.entities["acme"] = Entity(name="Acme", type="Company")
    store.entities["beta"] = Entity(name="Beta", type="Company")
    return store


class _DeadRelationalStore(FakeRelationalStore):
    """Postgres answers for the schema, then dies before the query runs.

    The realistic shape of an outage on this leg: the schema summary is cached,
    so the first thing that actually needs a connection is execution.
    """

    async def execute_readonly(
        self, sql: str, params: Sequence[Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        raise StoreUnavailable("postgres", "connection pool exhausted")


class _DeadGraphStore(FakeGraphStore):
    """Neo4j reachable for the schema, gone by the time the statement runs."""

    async def execute_readonly(
        self, cypher: str, params: dict[str, Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        raise StoreUnavailable("neo4j", "bolt handshake failed")


class _RecordingRelationalStore(FakeRelationalStore):
    """``FakeRelationalStore`` plus the full-text call arguments it drops.

    The fake records executed SQL but not the keyword arguments of
    ``fulltext_search``, and the tenant scope arrives as one of those.
    """

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__(rows=rows)
        self.fulltext_calls: list[dict[str, Any]] = []

    async def fulltext_search(
        self, query: str, *, top_k: int = 10, **kwargs: Any
    ) -> list[ScoredChunk]:
        self.fulltext_calls.append({"query": query, "top_k": top_k, **kwargs})
        return await super().fulltext_search(query, top_k=top_k, **kwargs)


class _PgVectorStore:
    """The pgvector surface (``vector_search``) over ``FakeVectorStore``.

    Deliberately an adapter and not a new fake: ranking, filtering and rank
    stamping stay the existing store's real cosine implementation, so the only
    thing added is the signature this retriever calls and a record of the scoping
    arguments it is responsible for forwarding.
    """

    def __init__(self, inner: FakeVectorStore) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    async def vector_search(
        self,
        query_vector: FloatArray,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        self.calls.append({"top_k": top_k, "filters": filters, "tenant_id": tenant_id})
        probe = Query(text="", top_k=top_k or 10, dense=query_vector)
        return await self.inner.search(probe, top_k=top_k, filters=filters)


# ---------------------------------------------------------------------------
# SQL leg: failure policy
# ---------------------------------------------------------------------------
async def test_sql_retriever_lets_a_dead_postgres_through(sql_settings: Settings) -> None:
    """An outage has to reach the fan-out layer, which is the only place that
    knows this coroutine was the Postgres leg and can name it in
    ``RetrievalResult.errors``. Swallowed here it becomes an empty result, and an
    empty result reads as "consulted, found nothing" — so the pipeline answers
    from the vector leg alone while believing it also asked the database."""
    store = _DeadRelationalStore()
    retriever = SQLRetriever(sql_llm("SELECT country FROM orders"), store, settings=sql_settings)

    with pytest.raises(StoreUnavailable) as caught:
        await retriever.retrieve(Query(text="which countries order most?"))

    assert caught.value.store == "postgres", "the fan-out layer keys the outage off this name"


async def test_sql_retriever_degrades_when_the_guard_refuses_the_draft(
    sql_settings: Settings,
) -> None:
    """The mirror image: a rejected statement must *not* fail a request the other
    legs can still answer, and must not reach the database on its way out. A
    blocked pattern is refused twice (the repair cannot argue the guard out of
    it), which is what turns into ``ConstructionError``."""
    store = FakeRelationalStore()
    llm = sql_llm("DROP TABLE customers")
    retriever = SQLRetriever(llm, store, settings=sql_settings)

    assert await retriever.retrieve(Query(text="drop the customers table")) == []
    assert store.executed == [], "the guard refused, so nothing may have reached Postgres"
    assert len([c for c in llm.calls if c["kind"] == "structured"]) == 2, "one draft, one repair"


async def test_sql_retriever_degrades_when_isolation_forbids_generated_sql() -> None:
    """With tenant isolation on, this library will not scope a generated
    statement, so the leg refuses — and that refusal is a policy decision, not an
    outage. It must degrade like any other rejection, before a token is spent."""
    settings = Settings(
        **_BASE,
        security={"enforce_tenant_isolation": True, "generated_query_isolation": "reject"},
        tenant_id="acme",
    )
    store = FakeRelationalStore()
    llm = sql_llm("SELECT country FROM orders")
    retriever = SQLRetriever(llm, store, settings=settings)

    assert await retriever.retrieve(Query(text="orders by country", tenant_id="acme")) == []
    assert store.executed == []
    assert llm.call_count == 0, "refused before the model was asked, not after"


async def test_sql_retriever_without_a_store_is_a_configuration_error(
    sql_settings: Settings,
) -> None:
    """A missing store is a wiring mistake, and the one failure that must be loud:
    degrading it to ``[]`` would hide a leg that never ran in any request."""
    retriever = SQLRetriever(sql_llm("SELECT 1 AS n"), None, settings=sql_settings)

    with pytest.raises(RetrievalError):
        await retriever.retrieve(Query(text="anything"))

    late_store = FakeRelationalStore(rows=[{"n": 1}])
    out = await retriever.retrieve(Query(text="anything"), store=late_store)
    assert [c.chunk.content for c in out] == ["n: 1"], "a per-call store is accepted"


# ---------------------------------------------------------------------------
# SQL leg: what reaches the database, and what comes back
# ---------------------------------------------------------------------------
async def test_sql_retriever_runs_and_cites_the_guarded_rewrite(sql_settings: Settings) -> None:
    """The statement that executes is the guard's rewrite, not the model's draft —
    the draft had no ``LIMIT`` — and the evidence chunk quotes that same rewrite.
    Provenance that cited the draft would be unresolvable: a reader auditing the
    answer would re-run a query that never ran."""
    store = FakeRelationalStore(
        rows=[{"country": "US", "total": 4200}, {"country": "GB", "total": 3100}]
    )
    retriever = SQLRetriever(
        sql_llm("SELECT country, sum(total) AS total FROM orders GROUP BY country"),
        store,
        settings=sql_settings,
    )

    out = await retriever.retrieve(Query(text="revenue by country"))

    assert store.executed == [
        "SELECT country, SUM(total) AS total FROM orders GROUP BY country LIMIT 200"
    ]
    assert len(out) == 1, "a result set is one piece of evidence, not one chunk per row"
    assert out[0].chunk.metadata["sql"] == store.executed[0]
    assert out[0].chunk.metadata["row_count"] == 2
    assert out[0].chunk.content == (
        "| country | total |\n| --- | --- |\n| US | 4200 |\n| GB | 3100 |"
    )


async def test_sql_evidence_names_the_leg_that_produced_it(sql_settings: Settings) -> None:
    """Provenance is the retriever's own to assert: the row renderer knows how to
    format a result set, only the retriever knows which leg fetched it. The
    context packer prints that to the generator and fusion keys explanations off
    it, so an unlabelled chunk is evidence nobody can attribute."""
    store = FakeRelationalStore(rows=[{"n": 7}])
    retriever = SQLRetriever(
        sql_llm("SELECT count(*) AS n FROM orders"), store, settings=sql_settings
    )

    out = await retriever.retrieve(Query(text="how many orders?"))

    assert [c.source for c in out] == [RetrievalSource.SQL]
    assert [c.rank for c in out] == [0]
    assert out[0].explain["retriever"] == "sql"


async def test_sql_evidence_declares_its_score_a_constant(sql_settings: Settings) -> None:
    """A ``SELECT`` matched or it did not — there is no gradient — so the score is
    whatever confidence the operator configured for the leg, and downstream code
    has to be able to see that before it fuses on magnitude or applies a relative
    cutoff to it."""
    store = FakeRelationalStore(rows=[{"n": 7}])
    retriever = SQLRetriever(
        sql_llm("SELECT count(*) AS n FROM orders"),
        store,
        confidence=0.42,
        settings=sql_settings,
    )

    out = await retriever.retrieve(Query(text="how many orders?"))

    assert out[0].score == 0.42, "the retriever's confidence, not the constructor's default"
    assert out[0].component_scores == {"sql": 0.42}
    assert out[0].explain["constant_score"] is True
    assert out[0].explain["fusion_note"] == "rank-based fusion only; the score is a constant"


@pytest.mark.parametrize(
    ("query_tenant", "expected"),
    [("beta", "beta"), (None, "acme")],
    ids=["query wins", "settings fallback"],
)
async def test_sql_evidence_carries_the_resolved_tenant(
    query_tenant: str | None, expected: str
) -> None:
    """A generated statement cannot be tenant-scoped, so the tenant on the chunk
    is all the downstream isolation checks have to work with. Losing the
    configured fallback would let a single-tenant deployment emit unlabelled
    evidence that a later filter cannot place."""
    settings = Settings(
        **_BASE,
        security={"enforce_tenant_isolation": False, "generated_query_isolation": "trusted"},
        tenant_id="acme",
    )
    store = FakeRelationalStore(rows=[{"n": 1}])
    retriever = SQLRetriever(sql_llm("SELECT count(*) AS n FROM orders"), store, settings=settings)

    out = await retriever.retrieve(Query(text="how many?", tenant_id=query_tenant))

    assert [c.chunk.tenant_id for c in out] == [expected]


# ---------------------------------------------------------------------------
# Cypher leg: the same policy, enforced separately
# ---------------------------------------------------------------------------
async def test_cypher_retriever_lets_a_dead_neo4j_through(cypher_settings: Settings) -> None:
    """Same reason as the relational leg: a swallowed outage becomes an empty
    traversal, and "the graph knows nothing about this" is a very different
    answer from "we never asked the graph"."""
    store = _DeadGraphStore()
    retriever = CypherRetriever(
        cypher_llm("MATCH (a:Company) RETURN a.name"), store, settings=cypher_settings
    )

    with pytest.raises(StoreUnavailable) as caught:
        await retriever.retrieve(Query(text="which companies exist?"))

    assert caught.value.store == "neo4j"


async def test_cypher_retriever_degrades_when_the_guard_refuses_a_write(
    cypher_settings: Settings,
) -> None:
    """A write the guard blocked must cost the request nothing but the graph leg,
    and must never reach the database — not even as the repair's second attempt,
    which is refused for the same keyword."""
    store = graph_store_with_two_companies()
    retriever = CypherRetriever(
        cypher_llm("MATCH (n) DETACH DELETE n RETURN n"), store, settings=cypher_settings
    )

    assert await retriever.retrieve(Query(text="wipe the graph")) == []
    assert store.executed == [], "a refused statement must not reach Neo4j"


async def test_cypher_retriever_degrades_when_isolation_forbids_generated_cypher() -> None:
    """The graph leg refuses under enforced isolation for the same reason the
    relational one does, and the refusal degrades rather than failing the
    request."""
    settings = Settings(
        **_BASE,
        security={
            "enforce_tenant_isolation": True,
            "generated_query_isolation": "reject",
            "cypher_explain_dryrun": False,
        },
        tenant_id="acme",
    )
    store = graph_store_with_two_companies()
    llm = cypher_llm("MATCH (a:Company) RETURN a.name")
    retriever = CypherRetriever(llm, store, settings=settings)

    assert await retriever.retrieve(Query(text="companies", tenant_id="acme")) == []
    assert store.executed == []
    assert llm.call_count == 0, "refused before the model was asked"


async def test_cypher_retriever_without_a_store_is_a_configuration_error(
    cypher_settings: Settings,
) -> None:
    """As with SQL: an unwired leg must be visible, not silently absent."""
    retriever = CypherRetriever(
        cypher_llm("MATCH (a:Company) RETURN a.name"), None, settings=cypher_settings
    )

    with pytest.raises(RetrievalError):
        await retriever.retrieve(Query(text="anything"))

    out = await retriever.retrieve(Query(text="anything"), store=graph_store_with_two_companies())
    assert [c.source for c in out] == [RetrievalSource.CYPHER], "a per-call store is accepted"


async def test_cypher_retriever_runs_and_cites_the_bounded_rewrite(
    cypher_settings: Settings,
) -> None:
    """The guard appends the row bound, and the statement that ran is the one the
    chunk cites — an unbounded ``MATCH`` is an outage, so the draft is never the
    thing that executes or the thing provenance points at."""
    store = graph_store_with_two_companies()
    retriever = CypherRetriever(
        cypher_llm("MATCH (a:Company) RETURN a.name, a.type"), store, settings=cypher_settings
    )

    out = await retriever.retrieve(Query(text="which companies exist?"))

    assert store.executed == ["MATCH (a:Company) RETURN a.name, a.type\nLIMIT 200"]
    assert len(out) == 1, "one chunk for the whole result: rows are meaningless apart"
    assert out[0].chunk.metadata["cypher"] == store.executed[0]
    assert out[0].chunk.content == "name: Acme | type: Company\nname: Beta | type: Company"


async def test_cypher_evidence_names_its_leg_and_flags_the_constant(
    cypher_settings: Settings,
) -> None:
    """``MATCH`` is a pattern match, so the ordering came from an ``ORDER BY`` the
    model wrote and the score means nothing. Rank fusion is the only sound merge,
    and it can only be chosen if the chunk says so."""
    store = graph_store_with_two_companies()
    retriever = CypherRetriever(
        cypher_llm("MATCH (a:Company) RETURN a.name"),
        store,
        confidence=0.42,
        settings=cypher_settings,
    )

    out = await retriever.retrieve(Query(text="which companies exist?"))

    assert [c.source for c in out] == [RetrievalSource.CYPHER]
    assert [c.rank for c in out] == [0]
    assert out[0].score == 0.42, "the retriever's confidence, not the constructor's 0.95"
    assert out[0].component_scores == {"cypher": 0.42}
    assert out[0].explain["retriever"] == "cypher"
    assert out[0].explain["constant_score"] is True
    assert out[0].explain["fusion_note"] == "rank-based fusion only; the score is a constant"


@pytest.mark.parametrize(
    ("query_tenant", "expected"),
    [("beta", "beta"), (None, "acme")],
    ids=["query wins", "settings fallback"],
)
async def test_cypher_evidence_carries_the_resolved_tenant(
    query_tenant: str | None, expected: str
) -> None:
    """Graph evidence is stamped from the same two sources in the same order, and
    the chunk id is derived from the tenant — an unstamped chunk collides across
    tenants in the cache as well as escaping the filter."""
    settings = Settings(
        **_BASE,
        security={
            "enforce_tenant_isolation": False,
            "generated_query_isolation": "trusted",
            "cypher_explain_dryrun": False,
        },
        tenant_id="acme",
    )
    store = graph_store_with_two_companies()
    retriever = CypherRetriever(
        cypher_llm("MATCH (a:Company) RETURN a.name"), store, settings=settings
    )

    out = await retriever.retrieve(Query(text="companies", tenant_id=query_tenant))

    assert [c.chunk.tenant_id for c in out] == [expected]


# ---------------------------------------------------------------------------
# pgvector
# ---------------------------------------------------------------------------
async def test_pgvector_forwards_the_tenant_and_filters_to_the_store(
    sql_settings: Settings, chunks: list[Chunk]
) -> None:
    """Scoping only isolates anything if it survives the hop into the store. This
    leg holds both halves of it — the tenant and the caller's filters — and the
    explicit ``top_k`` has to beat the query's, or a caller's over-fetch for
    reranking silently collapses to the query default."""
    inner = FakeVectorStore()
    await inner.upsert(chunks)
    store = _PgVectorStore(inner)
    retriever = PgVectorRetriever(store, settings=sql_settings)
    query = Query(text="refunds", top_k=7, dense=chunks[0].dense, tenant_id="acme")

    await retriever.retrieve(query, top_k=3, filters={"document_id": "doc-1"})

    assert store.calls == [{"top_k": 3, "filters": {"document_id": "doc-1"}, "tenant_id": "acme"}]


async def test_pgvector_reuses_the_vector_the_pipeline_already_embedded(
    sql_settings: Settings, chunks: list[Chunk]
) -> None:
    """Every dense leg of a fan-out shares one query vector. Re-embedding here
    would double the cost for an identical result, and a second embedder with a
    different asymmetric prefix would search a space the index was not built in —
    a silent quality loss, not an error."""
    store = _PgVectorStore(FakeVectorStore())
    embedder = StubEmbedder(dimension=32)
    retriever = PgVectorRetriever(store, embedder=embedder, settings=sql_settings)

    await retriever.retrieve(Query(text="refunds", dense=chunks[0].dense))
    assert embedder.calls == [], "the vector was already on the query"

    fresh = Query(text="refunds")
    await retriever.retrieve(fresh)
    assert embedder.calls == [("embed_query", 1)], "embedded exactly once when absent"
    assert fresh.dense is not None, "and cached on the query for the next leg"


async def test_pgvector_without_a_vector_or_an_embedder_is_an_error(
    sql_settings: Settings,
) -> None:
    """Searching with no query vector cannot be degraded into a narrower answer —
    there is nothing to search with — so it must be raised rather than returning
    an empty list a caller would read as "no matches"."""
    retriever = PgVectorRetriever(_PgVectorStore(FakeVectorStore()), settings=sql_settings)

    with pytest.raises(RetrievalError):
        await retriever.retrieve(Query(text="refunds"))


async def test_pgvector_labels_results_dense_and_ranks_them_in_store_order(
    sql_settings: Settings, chunks: list[Chunk]
) -> None:
    """pgvector scores are cosine similarities, directly comparable with Qdrant's,
    so this leg is labelled ``dense`` and not something structured — mislabel it
    and score-based fusion would treat a real gradient as a constant. Rank is
    re-stamped from the store's order because fusion reads rank as truth."""
    inner = FakeVectorStore()
    await inner.upsert(chunks)
    retriever = PgVectorRetriever(_PgVectorStore(inner), settings=sql_settings)

    out = await retriever.retrieve(Query(text="refunds", dense=chunks[2].dense), top_k=3)

    assert [c.chunk.id for c in out] == ["c2", "c1", "c3"], "nearest first, by cosine"
    assert [c.source for c in out] == [RetrievalSource.DENSE] * 3
    assert [c.rank for c in out] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Postgres full text
# ---------------------------------------------------------------------------
async def test_pg_fulltext_skips_a_blank_question_without_a_round_trip(
    sql_settings: Settings,
) -> None:
    """``to_tsquery('')`` matches nothing, so the round trip is pure latency on a
    query the caller already knows is empty — and a whitespace-only question is
    what an over-eager rewrite produces."""
    store = _RecordingRelationalStore(rows=[{"content": "refunds take 14 days"}])
    retriever = PgFullTextRetriever(store, settings=sql_settings)

    assert await retriever.retrieve(Query(text="   \n ")) == []
    assert store.fulltext_calls == [], "nothing to search for means no query"


async def test_pg_fulltext_forwards_the_tenant_and_labels_results_fulltext(
    sql_settings: Settings,
) -> None:
    """The lexical leg is scoped like every other, and its label matters twice
    over: ``ts_rank_cd`` is a gradient on a scale unrelated to cosine similarity,
    so fusion has to know which leg it came from to weight it at all."""
    store = _RecordingRelationalStore(
        rows=[{"content": "refunds take 14 days"}, {"content": "shipping is free"}]
    )
    retriever = PgFullTextRetriever(store, settings=sql_settings)
    query = Query(text="  refunds  ", top_k=5, filters={"lang": "en"}, tenant_id="acme")

    out = await retriever.retrieve(query)

    assert store.fulltext_calls == [
        {"query": "refunds", "top_k": 5, "filters": {"lang": "en"}, "tenant_id": "acme"}
    ]
    assert [c.chunk.id for c in out] == ["pg-0", "pg-1"]
    assert [c.source for c in out] == [RetrievalSource.FULLTEXT] * 2
    assert [c.rank for c in out] == [0, 1]


# ---------------------------------------------------------------------------
# label_and_rank
# ---------------------------------------------------------------------------
def test_label_and_rank_replaces_a_stale_rank() -> None:
    """Rank is re-stamped rather than trusted: a caller may have filtered or
    reordered between the store and here, and RRF reads whatever integer it finds
    as the position — a leftover rank 7 on a first-place hit contributes a
    seventh-place vote."""
    stale = [
        ScoredChunk(chunk=Chunk(id="a", content="a"), score=0.9, rank=7),
        ScoredChunk(chunk=Chunk(id="b", content="b"), score=0.5, rank=3),
    ]

    out = label_and_rank(stale, retriever="pgvector", source=RetrievalSource.DENSE)

    assert [c.chunk.id for c in out] == ["a", "b"], "order is preserved, only rank is rewritten"
    assert [c.rank for c in out] == [0, 1]


def test_label_and_rank_overrides_the_source_the_builder_guessed() -> None:
    """The source is asserted by the retriever, never inherited from whatever
    built the chunk. A row renderer knows how to format a result set and defaults
    the label to something plausible; only the retriever knows which leg fetched
    it, and the context packer prints that provenance to the generator."""
    guessed = [ScoredChunk(chunk=Chunk(id="a", content="a"), score=1.0)]

    out = label_and_rank(guessed, retriever="sql", source=RetrievalSource.SQL)

    assert [c.source for c in out] == [RetrievalSource.SQL], "the builder guessed dense"


def test_label_and_rank_adds_a_leg_without_overwriting_an_earlier_one() -> None:
    """A chunk two legs both found carries both component scores, and the one
    already recorded is the one that leg reported. Overwriting it would rewrite
    history — "why did this rank third" is answered from these numbers after
    fusion has collapsed them into one."""
    scored = ScoredChunk(
        chunk=Chunk(id="a", content="a"),
        score=1.0,
        component_scores={"dense": 0.4, "sql": 0.5},
    )

    out = label_and_rank([scored], retriever="sql", source=RetrievalSource.SQL)

    assert out[0].component_scores == {"dense": 0.4, "sql": 0.5}
