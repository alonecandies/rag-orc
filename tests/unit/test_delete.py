"""Taking a document out of the index.

There was no way to. The API was ``/health /metrics /query /query/stream /ingest
/eval`` and the CLI was ``init ingest query eval bench serve inspect
alias-swap``; the only deletion anywhere was the stale-purge inside re-ingest,
which *replaces* a document and cannot remove one. An operator asked to delete
had to drive three store objects by hand, and would still have missed the graph
— which had no delete at all — and the answer cache, which had no invalidation.

The interesting part is not that a delete exists but what it does when it only
half works. A delete that stops at the first failure is the worst outcome
available: the caller believes the document is gone while it is still
retrievable from whichever store was not reached. So every store is attempted
and the failures are reported.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.errors import StoreUnavailable
from ragorc.core.models import Chunk
from ragorc.core.settings import Settings
from ragorc.pipeline.builder import DeleteReport, RAGPipeline
from tests.fakes import FakeDocumentStore, FakeVectorStore


class FakeGraphDeleter:
    def __init__(self) -> None:
        self.deleted: list[list[str]] = []

    async def delete_chunks(self, chunk_ids: Any) -> dict[str, int]:
        ids = list(chunk_ids)
        self.deleted.append(ids)
        return {"chunks": len(ids), "entities": 2, "communities": 1}


class FakeAnswerCache:
    def __init__(self) -> None:
        self.invalidated: list[tuple[list[str], str | None]] = []

    async def invalidate(self, document_ids: Any, *, tenant_id: str | None = None) -> int:
        self.invalidated.append((list(document_ids), tenant_id))
        return len(list(document_ids))


def _settings(*, graph: bool = True, isolation: bool = False) -> Settings:
    return Settings(
        llm={"api_key": "k"},
        graph={"enabled": graph},
        cache={"enabled": False},
        embedding={"dense_dimension": 32},
        security={"enforce_tenant_isolation": isolation},
    )


async def _pipeline(
    *,
    graph: Any = None,
    cache: Any = None,
    relational: Any = None,
    settings: Settings | None = None,
    chunks: list[Chunk] | None = None,
) -> tuple[RAGPipeline, FakeVectorStore]:
    store = FakeVectorStore()
    if chunks:
        await store.upsert(chunks)
    pipeline = RAGPipeline(
        settings=settings or _settings(),
        vector_store=store,
        relational_store=relational,
        graph_store=graph,
    )
    pipeline._semantic_cache = cache
    return pipeline, store


def _chunk(cid: str, doc: str, tenant: str | None = None) -> Chunk:
    return Chunk(id=cid, content=f"body of {cid}", document_id=doc, tenant_id=tenant)


# ---------------------------------------------------------------------------
# It reaches every store
# ---------------------------------------------------------------------------
async def test_a_delete_reaches_all_four_stores() -> None:
    graph, cache = FakeGraphDeleter(), FakeAnswerCache()
    relational = FakeDocumentStore()
    pipeline, store = await _pipeline(
        graph=graph,
        cache=cache,
        relational=relational,
        chunks=[_chunk("c1", "policy"), _chunk("c2", "policy"), _chunk("c3", "faq")],
    )

    report = await pipeline.delete(["policy"])

    assert report.complete, report.errors
    assert [c.id async for c in store.scroll(filters={"document_id": ["policy"]})] == []
    assert [c.id async for c in store.scroll(filters={"document_id": ["faq"]})] == ["c3"]
    assert graph.deleted == [["c1", "c2"]], "the graph was given the deleted chunks' ids"
    assert cache.invalidated == [(["policy"], None)]
    assert "delete_document" in relational.calls


async def test_a_single_id_does_not_have_to_be_a_list() -> None:
    """`delete("policy")` iterating the string into six one-character ids is the
    kind of thing a caller finds out about in production."""
    graph = FakeGraphDeleter()
    pipeline, _store = await _pipeline(graph=graph, chunks=[_chunk("c1", "policy")])

    report = await pipeline.delete("policy")

    assert report.documents == 1


async def test_the_chunk_ids_are_read_before_anything_is_removed() -> None:
    """Ordering, not decoration. A `Chunk` node in Neo4j carries an id and no
    document reference, so once the vectors are gone there is no way left to find
    the nodes that referenced them — the graph would silently keep them forever.
    """
    graph = FakeGraphDeleter()
    pipeline, _store = await _pipeline(
        graph=graph, chunks=[_chunk("c1", "policy"), _chunk("c2", "policy")]
    )

    await pipeline.delete(["policy"])

    assert graph.deleted == [["c1", "c2"]], (
        "the graph got no ids, so the vectors were deleted before they were read"
    )


class _CountingScroll(FakeVectorStore):
    """Records how often the chunk-id lookup runs."""

    scrolls = 0

    async def scroll(self, **kwargs: Any):  # type: ignore[override]
        type(self).scrolls += 1
        async for chunk in super().scroll(**kwargs):
            yield chunk


async def test_the_graph_is_skipped_when_it_is_not_enabled() -> None:
    graph = FakeGraphDeleter()
    pipeline, _store = await _pipeline(
        graph=graph, settings=_settings(graph=False), chunks=[_chunk("c1", "policy")]
    )

    report = await pipeline.delete(["policy"])

    assert graph.deleted == []
    assert report.entities == 0


async def test_the_lookup_runs_even_with_the_graph_off() -> None:
    """A deliberate reversal, recorded because the previous test asserted the
    opposite.

    The scroll used to be skipped when the graph was off, on the grounds that the
    chunk ids existed only to hand to Neo4j. It has a second reader now: the
    report cannot say whether anything was *found* without asking, and every
    store's delete is "remove what matches", which succeeds identically over
    nothing. One scroll bounded by the documents being deleted is the price of a
    report that can tell a completed deletion from a typo.
    """
    _CountingScroll.scrolls = 0
    store = _CountingScroll()
    await store.upsert([_chunk("c1", "policy")])
    pipeline = RAGPipeline(
        settings=_settings(graph=False), vector_store=store, graph_store=FakeGraphDeleter()
    )
    pipeline._semantic_cache = None

    report = await pipeline.delete(["policy"])

    assert _CountingScroll.scrolls == 1
    assert report.found == 1 and report.deleted


async def test_deleting_nothing_is_not_an_error() -> None:
    pipeline, _store = await _pipeline()
    assert await pipeline.delete([]) == DeleteReport(documents=0)


async def test_deleting_what_is_already_gone_succeeds() -> None:
    """Idempotence is what makes a retry after a partial failure safe."""
    pipeline, _store = await _pipeline(graph=FakeGraphDeleter())

    first = await pipeline.delete(["never-existed"])
    second = await pipeline.delete(["never-existed"])

    assert first.complete and second.complete
    assert first.vectors == second.vectors == 0


# ---------------------------------------------------------------------------
# What it does when a store is down
# ---------------------------------------------------------------------------
class _BrokenGraph:
    async def delete_chunks(self, chunk_ids: Any) -> dict[str, int]:
        raise StoreUnavailable("neo4j is down")


async def test_one_store_failing_does_not_stop_the_others() -> None:
    """The whole design decision. Stopping at the first failure leaves the caller
    believing the document is gone while it is still retrievable elsewhere."""
    cache = FakeAnswerCache()
    pipeline, store = await _pipeline(
        graph=_BrokenGraph(), cache=cache, chunks=[_chunk("c1", "policy")]
    )

    report = await pipeline.delete(["policy"])

    assert not report.complete
    assert "graph" in report.errors
    assert [c.id async for c in store.scroll()] == [], "the vector store was still cleared"
    assert cache.invalidated, "the cache was still invalidated"


async def test_a_failure_is_reported_rather_than_raised() -> None:
    """A raise would discard the counts for the stores that succeeded, which is
    exactly what a caller needs in order to know what to retry."""
    pipeline, _store = await _pipeline(graph=_BrokenGraph(), chunks=[_chunk("c1", "policy")])

    report = await pipeline.delete(["policy"])

    assert isinstance(report, DeleteReport)
    assert report.vectors == 1
    assert report.errors["graph"].startswith("neo4j is down")


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------
async def test_a_delete_is_scoped_to_its_tenant() -> None:
    """A caller able to write under another tenant's id needs a second step to
    cause harm. One able to delete under it does not."""
    pipeline, store = await _pipeline(
        settings=_settings(isolation=True),
        chunks=[_chunk("c1", "policy", "acme"), _chunk("c2", "policy", "globex")],
    )

    await pipeline.delete(["policy"], tenant_id="globex")

    survivors = [c.id async for c in store.scroll()]
    assert survivors == ["c1"], f"acme's chunk was deleted by a globex request: {survivors}"


async def test_the_cache_invalidation_carries_the_tenant() -> None:
    cache = FakeAnswerCache()
    pipeline, _store = await _pipeline(
        cache=cache, settings=_settings(isolation=True), chunks=[_chunk("c1", "policy", "acme")]
    )

    await pipeline.delete(["policy"], tenant_id="acme")

    assert cache.invalidated == [(["policy"], "acme")]


async def test_an_unscoped_delete_is_refused_when_isolation_is_enforced() -> None:
    """`_scoped_tenant` is `require_tenant`, which is what makes "delete
    everything named policy" impossible to issue by accident."""
    from ragorc.core.errors import GuardrailViolation

    pipeline, _store = await _pipeline(settings=_settings(isolation=True))

    with pytest.raises(GuardrailViolation):
        await pipeline.delete(["policy"])


# ---------------------------------------------------------------------------
# It is audited
# ---------------------------------------------------------------------------
async def test_the_attempt_is_audited_even_if_it_fails() -> None:
    """A partial delete is the outcome an auditor most needs to see, so the record
    is written before the stores are touched."""
    pipeline, _store = await _pipeline(graph=_BrokenGraph(), chunks=[_chunk("c1", "policy")])
    events: list[Any] = []
    pipeline._audit.record = events.append  # type: ignore[method-assign]

    await pipeline.delete(["policy"])

    actions = [getattr(e, "action", None) for e in events]
    assert "delete" in actions, f"a delete left no audit record: {actions}"


# ---------------------------------------------------------------------------
# The report has to be able to say "nothing happened"
# ---------------------------------------------------------------------------
async def test_a_refused_cross_tenant_delete_does_not_read_as_success() -> None:
    """The worst thing this API can return. Isolation correctly protects the
    chunk, and the report used to say `documents=1, complete=True, errors={}` —
    which for a compliance-driven removal is a false confirmation that data is
    gone."""
    pipeline, store = await _pipeline(
        settings=_settings(isolation=True),
        chunks=[_chunk("c1", "policy", "acme")],
    )

    report = await pipeline.delete(["policy"], tenant_id="globex")

    assert report.documents == 1, "the request count is still what was asked"
    assert report.found == 0, "nothing belonging to globex resolved"
    assert not report.deleted, "the report claimed a deletion that did not happen"
    assert report.complete, "no store errored, which is a different question"
    assert [c.id async for c in store.scroll()] == ["c1"]


async def test_deleting_an_unknown_id_is_reported_as_finding_nothing() -> None:
    pipeline, _store = await _pipeline(
        graph=FakeGraphDeleter(), chunks=[_chunk("c1", "policy")]
    )

    report = await pipeline.delete(["never-existed"])

    assert (report.documents, report.found) == (1, 0)
    assert not report.deleted
    assert report.complete


async def test_a_real_delete_reports_both_numbers_equal() -> None:
    pipeline, _store = await _pipeline(
        graph=FakeGraphDeleter(), chunks=[_chunk("c1", "policy"), _chunk("c2", "faq")]
    )

    report = await pipeline.delete(["policy", "faq"])

    assert (report.documents, report.found) == (2, 2)
    assert report.deleted and report.complete


async def test_a_partial_match_is_neither_success_nor_failure() -> None:
    """One good id and one typo. `complete` is true — every store answered — and
    `deleted` is false, because the caller did not get what they asked for."""
    pipeline, _store = await _pipeline(
        graph=FakeGraphDeleter(), chunks=[_chunk("c1", "policy")]
    )

    report = await pipeline.delete(["policy", "typo"])

    assert (report.documents, report.found) == (2, 1)
    assert report.complete and not report.deleted


async def test_complete_and_deleted_answer_different_questions() -> None:
    """A store failing must not be confused with an id not existing, and vice
    versa — the two states need different remedies (retry, versus check the id)."""
    pipeline, _store = await _pipeline(
        graph=_BrokenGraph(), chunks=[_chunk("c1", "policy")]
    )

    report = await pipeline.delete(["policy"])

    assert report.found == 1, "the document was there"
    assert not report.complete, "but a store did not answer"
    assert not report.deleted, "so the removal cannot be confirmed"


# ---------------------------------------------------------------------------
# Finding an id in the first place
# ---------------------------------------------------------------------------
async def test_documents_lists_what_is_indexed() -> None:
    """`delete` had nothing to name. Its help pointed at `query --json`, which
    needs a working LLM, and `inspect` reports counts rather than ids — so a
    deployment out of model credit could not discover what it had indexed."""
    pipeline, store = await _pipeline()
    await store.upsert(
        [_chunk("c1", "policy"), _chunk("c2", "policy"), _chunk("c3", "faq")]
    )

    rows = await pipeline.documents()

    assert [(r["document_id"], r["chunks"]) for r in rows] == [("faq", 1), ("policy", 2)]


async def test_documents_is_scoped_to_the_tenant() -> None:
    """A list of what is indexed is a list of what exists — the enumeration the
    400-not-403 choice elsewhere exists to avoid leaking."""
    pipeline, store = await _pipeline(settings=_settings(isolation=True))
    await store.upsert(
        [_chunk("c1", "acme-doc", "acme"), _chunk("c2", "globex-doc", "globex")]
    )

    rows = await pipeline.documents(tenant_id="globex")

    assert [r["document_id"] for r in rows] == ["globex-doc"]


async def test_documents_can_be_filtered_by_source() -> None:
    """How you get from "I deleted this file" to the id that removes it."""
    pipeline, store = await _pipeline()
    a = _chunk("c1", "policy")
    a.metadata = {"source": "/corpus/policy.md"}
    b = _chunk("c2", "faq")
    b.metadata = {"source": "/corpus/faq.md"}
    await store.upsert([a, b])

    rows = await pipeline.documents(source="policy")

    assert [r["document_id"] for r in rows] == ["policy"]


async def test_the_limit_counts_documents_not_chunks() -> None:
    """A document with a thousand chunks must not exhaust a limit of ten."""
    pipeline, store = await _pipeline()
    await store.upsert([_chunk(f"c{i}", "policy") for i in range(20)])
    await store.upsert([_chunk("z1", "faq")])

    rows = await pipeline.documents(limit=1)

    assert len(rows) == 1
