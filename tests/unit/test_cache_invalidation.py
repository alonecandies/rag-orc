"""What happens to a cached answer when the document under it changes.

The semantic cache skips the entire pipeline on a hit, which is what makes it the
largest cost lever here and what makes a stale entry expensive. It had no
invalidation, and — the part that mattered — it could not have had one: the
payload was question, answer, timestamp, tenant and scope, so there was nothing
to invalidate *by*. The only removal available was ``clear()``, which drops the
collection for every tenant at once, and nothing on the ingest path called it.

Reproduced against a live stack before the fix. A policy edited from "30 days" to
"7 days" and re-ingested was still answered::

    2. answer after the correction: 'The refund window is 30 days.'
       served from cache: {'tier': 'semantic', 'score': 1.0, ...}
       cited: []

with no citations attached, because a cache hit carries none — so the reader
could not have noticed the answer came from a document that no longer said that.

The module already contained the argument for this fix, applied to the opposite
case: it refuses to cache an abstention because "it is a statement about the
index at one moment, and serving it later hides content that has since been
added". A positive answer is the same statement with the sign flipped.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Chunk, Document, ScoredChunk
from ragorc.core.settings import Settings


class RecordingCache:
    """An answer cache that records what it was asked to invalidate.

    A double rather than a live Qdrant collection because what is under test is
    the *call*: the primitive was verified against the real store, and the defect
    was that nothing reached it.
    """

    def __init__(self) -> None:
        self.invalidated: list[tuple[tuple[str, ...], str | None]] = []
        self.built = 0

    async def invalidate(self, document_ids: Any, *, tenant_id: str | None = None) -> int:
        ids = tuple(document_ids)
        self.invalidated.append((ids, tenant_id))
        return len(ids)


def _settings() -> Settings:
    return Settings(
        llm={"api_key": "k"},
        embedding={"dense_dimension": 32},
        cache={"enabled": True, "semantic_enabled": True},
    )


def _pipeline(cache: Any) -> Any:
    from ragorc.index.pipeline import IngestPipeline

    return IngestPipeline(settings=_settings(), answer_cache=cache)


def _doc(doc_id: str, tenant: str | None = None) -> Document:
    return Document(id=doc_id, content="body", source=f"{doc_id}.md", tenant_id=tenant)


# ---------------------------------------------------------------------------
# The call the purge was missing
# ---------------------------------------------------------------------------
async def test_the_purge_invalidates_the_answers_built_from_what_it_replaced() -> None:
    from ragorc.index.pipeline import IngestReport

    cache = RecordingCache()
    report = IngestReport()

    await _pipeline(cache)._purge([_doc("policy"), _doc("faq")], report)

    assert cache.invalidated, "the purge did not reach the answer cache"
    ids, _tenant = cache.invalidated[0]
    assert set(ids) == {"policy", "faq"}
    assert report.answers_invalidated == 2


async def test_invalidation_is_scoped_to_the_tenant_being_re_ingested() -> None:
    """One tenant's re-ingest must not evict another's answers. Not a correctness
    problem — it is their hit rate being spent — but the purge already groups by
    tenant for the vector delete, so passing it on costs nothing."""
    from ragorc.index.pipeline import IngestReport

    cache = RecordingCache()

    await _pipeline(cache)._purge(
        [_doc("policy", "acme"), _doc("handbook", "globex")], IngestReport()
    )

    by_tenant = {tenant: set(ids) for ids, tenant in cache.invalidated}
    assert by_tenant == {"acme": {"policy"}, "globex": {"handbook"}}


async def test_an_ingest_with_no_answer_cache_still_purges() -> None:
    """The cache is optional, and a library caller assembling an IngestPipeline
    directly will not have one."""
    from ragorc.index.pipeline import IngestReport

    report = IngestReport()
    await _pipeline(None)._purge([_doc("policy")], report)
    assert report.answers_invalidated == 0


async def test_a_failing_cache_does_not_fail_the_ingest() -> None:
    """The operator is running this ingest *to fix that document*. Not
    invalidating costs a stale answer for up to one TTL; raising costs them the
    fix."""
    from ragorc.index.pipeline import IngestReport

    class Broken:
        async def invalidate(self, document_ids: Any, *, tenant_id: str | None = None) -> int:
            raise RuntimeError("qdrant is down")

    with pytest.raises(RuntimeError):
        # The guarantee lives in SemanticCache.invalidate, which traps its own
        # failures; a double that does not trap is expected to propagate. Pinned
        # so the trap is not quietly moved out of the primitive and into here,
        # where every other cache implementation would lose it.
        await _pipeline(Broken())._purge([_doc("policy")], IngestReport())


# ---------------------------------------------------------------------------
# The provenance that makes invalidation possible
# ---------------------------------------------------------------------------
async def test_an_answer_records_the_documents_it_was_built_from() -> None:
    """Without this field there is nothing to invalidate by, which is why the
    cache went nine rounds with no invalidation rather than a broken one."""
    from ragorc.cache.semantic import SemanticCache

    stored: dict[str, Any] = {}

    class Collecting(SemanticCache):
        """Records the write and never reaches a store — including on the *read*,
        which `query` performs first and which would otherwise open a real Qdrant
        connection from a unit test."""

        async def get(self, question: str, **kw: Any) -> None:
            return None

        async def set(self, question: str, answer: dict[str, Any], **kw: Any) -> None:
            stored.update(kw)

    from ragorc.core.models import Query, RetrievalResult, RetrievalSource
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.pipeline.builder import RAGPipeline
    from tests.fakes import StubEmbedder, StubLLM

    chunks = [
        ScoredChunk(
            chunk=Chunk(id="c1", content="30 days", document_id="policy"),
            score=1.0,
            source=RetrievalSource.DENSE,
            rank=0,
        ),
        ScoredChunk(
            chunk=Chunk(id="c2", content="see also", document_id="faq"),
            score=0.9,
            source=RetrievalSource.DENSE,
            rank=1,
        ),
    ]

    class Corpus:
        name = "corpus"

        async def retrieve(self, query: Query, **kw: Any) -> list[ScoredChunk]:
            return chunks

        async def retrieve_detailed(self, query: Query, **kw: Any) -> RetrievalResult:
            result = RetrievalResult()
            result.chunks = chunks
            return result

    settings = _settings()
    llm = StubLLM()
    pipeline = RAGPipeline(
        settings=settings, llm=llm, retriever=Corpus(), generator=AnswerGenerator(llm, settings)
    )
    pipeline._semantic_cache = Collecting(StubEmbedder(dimension=32), settings=settings)

    await pipeline.query("what is the refund window?")

    assert set(stored.get("document_ids") or ()) == {"policy", "faq"}


async def test_the_payload_carries_the_ids_the_filtered_delete_matches_on() -> None:
    """The schema both halves depend on. ``set`` writes ``document_ids`` and
    ``invalidate`` matches it with ``MatchAny``; if either side renames the key the
    cache silently stops being invalidatable and nothing else fails.
    """
    from ragorc.cache.semantic import SemanticCache
    from tests.fakes import StubEmbedder

    written: list[Any] = []

    class Client:
        async def upsert(self, *, collection_name: str, points: Any, **kw: Any) -> None:
            written.extend(points)

    cache = SemanticCache(StubEmbedder(dimension=32), client=Client(), settings=_settings())
    cache._ready = True

    await cache.set("q", {"text": "a"}, tenant_id="acme", document_ids=["b", "a", "b"])

    payload = written[0].payload
    assert payload["document_ids"] == ["a", "b"], "deduped and ordered, so the id is stable"


async def test_an_answer_with_no_sources_stores_no_provenance() -> None:
    """An empty list would match nothing and cost a payload field on every entry."""
    from ragorc.cache.semantic import SemanticCache
    from tests.fakes import StubEmbedder

    written: list[Any] = []

    class Client:
        async def upsert(self, *, collection_name: str, points: Any, **kw: Any) -> None:
            written.extend(points)

    cache = SemanticCache(StubEmbedder(dimension=32), client=Client(), settings=_settings())
    cache._ready = True

    await cache.set("q", {"text": "a"}, tenant_id="acme", document_ids=[])

    assert "document_ids" not in written[0].payload


async def test_invalidating_nothing_makes_no_round_trip() -> None:
    """Called once per tenant per purge, including on runs that changed nothing.

    Asserted on the *call*, not on the return value. ``invalidate`` traps its own
    exceptions and returns 0, so a probe that raises inside the client is
    swallowed and a test reading only the return value passes either way — which
    is exactly what happened: this test's first version missed the mutation that
    sends an empty id list to the store.
    """
    from ragorc.cache.semantic import SemanticCache
    from tests.fakes import StubEmbedder

    calls: list[Any] = []

    class Counting:
        async def delete(self, **kw: Any) -> None:
            calls.append(kw)

    cache = SemanticCache(StubEmbedder(dimension=32), client=Counting(), settings=_settings())
    cache._ready = True

    assert await cache.invalidate([]) == 0
    assert await cache.invalidate(["", None]) == 0  # type: ignore[list-item]
    assert calls == [], f"an empty id list still reached the store: {calls}"

    # A mixed list is the case where the filter changes what is *sent* rather than
    # whether anything is: an empty string is a valid MatchAny member, so without
    # the filter it becomes a predicate that matches an id no document has.
    assert await cache.invalidate(["policy", ""]) == 1
    matched = calls[0]["points_selector"].filter.must[0].match.any
    assert matched == ["policy"], f"an empty id was sent as a predicate: {matched}"
