"""What the generator is handed when the index holds something else.

All three multi-representation stages index a *stand-in*: parent-document a small
child, summary-index an LLM summary, dense-X a rewritten proposition. All three
ship a query-time step that resolves the stand-in back to the source, and none of
those steps had a caller — so the generator answered from a 299-character child,
or from a paraphrase, while citing the document.

Nothing in the class or the setting was wrong. `expand_parents` works,
`ParentDocumentRetriever` works, `ContextPacker._expand` works, and
``retrieval.parent_expansion`` defaults to on. What made it invisible is that the
packer's *other* key — ``window_text``, written at index time by the
sentence-window splitter — does arrive, so the branch that substitutes ran on
every query and simply never found a parent to substitute.

These tests therefore assert on the prompt, not on the plumbing.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Chunk, Modality, Query, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings

_PARENT = (
    "Refunds are available for thirty days after delivery. After that window the "
    "item is yours. Shipping costs are not refundable in either case, and the "
    "original packaging must be intact for a refund to be issued at all."
)
_CHILD = "Refunds are available for thirty days after delivery."
_SUMMARY = "The vendor offers generous money-back arrangements with no fixed deadline."


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
        # Not a tenancy test. See tests/conftest.py on the neutralized .env.
        "security": {"enforce_tenant_isolation": False},
    }
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# The predicate, defined once
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flag", ["parent_document_enabled", "summary_index_enabled", "dense_x_enabled"]
)
def test_every_representation_counts_as_multirep(flag: str) -> None:
    """Asserted per flag. The index side had this predicate inline and the query
    side did not have it at all; a version that named two of the three would look
    correct and leave one representation dead exactly as before."""
    assert _settings(indexing={flag: True}).indexing.multirep_enabled, flag


def test_a_default_deployment_is_not_multirep() -> None:
    assert not _settings().indexing.multirep_enabled


def test_the_index_side_reads_the_same_predicate() -> None:
    """The two sides drifting is the defect, so the guard is that there is only
    one definition — not that two definitions currently agree."""
    import inspect

    from ragorc.index.pipeline import IngestPipeline

    source = inspect.getsource(IngestPipeline._stage_enabled)
    assert "multirep_enabled" in source
    assert "summary_index_enabled" not in source, (
        "the index side spelled the predicate out again; it will drift"
    )

    # And the answer, because `return not self.config.multirep_enabled` satisfies
    # both greps while inverting the stage in every deployment.
    pipeline = object.__new__(IngestPipeline)
    for flag in ("parent_document_enabled", "summary_index_enabled", "dense_x_enabled"):
        pipeline.config = _settings(indexing={flag: True}).indexing
        assert pipeline._stage_enabled("multirep") is True, flag
    pipeline.config = _settings().indexing
    assert pipeline._stage_enabled("multirep") is False


# ---------------------------------------------------------------------------
# parent_leg
# ---------------------------------------------------------------------------
class _Inner:
    name = "hybrid"

    def __init__(self, chunks: list[ScoredChunk] | None = None) -> None:
        self.chunks = chunks or []
        self.asked: list[int | None] = []

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> list[ScoredChunk]:
        self.asked.append(top_k)
        return list(self.chunks)


def test_parent_leg_is_the_identity_without_a_representation() -> None:
    """A default deployment must build the object graph it built before."""
    from ragorc.retrieve.parent import parent_leg

    inner = _Inner()
    assert parent_leg(inner, object(), settings=_settings()) is inner


@pytest.mark.parametrize(
    "flag", ["parent_document_enabled", "summary_index_enabled", "dense_x_enabled"]
)
def test_parent_leg_wraps_for_each_representation(flag: str) -> None:
    from ragorc.retrieve.parent import ParentDocumentRetriever, parent_leg

    wrapped = parent_leg(_Inner(), object(), settings=_settings(indexing={flag: True}))
    assert isinstance(wrapped, ParentDocumentRetriever), flag


def test_parent_expansion_off_means_no_fetch() -> None:
    """The flag the packer asks before substituting. Fetching bodies that will
    not be substituted is a docstore round trip with no output."""
    from ragorc.retrieve.parent import parent_leg

    inner = _Inner()
    settings = _settings(
        indexing={"summary_index_enabled": True}, retrieval={"parent_expansion": False}
    )
    assert parent_leg(inner, object(), settings=settings) is inner


# ---------------------------------------------------------------------------
# The wrapper must not change how callers retrieve
# ---------------------------------------------------------------------------
class _DetailedInner(_Inner):
    """A leg with `retrieve_detailed`, which is what the real vector leg is."""

    async def retrieve_detailed(self, query: Query, *, top_k: int | None = None, **kw: Any) -> Any:
        from ragorc.core.models import RetrievalResult

        self.asked.append(top_k)
        result = RetrievalResult()
        result.chunks = list(self.chunks)
        result.per_store = {"dense": list(self.chunks)}
        result.timings_ms = {"dense": 1.5}
        result.total_candidates = len(self.chunks)
        return result


def _answer_prompt(llm: Any) -> str:
    """The rendered context the answer was generated from."""
    calls = [c for c in llm.calls if c.get("stage") == "answer"]
    assert calls, f"no answer call was made: {[c.get('stage') for c in llm.calls]}"
    return str(calls[-1]["prompt"])


def _scored(chunk: Chunk, score: float = 1.0, rank: int = 0) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, source=RetrievalSource.DENSE, rank=rank)


class _Docstore:
    def __init__(self, *parents: Chunk) -> None:
        self.parents = {p.id: p for p in parents}
        self.asked: list[list[str]] = []

    async def get_chunks(self, ids: Any, **kw: Any) -> list[Chunk]:
        self.asked.append(list(ids))
        return [self.parents[i] for i in ids if i in self.parents]


async def test_the_wrapper_keeps_retrieve_detailed() -> None:
    """Five call sites pick their code path with
    ``getattr(retriever, "retrieve_detailed", None)`` — including both of the
    pipeline's retrieval nodes. A wrapper without it does not just lose the
    per-store diagnostics; it silently moves those callers onto their fallback
    branch, changing how the pipeline retrieves.
    """
    from ragorc.retrieve.parent import ParentDocumentRetriever

    parent = Chunk(id="p1", content=_PARENT, document_id="d1", start_char=40)
    child = Chunk(id="c1", content=_CHILD, document_id="d1", parent_id="p1")
    inner = _DetailedInner([_scored(child)])
    store = _Docstore(parent)

    wrapped = ParentDocumentRetriever(inner, store, settings=_settings())
    result = await wrapped.retrieve_detailed(Query(text="refunds?"), top_k=5)

    assert [c.chunk.id for c in result.per_store["dense"]] == ["c1"], (
        "per_store must report what each leg found, which is children"
    )
    assert result.timings_ms == {"dense": 1.5}, "the inner leg's diagnostics were dropped"
    assert result.chunks[0].chunk.metadata["parent_text"] == _PARENT


async def test_the_wrapper_overfetches_children() -> None:
    """Children of one parent collapse, so asking for `top_k` children returns
    fewer than `top_k` parents."""
    from ragorc.retrieve.parent import ParentDocumentRetriever

    inner = _Inner([])
    wrapped = ParentDocumentRetriever(inner, None, settings=_settings(), overfetch=3)
    await wrapped.retrieve(Query(text="q"), top_k=10)

    assert inner.asked == [30]


async def test_expansion_degrades_rather_than_failing_the_query() -> None:
    """A docstore that is down loses breadth. Raising would lose the evidence."""
    from ragorc.retrieve.parent import ParentDocumentRetriever

    class _Broken:
        async def get_chunks(self, ids: Any, **kw: Any) -> list[Chunk]:
            raise RuntimeError("postgres is down")

    child = Chunk(id="c1", content=_CHILD, document_id="d1", parent_id="p1")
    wrapped = ParentDocumentRetriever(_Inner([_scored(child)]), _Broken(), settings=_settings())

    out = await wrapped.retrieve(Query(text="refunds?"), top_k=5)

    assert [s.chunk.content for s in out] == [_CHILD]


# ---------------------------------------------------------------------------
# End to end: what reaches the prompt
# ---------------------------------------------------------------------------
async def _pipeline(indexing: dict[str, Any], hits: list[Chunk], parents: list[Chunk]) -> Any:
    """A pipeline whose vector leg returns derived units and whose docstore holds
    the sources — which is exactly the state an ingest with the stage on leaves."""
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.pipeline.builder import RAGPipeline
    from tests.fakes import FakeVectorStore, StubLLM

    settings = _settings(indexing=indexing, generation={"check_groundedness": False})
    llm = StubLLM(text="Thirty days [1].")

    class _Leg:
        name = "hybrid"

        async def retrieve(self, query: Query, *, top_k: int | None = None, **kw: Any) -> Any:
            return [_scored(c, score=1.0 - i / 10, rank=i) for i, c in enumerate(hits)]

    pipeline = RAGPipeline(
        settings=settings,
        llm=llm,
        generator=AnswerGenerator(llm, settings),
        vector_store=FakeVectorStore(),
        relational_store=_Docstore(*parents),
    )
    # The leg the wrap goes around. Injected after construction so the builder's
    # own `vector_leg` wiring is what does the wrapping — the call site is the
    # thing under test, not the retriever.
    pipeline._hybrid = _Leg()
    return pipeline, llm


@pytest.mark.parametrize(
    ("flag", "modality", "unit_text"),
    [
        ("parent_document_enabled", Modality.TEXT, _CHILD),
        ("summary_index_enabled", Modality.SUMMARY, _SUMMARY),
        ("dense_x_enabled", Modality.PROPOSITION, "Refunds last thirty days."),
    ],
)
async def test_the_generator_sees_the_source_not_the_derived_unit(
    flag: str, modality: Modality, unit_text: str
) -> None:
    """The assertion the whole round turns on, made on the rendered prompt.

    Every earlier check would have passed with the bug in place: the setting was
    honoured, the packer branch ran, the retriever existed and was registered. The
    only observable difference is which text the model was shown.
    """
    parent = Chunk(id="p1", content=_PARENT, document_id="d1", start_char=0)
    unit = Chunk(
        id="u1",
        content=unit_text,
        document_id="d1",
        parent_id="p1",
        modality=modality,
    )
    pipeline, llm = await _pipeline({flag: True}, [unit], [parent])

    await pipeline.query("how long do refunds take?", pipeline="naive")

    prompt = _answer_prompt(llm)
    assert _PARENT in prompt, f"the generator was handed the {modality.value}, not the source"


async def test_a_default_pipeline_still_sees_what_it_retrieved() -> None:
    """No representation on: the retrieved text is the answer's evidence, and the
    fix must not start expanding chunks that have nothing to expand into."""
    chunk = Chunk(id="c1", content=_CHILD, document_id="d1")
    pipeline, llm = await _pipeline({}, [chunk], [])

    await pipeline.query("how long do refunds take?", pipeline="naive")

    prompt = _answer_prompt(llm)
    assert _CHILD in prompt
    assert _PARENT not in prompt


# ---------------------------------------------------------------------------
# Both wirings, not just the one that was fixed first
# ---------------------------------------------------------------------------
def test_the_builder_routes_every_vector_consumer_through_the_wrap() -> None:
    """`hybrid_retriever` had two consumers and the server had five. Naming the
    property is what keeps the sixth from being added unwrapped."""
    import inspect

    from ragorc.pipeline.builder import RAGPipeline

    source = inspect.getsource(RAGPipeline)
    body = source[source.index("def vector_leg") :]
    assert "self.hybrid_retriever" not in body.replace(
        "parent_leg(\n                self.hybrid_retriever", ""
    ), "a consumer after vector_leg still reads hybrid_retriever directly"


def test_the_server_routes_every_pipeline_through_the_wrap() -> None:
    """Five routes read the vector leg — crag, graphrag, multihop, adaptive and
    the naive default. The one that keeps `self.hybrid` is the debug path in the
    CLI, which inspects the raw hybrid deliberately."""
    import inspect

    from ragorc.server.app import _LinearEngine

    source = inspect.getsource(_LinearEngine._build_retriever)
    assert "self.hybrid" not in source, "a route still retrieves through the unwrapped leg"
    assert source.count("self.vector_leg") == 5


def test_the_server_actually_wraps_the_leg_it_routes_through() -> None:
    """The other half, and the one that mattered.

    The test above is about how the five routes *read* the attribute. The
    attribute is *assigned* one method up, and replacing that assignment with
    `self.vector_leg = self.hybrid` restores the entire defect while every route
    still reads `self.vector_leg` five times — the whole suite stayed green.
    A reader test and a writer test are not the same test.
    """
    import inspect

    from ragorc.server.app import _LinearEngine

    build = inspect.getsource(_LinearEngine.build)
    assignments = [
        line.strip() for line in build.splitlines() if line.strip().startswith("self.vector_leg")
    ]
    assert assignments == ["self.vector_leg = parent_leg(self.hybrid, self.relational, settings=s)"], (
        f"the server's vector leg is not the wrapped one: {assignments}"
    )


# ---------------------------------------------------------------------------
# The scope the parent fetch runs under
# ---------------------------------------------------------------------------
class _ScopedStore:
    """A docstore that scopes its reads, which is what both shipped stores do."""

    def __init__(self, *parents: Chunk, enforce: bool = True) -> None:
        self.parents = {p.id: p for p in parents}
        self.asked: list[str | None] = []
        self.enforce = enforce

    async def get_chunks(self, ids: Any, *, tenant_id: str | None = None) -> list[Chunk]:
        self.asked.append(tenant_id)
        if self.enforce and tenant_id is None:
            from ragorc.core.errors import GuardrailViolation

            raise GuardrailViolation("tenant_id is required when tenant isolation is enabled")
        return [
            p
            for i in ids
            if (p := self.parents.get(i)) is not None
            and (tenant_id is None or p.tenant_id == tenant_id)
        ]


@pytest.mark.parametrize("isolation", [True, False])
async def test_the_parent_fetch_carries_the_querys_tenant(isolation: bool) -> None:
    """The fetch is scoped by the docstore, so omitting the tenant does not widen
    it — it *fails* it, and `_fetch_parents` degrades on any exception. So on the
    library's default configuration the whole fix was a silent no-op: the only
    evidence was a `parent_fetch_failed` warning and `resolved=0` at debug level.

    Asserted with isolation off as well, because an unscoped read is the hole the
    scoping exists to close whether or not the guard is armed.
    """
    from ragorc.retrieve.parent import ParentDocumentRetriever

    parent = Chunk(id="p1", content=_PARENT, document_id="d1", tenant_id="acme")
    child = Chunk(id="c1", content=_CHILD, document_id="d1", parent_id="p1", tenant_id="acme")
    store = _ScopedStore(parent, enforce=isolation)
    settings = _settings(
        indexing={"summary_index_enabled": True},
        security={"enforce_tenant_isolation": isolation},
    )

    wrapped = ParentDocumentRetriever(_Inner([_scored(child)]), store, settings=settings)
    out = await wrapped.retrieve(Query(text="refunds?", tenant_id="acme"), top_k=5)

    assert store.asked == ["acme"], f"the docstore was asked with tenant={store.asked}"
    assert out[0].chunk.metadata.get("parent_text") == _PARENT


async def test_a_foreign_parent_is_not_returned() -> None:
    """The scoping is not decoration: a child whose parent belongs to another
    tenant must come back unexpanded rather than carrying that tenant's text."""
    from ragorc.retrieve.parent import ParentDocumentRetriever

    foreign = Chunk(id="p1", content=_PARENT, document_id="d1", tenant_id="globex")
    child = Chunk(id="c1", content=_CHILD, document_id="d1", parent_id="p1", tenant_id="acme")
    store = _ScopedStore(foreign)

    wrapped = ParentDocumentRetriever(
        _Inner([_scored(child)]), store, settings=_settings(indexing={"dense_x_enabled": True})
    )
    out = await wrapped.retrieve(Query(text="refunds?", tenant_id="acme"), top_k=5)

    assert "parent_text" not in out[0].chunk.metadata
    assert out[0].chunk.content == _CHILD


async def test_the_deployment_tenant_is_the_fallback() -> None:
    """Same resolution the graph leg uses: the query's tenant, then the
    deployment's."""
    from ragorc.retrieve.parent import ParentDocumentRetriever

    parent = Chunk(id="p1", content=_PARENT, document_id="d1", tenant_id="acme")
    child = Chunk(id="c1", content=_CHILD, document_id="d1", parent_id="p1")
    store = _ScopedStore(parent)
    settings = _settings(indexing={"summary_index_enabled": True}, tenant_id="acme")

    wrapped = ParentDocumentRetriever(_Inner([_scored(child)]), store, settings=settings)
    await wrapped.retrieve(Query(text="refunds?"), top_k=5)

    assert store.asked == ["acme"]
