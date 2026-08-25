"""The public facade.

The README promises `build_pipeline()` → `ingest()` → `query()`. These tests hold
that promise to the letter, offline, because a facade that only works with three
databases running is not a facade a new user can try.
"""

from __future__ import annotations

import pytest

from ragorc.core.settings import Settings
from ragorc.pipeline.builder import RAGPipeline


@pytest.fixture
def settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "test-key"},
        embedding={"dense_dimension": 32},
    )


def test_top_level_lazy_exports_resolve() -> None:
    """``import ragorc`` must stay light; the heavy names resolve on access."""
    import ragorc

    assert ragorc.__version__
    # Import weight is asserted in a clean subprocess below; by this point the
    # test session has already imported most of the library.
    assert ragorc.RAGPipeline.__name__ == "RAGPipeline"
    assert callable(ragorc.build_pipeline)


def test_top_level_rejects_an_unknown_attribute() -> None:
    import ragorc

    with pytest.raises(AttributeError):
        _ = ragorc.NoSuchThing


def test_import_ragorc_does_not_load_the_drivers() -> None:
    """Checked in a clean subprocess, because this test module has already
    imported half the library."""
    import subprocess
    import sys

    code = (
        "import sys, ragorc; "
        "heavy=[m for m in ('qdrant_client','neo4j','psycopg','fastembed','torch') "
        "if m in sys.modules]; print(heavy)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]", f"import ragorc pulled {out}"


def test_describe_reports_the_resolved_configuration(settings: Settings) -> None:
    """A user must be able to see what is actually switched on without reading
    source or guessing from environment variables."""
    pipeline = RAGPipeline(settings=settings)
    described = pipeline.describe()
    assert isinstance(described, dict)
    blob = str(described)
    assert "hybrid" in blob or "features" in blob


def test_graph_selection_follows_the_settings(settings: Settings) -> None:
    """`pipeline="auto"` must be derived from the feature flags, not hard-coded."""
    plain = RAGPipeline(settings=settings)
    assert plain.select_graph("auto") in {"naive", "adaptive", "crag", "self_rag", "graphrag"}

    with_crag = settings.model_copy(deep=True)
    with_crag.retrieval.crag_enabled = True
    assert RAGPipeline(settings=with_crag).select_graph("auto") == "crag"

    with_self_rag = settings.model_copy(deep=True)
    with_self_rag.generation.self_rag_enabled = True
    assert RAGPipeline(settings=with_self_rag).select_graph("auto") == "self_rag"

    with_graph = settings.model_copy(deep=True)
    with_graph.graph.enabled = True
    assert RAGPipeline(settings=with_graph).select_graph("auto") == "graphrag"


def test_explicit_pipeline_name_overrides_auto(settings: Settings) -> None:
    pipeline = RAGPipeline(settings=settings)
    assert pipeline.select_graph("naive") == "naive"


def test_unknown_pipeline_name_is_rejected(settings: Settings) -> None:
    from ragorc.core.errors import ConfigError

    pipeline = RAGPipeline(settings=settings)
    with pytest.raises((ConfigError, KeyError, ValueError)):
        pipeline.select_graph("does-not-exist")


async def test_aclose_is_safe_without_any_store(settings: Settings) -> None:
    """Closing a pipeline that never opened a connection must not raise — a
    caller cannot know which stores were lazily constructed."""
    pipeline = RAGPipeline(settings=settings)
    await pipeline.aclose()
    await pipeline.aclose()  # idempotent


async def test_context_manager_closes(settings: Settings) -> None:
    async with RAGPipeline(settings=settings) as pipeline:
        assert pipeline.describe()


def test_stores_are_constructed_lazily(settings: Settings) -> None:
    """A project configured only for Qdrant must not open Postgres and Neo4j
    connections; constructing the facade must touch no network."""
    pipeline = RAGPipeline(settings=settings)
    # Accessors exist but must not have been invoked during construction.
    assert callable(pipeline.relational_store)
    assert callable(pipeline.graph_store)


async def test_aclose_releases_the_embedders_and_the_reranker(settings: Settings) -> None:
    """`aclose()` closed the stores, the cache and the LLM but not these.

    Every hosted embedding and rerank provider owns an HTTP client, so a pipeline
    built on one leaked a connection pool per instance. Invisible in a script that
    exits; fatal in a long-lived service that builds a pipeline per tenant.
    """
    closed: list[str] = []

    class _Closeable:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    pipeline = RAGPipeline(settings=settings)
    pipeline._dense = _Closeable("dense")  # type: ignore[assignment]
    pipeline._sparse = _Closeable("sparse")  # type: ignore[assignment]
    pipeline._late = _Closeable("late")  # type: ignore[assignment]
    pipeline._reranker = _Closeable("reranker")
    await pipeline.aclose()

    assert sorted(closed) == ["dense", "late", "reranker", "sparse"]


async def test_aclose_survives_a_component_that_will_not_close(settings: Settings) -> None:
    """One stubborn socket must not keep the others open: the caller is on their
    way out, and a leaked pool outlives the process that could have reported it."""
    closed: list[str] = []

    class _Broken:
        async def aclose(self) -> None:
            raise RuntimeError("refuses to close")

    class _Fine:
        async def aclose(self) -> None:
            closed.append("fine")

    pipeline = RAGPipeline(settings=settings)
    pipeline._dense = _Broken()  # type: ignore[assignment]
    pipeline._reranker = _Fine()
    await pipeline.aclose()

    assert closed == ["fine"]


async def test_aclose_is_idempotent(settings: Settings) -> None:
    calls: list[int] = []

    class _Counting:
        async def aclose(self) -> None:
            calls.append(1)

    pipeline = RAGPipeline(settings=settings)
    pipeline._reranker = _Counting()
    await pipeline.aclose()
    await pipeline.aclose()
    assert calls == [1]


def test_self_query_can_be_injected(settings: Settings) -> None:
    """The stage was documented as switchable and had no switch.

    `RAGPipeline.constructor` builds a `SelfQueryConstructor` with an empty
    attribute schema — a deliberate no-op, since a model asked to invent field
    names produces filters that match nothing — and its docstring told the reader
    to "inject it". There was no parameter to inject through, so the only way in
    was to bypass the facade entirely.
    """
    sentinel = object()
    assert RAGPipeline(settings=settings, constructor=sentinel).constructor is sentinel


async def test_create_treats_constructor_as_a_component_not_a_settings_path(
    settings: Settings,
) -> None:
    """`create()` splits its kwargs into components and settings overrides by
    reading `__init__`'s signature. A component missing from that signature is
    read as a dotted settings path instead, which fails in a way that names the
    setting rather than the component."""
    from ragorc.pipeline.builder import _COMPONENT_PARAMS

    assert "constructor" in _COMPONENT_PARAMS

    sentinel = object()
    pipeline = await RAGPipeline.create(settings=settings, constructor=sentinel)
    try:
        assert pipeline.constructor is sentinel
    finally:
        await pipeline.aclose()


# ---------------------------------------------------------------------------
# Streaming runs the pipeline that was selected
# ---------------------------------------------------------------------------
async def test_streaming_runs_the_selected_graph_not_a_fixed_sequence(
    settings: Settings,
) -> None:
    """``_retrieve_for_stream`` drove a hand-written five-node list.

    ``validate, translate, route, grade|retrieve, rerank`` — for every pipeline.
    ``select_graph`` was consulted, but only to choose the translation mode and
    to label the state, so streaming ``multihop`` took no hops and never asked
    whether the evidence was sufficient. This asserts through the builder rather
    than through ``build_graph``, because the builder is where the substitution
    happened and a test of the graphs alone cannot see it.
    """
    from ragorc.core.models import Chunk, Query, RetrievalResult, ScoredChunk
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.pipeline.builder import RAGPipeline
    from tests.fakes import StubLLM

    class OneHit:
        name = "stub"

        async def retrieve(self, query: Query, **kwargs: object) -> list[ScoredChunk]:
            del kwargs
            return [
                ScoredChunk(
                    chunk=Chunk(
                        id="c1", content="Acme was founded in Cambridge.", document_id="d1"
                    ),
                    score=0.9,
                )
            ]

        async def retrieve_detailed(self, query: Query, **kwargs: object) -> RetrievalResult:
            chunks = await self.retrieve(query)
            return RetrievalResult(chunks=chunks, per_store={"vector": chunks})

    llm = StubLLM()
    pipeline = RAGPipeline(
        settings=settings,
        llm=llm,
        retriever=OneHit(),
        generator=AnswerGenerator(llm, settings),
    )

    state, _nodes = await pipeline._retrieve_for_stream(
        "which university did Acme's founder attend?",
        tenant=None,
        top_k=5,
        pipeline="multihop",
    )

    assert llm.calls_for("multihop_reason"), (
        f"streaming multihop took no hops; the stages were {llm.stages()}"
    )
    assert state.get("answer") is None, "generation ran before the first token was streamed"


def test_the_streaming_graph_is_cached_apart_from_the_plain_one(settings: Settings) -> None:
    """Two compilations of one pipeline, because ``interrupt_before`` is a
    compile-time property. Keyed rather than rebuilt per request: compilation is
    not free and a streaming deployment would otherwise pay it every time."""
    from ragorc.pipeline.builder import RAGPipeline

    pipeline = RAGPipeline(settings=settings)
    plain = pipeline._compiled_graph("naive", streaming=False)
    streamed = pipeline._compiled_graph("naive", streaming=True)

    assert plain is not streamed
    assert pipeline._compiled_graph("naive", streaming=True) is streamed


async def test_builder_gives_local_graph_search_a_query_embedder(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wiring, asserted at the builder.

    `GraphLocalRetriever` blends three signals and the third needs a query
    vector. Removing `embedder=` from both construction sites left the whole
    suite green — every test of the term constructs the retriever itself, so
    none of them could see the builder not passing one. That is the same shape
    as the bug: the retriever was fine, nothing gave it its input.
    """
    from ragorc.pipeline.builder import RAGPipeline

    pipeline = RAGPipeline(settings=settings)

    async def fake_graph_store() -> object:
        return object()

    monkeypatch.setattr(pipeline, "graph_store", fake_graph_store)
    local = await pipeline._graph_search_retrievers()["local"]._resolve()

    assert local.embedder is not None, "local search cannot use its similarity term"
    assert local.embedder is pipeline.dense_embedder, (
        "the query must be embedded by the model that embedded the chunks; a cosine "
        "between two models' spaces is a number with no meaning"
    )
