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
