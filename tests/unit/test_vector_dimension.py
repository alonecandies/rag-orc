"""The pgvector column width, and the one path that never reached it.

``Settings.model_post_init`` keeps Qdrant and pgvector "in lockstep" — its
comment says so — but only when ``embedding.dense_dimension`` is explicitly set.
That field is documented as "auto-detected from the model when left unset", and
``dense_model``'s docstring recommends two larger models *by name*. On that path
the embedder resolves 768 from the model registry, Qdrant creates a 768-wide
collection, and ``postgres.vector_dimension`` stays at its literal default of 384:

    postgres_schema_ready ... dimension=384
    real vector width: (768,)
    UPSERT FAILED: query vector dimension mismatch (got=768 expected=384)

Two fixes, because the two failure modes are different. The width now comes from
the embedder at every construction site, and a table that already exists at the
wrong width is refused at schema time instead of thousands of writes later.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.errors import ConfigError
from ragorc.core.settings import Settings
from ragorc.stores.postgres.store import PostgresStore, _pg_at_dimension


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {"llm": {"api_key": "k"}, "cache": {"enabled": False}}
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# The override reaches every reader
# ---------------------------------------------------------------------------
def test_a_measured_width_overrides_the_default() -> None:
    store = PostgresStore(_settings(), dimension=768)
    assert store.pg.vector_dimension == 768


def test_no_measurement_leaves_the_setting_alone() -> None:
    """A caller with no embedder — text-to-SQL only — must not be forced to
    invent a number."""
    settings = _settings(postgres={"vector_dimension": 384})
    assert PostgresStore(settings, dimension=None).pg.vector_dimension == 384
    assert PostgresStore(settings, dimension=0).pg.vector_dimension == 384


def test_the_override_keeps_the_dsn() -> None:
    """The pool is keyed on the DSN, so a re-pointed store must share the pool it
    would have shared anyway rather than opening a second one."""
    pg = _settings().postgres
    assert _pg_at_dimension(pg, 768).dsn == pg.dsn


@pytest.mark.parametrize(
    "attribute",
    ["vector_dimension"],
)
def test_every_width_reader_follows_the_override(attribute: str) -> None:
    """`self.pg` is what the DDL, the query-vector coercion and the row builder
    all read. Overriding anything narrower than that would fix the schema and
    leave `search()` coercing to the old width."""
    import inspect

    source = inspect.getsource(PostgresStore)
    assert f"self.settings.postgres.{attribute}" not in source, (
        "a reader bypasses self.pg and would keep the stale width"
    )


# ---------------------------------------------------------------------------
# Both wirings resolve it
# ---------------------------------------------------------------------------
class _Embedder:
    dimension = 768
    name = "stub"


def test_the_builder_resolves_the_width_from_its_embedder() -> None:
    from ragorc.pipeline.builder import RAGPipeline

    pipeline = RAGPipeline(settings=_settings(embedding={"dense_dimension": 32}), llm=object())
    pipeline._dense = _Embedder()
    assert pipeline._relational_dimension() == 768


def test_the_builder_degrades_when_no_embedder_can_be_built() -> None:
    """A width that cannot be resolved is not fatal: a text-to-SQL-only
    deployment must not fail at startup over a number it never reads."""
    from ragorc.pipeline.builder import RAGPipeline

    pipeline = RAGPipeline(settings=_settings(), llm=object())

    class _Broken:
        @property
        def dimension(self) -> int:
            raise RuntimeError("no model available")

    pipeline._dense = _Broken()
    assert pipeline._relational_dimension() is None


def test_the_server_passes_its_embedder_width() -> None:
    import inspect

    from ragorc.server.app import _LinearEngine

    source = inspect.getsource(_LinearEngine.build)
    assert "PostgresStore(\n            s, cache=self.cache, dimension=" in source, (
        "the server builds its relational store at the settings default again"
    )


# ---------------------------------------------------------------------------
# A table that already exists at the wrong width
# ---------------------------------------------------------------------------
class _Conn:
    """Just enough of an AsyncConnection for the catalog probe."""

    def __init__(self, typmod: int | None) -> None:
        self.typmod = typmod
        self.queries: list[str] = []

    async def execute(self, sql: Any, params: Any = None) -> Any:
        self.queries.append(str(sql))
        typmod = self.typmod

        class _Cur:
            async def fetchone(self) -> Any:
                return None if typmod is None else (typmod,)

        return _Cur()


async def test_an_existing_column_of_the_wrong_width_is_refused() -> None:
    """`CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists,
    so the mismatch used to surface as an insert error thousands of writes later
    — naming neither the model nor the setting that disagree."""
    from ragorc.stores.postgres.ddl import _assert_vector_width

    pg = _settings(postgres={"vector_dimension": 768}).postgres

    with pytest.raises(ConfigError) as caught:
        await _assert_vector_width(_Conn(384), pg)

    detail = caught.value.detail
    assert detail["column_dimension"] == 384, detail
    assert detail["configured_dimension"] == 768, detail
    assert "vector_dimension" in str(detail["hint"]), "the message must name the setting to change"
    # `detail` is what the API surfaces and traces record. Asserting on the
    # rendered string alone let a version that dropped a structured field pass,
    # because the prose hint happened to interpolate the same number.
    assert "384" in str(caught.value) and "768" in str(caught.value)


async def test_a_matching_column_passes() -> None:
    from ragorc.stores.postgres.ddl import _assert_vector_width

    pg = _settings(postgres={"vector_dimension": 384}).postgres
    await _assert_vector_width(_Conn(384), pg)  # must not raise


async def test_a_fresh_database_is_not_a_mismatch() -> None:
    """No column yet — the table is being created in this very transaction."""
    from ragorc.stores.postgres.ddl import _assert_vector_width

    await _assert_vector_width(_Conn(None), _settings().postgres)


async def test_an_unconstrained_vector_column_is_not_a_mismatch() -> None:
    """`vector` with no declared dimension reports a negative typmod. It accepts
    any width, so it is not a disagreement."""
    from ragorc.stores.postgres.ddl import _assert_vector_width

    await _assert_vector_width(_Conn(-1), _settings(postgres={"vector_dimension": 768}).postgres)


async def test_the_check_runs_inside_ensure_schema() -> None:
    """The call site. A guard nothing calls is the defect this round is named
    for, so this asserts on `ensure_schema`, not on the helper."""
    import inspect

    from ragorc.stores.postgres import ddl

    source = inspect.getsource(ddl.ensure_schema)
    assert "_assert_vector_width(conn, settings)" in source
