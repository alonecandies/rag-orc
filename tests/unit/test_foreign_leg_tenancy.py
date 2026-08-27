"""What an adapted retriever does when it cannot honour tenant isolation.

:func:`~ragorc.adapters.langchain.from_langchain_retriever` makes someone else's
retriever one leg of an ensemble, fused with ours. That leg is outside every
mechanism the other tenancy settings describe: it queries its own store, so no
filter of ours reaches it, and the ``tenant_id`` on what comes back is read out
of the foreign document's own metadata — the retriever's claim, not a fact we
checked.

Reproduced before this guard existed. With ``enforce_tenant_isolation`` on, an
ensemble holding one adapted leg answered a query scoped to ``globex`` with::

    id=acme-secret  tenant_id='acme'  content='ACME Q3 REVENUE: $41.2M (CONFIDENTIAL)'

Two things were wrong and both are tested here. The wrapped retriever was handed
the question text and nothing else, so a multi-tenant one could not have scoped
itself; and nothing on the way back compared the chunk's declared tenant against
the query's.

Unlike the graph this has a real middle ground, which is why the setting has
three modes: a label that is present can be checked, and one that is absent can
be dropped. Neither needs a schema this library does not control.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.errors import GuardrailViolation
from ragorc.core.models import Query, RetrievalSource
from ragorc.core.settings import Settings
from ragorc.security.tenancy import (
    foreign_leg_refused,
    foreign_retriever_isolation_warning,
    require_foreign_retriever_isolation,
)


def _settings(*, isolation: bool = True, mode: str = "reject") -> Settings:
    return Settings(
        security={
            "enforce_tenant_isolation": isolation,
            "foreign_retriever_tenant_isolation": mode,
        },
        cache={"enabled": False},
        llm={"api_key": "k"},
        embedding={"dense_dimension": 4},
    )


class _Doc:
    """The shape ``from_langchain_documents`` reads. Duck-typed on purpose: the
    adapter never isinstance-checks, and neither should a test of it."""

    def __init__(self, identifier: str, content: str, **metadata: Any) -> None:
        self.id = identifier
        self.page_content = content
        self.metadata = metadata


class _Foreign:
    """A third-party retriever holding two tenants' documents, plus one that
    declares no tenant at all — the case an operator cannot reason about."""

    def __init__(self) -> None:
        self.configs: list[Any] = []

    async def ainvoke(self, text: str, config: Any = None) -> list[Any]:
        del text
        self.configs.append(config)
        return [
            _Doc("acme-secret", "ACME Q3 REVENUE", tenant_id="acme", document_id="board-deck"),
            _Doc("globex-public", "Globex public FAQ", tenant_id="globex", document_id="faq"),
            _Doc("unlabelled", "who knows whose this is", document_id="mystery"),
        ]


def _leg(mode: str = "reject", *, isolation: bool = True) -> Any:
    from ragorc.adapters.langchain import from_langchain_retriever

    return from_langchain_retriever(_Foreign(), settings=_settings(isolation=isolation, mode=mode))


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("isolation", "mode", "refused"),
    [
        pytest.param(True, "reject", True, id="isolated-and-rejecting"),
        pytest.param(True, "filter", False, id="scoped-by-declared-tenant"),
        pytest.param(True, "trusted", False, id="operator-asserted-single-tenant"),
        pytest.param(False, "reject", False, id="single-tenant"),
    ],
)
def test_the_predicate_and_the_guard_agree(isolation: bool, mode: str, refused: bool) -> None:
    """``foreign_leg_refused`` exists so callers can *decide* without catching.

    Control flow through a guard's exception is how a guard ends up caught and
    ignored, so the predicate and the raise must never disagree.
    """
    settings = _settings(isolation=isolation, mode=mode)
    assert foreign_leg_refused(settings) is refused
    if refused:
        with pytest.raises(GuardrailViolation) as caught:
            require_foreign_retriever_isolation("langchain:X", settings)
        assert caught.value.rule == "foreign_retriever_tenant_isolation"
    else:
        require_foreign_retriever_isolation("langchain:X", settings)


async def test_an_adapted_leg_refuses_under_isolation() -> None:
    with pytest.raises(GuardrailViolation, match="foreign retriever"):
        await _leg().retrieve(Query(text="what was revenue?", tenant_id="globex"))


def test_the_warning_lands_where_the_mistake_is() -> None:
    """A foreign leg is wired up once and fails on the first request, which is a
    long way from the line that caused it."""
    warning = foreign_retriever_isolation_warning("langchain:X", _settings())
    assert warning is not None and "will refuse every query" in warning
    assert foreign_retriever_isolation_warning("langchain:X", _settings(mode="filter")) is None
    assert foreign_retriever_isolation_warning("langchain:X", _settings(isolation=False)) is None


# ---------------------------------------------------------------------------
# What the leg is told, and what it is allowed to return
# ---------------------------------------------------------------------------
async def test_the_wrapped_retriever_is_told_which_tenant_is_asking() -> None:
    """The half a guard cannot supply. Refusing keeps a leg from leaking, but a
    retriever that *is* capable of scoping still has to be handed the scope, and
    it used to receive the question text and nothing else."""
    leg = _leg("filter")
    query = Query(text="q", tenant_id="globex", filters={"dept": "sales"})

    await leg.retrieve(query, tenant_id="globex", filters=query.filters)

    config = leg.retriever.configs[0]
    assert config is not None, "the wrapped retriever was given no config at all"
    metadata = config.get("metadata") or {}
    assert metadata.get("tenant_id") == "globex"
    assert metadata.get("filters") == {"dept": "sales"}


async def test_filter_keeps_only_what_declares_the_querying_tenant() -> None:
    out = await _leg("filter").retrieve(Query(text="q", tenant_id="globex"))
    assert [c.chunk.id for c in out] == ["globex-public"]


async def test_an_unlabelled_chunk_is_dropped_rather_than_stamped() -> None:
    """The decision the mode turns on. Stamping an unlabelled chunk with the
    querying tenant's id would forge exactly the provenance that made the graph
    leak worse than an unlabelled one: the label would say the content is yours.
    """
    out = await _leg("filter").retrieve(Query(text="q", tenant_id="globex"))
    assert "unlabelled" not in [c.chunk.id for c in out]


async def test_filtering_is_not_applied_when_the_query_names_no_tenant() -> None:
    """With no tenant to compare against there is nothing to filter *by*, and
    dropping everything would turn an unscoped query into an empty result rather
    than an error."""
    out = await _leg("filter").retrieve(Query(text="q"))
    assert len(out) == 3


async def test_the_tenant_can_arrive_as_a_keyword_rather_than_on_the_query() -> None:
    """Which of the two is authoritative, pinned.

    :class:`~ragorc.retrieve.ensemble.EnsembleRetriever` forwards ``tenant_id`` in
    ``**kw`` to every leg, and it does not have to match what the ``Query`` object
    carries — a caller scoping by keyword leaves ``query.tenant_id`` unset. Reading
    only the query there silently turns filtering off, and the mode that was asked
    for becomes a pass-through. Caught by mutation: this is the one revert the
    other tests all missed.
    """
    leg = _leg("filter")

    out = await leg.retrieve(Query(text="q"), tenant_id="globex")

    assert [c.chunk.id for c in out] == ["globex-public"]
    assert (leg.retriever.configs[0].get("metadata") or {}).get("tenant_id") == "globex"


async def test_trusted_does_not_second_guess_the_operator() -> None:
    """``trusted`` asserts the wrapped retriever holds one tenant's data. Filtering
    anyway would make the mode indistinguishable from ``filter`` and silently drop
    results from a correctly-configured single-tenant leg whose documents carry no
    tenant label — which is the normal case for one."""
    out = await _leg("trusted").retrieve(Query(text="q", tenant_id="globex"))
    assert len(out) == 3


async def test_the_leak_this_closes(caplog: Any) -> None:
    """End to end through the ensemble, which is how it was found: the fused
    result for one tenant contained another tenant's chunk."""
    from ragorc.retrieve.ensemble import EnsembleRetriever

    settings = _settings(mode="filter")
    ensemble = EnsembleRetriever(
        {"legacy": _leg("filter")}, settings=settings, weights={"legacy": 1.0}
    )
    query = Query(text="what was revenue?", tenant_id="globex")

    out = await ensemble.retrieve(query, top_k=5, tenant_id="globex")

    bodies = " ".join(c.chunk.content for c in out)
    assert "ACME" not in bodies, f"acme's document reached a globex query: {bodies!r}"
    assert all(c.chunk.tenant_id == "globex" for c in out)


async def test_the_source_is_still_the_adapters(caplog: Any) -> None:
    """Scoping must not disturb what the leg reports itself as, because fusion
    weights and the citation footer both read ``source``."""
    out = await _leg("trusted").retrieve(Query(text="q", tenant_id="globex"))
    assert {c.source for c in out} == {RetrievalSource.DENSE}


def test_the_setting_admits_no_fourth_mode() -> None:
    with pytest.raises(ValueError, match="foreign_retriever_tenant_isolation"):
        Settings(llm={"api_key": "k"}, security={"foreign_retriever_tenant_isolation": "rls"})
