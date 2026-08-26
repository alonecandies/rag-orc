"""What the knowledge graph does when it cannot honour tenant isolation.

Neo4j holds no tenant. Entities merge on ``name``, communities on a membership
hash, chunk links on a chunk id — none of them namespaced — so two tenants
writing about the same company converge on one node and a traversal from it
reaches both. Reproduced end to end against the live stack before this guard
existed: a query scoped to one tenant returned another tenant's chunk body
verbatim, and the verbalized subgraph carried a third party's entity
descriptions while being *stamped* with the querying tenant's id.

``require_generated_query_isolation`` already existed for exactly this shape of
problem, and did not cover it: that guard is on Cypher an LLM *wrote*, while
local, global, DRIFT and bridge search all run parameterized traversals.

Two layers are tested here. The guard refuses the legs, which is the honest
default because scoping the graph properly is a schema change. The by-id chunk
fetch is scoped independently, so a foreign chunk id resolves to nothing even
when an operator has asserted ``trusted``. The last test pins what is
deliberately *not* fixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ragorc.core.errors import GuardrailViolation
from ragorc.core.models import Chunk, Entity, Query, Relation
from ragorc.core.settings import Settings
from ragorc.retrieve.graph import GraphGlobalRetriever, GraphLocalRetriever
from ragorc.security.tenancy import (
    graph_isolation_warning,
    graph_legs_refused,
    require_graph_tenant_isolation,
)
from tests.fakes import FakeGraphStore, FakeVectorStore, StubLLM


def _settings(*, isolation: bool = True, mode: str = "reject", graph: bool = True) -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": isolation, "graph_tenant_isolation": mode},
        cache={"enabled": False},
        llm={"api_key": "k"},
        embedding={"dense_dimension": 4},
        graph={"enabled": graph},
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("isolation", "mode", "refused"),
    [
        pytest.param(True, "reject", True, id="isolated-and-rejecting"),
        pytest.param(True, "trusted", False, id="operator-asserted-one-graph-per-tenant"),
        pytest.param(False, "reject", False, id="single-tenant"),
    ],
)
def test_the_predicate_and_the_guard_agree(isolation: bool, mode: str, refused: bool) -> None:
    """``graph_legs_refused`` exists so callers can *decide* without catching.

    Control flow through a guard's exception is how a guard ends up caught and
    ignored, so the predicate and the raise must never disagree.
    """
    settings = _settings(isolation=isolation, mode=mode)
    assert graph_legs_refused(settings) is refused
    if refused:
        with pytest.raises(GuardrailViolation) as caught:
            require_graph_tenant_isolation("local search", settings)
        assert caught.value.rule == "graph_tenant_isolation"
    else:
        require_graph_tenant_isolation("local search", settings)


async def test_local_search_refuses_under_isolation() -> None:
    retriever = GraphLocalRetriever(FakeGraphStore(), FakeVectorStore(), settings=_settings())
    with pytest.raises(GuardrailViolation, match="graph local search is disabled"):
        await retriever.retrieve(Query(text="who is Acme?", tenant_id="acme"))


async def test_expand_refuses_too_because_drift_calls_it_directly() -> None:
    """A guard on ``retrieve`` alone would leave the DRIFT path open: it seeds
    with a vector search and then calls ``expand`` with the seeds in hand."""
    retriever = GraphLocalRetriever(FakeGraphStore(), FakeVectorStore(), settings=_settings())
    seed = (Entity(name="Acme", type="ORG", description="d"), 1.0)
    with pytest.raises(GuardrailViolation, match="graph local search is disabled"):
        await retriever.expand(Query(text="q", tenant_id="acme"), [seed])


async def test_global_search_refuses_under_isolation() -> None:
    """Community ids are a hash of level plus membership, with no tenant in them,
    so ``communities()`` returns every tenant's reports."""
    retriever = GraphGlobalRetriever(StubLLM(), FakeGraphStore(), settings=_settings())
    with pytest.raises(GuardrailViolation, match="graph global search is disabled"):
        await retriever.retrieve(Query(text="what are the themes?", tenant_id="acme"))


class _NoMatches(FakeGraphStore):
    """A graph the question names nothing in. Enough to prove the guard let the
    call through, without needing a Lucene index."""

    async def fulltext_entities(
        self, query: str, *, limit: int | None = None
    ) -> list[tuple[Entity, float]]:
        del query, limit
        return []


async def test_a_trusted_operator_is_let_through() -> None:
    """``trusted`` is an explicit assertion that the graph holds one tenant, so it
    cannot be arrived at by accident — and it must actually work."""
    retriever = GraphLocalRetriever(
        _NoMatches(), FakeVectorStore(), settings=_settings(mode="trusted")
    )
    assert await retriever.retrieve(Query(text="who is Acme?", tenant_id="acme")) == []


# ---------------------------------------------------------------------------
# Auto-selection and visibility
# ---------------------------------------------------------------------------
def test_auto_does_not_select_a_pipeline_that_is_guaranteed_to_refuse() -> None:
    """Otherwise a configuration decision becomes a 400 on every query."""
    from ragorc.pipeline.builder import RAGPipeline

    assert RAGPipeline(settings=_settings()).select_graph("auto") == "adaptive"
    assert RAGPipeline(settings=_settings(mode="trusted")).select_graph("auto") == "graphrag"


def test_asking_for_graphrag_by_name_still_gets_the_real_error() -> None:
    """Auto-selection substitutes; an explicit request must not be silently
    rerouted to a different pipeline than the one that was named."""
    from ragorc.pipeline.builder import RAGPipeline

    assert RAGPipeline(settings=_settings()).select_graph("graphrag") == "graphrag"


def test_health_says_the_graph_is_configured_but_unusable() -> None:
    warning = graph_isolation_warning(_settings())
    assert warning is not None and "every graph leg is refused" in warning
    assert graph_isolation_warning(_settings(mode="trusted")) is None
    assert graph_isolation_warning(_settings(graph=False)) is None


# ---------------------------------------------------------------------------
# The second layer, and the limit of it
# ---------------------------------------------------------------------------
def _chunk(cid: str, tenant: str, text: str) -> Chunk:
    chunk = Chunk(id=cid, content=text, document_id=f"doc-{tenant}", tenant_id=tenant)
    chunk.dense = np.ones(4, dtype=np.float32) / 2.0
    return chunk


async def test_a_by_id_fetch_will_not_cross_tenants() -> None:
    """The chokepoint. An id is not a filter, so naming a chunk used to fetch it
    whoever owned it — and on the graph path the ids come out of Neo4j, which
    stores no tenant at all."""
    store = FakeVectorStore()
    await store.upsert([_chunk("a", "acme", "ACME CONFIDENTIAL"), _chunk("b", "globex", "public")])

    both = await store.get(["a", "b"])
    assert {c.id for c in both} == {"a", "b"}, "unscoped, both are visible"

    scoped = await store.get(["a", "b"], tenant_id="globex")
    assert [c.id for c in scoped] == ["b"], "a foreign id must resolve to nothing"


async def test_local_search_passes_the_querys_tenant_to_the_body_fetch() -> None:
    """The call site, not just the primitive. Deleting the argument leaves the
    scoping intact and unreached, which is how the leak existed in the first
    place."""
    graph = FakeGraphStore()
    graph.entities["acme"] = Entity(
        name="Acme", type="ORG", description="d", source_chunk_ids=("a", "b")
    )
    graph.relations.append(Relation("Acme", "Acme", "SELF", weight=1.0))
    store = FakeVectorStore()
    await store.upsert([_chunk("a", "acme", "ACME CONFIDENTIAL"), _chunk("b", "globex", "public")])

    retriever = GraphLocalRetriever(graph, store, settings=_settings(mode="trusted"))
    out, _detail = await retriever.expand(
        Query(text="who is Acme?", tenant_id="globex"),
        [(graph.entities["acme"], 1.0)],
    )

    bodies = [c.chunk.id for c in out if c.chunk.document_id != "graph:local"]
    assert "a" not in bodies, f"acme's chunk reached a globex query: {bodies}"


async def test_trusted_mode_does_not_scope_the_graphs_own_output() -> None:
    """The limit of the fix, pinned so it is not mistaken for a guarantee.

    ``trusted`` means the operator asserted one graph per tenant. If that is
    untrue, entity names, descriptions and relationships still cross — the
    verbalized subgraph is built from whatever the traversal reached — and it is
    stamped with the querying tenant's id, which is worse than an unlabelled
    leak because the label says the content is yours. Scoping *that* is a schema
    change (entity identity becomes ``(tenant, name)``), which is why the default
    is to refuse rather than to filter.
    """
    graph = FakeGraphStore()
    private = Entity(name="Project Redacted", type="PROJECT", description="ACME INTERNAL")
    graph.entities["project redacted"] = private
    retriever = GraphLocalRetriever(graph, None, settings=_settings(mode="trusted"))

    out, _detail = await retriever.expand(Query(text="q", tenant_id="globex"), [(private, 1.0)])

    subgraph = next(c for c in out if c.chunk.document_id == "graph:local")
    assert "ACME INTERNAL" in subgraph.chunk.content
    assert subgraph.chunk.tenant_id == "globex", (
        "documented consequence: the graph's own text is stamped with the querying tenant"
    )


def test_the_setting_admits_no_middle_ground(caplog: Any) -> None:
    """There is deliberately no ``rls``-equivalent: nothing this library can do at
    query time makes an untenanted graph tenant-safe."""
    with pytest.raises(ValueError, match="graph_tenant_isolation"):
        Settings(llm={"api_key": "k"}, security={"graph_tenant_isolation": "rls"})
