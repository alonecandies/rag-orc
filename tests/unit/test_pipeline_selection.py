"""What `auto` resolves to, on each of the two wirings.

The library builds its components twice — `RAGPipeline` for library and CLI use,
`_LinearEngine` for the HTTP service — and each resolved `auto` with its own
logic. The server read two flags; the library read six. Four of five
configurations disagreed, including the shipped default:

    configuration                      builder    server
    graph + multihop, no communities   adaptive   graphrag
    self_rag only                      self_rag   naive
    crag + self_rag                    agentic    crag
    plain                              adaptive   naive

`adaptive` fans out to four stores; `naive` runs one hybrid leg. Same
configuration, different recall, different cost, different answer — and
`/health` reported the library's choice, so the service advertised a pipeline it
would not use.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.settings import Settings
from ragorc.pipeline.builder import RAGPipeline, select_pipeline
from ragorc.server.app import _LINEAR_UNSUPPORTED, PipelineName, _LinearEngine


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {"llm": {"api_key": "k"}, "cache": {"enabled": False}}
    base.update(over)
    return Settings(**base)


def _server(settings: Settings) -> Any:
    engine = object.__new__(_LinearEngine)
    engine.settings = settings
    return engine


CONFIGURATIONS = [
    ("plain", {}),
    ("crag", {"retrieval": {"crag_enabled": True}}),
    ("self_rag", {"generation": {"self_rag_enabled": True}}),
    (
        "crag+self_rag",
        {"retrieval": {"crag_enabled": True}, "generation": {"self_rag_enabled": True}},
    ),
    ("graph", {"graph": {"enabled": True}, "security": {"enforce_tenant_isolation": False}}),
    (
        "graph+multihop",
        {
            "graph": {"enabled": True, "multihop_enabled": True, "detect_communities": False},
            "security": {"enforce_tenant_isolation": False},
        },
    ),
    (
        "graph+communities",
        {
            "graph": {"enabled": True, "detect_communities": True},
            "security": {"enforce_tenant_isolation": False},
        },
    ),
]


@pytest.mark.parametrize(("name", "over"), CONFIGURATIONS, ids=[c[0] for c in CONFIGURATIONS])
def test_both_wirings_resolve_auto_the_same_way(name: str, over: dict[str, Any]) -> None:
    """Or, where they cannot, say so.

    The linear engine cannot run an agentic graph — that is the one legitimate
    difference, and it already has a substitution-with-warning path. Everything
    else must agree, because a deployment does not stop being the same deployment
    when it is reached over HTTP.
    """
    settings = _settings(**over)
    library = RAGPipeline(settings=settings, llm=object()).select_graph("auto")
    resolved, warnings = _server(settings)._resolve(PipelineName.AUTO)

    if PipelineName(library) in _LINEAR_UNSUPPORTED:
        assert resolved is PipelineName.ADAPTIVE, f"{name}: unsupported must substitute"
        assert warnings, f"{name}: the substitution was silent"
        assert library in warnings[0], f"{name}: the warning does not name what was asked for"
    else:
        assert resolved.value == library, f"{name}: library={library} server={resolved.value}"
        assert not warnings, f"{name}: an agreeing resolution warned anyway: {warnings}"


def test_the_server_reads_the_shared_resolver() -> None:
    """The guard is that there is one definition, not that two currently agree.
    They agreed on one of five configurations before this, which is what two
    definitions drifting looks like partway through.

    Walked over the AST. The first draft greped the source for `crag_enabled` and
    failed on the *comment* explaining the removal — a docstring is source text
    too, which is the same trap that has now caught three tests in this repo.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(_LinearEngine._resolve)))
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "select_pipeline" in calls, "the server does not use the shared resolver"

    read = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not read & {"crag_enabled", "self_rag_enabled", "detect_communities"}, (
        f"the server resolves auto from settings again: {sorted(read)}"
    )


def test_an_explicit_pipeline_is_never_re_resolved() -> None:
    """Asking for a graph by name must reach it — or be refused by name, not
    silently swapped for whatever `auto` would have chosen."""
    settings = _settings(retrieval={"crag_enabled": True})
    resolved, warnings = _server(settings)._resolve(PipelineName.NAIVE)
    assert resolved is PipelineName.NAIVE
    assert not warnings


def test_the_resolver_reports_its_reason() -> None:
    """A selection whose rationale is not recorded is one nobody can debug."""
    name, reason = select_pipeline(_settings(retrieval={"crag_enabled": True}))
    assert name == "crag"
    assert reason


def test_every_resolvable_name_is_a_pipeline_the_server_knows() -> None:
    """`select_pipeline` returns graph names and the server converts them to its
    enum. A name in one vocabulary and not the other is a ValueError on a live
    query, which is worse than the divergence it replaced."""
    known = {p.value for p in PipelineName}
    for _name, over in CONFIGURATIONS:
        chosen, _ = select_pipeline(_settings(**over))
        assert chosen in known, f"{chosen!r} is not a PipelineName"
