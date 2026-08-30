"""What a LangChain consumer gets, compared with a ragorc consumer.

The adapter is a second construction path over the same components, which is the
shape half this library's defects take. Four ways it diverged from the pipeline it
wraps, on the same index and the same settings:

* the tenant filter was inert whenever the tenant came from `settings.tenant_id`,
* the parent/window substitution never ran, so the generator saw the child,
* `top_k` was pinned to `Query`'s hardcoded 10, suppressing the settings fallback,
* an LCEL dict raised `TypeError`, and a `Query` input was asked as its repr.

None of them is visible from inside the adapter. Each is only a defect *relative*
to what `rag.query()` does with the same inputs, which is why these tests assert
the two agree rather than asserting the adapter is self-consistent.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Chunk, Query, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings

_PARENT = "ACME FY24 REPORT. Revenue grew 40%. Margin fell 3pts. Headcount +12%."
_CHILD = "Revenue grew 40%."


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
    }
    base.update(over)
    return Settings(**base)


def _scored(chunk: Chunk, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, source=RetrievalSource.DENSE, rank=0)


class _Leg:
    name = "inner"

    def __init__(self, *chunks: Chunk) -> None:
        self.chunks = chunks
        self.asked: list[int | None] = []

    async def retrieve(self, query: Query, *, top_k: int | None = None, **kw: Any) -> Any:
        self.asked.append(top_k)
        return [_scored(c) for c in self.chunks]


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------
async def test_the_adapter_retrieves_at_the_configured_width() -> None:
    """`Query.top_k` defaults to a hardcoded 10 and that default is *truthy*, so
    passing it down suppressed every `top_k or query.top_k or settings...`
    fallback: a consumer with `retrieval.top_k = 25` got 10 from the wrapped
    retriever and 25 from `rag.query()` on the same settings."""
    from ragorc.adapters.langchain import _retrieve_as_documents

    settings = _settings(retrieval={"top_k": 25})
    leg = _Leg(Chunk(id="c1", content=_CHILD, document_id="d1"))

    await _retrieve_as_documents(
        leg, "revenue?", top_k=None, filters=None, tenant_id=None, settings=settings, extra=None
    )

    assert leg.asked == [25], f"asked the inner leg for {leg.asked}, settings say 25"


async def test_an_explicit_width_still_wins() -> None:
    from ragorc.adapters.langchain import _retrieve_as_documents

    leg = _Leg(Chunk(id="c1", content=_CHILD, document_id="d1"))
    await _retrieve_as_documents(
        leg, "q", top_k=3, filters=None, tenant_id=None, settings=_settings(retrieval={"top_k": 25}),
        extra=None,
    )
    assert leg.asked == [3]


# ---------------------------------------------------------------------------
# The substitution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["parent_text", "window_text"])
async def test_a_langchain_consumer_sees_the_source_not_the_derived_unit(key: str) -> None:
    """The same swap `ContextPacker` performs before a prompt is rendered."""
    from ragorc.adapters.langchain import _retrieve_as_documents

    child = Chunk(id="c1", content=_CHILD, document_id="d1", parent_id="p1", metadata={key: _PARENT})
    docs = await _retrieve_as_documents(
        _Leg(child), "revenue?", top_k=None, filters=None, tenant_id=None,
        settings=_settings(retrieval={"parent_expansion": True}), extra=None,
    )

    assert docs[0].page_content == _PARENT, f"the consumer was handed {docs[0].page_content!r}"


async def test_the_substitution_honours_the_same_flag() -> None:
    """Off means off in both places, or the adapter is a third policy."""
    from ragorc.adapters.langchain import _retrieve_as_documents

    child = Chunk(
        id="c1", content=_CHILD, document_id="d1", parent_id="p1", metadata={"parent_text": _PARENT}
    )
    docs = await _retrieve_as_documents(
        _Leg(child), "revenue?", top_k=None, filters=None, tenant_id=None,
        settings=_settings(retrieval={"parent_expansion": False}), extra=None,
    )
    assert docs[0].page_content == _CHILD


def test_both_paths_share_one_definition_of_the_substitution() -> None:
    """The guard is that there is one, not that two currently agree."""
    import inspect

    from ragorc.context.pack import ContextPacker, expand_units

    assert "expand_units(" in inspect.getsource(ContextPacker._expand)
    assert callable(expand_units)


# ---------------------------------------------------------------------------
# Tenant scoping
# ---------------------------------------------------------------------------
class _Foreign:
    """A LangChain retriever holding one tenant's document and someone else's."""

    async def ainvoke(self, value: Any, config: Any = None) -> list[Any]:
        from ragorc.adapters.langchain import to_langchain_documents

        return to_langchain_documents(
            [
                _scored(Chunk(id="ours", content="ours", document_id="d", tenant_id="globex")),
                _scored(
                    Chunk(
                        id="theirs",
                        content="ACME Q3 REVENUE: $41.2M (CONFIDENTIAL)",
                        document_id="d",
                        tenant_id="acme",
                    )
                ),
            ]
        )


@pytest.mark.parametrize(
    "where", ["query", "settings"], ids=["tenant on the query", "tenant in settings"]
)
async def test_the_filter_mode_scopes_whichever_way_the_tenant_arrives(where: str) -> None:
    """`filter` returned early when it had no tenant to compare against, and the
    tenant was only ever read from the call. A single-tenant deployment names it
    once, in settings — and got the mode's protection not at all."""
    from ragorc.adapters.langchain import LangChainRetriever

    settings = _settings(
        security={
            "enforce_tenant_isolation": False,
            "foreign_retriever_tenant_isolation": "filter",
        },
        **({"tenant_id": "globex"} if where == "settings" else {}),
    )
    leg = LangChainRetriever(_Foreign(), settings=settings)
    query = Query(text="revenue?", tenant_id="globex" if where == "query" else None)

    chunks = await leg.retrieve(query, top_k=10)

    ids = [c.chunk.id for c in chunks]
    assert ids == ["ours"], f"a foreign document survived the filter: {ids}"


async def test_trusted_mode_is_unchanged() -> None:
    """`trusted` is the operator asserting the wrapped retriever holds one tenant,
    which this must not second-guess."""
    from ragorc.adapters.langchain import LangChainRetriever

    settings = _settings(
        tenant_id="globex",
        security={
            "enforce_tenant_isolation": False,
            "foreign_retriever_tenant_isolation": "trusted",
        },
    )
    chunks = await LangChainRetriever(_Foreign(), settings=settings).retrieve(
        Query(text="q"), top_k=10
    )
    assert len(chunks) == 2


# ---------------------------------------------------------------------------
# as_runnable's query branch
# ---------------------------------------------------------------------------
class _Component:
    def __init__(self) -> None:
        self.asked: list[str] = []

    async def query(self, question: str, **kw: Any) -> str:
        self.asked.append(question)
        return "answered"


async def test_a_query_object_is_not_asked_as_its_repr() -> None:
    """`str(value)` turned a `Query` into its dataclass repr and asked *that*:
    "Query(text='why is late chunking cheaper?', original=..., top_k=10, ...)"."""
    from ragorc.adapters.langchain import _dispatch

    component = _Component()
    await _dispatch(
        component, "query", Query(text="why is late chunking cheaper?"), {}, _settings()
    )

    assert component.asked == ["why is late chunking cheaper?"], component.asked


async def test_an_lcel_dict_with_extra_keys_does_not_raise() -> None:
    """`chat_history` is the canonical one. Forwarding it verbatim raised
    `TypeError: query() got an unexpected keyword argument`, so adding memory to a
    chain broke retrieval."""
    from ragorc.adapters.langchain import _dispatch

    component = _Component()
    result = await _dispatch(
        component, "query", {"question": "why?", "chat_history": ["earlier turn"]}, {}, _settings()
    )

    assert result == "answered"
    assert component.asked == ["why?"]


async def test_a_keyword_the_component_accepts_is_still_forwarded() -> None:
    """Dropping unknown keys must not drop known ones."""
    from ragorc.adapters.langchain import _dispatch

    seen: dict[str, Any] = {}

    class _Narrow:
        async def query(self, question: str, pipeline: str = "auto") -> str:
            seen["pipeline"] = pipeline
            return "ok"

    await _dispatch(
        _Narrow(), "query", {"question": "why?", "pipeline": "crag", "junk": 1}, {}, _settings()
    )
    assert seen == {"pipeline": "crag"}
