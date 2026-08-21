"""Two invariants the source states in a comment and nothing checked.

A comment is not a test. Both behaviours below are load-bearing, both are
asserted in prose next to the code that implements them, and both can be deleted
without turning the suite red — which means the prose is the only thing holding
them in place.

* Multi-document HyDE renders the *same* prompt N times, so ``temperature=0.8``
  is the entire source of diversity between the N samples. Drop the kwarg and the
  provider layer sends ``0.0``, the LLM cache collapses the N identical prompts
  into one answer, and mean-pooling N copies of one vector returns that vector:
  N times the cost of single-document HyDE for none of the benefit.
* ``require_generated_query_isolation`` guards the *first* line of
  ``construct_and_execute``. Its position is the point — a refusal that arrives
  after ``execute_readonly`` has already read every tenant's rows is not a
  refusal, it is a log line. The function is covered as a standalone unit, so
  moving the call below the execution keeps the suite green.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from ragorc.construct.text_to_cypher import TextToCypherConstructor
from ragorc.construct.text_to_sql import TextToSQLConstructor
from ragorc.core.errors import GuardrailViolation
from ragorc.core.models import Query, Usage
from ragorc.core.protocols import BatchStructuredLLM
from ragorc.core.schemas import CypherQuery, HyDEOutput, SQLQuery
from ragorc.core.settings import Settings
from ragorc.llm.prompts import get_prompt
from ragorc.translate.hyde import HyDETranslator
from tests.fakes import FakeGraphStore, FakeRelationalStore, StubLLM

HYDE_QUESTION = "How long do refunds take?"


# ---------------------------------------------------------------------------
# Recording doubles
# ---------------------------------------------------------------------------
class BatchRecordingLLM(StubLLM):
    """``StubLLM`` plus the kwargs handed to its *native* fan-out.

    ``StubLLM`` already records every per-prompt call, but the native branch makes
    exactly one call — ``batch_structured`` — and whether the sampling parameters
    survive that call is the thing under test.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.batch_calls: list[dict[str, Any]] = []

    async def batch_structured(
        self,
        prompts: Sequence[str],
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> list[tuple[Any, Usage]]:
        self.batch_calls.append({"prompts": list(prompts), **kwargs})
        return await super().batch_structured(prompts, schema, **kwargs)


class SequentialOnlyLLM:
    """Exactly the ``LLM`` protocol — deliberately no ``batch_structured``.

    ``batch_structured`` is a convenience the protocol does not require, so anyone
    supplying their own client takes the sequential fallback. That fallback is a
    second, independent copy of the sampling parameters, which is precisely how
    the two branches drift apart.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, prompt: str, *, system: str | None = None, model: str | None = None, **kwargs: Any
    ) -> tuple[str, Usage]:
        return "", Usage()

    async def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Usage]:
        self.calls.append({"prompt": prompt, "system": system, "model": model, **kwargs})
        return HyDEOutput(document=f"hypothesis {len(self.calls)}"), Usage(calls=1)

    async def stream(
        self, prompt: str, *, system: str | None = None, model: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        yield ""

    async def batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, Usage]]:
        return [("", Usage()) for _ in prompts]


# ---------------------------------------------------------------------------
# (1) Multi-document HyDE: temperature is the only source of diversity
# ---------------------------------------------------------------------------
async def test_multi_document_hyde_samples_the_batch_llm_at_a_non_zero_temperature(
    settings: Settings,
) -> None:
    """N hypotheses average away a single hallucination only if the N differ.

    Nothing else in this call can make them differ: the prompts are one rendered
    string repeated N times, and the provider layer defaults an absent
    ``temperature`` to ``0.0`` — at which point the LLM cache, keyed on the
    prompt, returns the first answer N times and the mean of N identical vectors
    is that vector. So the value has to reach the model, and the identical
    prompts asserted here are why.
    """
    llm = BatchRecordingLLM(responses={"HyDEOutput": HyDEOutput(document="A refunds passage.")})
    assert isinstance(llm, BatchStructuredLLM), "this fixture must exercise the native branch"

    translator = HyDETranslator(llm, settings, n_documents=3)
    await translator.translate(Query(text=HYDE_QUESTION))

    rendered = get_prompt("hyde").render(question=HYDE_QUESTION)
    assert len(llm.batch_calls) == 1, "the native fan-out is one call, not one per document"
    assert llm.batch_calls[0]["prompts"] == [rendered, rendered, rendered], (
        "the N prompts are identical, which is what leaves temperature as the "
        "only thing that can vary between samples"
    )
    assert llm.batch_calls[0].get("temperature") == 0.8
    assert [c.get("temperature") for c in llm.calls_for("hyde")] == [0.8, 0.8, 0.8], (
        "and the fan-out must pass it down to every sample rather than keep it"
    )


async def test_multi_document_hyde_samples_the_sequential_fallback_at_the_same_temperature(
    settings: Settings,
) -> None:
    """The fallback must not be the branch where diversity quietly disappears.

    It is reached by every caller who plugged in their own client, so a
    temperature that lives only on the native path means those callers pay N times
    for N copies of one hypothesis while the test suite, exercising the batch
    stub, sees nothing wrong. Same prompts, same temperature, three distinct
    documents in the order the samples were requested.
    """
    llm = SequentialOnlyLLM()
    assert not isinstance(llm, BatchStructuredLLM), (
        "the fixture must lack the native fan-out, or this tests the other branch"
    )

    translator = HyDETranslator(llm, settings, n_documents=3)
    out, usage = await translator.translate(Query(text=HYDE_QUESTION))

    rendered = get_prompt("hyde").render(question=HYDE_QUESTION)
    assert [c["prompt"] for c in llm.calls] == [rendered, rendered, rendered]
    assert [c.get("temperature") for c in llm.calls] == [0.8, 0.8, 0.8]
    assert out.metadata["hyde_documents"] == ["hypothesis 1", "hypothesis 2", "hypothesis 3"]
    assert out.hypothetical == "hypothesis 1"
    assert usage.calls == 3, "every sample is billed, so every sample must be a real call"


# ---------------------------------------------------------------------------
# (2) The generated-query guard runs before execution, at both call sites
# ---------------------------------------------------------------------------
@pytest.fixture
def refusing() -> Settings:
    """Isolation on with the default ``reject`` mode: the one configuration in
    which a generated statement must never reach a store."""
    return Settings(
        security={"enforce_tenant_isolation": True, "generated_query_isolation": "reject"},
        cache={"enabled": False},
        llm={"api_key": "test-key"},
    )


@pytest.fixture
def declared() -> Settings:
    """Isolation on, but the operator has declared how it is enforced — the
    control that proves a refusal is what stopped execution, not a store that
    never records."""
    return Settings(
        security={"enforce_tenant_isolation": True, "generated_query_isolation": "trusted"},
        cache={"enabled": False},
        llm={"api_key": "test-key"},
    )


async def test_text_to_sql_refuses_before_the_database_is_read(
    refusing: Settings, declared: Settings
) -> None:
    """A refusal that arrives after the rows have been read is not a refusal.

    The leg exists because a vector index cannot count, and counting reads whole
    tables — so under enforced isolation the one thing that must not happen is
    ``execute_readonly`` running an unscoped statement and *then* raising. The
    guard's value is entirely in its position at the top of the method, and the
    only way to observe position is to observe that the store recorded nothing.
    """
    sql = "SELECT count(*) FROM orders"
    query = Query(text="how many orders are there?", tenant_id="acme")

    refused_store = FakeRelationalStore()
    refused_llm = StubLLM(responses={"SQLQuery": SQLQuery(sql=sql)})
    with pytest.raises(GuardrailViolation) as exc:
        await TextToSQLConstructor(
            refused_llm, refused_store, settings=refusing
        ).construct_and_execute(query)
    assert exc.value.rule == "generated_query_isolation"
    assert refused_store.executed == [], "a refused leg must read zero rows from the database"
    assert refused_llm.calls == [], "and must not pay for a statement it will not run"

    executed_store = FakeRelationalStore()
    rows, validation, _ = await TextToSQLConstructor(
        StubLLM(responses={"SQLQuery": SQLQuery(sql=sql)}), executed_store, settings=declared
    ).construct_and_execute(query)
    assert executed_store.executed == [validation.sql], (
        "control: the same store, model and query do execute exactly once — and "
        "only the guard's rewritten SQL — once isolation is declared"
    )
    assert rows == [{"count": 3}]


async def test_text_to_cypher_refuses_before_the_graph_is_traversed(
    refusing: Settings, declared: Settings
) -> None:
    """The graph leg needs the same ordering, and got it from a separate line.

    Two call sites means two chances to lose it, and the guard is covered only as
    a standalone function — so the Cypher path could keep reading every tenant's
    subgraph while the SQL path was fixed, with nothing red.
    """
    cypher = "MATCH (a:Entity)-[r:WORKS_FOR]->(b:Entity) RETURN a.name, b.name"
    query = Query(text="who works for whom?", tenant_id="acme")

    refused_store = FakeGraphStore()
    refused_llm = StubLLM(responses={"CypherQuery": CypherQuery(cypher=cypher)})
    with pytest.raises(GuardrailViolation) as exc:
        await TextToCypherConstructor(
            refused_llm, refused_store, settings=refusing
        ).construct_and_execute(query)
    assert exc.value.rule == "generated_query_isolation"
    assert refused_store.executed == [], (
        "a refused leg must traverse nothing — not even the EXPLAIN dry run, which "
        "compiles the unscoped query against the operator's graph"
    )
    assert refused_llm.calls == [], "and must not pay for a statement it will not run"

    executed_store = FakeGraphStore()
    rows, validation, _ = await TextToCypherConstructor(
        StubLLM(responses={"CypherQuery": CypherQuery(cypher=cypher)}),
        executed_store,
        settings=declared,
    ).construct_and_execute(query)
    assert executed_store.executed == [f"EXPLAIN {validation.cypher}", validation.cypher], (
        "control: with isolation declared, the dry run and then the guard's "
        "bounded Cypher both reach the store, in that order"
    )
    assert rows == []
