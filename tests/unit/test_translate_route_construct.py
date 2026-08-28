"""Query translation, routing and construction."""

from __future__ import annotations

import numpy as np
import pytest

from ragorc.construct.self_query import AttributeInfo, SelfQueryConstructor
from ragorc.construct.text_to_cypher import TextToCypherConstructor
from ragorc.construct.text_to_sql import TextToSQLConstructor
from ragorc.core.errors import ConstructionError
from ragorc.core.models import DataStore, Query, Usage
from ragorc.core.protocols import BatchStructuredLLM
from ragorc.core.schemas import (
    CypherQuery,
    DecompositionOutput,
    FilterCondition,
    HyDEOutput,
    MetadataFilterOutput,
    MultiQueryOutput,
    RouteOutput,
    SQLQuery,
    StepBackOutput,
)
from ragorc.core.settings import Settings
from ragorc.route.hybrid import HybridRouter, rule_route
from ragorc.route.logical import LogicalRouter
from ragorc.route.semantic import SemanticRouter
from ragorc.translate import (
    CompositeTranslator,
    DecompositionTranslator,
    HyDETranslator,
    MultiQueryTranslator,
    StepBackTranslator,
    clean_variants,
)
from tests.fakes import StubLLM


@pytest.fixture
def settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k"},
    )


# ---------------------------------------------------------------------------
# Variant cleaning — the filter that stops fan-out being pure waste
# ---------------------------------------------------------------------------
def test_clean_variants_strips_enumeration_and_quotes() -> None:
    out = clean_variants(
        [
            "1. When is money returned?",
            '"How soon is a refund paid?"',
            "- What is the payout delay?",
        ],
        original="How long do refunds take?",
    )
    assert all(not v[0].isdigit() and not v.startswith(('"', "-")) for v in out)
    assert len(out) == 3


def test_clean_variants_drops_near_duplicates_of_the_original() -> None:
    out = clean_variants(
        ["How long do refunds take", "When will I receive my money back?"],
        original="How long do refunds take?",
    )
    assert out == ["When will I receive my money back?"]


def test_clean_variants_drops_duplicates_of_each_other() -> None:
    out = clean_variants(
        [
            "When is the money returned?",
            "When is the money returned!",
            "What is the payout window?",
        ],
        original="original question here",
    )
    assert len(out) == 2


def test_clean_variants_respects_the_cap() -> None:
    """The cap applies to variants that survive the similarity filter, so the
    fixture uses genuinely different vocabulary rather than numbered near-copies —
    which the near-duplicate filter would (correctly) remove first."""
    candidates = [
        "What is the reimbursement timeline for cancelled orders?",
        "How quickly does the payment provider settle a chargeback?",
        "When does store credit appear on a customer account?",
        "Which department approves an exceptional payout?",
        "Are partial returns handled on a separate schedule?",
    ]
    out = clean_variants(candidates, original="an unrelated original question", max_variants=3)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Translators
# ---------------------------------------------------------------------------
async def test_multi_query_populates_variants(settings: Settings) -> None:
    llm = StubLLM(
        responses={
            "MultiQueryOutput": MultiQueryOutput(
                queries=[
                    "What is the payout window for returned orders?",
                    "How soon does a customer receive reimbursement?",
                ]
            )
        }
    )
    out, usage = await MultiQueryTranslator(llm, settings, n=2).translate(
        Query(text="How long do refunds take?")
    )
    assert len(out.variants) == 2
    assert out.text == "How long do refunds take?", "the original query must be preserved"
    assert usage.calls == 1


async def test_translation_returns_a_copy_not_a_mutation(settings: Settings) -> None:
    """The pipeline keeps the pre-translation query for grading and the trace."""
    llm = StubLLM(
        responses={
            "MultiQueryOutput": MultiQueryOutput(queries=["a wholly different phrasing here"])
        }
    )
    original = Query(text="How long do refunds take?")
    out, _ = await MultiQueryTranslator(llm, settings).translate(original)
    assert original.variants == ()
    assert out.variants


async def test_step_back_records_the_general_question(settings: Settings) -> None:
    llm = StubLLM(
        responses={
            "StepBackOutput": StepBackOutput(
                step_back_question="What is the company's refund and returns policy?",
                reasoning="broader context",
            )
        }
    )
    out, _ = await StepBackTranslator(llm, settings).translate(
        Query(text="How long do refunds take?")
    )
    assert "refund and returns policy" in out.metadata["step_back"]
    assert any("policy" in v for v in out.variants)


async def test_decomposition_respects_atomic_questions(settings: Settings) -> None:
    """Forcing a split on an atomic question multiplies cost for nothing."""
    llm = StubLLM(
        responses={
            "DecompositionOutput": DecompositionOutput(
                sub_questions=["How long do refunds take?"], is_decomposable=False
            )
        }
    )
    query = Query(text="How long do refunds take?")
    out, _ = await DecompositionTranslator(llm, settings).translate(query)
    assert out.variants == ()
    assert "sub_questions" not in out.metadata


async def test_decomposition_splits_a_compound_question(settings: Settings) -> None:
    llm = StubLLM(
        responses={
            "DecompositionOutput": DecompositionOutput(
                sub_questions=[
                    "What is the refund processing window?",
                    "What is the shipping cost threshold?",
                ],
                is_decomposable=True,
            )
        }
    )
    out, _ = await DecompositionTranslator(llm, settings).translate(
        Query(text="What are the refund and shipping policies?")
    )
    assert len(out.metadata["sub_questions"]) == 2


async def test_hyde_blends_question_and_hypothesis(settings: Settings, embedder) -> None:
    """Pure HyDE drifts when the hypothesis is topically wrong; the blend bounds
    that risk, so the search vector must differ from both endpoints."""
    llm = StubLLM(
        responses={
            "HyDEOutput": HyDEOutput(
                document="Refunds are issued to the original payment method within fourteen days."
            )
        }
    )
    translator = HyDETranslator(llm, settings, blend=0.3)
    query = Query(text="How long do refunds take?")
    out, _ = await translator.translate(query)
    assert out.hypothetical

    blended = await translator.embed_for_search(out, embedder)
    question_only = await embedder.embed_query(out.text)
    hypothesis_only = (await embedder.embed_documents([out.hypothetical]))[0]

    assert np.linalg.norm(blended) == pytest.approx(1.0, abs=1e-5), "must be renormalized"
    assert not np.allclose(blended, question_only)
    assert not np.allclose(blended, hypothesis_only)


async def test_hyde_pure_mode_uses_only_the_hypothesis(settings: Settings, embedder) -> None:
    llm = StubLLM(
        responses={"HyDEOutput": HyDEOutput(document="A confident passage about refunds.")}
    )
    translator = HyDETranslator(llm, settings, blend=0.0)
    out, _ = await translator.translate(Query(text="How long do refunds take?"))
    blended = await translator.embed_for_search(out, embedder)
    hypothesis = (await embedder.embed_documents([out.hypothetical]))[0]
    assert np.allclose(blended, hypothesis / np.linalg.norm(hypothesis), atol=1e-5)


async def test_hyde_multi_document_works_without_a_batch_capable_llm(
    settings: Settings,
) -> None:
    """``batch_structured`` is a convenience the LLM protocol does not require.

    Multi-document HyDE used to call it unconditionally, so anyone plugging in
    their own client got an ``AttributeError`` the moment ``n_documents > 1``. The
    fallback must produce the same documents and the same bill.
    """

    class MinimalLLM:
        """Exactly the LLM protocol, nothing more."""

        def __init__(self) -> None:
            self.structured_calls = 0

        async def complete(self, prompt, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return "", Usage()

        async def structured(self, prompt, schema, **kwargs):  # noqa: ANN001, ANN003, ANN201
            self.structured_calls += 1
            return HyDEOutput(document=f"hypothesis {self.structured_calls}"), Usage(
                calls=1, prompt_tokens=7
            )

        async def stream(self, prompt, **kwargs):  # noqa: ANN001, ANN003, ANN201
            yield ""

        async def batch(self, prompts, **kwargs):  # noqa: ANN001, ANN003, ANN201
            return [("", Usage()) for _ in prompts]

    assert not isinstance(MinimalLLM(), BatchStructuredLLM), (
        "the fixture must lack the capability, or the test proves nothing"
    )

    llm = MinimalLLM()
    translator = HyDETranslator(llm, settings, n_documents=3)
    out, usage = await translator.translate(Query(text="How long do refunds take?"))

    assert llm.structured_calls == 3, "one call per requested document"
    assert len(out.metadata["hyde_documents"]) == 3
    assert usage.prompt_tokens == 21, "every call must be billed, not just the last"


async def test_composite_survives_a_failing_translator(settings: Settings) -> None:
    """Translation is an enhancement: a failure must degrade to the original
    query, never fail the request."""

    class Broken:
        name = "broken"

        async def translate(self, query):
            raise RuntimeError("model down")

    good = MultiQueryTranslator(
        StubLLM(
            responses={
                "MultiQueryOutput": MultiQueryOutput(queries=["an entirely different phrasing"])
            }
        ),
        settings,
    )
    composite = CompositeTranslator([Broken(), good])
    out, _ = await composite.translate(Query(text="How long do refunds take?"))
    assert out.variants, "the working translator must still contribute"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many enterprise customers signed up in Q1?", DataStore.RELATIONAL),
        ("What is the average order total per region since 2023?", DataStore.RELATIONAL),
        ("Show me the top 5 customers by revenue", DataStore.RELATIONAL),
        ("How is Northwind connected to Contoso?", DataStore.GRAPH),
        ("What is the shortest path between these two teams?", DataStore.GRAPH),
        ("Who works with the platform team?", DataStore.GRAPH),
    ],
)
def test_rule_fast_path_identifies_the_store(question: str, expected: DataStore) -> None:
    decision = rule_route(question)
    assert decision is not None, f"no rule fired for {question!r}"
    assert expected in decision.stores


def test_rule_fast_path_handles_both_cues() -> None:
    decision = rule_route("how many suppliers are connected to Acme?")
    assert decision is not None
    assert {DataStore.RELATIONAL, DataStore.GRAPH} <= set(decision.stores)


def test_rule_fast_path_detects_conversational_input() -> None:
    for greeting in ("hello", "hi!", "thanks", "who are you"):
        decision = rule_route(greeting)
        assert decision is not None and DataStore.NONE in decision.stores


def test_rule_fast_path_defers_on_semantic_questions() -> None:
    """A false positive here silently routes to a store that cannot answer, so
    the patterns must stay narrow."""
    for question in (
        "Why did we choose late chunking?",
        "Explain how reranking works.",
        "What is the refund policy?",
    ):
        assert rule_route(question) is None


async def test_logical_router_maps_stores(settings: Settings) -> None:
    llm = StubLLM(
        responses={"RouteOutput": RouteOutput(datastores=["relational", "vector"], confidence=0.8)}
    )
    decision, _ = await LogicalRouter(llm, settings).route(Query(text="how many customers?"))
    assert set(decision.stores) == {DataStore.RELATIONAL, DataStore.VECTOR}


def test_route_output_schema_rejects_an_unknown_store() -> None:
    """First line of defence: the ``Literal`` in the schema means an invented
    store name cannot even be constructed, so it never reaches the router."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RouteOutput(datastores=["elasticsearch"])


def test_logical_router_coercion_defaults_to_vector(settings: Settings) -> None:
    """Second line of defence, for the non-strict ``json_object`` path where a
    provider that ignores ``response_format`` can return looser output: coercion
    drops what it does not recognize and falls back rather than raising."""
    router = LogicalRouter(StubLLM(), settings)
    assert router._coerce(["elasticsearch", "mongo"]) == (DataStore.VECTOR,)
    assert router._coerce([]) == (DataStore.VECTOR,)
    # Deduplicated, and paired with vector — see the companion test below.
    assert router._coerce(["graph", "graph"]) == (DataStore.GRAPH, DataStore.VECTOR)


def test_a_structured_route_always_carries_vector(settings: Settings) -> None:
    """Generated SQL and Cypher fail in ways a fixed query does not — a
    hallucinated column, a misread schema, a guard rejection. Alone, any of those
    turns into "0 chunks" and the request abstains on a question the corpus
    answers. Vector search is the only leg that can attempt any question, so it
    rides along.
    """
    router = LogicalRouter(StubLLM(), settings)
    for requested in (["relational"], ["graph"], ["web"], ["relational", "graph"]):
        stores = router._coerce(requested)
        assert DataStore.VECTOR in stores, f"{requested} lost its vector companion"

    # Vector alone stays alone, and NONE is never paired with anything.
    assert router._coerce(["vector"]) == (DataStore.VECTOR,)
    assert router._coerce(["none"]) == (DataStore.NONE,)


async def test_logical_router_never_fails_the_request(settings: Settings) -> None:
    class Broken(StubLLM):
        async def structured(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("router model down")

    decision, _ = await LogicalRouter(Broken(), settings).route(Query(text="anything"))
    assert decision.stores == (DataStore.VECTOR,)
    assert decision.method == "fallback"


async def test_logical_router_drops_none_when_combined(settings: Settings) -> None:
    llm = StubLLM(responses={"RouteOutput": RouteOutput(datastores=["none", "vector"])})
    decision, _ = await LogicalRouter(llm, settings).route(Query(text="q"))
    assert DataStore.NONE not in decision.stores


async def test_semantic_router_costs_no_llm_call(settings: Settings, embedder) -> None:
    router = SemanticRouter(embedder, settings=settings)
    decision, usage = await router.route(Query(text="what is the exact syntax for this command"))
    assert decision.prompt_name in ("answer_technical", "answer_concise", "answer_default")
    assert usage.calls == 0, "semantic routing must never call a model"


async def test_semantic_router_embeds_exemplars_once(settings: Settings, embedder) -> None:
    router = SemanticRouter(embedder, settings=settings)
    await router.route(Query(text="how do I configure this"))
    calls_after_first = len(embedder.calls)
    await router.route(Query(text="when was it released"))
    # One additional call for the second query's own embedding, and no re-embed
    # of the exemplar matrix.
    assert len(embedder.calls) == calls_after_first + 1


async def test_hybrid_router_skips_the_llm_when_a_rule_fires(settings: Settings, embedder) -> None:
    llm = StubLLM(responses={"RouteOutput": RouteOutput(datastores=["vector"])})
    router = HybridRouter(
        logical=LogicalRouter(llm, settings),
        semantic=SemanticRouter(embedder, settings=settings),
        settings=settings,
    )
    decision, _ = await router.route(Query(text="How many customers are in the US?"))
    assert DataStore.RELATIONAL in decision.stores
    assert not any(c.get("stage") == "route" for c in llm.calls), "rule path must be free"


async def test_hybrid_router_calls_the_llm_for_semantic_questions(
    settings: Settings, embedder
) -> None:
    llm = StubLLM(responses={"RouteOutput": RouteOutput(datastores=["vector"], confidence=0.9)})
    router = HybridRouter(
        logical=LogicalRouter(llm, settings),
        semantic=SemanticRouter(embedder, settings=settings),
        settings=settings,
    )
    decision, _ = await router.route(Query(text="Why did the team choose this design?"))
    assert any(c.get("stage") == "route" for c in llm.calls)
    assert decision.prompt_name is not None, "the semantic leg must still select a prompt"


async def test_hybrid_router_survives_a_dead_leg(settings: Settings, embedder) -> None:
    class Broken:
        async def route(self, query):
            raise RuntimeError("leg down")

    router = HybridRouter(
        logical=Broken(), semantic=SemanticRouter(embedder, settings=settings), settings=settings
    )
    decision, _ = await router.route(Query(text="Why did the team choose this design?"))
    assert decision.stores, "a dead leg must not empty the route"


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
async def test_text_to_sql_validates_before_returning(settings: Settings, relational_store) -> None:
    llm = StubLLM(
        responses={"SQLQuery": SQLQuery(sql="SELECT name FROM customers WHERE country = 'US'")}
    )
    validated, usage = await TextToSQLConstructor(
        llm, relational_store, settings=settings
    ).construct(Query(text="which customers are in the US?"))
    assert "LIMIT" in validated.sql.upper(), "the guard must bound every query"
    assert usage.calls == 1


async def test_text_to_sql_repairs_once_then_gives_up(settings: Settings, relational_store) -> None:
    """A blocked pattern is still blocked on retry, so the budget is one."""
    llm = StubLLM(responses={"SQLQuery": SQLQuery(sql="DROP TABLE customers")})
    with pytest.raises(ConstructionError):
        await TextToSQLConstructor(llm, relational_store, settings=settings).construct(
            Query(text="delete everything")
        )
    structured = [c for c in llm.calls if c["kind"] == "structured"]
    assert len(structured) == 2, f"expected one attempt plus one repair, got {len(structured)}"


async def test_text_to_sql_renders_rows_as_evidence(settings: Settings, relational_store) -> None:
    llm = StubLLM(responses={"SQLQuery": SQLQuery(sql="SELECT count(*) FROM orders")})
    constructor = TextToSQLConstructor(llm, relational_store, settings=settings)
    rows = [{"country": "US", "total": 4200}, {"country": "GB", "total": 3100}]
    chunks = constructor.to_chunks(rows, "SELECT ...")
    assert len(chunks) == 1, "a result set is one piece of evidence, not one per row"
    assert "US" in chunks[0].chunk.content and "4200" in chunks[0].chunk.content
    assert constructor.to_chunks([], "SELECT ...") == [], "no rows means no evidence chunk"


async def test_text_to_cypher_validates_and_bounds(settings: Settings, graph_store) -> None:
    llm = StubLLM(
        responses={
            "CypherQuery": CypherQuery(
                cypher="MATCH (a:Entity)-[r:WORKS_FOR]->(b:Entity) RETURN a.name, b.name"
            )
        }
    )
    validated, _ = await TextToCypherConstructor(llm, graph_store, settings=settings).construct(
        Query(text="who works for whom?")
    )
    assert "LIMIT" in validated.cypher.upper()


async def test_text_to_cypher_rejects_a_write(settings: Settings, graph_store) -> None:
    llm = StubLLM(
        responses={"CypherQuery": CypherQuery(cypher="MATCH (n) DETACH DELETE n RETURN 1")}
    )
    with pytest.raises(ConstructionError):
        await TextToCypherConstructor(llm, graph_store, settings=settings).construct(
            Query(text="remove everything")
        )


async def test_self_query_splits_and_validates(settings: Settings) -> None:
    attributes = [
        AttributeInfo("year", "int", "publication year", (2021, 2022)),
        AttributeInfo("author", "string", "first author surname", ("Ho",)),
    ]
    llm = StubLLM(
        responses={
            "MetadataFilterOutput": MetadataFilterOutput(
                query="diffusion models",
                conditions=[
                    FilterCondition(field="year", op="gt", value="2022"),
                    FilterCondition(field="author", op="eq", value="Ho"),
                    FilterCondition(field="venue", op="eq", value="NeurIPS"),
                ],
            )
        }
    )
    result, _ = await SelfQueryConstructor(llm, attributes, settings=settings).construct(
        Query(text="papers on diffusion models after 2022 by Ho")
    )
    assert result.query_text == "diffusion models", "constraints must be removed from the text"
    assert result.filters == {"year": {"gt": 2022}, "author": "Ho"}
    assert any("venue" in d for d in result.dropped), (
        "an invented field must be dropped, not passed"
    )
    assert any("2022" in c for c in result.coerced)


async def test_self_query_merges_two_bounds_on_one_field(settings: Settings) -> None:
    """The second constraint must not overwrite the first — that would silently
    discard half of what the user asked for."""
    attributes = [AttributeInfo("year", "int", "publication year")]
    llm = StubLLM(
        responses={
            "MetadataFilterOutput": MetadataFilterOutput(
                query="diffusion",
                conditions=[
                    FilterCondition(field="year", op="gte", value=2022),
                    FilterCondition(field="year", op="lte", value=2024),
                ],
            )
        }
    )
    result, _ = await SelfQueryConstructor(llm, attributes, settings=settings).construct(
        Query(text="papers between 2022 and 2024")
    )
    assert result.filters == {"year": {"gte": 2022, "lte": 2024}}


async def test_self_query_rejects_ordering_on_a_string(settings: Settings) -> None:
    attributes = [AttributeInfo("status", "string", "order status")]
    llm = StubLLM(
        responses={
            "MetadataFilterOutput": MetadataFilterOutput(
                query="orders",
                conditions=[FilterCondition(field="status", op="gt", value="pending")],
            )
        }
    )
    result, _ = await SelfQueryConstructor(llm, attributes, settings=settings).construct(
        Query(text="orders after pending")
    )
    assert result.filters == {}
    assert result.dropped


async def test_self_query_is_a_noop_without_a_schema(settings: Settings) -> None:
    llm = StubLLM()
    result, usage = await SelfQueryConstructor(llm, [], settings=settings).construct(
        Query(text="anything")
    )
    assert result.filters == {}
    assert usage.calls == 0, "with no schema there is nothing to ask about"


# ---------------------------------------------------------------------------
# HyDE has to reach the search vector, not just the Query object
# ---------------------------------------------------------------------------
async def test_the_hypothetical_document_decides_what_is_searched() -> None:
    """The call site. `hyde_search_vector` was written, documented as "called by
    the retriever instead of the plain query embedding", covered by the two tests
    above — and called by nothing. The translator billed an LLM call for a
    document, left `Query.dense` as None, and the store embedded `query.text`.

    So the module's opening line, "HyDE: embed a hypothetical answer instead of
    the question", described the opposite of what happened.
    """
    import numpy as np

    from ragorc.core.models import Query
    from ragorc.core.settings import Settings
    from ragorc.retrieve.vector import VectorRetriever
    from ragorc.translate.hyde import HyDETranslator
    from tests.fakes import StubEmbedder, StubLLM

    settings = Settings(llm={"api_key": "k"}, embedding={"dense_dimension": 32})
    embedder = StubEmbedder(dimension=32)
    llm = StubLLM(text="Rotating the signing key requires keyctl rotate with the admin role.")

    translated, _usage = await HyDETranslator(llm, settings=settings).translate(
        Query(text="how do I rotate the signing key?")
    )
    assert translated.hypothetical, "the translator produced no document to embed"

    retriever = VectorRetriever(store=None, embedder=embedder, settings=settings)
    searched = (await retriever.embed_texts(translated, [translated.text]))[0]

    def cosine(a: object, b: object) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    question = await embedder.embed_query(translated.text)
    document = (await embedder.embed_documents([translated.hypothetical]))[0]

    assert cosine(searched, document) > cosine(searched, question), (
        "the question still dominates the search vector, so HyDE is not being used"
    )
    assert cosine(searched, question) < 0.99, "the search vector is just the question"


async def test_a_query_without_a_hypothetical_is_embedded_normally() -> None:
    """HyDE is off by default and the trigger must not fire without it — the hot
    path pays only a dict lookup."""
    import numpy as np

    from ragorc.core.models import Query
    from ragorc.core.settings import Settings
    from ragorc.retrieve.vector import VectorRetriever
    from tests.fakes import StubEmbedder

    settings = Settings(llm={"api_key": "k"}, embedding={"dense_dimension": 32})
    # Unnormalized on purpose. `hyde_search_vector`'s no-document fallback returns
    # `_l2(embed_query(text))`, which is indistinguishable from the plain path when
    # the embedder already returns unit vectors — so with a normalizing stub, a
    # trigger that fires on *every* query looks identical to one that fires on
    # none. Caught by mutation.
    embedder = StubEmbedder(dimension=32, normalize=False)
    query = Query(text="how do I rotate the signing key?")

    retriever = VectorRetriever(store=None, embedder=embedder, settings=settings)
    searched = (await retriever.embed_texts(query, [query.text]))[0]

    expected = await embedder.embed_query(query.text)
    assert np.allclose(searched, expected)
    assert not np.isclose(float(np.linalg.norm(searched)), 1.0), (
        "the vector was normalized, so it went through the HyDE path"
    )


async def test_a_caller_supplied_vector_still_wins() -> None:
    """`query.dense` is authoritative when already set — a caller who computed
    their own search vector must not have it recomputed from a hypothetical."""
    import numpy as np

    from ragorc.core.models import Query
    from ragorc.core.settings import Settings
    from ragorc.retrieve.vector import VectorRetriever
    from tests.fakes import StubEmbedder

    settings = Settings(llm={"api_key": "k"}, embedding={"dense_dimension": 32})
    embedder = StubEmbedder(dimension=32)
    mine = np.ones(32, dtype=np.float32) / np.sqrt(32)
    query = Query(text="q", hypothetical="a hypothetical answer document")
    query.dense = mine

    retriever = VectorRetriever(store=None, embedder=embedder, settings=settings)
    searched = (await retriever.embed_texts(query, [query.text]))[0]

    assert np.allclose(searched, mine)
