# `ragorc.translate` — query translation

Five ways to turn one question into better search input. Every translator returns an
enriched `Query` rather than a bare list of strings, so a downstream stage can see
*why* each variant exists — which matters because the fusion layer and the trace
both read that provenance.

None of them mutate the input query. `Query.original` survives every rewrite, so the
final answer is graded against what was actually asked.

## Key classes

```python
BaseTranslator(llm: LLM, settings=None, *, router: ModelRouter | None = None)
    name: str
    # Construction and variant bookkeeping only. `translate` is declared by the
    # `ragorc.core.protocols.QueryTranslator` protocol and implemented by each
    # concrete translator below — not inherited from here:
    #     async translate(query: Query) -> tuple[Query, Usage]

MultiQueryTranslator(llm, settings=None, *, n=3)      # n paraphrases, same information need
RAGFusionTranslator(llm, settings=None, *, n=4, rrf_k=None)   # + RRF over per-variant rankings
StepBackTranslator(llm, settings=None)                # one more general question
DecompositionTranslator(llm, settings=None, *, max_sub=4)
HyDETranslator(llm, settings=None, *, n_documents=1, blend=0.3)
    async embed_for_search(query, embedder) -> FloatArray

CompositeTranslator(translators, *, max_variants=8)
RecursiveDecomposer(translator: DecompositionTranslator, *, max_steps=4)
    async run(query, retrieve_fn, answer_fn) -> tuple[list[SubAnswer], Usage]
build_translator(name, llm, settings=None, **kw) -> BaseTranslator
build_translators(names, llm, settings=None, **kw) -> CompositeTranslator
```

Registry names: `multi_query`, `rag_fusion`, `step_back`, `decomposition`, `hyde`.

## What each one is for

| Translator | The failure it fixes |
|---|---|
`multi_query` | one phrasing lands in one neighbourhood of the embedding space; three phrasings cover more of it |
`rag_fusion` | same, plus RRF over the per-variant rankings, so a document several variants agree on outranks one that only the best variant found |
`step_back` | a question too specific to match anything ("does the Gold plan cover Saturdays?" → "what are the support plan coverage hours?") |
`decomposition` | a question whose answer is a *composition* of facts stated separately; `RecursiveDecomposer` chains the sub-answers |
`hyde` | a question whose vocabulary does not overlap its answer's — embed a hypothetical answer instead of the question |

`CompositeTranslator` chains them, and order matters: decomposition before
multi-query multiplies the variant count, so `max_variants` is a hard cap rather
than advice.

## Usage

```python
from ragorc.translate import build_translators

translator = build_translators(["step_back", "multi_query"], llm)
enriched, usage = await translator.translate(Query(text="does Gold cover Saturdays?"))

print(enriched.all_texts)  # original + variants, de-duplicated, order preserved
hits = await retriever.retrieve(enriched)  # retrievers fan out over all_texts
```

`HyDE` is the one that needs an extra step, because its output is a *document* rather
than a query:

```python
from ragorc.translate import HyDETranslator

hyde = HyDETranslator(llm)
enriched, usage = await hyde.translate(query)
enriched.dense = await hyde.embed_for_search(enriched, dense_embedder)  # blended vector
```

Blended, not replaced: a pure hypothetical-document vector drifts when the model
invents specifics, so the question's own vector stays in the mix.

## Cost

Every translator is one `fast_model` call (`Task.MULTI_QUERY`, `STEP_BACK`,
`DECOMPOSE`, `REWRITE`) except HyDE, which is `Task.HYDE` — also the fast tier,
because HyDE needs fluency rather than depth. The expensive part is downstream:
N variants means N retrievals, so `retrieval.max_concurrent_retrievers` and
`fetch_k` decide what a translator actually costs.

## Settings

| Setting | Effect |
|---|---|
`llm.fast_model` | every translator runs here (ADR-0005) |
`llm.max_concurrency` | bounds the variant fan-out |
`retrieval.fetch_k` | per-variant candidate window |
`retrieval.fusion` · `rrf_k` | how per-variant rankings are combined |
`retrieval.max_concurrent_retrievers` | ceiling on simultaneous variant searches |
`cost.max_llm_calls_per_query` | a composite translator plus decomposition can spend several calls before retrieval starts |
