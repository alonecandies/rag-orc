# `ragorc.route` — which datastore, which prompt

Two orthogonal decisions, both called "routing" in the literature:

- **Logical routing** — *which datastore* can answer this? "How much ARR does
  Contoso represent?" is a Postgres question; "who is on call for the Graph
  Service?" is a graph question; "why is late chunking cheaper?" is a vector
  question. Wrong store, no answer, however good the retriever.
- **Semantic routing** — *which prompt* should answer it? A pricing question and an
  incident-runbook question want different system prompts, and this decision needs
  no model call at all.

## Key classes

```python
LogicalRouter(llm, settings=None, *, router=None,
              schema_hint: Callable[[], str | Awaitable[str]] | None = None,
              allowed: tuple[DataStore, ...] | None = None)
    name = "logical"
    async route(query: Query) -> tuple[RouteDecision, Usage]

SemanticRouter(embedder, routes: dict[str, list[str]] | None = None, settings=None,
               *, min_similarity=0.30, default_route="answer_default")
    name = "semantic"
    async route(query) -> tuple[RouteDecision, Usage]     # Usage is empty: no LLM call

HybridRouter(logical=None, semantic=None, settings=None, *, use_rules=True)
    name = "hybrid"
    async route(query) -> tuple[RouteDecision, Usage]

rule_route(question) -> RouteDecision | None  # free keyword fast path
build_router(name, *, llm=None, embedder=None, settings=None, **kw)
```

`RouteDecision(stores, prompt_name, confidence, reasoning, method)` — `stores` is a
tuple of `DataStore` members, and `DataStore.NONE` is a legitimate outcome meaning
"this question needs no retrieval at all".

## Why hybrid is the default shape

`HybridRouter` tries three things in ascending cost:

1. `rule_route` — a keyword fast path. Free, and it resolves a surprising share of
   real traffic ("SQL", "how many", an obvious entity name). A routing decision that
   costs nothing is worth taking even at modest accuracy, because the fallback is
   immediately behind it.
2. `SemanticRouter` — one embedding, no LLM call, picks the prompt.
3. `LogicalRouter` — one `fast_model` call, picks the stores.

`build_router("hybrid", llm=..., embedder=...)` assembles whichever legs it was given:
with no embedder it still gives rule plus logical routing rather than an error.

## Usage

```python
from ragorc.route import build_router
from ragorc.retrieve.multi_store import MultiStoreRetriever

router = build_router("hybrid", llm=llm, embedder=dense)
decision, usage = await router.route(query)
print(decision.stores, decision.prompt_name, decision.confidence, decision.reasoning)

fanout = MultiStoreRetriever(vector=hybrid, relational=sql, graph=graph_local)
result = await fanout.retrieve_detailed(query, route=decision)
print(result.per_store.keys(), result.errors)  # per-store diagnostics
```

A route that selects a store nobody configured lands in `result.errors` by name,
rather than disappearing. That is deliberate: silently answering from two stores
when the router asked for three hides a configuration error indefinitely.

Custom semantic routes are a list of (prompt name, example utterances):

```python
from ragorc.route import SemanticRouter

router = SemanticRouter(
    dense,
    routes={
        "answer_concise": ["how much does X cost", "what is the list price"],
        "answer_default": ["the service is down, who is on call", "walk me through the runbook"],
    },
)
```

The exemplars are embedded once, under a lock, on the first call: without the lock
concurrent cold-start requests each embed the whole exemplar set, which is the most
expensive thing the process does. `DEFAULT_ROUTES` covers the shipped prompts.

## Settings

| Setting | Effect |
|---|---|
`llm.fast_model` | the logical router's model (`Task.ROUTE`) |
`generation.prompt_name` | the fallback when semantic routing declines |
`retrieval.per_store_timeout_s` | a routed store that is slow is dropped, not waited for |
`retrieval.max_concurrent_retrievers` | fan-out ceiling once the route is known |
`retrieval.fusion` | RRF by default, because a SQL leg's constant score has no magnitude to compare |
`cache.cache_schema` | the logical router's schema hint is introspected once and cached |
