# `ragorc.llm` — model transport, prompts and the cost cascade

One HTTP client against OpenRouter, one prompt library, and one policy object that
decides which model each stage gets. The client is ~400 lines of `httpx` rather
than a wrapper around a wrapper, because the four things it has to do —
provider routing, provider-reported per-call cost, a client-side token bucket, and
a structured-output repair loop — are exactly the four things a generic chat
abstraction hides.

Related: [ADR-0005 — model cascade](../adr/0005-model-cascade.md) ·
[ADR-0001 — LangGraph only](../adr/0001-langgraph-for-orchestration.md).

## Key classes

```python
OpenRouterLLM(settings: LLMSettings | None = None, *, cache=None, client: httpx.AsyncClient | None = None)

async complete(prompt, *, system=None, model=None, temperature=None,
               max_tokens=None, stop=None, **kw) -> tuple[str, Usage]
async structured(prompt, schema: type[BaseModel], *, system=None, model=None,
                 temperature=None, **kw) -> tuple[Any, Usage]
async stream(prompt, *, system=None, model=None, **kw) -> AsyncIterator[str]
async batch(prompts, *, system=None, model=None, **kw) -> list[tuple[str, Usage]]
async batch_structured(...)                      # the map-stage workhorse
async fetch_model_prices() -> dict[str, dict[str, float]]
```

Every call returns a `Usage`, so the bill reaches the call site and cost accounting
is impossible to forget. `structured` is the one most stages use: the routers,
graders, self-query constructor and entity extractor are schema-constrained calls,
not free text to be regex-parsed.

```python
ModelRouter(settings=None, overrides: dict[Task, str] | None = None,
            tiers: dict[Task, ModelTier] | None = None, prices=None)
.model_for(task: Task, *, escalate: bool = False) -> str
.model_for_tier(tier: ModelTier) -> str
.estimate_cost(model, prompt_tokens, completion_tokens) -> float
.should_escalate(confidence, *, threshold=None) -> bool
```

`Task` enumerates **every** LLM-using stage. Adding a stage means adding a member,
which forces an explicit decision about what it should cost. Three tiers: `FAST`
for the 10-40 classification calls, `BALANCED` for synthesis and summaries,
`STRONG` for escalation only.

```python
Prompt(name, template, system)     # .render(**kwargs) -> str
get_prompt(name) -> Prompt
register_prompt(prompt) -> Prompt
LLMCache(backend: Cache, settings: CacheSettings | None = None)
```

Prompts are never inlined at a call site — a prompt used by two stages that drift
apart is a bug nobody can see in a diff.

## Usage

```python
from ragorc.llm import ModelRouter, OpenRouterLLM, Task, get_prompt
from ragorc.core.schemas import RelevanceGrade

llm = OpenRouterLLM()
router = ModelRouter()

prompt = get_prompt("grade_relevance")
grade, usage = await llm.structured(
    prompt.render(question="what is the SEV-1 response time?", document=text),
    RelevanceGrade,
    system=prompt.system,
    model=router.model_for(Task.GRADE_RELEVANCE),  # the cheap tier
    stage="grade_relevance",  # attributes cost on the ledger
)
answer, usage = await llm.complete(rendered, model=router.model_for(Task.ANSWER))
await llm.aclose()
```

Pass `stage=` on every call. It is what makes `CostLedger.report()["by_stage"]`
an itemized bill rather than one number.

## Settings

| Setting | Effect |
|---|---|
`llm.api_key` · `llm.base_url` | any OpenAI-compatible endpoint works |
`llm.model` · `fast_model` · `strong_model` | the three cascade slots; `strong_model` is reached only by the Text-to-SQL guard repair |
`llm.temperature` | 0.0 — RAG answers should be reproducible and classifiers want argmax |
`llm.max_concurrency` | shared semaphore over all in-flight requests |
`llm.max_retries` · `retry_base_delay_s` · `retry_max_delay_s` | backoff with full jitter |
`llm.requests_per_minute` · `tokens_per_minute` | client-side bucket, strictly better than absorbing 429s |
`llm.http2` | multiplexes 16 concurrent calls over one connection |
`llm.provider_sort` | `price` routes to the cheapest provider serving the model |
`llm.require_parameters` | only providers that support `response_format`, so structured output cannot silently degrade |
`llm.data_collection` | `deny` excludes providers that train on prompts |
`llm.enable_prompt_cache` | emits provider cache hints; large static systems then cost ~10% on repeats |
`cost.cascade_enabled` · `cascade_confidence_threshold` | **inert** — `should_escalate` implements the gate and nothing calls it (ADR-0005) |
`cost.refresh_prices` | pull live prices from `/models` so cost stays correct |
`cache.cache_llm` | route completions through `LLMCache` |
