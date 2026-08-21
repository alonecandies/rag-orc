# `ragorc.core` — contracts

Everything above this package depends on it, and it depends on nothing. Six
concerns: the data model, the interfaces, configuration, errors, concurrency and
telemetry. If a change here is needed to add a feature, the feature is probably
in the wrong layer.

Related: [ADR-0008 — dataclasses in the hot path](../adr/0008-dataclasses-in-hot-path.md).

## Data model — `core/models.py`

`@dataclass(slots=True)`, because an ingest run creates millions of `Chunk`
objects: ~2-3x faster to construct than a pydantic model and ~40% less memory.
Vectors are `numpy` arrays, never `list[float]` (4 KiB vs ~40 KiB at 1024 dims).

| Type | Notes |
|---|---|
`Document(id, content, metadata, source, title, modality, checksum, tenant_id)` | `checksum` is what makes ingest idempotent |
`Chunk(id, content, document_id, index, start_char, end_char, ...)` | carries `dense`, `sparse` **and** `multi` simultaneously |
`ScoredChunk(chunk, score, source, rank, component_scores, explain)` | `component_scores` is why "why did this rank third?" is answerable |
`Query(text, original, variants, hypothetical, filters, top_k, dense, sparse, multi, tenant_id)` | enriched in flight; `original` survives every rewrite |
`RetrievalResult(chunks, per_store, timings_ms, errors, grade, total_candidates)` | a failed store lands in `errors`, it does not raise |
`Answer(text, citations, chunks, usage, grounded, groundedness, confidence, abstained, trace, route)` | the terminal object |
`Usage`, `StepTrace`, `Citation`, `Entity`, `Relation`, `Community`, `GraphPath`, `SparseVector` | |

Enums: `DataStore`, `RetrievalSource`, `ChunkingStrategy`, `FusionMethod`,
`GradeLabel`, `Modality`. Scores are **always higher-is-better**; a backend that
returns a distance converts, and says so in a comment.

## Protocols — `core/protocols.py`

`typing.Protocol`, not ABCs: a component satisfies an interface by *shape*, so a
third-party store or retriever drops in with no inheritance and no adapter.

```python
class Retriever(Protocol):
    name: str

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]: ...
```

`LLM`, `DenseEmbedder`, `SparseEmbedder`, `LateInteractionEmbedder`, `Reranker`,
`VectorStore`, `RelationalStore`, `GraphStore`, `Cache`, `Loader`, `Splitter`,
`BatchStructuredLLM · QueryTranslator`, `Router`, `QueryConstructor`, `Retriever`, `Compressor`,
`Grader`, `ContextPacker`, `Generator`.

## Settings, registry, ids, tokens

```python
from ragorc.core import get_settings, register, resolve, chunk_id, count_tokens

settings = get_settings()  # process-wide, lru_cached
cls = resolve("retriever", "hybrid")  # populated by @register decorators
cid = chunk_id(doc_id, index, content)  # content-derived, never random
```

Content-derived ids buy three things at once: idempotent ingest, cross-store
joins with no mapping table, and a cache key that *is* the chunk id.

## Errors — `core/errors.py`

Each type encodes a decision, not just a message: `TransientError` is retried,
`StoreUnavailable` trips a breaker and degrades the query, `GuardrailViolation` is
never retried (a blocked statement is still blocked), `BudgetExceeded` aborts.

## Concurrency — `core/concurrency.py`

```python
from ragorc.core import bounded_gather, safe_gather, gather_dict, CircuitBreaker, retry_async
```

Fan-out is always bounded — an unbounded `asyncio.gather` over 50k chunks
exhausts memory and gets rate-limited before it produces an answer. `safe_gather`
returns successes and failures separately, which is the shape the multi-store
retriever needs. `retry_async` uses full jitter, because a fleet retrying in
lockstep reproduces the burst that caused the 429.

## Telemetry — `core/telemetry.py`

```python
from ragorc.core.telemetry import new_request_context, timed, trace_step

with new_request_context(request_id="req-1", max_cost_usd=0.5, max_calls=40) as (trace, ledger):
    with timed("retrieve"):
        ...
    print(ledger.report())
```

Trace and ledger live in `contextvars`, so concurrent requests in one event loop
keep separate budgets without threading either through every signature.
`CostLedger.check()` runs **before** each model call: a pipeline with loops and
retries has no natural spending bound otherwise.

## Settings that live here

| Setting | Effect |
|---|---|
`environment` | `prod` force-enables the SQL and Cypher guards and disables prompt logging |
`tenant_id` | default tenant applied to ingest and query |
`cost.max_cost_per_query_usd` · `cost.max_llm_calls_per_query` · `cost.max_tokens_per_query` | the ledger's hard ceilings |
`observability.log_level` · `log_json` · `log_prompts` · `slow_query_ms` | `log_prompts` is off by default: prompts contain retrieved customer data |
