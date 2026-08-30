# Cost

The reason RAG prototypes cost 30x what they should, and the five mechanisms that
fix it.

## Count the calls

A single query through the full pipeline:

| Stage | Calls | Reads it? |
|---|---|---|
Routing | 1 | no |
Query translation (multi-query / step-back / decomposition) | 1-5 | no |
Query construction (SQL / Cypher / self-query) | 0-3 | no |
CRAG document grading | 5-20 | no |
Compression extracts | 0-20 | no |
Claim decomposition + verification | 0-15 | no |
**Answer synthesis** | **1** | **yes** |
Groundedness + utility grading | 2 | no |
CRAG / Self-RAG retries | ×1-3 on the above | no |

**10-40 calls, of which one produces text a human reads.** Sending all of them to
a frontier model is the single most common cost mistake in RAG.

## 1. The model cascade ([ADR-0005](adr/0005-model-cascade.md))

Three tiers; every stage declares its `Task` and gets the cheapest model that can
do the job.

```bash
RAGORC_LLM__FAST_MODEL=google/gemini-2.5-flash-lite      # ~14 of the stages above
RAGORC_LLM__MODEL=anthropic/claude-sonnet-4.5            # synthesis, summaries, extraction
RAGORC_LLM__STRONG_MODEL=anthropic/claude-opus-4.5       # escalation only
```

Illustrative arithmetic for one query with 20 classification calls at ~800 prompt
tokens each, plus one 6k-token synthesis:

```
all-frontier:   21 calls × frontier pricing
cascade:        20 calls × cheap pricing + 1 × frontier
```

The classification calls become a rounding error instead of the bill. Exact
savings depend on the models you pick; the *shape* — one expensive call, many
cheap ones — is what matters.

Two stages are deliberately on the middle tier despite being high volume:
**summarization** and **graph extraction**. Their output *becomes the index*, so a
bad result is permanently bad in a way a one-off grade is not.

Override per task when you disagree:

```python
from ragorc.llm.router import ModelRouter, Task

router = ModelRouter(overrides={Task.GRADE_RELEVANCE: "meta-llama/llama-3.3-70b-instruct"})
```

## 2. Caching ([ADR-0007](adr/0007-cache-tiers.md))

| Tier | Hits when | Typical rate |
|---|---|---|
memory / redis (exact) | the identical prompt recurs — constantly in RAG | 15-30% |
**semantic** | a question already answered comes back | measure it — see below |
embedding cache | re-ingesting unchanged content | ~100% on re-ingest |

One semantic hit is the largest single saving available, because it skips the
*entire* pipeline rather than one call. How often it hits is a different question,
and it depends on the embedding model at least as much as on the threshold.
Measured on this repo's corpus questions with the shipped default embedder
(`BAAI/bge-small-en-v1.5`, cosine), which is what decides the numbers:

```
same question, different case / punctuation      0.995-1.000   hits at 0.97
paraphrase ("what is X" / "explain X")           0.93-0.97     does not hit
different question, same topic                   0.57-0.94     must not hit
"...over $500?" vs "...under $500?"              0.9924        hits, and is wrong
```

So with the shipped model this tier is a case- and punctuation-insensitive cache
over repeated questions rather than the paraphrase cache it was once documented
as, and it can still hit on a one-word inversion. Budget it from your own traffic,
and read the docstring on `cache.semantic_threshold` before moving the number.

```bash
RAGORC_CACHE__SEMANTIC_ENABLED=true
RAGORC_CACHE__SEMANTIC_THRESHOLD=0.97   # 0.995 excludes every wrong pair above
RAGORC_CACHE__REDIS_URL=redis://localhost:6379/0   # required with >1 worker
```

Without Redis, each worker warms its own cache and the effective hit rate divides
by the worker count.

## 3. Provider routing

```bash
RAGORC_LLM__PROVIDER_SORT=price           # cheapest provider serving the model
RAGORC_LLM__ALLOW_FALLBACKS=true
RAGORC_LLM__REQUIRE_PARAMETERS=true       # only providers supporting response_format
RAGORC_LLM__ENABLE_PROMPT_CACHE=true      # long static system prompts at ~10%
```

`require_parameters` matters for cost indirectly: without it, a provider that
ignores `response_format` produces unparseable output, and the repair round trip
costs more than the routing saved.

## 4. Hard ceilings, checked before the spend

```bash
RAGORC_COST__MAX_COST_PER_QUERY_USD=0.50
RAGORC_COST__MAX_LLM_CALLS_PER_QUERY=40
RAGORC_COST__MAX_TOKENS_PER_QUERY=200000
```

`CostLedger.check()` runs *before* every call, so an over-budget request raises
`BudgetExceeded` instead of being discovered afterwards. This matters because CRAG
and Self-RAG loops have no natural upper bound: a pathological query that keeps
failing its groundedness check will keep retrying.

Cost is read from the provider's own report (`usage: {include: true}`), not
estimated from a local price table that will drift.

## 5. Avoid the call entirely

| Mechanism | Saves |
|---|---|
Rule fast-path in the hybrid router | the routing call on obvious queries |
Semantic router (embeddings, not LLM) | the prompt-selection call, always |
Entity matching via graph fulltext index | an extraction call per query |
`multihop_stop_on_sufficient` | 1-2 full retrieve+reason rounds |
`indexing.skip_unchanged` | the entire embedding cost of a re-ingest |
`EmbeddingFilterCompressor` as default | one LLM call per chunk |

## Ingest cost

Embeddings are local and free by default (FastEmbed). What costs money at ingest
is optional and per-chunk:

| Feature | Cost | Worth it when |
|---|---|---|
`contextual_enabled` | 1 cheap call/chunk (prompt-cache friendly) | references and pronouns span chunks |
`summary_index_enabled` | 1 call/chunk | long, meandering source text |
`dense_x_enabled` | 1 call/chunk | precision matters more than anything |
`raptor_enabled` | ~1 call per cluster per level | broad, thematic questions |
`graph.enabled` | 1-2 calls/chunk + 1/community | relationship and multi-hop questions |

Note that **late chunking is cheaper than early chunking** — one forward pass per
document instead of one per chunk. It is *preferred*, not the default: FastEmbed
returns pooled vectors only, so with the zero-dependency install `auto` resolves
to `early` and the saving is not being taken. Install `ragorc[local]` with a
token-capable model to collect it ([ADR-0002](adr/0002-late-chunking.md)).

## Reading the bill

The bill comes back **on the answer**, not from a ledger you install yourself:

```python
from ragorc import build_pipeline

rag = await build_pipeline()
answer = await rag.query("why is late chunking cheaper than early chunking?")

print(answer.usage.cost_usd, answer.usage.calls)  # the total
print(answer.metadata["cost"])  # itemized, by model and by stage
# {'total_cost_usd': 0.000102, 'calls': 2, 'cached_calls': 0, 'cache_hit_rate': 0.0,
#  'total_tokens': ..., 'prompt_tokens': ..., 'completion_tokens': ...,
#  'by_model': {'google/gemini-2.5-flash-lite': {...}},
#  'by_stage': {'route': {...}, 'answer': {...}}}
```

`answer.metadata["cost"]` is `CostLedger.report()` for that one request — by stage
*and* by model, with `cache_hit_rate` in the summary rather than buried, because a
cost report without it is unreadable.

Wrapping the call in `new_request_context(...)` and reading *that* ledger does not
work, and this doc used to show exactly that. `query()` installs its own request
context — which is what enforces the ceilings above — and the inner context
replaces the contextvars for the duration of the call, so the outer ledger stays at
`{'total_cost_usd': 0.0, 'calls': 0}` however many calls the query made. For the
same reason a `max_cost_usd=` passed to a wrapping context is ignored: the ceiling
comes from `cost.*`, so tighten it there
(`build_pipeline(cost__max_cost_per_query_usd=0.05)`).

One caveat on the two fields. `answer.usage` is always populated; `metadata["cost"]`
comes from the ledger the LLM client writes to, so an injected third-party client
that satisfies the protocol without writing to the ledger leaves it at zero while
`answer.usage` — summed from what the nodes reported — still holds the total.


## An ingest is not a query

`cost.max_llm_calls_per_query` bounds one request. An ingest is a corpus, and the
HTTP ingest route used to run inside the per-query ledger — so a 60-document
corpus with RAPTOR on stopped enriching after 40 documents and still reported
success:

```
documents indexed: 60
documents that got a RAPTOR summary: 40 of 60
warnings: ['raptor stage disabled: LLM call budget exhausted (calls=40 limit=40)']
```

Ingest has its own ceilings — `cost.max_llm_calls_per_ingest`,
`max_cost_per_ingest_usd`, `max_tokens_per_ingest`. Only the **cost** one has a
default (`10.0`), because spend is the quantity that does not scale with the
corpus: twenty times the per-query ceiling, and still an unmistakable runaway.
Call and token counts *do* scale, and an arbitrary default for them would only
move the same silent truncation to a different corpus size.

What makes leaving those two open safe is that exhaustion is now loud. Every
stage propagates `BudgetExceeded`: the multi-representation indexers re-raise it
out of their `build()` gather rather than letting `return_exceptions=True`
reclassify it as one chunk's bad luck, and `IngestPipeline._enrich` lets it
through instead of appending "stage disabled" to a report field most callers
never read.

Two things that do **not** bound an ingest, both of which were once believed to:

* `RaptorIndexer._check_budget` returns early when the ledger has no call
  ceiling, so its pre-flight forecast refuses nothing unless you set
  `max_llm_calls_per_ingest`. Set one when ingest is caller-triggered.
* With `track_costs=false` the cost ceiling is lifted too, exactly as on the
  query path.

## Ceilings are reserved, not merely checked

`cost.max_llm_calls_per_query`, `max_cost_per_query_usd` and
`max_tokens_per_query` are enforced by `CostLedger.reserve()`, which claims the
budget *before* a provider request is issued and releases it when the call
finishes however it finishes.

A plain pre-flight check is not enough under a fan-out. A call is recorded only
after its round trip, so every coroutine in a `gather` reads a ledger that still
says zero and every one of them passes. Measured with 40 prompts against a
five-call ceiling: 40 requests served, the ceiling read forty times and enforced
none. The overshoot was the full fan-out width rather than `llm.max_concurrency`,
because the check preceded the semaphore.

`max_tokens` is reserved exactly, being the most a call can spend and known
before it is made. Cost is reserved at zero, so the cost ceiling is bounded
transitively by the call ceiling — which is why clearing
`max_llm_calls_per_query` while keeping a cost ceiling reopens the window.
