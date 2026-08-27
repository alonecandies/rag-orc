# `ragorc.cache` — memory → Redis → semantic

Three tiers with wildly different latencies and hit semantics, checked
cheapest-first with read-through promotion. Caching is not a micro-optimization
here: a query makes 10-40 model calls, and the cache tiers are the difference
between a viable and an unviable unit cost.

```
memory (~200ns)  →  redis (~200µs)  →  semantic (~2ms)  →  run the pipeline
```

Related: [ADR-0007 — cache tiers](../adr/0007-cache-tiers.md) ·
[docs/cost.md](../cost.md).

## Key classes

```python
MemoryCache(max_items=20_000, ttl_s=900.0)     # TTLCache from cachetools, per process
RedisCache(url, *, prefix="ragorc", ttl_s=86_400.0)          # ragorc[redis]
TieredCache(tiers: list[Cache])                # get promotes upward, set writes to all
NullCache()                                     # for benchmarks that must not be memoized
build_cache(settings: CacheSettings | None = None) -> Cache

SemanticCache(embedder, client=None, settings=None)
    async get(question, *, tenant_id=None) -> SemanticHit | None
    async set(question, answer: dict, *, tenant_id=None) -> None
    async clear() -> None
    def stats() -> dict
SemanticHit(answer, question, score, stored_at)
```

`Cache` is bytes-in/bytes-out (`ragorc.core.protocols.Cache`). Callers serialize with
`orjson`, so no tier ever needs to know about our types. Two adapters sit on top of
it and own their key namespaces: `ragorc.llm.cache.LLMCache` for completions and
`ragorc.embed.cache.EmbeddingCache` for dense, sparse and multivector frames.

## The two tiers worth understanding

**Redis is not optional past one worker.** With four API workers and no shared tier,
each one warms its own memory cache and the effective hit rate divides by four.
`build_cache` adds the Redis tier only when `cache.redis_url` is set, and logs a
warning naming the extra if the import fails rather than silently running one tier.

**The semantic tier is the largest cost lever and the most dangerous setting.** It
embeds the question and looks for a *near* neighbour among previously answered ones,
so a hit skips the entire pipeline — an exact hit saves one call, a semantic hit
saves twenty. The danger is symmetrical: below about 0.95 similarity you start
returning confident answers to questions nobody asked, which is worse than any
miss. Default 0.97, and be conservative moving it.

One behaviour worth knowing: `SemanticCache.set` refuses to store an abstention. An
abstention is a statement about what the index held at one moment, and replaying it
later hides content that has since been ingested.

## Usage

```python
from ragorc.cache import build_cache
from ragorc.cache.semantic import SemanticCache
from ragorc.embed.cache import EmbeddingCache
from ragorc.llm.cache import LLMCache
from ragorc.llm import OpenRouterLLM

backend = build_cache()  # memory (+ redis if configured)
llm = OpenRouterLLM(cache=LLMCache(backend))
embeddings = EmbeddingCache(backend)

answers = SemanticCache(dense_embedder)
hit = await answers.get(question, tenant_id="acme")
if hit is not None:
    return hit.answer  # score >= semantic_threshold
answer = await pipeline.query(question)
await answers.set(question, {"text": answer.text}, tenant_id="acme")
```

Note the `tenant_id` on both calls. A semantic cache without tenant scoping is a
cross-tenant data leak with a latency improvement.

## Settings

| Setting | Effect |
|---|---|
`cache.enabled` | `False` makes `build_cache` return `NullCache` |
`cache.memory_max_items` · `memory_ttl_s` | the in-process tier |
`cache.redis_url` · `redis_ttl_s` · `redis_prefix` | empty URL disables the shared tier |
`cache.semantic_enabled` · `semantic_threshold` | 0.97; treat downward moves as a correctness change |
`cache.semantic_collection` · `semantic_ttl_s` | a small Qdrant collection, TTL by payload filter |
`cache.cache_llm` | completions and structured output |
`cache.cache_embeddings` | content-hash keyed vectors |
`cache.cache_rerank` | cross-encoder scores, keyed on (query, document) |
`cache.cache_schema` | Postgres/Neo4j introspection for the Text-to-SQL prompt |

## Why the semantic cache is keyed on more than the question

Tenant scoping was there from the start — a cache keyed only on question text
leaks one tenant's answer to another. `filters` and `top_k` are the same problem
one level down: filters decide which passages are admissible, `top_k` decides how
many are used, and a caller who filters to a subset is often doing so because that
subset is what they are entitled to see. Two requests with the same text and
different scopes are not the same request, and `scope_key()` is what keeps them
apart. It is matched exactly, unlike the question itself — a near-miss on the
question is the point of a semantic cache; a near-miss on the scope is a wrong
answer.

## Staleness, and why TTL is not the whole answer

An answer is a statement about the index at the moment it was computed. The
semantic cache already refused to store an abstention for that reason — serving
one later hides content that has since been added — and the same sentence with
the sign flipped is why a positive answer needs invalidating: serving it later
shows content that has since been corrected or removed.

Reproduced against a live stack: a policy edited from "30 days" to "7 days" and
re-ingested was still answered "30 days", from a cache hit that carried no
citations, so nothing in the response pointed at the document that no longer
said it.

Each entry now records the documents its answer was built from, and
`IngestPipeline._purge` calls `SemanticCache.invalidate` for every document a run
replaces. Three things follow from how that is built:

* **Provenance is the chunks the generator saw, not the citations.** An uncited
  passage still went into the prompt, so an edit to it can still change the
  answer.
* **Only the purge set.** A newly-added document cannot appear in an answer
  computed before it existed, so widening this to every ingested document would
  evict the whole cache on a first load.
* **Entries written before this field existed are not matched.** A filtered
  delete cannot match an absent field; they expire by TTL as they always did.

Invalidation never fails an ingest. Not invalidating costs a stale answer for up
to one TTL; raising costs the operator the fix they were in the middle of
applying.
