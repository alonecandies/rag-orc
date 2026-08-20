# ADR-0007: Three cache tiers, including a semantic one

**Status:** accepted · **Date:** 2026-08-19

## Context

The same work recurs constantly in RAG, at several granularities: the same
document is graded for the same query, the same database schema summary is
embedded in every Text-to-SQL prompt, the same chunk is re-embedded on every
re-ingest, and — most valuably — users ask the same question in different words.

## Decision

Three tiers, checked cheapest-first with read-through promotion:

| Tier | Latency | Scope | Keyed by |
|---|---|---|---|
In-process TTL-LRU | ~200 ns | one worker | exact content hash |
Redis (optional) | ~200 µs | whole fleet | exact content hash |
Semantic (Qdrant) | ~2 ms | whole fleet | **embedding proximity** |

What gets cached: LLM completions and structured outputs, embeddings, rerank
scores, retrieval results, and DB schema introspection — each individually
switchable.

Three rules that keep it correct:

1. **Never cache a sample.** `temperature > 0` bypasses the cache entirely;
   memoizing a sampled generation defeats the reason for sampling.
2. **The semantic threshold is strict (0.97), and strict is not the same as
   safe.** Below ~0.95 you begin serving answers to questions that were merely
   similar. Above it you can still serve the *opposite* answer: on the shipped
   `BAAI/bge-small-en-v1.5`, "...over $500?" against "...under $500?" scores
   0.9924, while an honest paraphrase of the same question scores 0.9596. The
   similarity ordering the model provides is not the ordering this cache needs,
   so the number belongs to the embedder — read the docstring on
   `cache.semantic_threshold` before moving it.
3. **Content-hash keys, not identity keys.** Because chunk ids are derived from
   content (see `core/ids.py`), an unchanged document's chunks hash identically
   and a full re-ingest becomes a no-op instead of a full re-embed.

Redis failures degrade to a miss and are logged. A cache outage must never become
a service outage.

## Consequences

- The in-memory tier alone divides the effective hit rate by the worker count,
  so `cache.redis_url` should be set as soon as more than one process serves
  traffic.
- Semantic hit rates cannot be quoted in the abstract, and this ADR should not
  have tried: the 20-40% / "largest cost lever" figure it carried was measured on
  nobody's traffic, and against the shipped embedder (2026-08-20) the paraphrase
  premise in *Context* does not hold — paraphrases land at 0.93-0.97, tangled up
  with questions whose answers differ. What the tier does catch is the same
  question re-asked with different case, punctuation or whitespace (0.995-1.000),
  which is common in a chat UI and worth having. Measure it on your own traffic
  before budgeting for it; [cost.md](../cost.md) carries the numbers.
- Cache statistics are first-class: `CostLedger.report()` includes
  `cache_hit_rate`, because a cost report without it is unreadable.
