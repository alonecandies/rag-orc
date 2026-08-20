# ADR-0002: Late chunking is the preferred strategy, and degrades honestly

**Status:** accepted · **Date:** 2026-08-19

## Context

Chunking decides the ceiling on retrieval quality: nothing downstream can
recover information destroyed at split time. There are three strategies, and the
choice is usually made by accident.

**Early chunking** (what almost everyone does) splits the document, then embeds
each chunk independently. The flaw is structural: a chunk reading

> Its revenue grew 40% in the same period, driven by the enterprise segment.

has no idea what *its* refers to. The document knew. The chunk does not, and
therefore neither does its vector. A query for "Acme revenue growth" will not
retrieve it.

**Late chunking** (Günther et al., 2024) inverts the order: embed the *whole*
document in one forward pass, keep the per-token embeddings, then mean-pool over
each chunk's token span. Every chunk vector is conditioned on the full document,
so the pooled vector for that sentence carries "Acme" even though the text does
not.

**Contextual retrieval** (Anthropic, 2024) prepends an LLM-written sentence or
two situating each chunk before embedding it. It solves the same problem at the
prompt level rather than the pooling level.

## Decision

**Prefer late chunking**, and resolve `ChunkingStrategy.AUTO` down a ladder:

```
LATE  ──if the dense model cannot expose token embeddings──▶  CONTEXTUAL
                                                                │
                        ──if contextual is disabled──▶        EARLY (+ parent expansion)
```

### What you actually get, and why

| Configuration | `AUTO` resolves to |
|---|---|
`pip install ragorc` (FastEmbed default) | **`early`** |
`ragorc[local]` + a token-capable model (`jinaai/jina-embeddings-v2-base-en`) | **`late`** |
`indexing.contextual_enabled = true` | `contextual` |
A hosted provider (OpenAI / Voyage / Cohere) | `early`, or `contextual` if enabled |

The base install lands on `early`, and that is a consequence of the technique
rather than an unfinished edge:

**Late chunking needs one model emitting both the token vectors and the pooled
document vector.** The pooled result is only usable as a chunk vector if it lands
in the same space as the *query* vector, and the query is embedded by the dense
model. FastEmbed's `TextEmbedding` returns pooled output only, so with the
zero-dependency default there is no token source for BGE, and late chunking is
genuinely unavailable rather than merely unconfigured.

Worth recording because an earlier implementation of this ADR got it wrong in an
instructive way: it substituted the ColBERT late-interaction model as the token
source, reasoning that ColBERT already emits one vector per token. True, and the
wrong conclusion — ColBERT is a different model in a different space at a
different width (128 vs 384), so the pooled vectors were not comparable to any
query vector. Qdrant rejected them on dimension, which was *luck*: at equal width
it would have accepted them and returned quietly meaningless neighbours for the
life of the index.

So the ladder degrades rather than substituting, and logs which rung it landed
on. `supports_late_chunking()` and `resolve_strategy()` are required to agree —
there is a test for it — because advertising a capability the resolver then
declines is how a corpus gets indexed one way while the operator believes it was
indexed another.

Rationale, in order of weight:

1. **Late chunking is cheaper, not more expensive.** One forward pass per
   document beats one per chunk. A 40-chunk document costs 1 pass instead of 40.
   This is the unusual case where the better technique also costs less.
2. **It composes with semantic chunking.** Boundaries are still chosen
   semantically; late chunking only changes how the vector for each span is
   computed. We get both.
3. **It requires no LLM calls**, unlike contextual retrieval — which matters
   because contextual retrieval costs one model call per chunk at ingest.

Why the ladder exists: hosted embedding APIs (OpenAI, Voyage, Cohere) return
only pooled vectors, so late chunking is *impossible* against them. Rather than
silently producing worse vectors, `resolve_strategy()` detects the capability and
degrades explicitly, with a log line saying which strategy was chosen and why.

`CONTEXTUAL` sits above `EARLY` in the ladder because Anthropic measured a
35%→49% reduction in retrieval failures when combined with BM25 — but it is
opt-in (`indexing.contextual_enabled`) since it costs one LLM call per chunk.

## Consequences

- The splitter API returns **spans**, not embedded chunks. Splitting and
  embedding are separate stages; a splitter that embedded as it split would make
  late chunking impossible. This is why `Splitter.split()` returns `Chunk`
  objects with `start_char`/`end_char` and no vectors.
- Documents longer than the model's context are processed in overlapping macro
  windows, and each chunk is pooled from the window that contains it most fully.
  Without the overlap, chunks near a window boundary lose exactly the context
  that late chunking exists to preserve.
- Prefer a long-context embedding model. `jinaai/jina-embeddings-v2-base-en`
  (8192 tokens) is the reference choice; the 512-token default caps how much
  document context any single pass can carry.
