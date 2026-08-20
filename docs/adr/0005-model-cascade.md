# ADR-0005: A task-tiered model cascade

**Status:** accepted · **Date:** 2026-08-19

## Context

A single query through the full pipeline makes far more model calls than people
expect. Counted from the graph: 1 routing call, 1-5 query-translation calls, 1
query-construction call per structured store, 5-20 document relevance grades, 0-N
compression extracts, 1 answer, 1 groundedness grade, 1 utility grade, plus
whatever the CRAG and Self-RAG loops add on retry.

That is 10-40 calls, of which **exactly one** produces text a human reads.

Routing everything to a frontier model is the standard reason a RAG prototype
costs 30x what it should. The classifiers are binary or short-list decisions
where a small model matches a large one.

## Decision

Three named tiers, and every stage declares which it needs via the `Task` enum:

| Tier | Setting | Used by |
|---|---|---|
`fast` | `llm.fast_model` | routing, all grading, rewriting, decomposition, self-query, compression, claim verification, sufficiency checks, HyDE |
`balanced` | `llm.model` | the answer, summarization, proposition extraction, graph extraction, community reports, RankGPT |
`strong` | `llm.strong_model` | escalation only, when a cheap answer's confidence is below `cost.cascade_confidence_threshold` |

Summarization sits in `balanced` despite being high-volume, because a summary
*becomes the retrieval target* — a bad summary is permanently bad, in a way a bad
one-off grade is not. Graph extraction is `balanced` for the same reason:
extraction quality determines whether traversal finds anything at all.

Supporting mechanisms:

- `provider.sort = "price"` so OpenRouter picks the cheapest provider serving the
  chosen model.
- `usage: {include: true}` so the **provider's actual charge** is recorded per
  call rather than estimated from a price table that will drift.
- A per-request `CostLedger` with hard ceilings on cost, calls and tokens,
  checked *before* each call. Loops and retries have no natural upper bound on
  spend; this gives them one.
- Prompt-cache hints on long static system prompts.

## Consequences

- Cost is dominated by the one synthesis call plus retrieval, which is the
  correct shape.
- Every stage's spend is attributable: `CostLedger.report()` breaks down by model
  *and* by stage, so an expensive query can be diagnosed instead of guessed at.
- Adding a stage means adding a `Task` member, which forces an explicit decision
  about what that stage is allowed to cost. That friction is intentional.
