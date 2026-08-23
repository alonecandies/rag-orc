# `ragorc.context` — token budget, packing, overflow compression

Three questions about the prompt, in the order they have to be answered: **how much
room is there**, **what goes in it**, and **what to do when the answer to the second
does not fit the first**.

## Key classes

```python
ContextBudgeter(settings=None)
    plan(*, system_prompt="", question="", shares=None, window=None) -> BudgetPlan
    measure(chunks) -> list[int]                  # one batched tiktoken call, cached on the chunk
    fits(chunks, plan) -> bool
    decide_strategy(chunks, plan) -> str          # "fit" | "truncate" | "summarize"
BudgetPlan(budget: TokenBudget, per_source, context_tokens, overflow, dropped_chunks, strategy)
    # per_source is computed and reported but not enforced — see OPEN-ITEMS
DEFAULT_SHARES = {"vector": 0.55, "graph": 0.20, "relational": 0.10, "summary": 0.10, "web": 0.05}

ContextPacker(settings=None)
    build(chunks, *, budget, isolate=True, expand_parents=None) -> ContextPack
    async pack(query, chunks, *, budget, isolate=True) -> tuple[list[ScoredChunk], str]
ContextPack(text, chunks, tokens, dropped, truncated, order)
reorder_lost_in_middle(chunks) -> list[ScoredChunk]

ContextSummarizer(llm, settings=None, *, router=None)
    async fit(question, chunks, *, budget, strategy="map_reduce", max_levels=3) -> SummarizationResult
SummarizationResult(chunks, usage, strategy, input_tokens, output_tokens)  # .compression_ratio
```

## Packing is a knapsack, not a top-k

Each chunk has a token cost and a relevance value; the window is the capacity.
Filling greedily by score alone wastes capacity, because a 900-token chunk scoring
0.81 displaces three 250-token chunks scoring 0.79. Selection is by **relevance per
token**; rank order is restored afterwards for presentation. The single best chunk is
admitted unconditionally, truncated if it alone exceeds the budget — returning nothing
because the top result is large is worse than returning it clipped.

## Placement changes accuracy

Transformer attention over long contexts is U-shaped (*lost in the middle*, Liu et
al. 2023), so packing in plain rank order buries the second- and third-best passages
in the dead zone. `reorder_lost_in_middle` interleaves outward from both ends —
`[0,1,2,3,4,5]` becomes `[0,2,4,5,3,1]` — so both high-attention extremes are held by
the highest-ranked passages and the weakest end up where being ignored costs least.

## Two substitutions happen at pack time, not retrieval time

`_expand` swaps in the wider text a chunk points at: `metadata["window_text"]` from the
sentence-window splitter, `metadata["parent_text"]` from the parent-document retriever.
Both patterns search a *precise* unit and generate from a *wide* one, and the swap
happens **after** ranking so precision is preserved where it matters. Several children
sharing one parent emit the parent once.

Rendering is 1-based, numbered and stable, because the generator cites `[n]` and the
validator resolves those numbers back to these chunks. Each body is wrapped by
`ragorc.security.wrap_untrusted` unless `isolate=False`.

## Overflow

`decide_strategy` returns `fit`, `truncate` or `summarize`. Summarization has three
shapes: `map_reduce` (default — parallel, one pass), `refine` (sequential, better on
narrative but serial by construction), `hierarchical` (repeated map-reduce for very
large sets). Compression costs model calls, so it runs only when the budget says it
must.

## Usage

```python
from ragorc.context import ContextBudgeter, ContextPacker, ContextSummarizer

budgeter, packer = ContextBudgeter(), ContextPacker()
plan = budgeter.plan(system_prompt=prompt.system, question=query.text)

if budgeter.decide_strategy(chunks, plan) == "summarize":
    compressed = await ContextSummarizer(llm).fit(
        query.text, chunks, budget=plan.budget.available_context, strategy="map_reduce"
    )
    chunks = compressed.chunks

pack = packer.build(chunks, budget=plan.budget.available_context)
rendered = prompt.render(context=pack.text, question=query.text)
print(pack.tokens, pack.dropped, pack.order)  # "lost_in_middle" or "rank"
```

`AnswerGenerator` already does exactly this; call the pieces directly only when you
are building a different terminal stage.

## Settings

| Setting | Effect |
|---|---|
`llm.context_window` | the total the budget divides |
`generation.reserved_output_tokens` · `max_answer_tokens` | room held back for the answer |
`retrieval.reorder_lost_in_middle` | on by default; the placement fix above |
`retrieval.parent_expansion` | enables the parent/window substitution in `_expand` |
`retrieval.compression_enabled` · `compression_ratio` | post-retrieval compression, distinct from overflow compression |
`llm.model` | summarization runs on `Task.SUMMARIZE` (balanced tier — a summary becomes the retrieval target) |
