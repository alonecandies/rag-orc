# `ragorc.pipeline` — the LangGraph state machines

A RAG query is not a chain. It is a **state machine with two feedback loops**: CRAG
grades the retrieved documents and may rewrite or search the web before re-grading;
Self-RAG grades the *generated answer* and may loop back to retrieval. Both are cyclic
with conditional edges and a bounded iteration count — exactly what LangGraph expresses
and what nested `async` functions do not.

Related: [ADR-0001](../adr/0001-langgraph-for-orchestration.md).

## The three layers

`graphs/` is one module per topology (`NAME`, `build(nodes)`,
`recursion_limit(settings)`); `nodes.py` holds every stage as an async callable, shared
by all of them; `state.py` is what flows through; `builder.py` is the public facade.

```python
RAGState(TypedDict, total=False)
    question, tenant_id, top_k, pipeline, prompt_name    # request
    query, route, retrieval, answer                      # stage outputs
    candidates, per_store, web_chunks, usages, errors, warnings, rewrites, tools_used
    grade, grounded, groundedness, useful, utility, sufficient, follow_up
    retrieve_iterations, generate_iterations, search_mode, metadata

initial_state(question, *, tenant_id=None, top_k=None, pipeline="auto", prompt_name=None)
total_usage(state) -> Usage  ·  evidence(state) -> list[ScoredChunk]
failure(stage, exc) -> dict  ·  merge_store_lists(left, right)   # the per_store reducer
```

`total=False` because a node returns a **partial** state and LangGraph materializes only
the channels that were written; nodes read with `state.get(...)` and never index a key
they did not set. The accumulator channels carry `Annotated[..., operator.add]` so
parallel supersteps merge instead of overwriting, which is what makes a three-store
fan-out safe. One state type serves every graph: the nodes are shared, so a per-graph
state would mean a per-graph copy of every node.

## Nodes

```python
PipelineNodes(*, llm, generator: AnswerGenerator, retriever: Retriever, settings=None,
              store_retrievers={}, validator=None, translator=None, router=None,
              constructor=None, reranker=None, compressor=None, crag=None,
              self_rag=None, web=None, graph_retrievers={}, bridge_retriever=None,
              multihop=None, model_router=None, noise=None, packer=None, budgeter=None)
```

Only `llm`, `generator` and `retriever` are required. Each node checks for the
collaborator it needs and **degrades**: a missing translator means the query is not
translated, not that the query fails — which is what lets one node set serve seven
topologies. Three exceptions escape a node rather than being recorded as a degraded
stage (`GuardrailViolation`, `ValidationFailed`, `BudgetExceeded`): each is a decision
rather than an outage.

Stages: `validate` · `translate` · `route` · `construct` · `retrieve` ·
`store_node(store)` · `fuse` · `rerank` · `compress` · `grade` · `rewrite` · `web_search`
· `generate` · `verify_groundedness` · `verify_utility` · `abstain` · `bridge` ·
`check_sufficiency` · `hop` · `graph_node(mode)` · `multihop_retrieve`.

## The seven graphs

| Module | Shape | Use it when |
|---|---|---|
`naive` | validate → retrieve → generate | the baseline every measurement is compared against |
`adaptive` | validate → translate → route → **parallel stores** → fuse → rerank → generate | the general-purpose pipeline |
`crag` | adds grade → {refine, rewrite, web} → re-grade | the corpus sometimes cannot answer |
`self_rag` | adds judge → {accept, rewrite+retry, abstain} | answer quality must be gated, not hoped for |
`graphrag` | local / global / DRIFT search, then reduce | entity, relationship and corpus-wide questions |
`multihop` | retrieve → reason → hop, with a sufficiency early exit | answers that are a composition of facts |
`agentic` | tool selection with a bounded loop | heterogeneous questions in one endpoint |

Each `recursion_limit(settings)` is **derived from the topology and the configured loop
bounds**, not left to LangGraph's default of 25 — a limit nobody derived is a limit
nobody notices is wrong. And because every graph is compiled, the control flow is
printable (`graph.get_graph().draw_mermaid()`) rather than reconstructed from call sites.

## Usage

```python
import asyncio
from ragorc import build_pipeline


async def main() -> None:
    async with await build_pipeline() as rag:  # reads .env / environment
        await rag.ingest("examples/corpus")
        answer = await rag.query("who is on call for the Graph Service?")
        print(answer.text, answer.grounded, answer.usage.cost_usd)


asyncio.run(main())
```

To drive one graph directly:

```python
from ragorc.pipeline.graphs import crag
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import initial_state, total_usage

nodes = PipelineNodes(llm=llm, generator=generator, retriever=hybrid, crag=corrective)
graph = crag.build(nodes)
final = await graph.ainvoke(
    initial_state("why is late chunking cheaper?"),
    config={"recursion_limit": crag.recursion_limit(settings)},
)
print(final["answer"].text, total_usage(final).cost_usd, final.get("errors"))
```

Wrap any direct invocation in `ragorc.core.telemetry.new_request_context` so the trace
and the cost ceilings apply; `RAGPipeline.query` already does.

## Settings

| Setting | Effect |
|---|---|
`retrieval.crag_enabled` | selects the CRAG topology under `pipeline="auto"` |
`generation.self_rag_enabled` · `self_rag_max_retries` | the answer-side loop and its bound |
`generation.rrr_enabled` · `rrr_max_rewrites` | pre-retrieval rewriting |
`graph.enabled` · `multihop_enabled` · `multihop_max_iterations` · `multihop_stop_on_sufficient` | graph and multi-hop topologies |
`graph.local_search_*` · `global_search_*` | which mode the `search_mode` channel selects |
`retrieval.max_concurrent_retrievers` · `per_store_timeout_s` | superstep width and deadline |
`cost.max_llm_calls_per_query` · `max_cost_per_query_usd` | the only real bound on a cyclic graph |
`observability.trace_enabled` | per-stage timings on `Answer.trace`; off withholds them (a trace records what each stage did with the retrieved text) |
`observability.slow_query_ms` | warn-log threshold on the query's **wall time** — independent of `trace_enabled`, and not the sum of the steps, which double-counts nesting and adds concurrent legs |
