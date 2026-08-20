# ADR-0001: LangGraph for pipeline orchestration, nothing else from LangChain

**Status:** accepted · **Date:** 2026-08-19

## Context

The architecture has two feedback loops that make it a state machine rather than
a chain:

- **CRAG** grades retrieved documents and, if they are irrelevant, rewrites the
  query and/or retrieves from the web — then re-enters grading.
- **Self-RAG / RRR** grades the *generated answer* for groundedness and utility
  and, on failure, loops back to retrieval or rewriting.

Both are cyclic with conditional edges and a bounded iteration count. Expressing
that as nested `async` functions produces code where the control flow is
implicit, the state is threaded by hand through every signature, and partial
progress is lost on failure.

## Decision

Use **LangGraph** as the pipeline engine, and take nothing else from the
LangChain ecosystem into the core.

LangGraph earns its place because it provides exactly the four things this
problem needs and none of them are trivial to write well:

1. A typed, reducer-based state object shared across nodes.
2. Conditional edges and cycles with a recursion limit.
3. Parallel "supersteps" — fan-out nodes execute concurrently automatically.
4. Checkpointing and token-level streaming out of the box.

What we deliberately do **not** use:

| LangChain component | Why not |
|---|---|
`ChatOpenAI` / LCEL | We need OpenRouter provider routing, per-call cost from the provider, a token bucket, a semantic cache and a structured-output repair loop. Wrapping a wrapper to get them is worse than 400 lines of `httpx`. |
`QdrantVectorStore` | Cannot express multivectors, quantization search params, or server-side prefetch fusion — see ADR-0003. Those are the performance features. |
Retrievers / splitters | Pure-Python loops where we want vectorized numpy, and no hook for late chunking. |

Crucially, **LangGraph nodes are plain async callables**. Using LangGraph does
not oblige us to use LangChain's model or retriever abstractions, so we get the
orchestration without the overhead.

## Consequences

- `langgraph` + its `langchain-core` dependency are in the base install. That is
  the whole cost, and it is small.
- Every pipeline lives in `ragorc/pipeline/graphs/` as a compiled graph, so the
  control flow is a diagram you can print (`graph.get_graph().draw_mermaid()`),
  not something you reconstruct by reading call sites.
- Interop runs the other way too: `ragorc/adapters/langchain.py` exposes our
  retrievers as LangChain `BaseRetriever`s for projects already invested in LCEL.
