# Documentation

## Start here

| If you want to… | Read |
|---|---|
Understand how the system fits together | [architecture.md](architecture.md) |
Know why a particular choice was made | [adr/](adr/) — nine decision records |
See the flow as a picture | [diagrams/](diagrams/) — six Mermaid diagrams |
Make it faster | [performance.md](performance.md) |
Make it cheaper | [cost.md](cost.md) |
Deploy it safely | [security.md](security.md) |
Run it in production | [operations.md](operations.md) |
Use one specific module | [modules/](modules/) |

## The nine decisions that shape everything else

| ADR | Decision | Why it matters |
|---|---|---|
[0001](adr/0001-langgraph-for-orchestration.md) | LangGraph for orchestration, nothing else from LangChain | The pipeline has cycles (CRAG, Self-RAG); it is a state machine, not a chain |
[0002](adr/0002-late-chunking.md) | **Late chunking by default**, degrading explicitly | Better *and* cheaper than early chunking — one forward pass per document, not per chunk |
[0003](adr/0003-server-side-fusion.md) | Hybrid search fused inside Qdrant, one round trip | No second search engine, no index-sync job, fusion in Rust next to the data |
[0004](adr/0004-fastembed-onnx.md) | FastEmbed/ONNX, no PyTorch in the base install | OpenRouter has no embeddings endpoint; ~2.5 GB smaller install, works offline |
[0005](adr/0005-model-cascade.md) | A task-tiered model cascade | One query makes 10-40 model calls and exactly one produces text a human reads |
[0006](adr/0006-layered-query-guards.md) | Three independent layers of query guarding | Text-to-SQL is an arbitrary-query primitive driven by user input |
[0007](adr/0007-cache-tiers.md) | Three cache tiers, including a semantic one | A semantic hit skips the *entire* pipeline, not just one call |
[0008](adr/0008-dataclasses-in-hot-path.md) | Dataclasses in the hot path, pydantic at the boundaries | Millions of `Chunk` objects per ingest make construction cost architectural |
[0009](adr/0009-dependency-pinning.md) | Floors are the versions we verified; majors are capped | `sqlglot>=25` with code tested on 30 is an untested claim — and the SQL guard matches on its AST node names |

## Diagrams

Rendered with any Mermaid tool (`make diagrams` needs `mmdc`), or viewed directly
on GitHub, which renders `.mmd` inline.

| Diagram | Shows |
|---|---|
[pipeline.mmd](diagrams/pipeline.mmd) | question → validate → translate → route → construct → retrieve → verify → answer, with both feedback loops |
[component-map.mmd](diagrams/component-map.mmd) | every architecture panel mapped to its module |
[hybrid-search.mmd](diagrams/hybrid-search.mmd) | the single gRPC round trip: dense + BM25 + ColBERT fused server-side |
[ingestion.mmd](diagrams/ingestion.mmd) | ingest with the chunking-strategy ladder (LATE → CONTEXTUAL → EARLY) |
[graphrag.mmd](diagrams/graphrag.mmd) | graph construction plus local / global / DRIFT search |
[feedback-loops.mmd](diagrams/feedback-loops.mmd) | CRAG and Self-RAG as one state machine |

## Reading order for a new contributor

1. [architecture.md](architecture.md) §1-2 — the shape of the system and the data model.
2. `ragorc/core/protocols.py` — every seam in the library is one of these interfaces.
3. [ADR-0002](adr/0002-late-chunking.md) and [ADR-0003](adr/0003-server-side-fusion.md) — the two decisions that most affect retrieval quality.
4. `tests/unit/test_security_guards.py` — the attack corpus explains the security model faster than prose does.
5. `docs/internal/CONTRACTS.md` — the rules every module follows.
[Open items](internal/OPEN-ITEMS.md) — known gaps, each verified against the code

## What is deliberately not claimed

Stated here so nobody has to discover it by reading source:

- **Injection detection is heuristic.** It raises the cost of an attack; structural
  isolation and least privilege are the actual controls. See [security.md](security.md).
- **PII detection is regex-based.** It will miss names, addresses and free-text
  identifiers. Use Presidio for coverage rather than best effort.
- **Published benchmark numbers do not transfer.** Every default in this library is
  a reasonable prior, not a substitute for running [the eval harness](modules/) on
  your own corpus.
- **Degradation is silent by design.** A dead store is recorded in
  `RetrievalResult.errors` and the query proceeds. That is correct behaviour and a
  monitoring obligation — see the failure-mode table in [operations.md](operations.md).
