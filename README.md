# rag-orc

A complete, production-grade implementation of the **RAG reference architecture** —
every component of the canonical "RAG From Scratch" map — as a reusable Python
library.

- **Relational:** PostgreSQL + pgvector
- **Graph:** Neo4j
- **Vector:** Qdrant
- **LLM:** OpenRouter
- **Embeddings:** FastEmbed (local ONNX, no PyTorch) · OpenAI · Voyage · Cohere
- **Orchestration:** LangGraph for the cyclic pipelines, nothing else from LangChain

Built for performance and cost from the first line: async throughout, bounded
fan-out, gRPC to Qdrant, server-side hybrid fusion in one round trip, vectorized
scoring, a three-tier cache, and a model cascade that keeps the 20 classification
calls per query off your frontier model.

---

## What's implemented

Every box in the architecture, mapped to real code:

### Query Translation — `ragorc/translate/`
| Component | Module |
|---|---|
Multi-Query | `multi_query.py` |
RAG-Fusion (reciprocal rank fusion) | `rag_fusion.py` |
Step-Back prompting | `step_back.py` |
Query Decomposition (+ answer chaining) | `decomposition.py` |
HyDE (with question/hypothesis vector blending) | `hyde.py` |

### Routing — `ragorc/route/`
| Component | Module |
|---|---|
Logical routing (LLM picks the datastore) | `logical.py` |
Semantic routing (embedding picks the prompt — no LLM call) | `semantic.py` |
Hybrid, with a free rule-based fast path | `hybrid.py` |

### Query Construction — `ragorc/construct/`
| Component | Module |
|---|---|
Text-to-SQL → Postgres | `text_to_sql.py` |
Text-to-Cypher → Neo4j | `text_to_cypher.py` |
Self-query retriever (auto metadata filters) | `self_query.py` |

All three pass through `ragorc/security/` guards before anything executes.

### Indexing — `ragorc/index/`
| Component | Module |
|---|---|
Semantic splitter (4 breakpoint methods) | `split/semantic.py` |
Recursive · token · markdown · code · sentence-window | `split/` |
**Late chunking** (preferred; needs `[local]`) | `../embed/late_chunking.py` |
Contextual retrieval (Anthropic-style prefixes) | `contextual.py` |
Parent-document indexing | `multirep/parent_document.py` |
Summary indexing | `multirep/summary.py` |
Dense-X / proposition indexing | `multirep/dense_x.py` |
ColBERT late interaction (Qdrant multivectors) | `colbert.py` |
RAPTOR (UMAP → soft GMM → summarize → recurse) | `raptor.py` |
GraphRAG construction (extract → resolve → Leiden → reports) | `graph/` |
Ingest orchestration (idempotent, backpressured) | `pipeline.py` |

### Retrieval — `ragorc/retrieve/`
| Component | Module |
|---|---|
Hybrid search — dense + BM25 + SPLADE + ColBERT, **fused server-side** | `hybrid.py` |
Fusion: RRF · DBSF · weighted · relative-score | `fusion.py` |
Cross-encoder reranking (ONNX) | `rerank.py` |
RankGPT listwise reranking (sliding window) | `rankgpt.py` |
Contextual compression (extract · embedding filter · sentence) | `compress.py` |
**Noise handling** — dedupe, near-dupe, relative cutoff, MMR, lost-in-the-middle | `noise.py` |
CRAG — grade → refine / rewrite / web, with knowledge strip | `crag.py` |
GraphRAG local · global · DRIFT search | `graph.py` |
**Multi-hop** — IRCoT iterative + bridge-entity paths | `multihop.py` |
Multi-store fan-out with circuit breakers | `multi_store.py` |

### Generation — `ragorc/generate/`
| Component | Module |
|---|---|
Grounded answering with citations | `answer.py` |
Span-level citation attribution | `citations.py` |
**Hallucination control** — claim decomposition + per-claim entailment | `groundedness.py` |
Self-consistency sampling | `consistency.py` |
**Abstention policy** | `abstain.py` |
Self-RAG (ISSUP/ISUSE grading loop) | `self_rag.py` |
RRR (rewrite–retrieve–read) | `rrr.py` |

### Cross-cutting
| Concern | Module |
|---|---|
**Context overflow** — budget, knapsack packing, map-reduce/refine compression | `context/` |
**Security** — SQL AST guard, Cypher guard, injection defence, PII, tenancy, audit | `security/` |
**Validation** — inbound query, outbound answer, ingest documents | `validate/` |
**Cache** — memory → Redis → semantic | `cache/` |
Cost cascade, ledger and hard ceilings | `llm/router.py`, `core/telemetry.py` |
LangGraph pipelines: naive · adaptive · CRAG · Self-RAG · GraphRAG · multi-hop | `pipeline/` |
Evaluation harness | `eval/` |
FastAPI service · Typer CLI · LangChain adapters | `server/`, `cli.py`, `adapters/` |

---

## Quickstart

```bash
make install            # uv venv + deps
cp .env.example .env    # add your OpenRouter key
make up                 # Postgres + Neo4j + Qdrant via docker compose
make schema             # create collections, tables, indexes
make seed               # ingest the example corpus
make ask Q="what is late chunking and why is it cheaper?"
```

```python
import asyncio
from ragorc import build_pipeline


async def main():
    rag = await build_pipeline()  # reads .env / environment
    await rag.ingest("./docs")

    answer = await rag.query("Why is late chunking cheaper than early chunking?")
    print(answer.text)
    print(
        f"grounded={answer.grounded} ({answer.groundedness:.2f})  cost=${answer.usage.cost_usd:.4f}"
    )
    for c in answer.citations:
        print(f"  [{c.chunk_id[:8]}] {c.quote[:70]}")


asyncio.run(main())
```

Turn features on individually — everything is a setting, and nothing exotic is on
by default:

```bash
RAGORC_RETRIEVAL__CRAG_ENABLED=true
RAGORC_GENERATION__SELF_RAG_ENABLED=true
RAGORC_GRAPH__ENABLED=true                 # GraphRAG + multi-hop
RAGORC_INDEXING__RAPTOR_ENABLED=true
RAGORC_INDEXING__CHUNKING_STRATEGY=late
```

---

## Design decisions worth knowing

Each links to the ADR with the full reasoning.

**Late chunking, preferred and honestly degraded** ([ADR-0002](docs/adr/0002-late-chunking.md)) —
one forward pass over the whole document, then mean-pool token embeddings per
chunk span, so a chunk saying *"its revenue grew 40%"* keeps who *it* is. Both
better than early chunking **and cheaper** (one pass per document, not per chunk).

It requires one model emitting both token and pooled vectors, so the
zero-dependency default (FastEmbed, pooled output only) resolves to `early`.
Install `ragorc[local]` with a token-capable model such as
`jinaai/jina-embeddings-v2-base-en` to get `late`. The resolver logs which rung of
the ladder it landed on, and never substitutes another model's vectors to
manufacture the answer.

**Hybrid search in one round trip** ([ADR-0003](docs/adr/0003-server-side-fusion.md)) —
dense, BM25-as-sparse (with Qdrant's IDF modifier), SPLADE and ColBERT
multivectors all live in Qdrant, fused server-side by `FusionQuery(RRF)`. No
Elasticsearch, no index-sync job, no client-side score normalization.

**A model cascade, not one model** ([ADR-0005](docs/adr/0005-model-cascade.md)) —
one query makes 10-40 model calls and exactly one produces text a human reads. The
graders, routers and rewriters run on a cheap model; only synthesis and escalation
use the expensive one.

**Three independent layers of query guarding** ([ADR-0006](docs/adr/0006-layered-query-guards.md)) —
`sqlglot` AST validation, read-only transactions, and a `SELECT`-only role. A
substring blocklist misses `WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x`;
an AST walk does not.

**Retrieved documents are untrusted input** — scanned for prompt injection,
NFKC-normalized with invisible and bidi characters stripped, and structurally
isolated in delimiters their own content cannot break out of.

**Abstention is a success state** — a system that always answers cannot signal
inadequate evidence, so its worst answers look exactly like its best.

**Dataclasses in the hot path, pydantic at the boundaries**
([ADR-0008](docs/adr/0008-dataclasses-in-hot-path.md)) — millions of `Chunk`
objects per ingest make construction cost and memory footprint architectural
concerns.

---

## Documentation

| Document | Contents |
|---|---|
[architecture.md](docs/architecture.md) | how the pieces fit, and why |
[performance.md](docs/performance.md) | library choices, where time goes, the dials |
[cost.md](docs/cost.md) | the 10-40 calls per query, and the five ways to cut them |
[security.md](docs/security.md) | threat model, controls, and what is *not* claimed |
[operations.md](docs/operations.md) | scaling, deployment, monitoring, failure modes |
[adr/](docs/adr/) | nine architecture decision records |
[diagrams/](docs/diagrams/) | Mermaid: pipeline, ingestion, hybrid search, GraphRAG, feedback loops, component map |

---

## Installation

```bash
pip install ragorc                      # core: 3 stores, OpenRouter, FastEmbed, LangGraph
pip install "ragorc[server]"            # FastAPI + uvicorn
pip install "ragorc[raptor]"            # UMAP + scikit-learn
pip install "ragorc[graphrag]"          # igraph + leidenalg + networkx
pip install "ragorc[loaders]"           # PDF, HTML, DOCX
pip install "ragorc[redis]"             # shared cache tier
pip install "ragorc[all]"               # everything except torch
pip install "ragorc[local,nli]"         # sentence-transformers + torch
```

The base install has **no PyTorch**: embeddings, sparse vectors, ColBERT and
reranking all run on ONNX Runtime via FastEmbed
([ADR-0004](docs/adr/0004-fastembed-onnx.md)).

### Store versions

`make up` pins **Qdrant v1.19**, **Postgres 16** (`pgvector/pgvector:pg16`) and
**Neo4j 5.26**, and that combination is what the integration suite runs against.

One hard floor is worth naming because it is not obvious from the driver
requirement: **Neo4j 5.9+**. Entity traversal uses quantified path patterns,
which is how it excludes the `MENTIONS` and `IN_COMMUNITY` scaffolding edges —
an untyped `-[*1..2]-` constrains only the endpoint, walks through `Chunk` nodes,
and returns co-occurrence as though it were an extracted relationship.

## Development

```bash
make check              # ruff + unit tests — no services, no API key, no downloads
make up                 # start Postgres + Neo4j + Qdrant
make doctor             # diagnose "the stack is up but nothing connects"
make test-integration   # integration tests from the host
make test-docker        # the same tests from inside the docker network (what CI does)
make test-e2e           # real embeddings + real stores + one real OpenRouter call
make cov                # coverage report
```

The unit suite runs entirely offline against fakes for all three stores and a
scripted LLM stub — no API keys, no containers, no model downloads. Integration
tests are deselected by default and opt in with `-m integration`.

`make test-docker` exists because it is how CI reaches the services (by compose
service name, not through published host ports) — and because on Docker Desktop
the host forwarder can wedge while the services stay perfectly healthy, which
`make doctor` will tell you.

## License

MIT
