# Architecture

`ragorc` implements every component of the RAG reference architecture — the six
coloured panels of the canonical "RAG From Scratch" map — as composable Python
modules over Postgres, Neo4j, Qdrant and OpenRouter.

This document explains how the pieces fit and, more importantly, **why each
non-obvious choice was made**. The decisions with lasting consequences live in
[`docs/adr/`](adr/) and are cross-referenced throughout.

```mermaid
%% docs/diagrams/component-map.mmd
```
See [`diagrams/component-map.mmd`](diagrams/component-map.mmd) for the module map,
and [`diagrams/pipeline.mmd`](diagrams/pipeline.mmd) for the query flow.

---

## 1. The shape of the system

A RAG query is not a chain. It is a **state machine with two feedback loops**:

- **CRAG** grades retrieved documents; if they are irrelevant it rewrites the
  query and/or searches the web, then re-enters grading.
- **Self-RAG / RRR** grades the *generated answer* for groundedness and utility;
  on failure it loops back to retrieval or rewriting.

That is why the orchestration layer is LangGraph ([ADR-0001](adr/0001-langgraph-for-orchestration.md))
and nothing else from LangChain is in the core. LangGraph nodes are plain async
callables, so we get cycles, conditional edges, parallel supersteps,
checkpointing and streaming without adopting any of its model or retriever
abstractions — which is exactly what we want, because those abstractions cannot
express the performance features this library is built on.

### Layering

```
                 ┌───────────────────────────────────────────────┐
   pipeline/     │ LangGraph state machines (naive, adaptive,     │
                 │ crag, self_rag, graphrag, multihop, agentic)   │
                 └───────────────────────────────────────────────┘
                             ▼ composes ▼
  translate/  route/  construct/  retrieve/  generate/  context/  eval/
                             ▼ built on ▼
        llm/        embed/       stores/      cache/     security/  validate/
                             ▼ contracts ▼
                 ┌───────────────────────────────────────────────┐
   core/         │ models · protocols · settings · errors ·       │
                 │ concurrency · telemetry · tokens · ids         │
                 └───────────────────────────────────────────────┘
```

Everything above `core/` depends only on **protocols**, never on concrete
classes. `Retriever`, `LLM`, `VectorStore` and the rest are
`typing.Protocol`s, so a component satisfies an interface by shape — your own
store or a third-party retriever drops in with no adapter and no inheritance.

---

## 2. Data model

Two kinds of object, split by trust boundary ([ADR-0008](adr/0008-dataclasses-in-hot-path.md)):

| | Used for | Why |
|---|---|---|
`@dataclass(slots=True)` | `Document`, `Chunk`, `ScoredChunk`, `Query`, `Entity`, `Relation`, `Usage`, `Answer` | Hot path — millions of instances per ingest. ~2-3x faster to construct than pydantic and ~40% less memory, which is what decides whether a 1M-chunk ingest fits in RAM. |
pydantic `BaseModel` | `Settings`, `core/schemas.py`, HTTP bodies | Trust boundaries, where validation is the point. LLM output schemas double as prompt content — field descriptions are sent to the model. |

Vectors are `numpy` arrays, never `list[float]`. A 1024-dim float32 array is
4 KiB; the equivalent Python list is ~40 KiB.

A `Chunk` can carry **three representations simultaneously** — `dense`,
`sparse`, and `multi` (an `(n_tokens, dim)` ColBERT matrix) — because the hybrid
retriever queries all three in one round trip.

---

## 3. Indexing

See [`diagrams/ingestion.mmd`](diagrams/ingestion.mmd).

### Splitting produces boundaries, not vectors

Every splitter returns `Chunk` objects with exact `start_char`/`end_char` and
**no vectors**. This separation is not tidiness — it is what makes late chunking
possible. A splitter that embedded as it split would foreclose the best available
chunking strategy.

Six splitters: `semantic` (default), `recursive`, `token`, `markdown`, `code`,
`sentence_window`.

The semantic splitter implements four breakpoint methods, all vectorized:
`percentile`, `stddev`, `interquartile`, and `gradient`. The gradient method
detects the *rate of change* in the similarity curve rather than its absolute
value, which is what finds boundaries in uniformly dense technical prose where
every consecutive pair looks similar.

### The chunking-strategy ladder ([ADR-0002](adr/0002-late-chunking.md))

```
LATE  ──embedder cannot expose token vectors──▶  CONTEXTUAL  ──disabled──▶  EARLY
```

**Late chunking is preferred** because it is both better *and cheaper*: one
forward pass over the whole document, then mean-pool token embeddings per chunk
span, so every chunk vector is conditioned on the full document. A chunk reading
*"its revenue grew 40%"* keeps who *it* is. One pass per document beats one pass
per chunk.

It needs a dense model that exposes token-level output — the *same* model, since
the pooled vector must be comparable to the query vector. The FastEmbed default
returns pooled vectors only, so `AUTO` resolves to `early` there; `ragorc[local]`
with a token-capable model gets you `late`.

It degrades explicitly, never silently: hosted embedding APIs return only pooled
vectors, so `resolve_strategy()` detects the capability and logs which strategy
was chosen and why.

### Multi-representation indexing

The unifying idea: **the retrieval unit and the generation unit need not be the
same object.** Search something small and precise; hand the model something large
and complete.

| Indexer | Searches | Generates from |
|---|---|---|
`parent_document` | small child chunks | the parent chunk |
`summary` | an LLM summary | the original text |
`dense_x` | atomic propositions | the source chunk |
`sentence_window` | one sentence | the surrounding window |
`raptor` | cluster summaries at every level | whichever level matched |

### RAPTOR

The real algorithm, not an approximation: UMAP dimensionality reduction → **soft**
Gaussian-Mixture clustering with BIC-selected component count → LLM cluster
summaries → recurse. Soft clustering matters because a chunk may legitimately
belong to several clusters; topics overlap. Two-stage (global then local)
clustering keeps large corpora from collapsing into one cluster.

All levels are indexed into the same collection so the collapsed-tree query can
search every abstraction level at once — the right level of abstraction is a
property of the *question*, not something to navigate to.

### GraphRAG construction

See [`diagrams/graphrag.mmd`](diagrams/graphrag.mmd).

`extract → resolve → write → detect communities → summarize`

**Entity resolution is the step that decides whether the graph works at all.**
Without it the graph fragments into "Acme", "Acme Corp" and "ACME Corporation" as
three unconnected nodes, and traversal stops finding anything. Three stages,
cheapest first: exact casefolded match, normalized form (legal suffixes and
punctuation stripped), then embedding similarity — *blocked* by first character
and type so it is not O(n²) over the whole graph.

---

## 4. Retrieval

### Hybrid search in one round trip ([ADR-0003](adr/0003-server-side-fusion.md))

See [`diagrams/hybrid-search.mmd`](diagrams/hybrid-search.mmd).

Every representation lives in Qdrant as a vector, so Qdrant does the fusion:

| Representation | Stored as | Catches |
|---|---|---|
Semantic | dense vector | paraphrase, concepts |
Lexical | **sparse BM25 vector** with `Modifier.IDF` | identifiers, error codes, rare proper nouns |
Learned sparse | SPLADE sparse vector (optional) | term expansion |
Late interaction | **multivector**, `MaxSim` comparator | precise reranking |

One `query_points` call carries a `prefetch` list and a `FusionQuery(RRF)`,
optionally nested inside a ColBERT `MaxSim` rerank. **No second search engine, no
index-sync job, no consistency window.**

Two details that are easy to get wrong: the sparse vector needs
`Modifier.IDF` or BM25-as-sparse degenerates into term frequency; and the ColBERT
multivector must have `hnsw m=0` because it is a reranking field reached through
prefetch, never a first-stage index.

### Recall here, precision later

`hybrid.py` returns `fetch_k` (default 50) candidates, not `top_k` (default 10).
Recall is set at the first stage and cannot be recovered afterwards — a reranker
can only reorder what it was given. Precision is the reranker's job.

### Noise handling

Five filters, ordered cheapest-first so expensive ones never run on rows already
discarded:

1. **exact dedupe** — by id *and* by word-normalized content hash, because the
   same passage indexed under two documents has two ids and one meaning
2. **relative score cutoff** — a fraction of the top score, not an absolute
   floor. An absolute similarity threshold is wrong per corpus and per embedding
   model, and fails in both directions
3. **near-duplicate collapse** — embedding cosine when vectors are present
   (one matmul), SimHash Hamming distance otherwise
4. **MMR diversity** — one matmul for the full pairwise matrix, then a greedy
   loop over slices
5. **lost-in-the-middle reordering** — strongest evidence at *both* ends of the
   context, because attention over long contexts is U-shaped

### CRAG

Grade the top-k documents → decide `CORRECT` / `AMBIGUOUS` / `INCORRECT` →
refine, or rewrite and search the web, or both. The **knowledge-strip** step that
most implementations skip is implemented: an otherwise-relevant document is split
into strips, the strips are graded, and only the relevant ones are reassembled.

---

## 5. Generation

The order of operations enforces every guarantee, and it is chosen so each step's
cost is only paid when the previous step allowed it:

```
abstain(pre) → budget → pack (+compress) → generate
             → citations → validate → ground → abstain(post)
```

**Abstain before generating.** With no usable evidence, the synthesis call is
waste — and a model given no evidence answers from its parameters, which produces
the most confident hallucination in the system.

**Validate citations before grading groundedness.** Citation validation is string
matching (free) and decisive; groundedness costs model calls. Running the cheap
check first means the expensive one is skipped on answers already disqualified.

### Hallucination control

| Mechanism | Catches | Cost |
|---|---|---|
Citation existence | `[7]` when six passages were supplied | free |
Quote verification | a plausible quote the cited document does not contain | free |
Holistic groundedness grade | gross unsupported answers | 1 cheap call |
**Claim decomposition + per-claim entailment** | composition errors and detail drift | N cheap calls |
NLI cross-encoder | same, locally, no API | free after model load |
Self-consistency | idiosyncratic fabrication | N × synthesis |
Abstention policy | everything above, as a decision | free |

The claim-level check exists because a single whole-answer grade anchors on
overall plausibility and misses the two failures that matter most: a causal claim
the context never makes, and a number that is *close* to the source but not the
source's.

Self-consistency compares **claims, not strings** — paraphrases share almost no
word forms, and two answers with different numbers share nearly all of them, so
naive text similarity scores both cases backwards.

**Abstention is a success state.** A system that always answers cannot signal
inadequate evidence, so its worst outputs are indistinguishable from its best.

---

## 6. Cross-cutting concerns

### Cost ([ADR-0005](adr/0005-model-cascade.md))

One query makes 10-40 model calls, of which **one** produces text a human reads.
Three tiers; every stage declares its `Task`, and only synthesis and escalation
reach the expensive model. Plus: OpenRouter price-floor routing, provider-reported
per-call cost, three cache tiers, and a `CostLedger` with hard ceilings checked
*before* each call — loops and retries have no natural spending bound otherwise.

### Security ([ADR-0006](adr/0006-layered-query-guards.md))

Text-to-SQL is an arbitrary-query primitive driven by user input. Three
independent layers: `sqlglot` **AST** validation (a statement type is a node type
and cannot be disguised by formatting), read-only transactions with server-side
timeouts, and a database role holding `SELECT` only.

Retrieved documents are untrusted input too — scanned for injection,
NFKC-normalized with invisible and bidi characters stripped, and structurally
isolated in delimiters their own content cannot break out of.

### Caching ([ADR-0007](adr/0007-cache-tiers.md))

`memory (200ns) → redis (200µs) → semantic (2ms)`, read-through with promotion.
The semantic tier — nearest-neighbour lookup over question embeddings — is the
largest single cost lever in a production deployment, and the one setting to be
conservative with: below ~0.95 similarity you start answering questions nobody
asked.

### Concurrency

Async throughout, with **bounded** fan-out. An unbounded `asyncio.gather` over
50k chunks exhausts memory and gets you rate-limited before it gets you an
answer. Per-store circuit breakers and timeouts mean a dead backend degrades the
query instead of defining its latency.

---

## 7. Extending it

| To add | Do this |
|---|---|
A splitter | satisfy `Splitter`, decorate `@register("splitter", "name")` |
A retriever | satisfy `Retriever`, decorate `@register("retriever", "name")` |
An embedding provider | satisfy `DenseEmbedder`, add a branch in `embed/factory.py` |
A store | satisfy `VectorStore` / `RelationalStore` / `GraphStore` |
A prompt | `register_prompt(Prompt(...))` in or alongside `llm/prompts.py` |
A pipeline | compose a LangGraph in `pipeline/graphs/` |

Nothing requires subclassing, because every seam is a Protocol.
