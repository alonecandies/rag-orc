# Performance

Every library choice and default in `ragorc` is a performance decision. This is
the reasoning, and the dials worth turning.

## The rules that produced these choices

1. **Rust/C cores over pure Python** where the work is mechanical.
2. **ONNX Runtime over PyTorch** for inference — no torch in the base install.
3. **One round trip instead of two**, wherever a server can do the merge.
4. **Vectorize** — no Python loop over vectors, ever.
5. **Bounded** fan-out — wide, but never unbounded.
6. **Never recompute what a hash can identify.**

## Library choices and what they buy

| Concern | Choice | Measured benefit |
|---|---|---|
Models/validation | pydantic v2 | Rust core; ~5-50x v1 |
Hot-path objects | `dataclass(slots=True)` | ~2-3x faster construction, ~40% less memory than pydantic |
JSON | orjson | 3-10x stdlib `json` |
Event loop | uvloop | 2-4x asyncio throughput — **only if you start the loop with it**, see below |
HTTP | httpx + **HTTP/2** | 16 concurrent LLM calls multiplex over one TCP connection |
Tokenizing | tiktoken | Rust BPE; `encode_batch` releases the GIL |
Embeddings | FastEmbed (ONNX) | no torch, quantized models, ~2.5 GB smaller install |
Vector store | Qdrant over **gRPC** | protobuf beats JSON float arrays 2-3x |
Hybrid search | Qdrant server-side fusion | 1 round trip instead of 2 + a Python merge |
Postgres | psycopg3 binary + pipeline | no text encode/decode of float arrays; batched round trips |
Bulk insert | `COPY` | ~10x `executemany` |
Graph writes | `UNWIND` batching | 1 round trip instead of 10k |
Vectors | numpy float32 | 4 KiB vs ~40 KiB per 1024-dim vector |

## Where the time actually goes

For a typical hybrid query with reranking, on a warm local stack. The two
CPU-inference lines are **measured on one machine** — Apple M1 Pro (10 cores),
macOS 26.6, ONNX sessions already loaded, shipped default models — because they
are the two that scale with your hardware and your chunk size, and an earlier
version of this table quoted a single small number for both:

```
embed query (bge-small, ONNX, warm session)  14-17 ms   (fresh string, so no cache hit)
Qdrant hybrid query (gRPC, server fusion)     5-25 ms
noise filters (numpy)                           <1 ms
cross-encoder rerank                        58-2300 ms  ← see the table below
context packing                                 <1 ms
LLM synthesis                              800-4000 ms  ← always the bottleneck
```

Rerank cost is a cross-encoder forward pass per candidate over *query + candidate
text*, so it is a function of both `rerank_top_k` and how big your chunks are —
one number for it is meaningless. `Xenova/ms-marco-MiniLM-L-6-v2`, p50 of 3 runs,
same machine:

```
chars per candidate       n=10     n=20     n=50
  200                     58 ms   110 ms   314 ms
  512  (chunk_size)      115 ms   246 ms   599 ms
  600                    132 ms   292 ms   690 ms
 2000  (max_chunk_size)  439 ms   936 ms  2261 ms
```

At the shipped defaults (`rerank_top_k=20`, `chunk_size=512`) that is ~250 ms, and
a corpus splitting near `max_chunk_size` with `rerank_top_k=50` pays over two
seconds. Expect a different machine to land somewhere else — measure yours with
`answer.trace` (below), which reports the rerank step directly.

Two conclusions follow, and they drive the defaults:

- **The LLM dominates**, but by less than this table used to imply. Optimizing
  retrieval below ~50 ms is still pointless next to a 2-second generation; the
  reranker is not in that category. Spend the effort on *avoiding* model calls:
  caching, the cheap tier for classifiers, and early exit.
- **Reranking is the one retrieval stage worth tuning**, and on long chunks it can
  rival the generation it feeds. It scales with `rerank_top_k` × candidate length,
  so cut whichever of the two your corpus made large.

## The dials

### Recall vs latency

```python
retrieval.fetch_k = 50  # candidates per retriever — sets the recall ceiling
retrieval.top_k = 10  # what the generator sees
retrieval.rerank_top_k = 20  # cross-encoder passes: linear cost
qdrant.hnsw_ef_search = 128  # beam width; the main recall/latency dial
```

Raising `fetch_k` costs almost nothing at the store (HNSW search is
logarithmic-ish in the limit) but costs linearly at the reranker. Raise `fetch_k`
freely; raise `rerank_top_k` deliberately.

### Memory vs accuracy

```python
qdrant.quantization = "scalar"  # int8: 4x memory cut, ~1% recall loss, SIMD-faster
qdrant.rescore = True  # re-score candidates at full precision
qdrant.oversampling = 2.0
qdrant.on_disk_vectors = False  # turn on past ~5M vectors
qdrant.quantization_always_ram = True  # what makes on-disk storage fast
```

`binary` quantization cuts 32x but only works for high-dimensional models
(≥1024) and needs oversampling. `rescore=False` with quantization on is the
configuration that quietly costs real accuracy.

### Ingest throughput

```python
indexing.batch_size = 128
indexing.max_concurrent_documents = 8
embedding.batch_size = 64  # raise to 256+ on GPU
qdrant.upsert_batch_size = 256
qdrant.parallel_upserts = 4
qdrant.wait_on_upsert = False  # fire-and-forget during bulk load
qdrant.indexing_threshold = 20_000  # or use bulk_load_mode()
```

The single biggest ingest speedup is `bulk_load_mode()`, which sets
`indexing_threshold=0` during the load and restores it afterwards. Building the
HNSW index incrementally per batch is dramatically slower than building it once at
the end.

The second biggest is `indexing.skip_unchanged`: content-derived ids plus
checksums turn a full re-ingest into a no-op.

### Postgres

```sql
maintenance_work_mem = 1GB   -- decides whether an HNSW build on 1M vectors
                             -- takes minutes or hours
random_page_cost = 1.1       -- the 4.0 default assumes spinning disks
jit = off                    -- JIT hurts on short, repetitive queries
```

`SET LOCAL hnsw.ef_search` is the per-query recall dial, mirroring Qdrant's.

## Cold start

First call downloads and initializes ONNX models. Mitigations:

- Model instances are cached per `(model, threads)` — constructing one loads an
  ONNX session and a tokenizer, far too expensive to repeat per request.
- Pre-warm on startup: embed one probe string during app init.
- Bake models into the container image rather than downloading at boot.

## Profiling what you have

Every answer carries its own trace and its own bill; there is nothing to install
around the call:

```python
rag = await build_pipeline()
answer = await rag.query("your question")

for step in answer.trace:
    print(f"{step.name:32} {step.duration_ms:8.1f} ms")

print(answer.metadata["cost"])  # the same request's spend, by model AND by stage
```

One line per stage the graph actually ran — `translate`, `route`,
`retrieve.<store>`, `rerank`, `pack_context`, `check_groundedness`, `generate` —
each with its own wall clock, plus `step.usage` where the stage called a model.

Per-stage timing is the difference between a guess and a fix: "the query took 4
seconds" becomes "reranking took 3.6 of the 4 seconds".

Wrapping the query in `new_request_context(...)` — which this page used to show —
returns an empty trace and a zero ledger: `query()` installs its own context, so
the outer one never sees the steps. See [cost.md](cost.md#reading-the-bill).

### uvloop, and when you actually get it

`install_uvloop()` sets the event-loop *policy*, and a policy is only read when a
loop is **created**. Calling it from inside a running loop cannot upgrade that
loop, so the row in the table above is only earned if you start the loop yourself:

```python
import uvloop

uvloop.run(main())  # preferred

from ragorc.core.concurrency import install_uvloop

install_uvloop()  # before the loop exists
asyncio.run(main())
```

It returns whether it took effect, and warns once (`uvloop_not_in_use`) when the
loop already running is not a uvloop one — which is what a library called from
`asyncio.run(main())` will report. The CLI and `ragorc.server` already do this
correctly; a library caller has to.

## Benchmarking retrieval strategies

```bash
make bench          # compares dense / sparse / hybrid / +rerank / +compress
```

Measure recall@k and nDCG on *your* corpus before committing to a configuration.
Every default here is a reasonable prior, not a substitute for measurement.
