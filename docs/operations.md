# Operations

Running `ragorc` beyond a laptop: scaling, deployment, monitoring, failure modes.

## Scaling the three stores

### Qdrant

| Corpus | Configuration |
|---|---|
< 100k vectors | defaults are fine; below `full_scan_threshold` Qdrant brute-forces and beats HNSW anyway |
100k - 1M | `quantization="scalar"`, `hnsw_m=16`, `default_segment_number` = CPU count |
1M - 10M | `on_disk_payload=true`, `hnsw_m=32`, raise `hnsw_ef_construct` to 256 |
> 10M | `on_disk_vectors=true` **with** `quantization_always_ram=true`, shard (`shard_number` > 1), consider `binary` quantization for ≥1024-dim models |

Zero-downtime reindex: build into a new collection with `--force`, then
`swap_alias()`. Never mutate a live collection's vector config — it requires a
rebuild.

`--force` is not optional here. The checksum skip asks Postgres whether a
document was ingested; it has no notion of which Qdrant collection is being
written, so an unchanged corpus re-ingested into a *new* collection skips every
document and reports success — and the alias then points at an empty index. If it
is forgotten, the ingest says so: skipping everything into an empty collection is
reported on `IngestReport.warnings`.

Multi-tenancy: keep one collection with a `tenant_id` payload index
(`is_tenant=True`). One collection per tenant does not scale past a few hundred
tenants — each carries its own HNSW graph and segment overhead.

### Postgres

The chunks table grows fastest. In order of when you will need them:

1. `maintenance_work_mem=1GB` before building an HNSW index on >100k rows.
2. Partition by `tenant_id` or by month past ~50M chunks.
3. Read replicas for the Text-to-SQL path — generated queries are unpredictable
   and belong away from your write path.
4. `pg_search` (ParadeDB) if lexical quality matters; `ts_rank_cd` is cover
   density, not BM25.

### Neo4j

Graph traversal is page-cache bound. `server_memory_pagecache_size` should exceed
the store size on disk; if it does not, traversals thrash and GraphRAG local search
becomes the slowest thing in the pipeline.

Community detection is a batch job, not a request-path operation. Run it after
ingest, on a schedule — not per query.

## Deployment

```
             ┌──────────────┐
   clients → │  API workers │ → Qdrant (gRPC)
             │  (uvicorn)   │ → Postgres (pgbouncer → primary + replica)
             └──────┬───────┘ → Neo4j
                    │
                 Redis  ← shared cache tier: mandatory with >1 worker
```

- **Workers**: async, so one worker per 2 cores is usually right. The LLM call
  dominates, so workers are mostly waiting — concurrency, not CPU, is the limit.
- **Redis is not optional above one worker.** Without it the semantic cache hit
  rate divides by the worker count.
- **Bake ONNX models into the image.** Downloading on first request adds seconds
  to a cold pod and can fail on a locked-down network.
- **Ingest is a separate workload.** It is CPU-bound (embedding) where serving is
  IO-bound. Run it as a job, not on the API workers.

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir "ragorc[server,redis,raptor,graphrag,loaders]"
# Pre-download the models so the first request does not pay for it
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
    TextEmbedding('BAAI/bge-small-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"
ENV RAGORC_ENVIRONMENT=prod
CMD ["uvicorn", "ragorc.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Monitoring

Watch these five. They are the ones that predict incidents.

| Signal | Where | Alert when |
|---|---|---|
`cache_hit_rate` | `CostLedger.report()` | drops sharply — usually a Redis outage or a prompt change invalidating keys |
`cost_usd` p95 per query | ledger | rising — a loop is retrying more, or the cascade is escalating more |
`abstained` rate | `Answer.abstained` | rising — index drift or a corpus gap, *not* a bug in generation |
`groundedness` p50 | `Answer.groundedness` | falling — retrieval quality regressed |
per-store `errors` | `RetrievalResult.errors` | non-empty — a circuit breaker is open and answers are silently degraded |

That last one deserves emphasis: **degradation is silent by design.** A dead store
is recorded in `RetrievalResult.errors` and the query proceeds with the others.
That is correct behaviour and a monitoring obligation — without an alert on it,
you will serve worse answers for days without noticing.

```python
answer.metadata["budget"]  # window, used, dropped chunks, overflow strategy
answer.metadata["validation"]  # citation validity, coverage, warnings
answer.trace  # per-stage timings
```

## When the stack is up but nothing connects

This failure signature is deceptive enough to deserve its own section, because
every ordinary check says the system is fine:

- `docker compose ps` reports all three containers **healthy**
- `lsof -iTCP:6333` shows the port **LISTENing**
- `nc -z localhost 6333` **succeeds**

…and yet every client times out. The reason all three mislead: container
healthchecks run *inside* the container, and Docker's port forwarder binds and
accepts the TCP connection before it can actually route it.

The distinguishing test is whether a **sibling container** can reach the service.
`make doctor` runs it:

```bash
make doctor
```

| Sibling reaches it | Host reaches it | Meaning |
|---|---|---|
yes | yes | healthy |
**yes** | **no** | **Docker's port forwarder is wedged** — restart Docker Desktop. Typically follows the machine sleeping. Not a configuration problem. |
no | no | the service itself is not ready — `docker compose logs` |

Because container-to-container traffic is unaffected, the integration suite stays
runnable through it:

```bash
make test-docker    # integration tests from inside the compose network
make test-e2e       # real embeddings, real stores, one real OpenRouter call
```

That is also how CI must run them — a runner reaches the services by compose
service name, never through published host ports — so these are the configurations
that get exercised on every push, not a workaround.

### Port collisions

A hardcoded host port collides with whatever the developer already runs. All of
them are overridable; remember to update the matching DSN:

```bash
RAGORC_PG_PORT=5433 docker compose up -d
```

## Failure modes and what they mean

| Symptom | Likely cause | Fix |
|---|---|---|
Answers ignore an obviously relevant document | `fetch_k` too low, or the term is lexical and sparse search is off | raise `fetch_k`; enable `use_sparse` |
Answers cite the right document but the wrong detail | chunks too large; the model is picking from a crowded passage | lower `chunk_size`, enable reranking |
Retrieval returns near-identical chunks | overlapping chunks and no dedupe | `dedupe_enabled`, lower `near_dupe_threshold`, consider MMR |
A chunk mentions "it" and is never retrieved | early chunking lost the antecedent | switch to a long-context embedder so `LATE` resolves |
`BudgetExceeded` on normal queries | a loop is retrying; groundedness threshold too strict | inspect `ledger.report()["by_stage"]` |
`StructuredOutputError` on one model | the provider ignores `response_format` | `require_parameters=true`, or pin `provider_order` |
Abstention rate spikes after a reindex | vectors written with a different embedding model | dimensions must match; rebuild the collection |
Reindex reported success but the new collection is empty | the checksum skip dropped every document — it knows Postgres, not which collection you are writing | re-run `ragorc ingest --force`; the report now warns when this happens |
`Abort trap: 6` / exit 134 at process exit, with a macOS crash report naming `Microsoft::Applications::Events` | ONNX Runtime's bundled telemetry client races static destruction while sessions are still alive | already mitigated — `ORT_DISABLE_TELEMETRY=1` plus an `atexit` session release in `ragorc/embed/_runtime.py`. The work had completed; only the exit was dirty |
Text-to-SQL always refuses, rule `generated_query_isolation` | tenant isolation is on and the deployment has not declared how tenants are separated — the default, and deliberately fail-closed | set `security.generated_query_isolation` to `rls` (recommended), `database` or `trusted`; see docs/security.md |
Text-to-SQL always refuses, other rules | `postgres.allowed_tables` too narrow, or the schema summary is stale | widen the allowlist; `schema_summary(refresh=True)` |
Graph search finds nothing | entity resolution did not run, so the graph is fragmented | `graph.resolve_entities=true`, lower `resolution_threshold` |
p99 latency tracks the slowest store | `per_store_timeout_s` too high | lower it; a slow store should be dropped, not waited for |

## Reindexing

Changing the embedding model, dimension, or chunking strategy requires a rebuild —
vectors from different models are not comparable, and a dimension mismatch is an
opaque insert error much later.

```bash
ragorc init --collection ragorc_v2      # new collection, new config
ragorc ingest ./corpus --collection ragorc_v2
ragorc alias-swap ragorc ragorc_v2      # atomic cutover
```

## Backups

- **Qdrant**: snapshot API, or back up the storage volume. Snapshots are
  point-in-time and safe while serving.
- **Postgres**: normal `pg_dump` / WAL archiving. This is your source of truth for
  chunk text — treat it as the durable copy.
- **Neo4j**: `neo4j-admin database dump`. Cheap to rebuild from Postgres chunks if
  you would rather re-extract than restore.

The vector index is *derived data*. If Postgres holds the chunks, Qdrant and Neo4j
can both be rebuilt — which is the right way to think about a restore.
