# Retrieval Engine

The platform product. Everything else Halcyon Data sells either extends it or is
metered by it. Current release is **3.4**; see
[Release process](09-release-process.md) for what the version number means.

## What it does

Retrieval Engine indexes a customer's documents and answers questions over them. A
single query runs dense (semantic) and BM25 (lexical) search in one round trip
against the same index, fuses the two rankings, and reranks the survivors with a
cross-encoder before generating a cited answer.

Hybrid search is not a toggle we recommend leaving off. Dense search alone misses
part numbers, error codes and rare proper nouns, because an embedding of a token
the model never saw is an embedding of *something like that*. Lexical search alone
misses paraphrase. Customers who disable one leg and then report poor recall are
almost always missing the other one.

## Pricing and packaging

Retrieval Engine is **$1,200 per month per environment**. An environment is one
isolated index with its own credentials; most customers run two, a production and a
staging. Document storage is included. Query volume is included. **Embedding
throughput is not** — that is billed separately as
[Embedding Credits](06-embedding-credits.md).

## Limits

| Limit | Value |
|---|---|
Documents per environment | 5,000,000 |
Chunk size ceiling | 2,048 tokens |
Queries per second, sustained | 50 |
Queries per second, burst | 200 for 60 seconds |
Maximum answer length | 1,024 tokens |

The sustained ceiling is a fair-use limit rather than a hard one; Yuki Tanabe's team
raises it on request for customers on Gold or Platinum
[support plans](07-support-plans.md).

## Services behind it

Retrieval Engine is not one process. Queries arrive at the **Query Gateway**, which
Priya Raman owns; indexing runs in the **Index Service**, owned by Bea Nkemelu;
usage metering is emitted to the **Billing Service**, owned by Ravi Menon. The
[Graph Add-on](05-graph-addon.md) introduces a fourth service. On-call ownership
for all four is in [On-call](08-engineering-oncall.md).

## Compatibility

The Graph Add-on requires Retrieval Engine **3.0 or later**. Embedding Credits
metering requires **2.8 or later**; before 2.8 usage was estimated from document
counts, and customers still on those versions are billed on the old estimate until
they upgrade.
