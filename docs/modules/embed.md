# `ragorc.embed` — three kinds of vector, plus late chunking

Hybrid search needs three representations of the same text, so there are three
embedder protocols rather than one. All of them run on **ONNX Runtime via
FastEmbed** by default, which is why the base install has no PyTorch.

Related: [ADR-0004 — FastEmbed/ONNX](../adr/0004-fastembed-onnx.md) ·
[ADR-0002 — late chunking](../adr/0002-late-chunking.md).

## Key classes

```python
FastEmbedDense(model_name=None, *, cache: EmbeddingCache | None = None, settings=None)
    dimension: int; model_name: str; max_tokens: int
    async embed_documents(texts) -> list[FloatArray]
    async embed_query(text) -> FloatArray
    async embed_queries(texts) -> list[FloatArray]
    async warmup() -> None                      # loads the ONNX session and pins `dimension`

FastEmbedSparse(...)            # BM25 or SPLADE -> SparseVector(indices, values)
FastEmbedLateInteraction(...)   # ColBERT, (n_tokens, dim); also the late-chunking backend
FastEmbedReranker(...)          # ONNX cross-encoder: rerank(query, documents, top_k=None)
```

`embed_documents` and `embed_query` are separate methods because asymmetric models
(E5, BGE, GTE) need different instruction prefixes on the two sides; using one call
for both is a silent recall loss, which is why the prefixes are explicit settings.

Hosted providers satisfy the same protocol: `OpenAIEmbedder`, `VoyageEmbedder`,
`CohereEmbedder` (plus `CohereReranker`), and `STEmbedder` /
`STCrossEncoderReranker` for sentence-transformers — each behind its own extra and
each registered under its provider name, so `embedding.provider` selects it.

```python
LateChunkingEmbedder(token_embedder=None, settings=None)
    supports_token_embeddings: bool
    async embed_chunks(document_text: str, spans: list[tuple[int, int]]) -> list[FloatArray]
    async embed_query(text) -> FloatArray

async resolve_strategy(requested: ChunkingStrategy, embedder, settings=None) -> ChunkingStrategy
EmbeddingCache(backend: Cache, settings=None)   # content-hash keyed, dense/sparse/multi
```

`resolve_strategy` implements the ADR-0002 ladder — `LATE -> CONTEXTUAL -> EARLY` —
once per ingest run, and logs which rung it landed on and why. It never downgrades
silently: a corpus indexed with the wrong strategy looks fine and retrieves badly.

## Why late chunking is preferred — and why you are probably not running it

One forward pass over the whole document, then mean-pool the token vectors over
each chunk's exact `start_char`/`end_char`. A chunk reading *"It is billed at $450
per month per seat"* keeps who *it* is, because its vector was computed in the
document's context. One pass per document also beats one pass per chunk on cost, so
this is the rare change that is both better and cheaper.

It is **not the default**, because the default embedder cannot do it: FastEmbed's
`TextEmbedding` returns pooled vectors only, so with the zero-dependency install
`resolve_strategy` logs `chunking_strategy_resolved chosen=early
reason=no_better_option` and `supports_late_chunking()` returns `False` — and an
explicit `chunking_strategy=late` is downgraded the same way, with
`late_chunking_requested_but_unavailable`. Install `ragorc[local]` with a
token-capable model (`jinaai/jina-embeddings-v2-base-en`) to get `late`
([ADR-0002](../adr/0002-late-chunking.md)).

`examples/corpus/05-graph-addon.md` was written to demonstrate that failure — its
pricing sentence starts with a pronoun whose antecedent is paragraphs earlier — but
it does not: with the shipped (early) configuration over the ingested example
corpus, "how much does the Graph Add-on cost" retrieves that chunk at **rank 1**
(dense score 0.80), because the `## Pricing` heading inside the same chunk carries
the signal the pronoun lost. Treat the corpus as a smoke test, not as evidence for
the technique.

## Usage

```python
from ragorc.cache import build_cache
from ragorc.embed.cache import EmbeddingCache
from ragorc.embed.fastembed_provider import FastEmbedDense, FastEmbedSparse

cache = EmbeddingCache(build_cache())
dense = FastEmbedDense(cache=cache)
sparse = FastEmbedSparse(cache=cache)
await dense.warmup()  # pin the dimension before creating a collection

vectors = await dense.embed_documents([c.content for c in chunks])
qv = await dense.embed_query("what is the SEV-1 response time?")
```

Build the embedders **once per process** and inject the same objects into the
ingest pipeline and the retriever: two instances mean two loaded ONNX sessions.

## Settings

| Setting | Effect |
|---|---|
`embedding.provider` | `fastembed` (default), `openai`, `voyage`, `cohere`, `sentence_transformers` |
`embedding.dense_model` · `dense_dimension` | dimension is auto-detected when unset |
`embedding.query_prefix` · `document_prefix` | required by asymmetric models |
`embedding.sparse_model` · `use_splade` · `splade_model` | BM25-as-sparse by default |
`embedding.late_interaction_model` · `enable_late_interaction` | ColBERT is ~100x the storage; worth it as a rerank stage, rarely as an index |
`embedding.reranker_model` | ONNX cross-encoder, no torch |
`embedding.batch_size` · `max_length` · `threads` | tuned for ONNX on CPU; raise batch size on GPU |
`embedding.normalize` | L2 so cosine reduces to a dot product |
`embedding.cache_embeddings` | content-hash cache; a re-ingest of unchanged text costs nothing |
`indexing.chunking_strategy` | `auto` resolves through the ADR-0002 ladder |
