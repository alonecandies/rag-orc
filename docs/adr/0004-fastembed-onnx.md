# ADR-0004: FastEmbed/ONNX for embeddings; no PyTorch in the base install

**Status:** accepted · **Date:** 2026-08-19

## Context

OpenRouter proxies chat completions only — **it has no embeddings endpoint**. So
the embedding provider is necessarily a separate decision from the LLM provider,
and RAG needs four distinct kinds of model:

1. dense embeddings, 2. sparse/BM25 embeddings, 3. late-interaction (ColBERT)
embeddings, 4. a cross-encoder reranker.

The obvious default, `sentence-transformers`, pulls PyTorch: ~2.5 GB installed,
slow cold start, and CUDA/MPS variance across machines. For a library meant to be
embedded in other projects, that is a heavy tax to impose by default.

## Decision

Default to **FastEmbed** (ONNX Runtime), and put torch behind the `[local]`
extra.

FastEmbed covers all four model kinds with one dependency and no torch:
`TextEmbedding`, `SparseTextEmbedding` (BM25 **and** SPLADE),
`LateInteractionTextEmbedding` (ColBERT), `TextCrossEncoder` (reranking). It also
ships quantized models, so CPU inference is genuinely fast rather than merely
possible — and it is maintained by Qdrant, so the output types line up with the
vector store's input types.

Default model: `BAAI/bge-small-en-v1.5` (384-dim). Small, strong for its size,
and 384 dimensions keeps both the Qdrant index and the pgvector column compact.

Hosted providers (OpenAI, Voyage, Cohere) are first-class alternatives behind the
`EmbeddingProvider` protocol, selected by one setting.

## Consequences

- `pip install ragorc` works offline, needs no API key for embeddings, and runs
  in CI without a GPU.
- Choosing a hosted provider **disables late chunking** (they return pooled
  vectors only) — the ladder in ADR-0002 handles this automatically.
- ONNX inference is CPU-bound and holds the GIL in places, so every embed call
  goes through `asyncio.to_thread`. Skipping that would stall the event loop for
  the duration of a batch and serialize the whole pipeline.
- Model instances are cached per `(model, threads)`: constructing one loads an
  ONNX session and a tokenizer, which is far too expensive to repeat per request.
