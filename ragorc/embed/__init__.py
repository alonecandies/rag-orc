"""Embeddings: dense, sparse, late-interaction and reranking.

OpenRouter proxies chat completions only — it has **no embeddings endpoint** — so
the embedding provider is necessarily a separate choice from the LLM provider.
The default is FastEmbed (ONNX Runtime, local, no PyTorch), which is the only
provider that covers all four model kinds this library uses. See
[ADR-0004](../../docs/adr/0004-fastembed-onnx.md).

Typical use goes through the factory rather than the provider classes::

    from ragorc.embed import build_dense_embedder, build_sparse_embedder

    dense = build_dense_embedder(settings)
    sparse = build_sparse_embedder(settings)   # None when hybrid search is off

Provider modules are imported lazily. Importing this package must not load an
ONNX session, and must not require the ``openai``, ``voyageai``, ``cohere`` or
``sentence-transformers`` packages that only one provider each needs.
"""

from __future__ import annotations

from typing import Any

from ragorc.embed.base import (
    BaseEmbedder,
    apply_prefix,
    batched,
    l2_normalize,
    l2_normalize_list,
    prefix_for,
    run_in_thread,
    to_float32_matrix,
)
from ragorc.embed.cache import EmbeddingCache
from ragorc.embed.factory import (
    PROVIDERS,
    build_dense_embedder,
    build_embedding_cache,
    build_late_chunking_embedder,
    build_late_interaction_embedder,
    build_reranker,
    build_sparse_embedder,
    supports_late_chunking,
)

__all__ = [
    "PROVIDERS",
    "BaseEmbedder",
    "CohereEmbedder",
    "CohereReranker",
    "EmbeddingCache",
    "FastEmbedDense",
    "FastEmbedLateInteraction",
    "FastEmbedReranker",
    "FastEmbedSparse",
    "LateChunkingEmbedder",
    "OpenAIEmbedder",
    "STCrossEncoderReranker",
    "STEmbedder",
    "VoyageEmbedder",
    "apply_prefix",
    "batched",
    "build_dense_embedder",
    "build_embedding_cache",
    "build_late_chunking_embedder",
    "build_late_interaction_embedder",
    "build_reranker",
    "build_sparse_embedder",
    "clear_model_cache",
    "l2_normalize",
    "l2_normalize_list",
    "prefix_for",
    "resolve_strategy",
    "run_in_thread",
    "supports_late_chunking",
    "to_float32_matrix",
]

#: Attribute -> module, resolved on first access. Keeps the optional provider
#: dependencies (and every ONNX session) out of the import path.
_LAZY: dict[str, str] = {
    "FastEmbedDense": "ragorc.embed.fastembed_provider",
    "FastEmbedSparse": "ragorc.embed.fastembed_provider",
    "FastEmbedLateInteraction": "ragorc.embed.fastembed_provider",
    "FastEmbedReranker": "ragorc.embed.fastembed_provider",
    "clear_model_cache": "ragorc.embed.fastembed_provider",
    "OpenAIEmbedder": "ragorc.embed.openai_provider",
    "VoyageEmbedder": "ragorc.embed.voyage_provider",
    "CohereEmbedder": "ragorc.embed.cohere_provider",
    "CohereReranker": "ragorc.embed.cohere_provider",
    "STEmbedder": "ragorc.embed.sentence_transformers_provider",
    "STCrossEncoderReranker": "ragorc.embed.sentence_transformers_provider",
    "LateChunkingEmbedder": "ragorc.embed.late_chunking",
    "resolve_strategy": "ragorc.embed.late_chunking",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so the next access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
