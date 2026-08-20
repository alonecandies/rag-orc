"""Cohere embeddings and reranking.

Two components, both lazy imports of ``cohere`` (``ragorc[cohere]``):

**CohereEmbedder** — like Voyage, the asymmetry lives in an API field rather than
in a text prefix: ``input_type="search_query"`` for the question and
``"search_document"`` for the corpus. Cohere is stricter about it than most
providers — v3 models *require* ``input_type`` — and mixing the two sides is a
recall loss that produces no error. So the two sides are separate code paths
here, and the client-side prefixes should stay empty.

**CohereReranker** — a hosted cross-encoder. It is worth having next to the local
ONNX reranker for one reason: it reranks 100 documents in a single round trip
with no model in your process, which is the difference between reranking being
free at idle and reranking pinning a CPU. The trade is a network hop inside the
query path, so it belongs behind ``settings.retrieval.rerank_top_k`` like any
other reranker.

Both batch to Cohere's 96-input ceiling and fan the resulting requests out
concurrently under the ingest concurrency bound. Errors are mapped by class name
into our retry taxonomy so ``retry_async`` owns backoff.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather, retry_async
from ragorc.core.errors import ConfigError, EmbeddingError, RateLimited, TransientError
from ragorc.core.models import FloatArray
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.embed.base import BaseEmbedder, batched, provider_concurrency
from ragorc.embed.cache import EmbeddingCache

log = structlog.get_logger(__name__)

__all__ = ["CohereEmbedder", "CohereReranker"]

_MAX_INPUTS = 96
"""Cohere's documented per-request input ceiling."""

_KNOWN_DIMENSIONS: dict[str, int] = {
    "embed-v4.0": 1536,
    "embed-english-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-v3.0": 1024,
    "embed-multilingual-light-v3.0": 384,
}

_DEFAULT_MODEL = "embed-english-v3.0"
_DEFAULT_RERANK_MODEL = "rerank-v3.5"
_MAX_MODEL_TOKENS = 512
"""v3 embedding models truncate at 512 tokens."""


def _client(settings: Settings, provider: str) -> Any:
    try:
        import cohere
    except ImportError as exc:
        raise ImportError(
            f"cohere {provider} needs the cohere client: pip install 'ragorc[cohere]'"
        ) from exc
    key = settings.embedding.api_key.get_secret_value()
    if not key:
        raise ConfigError(
            "no embedding API key configured",
            provider="cohere",
            hint="set RAGORC_EMBEDDING__API_KEY",
        )
    # V2 is the current surface; it is the one that returns typed embedding
    # buckets (`embeddings.float_`) instead of an untagged list.
    return cohere.AsyncClientV2(api_key=key)


def _map_error(exc: BaseException) -> BaseException:
    name = type(exc).__name__
    if name in {"TooManyRequestsError", "RateLimitError"}:
        return RateLimited(f"cohere rate limited: {exc}")
    if name in {
        "ServiceUnavailableError",
        "InternalServerError",
        "GatewayTimeoutError",
        "ApiError",
    }:
        status = getattr(exc, "status_code", None)
        if not isinstance(status, int) or status >= 500 or status in (408, 429):
            return TransientError(f"cohere transient failure: {exc}", status=status)
    return EmbeddingError(f"cohere call failed: {exc}")


@register("dense_embedder", "cohere")
class CohereEmbedder(BaseEmbedder):
    """Hosted dense embeddings with API-level query/document asymmetry."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        config = resolved.embedding
        # The default `dense_model` is a FastEmbed name; if the user switched the
        # provider without switching the model, a Cohere model name is the only
        # thing that can work.
        name = model_name or (
            config.dense_model if config.dense_model.startswith("embed-") else _DEFAULT_MODEL
        )
        if name != config.dense_model:
            log.info("cohere_model_defaulted", configured=config.dense_model, using=name)
        super().__init__(
            model_name=name,
            dimension=config.dense_dimension or _KNOWN_DIMENSIONS.get(name, 0),
            max_tokens=min(config.max_length or _MAX_MODEL_TOKENS, _MAX_MODEL_TOKENS),
            cache=cache,
            settings=resolved,
        )
        self.batch_size = max(1, min(config.batch_size, _MAX_INPUTS))
        self.concurrency = provider_concurrency(resolved)
        self._cohere: Any | None = None
        if config.query_prefix or config.document_prefix:
            log.warning(
                "cohere_manual_prefix_configured",
                hint="cohere applies the instruction via input_type; leave the prefixes empty",
            )

    def _api(self) -> Any:
        if self._cohere is None:
            self._cohere = _client(self.settings, "embeddings")
        return self._cohere

    async def _embed_batch(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        chunks = list(batched(texts, self.batch_size))
        if not chunks:
            return []
        input_type = "search_query" if is_query else "search_document"
        results = await bounded_gather(
            (self._embed_one_request(chunk, input_type) for chunk in chunks),
            limit=self.concurrency,
        )
        vectors: list[Any] = []
        for part in results:
            vectors.extend(part)
        return self._finalize(vectors)

    @retry_async(max_attempts=4, retry_on=(TransientError,))
    async def _embed_one_request(self, texts: list[str], input_type: str) -> list[FloatArray]:
        client = self._api()
        try:
            response = await client.embed(
                texts=texts,
                model=self.model_name,
                input_type=input_type,
                # Ask for float explicitly: without it v2 may return int8 or
                # binary buckets and `float_` comes back empty.
                embedding_types=["float"],
                truncate="END",
            )
        except Exception as exc:
            raise _map_error(exc) from exc
        vectors = getattr(response.embeddings, "float_", None) or getattr(
            response.embeddings, "float", None
        )
        if not vectors:
            raise EmbeddingError(
                "cohere returned no float embeddings", model=self.model_name, requested=len(texts)
            )
        vectors = list(vectors)
        if len(vectors) != len(texts):
            raise EmbeddingError(
                "cohere returned the wrong number of embeddings",
                requested=len(texts),
                returned=len(vectors),
            )
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]


@register("reranker", "cohere")
class CohereReranker:
    """Hosted cross-encoder reranking via Cohere's ``/rerank``.

    Returns ``(index, relevance_score)`` against the input order, sorted by score
    descending. Cohere's scores are already normalized to ``[0, 1]`` — unlike a
    local cross-encoder's raw logits — which makes them directly usable as a
    threshold in ``settings.retrieval.relative_score_cutoff``.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configured = self.settings.embedding.reranker_model
        self.model_name = model_name or (
            configured if configured.startswith("rerank") else _DEFAULT_RERANK_MODEL
        )
        self._cohere: Any | None = None

    def _api(self) -> Any:
        if self._cohere is None:
            self._cohere = _client(self.settings, "rerank")
        return self._cohere

    @retry_async(max_attempts=4, retry_on=(TransientError,))
    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        docs = list(documents)
        if not docs:
            return []
        client = self._api()
        try:
            response = await client.rerank(
                model=self.model_name,
                query=query,
                documents=docs,
                top_n=min(top_k, len(docs)) if top_k else len(docs),
            )
        except Exception as exc:
            raise _map_error(exc) from exc
        # The API already returns results sorted by relevance; sorting again is
        # cheap and makes the ordering guarantee ours.
        pairs = [(int(item.index), float(item.relevance_score)) for item in response.results]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs
