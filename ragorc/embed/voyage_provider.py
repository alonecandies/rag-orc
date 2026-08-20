"""Voyage AI embeddings.

Why Voyage gets its own provider rather than the OpenAI-compatible path
----------------------------------------------------------------------
``input_type`` is the whole point. Voyage's models are trained with an explicit
asymmetry between the two sides of retrieval: passing ``input_type="query"``
versus ``"document"`` makes the service prepend the instruction the model was
trained with, and the published ablations put the gap at a few points of
retrieval quality. An OpenAI-shaped client has nowhere to express that, so the
asymmetry would silently be lost.

Consequently ``settings.embedding.query_prefix`` / ``document_prefix`` should
stay **empty** for this provider: the instruction is applied server-side, and a
second client-side prefix is text the model was not trained to see. The mixin
still applies whatever is configured — it is the user's dial — but the default
of "" is the correct value here.

Batch ceiling is 128 inputs (and 120k tokens) per request, well below OpenAI's
2048, so batching matters more: requests are fired concurrently under the ingest
concurrency bound rather than sequentially.

The SDK is a lazy, optional import (``ragorc[voyage]``). Its errors are mapped
into our retry taxonomy by class name so ``retry_async`` handles 429s and 5xx
with jittered backoff instead of the SDK's own policy fighting ours.
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

__all__ = ["VoyageEmbedder"]

_MAX_INPUTS = 128
"""Voyage's documented per-request input ceiling."""

_KNOWN_DIMENSIONS: dict[str, int] = {
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-code-3": 1024,
    "voyage-finance-2": 1024,
    "voyage-law-2": 1024,
    "voyage-multilingual-2": 1024,
}

_KNOWN_CONTEXT: dict[str, int] = {
    "voyage-3-large": 32_000,
    "voyage-3.5": 32_000,
    "voyage-3.5-lite": 32_000,
    "voyage-3": 32_000,
    "voyage-3-lite": 32_000,
    "voyage-code-3": 32_000,
}

_RESIZABLE_PREFIXES = ("voyage-3-large", "voyage-3.5", "voyage-code-3")
"""Models trained with Matryoshka heads, i.e. the ones that accept
``output_dimension``. Sending it elsewhere is a 400 from the API."""


@register("dense_embedder", "voyage")
class VoyageEmbedder(BaseEmbedder):
    """Asymmetric hosted embeddings via the ``voyageai`` async client."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        config = resolved.embedding
        name = model_name or config.dense_model
        super().__init__(
            model_name=name,
            dimension=config.dense_dimension or _KNOWN_DIMENSIONS.get(name, 0),
            max_tokens=_KNOWN_CONTEXT.get(name, config.max_length),
            cache=cache,
            settings=resolved,
        )
        self.batch_size = max(1, min(config.batch_size, _MAX_INPUTS))
        self.concurrency = provider_concurrency(resolved)
        self._client: Any | None = None
        self._output_dimension = (
            config.dense_dimension
            if config.dense_dimension and name.startswith(_RESIZABLE_PREFIXES)
            else None
        )
        if config.query_prefix or config.document_prefix:
            log.warning(
                "voyage_manual_prefix_configured",
                hint="voyage applies the instruction via input_type; leave the prefixes empty",
            )

    # -- transport ---------------------------------------------------------
    def _voyage_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyage embeddings need the voyageai client: pip install 'ragorc[voyage]'"
            ) from exc
        key = self.config.api_key.get_secret_value()
        if not key:
            raise ConfigError(
                "no embedding API key configured",
                provider="voyage",
                hint="set RAGORC_EMBEDDING__API_KEY",
            )
        self._client = voyageai.AsyncClient(api_key=key)
        return self._client

    # -- inference ---------------------------------------------------------
    async def _embed_batch(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        chunks = list(batched(texts, self.batch_size))
        if not chunks:
            return []
        input_type = "query" if is_query else "document"
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
        client = self._voyage_client()
        kwargs: dict[str, Any] = {
            "texts": texts,
            "model": self.model_name,
            "input_type": input_type,
            # Truncate server-side rather than erroring: one overlong chunk must
            # not fail a 128-document batch mid-ingest.
            "truncation": True,
        }
        if self._output_dimension:
            kwargs["output_dimension"] = self._output_dimension
        try:
            response = await client.embed(**kwargs)
        except Exception as exc:
            raise _map_error(exc) from exc
        embeddings = list(response.embeddings)
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                "voyage returned the wrong number of embeddings",
                requested=len(texts),
                returned=len(embeddings),
            )
        return [np.asarray(vector, dtype=np.float32) for vector in embeddings]


def _map_error(exc: BaseException) -> BaseException:
    """Map ``voyageai.error`` classes into our taxonomy by name.

    Name matching avoids importing the SDK's error module (which would defeat the
    lazy import) and survives the SDK reorganizing it.
    """
    name = type(exc).__name__
    if name == "RateLimitError":
        return RateLimited(f"voyage rate limited: {exc}")
    if name in {"ServiceUnavailableError", "Timeout", "APIConnectionError", "APIError"}:
        return TransientError(f"voyage transient failure: {exc}")
    return EmbeddingError(f"voyage embeddings failed: {exc}")
