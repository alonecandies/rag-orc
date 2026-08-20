"""OpenAI embeddings over httpx, with the SDK as an optional accelerator.

Why httpx and not a hard dependency on ``openai``
-------------------------------------------------
The embeddings endpoint is one POST with a JSON body. Taking the SDK as a base
dependency would add httpx-plus-pydantic-plus-anyio machinery we already have,
its own retry policy competing with ours, and a version constraint on a library
that ships breaking changes regularly — to save about thirty lines. So the
transport here is our own ``httpx.AsyncClient`` (HTTP/2, so concurrent batches
multiplex over one connection) and the SDK is used only when it is already
installed, for users who need its org/proxy/Azure plumbing.

Three details that are worth the code:

**``encoding_format="base64"``.** The default response encodes each vector as a
JSON array of decimal floats: a 3072-dim ``text-embedding-3-large`` vector
becomes ~40 KB of text that has to be parsed into 3072 Python floats before it
can become an array. Base64 is one ``b64decode`` plus one ``frombuffer`` —
roughly 5x cheaper and ~3x less bytes on the wire. The SDK does the same thing
internally for exactly this reason.

**Re-normalization when ``dimensions`` is set.** OpenAI's v3 models are trained
with Matryoshka representation learning, so you can ask for a shorter vector and
keep most of the quality. The truncated vector is *not* unit-norm — the
documented procedure is to renormalize client-side, and skipping it breaks the
assumption that cosine similarity equals a dot product everywhere downstream.

**Result reordering by ``index``.** The API documents that ``data`` comes back in
request order; sorting by the ``index`` field anyway costs nothing and makes a
silent misalignment of vectors to chunks impossible.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
import orjson
import structlog

from ragorc.core.concurrency import bounded_gather, retry_async
from ragorc.core.errors import ConfigError, EmbeddingError, RateLimited, TransientError
from ragorc.core.models import FloatArray
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.embed.base import BaseEmbedder, batched, provider_concurrency
from ragorc.embed.cache import EmbeddingCache

log = structlog.get_logger(__name__)

__all__ = ["OpenAIEmbedder"]

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_MAX_INPUTS = 2048
"""Hard API ceiling on inputs per request. ``settings.embedding.batch_size`` may
lower this (memory, or a proxy with a smaller body limit) but never raise it."""

_MAX_MODEL_TOKENS = 8191
"""Per-input token limit shared by every current OpenAI embedding model."""

_KNOWN_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
"""Native output sizes, so the vector store can create its collection before the
first request. A model absent from this table is resolved by ``warmup()``."""

_MATRYOSHKA_PREFIX = "text-embedding-3"
"""Only the v3 family accepts the ``dimensions`` parameter."""

_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})


@register("dense_embedder", "openai")
class OpenAIEmbedder(BaseEmbedder):
    """Hosted dense embeddings from OpenAI (or any OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        base_url: str | None = None,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved = settings or get_settings()
        config = resolved.embedding
        name = model_name or config.dense_model
        dimension = config.dense_dimension or _KNOWN_DIMENSIONS.get(name, 0)
        super().__init__(
            model_name=name,
            dimension=dimension,
            max_tokens=min(config.max_length, _MAX_MODEL_TOKENS)
            if config.max_length
            else _MAX_MODEL_TOKENS,
            cache=cache,
            settings=resolved,
        )
        self.base_url = (base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.batch_size = max(1, min(config.batch_size, _MAX_INPUTS))
        self.concurrency = provider_concurrency(resolved)
        self._client = client
        self._owns_client = client is None
        self._sdk: Any | None = None
        self._sdk_checked = False

        # Only v3 honours `dimensions`; sending it to ada-002 is a 400.
        self._request_dimensions = (
            config.dense_dimension
            if config.dense_dimension and name.startswith(_MATRYOSHKA_PREFIX)
            else None
        )
        if config.dense_dimension and self._request_dimensions is None:
            log.warning(
                "embedding_dimensions_unsupported",
                model=name,
                requested=config.dense_dimension,
                hint="only text-embedding-3-* can be truncated",
            )

    # -- transport ---------------------------------------------------------
    def _api_key(self) -> str:
        key = self.config.api_key.get_secret_value()
        if not key:
            raise ConfigError(
                "no embedding API key configured",
                provider="openai",
                hint="set RAGORC_EMBEDDING__API_KEY",
            )
        return key

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self._api_key()}",
                    "Content-Type": "application/json",
                },
                http2=True,
                # There is no separate embedding timeout knob; the model-call
                # timeout is the right order of magnitude and keeps one dial.
                timeout=httpx.Timeout(
                    self.settings.llm.timeout_s, connect=self.settings.llm.connect_timeout_s
                ),
                limits=httpx.Limits(
                    max_connections=self.concurrency * 2,
                    max_keepalive_connections=self.concurrency,
                    keepalive_expiry=60.0,
                ),
            )
        return self._client

    def _sdk_client(self) -> Any | None:
        """The official SDK, if the user already has it installed.

        Checked once. Its presence buys Azure/proxy/organization handling that
        would be busywork to reimplement; its absence costs nothing because the
        httpx path is the primary one.
        """
        if self._sdk_checked:
            return self._sdk
        self._sdk_checked = True
        try:
            from openai import AsyncOpenAI
        except ImportError:
            log.debug("openai_sdk_absent", hint="pip install 'ragorc[openai]' to use it")
            return None
        self._sdk = AsyncOpenAI(api_key=self._api_key(), base_url=self.base_url)
        return self._sdk

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
        if self._sdk is not None:
            await self._sdk.close()
            self._sdk = None

    async def __aenter__(self) -> OpenAIEmbedder:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- inference ---------------------------------------------------------
    async def _embed_batch(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        """Split to the API ceiling, fire the requests concurrently, reassemble.

        Order is preserved because ``bounded_gather`` preserves it; that is what
        lets the caller zip vectors back onto chunks.
        """
        chunks = list(batched(texts, self.batch_size))
        if not chunks:
            return []
        results = await bounded_gather(
            (self._embed_one_request(chunk) for chunk in chunks), limit=self.concurrency
        )
        vectors: list[list[float] | FloatArray] = []
        for part in results:
            vectors.extend(part)
        return self._finalize(vectors)

    async def _embed_one_request(self, texts: list[str]) -> list[FloatArray]:
        sdk = self._sdk_client()
        if sdk is not None:
            return await self._via_sdk(sdk, texts)
        payload = await self._post(texts)
        rows = sorted(payload.get("data") or [], key=lambda row: int(row.get("index", 0)))
        if len(rows) != len(texts):
            raise EmbeddingError(
                "openai returned the wrong number of embeddings",
                requested=len(texts),
                returned=len(rows),
            )
        return [_decode_embedding(row.get("embedding")) for row in rows]

    def _body(self, texts: list[str]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "base64",
        }
        if self._request_dimensions:
            body["dimensions"] = self._request_dimensions
        return body

    @retry_async(max_attempts=4, retry_on=(TransientError,))
    async def _post(self, texts: list[str]) -> dict[str, Any]:
        try:
            response = await self.client.post(
                "/embeddings", content=orjson.dumps(self._body(texts))
            )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            raise TransientError(f"openai embeddings transport failure: {exc}") from exc

        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimited(
                "openai embeddings rate limited",
                retry_after=float(retry_after) if retry_after else None,
                body=response.text[:300],
            )
        if response.status_code in _RETRY_STATUS:
            raise TransientError(
                f"openai embeddings returned {response.status_code}",
                status=response.status_code,
                body=response.text[:300],
            )
        if response.status_code >= 400:
            raise EmbeddingError(
                f"openai embeddings returned {response.status_code}",
                status=response.status_code,
                body=response.text[:500],
            )
        return orjson.loads(response.content)

    @retry_async(max_attempts=4, retry_on=(TransientError,))
    async def _via_sdk(self, sdk: Any, texts: list[str]) -> list[FloatArray]:
        kwargs: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self._request_dimensions:
            kwargs["dimensions"] = self._request_dimensions
        try:
            response = await sdk.embeddings.create(**kwargs)
        except Exception as exc:
            # The SDK's exception classes are not importable without importing
            # the SDK, so they are mapped by name into our retry taxonomy.
            raise _map_sdk_error(exc) from exc
        rows = sorted(response.data, key=lambda row: row.index)
        return [np.asarray(row.embedding, dtype=np.float32) for row in rows]


def _decode_embedding(value: Any) -> FloatArray:
    """Base64 float32 (our request format) or a JSON float array (a proxy that
    ignored ``encoding_format``). Both shapes appear in the wild."""
    if isinstance(value, str):
        return np.frombuffer(base64.b64decode(value), dtype=np.float32).copy()
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    raise EmbeddingError(
        "openai returned an embedding of unexpected type", got=type(value).__name__
    )


def _map_sdk_error(exc: BaseException) -> BaseException:
    """Translate SDK exceptions into our retry taxonomy by class name.

    Matching on names rather than importing the exception classes keeps this
    working across SDK majors, where the module layout moves but the names do
    not.
    """
    name = type(exc).__name__
    if name == "RateLimitError":
        return RateLimited(f"openai embeddings rate limited: {exc}")
    if name in {"APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return TransientError(f"openai embeddings transient failure: {exc}")
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status in _RETRY_STATUS:
        return TransientError(f"openai embeddings returned {status}", status=status)
    return EmbeddingError(f"openai embeddings failed: {exc}")
