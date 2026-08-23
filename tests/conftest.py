"""Shared fixtures.

Every fixture here is offline. The only fixture that touches a service is behind
the ``integration`` marker, which is deselected by default in
``pyproject.toml``'s ``addopts``.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from ragorc.core.models import Chunk, Document, Query, ScoredChunk
from ragorc.core.settings import Settings, get_settings
from tests.fakes import (
    FakeCache,
    FakeGraphStore,
    FakeRelationalStore,
    FakeVectorStore,
    StubEmbedder,
    StubLLM,
    StubReranker,
    StubSparseEmbedder,
)

_REAL_CONNECT = socket.socket.connect
_REAL_CONNECT_EX = socket.socket.connect_ex


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest):
    """Fail a unit test that opens a real TCP connection *from Python*.

    The design goal in `tests/fakes` — "the entire unit suite runs with no
    network, no containers, no API keys and no model downloads" — was a promise
    in a docstring, and two tests had quietly stopped keeping it.

    **This is a partial net, and the gap is worth knowing.** It patches
    `socket.socket.connect`, so it only sees clients that connect through Python.
    Measured against this stack: an HTTP Qdrant client is caught; gRPC — which is
    the Qdrant default — and psycopg both connect inside C and sail straight
    past. So it will catch a stray OpenRouter call or an `httpx` request, and it
    would *not* have caught the two ingest tests that motivated it.

    What catches those is
    `test_the_offline_pipeline_helper_injects_its_stores`: a direct assertion on
    the helper, which is where the mistake is actually made. Keep both — this one
    is cheap (no measurable effect on suite runtime) and covers the case that
    assertion cannot, namely a test reaching a *hosted* API.

    Only AF_INET/AF_INET6 are blocked: `socket.socketpair()` is AF_UNIX and is how
    asyncio builds its self-pipe, so blocking that would break the event loop
    itself. Integration tests are exempt — reaching a service is their purpose.
    """
    if "integration" in request.keywords:
        yield
        return

    def _blocked(self, address, *args, **kwargs):  # noqa: ANN001, ANN202
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(
                f"unit test opened a real connection to {address!r}. "
                "Inject a fake store or embedder instead — see tests/fakes."
            )
        return _REAL_CONNECT(self, address, *args, **kwargs)

    def _blocked_ex(self, address, *args, **kwargs):  # noqa: ANN001, ANN202
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError(f"unit test opened a real connection to {address!r}")
        return _REAL_CONNECT_EX(self, address, *args, **kwargs)

    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked_ex
    try:
        yield
    finally:
        socket.socket.connect = _REAL_CONNECT
        socket.socket.connect_ex = _REAL_CONNECT_EX


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Settings are an lru_cached singleton; a test that changes them must not
    leak that change into the next test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Offline-safe defaults: tenant isolation off (most tests are single-tenant)
    and caching off (so tests measure the code, not the cache)."""
    return Settings(
        environment="dev",
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "test-key", "max_concurrency": 4},
        embedding={"dense_dimension": 32},
    )


@pytest.fixture
def llm() -> StubLLM:
    return StubLLM()


@pytest.fixture
def embedder() -> StubEmbedder:
    return StubEmbedder(dimension=32)


@pytest.fixture
def sparse_embedder() -> StubSparseEmbedder:
    return StubSparseEmbedder()


@pytest.fixture
def reranker() -> StubReranker:
    return StubReranker()


@pytest.fixture
def cache() -> FakeCache:
    return FakeCache()


@pytest.fixture
def vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def relational_store() -> FakeRelationalStore:
    return FakeRelationalStore()


@pytest.fixture
def graph_store() -> FakeGraphStore:
    return FakeGraphStore()


@pytest.fixture
def document() -> Document:
    return Document(
        id="doc-1",
        content=(
            "Refunds are processed within 14 days of the request. "
            "Shipping is free on orders above 50 USD. "
            "Enterprise plans include a dedicated account manager. "
            "Support replies within one business day on weekdays."
        ),
        source="policy.md",
        title="Customer Policy",
        checksum="abc123",
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    texts = [
        "Refunds are processed within 14 days of the request.",
        "Shipping is free on orders above 50 USD.",
        "Enterprise plans include a dedicated account manager.",
        "Support replies within one business day on weekdays.",
    ]
    out = []
    for i, text in enumerate(texts):
        chunk = Chunk(id=f"c{i}", content=text, document_id="doc-1", index=i)
        rng = np.random.default_rng(i)
        vector = rng.normal(size=32).astype(np.float32)
        chunk.dense = (vector / np.linalg.norm(vector)).astype(np.float32)
        out.append(chunk)
    return out


@pytest.fixture
def scored(chunks: list[Chunk]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=c, score=0.9 - i * 0.15, rank=i) for i, c in enumerate(chunks)]


@pytest.fixture
def query() -> Query:
    return Query(text="How long do refunds take?", top_k=3)
