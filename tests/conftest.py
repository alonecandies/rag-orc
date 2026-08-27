"""Shared fixtures.

Every fixture here is offline. The only fixture that touches a service is behind
the ``integration`` marker, which is deselected by default in
``pyproject.toml``'s ``addopts``.
"""

from __future__ import annotations

import os
import pathlib
import socket

import numpy as np
import pytest

#: Where tiktoken keeps the BPE file it downloads. Set before anything imports
#: tiktoken, because it reads this at import time.
#:
#: Its default is a directory under ``tempfile.gettempdir()``, which on macOS is
#: a per-boot path under ``/var/folders`` that the OS periodically purges. So the
#: unit suite passed or failed depending on whether *some earlier process on this
#: machine* had warmed a cache that nothing in the repo owns: a full green run,
#: then five failures an hour later with no change in between, all reported as
#: "unit test opened a real connection". Pinning it inside the repo makes the
#: download happen once per clone instead of once per temp-dir purge.
_CACHE_ROOT = pathlib.Path(__file__).resolve().parent.parent / ".cache"
_TIKTOKEN_CACHE = _CACHE_ROOT / "tiktoken"
_TIKTOKEN_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_TIKTOKEN_CACHE))

#: The same problem one library over. FastEmbed defaults to ``fastembed_cache``
#: in the system temp directory, so four ingest tests downloaded an ONNX model
#: whenever that directory had been purged — and then failed inside the network
#: guard, whose message ("inject a fake store or embedder") describes neither the
#: cause nor the fix.
_FASTEMBED_CACHE = _CACHE_ROOT / "fastembed"
_FASTEMBED_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(_FASTEMBED_CACHE))

# Imported after the environment is set, not before: `tiktoken` reads
# TIKTOKEN_CACHE_DIR at import time and `ragorc` pulls it in transitively.
from ragorc.core.models import Chunk, Document, Query, ScoredChunk  # noqa: E402
from ragorc.core.settings import Settings, get_settings  # noqa: E402
from tests.fakes import (  # noqa: E402
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


@pytest.fixture(scope="session", autouse=True)
def _tokenizer_cache() -> bool:
    """Warm the BPE cache once per session, before the network guard is armed.

    The guard is function-scoped and this is session-scoped, so this runs first
    and with real sockets — which is the only honest way to keep both promises at
    once. `tests/fakes` says the suite runs with "no network, no model
    downloads"; that is true of every *test*, and it was never true of the first
    thing a test asked for a token count.

    Returns whether an encoder is available, so a test that needs exact counts can
    skip with a reason instead of failing inside the guard with a message about
    injecting a fake store — which is not the problem and not the fix.

    After the first successful run the cache is on disk in the repo and no
    subsequent run touches the network at all.
    """
    from ragorc.core.tokens import load_encoder

    try:
        return load_encoder() is not None
    except Exception:  # noqa: BLE001 - availability is the answer, not the cause
        return False


@pytest.fixture(scope="session", autouse=True)
def _embedder_cache() -> bool:
    """Warm the default ONNX embedding model once, for the same reason.

    Session-scoped and therefore ahead of the function-scoped network guard, so
    the one download a fresh clone needs happens outside it. Four ingest tests
    build a real embedder — deliberately, since they are testing the dimension
    pin that only a real model has — and every one of them reached the network on
    a cold cache.
    """
    from ragorc.core.settings import get_settings

    try:
        from ragorc.embed.fastembed_provider import _load
    except ImportError:  # pragma: no cover - fastembed is an extra
        return False
    embedding = get_settings().embedding
    # Dense and sparse only. Those are the two the default ingest path builds, and
    # they are ~68 MB together; ColBERT is another order of magnitude and no unit
    # test needs a real one, so a clone does not pay for it.
    wanted = (("dense", embedding.dense_model), ("sparse", embedding.sparse_model))
    for kind, model in wanted:
        try:
            _load(kind, model, embedding.threads)
        except Exception:  # noqa: BLE001 - availability is the answer, not the cause
            return False
    return True


@pytest.fixture
def requires_tokenizer(_tokenizer_cache: bool) -> None:
    """Skip when the real BPE is unavailable — a cold cache with no network."""
    if not _tokenizer_cache:
        pytest.skip("tiktoken BPE unavailable offline; token counts fall back to an estimate")


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
