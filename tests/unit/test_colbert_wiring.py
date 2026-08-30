"""ColBERT: three consumers, and each wiring knew about a different subset.

`ColBERTReranker`'s class docstring names the property that distinguishes it from
a cross-encoder — "the document side is precomputable, so a chunk that came back
from Qdrant with its multivector attached is scored for *free*" — and
`_order_chunks` is written around exactly that, with an embed-the-gaps fallback
for the mixed candidate sets that are normal. The store never asked Qdrant for
the multivector under any configuration, so `missing` was always every candidate
and the fallback was the only path that ever ran:

    _search_vectors() -> ['dense']
    multi attached on results: [False, False, False, False, False]
    re-embedding 20 candidates (rerank_top_k default): 289.0 ms
    MaxSim over the same matrices                    :   0.86 ms

Two more of the same shape around it: the reranker built its own embedder with
`cache=None` while the deployment's cached one sat unused, and
`retrieval.colbert_rerank` on its own built no embedder at all because the
builder gated on `embedding.enable_late_interaction` — a condition the embedding
factory had already outgrown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ragorc.core.settings import Settings


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
    }
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "over",
    [
        {"embedding": {"enable_late_interaction": True}},
        {"retrieval": {"colbert_rerank": True}},
        {"retrieval": {"reranker": "colbert"}},
    ],
    ids=["enable_late_interaction", "colbert_rerank", "reranker=colbert"],
)
def test_every_consumer_needs_the_late_embedder(over: dict[str, Any]) -> None:
    """Asserted per consumer. The factory knew two of these, the builder and the
    server knew one, and nobody knew the third."""
    assert _settings(**over).late_interaction_needed


def test_a_default_deployment_does_not_need_it() -> None:
    """ColBERT multivectors are ~100x a dense vector in storage and an ONNX
    session in resident memory. Nothing may build one by accident."""
    assert not _settings().late_interaction_needed


def test_the_reranker_names_are_shared_with_the_factory() -> None:
    """`build_reranker` resolves three spellings to ColBERTReranker. A name
    added there that the predicate did not know would rebuild the exact bug."""
    import inspect

    from ragorc.core.settings import _COLBERT_RERANKER_NAMES
    from ragorc.retrieve.rerank import build_reranker

    source = inspect.getsource(build_reranker)
    assert "_COLBERT_RERANKER_NAMES" in source, "the factory spells the names out again"
    assert {"colbert", "late_interaction", "maxsim"} == _COLBERT_RERANKER_NAMES


@pytest.mark.parametrize("name", ["colbert", "late_interaction", "maxsim"])
def test_each_spelling_resolves_to_the_colbert_stage(name: str) -> None:
    from ragorc.retrieve.rerank import ColBERTReranker, build_reranker

    stub = _Embedder()
    stage = build_reranker(name, settings=_settings(), late_embedder=stub)
    assert isinstance(stage, ColBERTReranker)
    assert stage.embedder is stub


# ---------------------------------------------------------------------------
# The deployment's embedder, not a second one
# ---------------------------------------------------------------------------
class _Embedder:
    model_name = "colbert-ir/colbertv2.0"
    dimension = 128

    def __init__(self, **kw: Any) -> None:
        # Accepts the builder's `cache=`/`settings=` so it can stand in for the
        # provider class as well as be constructed bare in a reranker test.
        self.kwargs = kw
        self.cache = kw.get("cache", "the shared embedding cache")
        self.embedded: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[Any]:
        self.embedded.append(list(texts))
        return [np.ones((4, 8), dtype=np.float32) for _ in texts]

    async def embed_query(self, text: str) -> Any:
        return np.ones((3, 8), dtype=np.float32)


def test_a_cross_encoder_is_not_handed_a_late_embedder() -> None:
    """Routed rather than passed through `**kwargs`, which a cross-encoder would
    reject — the reason the argument was never plumbed in the first place."""
    from ragorc.retrieve.rerank import CrossEncoderReranker, build_reranker

    stage = build_reranker("cross_encoder", settings=_settings(), late_embedder=_Embedder())
    assert isinstance(stage, CrossEncoderReranker)


def test_the_builder_hands_its_own_embedder_to_the_reranker() -> None:
    """Object identity, not a grep.

    `late_embedder=self.late_embedder` already appears three times in this class
    — it is how the Qdrant store is built — so a source check for that string
    passed with the argument removed from the reranker call. Found by mutation,
    which is the only thing that distinguishes the two.
    """
    from ragorc.pipeline.builder import RAGPipeline

    pipeline = RAGPipeline(settings=_settings(retrieval={"reranker": "colbert"}), llm=object())
    pipeline._provider_class = lambda *a, **k: _Embedder  # type: ignore[method-assign]

    assert pipeline.reranker.embedder is pipeline.late_embedder, (
        "the reranker built a second, uncached ColBERT embedder"
    )


def _build_reranker_call(source: str) -> str:
    """The text of the `build_reranker(...)` call, and nothing else.

    `late_embedder=self.colbert` also appears in the QdrantStore construction a
    few lines above, so asserting against the whole method body is satisfied by
    the wrong line.
    """
    start = source.index("build_reranker(")
    depth, end = 0, start
    for end in range(start, len(source)):
        if source[end] == "(":
            depth += 1
        elif source[end] == ")":
            depth -= 1
            if depth == 0:
                break
    return source[start : end + 1]


def test_the_server_hands_its_own_embedder_to_the_reranker() -> None:
    import inspect

    from ragorc.server.app import _LinearEngine

    call = _build_reranker_call(inspect.getsource(_LinearEngine.build))
    assert "late_embedder=self.colbert" in call, f"the reranker call is {call!r}"


def test_the_server_builds_a_late_embedder_for_every_consumer() -> None:
    """The server's own gate, which read `enable_late_interaction` alone — so a
    `colbert_rerank` deployment reached `build_reranker` with `self.colbert=None`
    and got the uncached fallback anyway.

    Checked by orientation, not by presence. Swapping the ternary's branches keeps
    both strings a grep would look for and produces `self.colbert = None` for every
    deployment that needs ColBERT and an ONNX session for every one that does not —
    strictly worse than the bug this fixed. Found by mutation.
    """
    import ast
    import inspect
    import textwrap

    from ragorc.server.app import _LinearEngine

    tree = ast.parse(textwrap.dedent(inspect.getsource(_LinearEngine.build)))
    assigned = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "colbert" for t in node.targets
        )
    ]
    assert len(assigned) == 1, f"expected one `self.colbert = ...`, found {len(assigned)}"
    ternary = assigned[0]
    assert isinstance(ternary, ast.IfExp), ast.unparse(ternary)
    assert ast.unparse(ternary.test) == "s.late_interaction_needed", ast.unparse(ternary.test)
    assert "_embedder(" in ast.unparse(ternary.body), (
        f"the needed branch does not build one: {ast.unparse(ternary.body)}"
    )
    assert ast.unparse(ternary.orelse) == "None", ast.unparse(ternary.orelse)


@pytest.mark.parametrize(
    "over",
    [{"retrieval": {"colbert_rerank": True}}, {"retrieval": {"reranker": "colbert"}}],
    ids=["colbert_rerank", "reranker=colbert"],
)
def test_the_builder_builds_a_late_embedder_for_each_consumer(over: dict[str, Any]) -> None:
    """`colbert_rerank = True` alone produced `late_embedder -> None`,
    `_has_colbert -> False` and `want_colbert -> False` — the setting was on and
    the stage was absent, with nothing logged."""
    from ragorc.pipeline.builder import RAGPipeline

    built: list[Any] = []

    class _Stub(_Embedder):
        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            built.append(kw)

    pipeline = RAGPipeline(settings=_settings(**over), llm=object())
    pipeline._provider_class = lambda *a, **k: _Stub  # type: ignore[method-assign]

    assert pipeline.late_embedder is not None, "the consumer is on and no embedder was built"
    assert built and built[0].get("cache") is not None, "built without the shared cache"


def test_a_default_builder_builds_none() -> None:
    from ragorc.pipeline.builder import RAGPipeline

    assert RAGPipeline(settings=_settings(), llm=object()).late_embedder is None


# ---------------------------------------------------------------------------
# The store returns what the reuse branch reads
# ---------------------------------------------------------------------------
def _store(**over: Any) -> Any:
    from ragorc.stores.qdrant.store import QdrantStore

    return QdrantStore(_settings(**over), late_embedder=_Embedder())


def test_the_colbert_stage_gets_its_multivectors() -> None:
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR

    wanted = _store(retrieval={"reranker": "colbert"})._search_vectors(colbert_ready=True)
    assert wanted is not False and COLBERT_VECTOR in wanted, (
        f"the reuse branch can never run: {wanted}"
    )


def test_a_cross_encoder_deployment_pays_nothing() -> None:
    """Matrices are heavy on the wire — ~96 KB a chunk at the default token cap —
    so they must be asked for only when the stage that reads them is selected."""
    assert _store()._search_vectors(colbert_ready=True) is False


def test_reranking_switched_off_pays_nothing() -> None:
    assert (
        _store(retrieval={"reranker": "colbert", "rerank_enabled": False})._search_vectors(
            colbert_ready=True
        )
        is False
    )


def test_dense_and_colbert_coexist() -> None:
    """MMR wants dense, the reranker wants colbert, and a deployment may run both.
    Returning one name instead of a list would silently disable the other."""
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR, DENSE_VECTOR

    wanted = _store(retrieval={"reranker": "colbert", "mmr_enabled": True})._search_vectors(
        colbert_ready=True
    )
    assert set(wanted) == {DENSE_VECTOR, COLBERT_VECTOR}


def test_mmr_alone_still_asks_only_for_dense() -> None:
    from ragorc.stores.qdrant.collections import DENSE_VECTOR

    assert _store(retrieval={"mmr_enabled": True})._search_vectors(colbert_ready=True) == [
        DENSE_VECTOR
    ]


# ---------------------------------------------------------------------------
# Configuration says what we want; the collection says what is there
# ---------------------------------------------------------------------------
class _Collection:
    """A live collection whose named vectors are fixed at creation."""

    def __init__(self, *names: str) -> None:
        self.names = names

    async def get_collection(self, name: str) -> Any:
        vectors = {n: object() for n in self.names if n != "sparse"}
        sparse = {n: object() for n in self.names if n == "sparse"}
        params = type("P", (), {"vectors": vectors, "sparse_vectors": sparse})()
        config = type("C", (), {"params": params})()
        return type("I", (), {"config": config})()


def _store_over(collection: Any, **over: Any) -> Any:
    store = _store(**over)
    store._client = collection
    return store


async def test_a_legacy_collection_does_not_get_a_vector_it_lacks() -> None:
    """The regression the widened predicate introduced, and the reason
    `_has_colbert` is not the right question.

    A Qdrant collection's named vectors are fixed at creation and
    `ensure_collection` is create-if-not-exists, so switching on
    `retrieval.colbert_rerank` against an existing dense-only collection made
    every query name a vector that is not there. Reproduced against live Qdrant
    1.19: `Not existing vector name error: colbert`, on *every* search — a
    silently-dropped stage turned into a total outage.
    """
    from ragorc.stores.qdrant.collections import DENSE_VECTOR

    store = _store_over(_Collection(DENSE_VECTOR), retrieval={"reranker": "colbert"})

    assert store._has_colbert is True, "the deployment is configured for ColBERT"
    assert await store._colbert_ready() is False, "but the collection does not have the vector"
    assert store._search_vectors(colbert_ready=False) is False


async def test_a_collection_that_has_the_vector_is_used() -> None:
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR, DENSE_VECTOR

    store = _store_over(
        _Collection(DENSE_VECTOR, COLBERT_VECTOR), retrieval={"reranker": "colbert"}
    )
    assert await store._colbert_ready() is True


async def test_an_unreachable_collection_does_not_disable_the_stage() -> None:
    """Unknown is not the same as absent. A probe that failed must not silently
    switch off a feature the operator configured — that is the failure this
    round exists to stop, pointed the other way."""

    class _Down:
        async def get_collection(self, name: str) -> Any:
            raise RuntimeError("qdrant is unreachable")

    store = _store_over(_Down(), retrieval={"reranker": "colbert"})
    assert await store._colbert_ready() is True


async def test_the_probe_happens_once() -> None:
    """A startup-shaped question, not a per-query one."""
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR, DENSE_VECTOR

    calls = {"n": 0}

    class _Counting(_Collection):
        async def get_collection(self, name: str) -> Any:
            calls["n"] += 1
            return await super().get_collection(name)

    store = _store_over(
        _Counting(DENSE_VECTOR, COLBERT_VECTOR), retrieval={"reranker": "colbert"}
    )
    for _ in range(5):
        await store._colbert_ready()
    assert calls["n"] == 1


async def test_a_configuration_the_collection_cannot_serve_is_reported_once() -> None:
    """The remedy is to re-index into a new collection, which no amount of
    retrying discovers — so the operator has to be told, and told once."""
    from ragorc.stores.qdrant.collections import DENSE_VECTOR

    store = _store_over(_Collection(DENSE_VECTOR), retrieval={"reranker": "colbert"})
    assert store._colbert_warned is False
    await store._colbert_ready()
    assert store._colbert_warned is True


# ---------------------------------------------------------------------------
# And the reuse actually happens
# ---------------------------------------------------------------------------
async def test_an_attached_multivector_is_not_re_embedded() -> None:
    """The point of all of it. `_order_chunks` embeds only the gaps, and until the
    store returned the vectors every candidate was a gap."""
    from ragorc.core.models import Chunk, ScoredChunk
    from ragorc.retrieve.rerank import ColBERTReranker

    embedder = _Embedder()
    stage = ColBERTReranker(embedder, settings=_settings())

    stored = Chunk(id="a", content="already indexed with colbert")
    stored.multi = np.ones((4, 8), dtype=np.float32)
    fresh = Chunk(id="b", content="a sparse hit with no multivector")

    await stage._order_chunks(
        "q", [ScoredChunk(chunk=stored, score=1.0), ScoredChunk(chunk=fresh, score=0.9)], 2
    )

    assert embedder.embedded == [[fresh.content]], (
        f"re-embedded a chunk that arrived with its matrix: {embedder.embedded}"
    )


class _StubClient:
    """Enough of AsyncQdrantClient to see what a search actually requests."""

    def __init__(self, *names: str) -> None:
        self.names = names
        self.requests: list[dict[str, Any]] = []

    async def get_collection(self, name: str) -> Any:
        vectors = {n: object() for n in self.names if n != "sparse"}
        sparse = {n: object() for n in self.names if n == "sparse"}
        params = type("P", (), {"vectors": vectors, "sparse_vectors": sparse})()
        return type("I", (), {"config": type("C", (), {"params": params})()})()

    async def query_points(self, **kw: Any) -> Any:
        self.requests.append(kw)
        return type("R", (), {"points": []})()


async def test_a_search_never_names_a_vector_the_collection_lacks() -> None:
    """End to end through `search()`, because that is where the outage was.

    Reverting the readiness check to `self._has_colbert` — a configuration
    question — left every other test in the suite green while restoring
    `Not existing vector name error: colbert` on every query against a collection
    built before ColBERT was switched on.
    """
    from ragorc.core.models import Query
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR, DENSE_VECTOR
    from ragorc.stores.qdrant.store import QdrantStore
    from tests.fakes import StubEmbedder

    settings = _settings(
        retrieval={"reranker": "colbert", "colbert_rerank": True, "use_sparse": False},
        security={"enforce_tenant_isolation": False},
    )
    client = _StubClient(DENSE_VECTOR)
    store = QdrantStore(settings, dense_embedder=StubEmbedder(32), late_embedder=_Embedder())
    store._client = client

    await store.search(Query(text="refunds?"), top_k=5)

    assert client.requests, "no search was issued"
    for request in client.requests:
        assert request.get("using") != COLBERT_VECTOR, "queried a vector the collection lacks"
        wanted = request.get("with_vectors")
        if isinstance(wanted, list):
            assert COLBERT_VECTOR not in wanted, f"asked for {wanted}"
        prefetch = request.get("prefetch") or []
        for branch in prefetch:
            assert getattr(branch, "using", None) != COLBERT_VECTOR


async def test_a_search_does_name_it_when_the_collection_has_it() -> None:
    """The saving has to survive the fix: this is the whole point of the round."""
    from ragorc.core.models import Query
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR, DENSE_VECTOR
    from ragorc.stores.qdrant.store import QdrantStore
    from tests.fakes import StubEmbedder

    settings = _settings(
        retrieval={"reranker": "colbert", "use_sparse": False},
        security={"enforce_tenant_isolation": False},
    )
    client = _StubClient(DENSE_VECTOR, COLBERT_VECTOR)
    store = QdrantStore(settings, dense_embedder=StubEmbedder(32), late_embedder=_Embedder())
    store._client = client

    await store.search(Query(text="refunds?"), top_k=5)

    asked = [r.get("with_vectors") for r in client.requests]
    assert any(isinstance(w, list) and COLBERT_VECTOR in w for w in asked), (
        f"the stored multivectors were not requested: {asked}"
    )
