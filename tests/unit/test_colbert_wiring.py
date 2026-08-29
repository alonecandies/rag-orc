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
    and got the uncached fallback anyway."""
    import inspect

    from ragorc.server.app import _LinearEngine

    source = inspect.getsource(_LinearEngine.build)
    assert "if s.late_interaction_needed" in source, (
        "the server gates its ColBERT embedder on one consumer again"
    )
    assert "if s.embedding.enable_late_interaction" not in source


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

    wanted = _store(retrieval={"reranker": "colbert"})._search_vectors()
    assert wanted is not False and COLBERT_VECTOR in wanted, (
        f"the reuse branch can never run: {wanted}"
    )


def test_a_cross_encoder_deployment_pays_nothing() -> None:
    """Matrices are heavy on the wire — ~96 KB a chunk at the default token cap —
    so they must be asked for only when the stage that reads them is selected."""
    assert _store()._search_vectors() is False


def test_reranking_switched_off_pays_nothing() -> None:
    assert (
        _store(retrieval={"reranker": "colbert", "rerank_enabled": False})._search_vectors()
        is False
    )


def test_a_collection_without_multivectors_is_not_asked_for_them() -> None:
    from ragorc.stores.qdrant.store import QdrantStore

    store = QdrantStore(_settings(retrieval={"reranker": "colbert"}), late_embedder=None)
    assert store._has_colbert is False
    assert store._search_vectors() is False


def test_dense_and_colbert_coexist() -> None:
    """MMR wants dense, the reranker wants colbert, and a deployment may run both.
    Returning one name instead of a list would silently disable the other."""
    from ragorc.stores.qdrant.collections import COLBERT_VECTOR, DENSE_VECTOR

    wanted = _store(retrieval={"reranker": "colbert", "mmr_enabled": True})._search_vectors()
    assert set(wanted) == {DENSE_VECTOR, COLBERT_VECTOR}


def test_mmr_alone_still_asks_only_for_dense() -> None:
    from ragorc.stores.qdrant.collections import DENSE_VECTOR

    assert _store(retrieval={"mmr_enabled": True})._search_vectors() == [DENSE_VECTOR]


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
