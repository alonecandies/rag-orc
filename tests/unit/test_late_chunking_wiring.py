"""One model, or late chunking means nothing.

ADR-0002 is the decision this library is built on, and it states the rule in one
line: the pooled chunk vector only means something if it lands in the same space
as the query vector, so **one model** must produce both. Three call sites broke
it, and all three are wiring rather than mathematics — the primitives are right
and the callers hand them the wrong object.

The ADR even documents the first of them as a mistake already made and removed:
substituting ColBERT as the token source, "a different model in a different
space at a different width (128 vs 384)". It was live again in
``IngestPipeline._late_chunker``, reachable by a route the ADR does not mention —
switching on ColBERT *reranking* changed the *chunking strategy*.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import ChunkingStrategy
from ragorc.core.settings import Settings
from ragorc.embed.factory import supports_late_chunking
from ragorc.embed.late_chunking import resolve_strategy
from ragorc.index.pipeline import IngestPipeline


def _settings(*, colbert: bool = False) -> Settings:
    return Settings(
        llm={"api_key": "k"},
        embedding={"enable_late_interaction": colbert, "dense_dimension": 32},
        cache={"enabled": False},
    )


class _Embedder:
    """A stand-in that reports a name and a width, which is all the wiring reads."""

    def __init__(self, name: str, dimension: int = 32) -> None:
        self.model_name = name
        self.dimension = dimension
        self.max_tokens = 512

    async def embed_documents(self, texts: Any) -> list[Any]:
        return []

    async def embed_query(self, text: str) -> Any:
        return []


# ---------------------------------------------------------------------------
# The token source
# ---------------------------------------------------------------------------
def test_the_token_source_is_the_dense_embedder_not_colbert() -> None:
    """What late chunking *is*. Pooling token vectors yields a usable chunk vector
    only if the result lands in the query's space, and queries go through the
    dense model."""
    dense = _Embedder("dense-model")
    pipeline = IngestPipeline(
        settings=_settings(colbert=True),
        dense_embedder=dense,
        late_embedder=_Embedder("colbert-ir/colbertv2.0", dimension=128),
    )

    chunker = pipeline._late_chunker()

    assert chunker.token_embedder is dense, (
        f"the token source is {type(chunker.token_embedder).__name__} "
        f"{getattr(chunker.token_embedder, 'model_name', '')}, not the dense embedder"
    )


async def test_enabling_colbert_reranking_does_not_change_the_chunking_strategy() -> None:
    """The route the ADR does not mention. `FastEmbedLateInteraction` exposes token
    output, so handing it in as the token source made `supports_token_embeddings`
    true and flipped AUTO to LATE — for a caller who only wanted reranking.
    """
    from ragorc.pipeline.builder import RAGPipeline

    settings = _settings(colbert=True)
    rag = RAGPipeline(settings=settings)
    pipeline = IngestPipeline(
        settings=settings,
        dense_embedder=rag.dense_embedder,
        late_embedder=rag.late_embedder,
    )

    strategy = await resolve_strategy(ChunkingStrategy.AUTO, pipeline._late_chunker(), settings)

    assert strategy is not ChunkingStrategy.LATE


async def test_the_capability_check_and_the_resolver_agree() -> None:
    """ADR-0002 says there is a test for this disagreement. There was — it passed
    the *dense* embedder rather than the pipeline's chunker, so it resolved EARLY
    and passed while the shipped path resolved LATE.
    """
    from ragorc.pipeline.builder import RAGPipeline

    for colbert in (False, True):
        settings = _settings(colbert=colbert)
        rag = RAGPipeline(settings=settings)
        pipeline = IngestPipeline(
            settings=settings,
            dense_embedder=rag.dense_embedder,
            late_embedder=rag.late_embedder,
        )
        strategy = await resolve_strategy(
            ChunkingStrategy.AUTO, pipeline._late_chunker(), settings
        )
        capable = supports_late_chunking(settings)
        assert capable == (strategy is ChunkingStrategy.LATE), (
            f"colbert={colbert}: supports_late_chunking()={capable} "
            f"but resolve_strategy()={strategy}"
        )


# ---------------------------------------------------------------------------
# The query side
# ---------------------------------------------------------------------------
class _Store:
    """Only the attribute the binding touches."""

    def __init__(self, dense_embedder: Any) -> None:
        self.dense_embedder = dense_embedder


def test_an_injected_store_is_pointed_at_the_late_chunker() -> None:
    """`query_side` was computed in `_prepare` and read in exactly one branch —
    `if self.vector is None` — and every shipped construction site injects the
    store, so the branch was dead and the store kept embedding queries with the
    dense provider while ingest wrote vectors pooled by the chunker. ADR-0002 calls
    that a silent recall collapse.
    """
    dense = _Embedder("dense-model")
    chunker = _Embedder("late-chunker")
    pipeline = IngestPipeline(settings=_settings(), dense_embedder=dense)
    pipeline.vector = _Store(dense)  # type: ignore[assignment]

    pipeline._bind_query_side(chunker)

    assert pipeline.vector.dense_embedder is chunker  # type: ignore[union-attr]


def test_binding_is_a_no_op_when_the_store_already_agrees() -> None:
    """Under EARLY the query side *is* the dense embedder, so nothing should move
    and nothing should be logged."""
    dense = _Embedder("dense-model")
    pipeline = IngestPipeline(settings=_settings(), dense_embedder=dense)
    pipeline.vector = _Store(dense)  # type: ignore[assignment]

    pipeline._bind_query_side(dense)

    assert pipeline.vector.dense_embedder is dense  # type: ignore[union-attr]


def test_a_store_that_cannot_be_rebound_is_reported_not_crashed() -> None:
    """The stores are duck-typed. A third-party one without the attribute should
    warn rather than raise mid-ingest."""

    class Opaque:
        __slots__ = ()

    pipeline = IngestPipeline(settings=_settings(), dense_embedder=_Embedder("dense"))
    pipeline.vector = Opaque()  # type: ignore[assignment]

    pipeline._bind_query_side(_Embedder("late-chunker"))  # must not raise


# ---------------------------------------------------------------------------
# Derived units
# ---------------------------------------------------------------------------
def test_derived_units_use_the_same_embedder_as_the_leaves() -> None:
    """RAPTOR summaries, multi-representation units and parent documents land in
    the same collection as the leaves and are found by the same query vector. Under
    LATE they were embedded by the dense provider — a different space, in the same
    collection, ranked against the same query."""
    dense = _Embedder("dense-model")
    chunker = _Embedder("late-chunker")
    pipeline = IngestPipeline(settings=_settings(), dense_embedder=dense)

    assert pipeline._units() is dense, "before _prepare there is no strategy to honour"

    pipeline._unit_embedder = chunker
    assert pipeline._units() is chunker


def test_the_splitter_stays_on_the_dense_embedder() -> None:
    """Deliberately excluded. The splitter embeds to *decide* a breakpoint and
    stores nothing, and the chunker needs the splitter — routing it through
    `_units()` would be circular for no benefit."""
    dense = _Embedder("dense-model")
    pipeline = IngestPipeline(settings=_settings(), dense_embedder=dense)
    pipeline._unit_embedder = _Embedder("late-chunker")

    with pytest.MonkeyPatch.context() as patch:
        seen: list[Any] = []
        patch.setattr(
            "ragorc.index.pipeline.build_splitter",
            lambda *, embedder, settings: seen.append(embedder) or object(),
        )
        pipeline._splitter_for()

    assert seen == [dense]


# ---------------------------------------------------------------------------
# The call sites, not the helpers
# ---------------------------------------------------------------------------
# Every test above drives a helper directly. That is not enough here and the
# mutation run proved it: reverting `_ensure_stores` so it never calls
# `_bind_query_side`, and reverting the two stage sites so they call `_dense()`
# again, all survived — the helpers were still correct and still unreached. These
# four drive `_prepare` and the construction sites instead.
class _Stores:
    """Injected stores, which is what every shipped construction site does — and
    what made the `query_side` branch dead code."""

    def __init__(self, dense: Any) -> None:
        self.dense_embedder = dense

    async def ensure_collection(self, **kwargs: Any) -> None:
        return None

    async def ensure_schema(self, **kwargs: Any) -> None:
        return None


async def _prepared(monkeypatch: pytest.MonkeyPatch, strategy: ChunkingStrategy) -> Any:
    """Run `_prepare` with the strategy forced, and fakes for everything it touches."""
    dense = _Embedder("dense-model")
    pipeline = IngestPipeline(settings=_settings(), dense_embedder=dense)
    store = _Stores(dense)
    pipeline.vector = store  # type: ignore[assignment]
    pipeline.relational = store  # type: ignore[assignment]

    async def _resolved(*args: Any, **kwargs: Any) -> ChunkingStrategy:
        return strategy

    monkeypatch.setattr("ragorc.index.pipeline.resolve_strategy", _resolved)
    monkeypatch.setattr(IngestPipeline, "_pin_dimension", _dimension)
    monkeypatch.setattr(IngestPipeline, "_sparse", lambda self: None)
    monkeypatch.setattr(IngestPipeline, "_colbert", lambda self: None)
    await pipeline._prepare()
    return pipeline


async def _dimension(self: Any, embedder: Any, *, measure: bool = False) -> int:
    del embedder, measure
    return 32


async def test_prepare_points_the_injected_store_at_the_chunker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = await _prepared(monkeypatch, ChunkingStrategy.LATE)

    assert pipeline.vector.dense_embedder is pipeline.late_chunker, (  # type: ignore[union-attr]
        "the store still embeds queries with the dense provider under LATE"
    )


async def test_prepare_leaves_the_store_alone_under_early(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = await _prepared(monkeypatch, ChunkingStrategy.EARLY)

    assert pipeline.vector.dense_embedder is pipeline.dense_embedder  # type: ignore[union-attr]


async def test_prepare_records_the_unit_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_units()` cannot return the chunker if nothing ever stored it."""
    pipeline = await _prepared(monkeypatch, ChunkingStrategy.LATE)

    assert pipeline._unit_embedder is pipeline.late_chunker
    assert pipeline._units() is pipeline.late_chunker


async def test_the_stage_sites_ask_for_the_unit_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both sites that build something *stored*, asserted on the argument they
    pass rather than on `_units()` in isolation."""
    from ragorc.core.models import Document

    dense = _Embedder("dense-model")
    chunker = _Embedder("late-chunker")
    pipeline = IngestPipeline(settings=_settings(), dense_embedder=dense)
    pipeline._unit_embedder = chunker

    seen: list[Any] = []

    # -- the optional index stages (RAPTOR, multi-representation) --
    from ragorc.index.pipeline import _Plugin

    plugin = _Plugin(label="stage", modules=("m",), factories=("F",))
    monkeypatch.setattr("ragorc.index.pipeline._OPTIONAL_STAGES", (plugin,))
    monkeypatch.setattr(IngestPipeline, "_stage_enabled", lambda self, label: True)
    monkeypatch.setattr("ragorc.index.pipeline._load_plugin", lambda p: object())
    monkeypatch.setattr(
        "ragorc.index.pipeline._construct",
        lambda factory, **kw: seen.append(kw.get("embedder")) or object(),
    )
    pipeline._build_stages()

    # -- parent-document indexing --
    class Indexer:
        def __init__(self, *, embedder: Any, **kwargs: Any) -> None:
            seen.append(embedder)

        async def build(self, document: Any) -> Any:
            class _Index:
                children: tuple[Any, ...] = ()
                parents: tuple[Any, ...] = ()

            return _Index()

    monkeypatch.setattr("ragorc.index.multirep.ParentDocumentIndexer", Indexer)
    pipeline.config.parent_document_enabled = True
    await pipeline._split(Document(id="d", content="body", source="s.md"))

    assert seen and all(e is chunker for e in seen), (
        f"a stored unit was embedded by something other than the unit embedder: {seen}"
    )
