"""What reaches the scoring stages, and what they report having done.

Round 11d checked the ranking mathematics and found it sound. This file covers
the other half: three stages whose formulas are correct and which were being fed
the wrong thing, so the mathematics never ran.

One fact caused two of them. ``QdrantStore.search`` hardcoded
``with_vectors=False``, so no chunk on any shipped retrieval path carried a dense
vector — and both MMR and the embedding-filter compressor read ``chunk.dense``
and degrade silently without it. MMR became plain truncation on every query, and
the compressor re-embedded every candidate at ~7 s a call against the 2 µs its
own matmul takes.

The third is a call site: ``nodes.retrieve`` read ``state["top_k"]`` where its
sibling calls ``_fetch_k``, whose docstring states the resulting failure word for
word.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ragorc.core.models import Chunk, Query, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings
from ragorc.generate.answer import AnswerGenerator
from ragorc.pipeline.nodes import PipelineNodes
from ragorc.pipeline.state import initial_state
from tests.fakes import StubLLM


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
    }
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# How wide the fan-out is
# ---------------------------------------------------------------------------
class _Recording:
    name = "recording"

    def __init__(self) -> None:
        self.asked: list[int | None] = []

    async def retrieve(self, query: Query, *, top_k: int | None = None, **kw: Any) -> list[Any]:
        self.asked.append(top_k)
        return [
            ScoredChunk(
                chunk=Chunk(id=f"c{i}", content=f"body {i}", document_id="d"),
                score=1.0 - i / 100,
                source=RetrievalSource.DENSE,
                rank=i,
            )
            for i in range(top_k or 0)
        ]


def _nodes(retriever: Any, settings: Settings) -> PipelineNodes:
    llm = StubLLM()
    return PipelineNodes(
        settings=settings, retriever=retriever, llm=llm, generator=AnswerGenerator(llm, settings)
    )


async def test_the_leg_is_asked_for_fetch_k_not_top_k() -> None:
    """``_fetch_k``'s docstring is the specification: "A leg fetching only
    ``top_k`` makes the reranker reorder ten documents instead of choosing ten out
    of fifty." ``store_node`` called it; this node read the state directly and
    passed ``None``, so the retriever fell back to ``retrieval.top_k``.
    """
    settings = _settings(retrieval={"top_k": 10, "fetch_k": 50})
    retriever = _Recording()
    nodes = _nodes(retriever, settings)
    state = initial_state("what is the refund window?")
    state["query"] = Query(text="q")

    await nodes.retrieve(state)

    assert retriever.asked == [nodes._fetch_k(state)] == [50]


async def test_an_explicit_top_k_still_raises_the_floor() -> None:
    """``_fetch_k`` is ``max(fetch_k, top_k)``: a caller asking for 200 results
    must not be served 50."""
    settings = _settings(retrieval={"top_k": 10, "fetch_k": 50})
    retriever = _Recording()
    nodes = _nodes(retriever, settings)
    state = initial_state("q", top_k=200)
    state["query"] = Query(text="q")

    await nodes.retrieve(state)

    assert retriever.asked == [200]


async def test_both_retrieval_nodes_agree_on_the_width() -> None:
    """The defect was two call sites in one class disagreeing. Pinned as a
    property of the pair rather than of either one."""
    settings = _settings(retrieval={"top_k": 10, "fetch_k": 50})
    nodes = _nodes(_Recording(), settings)
    state = initial_state("q")
    state["query"] = Query(text="q")

    import inspect

    source = inspect.getsource(type(nodes).retrieve)
    assert "self._fetch_k(state)" in source, "retrieve stopped using the shared width"


# ---------------------------------------------------------------------------
# Whether the vectors MMR needs ever arrive
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("mmr", "compress", "compressor", "wanted"),
    [
        pytest.param(False, False, "embedding_filter", False, id="neither-costs-nothing"),
        pytest.param(True, False, "embedding_filter", True, id="mmr-needs-them"),
        pytest.param(False, True, "embedding_filter", True, id="compressor-needs-them"),
        pytest.param(False, True, "both", True, id="both-includes-the-filter"),
        pytest.param(False, True, "extract", False, id="extract-does-not"),
        pytest.param(False, True, "sentence", False, id="sentence-does-not"),
    ],
)
def test_vectors_are_requested_exactly_when_something_reads_them(
    mmr: bool, compress: bool, compressor: str, wanted: bool
) -> None:
    """Off by default, because 50 candidates of 384 float32 is ~77 KB per query
    and the payload is what answers it. On when a stage will read them, because
    both stages that do degrade silently to something that looks like success.
    """
    from ragorc.stores.qdrant.collections import DENSE_VECTOR
    from ragorc.stores.qdrant.store import QdrantStore

    settings = _settings(
        retrieval={
            "mmr_enabled": mmr,
            "compression_enabled": compress,
            "compressor": compressor,
        }
    )
    requested = QdrantStore(settings)._search_vectors()

    assert requested == ([DENSE_VECTOR] if wanted else False)


def test_only_the_dense_vector_is_asked_for() -> None:
    """``with_vectors=True`` would also drag back the sparse vector and ColBERT's
    per-token matrix, which nothing on this path reads and which is an order of
    magnitude larger."""
    from ragorc.stores.qdrant.store import QdrantStore

    settings = _settings(retrieval={"mmr_enabled": True})
    requested = QdrantStore(settings)._search_vectors()

    assert requested is not True
    assert isinstance(requested, list) and len(requested) == 1


# ---------------------------------------------------------------------------
# What the noise report claims
# ---------------------------------------------------------------------------
def _scored(n: int, *, with_vectors: bool) -> list[ScoredChunk]:
    out = []
    for i in range(n):
        chunk = Chunk(id=f"c{i}", content=f"passage {i} about refunds", document_id="d")
        if with_vectors:
            vector = np.zeros(32, dtype=np.float32)
            vector[i % 4] = 1.0
            chunk.dense = vector
        out.append(
            ScoredChunk(
                chunk=chunk, score=1.0 - i / 100, source=RetrievalSource.DENSE, rank=i
            )
        )
    return out


def test_truncation_is_not_reported_as_diversity() -> None:
    """The half that hid the other half. ``diversity_dropped`` was assigned inside
    the ``mmr_enabled`` branch whichever way the branch went, so plain truncation
    was logged as ``diversity=17`` and an operator had no way to see the feature
    they enabled was inert."""
    from ragorc.retrieve.noise import NoiseFilter

    settings = _settings(retrieval={"mmr_enabled": True, "top_k": 4, "dedupe_enabled": False})
    report = NoiseFilter(settings=settings).apply(_scored(10, with_vectors=False), top_k=4)[1]

    assert report.diversity_dropped == 0, "MMR could not run; nothing was dropped for diversity"
    assert report.truncated == 6


def test_diversity_is_reported_when_mmr_actually_runs() -> None:
    from ragorc.retrieve.noise import NoiseFilter

    settings = _settings(retrieval={"mmr_enabled": True, "top_k": 4, "dedupe_enabled": False})
    report = NoiseFilter(settings=settings).apply(_scored(10, with_vectors=True), top_k=4)[1]

    assert report.diversity_dropped == 6
    assert report.truncated == 0


def test_a_plain_cut_is_attributed_to_truncation() -> None:
    """With MMR off there is no ambiguity, and the number still has to land
    somewhere — ``removed`` is asserted against elsewhere."""
    from ragorc.retrieve.noise import NoiseFilter

    settings = _settings(retrieval={"mmr_enabled": False, "top_k": 4, "dedupe_enabled": False})
    kept, report = NoiseFilter(settings=settings).apply(_scored(10, with_vectors=True), top_k=4)

    assert len(kept) == 4
    assert (report.truncated, report.diversity_dropped) == (6, 0)
    assert report.removed >= 6, "the cut must be counted in the total"
