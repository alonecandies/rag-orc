"""The instrument every other judgement about this library is made with.

Round 11d checked the metric formulas and they are correct. Nobody had run
`--compare` end to end, and it was broken two ways at once: it raised on the
shipped dataset's only quality signal, and when it did not raise it measured one
pipeline twice.

Both are call-site defects against documentation that names the failure. 11d
taught `series_names()` to advertise the document-level series and never taught
`series()` to resolve them. `scope_key`'s docstring says, of the argument the
pipeline-level cache omitted: "whichever ran first answered for both — so a
benchmark comparing two pipelines measured one of them twice."
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Chunk, RetrievalResult, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings
from ragorc.eval.runner import RunReport, compare_runs


class _Docs:
    """A document-level retrieval report: only `per_query` is read here."""

    def __init__(self, **per_query: list[float]) -> None:
        self.per_query = per_query
        self.means = {f"doc_{k}": sum(v) / len(v) for k, v in per_query.items()}


def _report(name: str, **per_query: list[float]) -> RunReport:
    """A run over three cases where only the first two carry document labels.

    The shipped dataset has exactly this shape — 18 of 20 — and it is the shape
    that breaks a naive fix: the document vectors are shorter than `scored_ids`.
    """
    return RunReport(
        name=name,
        dataset="d",
        results=[],
        retrieval=None,
        document_retrieval=_Docs(**per_query),
        answer_scores={},
        scored_ids=("a", "b", "c"),
        document_ids=("a", "b"),
        ks=(10,),
    )


# ---------------------------------------------------------------------------
# Every advertised series must resolve
# ---------------------------------------------------------------------------
def test_every_name_series_names_advertises_can_be_resolved() -> None:
    """The invariant, rather than a list of the names that happened to break.

    `compare_runs` defaults to exactly what `series_names()` returns, so the two
    disagreeing is not a cosmetic mismatch — it is a crash on the first name.
    """
    report = _report("naive", **{"recall@10": [1.0, 0.5], "mrr": [1.0, 0.0]})

    for metric in report.series_names():
        report.series(metric)  # must not raise


def test_the_document_series_pairs_against_its_own_subset() -> None:
    """Not against `scored_ids`. The document vectors cover only the cases that
    carry a source document, so zipping them against every scored case raises on
    `strict=True` for precisely the datasets this series exists to grade."""
    report = _report("naive", **{"recall@10": [1.0, 0.5]})

    assert report.series("doc_recall@10") == {"a": 1.0, "b": 0.5}


def test_an_unknown_series_still_raises() -> None:
    """The prefix branch must not swallow a genuine typo into an empty dict."""
    report = _report("naive", **{"recall@10": [1.0, 0.5]})

    with pytest.raises(KeyError):
        report.series("doc_nonexistent@10")
    with pytest.raises(KeyError):
        report.series("nonsense")


def test_compare_runs_completes_on_a_document_labelled_dataset() -> None:
    """The call site. `RagService.evaluate` and `ragorc eval --compare` pass no
    `metrics=`, so they get the defaulted list and the first `doc_*` name took
    the whole comparison down — including the metrics that would have compared.
    """
    baseline = _report("naive", **{"recall@10": [1.0, 1.0], "mrr": [1.0, 1.0]})
    candidate = _report("graphrag", **{"recall@10": [0.5, 0.0], "mrr": [0.5, 0.0]})

    comparison = compare_runs(baseline, candidate)

    compared = comparison.to_dict()["metrics"]
    assert "doc_recall@10" in compared, f"the document series was not compared: {sorted(compared)}"
    assert compared["doc_recall@10"]["pairs"] == 2, "the pairing found no cases to compare"


def test_a_report_with_no_document_labels_is_unaffected() -> None:
    """A chunk-labelled dataset never advertised `doc_*` and must keep working."""
    report = RunReport(
        name="n", dataset="d", results=[], retrieval=None, document_retrieval=None,
        answer_scores={}, scored_ids=("a",), ks=(10,),
    )

    assert not [n for n in report.series_names() if n.startswith("doc_")]
    with pytest.raises(KeyError):
        report.series("doc_recall@10")


# ---------------------------------------------------------------------------
# Two pipelines must not share one cache entry
# ---------------------------------------------------------------------------
class _Mem:
    """A semantic cache that matches only on the scope it was handed."""

    def __init__(self) -> None:
        self.store: dict[tuple[Any, ...], Any] = {}

    async def get(self, question: str, *, tenant_id: Any = None, scope: Any = None) -> Any:
        payload = self.store.get((question, tenant_id, scope))
        if payload is None:
            return None

        class Hit:
            answer = payload
            question = ""
            score = 1.0
            stored_at = 0.0

        return Hit()

    async def set(self, question: str, answer: Any, **kw: Any) -> None:
        self.store[(question, kw.get("tenant_id"), kw.get("scope"))] = answer


class _Corpus:
    name = "corpus"

    async def retrieve(self, query: Any, **kw: Any) -> list[ScoredChunk]:
        return [
            ScoredChunk(
                chunk=Chunk(id="c1", content="the CFO approves it", document_id="d"),
                score=1.0,
                source=RetrievalSource.DENSE,
                rank=0,
            )
        ]

    async def retrieve_detailed(self, query: Any, **kw: Any) -> RetrievalResult:
        result = RetrievalResult()
        result.chunks = await self.retrieve(query)
        return result


async def _pipeline() -> Any:
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.pipeline.builder import RAGPipeline
    from tests.fakes import StubLLM

    settings = Settings(
        llm={"api_key": "k"},
        cache={"enabled": True, "semantic_enabled": True},
        embedding={"dense_dimension": 32},
        generation={"check_groundedness": False},
        # Not a tenancy test. The library fails closed by default, and these
        # queries carry no tenant; the suite used to inherit "off" from the
        # developer's .env, which is now neutralized in conftest.
        security={"enforce_tenant_isolation": False},
    )
    from tests.fakes import FakeVectorStore

    llm = StubLLM(text="whichever pipeline ran")
    pipeline = RAGPipeline(
        settings=settings,
        llm=llm,
        retriever=_Corpus(),
        generator=AnswerGenerator(llm, settings),
        # Injected even though `_Corpus` answers every retrieval: without it the
        # facade builds a real QdrantStore, whose client runs a compatibility
        # probe on a background thread. The conftest network guard cannot see
        # that — it patches `socket.connect` for the duration of the test, and
        # the thread outlives it — so the only signal was a stray UserWarning.
        vector_store=FakeVectorStore(),
    )
    # Assigned before any query, and the property returns it rather than building
    # a real SemanticCache — which would open a Qdrant client from a unit test.
    pipeline._semantic_cache = _Mem()
    assert pipeline.semantic_cache is pipeline._semantic_cache
    return pipeline


def _cached(answer: Any) -> bool:
    return bool((answer.metadata or {}).get("cache"))


async def test_a_second_pipeline_is_not_served_the_first_ones_answer() -> None:
    """`scope_key` takes `pipeline=` and its docstring says why: "graphrag and
    naive answer the same question differently on purpose". The HTTP layer passed
    it; this cache — the one `ragorc eval --compare` actually runs through — did
    not, so the candidate reported $0.00, zero LLM calls and 0.000 retrieval for a
    run that never happened."""
    pipeline = await _pipeline()
    question = "Who approves expenses over $500?"

    first = await pipeline.query(question, pipeline="naive")
    second = await pipeline.query(question, pipeline="graphrag")

    assert not _cached(first)
    assert not _cached(second), "graphrag was answered out of naive's cache entry"


async def test_the_same_pipeline_still_reuses_its_own_answer() -> None:
    """The saving the cache exists for has to survive the fix."""
    pipeline = await _pipeline()
    question = "Who approves expenses over $500?"

    await pipeline.query(question, pipeline="naive")
    again = await pipeline.query(question, pipeline="naive")

    assert _cached(again)


async def test_auto_and_the_name_it_resolves_to_share_an_entry() -> None:
    """The pipeline is keyed on the *resolved* graph, not the caller's word for
    it. `auto` selecting `adaptive` must hit `adaptive`'s entry, or the default
    caller pays twice for the same work."""
    pipeline = await _pipeline()
    question = "Who approves expenses over $500?"
    resolved = pipeline.select_graph("auto")

    await pipeline.query(question, pipeline=resolved)
    via_auto = await pipeline.query(question, pipeline="auto")

    assert _cached(via_auto), f"'auto' did not hit the entry '{resolved}' wrote"


async def test_the_runner_records_the_document_labelled_subset() -> None:
    """Through the real aggregation, not a hand-built report.

    Every test above constructs `RunReport(document_ids=...)` directly, so
    deleting the line in `_aggregate` that computes it left them all green — the
    call-site gap this codebase is named for, in my own new tests. Here the
    runner does the aggregating.

    Two of three cases carry a source document, which is the shipped dataset's
    shape (18 of 20) and the one that makes `document_ids` differ from
    `scored_ids`.
    """
    from ragorc.core.models import Answer
    from ragorc.eval.dataset import EvalCase
    from ragorc.eval.runner import EvalRunner

    cases = [
        EvalCase(
            question="who approves expenses?",
            expected_answer="the CFO",
            metadata={"source_document": "finance.md"},
            id="labelled-1",
        ),
        EvalCase(
            question="what is the refund window?",
            expected_answer="30 days",
            metadata={"source_document": "policy.md"},
            id="labelled-2",
        ),
        EvalCase(question="hello?", expected_answer="hi", metadata={}, id="unlabelled"),
    ]

    async def answer_fn(question: str, **kw: Any) -> Answer:
        return Answer(
            text="the CFO",
            chunks=[
                ScoredChunk(
                    chunk=Chunk(
                        id="c1",
                        content="the CFO approves it",
                        document_id="d1",
                        metadata={"source": "finance.md"},
                    ),
                    score=1.0,
                    source=RetrievalSource.DENSE,
                    rank=0,
                )
            ],
        )

    settings = Settings(llm={"api_key": "k"}, cache={"enabled": False})
    report = await EvalRunner(answer_fn, settings, metric_names=()).run(cases)

    assert report.scored_ids == ("labelled-1", "labelled-2", "unlabelled")
    assert report.document_ids == ("labelled-1", "labelled-2"), (
        "the document subset was not recorded, so its series cannot be paired"
    )
    for metric in report.series_names():
        paired = report.series(metric)
        if metric.startswith("doc_"):
            assert set(paired) == {"labelled-1", "labelled-2"}, paired
