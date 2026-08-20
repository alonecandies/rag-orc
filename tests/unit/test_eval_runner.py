"""Eval runner: document-level retrieval grading and report assembly.

The interesting case is a **hand-written** dataset. It cannot carry
``expected_chunk_ids`` — chunk ids are derived from chunk content, so they are
unknowable before an ingest and change whenever chunking settings change. Without
document-level grading, such a dataset yields no retrieval metrics at all, and
"unmeasured" silently reads as "fine".
"""

from __future__ import annotations

import pytest

from ragorc.core.models import Answer, Chunk, ScoredChunk
from ragorc.eval.dataset import EvalCase
from ragorc.eval.retrieval_metrics import evaluate_retrieval
from ragorc.eval.runner import CaseResult, _document_key


def scored(cid: str, source: str, *, doc_id: str = "content-derived-uuid") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=cid, content=cid, document_id=doc_id, metadata={"source": source}),
        score=0.9,
    )


def case_naming(document: str, *, key: str = "source_document") -> EvalCase:
    return EvalCase(
        id="c",
        question="who approves a large expense claim?",
        expected_answer="the Head of Finance",
        metadata={key: document},
    )


# ---------------------------------------------------------------------------
# _document_key
# ---------------------------------------------------------------------------
def test_document_key_prefers_the_human_readable_source() -> None:
    """A hand-written case names a file; ``document_id`` is a content-derived uuid
    the author could not have known. Comparing against the uuid would score every
    case zero and look like total retrieval failure."""
    chunk = Chunk(id="c1", content="x", document_id="uuid-abc", metadata={"source": "policy.md"})
    assert _document_key(chunk) == "policy.md"


def test_document_key_falls_back_to_document_id() -> None:
    chunk = Chunk(id="c1", content="x", document_id="uuid-abc")
    assert _document_key(chunk) == "uuid-abc"


def test_document_key_is_empty_when_nothing_identifies_the_source() -> None:
    assert _document_key(Chunk(id="c1", content="x")) == ""


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key", ["source_document", "source_documents", "expected_documents", "document_id"]
)
def test_expected_documents_reads_any_supported_key(key: str) -> None:
    result = CaseResult(case=case_naming("policy.md", key=key), answer=Answer(text=""))
    assert result.expected_documents == ("policy.md",)


def test_expected_documents_accepts_a_list() -> None:
    case = EvalCase(
        id="c",
        question="q",
        metadata={"expected_documents": ["a.md", "b.md"]},
    )
    result = CaseResult(case=case, answer=Answer(text=""))
    assert result.expected_documents == ("a.md", "b.md")


def test_expected_documents_is_empty_without_metadata() -> None:
    result = CaseResult(case=EvalCase(id="c", question="q"), answer=Answer(text=""))
    assert result.expected_documents == ()


def test_retrieved_documents_deduplicates_at_best_rank() -> None:
    """Three chunks from one document must count as that document once, at its
    best rank — otherwise document recall can exceed 1.0."""
    answer = Answer(
        text="x",
        chunks=[
            scored("c1", "policy.md"),
            scored("c2", "policy.md"),
            scored("c3", "handbook.md"),
            scored("c4", "policy.md"),
        ],
    )
    result = CaseResult(case=case_naming("policy.md"), answer=answer)
    assert result.retrieved_documents == ("policy.md", "handbook.md")


def test_retrieved_documents_is_empty_without_an_answer() -> None:
    assert CaseResult(case=case_naming("policy.md")).retrieved_documents == ()


def test_document_recall_cannot_exceed_one() -> None:
    """The property the deduplication protects, asserted end to end."""
    answer = Answer(text="x", chunks=[scored(f"c{i}", "policy.md") for i in range(5)])
    result = CaseResult(case=case_naming("policy.md"), answer=answer)
    report = evaluate_retrieval(
        [list(result.retrieved_documents)], [set(result.expected_documents)], ks=(5,)
    )
    assert report.mean()["recall@5"] == pytest.approx(1.0)


def test_document_grading_scores_a_hit_and_a_miss() -> None:
    hit = CaseResult(
        case=case_naming("policy.md"),
        answer=Answer(text="x", chunks=[scored("c1", "policy.md"), scored("c2", "other.md")]),
    )
    miss = CaseResult(
        case=case_naming("policy.md"),
        answer=Answer(text="x", chunks=[scored("c9", "release.md")]),
    )
    for result, expected in ((hit, 1.0), (miss, 0.0)):
        report = evaluate_retrieval(
            [list(result.retrieved_documents)], [set(result.expected_documents)], ks=(1,)
        )
        assert report.mean()["recall@1"] == pytest.approx(expected)


def test_case_result_serializes_both_granularities() -> None:
    """A report consumer must be able to tell which granularity a number came
    from, so both appear in the record."""
    result = CaseResult(
        case=case_naming("policy.md"),
        answer=Answer(text="x", chunks=[scored("c1", "policy.md")]),
    )
    record = result.to_dict()
    assert record["retrieved_documents"] == ["policy.md"]
    assert record["expected_documents"] == ["policy.md"]
    assert record["expected_chunk_ids"] == []


# ---------------------------------------------------------------------------
# The shipped dataset
# ---------------------------------------------------------------------------
async def test_shipped_eval_dataset_supports_document_grading() -> None:
    """The dataset in examples/ must actually be gradeable, or it is decoration."""
    from pathlib import Path

    from ragorc.eval.dataset import EvalDataset

    path = Path("examples/eval/questions.jsonl")
    if not path.is_file():
        pytest.skip("example dataset not present")

    dataset = await EvalDataset.load(str(path))
    cases = list(dataset)
    assert cases, "dataset loaded empty — are the comment lines being skipped?"

    graded = [c for c in cases if CaseResult(case=c, answer=Answer(text="")).expected_documents]
    assert graded, "no case names a source document, so retrieval cannot be graded at all"
    assert len(graded) >= len(cases) * 0.8, (
        f"only {len(graded)}/{len(cases)} cases carry a source document"
    )


# ---------------------------------------------------------------------------
# A/B verdict polarity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("metric", "lower_better"),
    [
        ("latency_ms_p50", True),
        ("latency_ms_p95", True),
        ("cost_usd", True),
        ("total_tokens", True),
        ("n_errors", True),
        ("recall@10", False),
        ("ndcg@5", False),
        ("faithfulness", False),
        ("answer_relevance", False),
        # Deliberately not classified: a rise can mean retrieval regressed *or*
        # that the guardrails correctly started declining unanswerable questions.
        ("abstention_rate", False),
    ],
)
def test_metric_polarity(metric: str, lower_better: bool) -> None:
    from ragorc.eval.runner import lower_is_better

    assert lower_is_better(metric) is lower_better


def test_slower_is_not_reported_as_better() -> None:
    """The bug this guards: a bare ``difference > 0`` labels a slower, costlier
    pipeline an improvement — backwards for the two numbers most likely to decide
    whether a change ships."""
    from ragorc.eval.runner import BootstrapResult

    slower = BootstrapResult(
        metric="latency_ms_p95",
        n_pairs=40,
        baseline=100.0,
        candidate=180.0,
        difference=80.0,
        ci_low=40.0,
        ci_high=120.0,
        p_value=0.001,
    )
    assert slower.significant
    assert slower.verdict == "worse", "an 80ms regression must never read as better"

    faster = BootstrapResult(
        metric="latency_ms_p95",
        n_pairs=40,
        baseline=180.0,
        candidate=100.0,
        difference=-80.0,
        ci_low=-120.0,
        ci_high=-40.0,
        p_value=0.001,
    )
    assert faster.verdict == "better"


def test_higher_recall_is_still_better() -> None:
    from ragorc.eval.runner import BootstrapResult

    improved = BootstrapResult(
        metric="recall@10",
        n_pairs=40,
        baseline=0.60,
        candidate=0.75,
        difference=0.15,
        ci_low=0.05,
        ci_high=0.25,
        p_value=0.002,
    )
    assert improved.verdict == "better"


def test_an_inconclusive_difference_is_not_a_verdict() -> None:
    """An interval spanning zero is noise, whichever direction the mean points."""
    from ragorc.eval.runner import BootstrapResult

    noise = BootstrapResult(
        metric="latency_ms_p95",
        n_pairs=40,
        baseline=100.0,
        candidate=104.0,
        difference=4.0,
        ci_low=-15.0,
        ci_high=23.0,
        p_value=0.6,
    )
    assert noise.verdict == "inconclusive"
