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


# ---------------------------------------------------------------------------
# The A/B table
# ---------------------------------------------------------------------------
def _rendered(comparison: object) -> str:
    import io

    from rich.console import Console

    from ragorc.cli import _comparison_table

    table, _hidden = _comparison_table(comparison.to_dict())  # type: ignore[attr-defined]
    console = Console(file=io.StringIO(), width=140)
    console.print(table)
    return console.file.getvalue()  # type: ignore[attr-defined]


def test_the_comparison_table_survives_a_metric_with_no_comparable_pairs() -> None:
    """`paired_bootstrap` returns `None` for every statistic when the two runs
    share no scored case — correct, since there is nothing to compare. The table
    then formatted `None` with `:.3f` and the whole `ragorc eval --compare`
    invocation died with a TypeError, taking the metrics that *did* compare with
    it."""
    from ragorc.eval.runner import Comparison, paired_bootstrap

    comparison = Comparison(
        baseline="a",
        candidate="b",
        results={
            "ndcg@10": paired_bootstrap({}, {}, metric="ndcg@10"),
            "faithfulness": paired_bootstrap(
                {"q1": 0.2, "q2": 0.3}, {"q1": 0.9, "q2": 0.8}, metric="faithfulness"
            ),
        },
    )

    out = _rendered(comparison)

    assert "ndcg@10" in out, out
    assert "faithfulness" in out, "a comparable metric must still be reported"


def test_the_comparison_table_prints_the_real_confidence_interval() -> None:
    """`BootstrapResult.to_dict()` emits `ci` as a `[low, high]` pair. The table
    read `ci_low` and `ci_high`, which do not exist, so `.get(..., 0.0)` made the
    column read `+0.000 to +0.000` on every row — the one number that says
    whether an A/B difference is real, permanently pinned to zero."""
    from ragorc.eval.runner import Comparison, paired_bootstrap

    baseline = {f"q{i}": 0.10 * (i % 3) for i in range(30)}
    candidate = {f"q{i}": 0.10 * (i % 3) + 0.5 for i in range(30)}
    result = paired_bootstrap(baseline, candidate, metric="faithfulness")
    low, high = result.to_dict()["ci"]
    assert low is not None and abs(low) > 1e-9, f"the fixture needs a real interval: {low}"

    out = _rendered(Comparison(baseline="a", candidate="b", results={"faithfulness": result}))
    row = next(line for line in out.splitlines() if "faithfulness" in line)

    assert "+0.000 to +0.000" not in row, f"the interval column is hard-zero: {row}"
    # The whole interval as one substring: matching the bound alone passes by
    # coincidence when it equals the delta printed in the neighbouring column.
    assert f"{low:+.3f} to {high:+.3f}" in row, (
        f"expected the real interval {low:+.3f} to {high:+.3f} in: {row}"
    )


# ---------------------------------------------------------------------------
# Document-level retrieval metrics have to reach a reader
# ---------------------------------------------------------------------------
def _report_with_document_labels():  # noqa: ANN202
    """A report shaped like the shipped dataset: source documents, no chunk ids."""
    from ragorc.eval.retrieval_metrics import evaluate_retrieval
    from ragorc.eval.runner import RunReport

    report = RunReport(name="run", dataset="examples/eval/questions.jsonl")
    report.document_retrieval = evaluate_retrieval(
        [["doc-a", "doc-b"], ["doc-c", "doc-a"]],
        [["doc-a"], ["doc-c"]],
        ks=(10,),
    )
    return report


def test_document_level_retrieval_metrics_reach_the_report() -> None:
    """`examples/eval/questions.jsonl` has 20 cases: none carry `expected_chunk_ids`
    and 18 carry a `source_document`. So the shipped `make eval` computes
    document-level retrieval quality for 18 cases and then throws it away —
    `to_dict()` (the `--json` record) omitted it, `series_names()` excluded it so
    `--compare` could never A/B it, and `to_markdown()` printed "no chunk labels:
    retrieval metrics not computed", which is not true: they were computed.
    """
    report = _report_with_document_labels()

    assert report.document_retrieval_metrics(), "the fixture must produce doc metrics"

    payload = report.to_dict()
    assert payload.get("document_retrieval"), (
        f"the JSON record drops document-level retrieval: {sorted(payload)}"
    )

    markdown = report.to_markdown()
    assert "not computed" not in markdown, f"claims nothing was measured:\n{markdown}"

    assert any(name.startswith("doc_") for name in report.series_names()), (
        f"--compare cannot A/B a metric it does not list: {report.series_names()}"
    )
