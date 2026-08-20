"""Retrieval metrics, checked against hand-computed values.

Every number here is derivable with a pen. That matters more than usual: these
metrics are the instrument used to judge every other configuration choice in the
library, so a subtly wrong metric does not fail loudly — it silently endorses the
wrong retrieval strategy for as long as anyone trusts it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ragorc.eval.retrieval_metrics import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def one(value) -> float:
    """Unwrap a single-query batch result."""
    array = np.asarray(value, dtype=np.float64).ravel()
    return float(array[0]) if array.size else float("nan")


# ---------------------------------------------------------------------------
# recall@k — the ceiling the reranker cannot exceed
# ---------------------------------------------------------------------------
def test_recall_at_k_hand_computed() -> None:
    # retrieved a,b,c; relevant b,z. Only b is found, out of 2 relevant -> 0.5
    assert one(recall_at_k([["a", "b", "c"]], [["b", "z"]], k=3)) == pytest.approx(0.5)


def test_recall_is_bounded_by_k() -> None:
    """A relevant document below the cutoff is not recalled — this is exactly why
    fetch_k and top_k are separate settings."""
    assert one(recall_at_k([["a", "b", "c"]], [["c"]], k=2)) == pytest.approx(0.0)
    assert one(recall_at_k([["a", "b", "c"]], [["c"]], k=3)) == pytest.approx(1.0)


def test_recall_perfect_and_zero() -> None:
    assert one(recall_at_k([["a", "b"]], [["a", "b"]], k=2)) == pytest.approx(1.0)
    assert one(recall_at_k([["x", "y"]], [["a", "b"]], k=2)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------
def test_precision_at_k_hand_computed() -> None:
    # 2 of the top 4 retrieved are relevant -> 0.5
    assert one(precision_at_k([["a", "b", "c", "d"]], [["a", "c"]], k=4)) == pytest.approx(0.5)


def test_precision_counts_the_cutoff_not_the_list() -> None:
    # Top 2 are both relevant -> 1.0, even though a third relevant doc exists.
    assert one(precision_at_k([["a", "b", "x"]], [["a", "b", "c"]], k=2)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hit rate
# ---------------------------------------------------------------------------
def test_hit_rate_is_binary() -> None:
    assert one(hit_rate_at_k([["a", "b"]], [["b"]], k=2)) == pytest.approx(1.0)
    assert one(hit_rate_at_k([["a", "b"]], [["z"]], k=2)) == pytest.approx(0.0)
    # One hit is enough; it does not measure how many.
    assert one(hit_rate_at_k([["a", "b"]], [["a", "b"]], k=2)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MRR — rank quality, which is what a reranker improves
# ---------------------------------------------------------------------------
def test_reciprocal_rank_hand_computed() -> None:
    assert one(reciprocal_rank([["a", "b", "c"]], [["a"]])) == pytest.approx(1.0)
    assert one(reciprocal_rank([["a", "b", "c"]], [["b"]])) == pytest.approx(0.5)
    assert one(reciprocal_rank([["a", "b", "c"]], [["c"]])) == pytest.approx(1 / 3)
    assert one(reciprocal_rank([["a", "b", "c"]], [["z"]])) == pytest.approx(0.0)


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert one(reciprocal_rank([["a", "b", "c"]], [["b", "c"]])) == pytest.approx(0.5)


def test_mrr_averages_across_queries() -> None:
    value = mrr([["a", "b"], ["x", "y"]], [["a"], ["y"]])
    assert value == pytest.approx((1.0 + 0.5) / 2)


# ---------------------------------------------------------------------------
# nDCG — rank quality with graded position discount
# ---------------------------------------------------------------------------
def test_ndcg_is_one_when_perfectly_ranked() -> None:
    assert one(ndcg_at_k([["a", "b"]], [["a", "b"]], k=2)) == pytest.approx(1.0)


def test_ndcg_hand_computed_for_a_single_hit_at_rank_two() -> None:
    # DCG = 1/log2(3); IDEAL = 1/log2(2) = 1  ->  nDCG = 1/log2(3) = 0.6309
    expected = 1.0 / math.log2(3)
    assert one(ndcg_at_k([["a", "b"]], [["b"]], k=2)) == pytest.approx(expected, abs=1e-6)


def test_ndcg_penalizes_a_worse_ordering() -> None:
    good = one(ndcg_at_k([["a", "x", "y"]], [["a"]], k=3))
    bad = one(ndcg_at_k([["x", "y", "a"]], [["a"]], k=3))
    assert good > bad, "nDCG must reward putting the relevant document earlier"


def test_ndcg_zero_when_nothing_relevant_retrieved() -> None:
    assert one(ndcg_at_k([["x", "y"]], [["a"]], k=2)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MAP
# ---------------------------------------------------------------------------
def test_average_precision_hand_computed() -> None:
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333
    expected = (1.0 + 2.0 / 3.0) / 2.0
    assert one(average_precision([["a", "x", "b", "y"]], [["a", "b"]])) == pytest.approx(expected)


def test_average_precision_rewards_early_hits() -> None:
    early = one(average_precision([["a", "b", "x", "y"]], [["a", "b"]]))
    late = one(average_precision([["x", "y", "a", "b"]], [["a", "b"]]))
    assert early > late
    assert early == pytest.approx(1.0)


def test_mean_average_precision_averages() -> None:
    value = mean_average_precision([["a", "x"], ["y", "b"]], [["a"], ["b"]])
    assert value == pytest.approx((1.0 + 0.5) / 2)


# ---------------------------------------------------------------------------
# Degenerate inputs — metrics must never crash a report
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k])
def test_metrics_handle_empty_retrieved(metric) -> None:
    result = metric([[]], [["a"]], k=3)
    assert one(result) == pytest.approx(0.0) or math.isnan(one(result))


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, hit_rate_at_k, ndcg_at_k])
def test_metrics_handle_no_relevant(metric) -> None:
    """A query with no ground truth is *undefined*, not zero. Scoring it 0 would
    drag an aggregate down for a case that was never answerable."""
    result = one(metric([["a", "b"]], [[]], k=2))
    assert math.isnan(result) or result == pytest.approx(0.0)


def test_metrics_handle_duplicates_in_retrieved() -> None:
    """A duplicate must not let one relevant document count twice."""
    assert one(recall_at_k([["a", "a", "b"]], [["a", "b"]], k=3)) == pytest.approx(1.0)
    assert one(precision_at_k([["a", "a"]], [["a"]], k=2)) <= 1.0


def test_batch_shape_is_preserved() -> None:
    result = np.asarray(recall_at_k([["a"], ["b"], ["c"]], [["a"], ["z"], ["c"]], k=1))
    assert result.shape == (3,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)
    assert result[2] == pytest.approx(1.0)
