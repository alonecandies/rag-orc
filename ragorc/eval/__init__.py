"""Evaluation harness: synthetic datasets, retrieval metrics, answer metrics, A/B runs.

Read the four modules in this order — it is the order the numbers depend on each
other in:

1. :mod:`~ragorc.eval.dataset` — where labelled questions come from when the
   corpus is private and nobody has any.
2. :mod:`~ragorc.eval.retrieval_metrics` — the recall *ceiling* versus the ranking
   quality measured inside it, which is the distinction that decides whether a
   configuration problem is upstream or downstream of the reranker.
3. :mod:`~ragorc.eval.answer_metrics` — what the generator did with the evidence,
   split so that "faithful but off-topic" and "on-topic but invented" are
   different numbers.
4. :mod:`~ragorc.eval.runner` — running a dataset end to end, and the paired
   bootstrap that decides whether a difference between two runs is real.
"""

from ragorc.eval.answer_metrics import (
    ALL_METRICS,
    AnswerMetrics,
    MetricScore,
    Scorecard,
    cheap_baseline,
    lexical_overlap,
    rouge_l_f1,
    token_f1,
)
from ragorc.eval.dataset import (
    DEFAULT_QUESTIONS_PER_CHUNK,
    MIN_CHUNK_CHARS,
    EvalCase,
    EvalDataset,
    SyntheticQuestionGenerator,
    SyntheticReport,
)
from ragorc.eval.retrieval_metrics import (
    DEFAULT_KS,
    RetrievalReport,
    average_precision,
    evaluate_retrieval,
    hit_rate_at_k,
    mean_average_precision,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_matrix,
)
from ragorc.eval.runner import (
    BootstrapResult,
    CaseResult,
    Comparison,
    EvalRunner,
    RunReport,
    as_answer_fn,
    compare_runs,
    paired_bootstrap,
)

__all__ = [
    "ALL_METRICS",
    "DEFAULT_KS",
    "DEFAULT_QUESTIONS_PER_CHUNK",
    "MIN_CHUNK_CHARS",
    "AnswerMetrics",
    "BootstrapResult",
    "CaseResult",
    "Comparison",
    "EvalCase",
    "EvalDataset",
    "EvalRunner",
    "MetricScore",
    "RetrievalReport",
    "RunReport",
    "Scorecard",
    "SyntheticQuestionGenerator",
    "SyntheticReport",
    "as_answer_fn",
    "average_precision",
    "cheap_baseline",
    "compare_runs",
    "evaluate_retrieval",
    "hit_rate_at_k",
    "lexical_overlap",
    "mean_average_precision",
    "mrr",
    "ndcg_at_k",
    "paired_bootstrap",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevance_matrix",
    "rouge_l_f1",
    "token_f1",
]
