"""Retrieval metrics, and which question each one answers.

There are only two questions worth asking about a ranked list, and confusing them
is how retrieval configurations get tuned in the wrong direction.

**"Did we retrieve it at all?"** — recall@k and hit_rate@k. This is a *ceiling*.
Whatever the reranker does afterwards, it can only reorder the candidates the
first stage returned; a passage that was never fetched cannot be promoted into the
answer by any amount of downstream cleverness. If recall@50 is 0.6, no reranker,
no compressor and no prompt will get the pipeline past 0.6 — the remaining 0.4 was
lost before the reranker was called and the only fixes are upstream: a different
embedding model, hybrid instead of dense-only, better chunking, a larger
``fetch_k``.

**"Did we rank it well?"** — MRR, nDCG@k and MAP. These are position-sensitive:
they change when the same set of documents is reordered, which recall cannot. This
is precisely what a reranker improves, and it is measured *within* the ceiling
recall already set.

That distinction is the entire reason ``fetch_k`` and ``top_k`` are separate
settings. ``fetch_k`` (50 by default) buys recall — the ceiling — at the price of
candidates the reranker must score. ``top_k`` (10) is what the generator sees and
is where ranking quality is cashed in. Tuning them as one number forces a trade
that does not exist: fetching 10 and reranking 10 cannot recover the document that
ranked 23rd in the first stage, and no ranking metric will tell you that happened
— only recall@50 vs recall@10 will.

Practical reading order for a new corpus:

1. ``recall@fetch_k`` — the ceiling. If it is low, stop and fix retrieval.
2. ``recall@top_k`` vs ``recall@fetch_k`` — how much the reranker has to work with.
3. ``ndcg@top_k`` / ``mrr`` — whether the reranker is actually earning its latency.
4. ``precision@k`` — how much of the context window is being spent on noise, but
   see the note on incomplete labels below.

On incomplete labels
--------------------
Synthetic datasets (:mod:`ragorc.eval.dataset`) label exactly the chunk a question
was generated from, even though near-duplicates, parents and summaries may answer
it equally well. Under incomplete labels, recall-style metrics stay meaningful (the
labelled chunk either came back or it did not), while precision@k and MAP are
*lower bounds* — an unlabelled-but-correct hit is counted as a miss. Optimizing
precision hard against synthetic labels therefore optimizes partly for noise.

Vectorization
-------------
Ids are strings, so the id-to-hit projection uses a Python set per query: for the
k≤100 range that matters here, a hash lookup beats ``np.isin``'s sort. Everything
after that — the rank discounts, cumulative sums, ideal DCGs and the reductions
over queries — is a single pass over one ``(n_queries, k)`` matrix, which is where
the float work actually is.

Undefined values are ``nan``, never ``0.0``: a query with no labels has no recall,
and scoring it as zero would make an unlabelled dataset look like a broken
retriever. Aggregation uses ``nanmean`` and reports how many queries contributed.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ragorc.core.models import FloatArray, IntArray

__all__ = [
    "DEFAULT_KS",
    "RetrievalReport",
    "average_precision",
    "evaluate_retrieval",
    "hit_rate_at_k",
    "mean_average_precision",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "relevance_matrix",
]

BoolArray = np.ndarray

RetrievedInput = Sequence[Sequence[str]] | Sequence[str]
RelevantInput = Sequence[Collection[str]] | Collection[str]

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
"""The ks worth reporting by default: 1 and 3 for "is the answer at the top",
5-10 for what the generator sees, 20-50 for the recall ceiling that ``fetch_k``
sets. Reporting a single k hides exactly the gap this module exists to expose."""


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------
def _first(items: Collection[Any]) -> Any:
    """First element of any collection, or ``None`` when it is empty.

    Indexing is not enough: ``relevant`` is legitimately a ``set``, which has no
    ``[0]``.
    """
    for item in items:
        return item
    return None


def _is_single_query(retrieved: RetrievedInput, relevant: RelevantInput) -> bool:
    """Whether the arguments describe one query rather than a batch of them.

    A single query is ``(["id1", "id2"], {"id2"})``; a batch is a sequence of
    those. Normally the shape of ``retrieved`` decides it — a batch's elements are
    sequences of ids, a single query's are ids — but ``retrieved=[]`` is ambiguous:
    it reads both as "one query that returned nothing" and as "no queries at all".
    A query that returned nothing is a first-class case here (``n_empty`` and
    ``n_returned`` exist to report it), so the tie is broken on ``relevant``, whose
    elements are label *sets* for a batch and ids for one query.
    """
    head = _first(retrieved)
    if head is not None:
        return isinstance(head, str)
    label_head = _first(relevant)
    return isinstance(label_head, str)


def _as_batch(
    retrieved: RetrievedInput, relevant: RelevantInput
) -> tuple[list[Sequence[str]], list[set[str]]]:
    """Accept one query or a batch of them."""
    rows: list[Sequence[str]]
    labels: list[set[str]]
    if _is_single_query(retrieved, relevant):
        # `_is_single_query` narrows both arguments at once, which no TypeGuard
        # can express — a TypeGuard narrows one parameter. The cast records what
        # the check just established rather than silencing the complaint.
        rows = [list(cast("Sequence[str]", retrieved))]
        labels = [set(cast("Sequence[str]", relevant))]
    else:
        rows = [list(row) for row in retrieved]
        labels = [set(item) for item in relevant]
    if len(rows) != len(labels):
        raise ValueError(
            f"retrieved/relevant length mismatch: {len(rows)} result lists "
            f"vs {len(labels)} label sets"
        )
    return rows, labels


def relevance_matrix(
    retrieved: RetrievedInput,
    relevant: RelevantInput,
    k: int,
    *,
    gains: Sequence[Mapping[str, float]] | None = None,
) -> tuple[BoolArray, IntArray, FloatArray, IntArray]:
    """Project ids onto a dense ``(n_queries, k)`` matrix.

    Returns ``(hits, n_relevant, gain_matrix, n_returned)``:

    * ``hits`` — boolean, position ``(i, j)`` is True when the j-th result for
      query i is relevant **and has not already been counted**. Padded with False
      where a query returned fewer than k.

      Marking only the *first* occurrence of a duplicated id is a correctness
      requirement, not a nicety. ``recall`` sums the hits and divides by the
      number of labelled documents, so counting one relevant document twice
      yields a recall above 1.0 — the metric would flatter a deduplication bug in
      the retriever rather than expose it, which is the opposite of what an
      instrument is for. Precision, nDCG and MAP inherit the same protection.
    * ``n_relevant`` — how many labelled documents exist per query. Needed by
      recall and MAP, and zero means "this query has no ground truth".
    * ``gain_matrix`` — graded gains for nDCG. Binary labels give 1.0/0.0; a
      ``gains`` mapping supplies per-id relevance grades when a dataset has them.
    * ``n_returned`` — the true length of each result list before padding, so
      "returned nothing" is distinguishable from "returned k misses".
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    rows, labels = _as_batch(retrieved, relevant)
    n = len(rows)
    if gains is not None and len(gains) != n:
        # Silently indexing a short `gains` would raise IndexError halfway through
        # the loop, after some queries had already been scored graded and the rest
        # would have been scored binary — a mixed-scale report is worse than a stop.
        raise ValueError(f"gains/retrieved length mismatch: {len(gains)} vs {n} queries")
    hits = np.zeros((n, k), dtype=bool)
    gain_matrix = np.zeros((n, k), dtype=np.float32)
    n_relevant = np.zeros(n, dtype=np.int64)
    n_returned = np.zeros(n, dtype=np.int64)

    for i, (row, label_set) in enumerate(zip(rows, labels, strict=True)):
        n_relevant[i] = len(label_set)
        window = row[:k]
        n_returned[i] = len(row)
        if not label_set or not window:
            continue
        grades = gains[i] if gains is not None else None
        # Credit each relevant id at most once, at its earliest rank.
        counted: set[str] = set()
        flags: list[bool] = []
        for item in window:
            is_new_hit = item in label_set and item not in counted
            if is_new_hit:
                counted.add(item)
            flags.append(is_new_hit)
        hits[i, : len(flags)] = flags
        if grades is None:
            gain_matrix[i, : len(flags)] = np.asarray(flags, dtype=np.float32)
        else:
            gain_matrix[i, : len(flags)] = [
                float(grades.get(item, 1.0)) if flag else 0.0
                for item, flag in zip(window, flags, strict=True)
            ]
    return hits, n_relevant, gain_matrix, n_returned


def _undefined(n_relevant: IntArray) -> BoolArray:
    """Queries with no ground truth. Their metrics are nan, not zero."""
    return n_relevant == 0


def _aggregate(values: FloatArray) -> float:
    """Mean over the queries where the metric is defined.

    The all-undefined case is checked rather than handed to ``nanmean``, which
    warns and returns nan for an empty slice. Both aggregation paths in this
    module — the standalone :func:`mrr` / :func:`mean_average_precision` and
    :meth:`RetrievalReport.mean` — go through here so a metric cannot read
    differently depending on which one the caller used. It reports 0.0 for
    "nothing was measurable"; ``RetrievalReport.n_labelled`` is what distinguishes
    that from a genuine zero, and the runner omits the retrieval block entirely
    when it is 0.
    """
    if values.size == 0 or bool(np.all(np.isnan(values))):
        return 0.0
    return float(np.nanmean(values))


# ---------------------------------------------------------------------------
# Set metrics — the recall ceiling
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: RetrievedInput, relevant: RelevantInput, k: int = 10) -> FloatArray:
    """Fraction of the labelled documents that appear in the top k. Per query.

    The ceiling metric. A reranker cannot raise it; only the first stage can.
    """
    hits, n_relevant, _, _ = relevance_matrix(retrieved, relevant, k)
    found = hits.sum(axis=1, dtype=np.float64)
    denom = np.maximum(n_relevant, 1).astype(np.float64)
    out = (found / denom).astype(np.float32)
    out[_undefined(n_relevant)] = np.nan
    return out


def precision_at_k(retrieved: RetrievedInput, relevant: RelevantInput, k: int = 10) -> FloatArray:
    """Fraction of the top k that is relevant. Per query.

    Divided by k, not by the number of results returned: a retriever that fills
    only three of ten slots has left seven slots' worth of context window unused,
    and that is a real cost, not a rounding convention.

    On synthetic labels this is a lower bound — see the module docstring.
    """
    hits, n_relevant, _, _ = relevance_matrix(retrieved, relevant, k)
    out = (hits.sum(axis=1, dtype=np.float64) / float(k)).astype(np.float32)
    out[_undefined(n_relevant)] = np.nan
    return out


def hit_rate_at_k(retrieved: RetrievedInput, relevant: RelevantInput, k: int = 10) -> FloatArray:
    """1.0 if *any* labelled document is in the top k, else 0.0. Per query.

    The metric to quote when one passage is enough to answer the question, which is
    the common case for factoid QA. It is also the least noisy of the set metrics
    on small datasets, because it does not divide by a label count that synthetic
    generation fixed at one.
    """
    hits, n_relevant, _, _ = relevance_matrix(retrieved, relevant, k)
    out = hits.any(axis=1).astype(np.float32)
    out[_undefined(n_relevant)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Rank metrics — what the reranker moves
# ---------------------------------------------------------------------------
def reciprocal_rank(
    retrieved: RetrievedInput, relevant: RelevantInput, k: int | None = None
) -> FloatArray:
    """1 / (rank of the first relevant result), 1-indexed. Per query.

    Rewards getting *one* answer to the very top and is indifferent to everything
    below it — which is exactly right when the generator will read the first
    passage and stop, and misleading when the answer needs several passages.
    """
    rows, labels = _as_batch(retrieved, relevant)
    width = k if k is not None else max((len(r) for r in rows), default=1)
    hits, n_relevant, _, _ = relevance_matrix(rows, labels, max(width, 1))
    any_hit = hits.any(axis=1)
    # argmax on a boolean row gives the first True; it returns 0 for all-False
    # rows, which is why the result is masked by `any_hit` afterwards.
    first = hits.argmax(axis=1).astype(np.float64)
    out = np.zeros(len(rows), dtype=np.float32)
    out[any_hit] = (1.0 / (first[any_hit] + 1.0)).astype(np.float32)
    out[_undefined(n_relevant)] = np.nan
    return out


def mrr(retrieved: RetrievedInput, relevant: RelevantInput, k: int | None = None) -> float:
    """Mean reciprocal rank over the batch."""
    return _aggregate(reciprocal_rank(retrieved, relevant, k))


def ndcg_at_k(
    retrieved: RetrievedInput,
    relevant: RelevantInput,
    k: int = 10,
    *,
    gains: Sequence[Mapping[str, float]] | None = None,
) -> FloatArray:
    """Normalized discounted cumulative gain at k. Per query.

    The rank metric to prefer when several documents are relevant and their order
    matters: unlike MRR it keeps counting after the first hit, and unlike
    precision@k it cares *where* in the window each hit landed.

    With binary labels — which is what synthetic generation produces — nDCG reduces
    to a log-discounted hit measure, and the ideal DCG is the first
    ``min(n_relevant, k)`` positions filled. ``gains`` supplies graded relevance
    when a hand-labelled dataset has it.
    """
    rows, labels = _as_batch(retrieved, relevant)
    _hits, n_relevant, gain_matrix, _ = relevance_matrix(rows, labels, k, gains=gains)
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = gain_matrix.astype(np.float64) @ discount

    # The ideal ranking is over the *labels*, not over what came back. Normalizing
    # by the best ordering of the retrieved gains alone would score a run 1.0
    # whenever the few relevant documents it found happen to be in the right
    # relative order — a retriever that missed the only highly-graded passage would
    # look perfect, and nDCG would stop being able to see the miss at all. It would
    # also disagree with the binary branch on gains that are all 1.0, which is the
    # same computation written twice.
    if gains is None:
        ideal_counts = np.minimum(n_relevant, k)
        cumulative = np.concatenate(([0.0], np.cumsum(discount)))
        idcg = cumulative[ideal_counts]
    else:
        idcg = np.zeros(len(rows), dtype=np.float64)
        for i, label_set in enumerate(labels):
            if not label_set:
                continue
            grades = gains[i]
            best = sorted((float(grades.get(item, 1.0)) for item in label_set), reverse=True)[:k]
            idcg[i] = float(np.asarray(best, dtype=np.float64) @ discount[: len(best)])

    out = np.zeros(len(dcg), dtype=np.float32)
    usable = idcg > 0
    out[usable] = (dcg[usable] / idcg[usable]).astype(np.float32)
    out[_undefined(n_relevant)] = np.nan
    return out


def average_precision(
    retrieved: RetrievedInput, relevant: RelevantInput, k: int | None = None
) -> FloatArray:
    """Average of precision@i taken at every rank i that holds a hit. Per query.

    The single number that is sensitive to both how many relevant documents were
    found and how early each one appeared, which is why it is the standard summary
    for multi-answer retrieval. It is also the metric most distorted by incomplete
    labels, because every unlabelled-but-correct hit lowers the precision at every
    subsequent rank.
    """
    rows, labels = _as_batch(retrieved, relevant)
    width = k if k is not None else max((len(r) for r in rows), default=1)
    width = max(width, 1)
    hits, n_relevant, _, _ = relevance_matrix(rows, labels, width)
    positions = np.arange(1, width + 1, dtype=np.float64)
    precision_at_i = np.cumsum(hits, axis=1, dtype=np.float64) / positions
    summed = (precision_at_i * hits).sum(axis=1)
    denom = np.minimum(n_relevant, width).astype(np.float64)
    out = np.zeros(len(rows), dtype=np.float32)
    usable = denom > 0
    out[usable] = (summed[usable] / denom[usable]).astype(np.float32)
    out[_undefined(n_relevant)] = np.nan
    return out


def mean_average_precision(
    retrieved: RetrievedInput, relevant: RelevantInput, k: int | None = None
) -> float:
    """MAP over the batch."""
    return _aggregate(average_precision(retrieved, relevant, k))


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RetrievalReport:
    """Per-query metric vectors plus their aggregates.

    The per-query arrays are kept, not just their means, because the A/B
    comparison in :mod:`ragorc.eval.runner` needs *paired* values to bootstrap:
    two means cannot tell you whether a difference is noise, and the pairing is
    what removes the between-question variance that dominates the estimate.
    """

    n_queries: int
    ks: tuple[int, ...]
    per_query: dict[str, FloatArray] = field(default_factory=dict)
    n_labelled: int = 0
    n_empty: int = 0
    mean_returned: float = 0.0

    def __getitem__(self, metric: str) -> FloatArray:
        return self.per_query[metric]

    def names(self) -> list[str]:
        return list(self.per_query)

    def mean(self) -> dict[str, float]:
        """Aggregates, ignoring queries where a metric is undefined."""
        return {name: _aggregate(values) for name, values in self.per_query.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "queries": self.n_queries,
            "labelled_queries": self.n_labelled,
            "empty_results": self.n_empty,
            "mean_results_returned": round(self.mean_returned, 2),
            "ks": list(self.ks),
            "metrics": {name: round(value, 4) for name, value in self.mean().items()},
        }

    def to_markdown(self) -> str:
        means = self.mean()
        lines = [
            f"| metric | {' | '.join(f'@{k}' for k in self.ks)} |",
            f"|---|{'---|' * len(self.ks)}",
        ]
        for family in ("recall", "precision", "hit_rate", "ndcg"):
            cells = [f"{means.get(f'{family}@{k}', float('nan')):.3f}" for k in self.ks]
            lines.append(f"| {family} | {' | '.join(cells)} |")
        lines.append("")
        lines.append(f"MRR **{means.get('mrr', 0.0):.3f}** · MAP **{means.get('map', 0.0):.3f}** ")
        lines.append(
            f"({self.n_labelled}/{self.n_queries} queries carry labels, "
            f"{self.n_empty} returned nothing)"
        )
        return "\n".join(lines)


def evaluate_retrieval(
    retrieved: RetrievedInput,
    relevant: RelevantInput,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    gains: Sequence[Mapping[str, float]] | None = None,
) -> RetrievalReport:
    """Compute every metric at every k in one pass.

    MRR and MAP are computed over the full returned list rather than at a k: they
    are summaries of the whole ranking, and truncating them to top_k would hide the
    thing they are best at showing — that the answer *was* retrieved, just too far
    down for the generator to see.
    """
    rows, labels = _as_batch(retrieved, relevant)
    n = len(rows)
    widest = max((len(row) for row in rows), default=1)
    valid_ks = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not valid_ks:
        valid_ks = (10,)

    per_query: dict[str, FloatArray] = {}
    for k in valid_ks:
        per_query[f"recall@{k}"] = recall_at_k(rows, labels, k)
        per_query[f"precision@{k}"] = precision_at_k(rows, labels, k)
        per_query[f"hit_rate@{k}"] = hit_rate_at_k(rows, labels, k)
        per_query[f"ndcg@{k}"] = ndcg_at_k(rows, labels, k, gains=gains)
    per_query["mrr"] = reciprocal_rank(rows, labels, widest)
    per_query["map"] = average_precision(rows, labels, widest)

    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    return RetrievalReport(
        n_queries=n,
        ks=valid_ks,
        per_query=per_query,
        n_labelled=int(sum(1 for label in labels if label)),
        n_empty=int((lengths == 0).sum()),
        mean_returned=float(lengths.mean()) if n else 0.0,
    )
