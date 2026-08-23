"""Client-side fusion: several ranked lists in, one ranked list out.

Qdrant already fuses dense and sparse inside the engine (ADR-0003), so why does
this module exist? Because **cross-store** fusion cannot happen there. Merging a
Qdrant result with a Postgres full-text result, a Neo4j path result and a web
result has to happen in our process, and so does fusion across *query variants*
(multi-query, RAG-Fusion), which are N separate searches by construction.
``retrieval.server_side_fusion=false`` also lands here, which is what the
offline test suite uses.

Rank-based versus score-based
-----------------------------
**RRF** (``sum over lists of w / (k + rank)``) only ever looks at rank position.
That is its strength and its weakness in the same sentence: it needs no
knowledge of how any retriever scores, so a cosine similarity in [0, 1], an
unbounded BM25 score and a cross-encoder logit combine without normalization and
without one scale swamping the others. What it throws away is the *margin*: a
list whose top hit scores 0.95 and whose runner-up scores 0.31 contributes
exactly the same 1/61 and 1/62 as a list where the two are 0.72 and 0.71. RRF
cannot tell "one clear winner" from "a coin flip".

**DBSF** keeps the margin. It z-scores each list against its own mean and
standard deviation and then adds the weighted results, so a hit two sigma above
its list's mean outvotes a hit half a sigma above another list's mean. It beats
RRF when the per-list score distributions are *comparable in shape* — two dense
retrievers over the same embedding model, one embedder over several query
variants, dense-plus-SPLADE where both are trained similarities. It is worse
than RRF when the distributions are heterogeneous (BM25's unbounded, long-tailed
scale next to a bounded cosine), when a list is short enough that its mean and
standard deviation are noise rather than statistics, or when a single outlier
inflates the standard deviation and flattens every real difference behind it.
Rule of thumb: same family of scorer, use DBSF; different families, use RRF.

Both are here, plus the three simpler combiners the ``FusionMethod`` enum
names, because the right choice is corpus-dependent and a library that only
ships RRF forces every user to re-implement the alternative.

Design notes
------------
* **Alignment happens once.** Every method operates on the same ``(n_lists,
  n_unique)`` matrix of scores and ranks built by ``_align``, with ``NaN`` for
  "this list did not return this chunk". Every combiner is then two or three
  numpy expressions over that matrix rather than a dict-of-dicts loop, which
  matters because fusion runs on the hot path over ``fetch_k`` x ``n_lists``
  candidates on every single query.

* **Absent is not average.** How a missing entry is filled is the subtlest
  correctness decision in score-based fusion. After min-max normalization,
  0 is the right fill: it means "no worse than the worst thing this list
  returned". After z-scoring, 0 means the list *mean*, so filling with it would
  make being absent from a list better than being returned near the bottom of
  it. DBSF therefore fills one z-unit *below* each list's minimum observed
  z-score — a truncated result list tells us only that the item ranked below
  everything the list did return, and the strict inequality is what keeps that
  claim from collapsing into "absent equals this list's worst hit". For a
  one-result list, where the single hit z-scores to 0 because it has no spread
  to measure, filling with the bare minimum made the whole row zero and deleted
  that list's vote entirely.

* **Per-list normalization is free insurance.** Min-max is a positive affine map
  per list, and RRF, DBSF, weighted and relative fusion are all invariant under
  one (ranks do not move; z-scores are affine-invariant; min-max is idempotent).
  ``max_fusion`` is the single exception, which is exactly why it is the only
  combiner that must be handed raw scores.

* **Provenance survives.** Each output carries the weighted per-list
  contribution under the list's own name in ``component_scores`` (so the values
  add up to — or take the max of — the final score) and the pre-fusion score
  under ``raw_<name>``, matching the convention in
  :func:`ragorc.retrieve.noise.normalize_scores`. A *non-zero* absence fill is a
  contribution too and is reported the same way, listed in
  ``explain["fusion_absent_fill"]`` and lacking a ``raw_<name>`` companion so it
  stays distinguishable from a real vote; without it the components of any DBSF
  row with an absent list came up short of the score they are supposed to
  explain. Ranks land in ``explain``. Without all this, "why is this document
  third?" is unanswerable the moment fusion has collapsed everything into one
  float.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import structlog

from ragorc.core.models import Chunk, FusionMethod, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_RRF_K",
    "FusionInput",
    "Weights",
    "distribution_based_score_fusion",
    "fuse",
    "max_fusion",
    "reciprocal_rank_fusion",
    "relative_score_fusion",
    "weighted_score_fusion",
]

FusionInput = Sequence[Sequence[ScoredChunk]] | Mapping[str, Sequence[ScoredChunk]]
"""Either positional lists (names are inferred from each list's
:class:`RetrievalSource`) or a name -> list mapping, which is what the ensemble
and hybrid retrievers use so the fusion audit trail carries real store names."""

Weights = Mapping[str, float] | Sequence[float] | None

DEFAULT_RRF_K = 60
"""The constant from Cormack et al. 2009, mirrored by ``retrieval.rrf_k``.
Larger flattens the contribution of rank position; smaller makes the head of
each list dominate. 60 is the published default and a sane one — at k=60 the
gap between rank 1 and rank 2 is 1.6%, so RRF is deliberately gentle about
disagreement near the top."""

_EPS = 1e-12
"""Degenerate-spread floor. A one-element list, or a list where every score is
identical, has no spread to normalize against; guarding here is what stops a
divide-by-zero from turning one list into ``NaN`` and poisoning every total."""


# ---------------------------------------------------------------------------
# Merging: the part everyone gets wrong
# ---------------------------------------------------------------------------
def _chunk_richness(chunk: Chunk) -> tuple[int, int, int]:
    """Sort key for "which copy of this chunk do we keep?".

    The same chunk id arrives from several lists with genuinely different
    payloads: Qdrant returns the full payload, a graph retriever may return only
    an id and a title, and a parent-expanded copy carries more text than the
    child it came from. Keeping whichever happened to be seen first would drop
    content that another leg already paid to fetch.
    """
    vectors = (chunk.dense is not None) + (chunk.sparse is not None) + (chunk.multi is not None)
    return (len(chunk.content), vectors, len(chunk.metadata))


@dataclass(slots=True)
class _Alignment:
    """Lists aligned onto one id axis, ready for vectorized combination."""

    names: list[str]
    weights: np.ndarray  # (n_lists,)
    chunks: list[Chunk]  # (n_unique,) richest copy of each chunk
    sources: list[RetrievalSource]  # (n_unique,) first-seen source
    components: list[dict[str, float]]  # (n_unique,) union of incoming component_scores
    explains: list[dict[str, object]]  # (n_unique,) union of incoming explain
    scores: np.ndarray  # (n_lists, n_unique) float64, NaN where absent
    ranks: np.ndarray  # (n_lists, n_unique) int64, -1 where absent

    @property
    def present(self) -> np.ndarray:
        return self.ranks >= 0


def _label_lists(
    result_lists: FusionInput, names: Sequence[str] | None
) -> tuple[list[str], list[list[ScoredChunk]]]:
    """Give every input list a stable, unique name."""
    if isinstance(result_lists, Mapping):
        labels = [str(key) for key in result_lists]
        lists = [list(value) for value in result_lists.values()]
    else:
        lists = [list(value) for value in result_lists]
        if names is not None:
            if len(names) != len(lists):
                raise ValueError(f"got {len(names)} names for {len(lists)} result lists")
            labels = [str(name) for name in names]
        else:
            # Fall back to the retriever that produced the list; an empty list has
            # nothing to introspect, so it gets a positional name.
            labels = [lst[0].source.value if lst else f"list{i}" for i, lst in enumerate(lists)]

    # Multi-query fusion hands over N lists that all say "dense". Suffixing keeps
    # each variant's contribution separately auditable instead of silently
    # collapsing them into one dict key.
    seen: dict[str, int] = {}
    unique: list[str] = []
    for label in labels:
        count = seen.get(label, 0)
        seen[label] = count + 1
        unique.append(label if count == 0 else f"{label}~{count + 1}")
    return unique, lists


def _weight_vector(names: Sequence[str], weights: Weights) -> np.ndarray:
    """Resolve weights to one float per list.

    A mapping is looked up by list name and defaults to 1.0 for anything it does
    not mention — ``retrieval.fusion_weights`` only names the four standard
    retrievers, and an unlisted custom retriever must not silently get 0.
    """
    if weights is None:
        return np.ones(len(names), dtype=np.float64)
    if isinstance(weights, Mapping):
        resolved = [
            float(weights.get(name, weights.get(name.split("~")[0], 1.0))) for name in names
        ]
    else:
        if len(weights) != len(names):
            raise ValueError(f"got {len(weights)} weights for {len(names)} result lists")
        resolved = [float(w) for w in weights]
    vector = np.asarray(resolved, dtype=np.float64)
    if np.any(vector < 0):
        raise ValueError("fusion weights must be non-negative")
    return vector


def _align(result_lists: FusionInput, names: Sequence[str] | None, weights: Weights) -> _Alignment:
    """Build the ``(n_lists, n_unique)`` score/rank matrices.

    Ranks come from each chunk's *position in its list*, not from its ``rank``
    attribute: a caller that concatenated, filtered or re-sorted a list leaves
    stale ranks behind, and RRF over stale ranks is silently wrong.
    """
    labels, lists = _label_lists(result_lists, names)
    weight_vector = _weight_vector(labels, weights)

    # Empty lists contribute nothing to any combiner but would produce all-NaN
    # rows, which makes every row-wise statistic below NaN. Drop them here, once.
    keep = [i for i, lst in enumerate(lists) if lst]
    if len(keep) != len(lists):
        log.debug(
            "fusion_empty_lists_dropped",
            dropped=[label for label, lst in zip(labels, lists, strict=True) if not lst],
        )
    labels = [labels[i] for i in keep]
    lists = [lists[i] for i in keep]
    weight_vector = weight_vector[keep]

    index: dict[str, int] = {}
    chunks: list[Chunk] = []
    sources: list[RetrievalSource] = []
    components: list[dict[str, float]] = []
    explains: list[dict[str, object]] = []
    rows: list[int] = []
    cols: list[int] = []
    raw: list[float] = []
    positions: list[int] = []

    for li, items in enumerate(lists):
        seen_here: set[str] = set()
        rank = 0
        for scored in items:
            cid = scored.chunk.id
            if cid in seen_here:
                # One list containing the same id twice would otherwise vote for
                # itself. Keep the first (best-ranked) occurrence.
                continue
            seen_here.add(cid)
            ci = index.get(cid)
            if ci is None:
                ci = len(chunks)
                index[cid] = ci
                chunks.append(scored.chunk)
                sources.append(scored.source)
                components.append(dict(scored.component_scores))
                explains.append(dict(scored.explain))
            else:
                if _chunk_richness(scored.chunk) > _chunk_richness(chunks[ci]):
                    chunks[ci] = scored.chunk
                # Union, first writer wins: an earlier list's own measurement of
                # its own score is the authoritative one.
                for key, value in scored.component_scores.items():
                    components[ci].setdefault(key, value)
                for key, value in scored.explain.items():
                    explains[ci].setdefault(key, value)
            rows.append(li)
            cols.append(ci)
            raw.append(float(scored.score))
            positions.append(rank)
            rank += 1

    n_lists = len(lists)
    n_unique = len(chunks)
    scores = np.full((n_lists, n_unique), np.nan, dtype=np.float64)
    ranks = np.full((n_lists, n_unique), -1, dtype=np.int64)
    if rows:
        row_idx = np.asarray(rows, dtype=np.int64)
        col_idx = np.asarray(cols, dtype=np.int64)
        scores[row_idx, col_idx] = np.asarray(raw, dtype=np.float64)
        ranks[row_idx, col_idx] = np.asarray(positions, dtype=np.int64)

    return _Alignment(
        names=labels,
        weights=weight_vector,
        chunks=chunks,
        sources=sources,
        components=components,
        explains=explains,
        scores=scores,
        ranks=ranks,
    )


# ---------------------------------------------------------------------------
# Normalization primitives, applied row-wise to the aligned matrix
# ---------------------------------------------------------------------------
def _minmax_rows(align: _Alignment) -> np.ndarray:
    """Per-list min-max into [0, 1]; absent entries become 0.

    A list with no spread (one result, or every score identical) maps to 1.0 for
    the entries it does contain — the same convention as
    :func:`ragorc.retrieve.noise.normalize_scores`, and the only one that does
    not silently delete a single-result list's vote.
    """
    scores = align.scores
    present = align.present
    lo = np.nanmin(scores, axis=1, keepdims=True)
    hi = np.nanmax(scores, axis=1, keepdims=True)
    span = hi - lo
    with np.errstate(invalid="ignore"):
        normed = np.where(span > _EPS, (scores - lo) / np.where(span > _EPS, span, 1.0), 1.0)
    return np.where(present, normed, 0.0)


def _zscore_rows(align: _Alignment) -> np.ndarray:
    """Per-list z-score; absent entries get one z-unit below that list's minimum.

    Qdrant's server-side DBSF instead maps ``mean +/- 3*sigma`` through min-max,
    which is the same statistic squashed into [0, 1]. Raw z is kept here because
    the squash discards the very thing DBSF is chosen for — the size of the gap
    between a strong hit and a mediocre one — and because client-side fusion has
    no bounded-scale requirement to satisfy.

    The ``- 1.0`` on the fill carries more weight than it looks. Filling with the
    bare minimum makes "absent from this list" numerically *identical* to "this
    list's worst returned hit", and on a one-result list it is worse than that:
    a single entry has no standard deviation, so it z-scores to 0, the minimum is
    0, the fill is 0, and the entire row is zero — a retriever that returned
    exactly one confident hit voted for nothing at all (observed with a graph leg
    whose only hit landed tied-last behind three dense hits it never saw). One
    z-unit of separation is the weakest statement that still orders a truncated
    list's own results ahead of what it never returned, which is the only
    ordering claim truncation supports. The degenerate row keeps z=0 rather than
    an invented positive: a list with no measured spread has no magnitude to
    report, only a preference, and the fill is what expresses the preference.
    """
    scores = align.scores
    present = align.present
    mean = np.nanmean(scores, axis=1, keepdims=True)
    std = np.nanstd(scores, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        z = np.where(std > _EPS, (scores - mean) / np.where(std > _EPS, std, 1.0), 0.0)
    z = np.where(present, z, np.nan)
    floor = np.nanmin(z, axis=1, keepdims=True) - 1.0
    return np.where(present, z, floor)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def _assemble(
    align: _Alignment,
    contributions: np.ndarray,
    totals: np.ndarray,
    *,
    method: str,
    top_k: int | None,
    extra_explain: dict[str, object] | None = None,
) -> list[ScoredChunk]:
    """Turn contributions into a ranked ``list[ScoredChunk]``.

    Ordering is ``(-total, best_rank, id)``: score first, then the best rank the
    chunk achieved in any list, then the id. The tiebreakers are not cosmetic —
    RRF produces exact ties constantly (any two chunks each found once at the
    same rank by different lists), and an unstable order there makes retrieval
    non-reproducible across runs, which makes evaluation meaningless.

    ``np.lexsort`` rather than ``argpartition`` on purpose: n here is bounded by
    ``fetch_k * n_lists`` (a few hundred), where the sort is free and total
    determinism is worth more than the asymptotics.
    """
    n = len(align.chunks)
    if n == 0:
        return []

    ranks = align.ranks
    best_rank = np.where(align.present, ranks, np.iinfo(np.int64).max).min(axis=0)
    ids = np.asarray([chunk.id for chunk in align.chunks])
    order = np.lexsort((ids, best_rank, -totals))
    if top_k is not None and top_k < n:
        order = order[:top_k]

    weighted = contributions * align.weights[:, None]
    out: list[ScoredChunk] = []
    for rank, ci in enumerate(order.tolist()):
        present = align.present[:, ci]
        component_scores = dict(align.components[ci])
        per_list_ranks: dict[str, int] = {}
        contributing: list[str] = []
        for li in np.flatnonzero(present).tolist():
            name = align.names[li]
            contributing.append(name)
            per_list_ranks[name] = int(ranks[li, ci])
            # The raw score keeps the pre-fusion measurement (mirroring the
            # ``raw_*`` convention in noise.normalize_scores); the bare name gets
            # the weighted contribution, so the components reconstruct `score`.
            component_scores[f"raw_{name}"] = float(align.scores[li, ci])
            component_scores[name] = float(weighted[li, ci])
        explain = dict(align.explains[ci])
        explain["fusion"] = method
        explain["fusion_sources"] = contributing
        explain["fusion_ranks"] = per_list_ranks
        # Absence is not always free, and when it is not, it is part of `score`.
        # DBSF fills a missing entry *below* the list's worst observed z, so
        # reporting only the lists that voted left the components unable to
        # reconstruct the number they exist to explain — a chunk found by one leg
        # alone reported ``{'graph': 0.0}`` against a score of -1.22, and the
        # audit trail could not answer "why is this last?". Filtering on the
        # value rather than on the method keeps this inert for every combiner
        # that fills with 0 (RRF, min-max, max), where there is nothing to
        # explain and a wall of zero-valued keys would only bury the real votes.
        filled: dict[str, float] = {}
        for li in np.flatnonzero(~present).tolist():
            fill = float(weighted[li, ci])
            if abs(fill) <= _EPS:
                continue
            component_scores[align.names[li]] = fill
            filled[align.names[li]] = fill
        if filled:
            explain["fusion_absent_fill"] = filled
        if extra_explain:
            explain.update(extra_explain)
        out.append(
            ScoredChunk(
                chunk=align.chunks[ci],
                score=float(totals[ci]),
                # FUSED means "more than one list actually voted for *this*
                # chunk", not "fusion ran". A chunk only the graph leg returned
                # keeps GRAPH_LOCAL, which is the provenance the context packer
                # prints and the citation layer resolves against; overwriting it
                # would make every result claim a consensus it never had.
                source=(RetrievalSource.FUSED if len(contributing) > 1 else align.sources[ci]),
                rank=rank,
                component_scores=component_scores,
                explain=explain,
            )
        )
    return out


# ---------------------------------------------------------------------------
# The combiners
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    result_lists: FusionInput,
    k: int = DEFAULT_RRF_K,
    weights: Weights = None,
    *,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: ``sum over lists of w / (k + rank + 1)``.

    Rank is 0-based here and 1-based in the paper, hence the ``+ 1``. Scale-free
    by construction, which is why it is the default for cross-store and
    cross-variant fusion where the score distributions have nothing in common.
    """
    align = _align(result_lists, names, weights)
    if not align.chunks:
        return []
    with np.errstate(divide="ignore"):
        contributions = np.where(align.present, 1.0 / (float(k) + align.ranks + 1.0), 0.0)
    totals = (contributions * align.weights[:, None]).sum(axis=0)
    return _assemble(
        align, contributions, totals, method="rrf", top_k=top_k, extra_explain={"rrf_k": int(k)}
    )


def distribution_based_score_fusion(
    result_lists: FusionInput,
    weights: Weights = None,
    *,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """DBSF: per-list z-score, then weighted sum.

    Use it when the lists come from the same family of scorer, where the *size*
    of a score gap carries information RRF discards. Avoid it when the scales are
    heterogeneous (BM25 next to cosine), when a list is too short for its mean
    and standard deviation to mean anything, or when one outlier can inflate a
    list's standard deviation and flatten every real difference behind it.
    """
    align = _align(result_lists, names, weights)
    if not align.chunks:
        return []
    contributions = _zscore_rows(align)
    totals = (contributions * align.weights[:, None]).sum(axis=0)
    return _assemble(align, contributions, totals, method="dbsf", top_k=top_k)


def weighted_score_fusion(
    result_lists: FusionInput,
    weights: Weights = None,
    *,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Per-list min-max into [0, 1], then weighted sum.

    The bounded cousin of DBSF: it keeps *some* margin information (the position
    of a score within its list's observed range) while being immune to the
    outlier problem that distorts a standard deviation. It pays for that with a
    different distortion — the range is set by two extreme values, so one
    unusually strong hit compresses everything below it.
    """
    align = _align(result_lists, names, weights)
    if not align.chunks:
        return []
    contributions = _minmax_rows(align)
    totals = (contributions * align.weights[:, None]).sum(axis=0)
    return _assemble(align, contributions, totals, method="weighted", top_k=top_k)


def relative_score_fusion(
    result_lists: FusionInput,
    weights: Weights = None,
    *,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Per-list min-max, then the weighted *maximum* across lists.

    "Best evidence wins" rather than "most votes win". Correct when the lists are
    known to have different coverage rather than different opinions — a graph
    retriever that answers 5% of queries superbly should not be penalized for
    contributing nothing to the other 95%, which a sum does and a max does not.
    """
    align = _align(result_lists, names, weights)
    if not align.chunks:
        return []
    contributions = _minmax_rows(align)
    totals = (contributions * align.weights[:, None]).max(axis=0)
    return _assemble(align, contributions, totals, method="relative", top_k=top_k)


def max_fusion(
    result_lists: FusionInput,
    weights: Weights = None,
    *,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Weighted maximum of the *raw* scores.

    The only combiner here that must not be handed normalized input, and the
    only one for which per-list normalization is not a no-op: normalizing first
    turns it into :func:`relative_score_fusion`. Use it when the scores are
    already commensurable — one embedder over several query variants, or one
    reranker applied to several candidate pools — where per-list normalization
    would manufacture differences that do not exist.
    """
    # A zero weight means "do not use this list", so drop it before alignment.
    # Multiplying instead mapped its every score to -0.0, which in an
    # all-negative column is the *largest* value there is — the one setting that
    # disables a source promoted it to first place. Dropping the list here, not
    # masking it later, also keeps chunks that only it contributed out of the
    # output entirely rather than leaving them with a NaN score.
    labels, lists = _label_lists(result_lists, names)
    weight_vector = _weight_vector(labels, weights)
    live = [i for i, w in enumerate(weight_vector) if w > 0.0]
    if not live:
        return []
    if len(live) != len(lists):
        log.debug(
            "fusion_zero_weight_lists_dropped",
            dropped=[labels[i] for i in range(len(labels)) if i not in set(live)],
        )
        lists = [lists[i] for i in live]
        labels = [labels[i] for i in live]
        weights = {labels[j]: float(weight_vector[i]) for j, i in enumerate(live)}

    align = _align(lists, labels, weights)
    if not align.chunks:
        return []
    # NaN marks "absent", and every column has at least one contributor by
    # construction, so nanmax is well defined and needs no fill value.
    contributions = np.where(align.present, align.scores, np.nan)

    # Multiplicative weighting needs a ratio scale, and these scores are raw and
    # may be negative: a cross-encoder logit of -9.0 at weight 0.1 becomes -0.9
    # and outranks a -5.0 at weight 1.0, so down-weighting a list *promotes* it.
    # Uniform positive weights are fine — scaling everything by one constant is
    # order-preserving — so only the mixed case is refused, and it is refused
    # loudly rather than returning a plausible, wrongly ordered list.
    if bool(np.any(contributions < 0.0)) and not bool(np.allclose(align.weights, align.weights[0])):
        raise ValueError(
            "max fusion cannot weight negative scores: multiplying by a weight "
            "below 1 moves a negative score up, so a down-weighted list outranks "
            "the list you trust. Use fusion='relative' (min-max per list, then "
            "max), or drop the per-list weights."
        )

    totals = np.nanmax(contributions * align.weights[:, None], axis=0)
    return _assemble(
        align, np.nan_to_num(contributions, nan=0.0), totals, method="max", top_k=top_k
    )


_DISPATCH = {
    FusionMethod.RRF: "rrf",
    FusionMethod.DBSF: "dbsf",
    FusionMethod.WEIGHTED: "weighted",
    FusionMethod.RELATIVE_SCORE: "relative",
    FusionMethod.MAX: "max",
}


def fuse(
    result_lists: FusionInput,
    method: FusionMethod | str = FusionMethod.RRF,
    *,
    weights: Weights = None,
    k: int | None = None,
    names: Sequence[str] | None = None,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> list[ScoredChunk]:
    """Dispatch to a combiner by :class:`FusionMethod`.

    ``k`` and ``weights`` fall back to ``retrieval.rrf_k`` and
    ``retrieval.fusion_weights`` so a caller that has already resolved settings
    can pass them and a caller that has not still gets the configured behaviour.
    """
    resolved = FusionMethod(method) if not isinstance(method, FusionMethod) else method
    cfg = (settings or get_settings()).retrieval
    if weights is None and cfg.fusion_weights:
        weights = cfg.fusion_weights
    rrf_k = cfg.rrf_k if k is None else k

    if resolved is FusionMethod.RRF:
        return reciprocal_rank_fusion(result_lists, rrf_k, weights, names=names, top_k=top_k)
    if resolved is FusionMethod.DBSF:
        return distribution_based_score_fusion(result_lists, weights, names=names, top_k=top_k)
    if resolved is FusionMethod.WEIGHTED:
        return weighted_score_fusion(result_lists, weights, names=names, top_k=top_k)
    if resolved is FusionMethod.RELATIVE_SCORE:
        return relative_score_fusion(result_lists, weights, names=names, top_k=top_k)
    return max_fusion(result_lists, weights, names=names, top_k=top_k)
