"""Retrieval noise handling.

Recall and precision pull in opposite directions, and a hybrid multi-store
retriever is deliberately tuned for recall: it fetches ``fetch_k`` from each of
several retrievers over several query variants. That is the right choice — a
document the first stage misses can never be recovered — but it means the
candidate set arrives full of noise:

* **Duplicates.** Overlapping chunks, the same passage reached by dense and
  sparse search, and genuinely repeated boilerplate across documents. Each
  duplicate consumes a context slot and re-asserts the same fact, which makes the
  model *more* confident in it — a real failure mode, not just waste.
* **Near-duplicates.** Templated pages, versioned documentation, quoted email
  threads. Not byte-identical, so exact hashing misses them.
* **Weak tail.** Results ranked 30-50 are usually irrelevant; keeping them
  dilutes the context and shifts the strong evidence toward the middle of the
  prompt, where attention is lowest.
* **Redundancy.** Ten chunks that all say the same thing crowd out the one that
  says the other half of the answer. This is what MMR addresses.
* **Contradictions.** Two passages that disagree. Filtering silently would hide
  a real conflict, so the default is to *mark* them and let the generator report
  the disagreement.

Everything here is vectorized. These filters run on every query over up to a few
hundred candidates, so a Python loop computing pairwise similarity would cost
more than the retrieval it is cleaning up.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from ragorc.core.ids import content_hash
from ragorc.core.models import RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

_FloatMatrix = npt.NDArray[np.floating[Any]]
"""A float matrix of unspecified width.

The vectors here are built as float32 and stay float32 at runtime (NEP 50 makes
a Python float a *weak* scalar, so ``float32_array / 1e-9`` is float32). numpy's
stubs predate that rule and widen the same expression to float64, so pinning the
local to ``NDArray[np.float32]`` would report a conversion that never happens.
Width-agnostic keeps the array-ness checked without asserting the wrong width."""

__all__ = ["NoiseFilter", "NoiseReport", "mmr_select", "normalize_scores", "simhash"]

_WORD = re.compile(r"\w+")

#: Words *or operators*, for :func:`simhash`. Deliberately not "words or any
#: punctuation": that separates prose differing only in a trailing ``.`` versus
#: ``!``, which is precisely the near-duplicate the stage exists to collapse.
#:
#: So the set is curated to characters that carry meaning where they appear:
#: comparison and arithmetic operators, which flip what a line of code or
#: configuration *does*. Sentence punctuation (``. , ; : ? " ( )``) is excluded.
#:
#: ``!=`` is matched as a unit rather than by including ``!`` in the class,
#: because ``!`` is an operator in ``x != 0`` and mere emphasis in ``the mat!``,
#: and only the paired form is load-bearing.
_TOKEN = re.compile(r"\w+|!=|[=<>+\-*/%&|^~]+")
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
def normalize_scores(chunks: Sequence[ScoredChunk], *, method: str = "minmax") -> list[ScoredChunk]:
    """Put scores from different retrievers on a comparable scale.

    Necessary before any score-based (as opposed to rank-based) fusion: cosine
    similarity lands in [0, 1], BM25 is unbounded and corpus-dependent, and a
    cross-encoder emits logits that can be negative. Averaging them raw lets
    whichever retriever happens to have the widest range dominate the result.

    ``zscore`` is the right choice when the distributions are roughly normal —
    it preserves relative gaps, where min-max flattens them.
    """
    if not chunks:
        return []
    scores = np.fromiter((c.score for c in chunks), dtype=np.float64, count=len(chunks))
    if method == "zscore":
        std = scores.std()
        normed = (scores - scores.mean()) / std if std > 1e-9 else np.zeros_like(scores)
        # Map to [0,1] with a logistic so downstream weighting stays positive.
        normed = 1.0 / (1.0 + np.exp(-normed))
    else:
        lo, hi = float(scores.min()), float(scores.max())
        normed = (scores - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(scores)
    out: list[ScoredChunk] = []
    for chunk, value in zip(chunks, normed, strict=True):
        clone = chunk.with_score(float(value))
        clone.component_scores = {
            **chunk.component_scores,
            f"raw_{chunk.source.value}": chunk.score,
        }
        out.append(clone)
    return out


#: Sources whose scores are ranks or rank-derived, not similarities.
_RANK_BASED_SOURCES = frozenset({RetrievalSource.FUSED})


def _is_similarity_scale(chunks: Sequence[ScoredChunk]) -> bool:
    """Whether these scores live on a comparable similarity scale.

    Conservative on purpose: a single rank-fused entry disables the relative
    cutoff for the batch, because mixing a rank score into a similarity
    comparison is exactly the error this guards. Being too cautious costs a few
    extra candidates reaching the reranker, which is the stage designed to reject
    them; being too eager costs the answer.
    """
    if not chunks:
        return False
    if any(c.source in _RANK_BASED_SOURCES for c in chunks):
        return False
    # A fusion step records the method it used; trust it over the source enum.
    return not any(
        str(c.explain.get("fusion") or c.explain.get("method") or "").lower()
        in {"rrf", "dbsf", "reciprocal_rank_fusion"}
        for c in chunks
    )


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------
def simhash(text: str, *, bits: int = 64, shingle: int = 3) -> int:
    """SimHash over word shingles.

    Chosen over MinHash because it is a single integer with a metric (Hamming
    distance) that survives small edits, needs no signature matrix, and can be
    computed in one pass.

    Shingles are built over words **and operator runs**, not words alone. Words
    alone were the same mistake the exact-dedupe key made one layer up: they
    discard punctuation, so ``assert x == 0`` and ``assert x != 0`` produce an
    *identical* hash and collapse at Hamming distance 0 — which no threshold can
    separate, so tuning ``near_dupe_threshold`` could not save the distinguishing
    line. For a corpus of code or configuration that is silent data loss.

    Whitespace is still normalized away (that is the difference templated
    near-duplicates actually have), and a run of punctuation collapses to one
    token so ``x  ==  0`` and ``x==0`` still match.
    """
    words = _TOKEN.findall(text.lower())
    if not words:
        return 0
    grams = (
        [" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)]
        if len(words) >= shingle
        else [" ".join(words)]
    )
    vector = np.zeros(bits, dtype=np.int32)
    mask = (1 << bits) - 1
    for gram in grams:
        h = int(content_hash(gram, size=8), 16) & mask
        # Vectorized bit expansion: no per-bit Python loop.
        bit_array = ((h >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.int32)
        vector += bit_array * 2 - 1
    out = 0
    for i in range(bits):
        if vector[i] > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------
def mmr_select(
    chunks: Sequence[ScoredChunk],
    *,
    k: int,
    lambda_mult: float = 0.6,
    query_vector: np.ndarray | None = None,
) -> list[ScoredChunk]:
    """Maximal Marginal Relevance: pick relevant *and* mutually diverse chunks.

    ``score = lambda * relevance - (1 - lambda) * max_similarity_to_selected``

    The greedy loop is unavoidable (each pick depends on the previous ones), but
    the expensive part is not: the full pairwise similarity matrix is computed
    once with a single matmul, and each iteration is then a vector max over a
    slice of it.

    Falls back to relevance order when the chunks carry no dense vectors — there
    is no diversity signal without them, and silently returning an arbitrary
    subset would be worse than returning the top-k.
    """
    if not chunks or k <= 0:
        return []
    if len(chunks) <= k:
        return list(chunks)

    vectors = [c.chunk.dense for c in chunks]
    if any(v is None for v in vectors):
        log.debug("mmr_skipped", reason="missing_dense_vectors", n=len(chunks))
        return list(chunks)[:k]

    matrix: _FloatMatrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-9)
    similarity = matrix @ matrix.T  # one matmul for every pair

    if query_vector is not None:
        q = np.asarray(query_vector, dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        relevance = matrix @ q
    else:
        relevance = np.fromiter((c.score for c in chunks), dtype=np.float32, count=len(chunks))
        span = relevance.max() - relevance.min()
        relevance = (relevance - relevance.min()) / span if span > 1e-9 else np.ones_like(relevance)

    selected: list[int] = [int(np.argmax(relevance))]
    candidates = set(range(len(chunks))) - set(selected)

    while len(selected) < k and candidates:
        idx = np.fromiter(candidates, dtype=np.int64, count=len(candidates))
        redundancy = similarity[np.ix_(idx, selected)].max(axis=1)
        mmr = lambda_mult * relevance[idx] - (1.0 - lambda_mult) * redundancy
        winner = int(idx[int(np.argmax(mmr))])
        selected.append(winner)
        candidates.discard(winner)

    out: list[ScoredChunk] = []
    for rank, i in enumerate(selected):
        clone = chunks[i].with_score(chunks[i].score)
        clone.rank = rank
        clone.explain = {**chunks[i].explain, "mmr": True}
        out.append(clone)
    return out


# ---------------------------------------------------------------------------
# The filter pipeline
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class NoiseReport:
    kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    below_threshold: int = 0
    diversity_dropped: int = 0
    """Dropped by MMR because they were redundant with a higher-ranked pick.

    Only when MMR actually ran. This used to be assigned inside the
    ``mmr_enabled`` branch whichever way the branch went, so when
    :func:`mmr_select` fell back to ``chunks[:k]`` — which it did on every query,
    because nothing on the search path carried a dense vector — plain truncation
    was reported as ``diversity=17``. An operator reading the log had no way to
    see that the feature they enabled was inert."""
    truncated: int = 0
    """Dropped by the plain ``[:k]`` cut, with no diversity signal involved."""

    @property
    def removed(self) -> int:
        return (
            self.exact_duplicates
            + self.near_duplicates
            + self.below_threshold
            + self.diversity_dropped
            + self.truncated
        )


class NoiseFilter:
    """Applies the noise filters in the order that costs least.

    Ordering is deliberate: exact deduplication is a dict lookup, so it runs
    first and shrinks the input for everything after it. Thresholding is a
    comparison. Near-duplicate detection needs a hash per chunk. MMR needs a
    similarity matrix. Doing them in the reverse order would compute a matrix
    over rows that were going to be discarded anyway.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def apply(
        self,
        chunks: Sequence[ScoredChunk],
        *,
        top_k: int | None = None,
        query_vector: np.ndarray | None = None,
    ) -> tuple[list[ScoredChunk], NoiseReport]:
        cfg = self.settings.retrieval
        report = NoiseReport()
        working = list(chunks)
        if not working:
            return working, report

        working.sort(key=lambda c: c.score, reverse=True)

        if cfg.dedupe_enabled:
            working, report.exact_duplicates = self._dedupe_exact(working)

        working, report.below_threshold = self._threshold(working)

        if cfg.dedupe_enabled:
            working, report.near_duplicates = self._dedupe_near(working, cfg.near_dupe_threshold)

        k = top_k or cfg.top_k
        if cfg.mmr_enabled and len(working) > k:
            before = len(working)
            # Asked *before* the call, because `mmr_select`'s fallback is
            # indistinguishable from its success by the return value alone — both
            # give k chunks in relevance order when the candidates happen to be
            # diverse. Attributing on the input is the only honest option.
            ran = all(c.chunk.dense is not None for c in working)
            working = mmr_select(
                working, k=k, lambda_mult=cfg.mmr_lambda, query_vector=query_vector
            )
            dropped = before - len(working)
            if ran:
                report.diversity_dropped = dropped
            else:
                report.truncated = dropped
                log.warning(
                    "mmr_inert",
                    dropped=dropped,
                    hint=(
                        "candidates carry no dense vectors, so retrieval.mmr_enabled "
                        "degraded to plain truncation"
                    ),
                )
        else:
            before = len(working)
            working = working[:k]
            report.truncated = before - len(working)

        for rank, chunk in enumerate(working):
            chunk.rank = rank
        report.kept = len(working)

        if report.removed:
            log.debug(
                "noise_filtered",
                kept=report.kept,
                exact=report.exact_duplicates,
                near=report.near_duplicates,
                threshold=report.below_threshold,
                diversity=report.diversity_dropped,
                truncated=report.truncated,
            )
        return working, report

    # -- individual filters -------------------------------------------------
    @staticmethod
    def _dedupe_exact(chunks: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
        """Collapse by chunk id and by normalized content.

        Content hashing as well as id matters: the same passage indexed under two
        documents has two ids and one meaning, and both dense and sparse search
        will happily return both.

        The normalization is case and whitespace *only*. Keying on ``\\w+`` tokens
        instead — the cheaper thing to reach for — throws away every operator, so
        ``a == b`` and ``a != b`` hash alike, as do ``x > 0`` and ``x < 0``, and
        ``Written in C++.`` and ``Written in C.``; any two chunks with no word
        characters at all (``'!!!'``, ``'###'``) collapse into one. On a corpus of
        code, configuration or threshold documentation the operator *is* the
        content, so the copy that gets dropped is precisely the one that
        contradicts the copy we keep — the "keeping it would surface a real
        conflict" case this module says it wants to preserve. Cosmetic punctuation
        differences in prose are still removed one stage later by
        :meth:`_dedupe_near`, which is word-shingled and is *allowed* to be fuzzy
        because it is not the stage that claims two passages are identical.
        """
        seen_ids: set[str] = set()
        seen_content: dict[str, ScoredChunk] = {}
        out: list[ScoredChunk] = []
        removed = 0
        for chunk in chunks:
            if chunk.chunk.id in seen_ids:
                removed += 1
                continue
            key = content_hash(_WHITESPACE.sub(" ", chunk.chunk.content).strip().casefold())
            prior = seen_content.get(key)
            if prior is not None:
                # Keep the higher-scoring copy but merge provenance, so "found by
                # both dense and sparse" stays visible to fusion diagnostics.
                prior.component_scores.update(chunk.component_scores)
                prior.explain.setdefault("duplicate_of", []).append(chunk.chunk.id)
                removed += 1
                continue
            seen_ids.add(chunk.chunk.id)
            seen_content[key] = chunk
            out.append(chunk)
        return out, removed

    def _threshold(self, chunks: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
        """Absolute floor plus a *relative* cutoff.

        The relative cutoff is the useful one. An absolute similarity floor is
        wrong per corpus and per embedding model, and it fails in both
        directions: on an easy query everything clears it, on a hard one nothing
        does. A fraction of the top score adapts to the query — if the best hit
        scores 0.42, a hit at 0.14 is noise relative to it, even though 0.14
        might be a fine absolute score elsewhere.

        **Applied only to similarity scores.** A rank-fusion score is not a
        similarity — it is a function of *position*, so "35% of the top score"
        means nothing on it and the outcome depends entirely on the fusion
        constant. Qdrant's server-side RRF uses k=2, giving 1.0, 0.667, 0.4,
        0.286…, so a 0.35 cutoff truncates at **rank 2 for every query ever
        run**: the default retriever was handing the reranker 3 candidates
        instead of 50, and discarding the relevant evidence to do it. Our own
        client-side RRF uses k=60, which clusters scores near 0.028 and made the
        same cutoff silently inert — so the bug looked like a server-side-only
        quirk rather than a category error.

        Rank-based results are already bounded by ``top_k``, which is the correct
        instrument for them.
        """
        cfg = self.settings.retrieval
        before = len(chunks)
        out = chunks
        if cfg.score_threshold is not None:
            out = [c for c in out if c.score >= cfg.score_threshold]

        if cfg.relative_score_cutoff is not None and out and _is_similarity_scale(out):
            floor = out[0].score * cfg.relative_score_cutoff
            # Only meaningful for non-negative scales; a cross-encoder logit can
            # be negative, where a fraction of the top score is not a floor.
            if out[0].score > 0:
                out = [c for c in out if c.score >= floor]
        return out, before - len(out)

    @staticmethod
    def _dedupe_near(chunks: list[ScoredChunk], threshold: float) -> tuple[list[ScoredChunk], int]:
        """Drop near-duplicates, preferring the embedding signal when present.

        With dense vectors, cosine similarity is the better measure and the whole
        matrix is one matmul. Without them, SimHash Hamming distance is the
        fallback; the threshold is converted from a similarity to a bit distance
        so one setting controls both paths.
        """
        if len(chunks) < 2:
            return chunks, 0

        vectors = [c.chunk.dense for c in chunks]
        if not any(v is None for v in vectors):
            matrix: _FloatMatrix = np.asarray(vectors, dtype=np.float32)
            matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
            similarity = matrix @ matrix.T
            np.fill_diagonal(similarity, 0.0)
            keep: list[int] = []
            for i in range(len(chunks)):
                # Chunks are score-sorted, so anything already kept outranks i.
                if keep and float(similarity[i, keep].max()) >= threshold:
                    continue
                keep.append(i)
            return [chunks[i] for i in keep], len(chunks) - len(keep)

        max_distance = round(64 * (1.0 - threshold))
        hashes = [simhash(c.chunk.content) for c in chunks]
        keep_idx: list[int] = []
        for i, h in enumerate(hashes):
            if any(_hamming(h, hashes[j]) <= max_distance for j in keep_idx):
                continue
            keep_idx.append(i)
        return [chunks[i] for i in keep_idx], len(chunks) - len(keep_idx)
