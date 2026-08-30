"""RAPTOR: a tree of recursive abstractive summaries over the corpus.

The retrieval failure this fixes
--------------------------------
Flat chunk retrieval can only answer questions whose evidence fits in a few
chunks. "What did the report conclude about margin pressure?" is not that kind of
question: the conclusion is distributed across eight sections, no single chunk
states it, and every chunk that touches it looks weakly relevant next to a chunk
that happens to repeat the query's words. Raising ``top_k`` does not help — it
adds noise faster than evidence, and the generator now has to do the synthesis
inside its context window with no idea which passages belong together.

RAPTOR (Sarthi et al., 2024, "RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval") builds the missing granularity at *index* time.
Cluster the leaf chunks, have an LLM write one summary per cluster, then treat
those summaries as the next level's input and repeat::

    level 0   the original chunks                         (what the text says)
    level 1   one summary per cluster of chunks           (what a section says)
    level 2   one summary per cluster of summaries        (what the corpus says)

A broad question now has something to retrieve that is actually *about* the
breadth it asks for, and a narrow question still retrieves the leaf — because
every level lives in the same collection and competes in the same ranking.

Why the clustering is a Gaussian mixture and not k-means
--------------------------------------------------------
Topics overlap, and hard assignment lies about that. A chunk discussing
"enterprise pricing in the EU under the new data rules" belongs to the pricing
cluster, the EU-regulation cluster and the enterprise-segment cluster; k-means
picks one and the other two summaries are then written *without* a passage that
was central to them. A Gaussian mixture gives a membership probability per
component, so a chunk joins every cluster where its probability clears
``indexing.raptor_gmm_threshold`` and appears in several summaries. That is not
sloppiness, it is the fact of the matter: the same sentence is evidence for more
than one theme, and a hierarchy that cannot express this produces summaries with
holes in them. This module implements soft membership; approximating it with
``argmax`` would remove the only reason to prefer GMM over k-means.

``n_components`` is selected by BIC over a bounded range. The number of themes
in a corpus is a property of the corpus, not something a user should be asked to
configure, and BIC is the right selector because its parameter penalty stops the
search from buying likelihood with components — a plain likelihood criterion is
maximized by one component per point.

Why clustering happens twice per level
--------------------------------------
UMAP's ``n_neighbors`` *is* the global-versus-local dial: large values preserve
the coarse shape of the manifold and blur neighbourhoods, small values do the
opposite. One pass cannot have both, and on a large corpus a single global pass
collapses everything into a handful of enormous clusters whose summaries read
like a table of contents — technically correct, useless for retrieval, and
extremely expensive because each one has to fit thousands of passages in a prompt.

So each level is clustered as the paper does it: a **global** pass with
``n_neighbors ~ sqrt(n-1)`` finds the broad regions, and then every region larger
than ``raptor_max_cluster_size`` is re-clustered **locally** with
``raptor_umap_neighbors`` neighbours. The result is clusters that are small enough
to summarize honestly and coherent enough to be worth summarizing, at every
corpus size. Dimensionality reduction comes first for a second reason: fitting a
full-covariance mixture to 384- or 1024-dimensional embeddings is a badly
conditioned problem (the covariance has O(d^2) parameters and there are tens of
points), while the same fit over ``raptor_umap_components`` dimensions is stable.

Why the collapsed tree beats top-down traversal
-----------------------------------------------
Every level is written into the *same* collection, tagged with ``Chunk.level``,
so an ordinary search queries all of them at once — one flat search over leaves
and summaries together, ranked against each other. There is no second mode and
no switch: retrieval does not know RAPTOR ran.

The alternative is tree traversal: search the root, descend into the best
subtree, repeat. It sounds principled and it is worse, for a structural reason.
The right level of abstraction for a question is a property of the *question* —
"what was the 2019 revenue" wants a leaf, "what is this document arguing" wants
the root — and traversal has to guess that before it has seen any evidence, at
the root, where it knows least. Every wrong descent is unrecoverable: the leaf
that held the answer is in a subtree the walk already abandoned. The collapsed
query has no such commitment, costs one round trip instead of one per level, and
the paper measured it as both simpler and better. This module therefore never
prunes a level from the output.

``indexing.raptor_collapse_tree`` used to sit here as a switch between the two.
It had no behavioural reader — one ``log.info`` field and one ``describe()``
entry — so both settings retrieved identically (``levels=[1, 0, 0, 0, 0, 0, 0,
0]``), and it has been removed rather than implemented: a knob whose False branch
this section argues against is a promise the module does not intend to keep.

Cost, and why it is reported before anything is spent
-----------------------------------------------------
This is the most expensive indexer in the library: one LLM call per cluster per
level, so a 50k-chunk corpus is thousands of calls. :meth:`estimate_llm_calls`
computes that number *before* the first call, logs it, and checks it against the
request's :class:`~ragorc.core.telemetry.CostLedger` so an over-budget build is
refused up front instead of discovered halfway through. Clustering itself is
CPU-bound (UMAP especially) and runs in worker threads throughout.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import BudgetExceeded, EmbeddingError, GuardrailViolation
from ragorc.core.ids import chunk_id, document_id
from ragorc.core.models import Chunk, FloatArray, IntArray, Modality, Usage
from ragorc.core.protocols import LLM, DenseEmbedder
from ragorc.core.registry import register
from ragorc.core.settings import IndexingSettings, Settings, get_settings
from ragorc.core.telemetry import current_ledger, timed
from ragorc.core.tokens import count_tokens, truncate_to_tokens
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["CallForecast", "RaptorIndexer", "RaptorLevel", "RaptorTree"]

_RAPTOR_EXTRA = 'pip install "ragorc[raptor]"'

_SEED = 42
"""Fixed seed for UMAP and the mixture. Determinism is not cosmetic here: chunk
ids are derived from cluster content (:func:`ragorc.core.ids.chunk_id`), so a
non-reproducible clustering would emit a different tree — and therefore a
different set of point ids — on every ingest, turning upserts into duplicates."""

_BIC_SEARCH_WIDTH = 8
"""How many candidate component counts BIC evaluates above the lower bound. Each
candidate is a full mixture fit; the range is bounded because the useful answers
live near ``n / cluster_size`` and searching the whole space would cost more than
the summaries it is planning."""

_CHARS_PER_TOKEN = 4
"""English average, used only to convert ``chunk_size`` (characters) into a
summary token ceiling."""

_MIN_SUMMARY_TOKENS = 96
"""Floor for a cluster summary. Below this the model starts dropping the entities
and figures that make a summary node retrievable at all."""

_MIN_PASSAGE_TOKENS = 64
"""Floor for one passage inside a summarization prompt. A cluster large enough to
push passages under this is a clustering failure, not a prompt-budget problem —
``raptor_max_cluster_size`` exists to prevent it."""

_PROMPT_BUDGET_FRACTION = 0.5
"""Share of ``llm.context_window`` a summarization prompt may occupy. Half,
because the remainder pays for the system prompt, the completion, and the fact
that our token count is an estimate from a different tokenizer than the
provider's."""

_MAX_SOURCE_DOCS = 16
"""Cap on the ``source_document_ids`` list stored in a node's payload. A cluster
can span thousands of documents and the payload is read on every retrieval hit;
the full membership is already in ``children_ids``."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class CallForecast:
    """Predicted LLM spend of a build, per level.

    Two walks are reported. ``per_level`` assumes clusters of the average
    permitted size and is what an operator wants to read; ``per_level_max``
    assumes every cluster lands on ``raptor_min_cluster_size``, which is the true
    ceiling because soft membership lets a node belong to several clusters.
    """

    per_level: tuple[int, ...] = ()
    per_level_max: tuple[int, ...] = ()

    @property
    def total(self) -> int:
        return sum(self.per_level)

    @property
    def upper_bound(self) -> int:
        return sum(self.per_level_max)

    def report(self) -> dict[str, Any]:
        return {
            "levels": len(self.per_level),
            "calls": self.total,
            "calls_upper_bound": self.upper_bound,
            "per_level": list(self.per_level),
        }


@dataclass(slots=True)
class RaptorLevel:
    """One level of summaries, plus how it was produced."""

    level: int
    nodes: list[Chunk] = field(default_factory=list)
    clusters: int = 0
    failed: int = 0
    """Clusters whose summary call failed. The leaves are indexed independently,
    so a failed summary costs abstraction, never content."""
    method: str = "umap+gmm"
    multi_membership: int = 0
    """Nodes from the level below that ended up in more than one cluster — the
    soft-clustering effect, counted so it can be seen rather than assumed."""

    def report(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "nodes": len(self.nodes),
            "clusters": self.clusters,
            "failed": self.failed,
            "method": self.method,
            "multi_membership": self.multi_membership,
        }


@dataclass(slots=True)
class RaptorTree:
    """The built hierarchy. ``leaves`` are the caller's chunks, unmodified except
    for a ``dense`` vector filled in where one was missing."""

    leaves: list[Chunk] = field(default_factory=list)
    levels: list[RaptorLevel] = field(default_factory=list)
    forecast: CallForecast = field(default_factory=CallForecast)
    llm_calls: int = 0

    @property
    def height(self) -> int:
        """Number of summary levels above the leaves."""
        return len(self.levels)

    def summaries(self) -> list[Chunk]:
        return [node for level in self.levels for node in level.nodes]

    def all_nodes(self) -> list[Chunk]:
        """Leaves and summaries together — the collapsed tree, which is exactly
        what gets upserted into the single shared collection."""
        return [*self.leaves, *self.summaries()]

    def report(self) -> dict[str, Any]:
        return {
            "leaves": len(self.leaves),
            "summaries": sum(len(level.nodes) for level in self.levels),
            "height": self.height,
            "llm_calls": self.llm_calls,
            "forecast": self.forecast.report(),
            "levels": [level.report() for level in self.levels],
        }


# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
def _module_present(name: str) -> bool:
    """Importability without importing. ``find_spec`` touches the filesystem, so
    every caller of this runs inside a worker thread already."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken sys.path entry
        return False


def _umap_embed(matrix: FloatArray, *, n_neighbors: int, n_components: int) -> FloatArray:
    """Reduce ``(n, dim)`` embeddings to ``(n, n_components)``.

    ``metric="cosine"`` because the vectors are L2-normalized by every embedder
    in this package, so cosine is the geometry they were trained in.
    ``random_state`` makes the result reproducible at the price of numba's
    parallel code path — the right trade here, since the ids of every node
    downstream depend on this output.
    """
    try:
        import umap
    except ImportError as exc:  # pragma: no cover - guarded by _module_present
        raise ImportError(f"RAPTOR needs umap-learn for clustering: {_RAPTOR_EXTRA}") from exc
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=_SEED,
    )
    return np.ascontiguousarray(reducer.fit_transform(matrix), dtype=np.float32)


def _as_float64(matrix: FloatArray) -> np.ndarray:
    """Widen to float64 for the mixture fit.

    Everything else in this package is float32 — that is the store's wire format
    and half the memory. The mixture is the one exception, because the fit ends in
    a Cholesky factorization of each component's covariance: in float32 that
    factorization fails outright on ordinary UMAP output (components whose spread
    is small next to the layout's scale), and sklearn reports it as an
    "ill-defined empirical covariance" ``ValueError`` that takes the whole build
    down. The array here is ``(n_nodes, raptor_umap_components)`` — a few hundred
    doubles — so the precision costs nothing worth counting.
    """
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _fit_mixture(matrix: FloatArray, n_components: int) -> Any:
    """Fit one full-covariance Gaussian mixture. sklearn is a lazy import."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise ImportError(
            f"RAPTOR needs scikit-learn for Gaussian-mixture clustering: {_RAPTOR_EXTRA}"
        ) from exc
    # Full covariance is what lets a component be an elongated topic rather than
    # a sphere; it is only estimable because UMAP already cut the dimensionality.
    return GaussianMixture(n_components=n_components, random_state=_SEED).fit(matrix)


def _fit_kmeans(matrix: FloatArray, n_clusters: int) -> IntArray:
    """Hard-assignment fallback used when UMAP is not installed."""
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            f"RAPTOR needs scikit-learn (and preferably umap-learn) to cluster: {_RAPTOR_EXTRA}"
        ) from exc
    model = KMeans(n_clusters=n_clusters, random_state=_SEED, n_init=10).fit(matrix)
    return np.asarray(model.labels_, dtype=np.int64)


# ---------------------------------------------------------------------------
# Cluster bookkeeping
# ---------------------------------------------------------------------------
def _dedupe_clusters(clusters: Sequence[IntArray]) -> list[IntArray]:
    """Drop clusters with identical membership.

    Soft clustering plus the local pass can produce the same member set twice
    (two mixture components covering one region). Left in, each duplicate costs
    an LLM call and adds a node that is a paraphrase of its twin — competing with
    it in the ranking and splitting the score it should have had.
    """
    seen: set[bytes] = set()
    out: list[IntArray] = []
    for cluster in clusters:
        members = np.asarray(cluster, dtype=np.int64)
        if members.size == 0:
            continue
        key = members.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(members)
    return out


def _chop(members: IntArray, limit: int) -> list[IntArray]:
    """Split an oversized cluster into contiguous pieces of at most ``limit``.

    The backstop after global and local clustering: the mixture is free to hand
    back a component with 400 members, and a 400-passage prompt is either
    truncated beyond usefulness or refused by the provider. Slicing loses the
    semantic ordering the clustering earned, which is why it runs last and only
    on what the two clustering stages could not size down.
    """
    if limit <= 0 or members.size <= limit:
        return [members]
    return [members[start : start + limit] for start in range(0, members.size, limit)]


# ---------------------------------------------------------------------------
# The indexer
# ---------------------------------------------------------------------------
@register("indexer", "raptor")
class RaptorIndexer:
    """Builds the RAPTOR tree for a set of leaf chunks.

    The embedder must be *the same one the rest of the ingest uses* — including a
    :class:`~ragorc.embed.late_chunking.LateChunkingEmbedder`. Summary nodes are
    stored in the same collection as the leaves and ranked against them, so
    vectors from a second model would put half the collection in a different
    space: no error, no log line, and a ranking that mixes two incomparable
    similarity scales.
    """

    name = "raptor"

    def __init__(
        self,
        llm: LLM,
        embedder: DenseEmbedder,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.settings = settings or get_settings()
        self.config: IndexingSettings = self.settings.indexing
        self.router = router or ModelRouter(self.settings.llm)

    # -- policy -----------------------------------------------------------
    @property
    def summary_tokens(self) -> int:
        """Token ceiling for one summary.

        Derived from ``chunk_size`` so a summary node is roughly the size of the
        leaves it competes with. This matters for ranking, not just for cost: a
        2000-token summary has far more surface area for a query to match than a
        128-token leaf, so oversized summaries would win on length rather than on
        being the right level of abstraction.
        """
        return max(_MIN_SUMMARY_TOKENS, self.config.chunk_size // _CHARS_PER_TOKEN)

    def estimate_llm_calls(self, leaves: int) -> CallForecast:
        """How many summary calls a tree over ``leaves`` chunks will need."""
        sizes = (self.config.raptor_min_cluster_size, self.config.raptor_max_cluster_size)
        average = max(2, (sizes[0] + sizes[1]) // 2)
        return CallForecast(
            per_level=self._walk(leaves, average),
            per_level_max=self._walk(leaves, max(sizes[0], 1)),
        )

    def _walk(self, leaves: int, cluster_size: int) -> tuple[int, ...]:
        """Project the level sizes for an assumed average cluster size."""
        counts: list[int] = []
        remaining = leaves
        for _ in range(max(self.config.raptor_max_levels, 0)):
            if remaining <= max(self.config.raptor_min_cluster_size, 1):
                break
            clusters = max(1, math.ceil(remaining / cluster_size))
            counts.append(clusters)
            if clusters >= remaining:
                break
            remaining = clusters
        return tuple(counts)

    # -- public API -------------------------------------------------------
    async def build(self, chunks: Sequence[Chunk]) -> tuple[RaptorTree, Usage]:
        """Build the tree level by level. Returns the tree and its total usage."""
        leaves = self._prepare_leaves(chunks)
        tree = RaptorTree(leaves=leaves)
        if len(leaves) <= max(self.config.raptor_min_cluster_size, 1):
            log.info("raptor_skipped", leaves=len(leaves), reason="too few chunks to cluster")
            return tree, Usage()

        tree.forecast = self.estimate_llm_calls(len(leaves))
        log.info(
            "raptor_plan",
            leaves=len(leaves),
            max_levels=self.config.raptor_max_levels,
            model=self.router.model_for(Task.RAPTOR_SUMMARY),
            **tree.forecast.report(),
        )
        self._check_budget(tree.forecast)

        usages: list[Usage] = []
        current = leaves
        with timed("raptor.build", leaves=len(leaves)):
            for level in range(1, self.config.raptor_max_levels + 1):
                if len(current) <= max(self.config.raptor_min_cluster_size, 1):
                    log.info("raptor_level_stop", level=level, nodes=len(current), reason="too few")
                    break
                built, usage = await self._build_level(level, current)
                # Accounted before the stop decision, both of them: a level that
                # ends the recursion because every one of its summaries came back
                # empty still *made* those calls, and dropping their usage here
                # under-reports real money in the value the caller bills against.
                usages.append(usage)
                if built is None:
                    break
                tree.llm_calls += built.clusters
                if not built.nodes:
                    break
                tree.levels.append(built)
                current = built.nodes

        if tree.levels:
            # A level is only embedded when the level *above* it is built, so the
            # top level would otherwise reach the store with no vector at all and
            # be rejected there (QdrantStore refuses a point with no vectors,
            # correctly — it would be permanently unretrievable). One extra
            # embedding batch buys the invariant that every node in the tree is
            # upsertable.
            await self._level_vectors(tree.levels[-1].nodes)

        total = Usage.sum(usages)
        log.info("raptor_built", cost_usd=round(total.cost_usd, 6), **tree.report())
        return tree, total

    async def index(self, chunks: Sequence[Chunk]) -> tuple[list[Chunk], Usage]:
        """Build the tree and return every node ready for a single upsert.

        Leaves and summaries go into one collection on purpose: that is what makes
        the collapsed-tree query possible, and the payload index on ``level``
        (see :mod:`ragorc.stores.qdrant.collections`) is what lets a retriever
        restrict to one level when it wants to.
        """
        tree, usage = await self.build(chunks)
        nodes = tree.all_nodes()
        log.info(
            "raptor_indexed",
            nodes=len(nodes),
            leaves=len(tree.leaves),
            summaries=len(nodes) - len(tree.leaves),
            height=tree.height,
        )
        return nodes, usage

    # -- one level --------------------------------------------------------
    async def _build_level(
        self, level: int, nodes: Sequence[Chunk]
    ) -> tuple[RaptorLevel | None, Usage]:
        vectors = await self._level_vectors(nodes)
        clusters, method = await asyncio.to_thread(self._cluster, vectors)
        if not clusters:  # pragma: no cover - _cluster always covers every node
            log.warning("raptor_level_empty", level=level, nodes=len(nodes))
            return None, Usage()
        if len(clusters) >= len(nodes):
            # No compression: the next level would be as large as this one and
            # the recursion would never terminate on anything but the level cap,
            # paying one LLM call per node to restate it.
            log.info(
                "raptor_level_stop",
                level=level,
                nodes=len(nodes),
                clusters=len(clusters),
                reason="clustering produced no abstraction",
            )
            return None, Usage()

        _, memberships = np.unique(np.concatenate(clusters), return_counts=True)
        multi = int(np.count_nonzero(memberships > 1))
        built, usage = await self._summarize_level(level, nodes, clusters, method, multi)
        if built.nodes:
            log.info("raptor_level_built", **built.report())
        else:
            # Every summary failed. The level is unusable and the caller will stop
            # on the empty node list, but it is returned rather than dropped so the
            # calls it already paid for are still counted.
            log.warning("raptor_level_abandoned", level=level, clusters=len(clusters))
        return built, usage

    async def _level_vectors(self, nodes: Sequence[Chunk]) -> FloatArray:
        """Embed whatever this level is missing, then stack into one matrix.

        Vectors already on the chunks are reused: leaves normally arrive embedded
        from the ingest pipeline, and re-embedding them would double the cost of
        the cheapest stage of this indexer for an identical result.
        """
        missing = [node for node in nodes if node.dense is None]
        if missing:
            vectors = await self.embedder.embed_documents([node.embed_text for node in missing])
            if len(vectors) != len(missing):
                raise EmbeddingError(
                    "embedder returned a different number of vectors than texts",
                    expected=len(missing),
                    received=len(vectors),
                )
            for node, vector in zip(missing, vectors, strict=True):
                node.dense = vector
        try:
            matrix = np.stack([np.asarray(node.dense, dtype=np.float32) for node in nodes])
        except ValueError as exc:
            # Ragged stack: two different embedding models produced this level.
            raise EmbeddingError(
                "level vectors have inconsistent dimensions",
                nodes=len(nodes),
                hint="every level must be embedded by one model",
                error=str(exc)[:200],
            ) from exc
        return np.ascontiguousarray(matrix, dtype=np.float32)

    # -- clustering (all of this runs in a worker thread) -----------------
    def _cluster(self, matrix: FloatArray) -> tuple[list[IntArray], str]:
        """Two-stage clustering: global regions, then local topics inside them.

        Returns index arrays into ``matrix``. Clusters may overlap — that is the
        soft-membership contract — and every row appears in at least one.
        """
        total = matrix.shape[0]
        max_size = max(self.config.raptor_max_cluster_size, 1)
        global_clusters, method = self._soft_cluster(
            matrix, stage="global", n_neighbors=self._global_neighbors(total)
        )

        out: list[IntArray] = []
        for members in global_clusters:
            if members.size <= max_size:
                out.append(members)
                continue
            local, _ = self._soft_cluster(
                matrix[members],
                stage="local",
                n_neighbors=self._local_neighbors(int(members.size)),
            )
            for piece in local:
                # `piece` indexes the sub-matrix; map it back to the level's
                # index space before it leaves this function.
                out.extend(_chop(members[piece], max_size))

        clusters = _dedupe_clusters(out)
        log.debug(
            "raptor_clustered",
            nodes=total,
            global_clusters=len(global_clusters),
            clusters=len(clusters),
            method=method,
        )
        return clusters, method

    def _global_neighbors(self, n: int) -> int:
        """``sqrt(n-1)``, as in the paper: the neighbourhood grows with the corpus
        so the global pass keeps seeing global structure instead of resolving
        finer and finer detail as the corpus gets bigger."""
        return max(2, min(n - 1, math.isqrt(max(n - 1, 1))))

    def _local_neighbors(self, n: int) -> int:
        """``raptor_umap_neighbors`` (10 by default), clamped to the sub-cluster."""
        return max(2, min(n - 1, self.config.raptor_umap_neighbors))

    def _soft_cluster(
        self, matrix: FloatArray, *, stage: str, n_neighbors: int
    ) -> tuple[list[IntArray], str]:
        rows = matrix.shape[0]
        if rows <= max(self.config.raptor_max_cluster_size, 1):
            # Already summarizable as one cluster; reducing and fitting a mixture
            # to a dozen points would invent structure rather than find it.
            return [np.arange(rows, dtype=np.int64)], "single"

        if not _module_present("umap"):
            # Explicit, logged degradation. Hard assignment loses the overlapping
            # membership that is the point of GMM, so it is a fallback and never
            # a default — but a single giant cluster would be worse than both.
            log.warning(
                "raptor_umap_unavailable",
                stage=stage,
                nodes=rows,
                effect="hard k-means assignment, no overlapping membership",
                hint=_RAPTOR_EXTRA,
            )
            return self._kmeans_clusters(matrix), "kmeans"

        components = min(self.config.raptor_umap_components, rows - 2)
        if components < 2:
            # UMAP's spectral initialization diagonalizes an (n x n) graph for
            # ``n_components + 1`` eigenvectors, so it needs strictly more rows
            # than that — and a level this small has no manifold to unfold in the
            # first place. Clamping the components instead would silently ask for
            # more eigenvectors than the matrix has and raise from inside scipy.
            log.warning(
                "raptor_umap_too_small",
                stage=stage,
                nodes=rows,
                effect="hard k-means assignment, no overlapping membership",
            )
            return self._kmeans_clusters(matrix), "kmeans"

        reduced = _umap_embed(matrix, n_neighbors=n_neighbors, n_components=components)
        model, chosen, bic = self._select_components(reduced)
        if model is None:
            # No component count in the window produced a usable mixture. Hard
            # assignment loses overlapping membership, but it is a tree; raising
            # here would cost the caller the whole ingest over one bad level.
            log.warning(
                "raptor_mixture_unfittable",
                stage=stage,
                nodes=rows,
                effect="hard k-means assignment, no overlapping membership",
            )
            return self._kmeans_clusters(matrix), "kmeans"
        probabilities = np.asarray(model.predict_proba(_as_float64(reduced)), dtype=np.float64)
        clusters = self._memberships(probabilities)
        log.debug(
            "raptor_soft_cluster",
            stage=stage,
            nodes=rows,
            n_neighbors=n_neighbors,
            components=components,
            selected_k=chosen,
            bic=round(bic, 2),
            clusters=len(clusters),
        )
        return clusters, "umap+gmm"

    def _select_components(self, reduced: FloatArray) -> tuple[Any, int, float]:
        """Pick ``n_components`` by BIC over a range the size limits allow.

        The range is derived, not guessed: at least ``n / max_cluster_size``
        components (or the average cluster is too big to summarize) and at most
        ``n / min_cluster_size`` (or the average cluster is too small to be worth
        a call). BIC then chooses inside that window, and the best *fitted* model
        is kept so the selection pass is not paid for twice.

        A candidate that cannot be fitted at all is skipped rather than fatal: on
        a small or duplicate-heavy level some component counts collapse a
        component onto a single point, and one unusable ``k`` is not a reason to
        lose the other seven. ``None`` comes back only when the whole window
        failed, which the caller turns into the k-means fallback.
        """
        rows = reduced.shape[0]
        lower = max(1, math.ceil(rows / max(self.config.raptor_max_cluster_size, 1)))
        upper = max(lower, min(rows - 1, rows // max(self.config.raptor_min_cluster_size, 1)))
        upper = min(upper, lower + _BIC_SEARCH_WIDTH)
        points = _as_float64(reduced)

        best_model: Any = None
        best_k = lower
        best_bic = math.inf
        for candidate in range(lower, upper + 1):
            try:
                model = _fit_mixture(points, candidate)
                bic = float(model.bic(points))
            except ValueError as exc:
                # sklearn's "ill-defined empirical covariance": this k has a
                # degenerate component on this data. Other k values usually do not.
                log.debug("raptor_mixture_fit_failed", k=candidate, error=str(exc)[:120])
                continue
            if bic < best_bic:
                best_model, best_k, best_bic = model, candidate, bic
        return best_model, best_k, best_bic

    def _memberships(self, probabilities: np.ndarray) -> list[IntArray]:
        """Turn responsibilities into (possibly overlapping) index arrays.

        A row joins every component whose probability clears
        ``raptor_gmm_threshold``. Rows that clear nothing — points in a low-density
        gap between components — join their ``argmax`` component, because a leaf
        with no parent is a leaf that no summary at any level covers, and that is
        a hole in the hierarchy rather than a saving.
        """
        mask = probabilities >= self.config.raptor_gmm_threshold
        orphans = np.flatnonzero(~mask.any(axis=1))
        if orphans.size:
            mask[orphans, probabilities[orphans].argmax(axis=1)] = True
        clusters = [np.flatnonzero(mask[:, column]) for column in range(mask.shape[1])]
        return self._enforce_min_size(clusters, probabilities)

    def _enforce_min_size(
        self, clusters: Sequence[IntArray], probabilities: np.ndarray
    ) -> list[IntArray]:
        """Fold clusters below ``raptor_min_cluster_size`` into their best neighbour.

        A one-member cluster produces a summary that is a paraphrase of a single
        chunk: an LLM call and an index entry that compete with the chunk they
        came from. Absorbed members keep their coverage — they are added to the
        surviving cluster they were most probably drawn from, so no node loses its
        parent.
        """
        minimum = max(self.config.raptor_min_cluster_size, 1)
        keep = [
            (column, members) for column, members in enumerate(clusters) if members.size >= minimum
        ]
        if not keep:
            # Every component is under the floor (a tiny or very diffuse level):
            # one cluster over everything is the only honest answer.
            return [np.arange(probabilities.shape[0], dtype=np.int64)]

        covered = np.zeros(probabilities.shape[0], dtype=bool)
        for _, members in keep:
            covered[members] = True
        orphans = np.flatnonzero(~covered)
        if orphans.size == 0:
            return [np.asarray(members, dtype=np.int64) for _, members in keep]

        columns = np.asarray([column for column, _ in keep], dtype=np.int64)
        winners = columns[probabilities[np.ix_(orphans, columns)].argmax(axis=1)]
        adopted: dict[int, list[int]] = {}
        for row, column in zip(orphans.tolist(), winners.tolist(), strict=True):
            adopted.setdefault(column, []).append(row)

        out: list[IntArray] = []
        for column, members in keep:
            extra = adopted.get(column)
            if extra:
                members = np.union1d(members, np.asarray(extra, dtype=np.int64))
            out.append(np.asarray(members, dtype=np.int64))
        return out

    def _kmeans_clusters(self, matrix: FloatArray) -> list[IntArray]:
        """k-means fallback, sized to the configured cluster window."""
        rows = matrix.shape[0]
        target = max(
            2, (self.config.raptor_min_cluster_size + self.config.raptor_max_cluster_size) // 2
        )
        clusters = int(min(max(math.ceil(rows / target), 1), rows))
        labels = _fit_kmeans(matrix, clusters)
        return [np.flatnonzero(labels == label) for label in range(clusters)]

    # -- summarization ----------------------------------------------------
    async def _summarize_level(
        self,
        level: int,
        nodes: Sequence[Chunk],
        clusters: Sequence[IntArray],
        method: str,
        multi_membership: int,
    ) -> tuple[RaptorLevel, Usage]:
        prompt = get_prompt("raptor_summary")
        model = self.router.model_for(Task.RAPTOR_SUMMARY)
        groups = [[nodes[index] for index in cluster.tolist()] for cluster in clusters]

        results = await bounded_gather(
            (self._summarize(prompt, model, group) for group in groups),
            limit=max(1, self.settings.llm.max_concurrency),
            return_exceptions=True,
        )

        built = RaptorLevel(
            level=level, clusters=len(groups), method=method, multi_membership=multi_membership
        )
        usages: list[Usage] = []
        for ordinal, (group, result) in enumerate(zip(groups, results, strict=True)):
            if isinstance(result, BudgetExceeded):
                # A spent budget is a stop signal for the whole build, not one
                # cluster's bad luck: every remaining call would raise too.
                raise result
            if isinstance(result, BaseException):
                # Degrade, do not fail: the children are indexed on their own, so
                # a missing summary costs abstraction at this level and nothing
                # else. Failing the ingest over it would be a worse trade.
                built.failed += 1
                log.warning(
                    "raptor_summary_failed",
                    level=level,
                    cluster=ordinal,
                    children=len(group),
                    error=str(result)[:200],
                )
                continue
            text, usage = result
            usages.append(usage)
            if not text:
                built.failed += 1
                continue
            built.nodes.append(self._make_node(level, ordinal, group, text))
        return built, Usage.sum(usages)

    async def _summarize(
        self, prompt: Any, model: str, children: Sequence[Chunk]
    ) -> tuple[str, Usage]:
        text, usage = await self.llm.complete(
            prompt.render(texts=self._render_passages(children), max_tokens=self.summary_tokens),
            system=prompt.system,
            model=model,
            max_tokens=self.summary_tokens,
            stage="raptor_summary",
        )
        return text.strip(), usage

    def _render_passages(self, children: Sequence[Chunk]) -> str:
        """Join the cluster's texts under a prompt budget.

        Truncation is per passage rather than over the concatenation, so a long
        first chunk cannot consume the window and silently hide the rest of its
        cluster from the summarizer — which would produce a summary that claims to
        cover passages it never saw.
        """
        budget = max(int(self.settings.llm.context_window * _PROMPT_BUDGET_FRACTION), 1)
        per_passage = max(_MIN_PASSAGE_TOKENS, budget // max(len(children), 1))
        parts: list[str] = []
        for child in children:
            content = child.content.strip()
            tokens = child.token_count or count_tokens(content)
            parts.append(
                content if tokens <= per_passage else truncate_to_tokens(content, per_passage)
            )
        return "\n\n".join(parts)

    def _make_node(self, level: int, ordinal: int, children: Sequence[Chunk], text: str) -> Chunk:
        """Wrap a summary as a level-``level`` chunk with a deterministic id."""
        source_ids = sorted({child.document_id for child in children if child.document_id})
        doc_id = self._node_document_id(source_ids, children)
        return Chunk(
            id=chunk_id(doc_id, ordinal, text, level),
            content=text,
            document_id=doc_id,
            index=ordinal,
            # A summary is not a span of any source document. Zero offsets say
            # exactly that; a citation verifier keys off `level > 0` and checks
            # the children instead.
            start_char=0,
            end_char=0,
            level=level,
            children_ids=tuple(child.id for child in children),
            modality=Modality.SUMMARY,
            metadata={
                "raptor": True,
                "raptor_level": level,
                "raptor_cluster": ordinal,
                "children": len(children),
                "source_document_ids": source_ids[:_MAX_SOURCE_DOCS],
            },
            token_count=count_tokens(text),
            tenant_id=children[0].tenant_id,
        )

    def _node_document_id(self, source_ids: Sequence[str], children: Sequence[Chunk]) -> str:
        """Owning document of a summary node.

        A cluster confined to one document keeps that document's id, so
        document-scoped filters and deletes reach its summaries too. A cluster
        spanning several gets a deterministic synthetic id derived from the set it
        covers — synthetic rather than blank, because an empty ``document_id``
        makes the node invisible to every document filter in the system, including
        the delete-by-document path that cleans up after a re-ingest.
        """
        if len(source_ids) == 1:
            return source_ids[0]
        tenant = children[0].tenant_id if children else None
        return document_id(f"raptor:{'|'.join(source_ids)}", tenant_id=tenant)

    # -- guards -----------------------------------------------------------
    def _prepare_leaves(self, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Filter the input down to usable level-0 leaves.

        Summary nodes from an earlier build are dropped rather than re-summarized:
        feeding level 1 back in as leaves would build a second, parallel
        hierarchy over the same text and double every level above it.
        """
        leaves = [chunk for chunk in chunks if chunk.level == 0 and chunk.content.strip()]
        dropped = len(chunks) - len(leaves)
        if dropped:
            log.warning(
                "raptor_leaves_filtered",
                dropped=dropped,
                kept=len(leaves),
                reason="empty content or level > 0",
            )
        tenants = {chunk.tenant_id for chunk in leaves}
        if len(tenants) > 1:
            # A cluster spanning two tenants produces one summary containing both
            # tenants' text, stored under one tenant_id. That is a cross-tenant
            # leak created at index time, which no query-side filter can undo.
            if self.settings.security.enforce_tenant_isolation:
                raise GuardrailViolation(
                    "RAPTOR cannot cluster chunks from more than one tenant",
                    rule="tenant_isolation",
                    tenants=len(tenants),
                    hint="build one tree per tenant",
                )
            log.warning("raptor_mixed_tenants", tenants=len(tenants))
        return leaves

    def _check_budget(self, forecast: CallForecast) -> None:
        """Refuse a build that the request's ledger cannot pay for.

        Checked against the *expected* call count, not the upper bound: refusing
        on the pessimistic figure would reject trees that comfortably fit. The
        ledger still enforces the real ceiling call by call inside the LLM client,
        so this is an early exit, not the only guard.

        And it is an early exit that does nothing when the ledger has no call
        ceiling: ``max_calls is None`` returns before the forecast is compared.
        Worth stating because ``cost.max_llm_calls_per_ingest`` defaults to
        ``None`` — this forecast is not what bounds an ingest.
        """
        ledger = current_ledger()
        if ledger is None:
            return
        ledger.check()
        if ledger.max_calls is None:
            return
        remaining = ledger.max_calls - ledger.total.calls
        if forecast.total > remaining:
            raise BudgetExceeded(
                "RAPTOR tree would exceed the LLM call budget",
                required=forecast.total,
                remaining=remaining,
                limit=ledger.max_calls,
                hint=(
                    "lower indexing.raptor_max_levels, or raise "
                    "cost.max_llm_calls_per_query (query path) / "
                    "cost.max_llm_calls_per_ingest (ingest path)"
                ),
            )

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_levels": self.config.raptor_max_levels,
            "cluster_size": [
                self.config.raptor_min_cluster_size,
                self.config.raptor_max_cluster_size,
            ],
            "gmm_threshold": self.config.raptor_gmm_threshold,
            "umap": [self.config.raptor_umap_neighbors, self.config.raptor_umap_components],
            "summary_tokens": self.summary_tokens,
            "model": self.router.model_for(Task.RAPTOR_SUMMARY),
        }
