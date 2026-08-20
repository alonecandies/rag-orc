"""Hybrid retrieval: the default retriever, and the two ways it can run.

Hybrid search is not optional in a serious RAG system. Dense retrieval misses
exact identifiers (part numbers, error codes, API names, rare proper nouns)
because an embedding of a token it never saw is an embedding of "something like
that"; lexical retrieval misses paraphrase because it matches strings. Combining
them is the single highest-value retrieval change available, and every other
knob in this package is a refinement on top of it.

Two paths, one contract
-----------------------
**Server-side (preferred).** One ``query_points`` call whose ``prefetch`` list
runs the dense and sparse branches inside Qdrant and whose outer query is a
``FusionQuery`` (RRF or DBSF), optionally nested inside a ColBERT MaxSim rerank
(ADR-0003). One round trip, fusion in Rust next to the data, ``fetch_k`` payloads
crossing the wire once. This is what ``retrieval.server_side_fusion`` selects and
it is on by default.

**Client-side (fallback).** The dense and sparse retrievers run concurrently and
:mod:`ragorc.retrieve.fusion` merges the two ranked lists in this process. Two
round trips, two payload decodes, and a Python merge — measurably slower for
identical recall. It exists for three real reasons: ``server_side_fusion=false``
(what the offline test suite uses), a fusion method Qdrant cannot do itself
(``weighted``, ``relative``, ``max``), and the case where the *per-modality
rankings* are needed rather than just their combination — auditing, evaluation,
or a downstream stage that wants to reweight. Which path ran is logged on every
query, because "why is this slower in staging?" is usually this.

The Postgres full-text leg (``retrieval.use_fulltext``) is always client-side by
construction: it lives in another database, so no server can fuse it with the
vector results. It runs concurrently with the Qdrant call and is fused in. It is
weighted 0.5 by default in ``retrieval.fusion_weights`` because ``ts_rank_cd`` is
a cover-density heuristic with no IDF term, i.e. a weaker ranker than either
vector branch — useful as a third opinion, wrong as a primary vote.

Recall here, precision later
----------------------------
This stage returns ``fetch_k`` candidates, not ``top_k``. That is the central
division of labour in the pipeline: **recall is bought here and precision is
bought by the reranker**. A document this stage does not return can never be
recovered — no reranker, compressor or generator can rank a passage it never
saw — whereas a bad candidate at position 40 costs one cross-encoder forward pass
to discard. So ``top_k`` is treated as a floor on the candidate window, not as
the output size, and the caller who actually needs ``top_k`` results is the
generator, several stages downstream.

:class:`~ragorc.retrieve.noise.NoiseFilter` runs after fusion, not before: fusion
is what reveals the duplicates (the same passage found by dense *and* sparse
search arrives twice) and what makes a relative score cutoff meaningful (there is
one score scale to be relative to).

State lives in the return value, never on the instance. One retriever object
serves concurrent requests; per-request diagnostics stored on ``self`` would be a
data race whose symptom is a plausible wrong answer rather than a crash — hence
:meth:`HybridRetriever.retrieve_detailed`.
"""

from __future__ import annotations

from collections.abc import Coroutine
from typing import Any, cast

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import FloatArray, Query, RetrievalResult, ScoredChunk, SparseVector
from ragorc.core.protocols import RelationalStore
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.retrieve.ensemble import LegResult, run_legs
from ragorc.retrieve.fusion import fuse
from ragorc.retrieve.noise import NoiseFilter
from ragorc.retrieve.sparse import SparseRetriever
from ragorc.retrieve.vector import VectorRetriever, clone_for_variant, resolve_filters
from ragorc.stores.qdrant.store import QdrantStore

log = structlog.get_logger(__name__)

__all__ = ["HybridRetriever"]


_PRESENT = object()
"""Sentinel for "this retriever has no ``embedder`` attribute to inspect"."""


def _ready(retriever: Any, vector: Any) -> bool:
    """Can this leg produce a query representation at all?

    True when the query already carries the vector, or when the retriever owns an
    embedder that can make one. A retriever that does not expose ``embedder``
    (someone's own implementation) is assumed capable — guessing that a
    third-party leg is broken would silently disable it.
    """
    if vector is not None:
        return True
    embedder = getattr(retriever, "embedder", _PRESENT)
    return embedder is not None


@register("retriever", "hybrid", "default")
class HybridRetriever:
    """Dense + sparse (+ optional Postgres full-text), fused and denoised."""

    name = "hybrid"

    def __init__(
        self,
        store: QdrantStore | None = None,
        *,
        postgres: RelationalStore | None = None,
        dense: VectorRetriever | None = None,
        sparse: SparseRetriever | None = None,
        noise: NoiseFilter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store if store is not None else QdrantStore(self.settings)
        # The sub-retrievers share the store, and therefore its embedders: the
        # dense and sparse ONNX sessions are loaded once per process, not once
        # per retriever.
        self.dense = dense or VectorRetriever(self.store, settings=self.settings)
        self.sparse = sparse or SparseRetriever(self.store, settings=self.settings)
        # Typed as the protocol rather than as PostgresStore so a dense-only
        # deployment never imports the relational stack to satisfy an annotation.
        self.postgres = postgres
        self.noise = noise or NoiseFilter(self.settings)

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> list[ScoredChunk]:
        """Candidates for the reranker. Returns ``fetch_k``, not ``top_k``."""
        result = await self.retrieve_detailed(query, top_k=top_k, **kw)
        return result.chunks

    async def retrieve_detailed(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> RetrievalResult:
        """Run the hybrid retrieval and report per-leg results, timings and errors.

        Keyword arguments: ``filters``, ``tenant_id``, ``fetch_k``, ``use_dense``,
        ``use_sparse``, ``use_fulltext``, ``use_variants``, ``server_side``.
        """
        rs = self.settings.retrieval
        k = int(top_k or query.top_k or rs.top_k)
        fetch_k = max(int(kw.pop("fetch_k", None) or rs.fetch_k), k)
        filters, tenant = resolve_filters(query, kw, self.settings)

        use_dense = bool(kw.get("use_dense", rs.use_dense))
        use_sparse = bool(kw.get("use_sparse", rs.use_sparse))
        use_fulltext = bool(kw.get("use_fulltext", rs.use_fulltext)) and self.postgres is not None
        use_variants = bool(kw.get("use_variants", True))
        texts = list(query.all_texts) if use_variants else [query.text]

        # A modality with neither a vector on the query nor an embedder to produce
        # one is unavailable no matter what the settings say. Resolving that here
        # is what stops a dense-only deployment from failing on the default
        # `use_sparse=True` instead of quietly running dense-only.
        if use_dense and not _ready(self.dense, query.dense):
            log.debug("hybrid_leg_unavailable", leg="dense", reason="no_embedder")
            use_dense = False
        if use_sparse and not _ready(self.sparse, query.sparse):
            log.debug("hybrid_leg_unavailable", leg="sparse", reason="no_embedder")
            use_sparse = False

        # Server-side fusion needs two branches to fuse. With one modality enabled
        # there is nothing to fuse server-side, so the single-leg client path is
        # both simpler and equivalent.
        server_side = (
            bool(kw.get("server_side", rs.server_side_fusion)) and use_dense and use_sparse
        )

        if not use_dense and not use_sparse and not use_fulltext:
            log.warning("hybrid_no_legs_enabled", hint="enable retrieval.use_dense/use_sparse")
            return RetrievalResult()

        with timed("retrieve.hybrid", path="server" if server_side else "client", fetch_k=fetch_k):
            if server_side:
                jobs = await self._server_side_jobs(
                    query, texts, fetch_k=fetch_k, filters=filters, tenant=tenant
                )
            else:
                jobs = self._client_side_jobs(
                    query,
                    fetch_k=fetch_k,
                    filters=filters,
                    tenant=tenant,
                    use_dense=use_dense,
                    use_sparse=use_sparse,
                    use_variants=use_variants,
                )
            if use_fulltext:
                jobs["fulltext"] = self._fulltext_job(
                    query, fetch_k=fetch_k, filters=filters, tenant=tenant
                )
            legs = await run_legs(
                jobs,
                timeout_s=rs.per_store_timeout_s,
                limit=rs.max_concurrent_retrievers,
                label=self.name,
            )

        return self._collect(query, legs, fetch_k=fetch_k, server_side=server_side)

    # -- the two paths -----------------------------------------------------
    async def _server_side_jobs(
        self,
        query: Query,
        texts: list[str],
        *,
        fetch_k: int,
        filters: dict[str, Any],
        tenant: str | None,
    ) -> dict[str, Coroutine[Any, Any, list[ScoredChunk]]]:
        """One Qdrant query per query text, each doing dense+sparse+fusion itself.

        Both modalities' vectors for *all* variants are computed first, in two
        batched calls that overlap — not two calls per variant. The store would
        otherwise embed each variant lazily inside its own search, turning N
        variants into 2N forward passes and 2N sequential waits.

        The searches are then independent, so a single-variant query is exactly
        one round trip and a five-variant query is five concurrent ones.
        """
        # One list of two differently-typed coroutines, so the gather's element
        # type collapses to their join. The casts put the two element types back;
        # they hold by construction, since the order of the results is the order
        # of the coroutines.
        embedded = await bounded_gather(
            [
                self.dense.embed_texts(query, texts),
                self.sparse.embed_texts(query, texts),
            ],
            limit=2,
        )
        dense_vectors = cast("list[FloatArray | None]", embedded[0])
        sparse_vectors = cast("list[SparseVector | None]", embedded[1])

        jobs: dict[str, Coroutine[Any, Any, list[ScoredChunk]]] = {}
        for i, text in enumerate(texts):
            dense_vector = dense_vectors[i]
            sparse_vector = sparse_vectors[i]
            if dense_vector is None and (sparse_vector is None or not len(sparse_vector)):
                continue  # nothing to search with; a variant we could not embed
            variant = clone_for_variant(
                query,
                text,
                filters=filters,
                top_k=fetch_k,
                dense=dense_vector,
                sparse=sparse_vector,
            )
            jobs["hybrid" if i == 0 else f"hybrid_v{i}"] = self.store.search(
                variant,
                # `top_k` caps the fused output, `fetch_k` the per-branch
                # candidate window. Both are fetch_k here: this stage is buying
                # recall, and truncating the fused list below the branch limits
                # would throw away candidates we already paid to retrieve.
                top_k=fetch_k,
                filters=filters,
                fetch_k=fetch_k,
                tenant_id=tenant,
            )
        return jobs

    def _client_side_jobs(
        self,
        query: Query,
        *,
        fetch_k: int,
        filters: dict[str, Any],
        tenant: str | None,
        use_dense: bool,
        use_sparse: bool,
        use_variants: bool,
    ) -> dict[str, Coroutine[Any, Any, list[ScoredChunk]]]:
        """One leg per modality, each fanning out over variants internally.

        Delegating the variant fan-out keeps the "one batched embed per modality"
        property (each sub-retriever embeds all its variants in one call) and keeps
        the per-modality rankings intact, which is the reason to be on this path
        at all.
        """
        jobs: dict[str, Coroutine[Any, Any, list[ScoredChunk]]] = {}
        if use_dense:
            jobs["dense"] = self.dense.retrieve(
                query,
                top_k=fetch_k,
                filters=filters,
                tenant_id=tenant,
                use_variants=use_variants,
            )
        if use_sparse:
            jobs["sparse"] = self.sparse.retrieve(
                query,
                top_k=fetch_k,
                filters=filters,
                tenant_id=tenant,
                use_variants=use_variants,
            )
        return jobs

    def _fulltext_job(
        self, query: Query, *, fetch_k: int, filters: dict[str, Any], tenant: str | None
    ) -> Coroutine[Any, Any, list[ScoredChunk]]:
        """Postgres full-text as a third opinion, on the primary query text only.

        Deliberately not fanned out over variants: this leg is weighted 0.5, and
        multiplying load on the relational database by the variant count to
        sharpen a half-weight vote is the wrong trade — the vector legs already
        cover the rephrasings.
        """
        assert self.postgres is not None
        return self.postgres.fulltext_search(
            query.text, top_k=fetch_k, filters=filters, tenant_id=tenant
        )

    # -- assembly ----------------------------------------------------------
    def _collect(
        self,
        query: Query,
        legs: list[LegResult],
        *,
        fetch_k: int,
        server_side: bool,
    ) -> RetrievalResult:
        rs = self.settings.retrieval
        result = RetrievalResult()
        lists: dict[str, list[ScoredChunk]] = {}
        for leg in legs:
            result.timings_ms[leg.name] = round(leg.ms, 2)
            if leg.error is not None:
                # Degrade, do not fail: a dead store costs recall, not the query.
                result.errors[leg.name] = leg.error
                continue
            result.per_store[leg.name] = leg.chunks
            if leg.chunks:
                lists[leg.name] = leg.chunks
        result.total_candidates = sum(len(v) for v in result.per_store.values())

        if not lists:
            log.warning(
                "hybrid_empty",
                path="server" if server_side else "client",
                legs=len(legs),
                errors=len(result.errors),
            )
            return result

        if len(lists) == 1:
            # Nothing to fuse. Passing through preserves the store's own score
            # scale, which is what `score_threshold` is calibrated against.
            fused = next(iter(lists.values()))
        else:
            fused = fuse(
                lists,
                rs.fusion,
                weights=rs.fusion_weights,
                top_k=None,
                settings=self.settings,
            )

        # Denoise last, on the fused list, because fusion is what *creates* the
        # duplicates (the same passage found by two legs) and what gives the
        # relative cutoff a single scale to be relative to.
        #
        # Worth knowing about that cutoff on this path: RRF scores are ~1/rrf_k
        # per contributing leg, so `relative_score_cutoff` stops being a
        # similarity floor and becomes a *consensus* filter — a chunk that only
        # one leg returned can fall below 35% of a chunk that three legs agreed
        # on, even at rank 0 of its own list. For hybrid search that is usually
        # the behaviour you want; set `relative_score_cutoff=None` when a single
        # store's unique finds matter more than agreement between stores.
        kept, report = self.noise.apply(fused, top_k=fetch_k, query_vector=query.dense)
        result.chunks = kept
        log.info(
            "hybrid_retrieved",
            path="server_side_fusion" if server_side else "client_side_fusion",
            fusion=rs.fusion.value,
            legs=sorted(lists),
            dropped_legs=sorted(result.errors),
            candidates=result.total_candidates,
            kept=len(kept),
            removed=report.removed,
            fetch_k=fetch_k,
        )
        return result
