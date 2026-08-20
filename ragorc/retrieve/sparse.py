"""Lexical retrieval expressed as a sparse vector search inside Qdrant.

Dense retrieval fails on exactly the queries where the user knows what they are
looking for: part numbers, error codes, API names, SKUs, surnames, version
strings. An embedding of ``ERR_2148204812`` is an embedding of "some error code",
so the nearest neighbours are other error codes. Lexical matching is not a legacy
fallback, it is the half of hybrid search that handles the identifiers, and the
literature is consistent that dense-plus-lexical beats either alone on any
realistic corpus.

Why the lexical half is a *vector*
----------------------------------
The conventional answer is a second engine — Elasticsearch, or Postgres
full-text — which means two systems to operate, an index-sync job, and a
consistency window during which the two disagree. Instead both BM25 and SPLADE
are stored as Qdrant sparse vectors (ADR-0003), so lexical search is the same API
call as semantic search against the same points, and hybrid fusion can happen
server-side because both branches live in one engine.

Two model families come through this class, and the difference matters:

* **BM25** (``Qdrant/bm25``): a hashed bag of stemmed terms. The client-side
  document weights carry term-frequency saturation and length normalization but
  deliberately *not* IDF, because IDF depends on corpus statistics the client
  does not have. Qdrant's ``Modifier.IDF`` supplies it at query time from the
  live collection, which is what makes this true BM25 rather than an
  approximation of it. Source is reported as ``BM25``.
* **SPLADE** (``prithivida/Splade_PP_en_v1``): a transformer that emits weights
  over the whole vocabulary, so "car" retrieves "automobile" without an embedding
  space. Better than BM25 on paraphrase, roughly 4x the index size, and *not*
  lexical in the strict sense — so it is reported as ``SPARSE`` and the
  distinction stays visible to fusion and to the evaluation harness.

Scale, and what must not be done to it
--------------------------------------
Sparse scores are dot products of unbounded term weights: corpus-dependent,
query-length-dependent, and nowhere near [0, 1]. Three consequences are baked in
here. The store never applies ``retrieval.score_threshold`` on this path (a
cosine floor would empty the result set or pass everything, depending on the
corpus). Fusion with the dense leg is rank-based by default, because normalizing
a BM25 distribution against a cosine distribution requires knowing both
distributions for *this* query. And an empty query vector — a query made entirely
of stopwords, or of terms absent from the BM25 vocabulary — is a legitimate empty
result, not an error.

Variant fan-out follows the same shape as the dense retriever: every variant's
sparse vector is computed in one batched call, the searches run concurrently, and
the lists are fused with RRF only when there is more than one of them.
"""

from __future__ import annotations

from collections.abc import Coroutine, Sequence
from typing import Any

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import RetrievalError
from ragorc.core.models import Query, ScoredChunk, SparseVector
from ragorc.core.protocols import SparseEmbedder
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.retrieve.fusion import reciprocal_rank_fusion
from ragorc.retrieve.vector import clone_for_variant, resolve_filters, run_variants
from ragorc.stores.qdrant.store import QdrantStore

log = structlog.get_logger(__name__)

__all__ = ["SparseRetriever"]


@register("retriever", "sparse", "splade")
class SparseRetriever:
    """Sparse (BM25 / SPLADE) search over the Qdrant sparse named vector."""

    name = "sparse"

    def __init__(
        self,
        store: QdrantStore | None = None,
        *,
        embedder: SparseEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = (
            store if store is not None else QdrantStore(self.settings, sparse_embedder=embedder)
        )
        self.embedder = embedder or getattr(self.store, "sparse_embedder", None)

    @property
    def is_lexical(self) -> bool:
        """True for BM25-family models, False for learned sparse (SPLADE).

        Read from the embedder when one is attached, because the model name in
        settings can be overridden per-instance; the settings flag is only the
        fallback for a store that was handed pre-computed vectors.
        """
        if self.embedder is not None:
            return bool(getattr(self.embedder, "is_lexical", True))
        return not self.settings.embedding.use_splade

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> list[ScoredChunk]:
        """Sparse retrieval, fanned out over query variants when there are any.

        Keyword arguments: ``filters``, ``tenant_id``, ``use_variants``.
        """
        k = int(top_k or query.top_k or self.settings.retrieval.top_k)
        filters, tenant = resolve_filters(query, kw, self.settings)
        use_variants = bool(kw.get("use_variants", True))

        texts = list(query.all_texts) if use_variants else [query.text]
        vectors = await self.embed_texts(query, texts)

        plan = [
            (text, vec)
            for text, vec in zip(texts, vectors, strict=True)
            # A zero-length sparse vector matches nothing; issuing the search
            # would spend a round trip to be told so.
            if vec is not None and len(vec) > 0
        ]
        if not plan:
            log.debug("sparse_query_empty", texts=len(texts), lexical=self.is_lexical)
            return []

        jobs: dict[str, Coroutine[Any, Any, list[ScoredChunk]]] = {}
        label = "bm25" if self.is_lexical else "sparse"
        for i, (text, vector) in enumerate(plan):
            name = label if i == 0 else f"{label}_v{i}"
            variant = clone_for_variant(query, text, filters=filters, top_k=k, sparse=vector)
            jobs[name] = self.store.search_sparse(
                variant, top_k=k, filters=filters, tenant_id=tenant
            )

        with timed("retrieve.sparse", variants=len(jobs), top_k=k, lexical=self.is_lexical):
            results, errors = await run_variants(
                jobs, label=self.name, limit=self.settings.retrieval.max_concurrent_retrievers
            )

        if len(results) <= 1:
            single = next(iter(results.values()), [])
            for rank, item in enumerate(single):
                item.rank = rank
            return single[:k]

        fused = reciprocal_rank_fusion(results, self.settings.retrieval.rrf_k, top_k=k)
        log.debug(
            "sparse_variants_fused",
            variants=len(results),
            failed=len(errors),
            candidates=len(fused),
        )
        return fused

    # -- embedding ---------------------------------------------------------
    async def embed_texts(self, query: Query, texts: Sequence[str]) -> list[SparseVector | None]:
        """Compute every missing sparse vector in one batched call.

        ``SparseEmbedder`` only promises ``embed_query`` for a single string, but
        every implementation in this library also offers ``embed_queries``, and
        the batch path is what keeps N variants to one forward pass. The
        single-item fallback is bounded rather than sequential so a third-party
        embedder without the batch method is still concurrent, not serial.
        """
        vectors: list[SparseVector | None] = [None] * len(texts)
        pending = list(range(len(texts)))
        if query.sparse is not None:
            vectors[0] = query.sparse
            pending = pending[1:]

        if not pending:
            return vectors
        if self.embedder is None:
            if vectors[0] is None:
                raise RetrievalError(
                    "no sparse query vector and no sparse embedder",
                    hint="pass embedder=... or set Query.sparse",
                )
            log.warning("sparse_variants_skipped", reason="no_sparse_embedder", n=len(pending))
            return vectors

        batch = [texts[i] for i in pending]
        embed_many = getattr(self.embedder, "embed_queries", None)
        if callable(embed_many):
            embedded = await embed_many(batch)
        else:
            embedded = await bounded_gather(
                [self.embedder.embed_query(text) for text in batch],
                limit=self.settings.retrieval.max_concurrent_retrievers,
            )
        for i, vector in zip(pending, embedded, strict=True):
            vectors[i] = vector
        if query.sparse is None and vectors[0] is not None:
            query.sparse = vectors[0]
        return vectors
