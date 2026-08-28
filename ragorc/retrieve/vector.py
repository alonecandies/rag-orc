"""Dense-only retrieval, and the multi-variant fan-out every translator needs.

This is the semantic half of hybrid search on its own: one named vector, one
Qdrant branch, cosine similarity out. The hybrid retriever prefers a single
server-side query that does dense *and* sparse in one round trip (ADR-0003), so
this class exists for the three cases where that is not what you want:

* **Dense-only deployments** — no sparse index, nothing to fuse.
* **Client-side fusion** — the ensemble retriever needs each modality's own
  ranking, because a fused score cannot be decomposed back into its parts.
* **Query translation** — multi-query, RAG-Fusion, step-back and decomposition
  all produce N query strings, and N searches have to be issued and fused
  regardless of what any single search does internally.

Two decisions carry most of the value here.

**Variants are embedded in one call, not N.** A hosted embedding provider bills
per input token but charges *latency* per request, and a local ONNX session
amortizes its batch overhead the same way: five variants in one call is one
round trip and one forward pass, five calls is five of each. Getting this wrong
is the most common way a "free" query-translation feature turns into the
dominant latency in the pipeline. So all missing variant vectors are embedded
together, once, before any search is issued.

**Variant fusion is RRF, and only when there is more than one list.** Across
variants the score distributions are as comparable as they ever get (same
embedder, same corpus), which argues for DBSF — but the *lists* are not
independent evidence about the same question, they are rephrasings, so what
matters is how many rephrasings agree on a document and how highly. That is
exactly the rank-consensus signal RRF measures, and it is what RAG-Fusion
specifies. When there is a single query text no fusion runs at all: RRF would
replace the cosine similarity with ~1/61, and ``retrieval.score_threshold`` is
calibrated in cosine units. Fusing changes the score *scale*, so it is only done
when it also changes the *order*.

A HyDE-populated ``Query.dense`` is reused rather than recomputed. HyDE embeds a
hypothetical answer document, not the question, so re-embedding ``query.text``
here would silently throw the whole technique away.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from typing import Any

import structlog

from ragorc.core.concurrency import gather_dict
from ragorc.core.errors import BudgetExceeded, GuardrailViolation, RetrievalError
from ragorc.core.models import FloatArray, Query, ScoredChunk
from ragorc.core.protocols import DenseEmbedder
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.retrieve.fusion import reciprocal_rank_fusion
from ragorc.security.tenancy import scope_filter
from ragorc.stores.qdrant.store import QdrantStore

log = structlog.get_logger(__name__)

__all__ = ["VectorRetriever", "clone_for_variant", "resolve_filters", "run_variants"]


# ---------------------------------------------------------------------------
# Shared variant plumbing (also used by the sparse and BM25 retrievers)
# ---------------------------------------------------------------------------
def resolve_filters(
    query: Query, kwargs: Mapping[str, Any], settings: Settings
) -> tuple[dict[str, Any], str | None]:
    """Merge caller filters with the query's own and apply tenant scoping.

    :func:`ragorc.security.tenancy.scope_filter` is the only sanctioned way to
    build a store filter: it fails closed when isolation is on and no tenant is
    known, and it rejects a caller trying to *override* the scope rather than
    quietly honouring the override. Both failure modes are data leaks, and both
    are silent if the predicate is assembled by hand at each call site.
    """
    tenant = kwargs.get("tenant_id") or query.tenant_id or settings.tenant_id
    filters = dict(query.filters)
    extra = kwargs.get("filters")
    if extra:
        filters.update(extra)
    return scope_filter(filters, tenant, settings.security), tenant


def clone_for_variant(
    query: Query,
    text: str,
    *,
    filters: dict[str, Any],
    top_k: int,
    dense: FloatArray | None = None,
    sparse: Any | None = None,
) -> Query:
    """A shallow :class:`Query` for one variant, carrying its own vectors.

    Cloning rather than mutating matters: the searches run concurrently, and one
    shared ``Query`` whose ``dense`` field is rewritten by each branch is a race
    that produces plausible-looking wrong results. ``original`` is preserved so
    the answer can still be graded against what the user actually asked.
    """
    return Query(
        text=text,
        original=query.original,
        filters=filters,
        top_k=top_k,
        dense=dense,
        sparse=sparse,
        tenant_id=query.tenant_id,
        # Copied, not shared: concurrent branches holding one mutable dict is the
        # same class of bug as sharing the vector field.
        metadata=dict(query.metadata),
    )


async def run_variants(
    jobs: Mapping[str, Coroutine[Any, Any, list[ScoredChunk]]],
    *,
    label: str,
    limit: int,
) -> tuple[dict[str, list[ScoredChunk]], dict[str, str]]:
    """Run per-variant searches concurrently, degrading on partial failure.

    Three tiers, deliberately different:

    * A guardrail or budget rejection is re-raised. Those are *decisions*, not
      outages — retrying or degrading past a tenant-isolation failure would turn
      a working guard into a data leak.
    * Some variants failing degrades to the ones that worked. Four rephrasings
      out of five is a slightly narrower recall net, not a failed query.
    * *Every* variant failing re-raises. Returning an empty list here would tell
      the pipeline "the corpus has nothing on this", which is a different and much
      worse answer than "this store is down" — and the hybrid and ensemble layers
      are written to catch the latter and answer from the other stores.
    """
    raw = await gather_dict(jobs, limit=limit, return_exceptions=True)
    ok: dict[str, list[ScoredChunk]] = {}
    errors: dict[str, str] = {}
    first: Exception | None = None
    for name, value in raw.items():
        if not isinstance(value, BaseException):
            ok[name] = value
            continue
        if not isinstance(value, Exception) or isinstance(value, asyncio.CancelledError):
            raise value  # cancellation and process-level signals are not results
        if isinstance(value, GuardrailViolation | BudgetExceeded):
            raise value
        errors[name] = f"{type(value).__name__}: {value}"
        first = first or value
        log.warning(
            "retriever_variant_failed",
            retriever=label,
            variant=name,
            error=str(value)[:200],
            error_type=type(value).__name__,
        )
    if not ok and first is not None:
        raise first
    return ok, errors


# ---------------------------------------------------------------------------
# The retriever
# ---------------------------------------------------------------------------
@register("retriever", "vector", "dense")
class VectorRetriever:
    """Dense search over :class:`~ragorc.stores.qdrant.store.QdrantStore`."""

    name = "vector"

    def __init__(
        self,
        store: QdrantStore | None = None,
        *,
        embedder: DenseEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # The store owns the embedder in the normal wiring, so the ingest path and
        # the query path share one loaded ONNX session instead of paying for two.
        self.store = (
            store if store is not None else QdrantStore(self.settings, dense_embedder=embedder)
        )
        self.embedder = embedder or getattr(self.store, "dense_embedder", None)

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> list[ScoredChunk]:
        """Dense retrieval, fanned out over query variants when there are any.

        Keyword arguments: ``filters``, ``tenant_id``, ``use_variants`` (default
        ``True``; the hybrid retriever turns it off when it is doing its own
        fan-out).
        """
        k = int(top_k or query.top_k or self.settings.retrieval.top_k)
        filters, tenant = resolve_filters(query, kw, self.settings)
        use_variants = bool(kw.get("use_variants", True))

        texts = list(query.all_texts) if use_variants else [query.text]
        vectors = await self.embed_texts(query, texts)

        # Drop variants we could not embed rather than searching with a stale
        # vector; the primary is guaranteed present by _embed_all.
        plan = [(text, vec) for text, vec in zip(texts, vectors, strict=True) if vec is not None]

        jobs: dict[str, Coroutine[Any, Any, list[ScoredChunk]]] = {}
        for i, (text, vector) in enumerate(plan):
            name = "dense" if i == 0 else f"dense_v{i}"
            variant = clone_for_variant(query, text, filters=filters, top_k=k, dense=vector)
            jobs[name] = self.store.search_dense(
                variant, top_k=k, filters=filters, tenant_id=tenant
            )

        with timed("retrieve.vector", variants=len(jobs), top_k=k):
            results, errors = await run_variants(
                jobs, label=self.name, limit=self.settings.retrieval.max_concurrent_retrievers
            )

        if len(results) <= 1:
            # One list: nothing to fuse, and fusing would replace cosine
            # similarity with an RRF score that no threshold is calibrated for.
            single = next(iter(results.values()), [])
            for rank, item in enumerate(single):
                item.rank = rank
            return single[:k]

        # A mapping input names each list after the variant that produced it, so
        # the audit trail in component_scores says which rephrasing found what.
        fused = reciprocal_rank_fusion(results, self.settings.retrieval.rrf_k, top_k=k)
        log.debug(
            "vector_variants_fused",
            variants=len(results),
            failed=len(errors),
            candidates=len(fused),
        )
        return fused

    # -- embedding ---------------------------------------------------------
    async def embed_texts(self, query: Query, texts: Sequence[str]) -> list[FloatArray | None]:
        """Embed every text that has no vector yet, in ONE batched call.

        ``query.dense`` is authoritative for ``texts[0]`` when it is already set:
        that is how HyDE injects the embedding of a hypothetical document, and
        re-deriving it from ``query.text`` would discard the technique. Whatever
        is computed for the primary text is written back onto the query so a
        second store, a retry, or the MMR stage does not embed it again.
        """
        vectors: list[FloatArray | None] = [None] * len(texts)
        pending = list(range(len(texts)))

        # The wire HyDE was missing. `HyDETranslator` produces a hypothetical
        # document, bills an LLM call for it and leaves it on the query — and
        # `hyde_search_vector`, which turns it into the blended search vector, was
        # documented as "called by the retriever instead of the plain query
        # embedding" and called by nothing but its own two unit tests. So the
        # question's vector did the searching and the document was paid for and
        # discarded.
        #
        # Here rather than in the translator because the translator has no
        # embedder, and `query.dense` is exactly the channel the paragraph below
        # already treats as authoritative.
        if query.dense is None and self.embedder is not None:
            from ragorc.translate.hyde import carries_hypothetical, hyde_search_vector

            if carries_hypothetical(query):
                query.dense = await hyde_search_vector(query, self.embedder)

        if query.dense is not None:
            vectors[0] = query.dense
            pending = pending[1:]

        if not pending:
            return vectors
        if self.embedder is None:
            if vectors[0] is None:
                raise RetrievalError(
                    "no dense query vector and no dense embedder",
                    hint="pass embedder=... or set Query.dense",
                )
            log.warning("vector_variants_skipped", reason="no_dense_embedder", n=len(pending))
            return vectors

        batch = [texts[i] for i in pending]
        embedded = await self.embedder.embed_queries(batch)
        for i, vector in zip(pending, embedded, strict=True):
            vectors[i] = vector
        if query.dense is None and vectors[0] is not None:
            query.dense = vectors[0]
        return vectors
