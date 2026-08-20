"""Structured-store retrieval: the relational leg of the three-way split.

Three retrievers here read the same Postgres instance through three different
access paths, and they differ in exactly one respect that the fusion layer cares
about — whether their scores mean anything.

* :class:`PgVectorRetriever` returns cosine *similarity*. pgvector's ``<=>`` is a
  cosine **distance** in ``[0, 2]``; the store converts it with ``1 - distance``
  in the projection (never in the ``ORDER BY``, or the HNSW index is unusable),
  so what arrives here is already higher-is-better, on a scale that is directly
  comparable to Qdrant's dense scores.
* :class:`PgFullTextRetriever` returns ``ts_rank_cd`` squashed into ``(0, 1)`` by
  normalization flag 32, or real BM25 when ParadeDB is enabled. A genuine
  gradient, but on a scale that has nothing to do with cosine similarity: it has
  no upper bound in the BM25 case and no IDF term in the ``ts_rank_cd`` case.
* :class:`SQLRetriever` returns a **constant**, and that has consequences big
  enough to deserve their own section.

Why a SQL row has no relevance score, and what that forces
----------------------------------------------------------
A ``SELECT`` either matched a row or it did not. There is no similarity, no
partial credit, no gradient to report: the query *is* the relevance judgement,
and it already ran. So every chunk this module produces from SQL carries the same
configurable confidence, and the honest reading of that number is "how much do we
trust the text-to-SQL leg in general", not "how relevant is this result".

That single fact determines how SQL may be merged with vector results:

**Score-based fusion is meaningless against a constant.** Weighted sum, DBSF and
relative-score fusion all combine *magnitudes*. A constant has no magnitude
relative to anything — it either sits above the vector leg's score distribution
(and always wins) or below it (and never appears), depending on a distribution
that changes with every query. Tuning the constant does not tune relevance; it
sets a global bias, and the correct value differs per query, which is another way
of saying there is no correct value.

**Rank-based fusion is well defined.** RRF looks only at position. The SQL result
enters at rank 0 and contributes ``1 / (rrf_k + 1)`` no matter what the constant
is, so the merge is insensitive to a number that carries no information. This is
why :class:`~ragorc.retrieve.multi_store.MultiStoreRetriever` defaults to RRF and
why routing a question to both Postgres and Qdrant is only sound under it.

**The constant must not reach a score-based filter unfused.** ``NoiseFilter``'s
``relative_score_cutoff`` drops anything below a fraction of the *top* score. Put
a constant 1.0 at the top of a score-sorted mixed list and the cutoff is suddenly
measuring cosine similarities against a number that is not a similarity — with
the default 0.35 that silently discards every dense hit under 0.35, which on a
hard query is all of them. Fuse first, filter second.

Failure policy
--------------
These are *leaf* retrievers, and they deliberately do **not** swallow
:class:`~ragorc.core.errors.StoreUnavailable`. Only the fan-out layer knows which
store a coroutine belonged to, so only it can record the outage against a name in
:attr:`~ragorc.core.models.RetrievalResult.errors`. Catching it here would turn a
dead database into an empty result set, and an empty result set is indistinguish-
able from "we looked and there was nothing" — the pipeline would then answer
confidently while believing it had consulted a store it never reached.

A *rejected generated query* is the opposite case and is degraded in place: the
guard refusing a statement, or text-to-SQL failing to write one the guard accepts,
says nothing about the health of Postgres and must not fail a request that the
vector leg can still answer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import structlog

from ragorc.construct.text_to_sql import TextToSQLConstructor
from ragorc.core.errors import ConstructionError, GuardrailViolation, RetrievalError
from ragorc.core.models import FloatArray, Query, RetrievalSource, ScoredChunk, Usage
from ragorc.core.protocols import LLM, DenseEmbedder, RelationalStore
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import Timer, timed, trace_step

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_SQL_CONFIDENCE",
    "PgFullTextRetriever",
    "PgVectorRetriever",
    "SQLRetriever",
    "VectorSearchStore",
    "label_and_rank",
]

DEFAULT_SQL_CONFIDENCE = 1.0
"""Confidence assigned to every SQL result.

1.0 rather than something modest, because the number is only ever compared
*within* the SQL leg (where it is constant, so the comparison is vacuous) or
consumed by rank fusion (which ignores it). Picking 0.6 to express "slightly less
trusted than a strong dense hit" would express nothing — see the module docstring
— while making the structured answer the first casualty of any relative cutoff
that runs before fusion.
"""


@runtime_checkable
class VectorSearchStore(Protocol):
    """The pgvector surface. Narrower than
    :class:`~ragorc.core.protocols.RelationalStore`, which does not declare ANN
    search because not every relational store has pgvector installed."""

    async def vector_search(
        self,
        query_vector: FloatArray,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]: ...


def label_and_rank(
    chunks: Sequence[ScoredChunk], *, retriever: str, source: RetrievalSource
) -> list[ScoredChunk]:
    """Stamp provenance: rank from 0, retrieval source, and the component score.

    Three things a retriever owns about every chunk it returns, and one of them is
    easy to leave to a collaborator by accident:

    * **rank** — re-stamped rather than trusted, because a caller may have
      filtered or reordered between the store and here, and rank fusion reads a
      stale rank as truth.
    * **source** — asserted here rather than taken from whatever built the chunk.
      A row renderer (``TextToSQLConstructor.to_chunks``) knows how to format a
      result set; it is the *retriever* that knows which leg fetched it, and the
      context packer prints that provenance to the generator.
    * **component score** — keyed by the source, which is how "why did this rank
      third?" stays answerable after fusion has collapsed several numbers into
      one. A chunk found by two legs ends up carrying both keys.
    """
    key = source.value
    out: list[ScoredChunk] = []
    for rank, scored in enumerate(chunks):
        scored.rank = rank
        scored.source = source
        scored.component_scores.setdefault(key, scored.score)
        scored.explain.setdefault("retriever", retriever)
        out.append(scored)
    return out


@register("retriever", "sql")
class SQLRetriever:
    """Text-to-SQL as a retriever: question in, evidence chunk out.

    Construction, validation and execution all live in
    :class:`~ragorc.construct.text_to_sql.TextToSQLConstructor` — nothing is
    generated here, and nothing reaches the database that the SQL guard has not
    already parsed and rewritten. This class exists to put that result on the same
    conveyor belt as every other retriever, so a computed answer goes through the
    same fusion, packing, citation and groundedness machinery as retrieved prose.
    """

    name = "sql"

    def __init__(
        self,
        llm: LLM,
        store: RelationalStore | None = None,
        *,
        constructor: TextToSQLConstructor | None = None,
        confidence: float = DEFAULT_SQL_CONFIDENCE,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.constructor = constructor or TextToSQLConstructor(llm, store, settings=self.settings)
        self.confidence = float(confidence)
        self.usage = Usage()
        """Cost of the last :meth:`retrieve`. The ``Retriever`` protocol has no
        usage channel — it returns chunks — so a text-to-SQL retriever has to
        expose its bill somewhere. It is also emitted to the request trace."""

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """Write SQL, run it, and return the result set as one evidence chunk.

        ``top_k`` is accepted for protocol compatibility and does not bound the
        output: a result set is a single chunk (splitting rows would let the
        reranker drop row 4 of a ranking and hand the generator a mutilated table
        it cannot tell is incomplete). The row ceiling is
        ``postgres.max_sql_rows``, enforced by the guard and again by the store.
        """
        self.usage = Usage()
        store = kwargs.get("store") or self.store
        if store is None:
            raise RetrievalError(
                "SQLRetriever needs a RelationalStore",
                hint="pass store= to the constructor or to retrieve()",
            )

        # A plain Timer rather than ``timed``: this stage reports a Usage as well
        # as a duration, and ``timed`` would append a second, usage-less trace
        # entry under the same name.
        with Timer("retrieve.sql") as timer:
            try:
                rows, validation, usage = await self.constructor.construct_and_execute(query, store)
            except (ConstructionError, GuardrailViolation) as exc:
                # The guard refusing a statement is not an outage. Degrade to no
                # structured evidence and let the other legs answer.
                log.warning(
                    "sql_retrieve_degraded", reason=type(exc).__name__, error=str(exc)[:300]
                )
                return []

        self.usage = usage
        chunks = self.constructor.to_chunks(
            rows,
            validation.sql,
            tenant_id=query.tenant_id or self.settings.tenant_id,
            score=self.confidence,
        )
        for scored in chunks:
            scored.explain["constant_score"] = True
            scored.explain["fusion_note"] = "rank-based fusion only; the score is a constant"
        trace_step(
            "retrieve.sql",
            duration_ms=timer.elapsed_ms,
            usage=usage,
            rows=len(rows),
            tables=list(validation.tables),
        )
        log.info(
            "sql_retrieved",
            rows=len(rows),
            chunks=len(chunks),
            tables=list(validation.tables),
            cost_usd=round(usage.cost_usd, 6),
        )
        return label_and_rank(chunks, retriever=self.name, source=RetrievalSource.SQL)


@register("retriever", "pgvector")
class PgVectorRetriever:
    """ANN search over the pgvector column.

    Worth having next to Qdrant rather than instead of it: when the corpus
    already lives in Postgres, this leg needs no second system to operate and its
    filters are ordinary SQL predicates evaluated against real columns and a
    ``jsonb_path_ops`` index. Qdrant wins on scale and on hybrid search in one
    round trip; pgvector wins on transactional consistency with the rows the
    chunks came from, which is the difference between a filter that is correct and
    one that is eventually correct.
    """

    name = "pgvector"

    def __init__(
        self,
        store: VectorSearchStore,
        *,
        embedder: DenseEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.embedder = embedder

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        k = int(top_k or query.top_k or self.settings.retrieval.top_k)
        vector = await self._query_vector(query)
        with timed("retrieve.pgvector", top_k=k):
            chunks = await self.store.vector_search(
                vector,
                top_k=k,
                filters=kwargs.get("filters") or query.filters or None,
                tenant_id=kwargs.get("tenant_id") or query.tenant_id,
            )
        log.debug("pgvector_retrieved", returned=len(chunks), top_k=k)
        return label_and_rank(chunks, retriever=self.name, source=RetrievalSource.DENSE)

    async def _query_vector(self, query: Query) -> FloatArray:
        """Reuse the vector the pipeline already computed, or embed once.

        Reuse is the point: the query is embedded once per request and shared by
        every dense leg. Embedding it again here would double the tokenizer and
        ONNX cost of a fan-out for an identical result, and — worse — a second
        embedder configured with a different asymmetric prefix would silently
        search a different space than the index was built in.
        """
        if query.dense is not None:
            return query.dense
        if self.embedder is None:
            raise RetrievalError(
                "no dense query vector and no embedder injected",
                hint="embed the query upstream or pass embedder= to PgVectorRetriever",
            )
        query.dense = await self.embedder.embed_query(query.text)
        return query.dense


@register("retriever", "pg_fulltext")
class PgFullTextRetriever:
    """Lexical search over the generated ``tsvector`` (or ParadeDB BM25).

    The lexical leg is not redundant with the dense one. Embeddings are lossy on
    exactly the tokens that identify a document: part numbers, error codes,
    surnames, version strings. A dense model maps ``ORA-01555`` and ``ORA-01652``
    to near-identical vectors; an inverted index does not, and that is the class
    of query where hybrid retrieval earns its second index.
    """

    name = "pg_fulltext"

    def __init__(self, store: RelationalStore, *, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.store = store

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        text = query.text.strip()
        if not text:
            return []
        k = int(top_k or query.top_k or self.settings.retrieval.top_k)
        with timed("retrieve.pg_fulltext", top_k=k):
            chunks = await self.store.fulltext_search(
                text,
                top_k=k,
                filters=kwargs.get("filters") or query.filters or None,
                tenant_id=kwargs.get("tenant_id") or query.tenant_id,
            )
        log.debug("pg_fulltext_retrieved", returned=len(chunks), top_k=k)
        return label_and_rank(chunks, retriever=self.name, source=RetrievalSource.FULLTEXT)
