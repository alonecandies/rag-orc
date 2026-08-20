"""Text-to-Cypher as a retriever: the graph leg's *query* path.

GraphRAG search (:mod:`ragorc.retrieve.graph`) walks a fixed traversal — match
entities, expand the ego network, collect the chunks. That covers "what is related
to X" and "what are the themes", and it cannot cover a question whose answer is a
property of the graph itself: *which* department has the most employees, *when*
did every acquisition in this subgraph happen, which nodes have no outgoing edges.
Those are aggregations and projections over the graph, and the only way to express
them is Cypher.

So this module is the graph's counterpart to :class:`~ragorc.retrieve.sql.
SQLRetriever`, and deliberately the same shape: the constructor writes and
validates the statement, the store executes it, and the rows become one evidence
chunk that enters the same fusion, packing, citation and groundedness path as
retrieved prose. Nothing is generated here.

Why the ranking argument from the SQL leg applies verbatim
----------------------------------------------------------
A Cypher result set has no relevance score either. ``MATCH`` is a pattern match:
a row either satisfied the pattern or it is not in the result. Whatever ordering
exists came from an ``ORDER BY`` the model wrote, which expresses the *question's*
notion of importance ("most employees") and not similarity to the question text.

Every chunk produced here therefore carries a single configurable confidence, with
the same consequence spelled out at length in :mod:`ragorc.retrieve.sql`: a
constant score can only be merged with vector results by **rank**. RRF puts this
chunk in at rank 0 for a contribution of ``1 / (rrf_k + 1)``, independent of the
constant. Any score-based fusion — weighted sum, DBSF, relative-score — is
comparing a constant against a cosine distribution and will either always prefer
the graph answer or never surface it, depending on the query.

Why Cypher is riskier than SQL, and where that is handled
---------------------------------------------------------
There is no maintained Python Cypher parser, so validation is lexical rather than
AST-based, and Cypher has two hazards SQL does not: an unbounded variable-length
pattern (``-[*]-``) walks the whole graph, and ``CALL apoc.load.json(...)`` is
server-side request forgery. Both are the
:class:`~ragorc.security.cypher_guard.CypherGuard`'s problem and are enforced
twice — once by the guard on the generated text and once by
:meth:`~ragorc.stores.neo4j.store.Neo4jStore.execute_readonly`, which re-checks the
forbidden keywords and can ``EXPLAIN`` the statement before running it. This module
must not add a third, weaker copy of that logic; its job is to react to the verdict.

Failure policy
--------------
Same split as the relational leg, for the same reason.
:class:`~ragorc.core.errors.StoreUnavailable` propagates, because only the fan-out
layer can record an outage against a store name — swallowing it would make a dead
Neo4j indistinguishable from an empty traversal. A
:class:`~ragorc.core.errors.ConstructionError` (the guard refused, or the model
could not write a statement it accepts) degrades to no graph evidence, because a
bad generated query says nothing about the health of the database and must not
fail a request the other legs can answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from ragorc.core.errors import ConstructionError, GuardrailViolation, RetrievalError
from ragorc.core.models import Query, RetrievalSource, ScoredChunk, Usage
from ragorc.core.protocols import LLM, GraphStore
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import Timer, trace_step
from ragorc.retrieve.sql import label_and_rank

if TYPE_CHECKING:
    from ragorc.construct.text_to_cypher import TextToCypherConstructor

log = structlog.get_logger(__name__)

__all__ = ["DEFAULT_CYPHER_CONFIDENCE", "CypherRetriever"]

DEFAULT_CYPHER_CONFIDENCE = 1.0
"""Confidence assigned to every Cypher result. See
:data:`ragorc.retrieve.sql.DEFAULT_SQL_CONFIDENCE` for why a smaller, more
"humble" constant would express nothing while making the structured answer the
first casualty of a relative score cutoff that runs before fusion."""


def _build_constructor(
    llm: LLM, store: GraphStore | None, settings: Settings
) -> TextToCypherConstructor:
    """Construct the default Text-to-Cypher constructor.

    Imported inside the function rather than at module scope so that a caller who
    injects its own constructor — which is the normal wiring, since the pipeline
    builds one per request context — never pays for the prompt library, the guard
    and the audit sink that the default drags in.
    """
    from ragorc.construct.text_to_cypher import TextToCypherConstructor

    return TextToCypherConstructor(llm, store, settings=settings)


@register("retriever", "cypher")
class CypherRetriever:
    """Question in, one graph-result evidence chunk out."""

    name = "cypher"

    def __init__(
        self,
        llm: LLM,
        store: GraphStore | None = None,
        *,
        constructor: TextToCypherConstructor | None = None,
        confidence: float = DEFAULT_CYPHER_CONFIDENCE,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store
        self.constructor = constructor or _build_constructor(llm, store, self.settings)
        self.confidence = float(confidence)
        self.usage = Usage()
        """Cost of the last :meth:`retrieve`. The ``Retriever`` protocol returns
        chunks and has no usage channel, so a model-using retriever has to publish
        its bill somewhere; it is also emitted to the request trace."""

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """Write Cypher, run it read-only, and return the rows as evidence.

        ``top_k`` is accepted for protocol compatibility and does not bound the
        output: the result set is rendered as a single chunk, because rows are only
        meaningful next to their header and next to each other. Splitting them
        would let the reranker drop the third row of a ranking and hand the
        generator a table it cannot tell is incomplete. The row ceiling is
        ``neo4j.max_cypher_rows``, applied by the store's result cursor rather than
        by appending a second ``LIMIT`` to a statement that may already have one.
        """
        self.usage = Usage()
        store = kwargs.get("store") or self.store
        if store is None:
            raise RetrievalError(
                "CypherRetriever needs a GraphStore",
                hint="pass store= to the constructor or to retrieve()",
            )

        with Timer("retrieve.cypher") as timer:
            try:
                rows, validation, usage = await self.constructor.construct_and_execute(query, store)
            except (ConstructionError, GuardrailViolation) as exc:
                log.warning(
                    "cypher_retrieve_degraded",
                    reason=type(exc).__name__,
                    error=str(exc)[:300],
                )
                return []

        self.usage = usage
        chunks = self.constructor.to_chunks(
            rows,
            validation.cypher,
            tenant_id=query.tenant_id or self.settings.tenant_id,
            score=self.confidence,
        )
        for scored in chunks:
            scored.explain["constant_score"] = True
            scored.explain["fusion_note"] = "rank-based fusion only; the score is a constant"
        trace_step(
            "retrieve.cypher",
            duration_ms=timer.elapsed_ms,
            usage=usage,
            rows=len(rows),
            hops=validation.max_hops,
        )
        log.info(
            "cypher_retrieved",
            rows=len(rows),
            chunks=len(chunks),
            max_hops=validation.max_hops,
            procedures=list(validation.procedures),
            cost_usd=round(usage.cost_usd, 6),
        )
        return label_and_rank(chunks, retriever=self.name, source=RetrievalSource.CYPHER)
