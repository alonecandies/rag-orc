"""Logical routing: let the model choose which datastore can answer.

The question this answers is not "which store *has* the data" but "which store
can *express* the query". Three shapes, three stores:

* "How many enterprise customers signed up in Q1?" is an aggregation over
  structured fields. A vector search cannot compute a COUNT, and no amount of
  embedding quality will make it able to.
* "How is Northwind connected to Contoso?" is a path query. The relationship may
  not be stated in any single document, which means no chunk contains the answer.
* "Why did we choose late chunking?" is semantic. There is no schema for it.

Routing wrongly is expensive in a specific way: the query still returns *something*
plausible from the wrong store, so the failure is silent. That is why the router
gets a schema hint — knowing what the relational store actually contains is most
of the decision — and why the hint is cached rather than re-derived per query.

Routing must never fail the request. An unparseable or empty decision falls back
to the vector store, which is the only one that can attempt any question.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog

from ragorc.core.models import DataStore, Query, RouteDecision, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import RouteOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["DEFAULT_SCHEMA_HINT", "LogicalRouter"]

DEFAULT_SCHEMA_HINT = """\
vector      — unstructured prose: documentation, articles, policies, transcripts.
relational  — PostgreSQL tables with typed columns; supports counts, sums, averages,
              rankings, date ranges and filters.
graph       — a knowledge graph of entities and typed relationships; supports paths,
              connections and multi-hop traversal.
web         — live internet search, for information outside or newer than the corpus."""

HintProvider = Callable[[], Awaitable[str]] | Callable[[], str] | None


@register("router", "logical")
class LogicalRouter:
    name = "logical"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        schema_hint: HintProvider = None,
        allowed: tuple[DataStore, ...] | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self._hint_provider = schema_hint
        self._hint_cache: str | None = None
        self.allowed = allowed or (
            DataStore.VECTOR,
            DataStore.RELATIONAL,
            DataStore.GRAPH,
            DataStore.WEB,
            DataStore.NONE,
        )

    async def _hint(self) -> str:
        """Resolve and cache the store description.

        Cached because it is derived from database introspection: re-reading
        ``information_schema`` and the Neo4j label list on every query is a round
        trip that buys nothing, since the schema does not change per request.
        """
        if self._hint_cache is not None:
            return self._hint_cache
        if self._hint_provider is None:
            self._hint_cache = DEFAULT_SCHEMA_HINT
            return self._hint_cache
        try:
            result = self._hint_provider()
            self._hint_cache = await result if hasattr(result, "__await__") else str(result)
        except Exception as exc:  # noqa: BLE001
            log.warning("schema_hint_failed", error=str(exc)[:200])
            self._hint_cache = DEFAULT_SCHEMA_HINT
        return self._hint_cache or DEFAULT_SCHEMA_HINT

    async def route(self, query: Query) -> tuple[RouteDecision, Usage]:
        prompt = get_prompt("logical_route")
        try:
            result, usage = await self.llm.structured(
                prompt.render(question=query.text, schema_hint=await self._hint()),
                RouteOutput,
                system=prompt.system,
                model=self.router.model_for(Task.ROUTE),
                stage="route",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("routing_failed", error=str(exc)[:200], fallback="vector")
            return self._fallback("router error"), Usage()

        stores = self._coerce(result.datastores)
        decision = RouteDecision(
            stores=stores,
            prompt_name=result.prompt_name,
            confidence=float(result.confidence),
            reasoning=result.reasoning,
            method="logical",
        )
        log.info(
            "routed",
            stores=[s.value for s in stores],
            confidence=round(decision.confidence, 2),
            prompt=result.prompt_name,
        )
        return decision, usage

    def _coerce(self, names: list[str]) -> tuple[DataStore, ...]:
        """Map model output to allowed stores, defaulting rather than raising."""
        out: list[DataStore] = []
        for name in names:
            try:
                store = DataStore(name)
            except ValueError:
                log.debug("unknown_datastore", name=name)
                continue
            if store in self.allowed and store not in out:
                out.append(store)
        if not out:
            log.info("routing_empty", requested=names, fallback="vector")
            return (DataStore.VECTOR,)
        # NONE only makes sense alone; combined with a store it is contradictory.
        if DataStore.NONE in out and len(out) > 1:
            out = [s for s in out if s is not DataStore.NONE]

        # Always carry the vector store alongside a structured one.
        #
        # SQL and Cypher are *generated*, so they fail in ways a fixed query does
        # not: a hallucinated column, a schema the model misread, a guard
        # rejection. When the structured leg is the only leg, any of those turns
        # into "retrieval returned 0 chunks" and the whole request abstains — even
        # when the prose answer was sitting in the vector store the entire time.
        # Observed exactly that: "who approves an expense claim above €5000?"
        # routed to relational alone, generated SQL against a column that does not
        # exist, and abstained on a question the corpus answers plainly.
        #
        # Vector search is the only leg that can attempt *any* question, so it is
        # the natural companion. The rule fast-path in ragorc/route/hybrid.py has
        # always paired them; this makes the model-driven path agree.
        structured = {DataStore.RELATIONAL, DataStore.GRAPH, DataStore.WEB}
        if structured.intersection(out) and DataStore.VECTOR not in out:
            out.append(DataStore.VECTOR)
            log.debug("routing_vector_companion_added", stores=[s.value for s in out])
        return tuple(out)

    @staticmethod
    def _fallback(reason: str) -> RouteDecision:
        return RouteDecision(
            stores=(DataStore.VECTOR,), confidence=0.0, reasoning=reason, method="fallback"
        )
