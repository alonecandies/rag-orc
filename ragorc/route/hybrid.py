"""Hybrid routing: rules first, then embeddings and the LLM in parallel.

Three tiers, and the ordering is the design:

1. **Rule fast path.** A question containing "how many", "average", "total" or
   "top 5" is an aggregation, and no model call is needed to know that. Likewise
   "connected to", "path between" and "related to" are graph questions. The
   patterns fire only on high-confidence phrasings, and when one fires the whole
   routing stage costs nothing.
2. **Semantic prompt selection.** One matmul, no model call, on every request.
3. **Logical store selection.** The only tier that costs a model call, and it runs
   concurrently with tier 2 because the two decisions are independent — prompt
   choice does not depend on store choice.

Running 2 and 3 concurrently rather than sequentially is worth stating plainly:
they are independent decisions about the same query, so serializing them adds the
semantic router's latency to the LLM router's for no reason.
"""

from __future__ import annotations

import re
from collections.abc import Coroutine
from typing import Any, Protocol

import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import DataStore, Query, RouteDecision, Usage
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["HybridRouter", "RouterLeg", "rule_route"]

#: Patterns that identify a store without ambiguity. Deliberately narrow: a false
#: positive here silently sends a question to a store that cannot answer it, which
#: is worse than paying for the model call.
_RELATIONAL = re.compile(
    r"\b(how many|how much|count of|number of|total (?:of|number|amount|revenue|sales)|"
    r"sum of|average|avg|median|mean of|top \d+|bottom \d+|highest|lowest|largest|smallest|"
    r"rank(?:ed|ing)? by|group(?:ed)? by|per (?:month|quarter|year|customer|region)|"
    r"between \d{4} and \d{4}|since \d{4}|year over year|month over month)\b",
    re.IGNORECASE,
)
_GRAPH = re.compile(
    r"\b(connected to|connection between|related to|relationship between|"
    r"path (?:from|between)|how (?:is|are|does) \w+ (?:related|connected|linked)|"
    r"who (?:works|worked) (?:with|for)|reports to|owned by|"
    r"downstream of|upstream of|depends on|influenced by|shortest path|"
    r"(?:two|three|multi)[- ]hop)\b",
    re.IGNORECASE,
)
_NO_RETRIEVAL = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening)|"
    r"who are you|what can you do|help)\b[\s!?.]*$",
    re.IGNORECASE,
)


def rule_route(question: str) -> RouteDecision | None:
    """Return a decision when a high-confidence pattern matches, else ``None``."""
    if _NO_RETRIEVAL.match(question.strip()):
        return RouteDecision(
            stores=(DataStore.NONE,), confidence=0.95, reasoning="conversational", method="rule"
        )

    relational = bool(_RELATIONAL.search(question))
    graph = bool(_GRAPH.search(question))

    if relational and graph:
        # Genuinely both, e.g. "how many suppliers are connected to Acme". Send it
        # to both rather than guessing which dominates.
        return RouteDecision(
            stores=(DataStore.RELATIONAL, DataStore.GRAPH, DataStore.VECTOR),
            confidence=0.7,
            reasoning="aggregation and relationship cues",
            method="rule",
        )
    if relational:
        # Vector is retained as a companion: an aggregation question often also
        # wants prose context, and the SQL path may return nothing.
        return RouteDecision(
            stores=(DataStore.RELATIONAL, DataStore.VECTOR),
            confidence=0.85,
            reasoning="aggregation cue",
            method="rule",
        )
    if graph:
        return RouteDecision(
            stores=(DataStore.GRAPH, DataStore.VECTOR),
            confidence=0.85,
            reasoning="relationship cue",
            method="rule",
        )
    return None


class RouterLeg(Protocol):
    """What both legs have in common, and all this class ever calls on them.

    Structural rather than nominal so a caller can supply its own router — the
    dead-leg tests do exactly that — without importing anything from here.
    """

    async def route(self, query: Query) -> tuple[RouteDecision, Usage]: ...


@register("router", "hybrid")
class HybridRouter:
    name = "hybrid"

    def __init__(
        self,
        logical: RouterLeg | None = None,
        semantic: RouterLeg | None = None,
        settings: Settings | None = None,
        *,
        use_rules: bool = True,
    ) -> None:
        self.logical = logical
        self.semantic = semantic
        self.settings = settings or get_settings()
        self.use_rules = use_rules

    async def route(self, query: Query) -> tuple[RouteDecision, Usage]:
        rule = rule_route(query.text) if self.use_rules else None

        # A conversational greeting needs neither retrieval nor a prompt lookup.
        if rule is not None and DataStore.NONE in rule.stores:
            log.info("routed", method="rule", stores=["none"])
            return rule, Usage()

        # Bound to locals so the ``is not None`` checks below narrow: a rule hit
        # makes the paid leg unnecessary, which is the whole point of tier 1.
        semantic = self.semantic
        logical = self.logical if rule is None else None
        need_logical = logical is not None

        tasks: list[Coroutine[Any, Any, tuple[RouteDecision, Usage]]] = []
        labels: list[str] = []
        if semantic is not None:
            tasks.append(semantic.route(query))
            labels.append("semantic")
        if logical is not None:
            tasks.append(logical.route(query))
            labels.append("logical")

        results = await bounded_gather(tasks, limit=2, return_exceptions=True) if tasks else []

        prompt_name: str | None = None
        stores: tuple[DataStore, ...] = rule.stores if rule else ()
        confidence = rule.confidence if rule else 0.0
        reasons: list[str] = [rule.reasoning] if rule and rule.reasoning else []
        usages: list[Usage] = []
        method_parts: list[str] = ["rule"] if rule else []

        for label, result in zip(labels, results, strict=True):
            if isinstance(result, BaseException):
                log.warning("router_leg_failed", leg=label, error=str(result)[:160])
                continue
            decision, usage = result
            usages.append(usage)
            method_parts.append(label)
            if label == "semantic":
                prompt_name = decision.prompt_name
                if decision.reasoning:
                    reasons.append(f"prompt: {decision.reasoning}")
            else:
                stores = decision.stores or stores
                confidence = max(confidence, decision.confidence)
                if decision.reasoning:
                    reasons.append(decision.reasoning)

        if not stores:
            stores = (DataStore.VECTOR,)
            reasons.append("defaulted to vector")

        decision = RouteDecision(
            stores=stores,
            prompt_name=prompt_name,
            confidence=confidence or 0.5,
            reasoning="; ".join(reasons),
            method="+".join(method_parts) or "default",
        )
        log.info(
            "routed",
            method=decision.method,
            stores=[s.value for s in stores],
            prompt=prompt_name,
            llm_call=need_logical,
        )
        return decision, Usage.sum(usages)
