"""Task-aware model selection — the cost cascade.

A RAG query is not one LLM call, it is one *hard* call and 10-40 *easy* ones:
route the question, grade five documents, rewrite a query, extract entities,
check groundedness. Those easy calls are binary classifications that a small
model does as well as a frontier model, at 1-5% of the price.

Sending everything to one strong model is the single most common reason a RAG
prototype is 30x more expensive than it needs to be. This module encodes the
policy: each *task* declares the capability it needs, and only synthesis and
escalation reach the expensive tier.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.settings import LLMSettings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["ModelRouter", "ModelTier", "Task"]


class ModelTier(str, enum.Enum):
    FAST = "fast"  # classification, grading, extraction, rewriting
    BALANCED = "balanced"  # synthesis, summarization
    STRONG = "strong"  # escalation: hard reasoning, low-confidence retries


class Task(str, enum.Enum):
    """Every LLM-using stage. Adding a stage means adding it here, which forces
    an explicit decision about what it should cost."""

    # --- cheap classification / transformation ---------------------------
    ROUTE = "route"
    GRADE_RELEVANCE = "grade_relevance"
    GRADE_GROUNDEDNESS = "grade_groundedness"
    GRADE_UTILITY = "grade_utility"
    REWRITE = "rewrite"
    MULTI_QUERY = "multi_query"
    STEP_BACK = "step_back"
    DECOMPOSE = "decompose"
    SELF_QUERY = "self_query"
    COMPRESS = "compress"
    VERIFY_CLAIM = "verify_claim"
    DECOMPOSE_CLAIMS = "decompose_claims"
    SUFFICIENCY = "sufficiency"
    CONTEXTUAL_PREFIX = "contextual_prefix"

    # --- middle: needs fluency, not deep reasoning -----------------------
    HYDE = "hyde"
    SUMMARIZE = "summarize"
    RAPTOR_SUMMARY = "raptor_summary"
    PROPOSITIONS = "propositions"
    EXTRACT_GRAPH = "extract_graph"
    COMMUNITY_REPORT = "community_report"
    GLOBAL_MAP = "global_map"
    RANK_GPT = "rank_gpt"

    # --- expensive: the user reads this output ---------------------------
    ANSWER = "answer"
    GLOBAL_REDUCE = "global_reduce"
    MULTIHOP_REASON = "multihop_reason"


#: Default policy. Overridable per-deployment via ``ModelRouter(overrides=...)``.
_DEFAULT_TIERS: dict[Task, ModelTier] = {
    Task.ROUTE: ModelTier.FAST,
    Task.GRADE_RELEVANCE: ModelTier.FAST,
    Task.GRADE_GROUNDEDNESS: ModelTier.FAST,
    Task.GRADE_UTILITY: ModelTier.FAST,
    Task.REWRITE: ModelTier.FAST,
    Task.MULTI_QUERY: ModelTier.FAST,
    Task.STEP_BACK: ModelTier.FAST,
    Task.DECOMPOSE: ModelTier.FAST,
    Task.SELF_QUERY: ModelTier.FAST,
    Task.COMPRESS: ModelTier.FAST,
    Task.VERIFY_CLAIM: ModelTier.FAST,
    Task.DECOMPOSE_CLAIMS: ModelTier.FAST,
    Task.SUFFICIENCY: ModelTier.FAST,
    Task.CONTEXTUAL_PREFIX: ModelTier.FAST,
    # HyDE writes prose that gets embedded — style matters, depth does not.
    Task.HYDE: ModelTier.FAST,
    # Summaries become the retrieval target; a bad summary is permanently bad,
    # so these get the balanced tier even though they are high volume.
    Task.SUMMARIZE: ModelTier.BALANCED,
    Task.RAPTOR_SUMMARY: ModelTier.BALANCED,
    Task.PROPOSITIONS: ModelTier.BALANCED,
    # Graph extraction quality determines whether traversal finds anything.
    Task.EXTRACT_GRAPH: ModelTier.BALANCED,
    Task.COMMUNITY_REPORT: ModelTier.BALANCED,
    Task.GLOBAL_MAP: ModelTier.FAST,
    Task.RANK_GPT: ModelTier.BALANCED,
    Task.ANSWER: ModelTier.BALANCED,
    Task.GLOBAL_REDUCE: ModelTier.BALANCED,
    Task.MULTIHOP_REASON: ModelTier.BALANCED,
}


@dataclass(slots=True)
class ModelRouter:
    """Maps a :class:`Task` to a concrete model id, and handles escalation."""

    # Resolved, not optional, so that every read is a plain attribute access
    # instead of a `# type: ignore[union-attr]` restating what the constructor
    # already guaranteed. `default_factory` covers the *omitted* argument; the
    # normalization in `__post_init__` covers the explicit `None`, which is a
    # documented way to say "use the defaults" and must keep working.
    settings: LLMSettings = field(default_factory=lambda: get_settings().llm)
    overrides: dict[Task, str] = field(default_factory=dict)
    tiers: dict[Task, ModelTier] = field(default_factory=lambda: dict(_DEFAULT_TIERS))
    prices: dict[str, dict[str, float]] | None = None

    def __post_init__(self) -> None:
        # These look dead to a type checker reading the annotations above, and they
        # are not: `ModelRouter(settings=None)` is a runtime call a checker never
        # sees, and dropping the guard turns it into an `AttributeError` on first
        # use rather than a defaulted router.
        if self.settings is None:
            self.settings = get_settings().llm
        if self.overrides is None:
            self.overrides = {}
        # A caller supplying `tiers` is overriding *some* tasks, not redefining the
        # whole policy, so the defaults stay underneath. Idempotent when the field
        # came from its own factory.
        self.tiers = {**_DEFAULT_TIERS, **(self.tiers or {})}

    def model_for(self, task: Task, *, escalate: bool = False) -> str:
        if escalate:
            return self.settings.strong_model
        if override := self.overrides.get(task):
            return override
        tier = self.tiers.get(task, ModelTier.FAST)
        return self.model_for_tier(tier)

    def model_for_tier(self, tier: ModelTier) -> str:
        return {
            ModelTier.FAST: self.settings.fast_model,
            ModelTier.BALANCED: self.settings.model,
            ModelTier.STRONG: self.settings.strong_model,
        }[tier]

    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Pre-flight cost estimate from the live OpenRouter price table.

        Used by the budget check *before* a call, so an over-budget request is
        refused instead of discovered after the fact.
        """
        if not self.prices:
            return 0.0
        entry = self.prices.get(model)
        if not entry:
            return 0.0
        return prompt_tokens * entry.get("prompt", 0.0) + completion_tokens * entry.get(
            "completion", 0.0
        )

    def should_escalate(self, confidence: float, *, threshold: float | None = None) -> bool:
        """Cascade decision: retry a low-confidence cheap answer on the strong
        model. Most queries never trigger this, so the average cost stays near
        the cheap tier while the tail gets the good model."""
        settings = get_settings().cost
        if not settings.cascade_enabled:
            return False
        limit = threshold if threshold is not None else settings.cascade_confidence_threshold
        return confidence < limit

    def describe(self) -> dict[str, Any]:
        return {
            "fast": self.settings.fast_model,
            "balanced": self.settings.model,
            "strong": self.settings.strong_model,
            "assignments": {t.value: self.model_for(t) for t in Task},
        }
