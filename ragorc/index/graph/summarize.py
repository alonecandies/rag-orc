"""Community reports: the text that GraphRAG global search actually reads.

A community is a set of entity and relation rows. Nothing can answer a question
from rows, so each community gets one LLM-written report, and that report — not
the graph — is what global search retrieves, maps over and reduces. The report is
therefore the *only* representation of the community that ever reaches a user's
question, which sets two hard requirements: it must stand alone, and it must be
built from the part of the community that carries its meaning.

**Why bottom-up.** Levels are summarized deepest-first, and a parent community
reads its children's finished reports instead of its members' raw rows. This is
cheaper and better, and the two reasons are separate.

Cheaper: a level-0 community with 800 entities and 2 000 relations is well over
100 000 tokens of rows. The same community has perhaps six children, whose
reports total ~3 000 tokens. Same coverage, ~30x less prompt, and the children
were going to be summarized anyway.

Better: without the children, the parent's rows have to be truncated to fit —
which means most of the community is simply *absent* from the prompt, and the
report describes the fraction that survived truncation while claiming to describe
the whole. With the children, every member was read by some call at full detail,
and the parent's job becomes synthesis of complete sub-summaries rather than a
skim of an arbitrary slice.

**Why truncation is by degree and by weight.** When rows must be dropped, the
ones to keep are the highest-degree entities and the highest-weight relations.
Degree is not a popularity contest here: the high-degree members are the ones
every other member connects *through*, so they are what makes the community
recognizable — drop a hub and the report describes an unrecognizable cluster,
drop a leaf and it loses one fact. Weight accumulates across the chunks that
asserted an edge, so a high-weight relation is corroborated by many documents
while a weight-1 relation is one sentence's claim; when there is only room for
some of the edges, the corroborated ones are the ones a report should be built on.

**Why the input budget is derived from the output budget.** A report capped at
``graph.community_summary_max_tokens`` cannot faithfully represent an unbounded
input. Past roughly an order of magnitude of compression the model is discarding
most of what it was sent, so paying for more input buys nothing — the prompt is
capped at a multiple of the report size rather than at the context window, which
also keeps a single pathological hub community from costing more than the rest of
the corpus put together.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.models import Community, Entity, Relation, Usage
from ragorc.core.protocols import LLM
from ragorc.core.schemas import CommunityReport
from ragorc.core.settings import Settings, get_settings
from ragorc.core.tokens import TokenBudget, count_tokens, count_tokens_batch
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["CommunitySummarizer", "SummarizationReport"]

#: How much larger the prompt may be than the report it produces. See the module
#: docstring: beyond this the extra input is compressed away unread.
_INPUT_TO_OUTPUT_RATIO = 16

#: Shares of the input budget. With children present the children's reports are
#: the highest-value tokens in the prompt — they already summarize the rows —
#: so they take the largest share and the raw rows become supporting detail.
_SHARES_WITH_CHILDREN = {"children": 0.45, "entities": 0.35, "relations": 0.20}
_SHARES_LEAF = {"children": 0.0, "entities": 0.60, "relations": 0.40}


@dataclass(slots=True)
class SummarizationReport:
    """Summarized communities plus the accounting for the run."""

    communities: list[Community] = field(default_factory=list)
    summarized: int = 0
    failed: int = 0
    reused_children: int = 0
    truncated_entities: int = 0
    truncated_relations: int = 0
    levels: int = 0

    def summary(self) -> dict[str, Any]:
        return {
            "communities": len(self.communities),
            "summarized": self.summarized,
            "failed": self.failed,
            "reused_children": self.reused_children,
            "truncated_entities": self.truncated_entities,
            "truncated_relations": self.truncated_relations,
            "levels": self.levels,
        }


class CommunitySummarizer:
    """One concurrent LLM call per community, levels processed bottom-up."""

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.cfg = self.settings.graph
        self.router = router or ModelRouter(self.settings.llm)
        self._prompt = get_prompt("community_report")

        self._max_output = max(64, self.cfg.community_summary_max_tokens)
        window = TokenBudget(
            total=self.settings.llm.context_window,
            reserved_output=self._max_output,
            reserved_system=count_tokens(self._prompt.system),
        )
        self._input_budget = max(
            256, min(window.available_context, _INPUT_TO_OUTPUT_RATIO * self._max_output)
        )

    async def summarize(
        self,
        communities: Sequence[Community],
        entities: Sequence[Entity],
        relations: Sequence[Relation],
    ) -> tuple[SummarizationReport, Usage]:
        """Write a report for every community, deepest level first."""
        report = SummarizationReport(communities=list(communities))
        if not communities or not self.cfg.summarize_communities:
            return report, Usage()

        entity_by_key = {entity.key: entity for entity in entities}
        relation_by_key = {relation.key: relation for relation in relations}
        children: dict[int, list[int]] = {}
        for position, community in enumerate(communities):
            if community.parent_id is not None:
                children.setdefault(community.parent_id, []).append(position)

        by_level: dict[int, list[int]] = {}
        for position, community in enumerate(communities):
            by_level.setdefault(community.level, []).append(position)
        report.levels = len(by_level)

        usages: list[Usage] = []
        model = self.router.model_for(Task.COMMUNITY_REPORT)
        # Deepest level first: a parent's prompt reads its children's finished
        # summaries, so the levels are a dependency chain and cannot overlap.
        # Within a level every call is independent and runs concurrently.
        for level in sorted(by_level, reverse=True):
            positions = by_level[level]
            prompts = await bounded_gather(
                (
                    self._build_prompt(
                        report.communities[position],
                        entity_by_key,
                        relation_by_key,
                        [
                            report.communities[child]
                            for child in children.get(report.communities[position].id, ())
                        ],
                        report,
                    )
                    for position in positions
                ),
                limit=max(1, self.settings.llm.max_concurrency),
            )
            outcomes = await bounded_gather(
                (self._write_report(prompt, model) for prompt in prompts),
                limit=max(1, self.settings.llm.max_concurrency),
                return_exceptions=True,
            )
            for position, prompt, outcome in zip(positions, prompts, outcomes, strict=True):
                community = report.communities[position]
                if isinstance(outcome, BaseException):
                    report.failed += 1
                    log.warning(
                        "graph_community_report_failed",
                        community_id=community.id,
                        level=community.level,
                        error=str(outcome)[:200],
                        error_type=type(outcome).__name__,
                    )
                    report.communities[position] = _fallback_report(community, prompt)
                    continue
                written, usage = outcome
                usages.append(usage)
                report.summarized += 1
                report.communities[position] = _apply_report(community, written)

        total = Usage.sum(usages)
        log.info(
            "graph_communities_summarized",
            **report.summary(),
            model=model,
            cost_usd=round(total.cost_usd, 6),
        )
        return report, total

    # -- one community -----------------------------------------------------
    async def _write_report(
        self, prompt: _CommunityPrompt, model: str
    ) -> tuple[CommunityReport, Usage]:
        return await self.llm.structured(
            self._prompt.render(entities=prompt.entities_text, relations=prompt.relations_text),
            CommunityReport,
            system=self._prompt.system,
            model=model,
            stage="community_report",
            max_tokens=self._max_output,
        )

    async def _build_prompt(
        self,
        community: Community,
        entity_by_key: Mapping[str, Entity],
        relation_by_key: Mapping[tuple[str, str, str], Relation],
        children: Sequence[Community],
        report: SummarizationReport,
    ) -> _CommunityPrompt:
        """Render and fit one community's rows.

        Offloaded to a thread because the fitting is a ``tiktoken`` batch encode
        over every candidate line, which is Rust that releases the GIL. On a
        large community that is milliseconds of CPU per community and there can
        be thousands of them, so leaving it inline would serialize the whole
        summarization stage behind token counting.
        """
        prompt = await asyncio.to_thread(
            _render_prompt,
            community,
            entity_by_key,
            relation_by_key,
            [child for child in children if child.summary],
            self._input_budget,
        )
        report.truncated_entities += prompt.dropped_entities
        report.truncated_relations += prompt.dropped_relations
        if prompt.child_count:
            report.reused_children += prompt.child_count
        return prompt


# ---------------------------------------------------------------------------
# Prompt construction (CPU-bound; runs in a thread)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _CommunityPrompt:
    entities_text: str
    relations_text: str
    top_entities: tuple[str, ...]
    dropped_entities: int
    dropped_relations: int
    child_count: int


def _render_prompt(
    community: Community,
    entity_by_key: Mapping[str, Entity],
    relation_by_key: Mapping[tuple[str, str, str], Relation],
    children: Sequence[Community],
    budget: int,
) -> _CommunityPrompt:
    shares = _SHARES_WITH_CHILDREN if children else _SHARES_LEAF
    child_budget = int(budget * shares["children"])
    entity_budget = int(budget * shares["entities"])
    relation_budget = budget - child_budget - entity_budget

    members = [
        entity_by_key[name.casefold()]
        for name in community.entity_names
        if name.casefold() in entity_by_key
    ]
    # Highest degree first; ties broken on the number of chunks that mentioned
    # the entity, then on the name so the prompt is byte-stable across runs.
    members.sort(key=lambda e: (-e.degree, -len(e.source_chunk_ids), e.name))
    entity_lines = [_entity_line(entity) for entity in members]
    kept_entities, dropped_entities = _fit_lines(entity_lines, entity_budget)

    edges = [
        relation_by_key[key]
        for key in dict.fromkeys(community.relation_keys)
        if key in relation_by_key
    ]
    edges.sort(key=lambda r: (-r.weight, r.source, r.type, r.target))
    relation_lines = [_relation_line(relation) for relation in edges]
    kept_relations, dropped_relations = _fit_lines(relation_lines, relation_budget)

    child_lines = [
        f"- {child.title or f'community {child.id}'}: {child.summary}" for child in children
    ]
    kept_children, _ = _fit_lines(child_lines, child_budget)

    entities_text = "\n".join(kept_entities) or "(no entity details available)"
    if kept_children:
        # ``community_report`` exposes only {entities} and {relations}, and the
        # prompt library is not edited from here, so the sub-reports are labelled
        # and prepended to the entity block rather than smuggled in unmarked.
        entities_text = (
            "Sub-community reports (already-summarized parts of this community):\n"
            + "\n".join(kept_children)
            + "\n\nEntities in this community:\n"
            + entities_text
        )

    return _CommunityPrompt(
        entities_text=entities_text,
        relations_text="\n".join(kept_relations) or "(no relationships available)",
        top_entities=tuple(entity.name for entity in members[:5]),
        dropped_entities=dropped_entities,
        dropped_relations=dropped_relations,
        child_count=len(kept_children),
    )


def _entity_line(entity: Entity) -> str:
    description = entity.description.replace("\n", " ").strip()
    head = f"{entity.name} ({entity.type}, degree {entity.degree})"
    return f"- {head}: {description}" if description else f"- {head}"


def _relation_line(relation: Relation) -> str:
    description = relation.description.replace("\n", " ").strip()
    head = (
        f"{relation.source} -[{relation.type}]-> {relation.target} (weight {relation.weight:.1f})"
    )
    return f"- {head}: {description}" if description else f"- {head}"


def _fit_lines(lines: Sequence[str], budget: int) -> tuple[list[str], int]:
    """Keep the longest prefix of ``lines`` that fits ``budget`` tokens.

    The lines are already in priority order, so this is a prefix cut rather than
    a selection. ``count_tokens_batch`` encodes them all in one Rust call and the
    cut point comes from a cumulative sum plus a ``searchsorted`` — counting
    line-by-line and stopping when the total is exceeded would call the tokenizer
    once per line, which on a 2 000-entity community is the difference between
    microseconds and tens of milliseconds.
    """
    if not lines or budget <= 0:
        return [], len(lines)
    # +1 per line for the joining newline, so the estimate does not drift under
    # the real prompt on a community with thousands of short rows.
    costs = np.asarray(count_tokens_batch(list(lines)), dtype=np.int64) + 1
    cumulative = np.cumsum(costs)
    keep = int(np.searchsorted(cumulative, budget, side="right"))
    if keep >= len(lines):
        return list(lines), 0
    return list(lines[:keep]), len(lines) - keep


# ---------------------------------------------------------------------------
# Applying the result
# ---------------------------------------------------------------------------
def _apply_report(community: Community, written: CommunityReport) -> Community:
    """Attach the report, blending its rating into the structural rank.

    Neither number alone ranks correctly. The structural rank measures how much
    graph evidence sits behind a community; the model's rating measures how
    significant its subject is. A tiny community about the corpus's central
    topic rates high with almost no evidence, and a huge cluster of boilerplate
    has all the evidence and no significance — averaging keeps global search from
    being dominated by either failure.
    """
    summary = written.summary.strip()
    if written.findings:
        summary = summary + "\n\n" + "\n".join(f"- {finding}" for finding in written.findings)
    return replace(
        community,
        title=written.title.strip() or community.title,
        summary=summary,
        rank=(community.rank + float(written.rating)) / 2.0,
    )


def _fallback_report(community: Community, prompt: _CommunityPrompt) -> Community:
    """Deterministic stand-in when the report call fails.

    A community with no summary is invisible to global search, which silently
    removes a slice of the corpus from every thematic question. The provisional
    title plus the top entity names is a poor report and a much better outcome
    than a hole in the index.
    """
    names = ", ".join(prompt.top_entities)
    return replace(
        community,
        summary=(
            f"Automatically generated placeholder: this community groups "
            f"{len(community.entity_names)} entities"
            + (f", including {names}." if names else ".")
            + " No model-written report is available for it."
        ),
    )
