"""Self-query: split a request into a semantic part and a structured filter.

The problem
-----------
"Papers about diffusion models published after 2022 by Ho" is two requests wearing
one sentence. "Diffusion models" is semantic and belongs in the embedding.
"After 2022" and "by Ho" are *constraints*, and an embedding encodes them badly or
not at all — vector similarity has no notion of ordering, so "after 2022" and
"before 2022" occupy nearly the same point in space.

Leaving the constraint in the query text is worse than dropping it: the tokens
still shift the query vector, so retrieval is degraded *and* unfiltered.

Two failure modes, and why the drop-with-warning policy
-------------------------------------------------------
A generated filter fails in two ways, and they need opposite treatment.

An **unknown field** (the model invents ``publication_year`` when the schema says
``year``) produces a filter that matches nothing. In a vector store, that is a
silent empty result — the query succeeds, returns zero hits, and the pipeline
reports "no relevant documents found" for a question the corpus could answer.
That is the worst outcome in this module, so invalid conditions are **dropped**
with a warning rather than passed through. A degraded-but-working search beats a
confidently empty one.

A **type mismatch** (``year eq "recent"``) is recoverable more often than it
looks: ``"2022"`` coerces to ``2022``, ``"true"`` to ``True``. Coercion is
attempted before dropping, because the model's intent is usually clear and
discarding a correct constraint over its serialization would lose real precision.

Both are reported on the result, so a caller can distinguish "no matches" from
"the filter was wrong".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

import structlog

from ragorc.core.models import Query, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import FilterCondition, MetadataFilterOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["AttributeInfo", "SelfQueryConstructor", "SelfQueryResult"]

#: Operators the filter dialect understands, mapped to the comparison they express.
#: This is the same dialect ``ragorc/stores/qdrant/filters.py`` consumes, so a
#: filter built here is directly executable.
_OPERATORS = frozenset({"eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "range"})

#: Operators that require an ordered type. ``eq`` on a string is fine; ``gt`` is
#: not, and a store will either error or silently compare lexicographically.
_ORDERED_ONLY = frozenset({"gt", "gte", "lt", "lte", "range"})

_TRUE = frozenset({"true", "yes", "1", "y"})
_FALSE = frozenset({"false", "no", "0", "n"})


@dataclass(slots=True, frozen=True)
class AttributeInfo:
    """One filterable metadata field, as described to the model.

    ``examples`` earns its place: given three real values, a model stops inventing
    plausible-looking ones. For a low-cardinality field it is the difference
    between ``status eq "Delivered"`` and ``status eq "delivered"``, and only one
    of those matches anything.
    """

    name: str
    type: str = "string"
    description: str = ""
    examples: tuple[Any, ...] = ()

    def render(self) -> str:
        line = f"- {self.name} ({self.type})"
        if self.description:
            line += f": {self.description}"
        if self.examples:
            shown = ", ".join(repr(e) for e in self.examples[:5])
            line += f" — e.g. {shown}"
        return line


@dataclass(slots=True)
class SelfQueryResult:
    query_text: str
    """The semantic remainder, with every extracted constraint removed."""
    filters: dict[str, Any] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)
    """Human-readable reasons, one per rejected condition. Surfaced rather than
    swallowed, so an empty result set can be diagnosed."""
    coerced: list[str] = field(default_factory=list)
    combinator: str = "and"

    @property
    def has_filters(self) -> bool:
        return bool(self.filters)

    def report(self) -> dict[str, Any]:
        return {
            "query": self.query_text,
            "filters": self.filters,
            "dropped": self.dropped,
            "coerced": self.coerced,
        }


@register("constructor", "self_query", "selfquery")
class SelfQueryConstructor:
    """Turns a natural-language request into (semantic query, metadata filter)."""

    name = "self_query"
    target = "filter"

    def __init__(
        self,
        llm: LLM,
        attributes: Sequence[AttributeInfo] = (),
        *,
        router: ModelRouter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.attributes = {a.name: a for a in attributes}
        self.router = router or ModelRouter(self.settings.llm)
        self.prompt = get_prompt("self_query")

    # -- construction ------------------------------------------------------
    async def construct(self, query: Query, **kwargs: Any) -> tuple[SelfQueryResult, Usage]:
        if not self.attributes:
            # With no schema there is nothing to filter on, and asking the model
            # to invent fields is strictly harmful. Pass the query through.
            log.debug("self_query_skipped", reason="no_attributes")
            return SelfQueryResult(query_text=query.text), Usage()

        parsed, usage = await self.llm.structured(
            self.prompt.render(schema=self._render_schema(), question=query.text),
            MetadataFilterOutput,
            system=self.prompt.system,
            model=self.router.model_for(Task.SELF_QUERY),
            stage="self_query",
        )
        result = self._validate(parsed, fallback_text=query.text)
        log.info(
            "self_query",
            filters=len(result.filters),
            dropped=len(result.dropped),
            coerced=len(result.coerced),
        )
        return result, usage

    async def apply(self, query: Query, **kwargs: Any) -> tuple[Query, SelfQueryResult, Usage]:
        """Return a *copy* of the query with the filter applied.

        A copy because the pre-filter query is what the generator grades the
        answer against, and because a downstream store that ignores filters should
        still see the semantic text the user actually asked for.
        """
        result, usage = await self.construct(query, **kwargs)
        updated = Query(
            text=result.query_text or query.text,
            original=query.original,
            variants=query.variants,
            hypothetical=query.hypothetical,
            filters={**query.filters, **result.filters},
            top_k=query.top_k,
            dense=None if result.query_text != query.text else query.dense,
            sparse=None if result.query_text != query.text else query.sparse,
            multi=query.multi,
            tenant_id=query.tenant_id,
            metadata={
                **query.metadata,
                "self_query": result.report(),
            },
        )
        return updated, result, usage

    # -- validation --------------------------------------------------------
    def _render_schema(self) -> str:
        return "\n".join(a.render() for a in self.attributes.values())

    def _validate(self, parsed: MetadataFilterOutput, *, fallback_text: str) -> SelfQueryResult:
        result = SelfQueryResult(
            query_text=(parsed.query or "").strip() or fallback_text,
            combinator=parsed.combinator,
        )
        clauses: list[dict[str, Any]] = []

        for condition in parsed.conditions:
            clause = self._validate_one(condition, result)
            if clause is not None:
                clauses.append(clause)

        if not clauses:
            return result

        if parsed.combinator == "or" and len(clauses) > 1:
            result.filters = {"$or": clauses}
        else:
            # AND is expressed by merging keys, which is what the store dialect
            # expects; a `$and` wrapper would be redundant.
            merged: dict[str, Any] = {}
            for clause in clauses:
                for key, value in clause.items():
                    if key in merged and merged[key] != value:
                        # Two constraints on one field (a range) must not overwrite
                        # each other — merge them into one condition object.
                        existing = merged[key]
                        if isinstance(existing, dict) and isinstance(value, dict):
                            merged[key] = {**existing, **value}
                        else:
                            merged.setdefault("$and", []).append({key: value})
                            continue
                    else:
                        merged[key] = value
            result.filters = merged
        return result

    def _validate_one(
        self, condition: FilterCondition, result: SelfQueryResult
    ) -> dict[str, Any] | None:
        field_name = (condition.field or "").strip()
        attribute = self.attributes.get(field_name)
        if attribute is None:
            # The single most damaging failure: an unknown field yields a filter
            # that matches nothing, and a silent empty result set.
            result.dropped.append(
                f"unknown field {field_name!r} (known: {sorted(self.attributes)})"
            )
            return None

        operator = (condition.op or "eq").strip().lower()
        if operator not in _OPERATORS:
            result.dropped.append(f"{field_name}: unsupported operator {operator!r}")
            return None

        if operator in _ORDERED_ONLY and attribute.type not in (
            "int",
            "integer",
            "float",
            "number",
            "date",
            "datetime",
        ):
            result.dropped.append(
                f"{field_name}: operator {operator!r} needs an ordered type, "
                f"schema says {attribute.type!r}"
            )
            return None

        value, note = self._coerce(condition.value, attribute, operator)
        if value is _INVALID:
            result.dropped.append(
                f"{field_name}: value {condition.value!r} is not a valid {attribute.type}"
            )
            return None
        if note:
            result.coerced.append(f"{field_name}: {note}")

        if operator == "eq":
            return {field_name: value}
        return {field_name: {operator: value}}

    def _coerce(
        self, value: Any, attribute: AttributeInfo, operator: str
    ) -> tuple[Any, str | None]:
        """Best-effort type coercion. Returns ``_INVALID`` when it cannot."""
        declared = attribute.type.lower()

        if operator in ("in", "nin"):
            if not isinstance(value, (list, tuple, set)):
                value = [value]
            coerced: list[Any] = []
            for item in value:
                one, _ = self._coerce(item, attribute, "eq")
                if one is _INVALID:
                    return _INVALID, None
                coerced.append(one)
            return coerced, None

        if operator == "range":
            if isinstance(value, dict):
                out: dict[str, Any] = {}
                for key in ("gte", "gt", "lte", "lt"):
                    if key in value:
                        one, _ = self._coerce(value[key], attribute, "gte")
                        if one is _INVALID:
                            return _INVALID, None
                        out[key] = one
                return (out, None) if out else (_INVALID, None)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                low, _ = self._coerce(value[0], attribute, "gte")
                high, _ = self._coerce(value[1], attribute, "lte")
                if low is _INVALID or high is _INVALID:
                    return _INVALID, None
                return {"gte": low, "lte": high}, None
            return _INVALID, None

        if declared in ("int", "integer"):
            if isinstance(value, bool):
                return _INVALID, None
            if isinstance(value, int):
                return value, None
            try:
                return int(str(value).strip().replace(",", "")), f"{value!r} -> int"
            except (TypeError, ValueError):
                return _INVALID, None

        if declared in ("float", "number"):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value), None
            try:
                return float(str(value).strip().replace(",", "")), f"{value!r} -> float"
            except (TypeError, ValueError):
                return _INVALID, None

        if declared in ("bool", "boolean"):
            if isinstance(value, bool):
                return value, None
            text = str(value).strip().lower()
            if text in _TRUE:
                return True, f"{value!r} -> True"
            if text in _FALSE:
                return False, f"{value!r} -> False"
            return _INVALID, None

        if declared in ("date", "datetime"):
            if isinstance(value, (date, datetime)):
                return value.isoformat(), None
            text = str(value).strip()
            # A bare year is a common and unambiguous model output for a date
            # field; normalizing it beats dropping a valid constraint.
            if text.isdigit() and len(text) == 4:
                return f"{text}-01-01", f"{value!r} -> {text}-01-01"
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat(), None
            except ValueError:
                return _INVALID, None

        if isinstance(value, (list, tuple, dict)):
            return _INVALID, None
        return str(value), None


class _Invalid:
    """Sentinel distinct from ``None``, because ``None`` is a legitimate filter
    value (``deleted_at eq null``) and must not be confused with failure."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<invalid>"


_INVALID = _Invalid()
