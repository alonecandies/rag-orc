"""Our filter dialect -> Qdrant ``Filter``.

Why a dialect at all
--------------------
Metadata filters arrive from three places that must not be allowed to diverge:
a user calling ``retrieve(filters={...})``, the self-query constructor turning a
question into a filter with an LLM, and the pipeline's own tenant scoping. One
dialect means one translator, one place where an unsupported operator is caught,
and one place where the security scoping is applied. The shape is deliberately
Mongo-like because that is what LLMs emit when asked for a filter, which keeps
the self-query prompt short and its output valid more often.

Correctness notes that are easy to get wrong
--------------------------------------------
* **Never drop an operator you do not understand.** Silently ignoring
  ``{"year": {"gtx": 2020}}`` returns *more* documents than the caller asked
  for, which in a multi-tenant or permission-filtered corpus is a data leak
  dressed as a recall win. Unknown operators raise.

* **Type decides the condition, not the operator.** Qdrant's ``MatchValue``
  takes ``bool | int | str`` only, so a float equality has to become a
  degenerate range, and a datetime comparison has to become ``DatetimeRange``
  rather than the numeric ``Range``. Getting this wrong is not an error, it is
  an empty result set.

* **Tenant scoping wraps, it does not append.** Appending a tenant condition
  into an existing ``must`` list is wrong whenever the caller's filter uses
  ``should``, because Qdrant evaluates ``must`` and ``should`` as siblings.
  Nesting the caller's whole filter as a single condition ANDs it with the
  tenant condition no matter what is inside it.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import structlog
from qdrant_client import models

from ragorc.core.errors import GuardrailViolation, ValidationFailed
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["TENANT_FIELD", "to_qdrant_filter", "with_tenant"]

TENANT_FIELD = "tenant_id"

_RANGE_OPS = frozenset({"gt", "gte", "lt", "lte"})
_SUPPORTED_OPS = frozenset({"eq", "ne", "in", "nin", "contains", "range", "gt", "gte", "lt", "lte"})
_AND, _OR, _NOT = "$and", "$or", "$not"

Condition = models.Condition
"""Anything Qdrant accepts in ``must``/``should``/``must_not``: a
``FieldCondition``, an ``IsEmptyCondition``, or a nested ``Filter``.

Aliased to the client's own union rather than to ``Any`` so that returning a
model Qdrant does not accept in a condition slot is a type error here instead
of a 400 from the server."""


def _strip_op(name: str) -> str:
    """``"$gte"`` and ``"gte"`` are the same operator.

    LLM-generated filters use the ``$`` prefix inconsistently; accepting both is
    cheaper than a repair round trip.
    """
    return (name[1:] if name.startswith("$") else name).lower()


def _as_list(value: Any) -> list[Any]:
    """A scalar is a one-element set. ``{"in": "text"}`` is a common LLM slip and
    treating the string as a sequence of characters would be worse than useless."""
    if isinstance(value, str | bytes) or not isinstance(value, list | tuple | set | frozenset):
        return [value]
    return list(value)


def _range(field: str, parts: dict[str, Any]) -> models.FieldCondition:
    """Build one range condition from every range operator seen for a field.

    Collecting them means ``{"gte": 1, "lt": 10}`` becomes a single interval
    condition instead of two conditions the engine has to intersect.
    """
    if any(isinstance(v, datetime | date | str) for v in parts.values()):
        # Qdrant parses RFC3339 strings itself; mixing a numeric Range with a
        # datetime payload silently matches nothing.
        return models.FieldCondition(key=field, range=models.DatetimeRange(**parts))
    try:
        numeric = {k: float(v) for k, v in parts.items()}
    except (TypeError, ValueError) as exc:
        raise ValidationFailed(
            "range bounds must be numeric or datetime", field=field, bounds=parts
        ) from exc
    return models.FieldCondition(key=field, range=models.Range(**numeric))


def _equality(field: str, value: Any) -> Condition:
    """``{"field": value}`` — the shape 90% of filters actually use."""
    if value is None:
        # "the key is absent or empty" is what a None filter means in practice;
        # MatchValue cannot express it at all.
        return models.IsEmptyCondition(is_empty=models.PayloadField(key=field))
    if isinstance(value, list | tuple | set | frozenset):
        return models.FieldCondition(key=field, match=models.MatchAny(any=list(value)))
    if isinstance(value, datetime | date):
        return _range(field, {"gte": value, "lte": value})
    if isinstance(value, bool | int | str):
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))
    if isinstance(value, float):
        # MatchValue has no float variant: an exact float match is a degenerate
        # closed interval.
        return _range(field, {"gte": value, "lte": value})
    raise ValidationFailed(
        "unsupported filter value type", field=field, value_type=type(value).__name__
    )


def _field_conditions(field: str, spec: dict[str, Any]) -> list[Condition]:
    """Translate ``{"field": {op: value, ...}}``."""
    conditions: list[Condition] = []
    range_parts: dict[str, Any] = {}

    for raw_op, value in spec.items():
        op = _strip_op(str(raw_op))
        if op in _RANGE_OPS:
            range_parts[op] = value
        elif op == "range":
            if not isinstance(value, dict):
                raise ValidationFailed(
                    "'range' expects a mapping of bounds", field=field, got=type(value).__name__
                )
            for raw_bound, bound in value.items():
                bound_op = _strip_op(str(raw_bound))
                if bound_op not in _RANGE_OPS:
                    raise ValidationFailed(
                        "unsupported range bound",
                        field=field,
                        bound=raw_bound,
                        supported=sorted(_RANGE_OPS),
                    )
                range_parts[bound_op] = bound
        elif op == "eq":
            conditions.append(_equality(field, value))
        elif op == "ne":
            # Negation has to be its own scope: `must_not` is a clause of a
            # Filter, not a property of a FieldCondition.
            conditions.append(models.Filter(must_not=[_equality(field, value)]))
        elif op == "in":
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchAny(any=_as_list(value)))
            )
        elif op == "nin":
            conditions.append(
                models.FieldCondition(
                    key=field, match=models.MatchExcept(**{"except": _as_list(value)})
                )
            )
        elif op == "contains":
            # Substring/token match, which needs a `text` payload index on the
            # field. Note that for *array* payload fields plain equality already
            # means "contains": Qdrant matches a value against any element.
            conditions.append(
                models.FieldCondition(key=field, match=models.MatchText(text=str(value)))
            )
        else:
            raise ValidationFailed(
                "unsupported filter operator",
                field=field,
                operator=raw_op,
                supported=sorted(_SUPPORTED_OPS),
            )

    if range_parts:
        conditions.append(_range(field, range_parts))
    return conditions


def _clause(value: Any, *, key: str) -> list[models.Filter]:
    """Sub-filters for a ``$and``/``$or``/``$not`` clause."""
    items = value if isinstance(value, list | tuple) else [value]
    filters: list[models.Filter] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValidationFailed(f"{key} expects mappings", got=type(item).__name__, clause=key)
        sub = to_qdrant_filter(item)
        if sub is not None:
            filters.append(sub)
    if not filters:
        raise ValidationFailed(f"{key} clause is empty", clause=key)
    return filters


def to_qdrant_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Translate our filter dialect into a Qdrant ``Filter``.

    Supported forms, all combinable::

        {"level": 0}                        # equality (list value -> any-of)
        {"tenant_id": None}                 # key absent / empty
        {"year": {"gte": 2020, "lt": 2024}} # interval, one condition
        {"modality": {"in": ["table"]}}     # any-of / not-any-of via "nin"
        {"author": {"ne": "bot"}}           # negation
        {"content": {"contains": "gRPC"}}   # full-text (needs a text index)
        {"$or": [{...}, {...}]}             # disjunction
        {"$and": [{...}, {...}]}            # explicit conjunction
        {"$not": {...}}                     # negated sub-filter

    Top-level keys are ANDed, so ``{"level": 0, "$or": [...]}`` reads the way it
    looks. Returns ``None`` for an empty filter so callers can pass it straight
    to Qdrant, which treats ``None`` as "no filter" rather than "match nothing".
    """
    if not filters:
        return None

    must: list[Condition] = []
    for key, value in filters.items():
        if key == _AND:
            must.extend(_clause(value, key=_AND))
        elif key == _OR:
            must.append(models.Filter(should=_clause(value, key=_OR)))
        elif key == _NOT:
            must.append(models.Filter(must_not=_clause(value, key=_NOT)))
        elif isinstance(value, dict):
            must.extend(_field_conditions(str(key), value))
        else:
            must.append(_equality(str(key), value))

    if not must:
        return None
    return models.Filter(must=must)


def with_tenant(
    query_filter: models.Filter | None,
    tenant_id: str | None,
    *,
    settings: Settings | None = None,
    field: str = TENANT_FIELD,
) -> models.Filter | None:
    """AND a tenant condition onto ``query_filter``.

    Raises :class:`GuardrailViolation` when ``security.enforce_tenant_isolation``
    is on and no tenant is supplied. That is the whole point of the setting: an
    un-scoped query in a multi-tenant corpus does not fail, it returns another
    customer's documents, and the failure surfaces as a support ticket rather
    than as an exception. Guardrail violations are never retried.
    """
    st = settings or get_settings()
    if tenant_id is None:
        if st.security.enforce_tenant_isolation:
            raise GuardrailViolation(
                "tenant_id is required while tenant isolation is enforced",
                rule="enforce_tenant_isolation",
                hint="set Query.tenant_id, settings.tenant_id, or disable the guard",
            )
        return query_filter

    condition = models.FieldCondition(key=field, match=models.MatchValue(value=tenant_id))
    if query_filter is None:
        return models.Filter(must=[condition])
    if query_filter.should is None and query_filter.must_not is None:
        # Flat `must`-only filter: append and keep the plan one level shallower.
        return query_filter.model_copy(update={"must": [*(query_filter.must or []), condition]})
    # Anything with should/must_not gets nested, so the tenant condition cannot
    # be reinterpreted as one more alternative in an OR.
    return models.Filter(must=[query_filter, condition])
