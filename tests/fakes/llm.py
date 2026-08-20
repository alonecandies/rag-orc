"""LLM doubles."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from ragorc.core.models import Usage


class StubLLM:
    """Returns canned values and records every call.

    ``structured`` builds an instance of the requested schema from ``responses``
    keyed by schema name, falling back to a minimal valid instance built from the
    schema's own defaults. That fallback matters: it lets a test exercise a
    pipeline end to end without having to script a response for all ~20 schemas.
    """

    def __init__(
        self,
        *,
        text: str = "A grounded answer about refunds [1].",
        responses: dict[str, Any] | None = None,
        cost_per_call: float = 0.0001,
    ) -> None:
        self.text = text
        self.responses = responses or {}
        self.cost_per_call = cost_per_call
        self.calls: list[dict[str, Any]] = []

    # -- bookkeeping -------------------------------------------------------
    def _record(self, kind: str, **kwargs: Any) -> Usage:
        self.calls.append({"kind": kind, **kwargs})
        return Usage(
            model=kwargs.get("model") or "stub/model",
            prompt_tokens=len(str(kwargs.get("prompt", ""))) // 4,
            completion_tokens=32,
            cost_usd=self.cost_per_call,
            calls=1,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def calls_for(self, stage: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c.get("stage") == stage]

    def stages(self) -> list[str]:
        return [c.get("stage", "?") for c in self.calls]

    # -- LLM protocol ------------------------------------------------------
    async def complete(
        self, prompt: str, *, system: str | None = None, model: str | None = None, **kwargs: Any
    ) -> tuple[str, Usage]:
        usage = self._record("complete", prompt=prompt, system=system, model=model, **kwargs)
        return self.text, usage

    async def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Usage]:
        usage = self._record(
            "structured",
            prompt=prompt,
            system=system,
            model=model,
            schema=schema.__name__,
            **kwargs,
        )
        canned = self.responses.get(schema.__name__)
        if canned is not None:
            obj = canned if isinstance(canned, schema) else schema.model_validate(canned)
            return obj, usage
        return _default_instance(schema), usage

    async def stream(
        self, prompt: str, *, system: str | None = None, model: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        self._record("stream", prompt=prompt, system=system, model=model, **kwargs)
        for word in self.text.split():
            yield word + " "

    async def batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, Usage]]:
        return [await self.complete(p, system=system, model=model, **kwargs) for p in prompts]

    async def batch_structured(
        self,
        prompts: Sequence[str],
        schema: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Any, Usage]]:
        return [
            await self.structured(p, schema, system=system, model=model, **kwargs) for p in prompts
        ]


class ScriptedLLM(StubLLM):
    """Returns a different response per call, in order.

    For testing loops: a CRAG or Self-RAG test needs the first grade to fail and
    the second to pass, which a single canned value cannot express.
    """

    def __init__(self, script: Sequence[Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.script = list(script)
        self.index = 0

    def _next(self) -> Any:
        if self.index >= len(self.script):
            return self.script[-1] if self.script else None
        item = self.script[self.index]
        self.index += 1
        return item

    async def complete(
        self, prompt: str, *, system: str | None = None, model: str | None = None, **kwargs: Any
    ) -> tuple[str, Usage]:
        usage = self._record("complete", prompt=prompt, system=system, model=model, **kwargs)
        item = self._next()
        return (item if isinstance(item, str) else self.text), usage

    async def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Usage]:
        usage = self._record(
            "structured",
            prompt=prompt,
            system=system,
            model=model,
            schema=schema.__name__,
            **kwargs,
        )
        item = self._next()
        if item is None:
            return _default_instance(schema), usage
        if isinstance(item, schema):
            return item, usage
        if isinstance(item, dict):
            return schema.model_validate(item), usage
        return _default_instance(schema), usage


def _default_instance(schema: type[BaseModel]) -> Any:
    """Build a minimal *valid* instance from a schema's fields.

    "Valid" is the hard part. Several schemas in the library constrain their
    lists with ``min_length=1`` (a router must name at least one datastore), so a
    naive empty-list default fails validation. This walks the field constraints
    and fills the minimum, recursing into nested models — which is what lets a
    test drive the whole pipeline without scripting a response for all ~20 schemas.
    """
    return schema.model_validate(_default_values(schema))


def _default_values(schema: type[BaseModel]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if not field.is_required():
            continue
        values[name] = _default_for(field.annotation, _min_items(field))
    return values


def _min_items(field: Any) -> int:
    """Read ``min_length`` off the field's constraint metadata."""
    for meta in getattr(field, "metadata", ()) or ():
        value = getattr(meta, "min_length", None)
        if isinstance(value, int):
            return value
    return 0


def _default_for(annotation: Any, min_items: int = 0) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return args[0] if args else "stub"
    if origin is Union or origin is UnionType:
        # Optional[X] -> use X, so a nullable-but-required field still validates.
        non_none = [a for a in args if a is not type(None)]
        return _default_for(non_none[0], min_items) if non_none else None
    if origin in (list, tuple, set, frozenset) or annotation in (list, tuple, set):
        item_type = args[0] if args else str
        items = [_default_for(item_type) for _ in range(max(min_items, 0))]
        return tuple(items) if origin in (tuple, frozenset) else items
    if origin is dict or annotation is dict:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _default_values(annotation)
    if annotation is bool:
        return True
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is str:
        return "stub"
    return "stub"
