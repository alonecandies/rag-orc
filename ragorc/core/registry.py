"""Name -> implementation registry.

Configuration says ``splitter: semantic``; something has to turn that string
into a class. A decorator-based registry does it without a hand-maintained
``if/elif`` chain, and keeps every implementation discoverable:

    @register("splitter", "semantic")
    class SemanticSplitter: ...

    cls = resolve("splitter", "semantic")

The registry also validates against a protocol at resolve time, so a
misconfigured component fails at startup with a useful message rather than
mid-request with an ``AttributeError``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from ragorc.core.errors import ConfigError

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {}

__all__ = ["available", "register", "resolve", "resolve_instance"]


def register(kind: str, name: str, *aliases: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        bucket = _REGISTRY.setdefault(kind, {})
        for key in (name, *aliases):
            existing = bucket.get(key)
            if existing is not None and existing is not cls:
                raise ConfigError(
                    f"duplicate registration for {kind}:{key}",
                    existing=existing.__name__,
                    new=cls.__name__,
                )
            bucket[key] = cls
        return cls

    return decorator


def resolve(kind: str, name: str, *, protocol: type | None = None) -> type:
    bucket = _REGISTRY.get(kind)
    if not bucket:
        raise ConfigError(f"no components registered for kind {kind!r}")
    cls = bucket.get(name)
    if cls is None:
        raise ConfigError(
            f"unknown {kind} {name!r}", available=sorted(bucket), hint="check your settings"
        )
    if protocol is not None and not issubclass_protocol(cls, protocol):
        raise ConfigError(
            f"{cls.__name__} does not satisfy {protocol.__name__}",
            missing=missing_members(cls, protocol),
        )
    return cls


def resolve_instance(kind: str, name: str, /, *args: Any, **kwargs: Any) -> Any:
    return resolve(kind, name)(*args, **kwargs)


def available(kind: str | None = None) -> dict[str, list[str]]:
    if kind:
        return {kind: sorted(_REGISTRY.get(kind, {}))}
    return {k: sorted(v) for k, v in sorted(_REGISTRY.items())}


def missing_members(cls: type, protocol: type) -> list[str]:
    expected = {
        name
        for name in dir(protocol)
        if not name.startswith("_") and callable(getattr(protocol, name, None))
    }
    return sorted(name for name in expected if not hasattr(cls, name))


def issubclass_protocol(cls: type, protocol: type) -> bool:
    """Structural check that works for non-runtime-checkable protocols too."""
    return not missing_members(cls, protocol)
