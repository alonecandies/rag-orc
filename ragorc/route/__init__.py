"""Routing: logical (datastore), semantic (prompt), hybrid (rules + both)."""

from __future__ import annotations

from typing import cast

from ragorc.core.protocols import LLM, DenseEmbedder, Router
from ragorc.core.registry import resolve
from ragorc.core.settings import Settings
from ragorc.route.hybrid import HybridRouter, rule_route
from ragorc.route.logical import DEFAULT_SCHEMA_HINT, LogicalRouter
from ragorc.route.semantic import DEFAULT_ROUTES, SemanticRouter

__all__ = [
    "DEFAULT_ROUTES",
    "DEFAULT_SCHEMA_HINT",
    "HybridRouter",
    "LogicalRouter",
    "SemanticRouter",
    "build_router",
    "rule_route",
]


def build_router(
    name: str,
    *,
    llm: LLM | None = None,
    embedder: DenseEmbedder | None = None,
    settings: Settings | None = None,
    **kwargs: object,
) -> Router:
    """Construct a router by registry name.

    ``hybrid`` assembles its own legs from whichever of ``llm`` and ``embedder``
    were supplied, so a caller with no embedder still gets rule + logical routing
    rather than an error.
    """
    # `protocol=` is the runtime structural check; it cannot narrow the static
    # type, because mypy treats `type[SomeProtocol]` as abstract and refuses to
    # instantiate it. Hence the cast — the check above is what makes it true.
    cls = resolve("router", name, protocol=Router)
    if name == "hybrid":
        logical = LogicalRouter(llm, settings) if llm is not None else None
        semantic = SemanticRouter(embedder, settings=settings) if embedder is not None else None
        return cast(Router, cls(logical=logical, semantic=semantic, settings=settings, **kwargs))
    if name == "semantic":
        return cast(Router, cls(embedder, settings=settings, **kwargs))
    return cast(Router, cls(llm, settings, **kwargs))
