"""Query construction: natural language into SQL, Cypher and metadata filters.

Every constructor here returns a *validated* artifact, never a raw string, and
execution is a separate call. That separation is the security boundary: the only
object carrying executable SQL or Cypher is the guard's own output, so nothing can
reach a datastore without having been validated first.
"""

from __future__ import annotations

from ragorc.construct.self_query import AttributeInfo, SelfQueryConstructor, SelfQueryResult
from ragorc.construct.text_to_cypher import TextToCypherConstructor
from ragorc.construct.text_to_sql import TextToSQLConstructor
from ragorc.core.registry import resolve

__all__ = [
    "AttributeInfo",
    "SelfQueryConstructor",
    "SelfQueryResult",
    "TextToCypherConstructor",
    "TextToSQLConstructor",
    "build_constructor",
]


def build_constructor(
    name: str, llm: object, store: object | None = None, **kwargs: object
) -> object:
    """Resolve a constructor by registry name."""
    cls = resolve("constructor", name)
    if name in ("self_query", "selfquery"):
        return cls(llm, **kwargs)
    return cls(llm, store, **kwargs)
