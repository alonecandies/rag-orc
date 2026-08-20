"""Interop with other frameworks. Nothing here is on the request path.

``import ragorc.adapters`` must stay free — it is reachable from
``ragorc.__init__`` walks, from ``dir()``-driven tooling and from anything that
enumerates the package, none of which intend to install ``langchain-core``. Yet
:mod:`ragorc.adapters.langchain` cannot avoid importing it, because
:class:`~ragorc.adapters.langchain.RagOrcRetriever` *subclasses* a LangChain class.

So this package resolves its exports on first attribute access (:pep:`562`):
``from ragorc.adapters import RagOrcRetriever`` works and raises a message naming
the extra when it is missing, while merely importing the package does nothing at
all. The alternative — a top-level import guarded by ``try/except ImportError`` —
would swallow a *real* import error inside the adapter and present it as a missing
optional dependency, sending the reader to install a package they already have.

The adapter module itself keeps its own ``langchain_core`` imports inside function
bodies, so the laziness holds one level deeper too: importing
``ragorc.adapters.langchain`` directly is also free, and only touching a name that
needs LangChain triggers the import.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LangChainRetriever",
    "RagOrcRetriever",
    "as_runnable",
    "from_langchain_documents",
    "from_langchain_retriever",
    "to_langchain_documents",
    "to_langchain_retriever",
]

_LANGCHAIN_EXPORTS = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Import the LangChain adapter on first use of one of its names."""
    if name in _LANGCHAIN_EXPORTS:
        import importlib

        value = getattr(importlib.import_module(f"{__name__}.langchain"), name)
        globals()[name] = value  # cache: __getattr__ only fires on a miss
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
