"""Retrieval layer: hybrid search, fusion, noise handling, reranking, compression.

Import strategy — why this package resolves names lazily
--------------------------------------------------------
Every other package in this library re-exports its public API with plain imports
at the top of ``__init__.py``. This one cannot, for two independent reasons.

**Import cost.** Pulling every submodule eagerly would load the Qdrant client, the
Postgres stack, the Neo4j driver, an ONNX runtime session and whatever a web
search provider drags in — on ``import ragorc.retrieve``, before a single query
runs. A dense-only deployment would pay for the graph and relational retrievers it
will never call, and a CLI would pay all of it to print ``--help``.

**Composition.** The retrievers in this package are written and deployed
independently — a sibling module may be absent (an optional extra is not
installed), broken (a provider SDK changed), or simply not there yet in a
partially-vendored checkout. A single missing sibling in an eager ``__init__``
takes down the entire package, including the parts that work. With a module-level
``__getattr__`` the failure is scoped to the name that actually needs the missing
module: ``HybridRetriever`` keeps importing when ``web.py`` cannot.

The mechanism, per :pep:`562`:

* Names this package knows statically (``_EXPORTS``) map to their module and are
  imported on first access, then cached in the module globals so the second
  access is a plain dict lookup.
* Any other public name is looked for across the remaining submodules, skipping
  the ones that are not importable. That is what keeps sibling retrievers
  addressable as ``ragorc.retrieve.X`` without this file having to know their
  symbol names in advance.
* Only names a module actually exports (its ``__all__``) are eligible, so the scan
  cannot accidentally hand out someone's ``log`` or ``np``.

:func:`load_all` is the counterpart for the factory layer: the registry in
:mod:`ragorc.core.registry` is populated by decorators at *class definition* time,
so ``resolve("retriever", "hybrid")`` only works once the defining module has been
imported. Calling ``load_all()`` once at startup imports everything importable and
reports what it skipped, which turns a missing optional dependency into a startup
log line instead of a mid-request ``ConfigError``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

__all__ = [
    "BM25_B",
    "BM25_K1",
    "DEFAULT_RRF_K",
    "EnsembleRetriever",
    "FusionInput",
    "HybridRetriever",
    "InMemoryBM25Retriever",
    "LegResult",
    "NoiseFilter",
    "NoiseReport",
    "SparseRetriever",
    "VectorRetriever",
    "Weights",
    "clone_for_variant",
    "distribution_based_score_fusion",
    "fuse",
    "load_all",
    "max_fusion",
    "mmr_select",
    "normalize_scores",
    "reciprocal_rank_fusion",
    "relative_score_fusion",
    "resolve_filters",
    "run_leg",
    "run_legs",
    "run_variants",
    "simhash",
    "weighted_score_fusion",
]

_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "noise": ("NoiseFilter", "NoiseReport", "mmr_select", "normalize_scores", "simhash"),
    "fusion": (
        "DEFAULT_RRF_K",
        "FusionInput",
        "Weights",
        "distribution_based_score_fusion",
        "fuse",
        "max_fusion",
        "reciprocal_rank_fusion",
        "relative_score_fusion",
        "weighted_score_fusion",
    ),
    "vector": ("VectorRetriever", "clone_for_variant", "resolve_filters", "run_variants"),
    "sparse": ("SparseRetriever",),
    "bm25": ("BM25_B", "BM25_K1", "InMemoryBM25Retriever"),
    "hybrid": ("HybridRetriever",),
    "ensemble": ("EnsembleRetriever", "LegResult", "run_leg", "run_legs"),
}

_EXPORTS: dict[str, str] = {
    symbol: module for module, symbols in _MODULE_EXPORTS.items() for symbol in symbols
}

_SIBLINGS: tuple[str, ...] = (
    "rerank",
    "rankgpt",
    "compress",
    "crag",
    "multihop",
    "multi_store",
    "sql",
    "cypher",
    "graph",
    "web",
    "parent",
)
"""Retrievers and post-processors that live in this package but whose symbol
names are not pinned here. Ordered cheapest-first: the scan stops at the first
module that exports the requested name, and ``web`` is last because it is the one
most likely to need an optional dependency."""

_ALL_MODULES: tuple[str, ...] = (*_MODULE_EXPORTS, *_SIBLINGS)


def _try_import(module: str) -> ModuleType | None:
    """Import one submodule, returning ``None`` when it simply is not there.

    Only a *missing module* is tolerated. Any other ImportError — a broken
    relative import, an optional dependency imported at module scope instead of
    lazily — propagates, because silently skipping those turns a real bug into a
    mysteriously absent retriever.
    """
    try:
        return importlib.import_module(f"{__name__}.{module}")
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing in (module, f"{__name__}.{module}"):
            return None
        raise


def _public_names(module: ModuleType) -> tuple[str, ...]:
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return tuple(declared)
    return tuple(name for name in vars(module) if not name.startswith("_"))


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    known = _EXPORTS.get(name)
    if known is not None:
        # Module is known and shipped: let any failure surface as-is.
        value = getattr(importlib.import_module(f"{__name__}.{known}"), name)
        globals()[name] = value
        return value

    unavailable: list[str] = []
    for candidate in _SIBLINGS:
        module = _try_import(candidate)
        if module is None:
            unavailable.append(candidate)
            continue
        if name in _public_names(module):
            value = getattr(module, name)
            globals()[name] = value
            return value

    hint = f" (not importable: {', '.join(unavailable)})" if unavailable else ""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}{hint}")


def __dir__() -> list[str]:
    return sorted(__all__)


def load_all() -> dict[str, list[str]]:
    """Import every submodule so the component registry is fully populated.

    Returns ``{"loaded": [...], "skipped": [...]}``. Call it once during startup,
    before resolving retriever names from configuration: a name that is only
    registered by an unimported module looks identical to a typo, and the failure
    lands mid-request rather than at boot.
    """
    loaded: list[str] = []
    skipped: list[str] = []
    for module in _ALL_MODULES:
        if _try_import(module) is None:
            skipped.append(module)
        else:
            loaded.append(module)
    return {"loaded": loaded, "skipped": skipped}
