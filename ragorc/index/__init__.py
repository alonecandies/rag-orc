"""Indexing: loaders, splitters, the ingest orchestrator, and the optional stages.

What this package imports eagerly and what it does not
------------------------------------------------------
The three things every ingest needs — the loaders, the splitters, the pipeline —
are imported here. The four *optional* indexing stages are not:

===============  ========================================================
``multirep``     parent-document, summary and Dense-X representations
``colbert``      late-interaction multivectors for MaxSim reranking
``raptor``       UMAP -> GMM -> summarize -> recurse hierarchical index
``graph``        GraphRAG entity/relation extraction and communities
===============  ========================================================

They are reached through a module-level ``__getattr__`` (PEP 562) instead, for two
independent reasons:

* **Extras.** ``raptor`` needs ``umap-learn`` and ``scikit-learn``, ``graph`` needs
  ``igraph`` and ``leidenalg``. Importing them from this ``__init__`` would make
  ``from ragorc.index import IngestPipeline`` — the ordinary case, and the one that
  needs no extras at all — fail with an ``ImportError`` about a clustering library
  nobody asked for.
* **Cost of import.** These modules pull in numpy-heavy scientific stacks whose
  import time is measured in seconds. A CLI that only splits text should not pay
  for a clustering library it will never call.

``__getattr__`` gives the same surface either way: ``ragorc.index.raptor`` resolves
on first attribute access, raises the underlying ``ImportError`` (naming the extra
to install) only if the stage is actually used, and stays absent from
``dir()``-driven eager imports. ``__all__`` advertises them so tooling and
``from ragorc.index import *`` still see the full package.

The pipeline itself reaches these stages by name at run time rather than through
this module, so an ingest with ``raptor_enabled`` and no ``ragorc[raptor]``
installed logs one warning and indexes the leaf chunks — see
:class:`ragorc.index.pipeline.IndexStage`.
"""

from __future__ import annotations

import importlib
from typing import Any

from ragorc.index import split
from ragorc.index.loaders import (
    LOADERS,
    BaseLoader,
    CSVLoader,
    DirectoryLoader,
    DocxLoader,
    HTMLLoader,
    JSONLLoader,
    JSONLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    load,
    loader_for,
)
from ragorc.index.pipeline import (
    BULK_LOAD_MIN_DOCUMENTS,
    IndexStage,
    IngestPipeline,
    IngestReport,
)
from ragorc.index.split import (
    SPLITTERS,
    BaseSplitter,
    Span,
    build_splitter,
)

__all__ = [
    "BULK_LOAD_MIN_DOCUMENTS",
    "LOADERS",
    "SPLITTERS",
    "BaseLoader",
    "BaseSplitter",
    "CSVLoader",
    "DirectoryLoader",
    "DocxLoader",
    "HTMLLoader",
    "IndexStage",
    "IngestPipeline",
    "IngestReport",
    "JSONLLoader",
    "JSONLoader",
    "MarkdownLoader",
    "PDFLoader",
    "Span",
    "TextLoader",
    "build_splitter",
    "colbert",
    "graph",
    "load",
    "loader_for",
    "multirep",
    "raptor",
    "split",
]

_LAZY_SUBMODULES = frozenset({"colbert", "graph", "multirep", "raptor"})
"""Submodules resolved on first access. Written in parallel with this package and
gated behind optional extras, so neither their absence nor their dependencies can
break ``import ragorc.index``."""


def __getattr__(name: str) -> Any:
    """Resolve an optional submodule on first attribute access (PEP 562)."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module  # cache it: __getattr__ runs only on a miss
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
