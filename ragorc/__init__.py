"""ragorc — a complete, production-grade RAG orchestration library.

Public API
----------
    import asyncio
    from ragorc import build_pipeline

    async def main():
        async with await build_pipeline() as rag:
            await rag.ingest("./docs")
            answer = await rag.query("why is late chunking cheaper?")
            print(answer.text, answer.groundedness, answer.usage.cost_usd)

    asyncio.run(main())

Why the heavy names are lazy
----------------------------
``RAGPipeline`` transitively imports LangGraph, the three database drivers and
the ONNX runtime. A project that only wants ``Settings`` or a ``Chunk`` — a
script computing token counts, a test collecting fixtures, a CLI printing
configuration — should not pay for any of that, so those imports happen on first
*attribute access* rather than at ``import ragorc``.

The effect is measurable: importing the data model alone is milliseconds, where
importing the pipeline is hundreds. The dataclasses, settings and errors are
eager because they are cheap and almost every caller wants them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ragorc.core.errors import (
    BudgetExceeded,
    ConfigError,
    ConstructionError,
    GuardrailViolation,
    LLMError,
    RagOrcError,
    RetrievalError,
    StoreUnavailable,
    ValidationFailed,
)
from ragorc.core.models import (
    Answer,
    Chunk,
    ChunkingStrategy,
    Citation,
    Community,
    DataStore,
    Document,
    Entity,
    FusionMethod,
    GradeLabel,
    GraphPath,
    Modality,
    Query,
    Relation,
    RetrievalResult,
    RetrievalSource,
    RouteDecision,
    ScoredChunk,
    SparseVector,
    Usage,
)
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging, new_request_context

if TYPE_CHECKING:  # pragma: no cover - import-time cost avoided at runtime
    from ragorc.pipeline.builder import RAGPipeline

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "BudgetExceeded",
    "Chunk",
    "ChunkingStrategy",
    "Citation",
    "Community",
    "ConfigError",
    "ConstructionError",
    "DataStore",
    "Document",
    "Entity",
    "FusionMethod",
    "GradeLabel",
    "GraphPath",
    "GuardrailViolation",
    "LLMError",
    "Modality",
    "Query",
    "RAGPipeline",
    "RagOrcError",
    "Relation",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalSource",
    "RouteDecision",
    "ScoredChunk",
    "Settings",
    "SparseVector",
    "StoreUnavailable",
    "Usage",
    "ValidationFailed",
    "__version__",
    "build_pipeline",
    "configure_logging",
    "get_settings",
    "new_request_context",
]

_LAZY = {
    "RAGPipeline": ("ragorc.pipeline.builder", "RAGPipeline"),
    "build_pipeline": ("ragorc.pipeline.builder", "build_pipeline"),
}


def __getattr__(name: str) -> Any:
    """Resolve the heavy exports on first access.

    A clear ``ConfigError`` is raised when the pipeline layer is unavailable,
    because "cannot import name RAGPipeline" tells a user nothing about which
    extra they are missing.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        import importlib

        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ConfigError(
            f"{name} requires the orchestration layer",
            missing=str(exc),
            hint='pip install "ragorc[all]"',
        ) from exc
    value = getattr(module, attribute)
    globals()[name] = value  # cache, so the next access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
