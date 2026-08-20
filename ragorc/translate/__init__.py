"""Query translation: multi-query, RAG-Fusion, step-back, decomposition, HyDE."""

from __future__ import annotations

from collections.abc import Sequence

from ragorc.core.protocols import LLM
from ragorc.core.registry import resolve
from ragorc.core.settings import Settings
from ragorc.translate.base import BaseTranslator, CompositeTranslator, clean_variants
from ragorc.translate.decomposition import (
    DecompositionTranslator,
    RecursiveDecomposer,
    SubAnswer,
)
from ragorc.translate.hyde import HyDETranslator
from ragorc.translate.multi_query import MultiQueryTranslator
from ragorc.translate.rag_fusion import RAGFusionTranslator, reciprocal_rank_fusion
from ragorc.translate.step_back import StepBackTranslator

__all__ = [
    "BaseTranslator",
    "CompositeTranslator",
    "DecompositionTranslator",
    "HyDETranslator",
    "MultiQueryTranslator",
    "RAGFusionTranslator",
    "RecursiveDecomposer",
    "StepBackTranslator",
    "SubAnswer",
    "build_translator",
    "build_translators",
    "clean_variants",
    "reciprocal_rank_fusion",
]


def build_translator(
    name: str, llm: LLM, settings: Settings | None = None, **kwargs: object
) -> BaseTranslator:
    """Resolve a translator by registry name."""
    cls = resolve("translator", name)
    return cls(llm, settings, **kwargs)


def build_translators(
    names: Sequence[str], llm: LLM, settings: Settings | None = None, **kwargs: object
) -> CompositeTranslator:
    """Chain several translators. Order matters — see CompositeTranslator."""
    return CompositeTranslator([build_translator(n, llm, settings, **kwargs) for n in names])
