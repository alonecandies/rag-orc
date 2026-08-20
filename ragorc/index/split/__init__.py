"""Chunk optimization: six splitting strategies behind one protocol.

All six satisfy :class:`ragorc.core.protocols.Splitter` and all six obey the same
contract from ADR-0002 — they return boundaries, never vectors. Which one to pick
is a property of the corpus, not a preference:

===================  =====================================================
``semantic``         Default. Cuts where the meaning changes, using one
                     embedding batch per document as a boundary detector.
``recursive``        No model, no network. The universal fallback and the
                     right answer for mixed or unknown content.
``token``            The only strategy whose size guarantee is in the same
                     unit as the embedder's limit. Use it for CJK, code and
                     anything where characters-per-token varies wildly.
``markdown``         Keeps the authored outline: heading path per chunk,
                     code fences and tables never cut.
``code``             Splits on definition boundaries and carries the
                     enclosing class and import block along.
``sentence_window``  One chunk per sentence for retrieval precision, with
                     the surrounding sentences stored for generation.
===================  =====================================================

:func:`build_splitter` resolves the name through the component registry, so a
deployment selects a strategy with a settings string and a third party can add one
with a ``@register("splitter", ...)`` decorator and no change here.

The one non-obvious behaviour is the fallback: ``semantic`` needs a
:class:`~ragorc.core.protocols.DenseEmbedder`, and asking for it without one is a
wiring mistake that would otherwise surface as a ``TypeError`` deep inside an
ingest job. The factory degrades to ``recursive`` and logs a warning instead —
ingest continuing with a documented downgrade beats an ingest that dies at
document one, and the log line names the fix.
"""

from __future__ import annotations

import structlog

from ragorc.core.protocols import DenseEmbedder, Splitter
from ragorc.core.registry import resolve
from ragorc.core.settings import Settings, get_settings
from ragorc.index.split.base import (
    UNBOUNDED_OVERLAP,
    BaseSplitter,
    Span,
    apply_overlap,
    merge_small_spans,
    normalize_spans,
    split_oversized_spans,
    split_sentences,
)
from ragorc.index.split.code import CodeSplitter
from ragorc.index.split.markdown import MarkdownSplitter
from ragorc.index.split.recursive import DEFAULT_SEPARATORS, RecursiveSplitter, recursive_spans
from ragorc.index.split.semantic import SemanticSplitter
from ragorc.index.split.sentence_window import SentenceWindowSplitter
from ragorc.index.split.token import TokenSplitter, token_spans

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_SEPARATORS",
    "SPLITTERS",
    "UNBOUNDED_OVERLAP",
    "BaseSplitter",
    "CodeSplitter",
    "MarkdownSplitter",
    "RecursiveSplitter",
    "SemanticSplitter",
    "SentenceWindowSplitter",
    "Span",
    "TokenSplitter",
    "apply_overlap",
    "build_splitter",
    "merge_small_spans",
    "normalize_spans",
    "recursive_spans",
    "split_oversized_spans",
    "split_sentences",
    "token_spans",
]

SPLITTERS: tuple[str, ...] = (
    "recursive",
    "token",
    "semantic",
    "markdown",
    "code",
    "sentence_window",
)
"""The built-in names, matching ``IndexingSettings.splitter``. Third-party
splitters are resolvable too; this tuple is what ships."""

_FALLBACK = "recursive"


def build_splitter(
    name: str | None = None,
    *,
    embedder: DenseEmbedder | None = None,
    settings: Settings | None = None,
) -> Splitter:
    """Construct a splitter by name, defaulting to ``settings.indexing.splitter``.

    ``embedder`` is only consulted by strategies that declare
    ``requires_embedder``; passing one to the others is harmless and lets a
    pipeline build any splitter from a single call site.
    """
    resolved = settings or get_settings()
    chosen = name or resolved.indexing.splitter
    cls = resolve("splitter", chosen, protocol=Splitter)

    if getattr(cls, "requires_embedder", False) and embedder is None:
        log.warning(
            "splitter_fallback",
            requested=chosen,
            chosen=_FALLBACK,
            reason="no dense embedder supplied",
            hint="pass embedder=... to build_splitter to keep semantic chunking",
        )
        chosen = _FALLBACK
        cls = resolve("splitter", chosen, protocol=Splitter)

    # ``resolve`` is typed ``-> type``, so instantiating it yields ``Any``. The
    # annotation is safe rather than wishful: ``protocol=Splitter`` above made
    # ``resolve`` verify the members structurally and raise ConfigError otherwise.
    splitter: Splitter = (
        cls(embedder, settings=resolved)
        if getattr(cls, "requires_embedder", False)
        else cls(settings=resolved)
    )
    log.debug("splitter_built", splitter=chosen, cls=cls.__name__)
    return splitter
