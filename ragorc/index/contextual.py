"""Contextual retrieval: give each chunk back the context the split took away.

A chunk reading

    Its revenue grew 40% in the same period, driven by the enterprise segment.

is unretrievable by anyone who searches for "Acme revenue growth". The document
knew what *its* referred to; the chunk does not, and therefore neither does its
vector or its BM25 posting list. Anthropic's contextual retrieval (2024) fixes
that at the prompt level: an LLM writes one or two sentences situating the chunk
inside its document, the prefix is prepended *before embedding*, and Anthropic
measured retrieval failures dropping from 5.7% to 2.9% for embeddings alone and
to 1.9% combined with contextual BM25 — a 35%-to-49% reduction depending on the
configuration.

Where it sits in the ladder (ADR-0002): one rung below late chunking. Both solve
the same problem — a chunk vector that does not know what the document knew — but
late chunking solves it by *pooling* (one forward pass per document, no model
calls) while this solves it by *rewriting* (one model call per chunk). So this is
the technique for hosted embedding APIs, which return only pooled vectors and
therefore make late chunking impossible.

Why prompt caching is not an optimization here but the premise
-------------------------------------------------------------
The whole document is sent with **every** chunk request. That makes the naive
ingest cost quadratic in document length: a 60k-token document split into 200
chunks sends 60k tokens 200 times, 12M prompt tokens for one document. At list
price that is not a technique, it is a bill.

Provider prompt caching is what changes the arithmetic — Anthropic charges roughly
10% of the input price on a cache hit — and three implementation choices are
required to actually get the hits, all of them made here:

1. **The document goes in the static portion.** ``OpenRouterLLM._messages``
   attaches ``cache_control: ephemeral`` to the *system* block, and only to it. A
   request with the document in the user message is a cache miss by construction,
   every time. So the system block is assembled as the prompt's instructions plus
   the document, and the user block carries only the chunk. Both halves are
   derived from the registered ``contextual_prefix`` template by splitting it at
   its own ``<chunk>`` marker — no prompt copy is written here, only the decision
   about which half is static.
2. **All chunks of one document are processed consecutively.** Interleaving two
   documents alternates the cached prefix and every request pays full price.
   ``documents_in_flight`` defaults to 1 for that reason.
3. **The cache is warmed with a single request before the fan-out.** The cache
   entry is written by the request that first sees the prefix; N requests issued
   concurrently before that write lands all miss it and all pay full price. So
   chunk one goes alone, and chunks two onward fan out against a warm prefix. This
   is the difference between paying for the document once and paying for it N
   times, and it costs one serialized round trip per document to get.

``LLM.structured`` also caches completions on ``(prompt, system, model, schema)``,
so re-ingesting an unchanged document skips the calls entirely — the same saving
one layer up.

Documents longer than the budget are windowed, not skipped
----------------------------------------------------------
``indexing.contextual_max_doc_tokens`` bounds what may be sent as the static
prefix. Above it, the document is tiled into windows of at most that many tokens
and each chunk is situated against the window that contains it — the region most
relevant to that chunk — rather than the request failing or the document being
left unenriched. Chunks are grouped by window so the cache still warms once per
window instead of once per chunk. The characters-per-token ratio used to size the
windows is measured on the document itself rather than assumed: it varies roughly
threefold between English prose, source code and CJK text, and a global constant
would either overshoot the limit or waste most of it.

The prefix is for the index only, never for the generator
---------------------------------------------------------
It is written to :attr:`Chunk.contextual_prefix`, and the data model already
enforces the split: :attr:`Chunk.embed_text` prepends the prefix (so it is
embedded, and so a sparse/BM25 embedder reading the same property gets contextual
BM25 for free), :attr:`Chunk.content` does not, and :meth:`Chunk.payload` does not
serialize the field at all — the vector store physically cannot hand it back to
the generator. That matters because the prefix restates context the prompt already
contains, and duplicated context is tokens spent to make the model more confident
about something it was already told.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial

import structlog

from ragorc.core.concurrency import map_concurrent
from ragorc.core.errors import RagOrcError
from ragorc.core.models import Chunk, Document, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import ContextualPrefix
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.core.tokens import count_tokens, truncate_to_tokens
from ragorc.llm.prompts import Prompt, get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["ContextualEnricher"]

_CHUNK_MARKER = "<chunk>"
"""Where the registered ``contextual_prefix`` template divides into a static half
(instructions + document) and a variable half (the chunk). Splitting the template
rather than writing two prompt strings keeps every word of prompt copy in
:mod:`ragorc.llm.prompts`, where it can be reviewed and versioned."""

_JSON_ENVELOPE_TOKENS = 32
"""Slack over ``contextual_prefix_tokens`` for the structured-output wrapper. The
completion is ``{"context": "..."}``; capping ``max_tokens`` at exactly the prose
budget truncates the closing brace and fails validation instead of shortening the
sentence."""


def _split_template(prompt: Prompt) -> tuple[str, str] | None:
    """Divide the prompt template into its static and variable halves.

    Returns ``None`` if the template no longer has the shape this optimization
    depends on, which is the honest answer: the enricher then falls back to a
    single user message that works correctly and simply does not get cache hits.
    """
    head, marker, tail = prompt.template.partition(_CHUNK_MARKER)
    if not marker or "{document}" not in head or "{chunk}" not in tail:
        return None
    return head, marker + tail


@dataclass(slots=True)
class _Window:
    """One static prefix and the chunks that will be situated against it."""

    text: str
    system: str
    chunks: list[Chunk] = field(default_factory=list)


@register("indexer", "contextual")
class ContextualEnricher:
    """Writes a situating prefix onto each chunk before it is embedded."""

    name = "contextual"

    def __init__(
        self,
        llm: LLM,
        *,
        router: ModelRouter | None = None,
        documents_in_flight: int = 1,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.config = self.settings.indexing
        self.router = router or ModelRouter(self.settings.llm)
        self.documents_in_flight = max(1, documents_in_flight)
        self.prompt = get_prompt("contextual_prefix")

        split = _split_template(self.prompt)
        if split is None:
            self._document_template: str | None = None
            self._user_template: str | None = None
            log.warning(
                "contextual_prompt_not_splittable",
                prompt=self.prompt.name,
                impact="document travels in the user message; prompt caching cannot engage",
            )
        else:
            self._document_template, self._user_template = split

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def enrich(
        self, document: Document, chunks: Sequence[Chunk]
    ) -> tuple[list[Chunk], Usage]:
        """Set ``contextual_prefix`` on every chunk of one document, in place.

        Windows are processed one at a time and, inside a window, the first chunk
        is sent alone to write the cache entry before the rest fan out. The
        serialized first call is the price of the cached prefix for all the others.
        """
        if not chunks:
            return [], Usage()

        windows = self._plan_windows(document, chunks)
        total = Usage()
        with timed("contextual_enrich", document_id=document.id, chunks=len(chunks)):
            # Sequential across windows: each window is its own cached prefix, and
            # overlapping two of them puts two prefixes in front of the provider.
            for window in windows:
                total = total + await self._enrich_window(window)

        enriched = sum(1 for chunk in chunks if chunk.contextual_prefix)
        log.info(
            "contextual_enriched",
            document_id=document.id,
            chunks=len(chunks),
            enriched=enriched,
            windows=len(windows),
            cost_usd=round(total.cost_usd, 6),
        )
        return list(chunks), total

    async def enrich_many(
        self, items: Sequence[tuple[Document, Sequence[Chunk]]]
    ) -> tuple[list[Chunk], Usage]:
        """Enrich several documents while keeping each document's cache warm.

        ``documents_in_flight`` is the throughput/cost dial and it defaults to 1:
        one document at a time guarantees that the static prefix in front of the
        provider is the one it just cached. Raising it to a small number is safe in
        practice — a provider keeps several prefixes alive for the cache's TTL — but
        every additional document in flight is another prefix competing for that
        window, and a corpus of short documents will thrash it.
        """
        if not items:
            return [], Usage()
        results = await map_concurrent(
            self._enrich_pair,
            list(items),
            limit=self.documents_in_flight,
            return_exceptions=True,
        )
        chunks: list[Chunk] = []
        usages: list[Usage] = []
        for (document, group), outcome in zip(items, results, strict=True):
            if isinstance(outcome, BaseException):
                # One document's enrichment failing must not lose the document:
                # unprefixed chunks still index, they just index worse.
                log.warning(
                    "contextual_document_failed",
                    document_id=document.id,
                    error=str(outcome),
                    error_type=type(outcome).__name__,
                )
                chunks.extend(group)
                continue
            enriched, usage = outcome
            chunks.extend(enriched)
            usages.append(usage)
        return chunks, Usage.sum(usages)

    async def _enrich_pair(
        self, item: tuple[Document, Sequence[Chunk]]
    ) -> tuple[list[Chunk], Usage]:
        document, chunks = item
        return await self.enrich(document, chunks)

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------
    def _plan_windows(self, document: Document, chunks: Sequence[Chunk]) -> list[_Window]:
        text = document.content
        budget = max(self.config.contextual_max_doc_tokens, 1)
        total = count_tokens(text)
        if total <= budget:
            return [_Window(text=text, system=self._system_for(text), chunks=list(chunks))]

        # Ratio measured on this document, not assumed (see the module docstring).
        chars_per_token = max(len(text) / max(total, 1), 1.0)
        window_chars = max(int(budget * chars_per_token), 1)
        count = max((len(text) + window_chars - 1) // window_chars, 1)

        buckets: list[list[Chunk]] = [[] for _ in range(count)]
        for chunk in chunks:
            # Assign by midpoint so a chunk straddling a window boundary goes to
            # the window that holds most of it. Derived chunks with no offsets fall
            # into the first window, which is the only defensible guess.
            midpoint = (chunk.start_char + chunk.end_char) // 2
            buckets[min(max(midpoint // window_chars, 0), count - 1)].append(chunk)

        windows: list[_Window] = []
        for index, bucket in enumerate(buckets):
            if not bucket:
                continue
            start = index * window_chars
            # truncate_to_tokens is the hard guarantee; the char estimate above is
            # only close, and "close" is not a limit.
            region = truncate_to_tokens(text[start : start + window_chars], budget)
            windows.append(_Window(text=region, system=self._system_for(region), chunks=bucket))

        log.warning(
            "contextual_document_windowed",
            document_id=document.id,
            document_tokens=total,
            limit=budget,
            windows=len(windows),
            reason="document exceeds contextual_max_doc_tokens; situating against regions",
        )
        return windows

    def _system_for(self, region: str) -> str:
        """The static half of the request: instructions plus the document region."""
        if self._document_template is None:
            return self.prompt.system
        return f"{self.prompt.system}\n\n{self._document_template.format(document=region)}".strip()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    async def _enrich_window(self, window: _Window) -> Usage:
        first, *rest = window.chunks
        usages = [await self._apply(window, first)]
        if rest:
            outcomes = await map_concurrent(
                partial(self._apply, window),
                rest,
                limit=max(1, self.settings.llm.max_concurrency),
                return_exceptions=True,
            )
            for chunk, outcome in zip(rest, outcomes, strict=True):
                if isinstance(outcome, BaseException):
                    log.warning(
                        "contextual_prefix_unexpected_failure",
                        chunk_id=chunk.id,
                        error=str(outcome),
                        error_type=type(outcome).__name__,
                    )
                    continue
                usages.append(outcome)
        return Usage.sum(usages)

    async def _apply(self, window: _Window, chunk: Chunk) -> Usage:
        """Generate one prefix and attach it. Never raises for a model failure."""
        if self._user_template is None:
            user = self.prompt.render(document=window.text, chunk=chunk.content)
        else:
            user = self._user_template.format(chunk=chunk.content)

        try:
            result, usage = await self.llm.structured(
                user,
                ContextualPrefix,
                system=window.system,
                model=self.router.model_for(Task.CONTEXTUAL_PREFIX),
                stage="contextual_prefix",
                max_tokens=self.config.contextual_prefix_tokens + _JSON_ENVELOPE_TOKENS,
            )
        except RagOrcError as exc:
            log.warning(
                "contextual_prefix_failed",
                chunk_id=chunk.id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return Usage()

        context = " ".join(result.context.split())
        if not context:
            log.debug("contextual_prefix_empty", chunk_id=chunk.id)
            return usage

        # A splitter may already have supplied a prefix — the markdown splitter
        # puts the heading path there. That one is exact and free, so it keeps its
        # place at the front: if the cap truncates anything, it truncates the
        # generated half.
        existing = (chunk.contextual_prefix or "").strip()
        combined = f"{existing} {context}" if existing else context
        chunk.contextual_prefix = truncate_to_tokens(
            combined, max(self.config.contextual_prefix_tokens, 1)
        )
        chunk.metadata["contextual"] = True
        return usage
