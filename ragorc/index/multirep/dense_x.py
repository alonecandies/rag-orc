"""Dense-X / proposition indexing: one fact per index entry.

The most precise multi-representation option in this package, and the most
expensive. A chunk is decomposed into atomic propositions — one fact each, every
pronoun resolved — and each proposition becomes its own searchable unit pointing
back at the chunk it came from via ``parent_id``. Retrieval matches a single fact
against a single question; generation still receives the whole chunk.

Why it is the most precise
--------------------------
Precision is lost at index time by *averaging*. A 512-token chunk that states six
facts has one vector, and that vector is the mean of six directions; a question
about the third fact is compared against a point that is five-sixths noise. A
proposition has nothing to average. It matches its own question almost exactly and
is nearly orthogonal to everything else, so the top-k stops being a lottery among
chunks that mention the right nouns (Chen et al., *Dense X Retrieval*, 2023).

Why it is the most expensive
----------------------------
Three costs, all real:

* **One LLM call per chunk at ingest**, on the balanced tier (see
  :mod:`ragorc.llm.router`: a bad decomposition is permanently bad). This is the
  same order of magnitude as GraphRAG's extraction pass, and it is why
  ``indexing.dense_x_enabled`` is off by default.
* **Index amplification.** One chunk becomes up to
  :data:`_MAX_PROPOSITIONS_PER_CHUNK` entries, so the vector count and the HNSW
  graph grow roughly an order of magnitude. That is memory and build time, not
  just disk.
* **Result-set crowding.** Ten propositions from one chunk can fill a ``top_k``
  of ten. Deduplicating by ``parent_id`` — which
  :func:`~ragorc.index.multirep.parent_document.expand_parents` does — is not
  optional here, it is what makes the representation usable.

The failure mode that makes propositions worthless
--------------------------------------------------
**A proposition must be context-independent or it is useless when retrieved
alone.** "It grew 40% in the same period" carries no information once it is
separated from the paragraph that named the company and the period — and
separation is the entire point of the technique, so the damage is guaranteed
rather than possible. The ``propositions`` prompt therefore requires every
pronoun and every "the company"/"this method" reference to be resolved into the
explicit entity, and this module enforces the mechanical half of the same
contract: propositions shorter than a clause are dropped, duplicates are
collapsed, and empty output degrades to indexing the chunk itself.

Verbatim propositions keep their source offsets
-----------------------------------------------
Most propositions are rewrites — that is what pronoun resolution is — but a
simple declarative sentence often comes back unchanged. When a normalized
substring test finds the proposition inside the source chunk, the unit is built
from the *source span* rather than the model's retyping: ``content`` becomes
``document.content[start:end]`` exactly, real offsets are attached, and the entry
is then indistinguishable from a splitter chunk. Two things follow, both worth
having: span-level citation verification works against it, and the author's own
wording is what gets embedded and matched by BM25 rather than the model's
paraphrase of it. Propositions that fail the test carry no document offsets at
all, for the same reason summaries do not — see :meth:`PropositionIndexer._unit`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.concurrency import map_concurrent
from ragorc.core.errors import BudgetExceeded, RagOrcError
from ragorc.core.ids import stable_uuid
from ragorc.core.models import Chunk, IntArray, Modality, ScoredChunk, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import PropositionOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.core.tokens import count_tokens
from ragorc.index.multirep.parent_document import expand_parents
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["PropositionIndexer"]

_MAX_PROPOSITIONS_PER_CHUNK = 12
"""Cap on units emitted per chunk.

``PropositionOutput`` allows 30, which is the schema's guard against a runaway
generation; this is the *indexing* limit and it is much lower on purpose. Past a
dozen the model stops finding facts and starts splitting hairs — restating the
same fact with a different subject, or promoting a subordinate clause to a
sentence — and each of those competes with its own siblings for a top-k slot
while adding nothing. A chunk with thirty genuine facts is a chunk that was split
too coarsely, so hitting the cap is recorded in metadata as the signal it is."""

_MIN_PROPOSITION_WORDS = 3
"""A proposition is a sentence. Fewer than three words cannot be subject, verb and
object, so it is a fragment or a heading — and a two-word entry with almost no
content to dilute the query match will outscore real evidence."""

_TERMINATORS = " .!?;:,"
"""Trailing punctuation stripped before the verbatim test. The model adds a full
stop to make a grammatical sentence; the source may end the span with a comma or
nothing at all, and that difference alone should not lose the offsets."""

_JSON_ENVELOPE_TOKENS = 64
"""Slack for the structured-output wrapper around a proposition list. Sized larger
than the summary indexer's because the envelope here is an array of strings, so
the per-item quoting and commas add up."""

_PROPOSITION_TOKENS = 48
"""Assumed upper bound per proposition when sizing ``max_tokens``. Propositions
are single sentences; this bounds the completion without capping it so tightly
that a long list is truncated mid-JSON and fails validation."""


def _normalized_index(text: str) -> tuple[str, IntArray]:
    """Casefolded, whitespace-collapsed text plus a map back to real offsets.

    The map is what makes the verbatim test usable rather than merely decorative:
    a match position in normalized space converts straight back to a character
    offset in the original document. A collapsed whitespace run maps to the offset
    of its *first* character, so an exclusive end landing on a space excludes the
    whitespace instead of trailing it.

    Case folding can change length (German ``ß`` folds to ``ss``), so origins are
    extended once per output character rather than once per input character; the
    result stays index-aligned with the returned string. The loop is per-chunk —
    a few hundred characters — where a numpy round trip would cost more than it
    saves; the array is built once at the end because :meth:`np.ndarray.__getitem__`
    is what the caller uses it for.
    """
    pieces: list[str] = []
    origins: list[int] = []
    pending_space = False
    space_at = 0
    for index, char in enumerate(text):
        if char.isspace():
            if pieces and not pending_space:
                pending_space = True
                space_at = index
            continue
        if pending_space:
            pieces.append(" ")
            origins.append(space_at)
            pending_space = False
        folded = char.casefold()
        pieces.append(folded)
        origins.extend([index] * len(folded))
    # Sentinel so an exclusive end can address one past the last character.
    origins.append(len(text))
    return "".join(pieces), np.asarray(origins, dtype=np.int64)


def _normalize(text: str) -> str:
    """The same normalization, without the offset map, for needles and checks."""
    return " ".join(text.casefold().split())


@register("indexer", "dense_x")
class PropositionIndexer:
    """Decomposes chunks into propositions and indexes each one separately."""

    name = "dense_x"

    def __init__(
        self,
        llm: LLM,
        *,
        router: ModelRouter | None = None,
        embedder: Any | None = None,
        max_per_chunk: int = _MAX_PROPOSITIONS_PER_CHUNK,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.config = self.settings.indexing
        self.router = router or ModelRouter(self.settings.llm)
        self.embedder = embedder
        self.max_per_chunk = max(1, max_per_chunk)
        self.prompt = get_prompt("propositions")

    # ------------------------------------------------------------------
    # Building the representation
    # ------------------------------------------------------------------
    async def build(self, chunks: Sequence[Chunk]) -> tuple[list[Chunk], Usage]:
        """Decompose every chunk concurrently and return the proposition units.

        A chunk whose decomposition fails or comes back empty is indexed as itself
        rather than dropped, on the same reasoning as everywhere else in this
        package: a slightly worse index entry is recoverable at query time, a
        missing one is not.
        """
        if not chunks:
            return [], Usage()

        with timed("dense_x_build", chunks=len(chunks)):
            outcomes = await map_concurrent(
                self._decompose,
                list(chunks),
                limit=max(1, self.settings.llm.max_concurrency),
                return_exceptions=True,
            )

        units: list[Chunk] = []
        usages: list[Usage] = []
        verbatim = 0
        failures = 0
        for source, outcome in zip(chunks, outcomes, strict=True):
            if isinstance(outcome, BudgetExceeded):
                # A spent budget is a stop signal for the whole stage, not one
                # chunk's bad luck: every remaining call would raise too. The
                # re-raise in `_decompose` is undone here without this — `map_concurrent`
                # returns the exception and the branch below reclassifies it as an
                # unexpected per-chunk failure, which is exactly the outcome the
                # re-raise exists to prevent. RaptorIndexer._summarize_level has
                # had this line all along.
                raise outcome
            if isinstance(outcome, BaseException):
                failures += 1
                log.warning(
                    "propositions_unexpected_failure",
                    chunk_id=source.id,
                    error=str(outcome),
                    error_type=type(outcome).__name__,
                )
                units.append(self._as_source_unit(source, reason="error"))
                continue
            produced, usage = outcome
            usages.append(usage)
            verbatim += sum(1 for unit in produced if unit.metadata.get("verbatim"))
            units.extend(produced)

        total = Usage.sum(usages)
        log.info(
            "dense_x_built",
            chunks=len(chunks),
            propositions=len(units),
            amplification=round(len(units) / len(chunks), 2),
            verbatim=verbatim,
            failures=failures,
            cost_usd=round(total.cost_usd, 6),
        )
        return units, total

    async def index(
        self,
        chunks: Sequence[Chunk],
        *,
        vector_store: Any | None = None,
        docstore: Any | None = None,
    ) -> tuple[list[Chunk], Usage]:
        """Build, embed the propositions and write both representations."""
        units, usage = await self.build(chunks)
        if not units:
            return units, usage

        await self._embed(units)
        if docstore is not None:
            referenced = {unit.parent_id for unit in units if unit.parent_id}
            sources = [c for c in chunks if c.id in referenced]
            if sources:
                await docstore.upsert_chunks(sources)
        if vector_store is not None:
            await vector_store.upsert(units)
        return units, usage

    async def expand(self, chunks: Sequence[ScoredChunk], store: Any) -> list[ScoredChunk]:
        """Swap each retrieved proposition back for its source chunk.

        Mandatory rather than optional for this representation: without the
        ``parent_id`` collapse, several propositions from one chunk occupy several
        context slots to assert several halves of the same paragraph.
        """
        return await expand_parents(chunks, store)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _decompose(self, source: Chunk) -> tuple[list[Chunk], Usage]:
        prompt = self.prompt.render(text=source.content)
        try:
            result, usage = await self.llm.structured(
                prompt,
                PropositionOutput,
                system=self.prompt.system,
                model=self.router.model_for(Task.PROPOSITIONS),
                stage="dense_x",
                max_tokens=self.max_per_chunk * _PROPOSITION_TOKENS + _JSON_ENVELOPE_TOKENS,
            )
        except BudgetExceeded:
            # A spent budget is a stop signal for the whole stage, not one chunk's
            # bad luck: every remaining call would raise too, and swallowing it
            # per chunk turns "the budget ran out after 40 of 720 chunks" into 680
            # units silently indexed as their own source text, with `usage.calls`
            # reporting zero and nothing on the report. RAPTOR already treats it
            # this way (`_summarize_level`); these two did not.
            raise
        except RagOrcError as exc:
            log.warning(
                "propositions_failed",
                chunk_id=source.id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return [self._as_source_unit(source, reason="llm_error")], Usage()

        kept = self._clean(result.propositions)
        if not kept:
            # Nothing survived filtering — a chunk of headings or a table, most
            # likely. Index it as itself rather than removing it from the corpus.
            log.debug("propositions_empty", chunk_id=source.id, returned=len(result.propositions))
            return [self._as_source_unit(source, reason="empty")], usage

        truncated = len(kept) > self.max_per_chunk
        if truncated:
            # The model emits propositions in source order, so keeping the head
            # keeps the earliest facts rather than an arbitrary subset.
            kept = kept[: self.max_per_chunk]

        normalized, origins = _normalized_index(source.content)
        units = [
            self._unit(source, text, normalized, origins, truncated=truncated) for text in kept
        ]
        return units, usage

    def _clean(self, raw: Sequence[str]) -> list[str]:
        """Drop empties and fragments, collapse duplicates, preserve order.

        Duplicates are compared after normalization because the same fact restated
        with different capitalization or spacing is one index entry's worth of
        information and two entries' worth of crowding.
        """
        seen: set[str] = set()
        out: list[str] = []
        for item in raw:
            text = " ".join(item.split())
            key = _normalize(text)
            if not key or len(key.split()) < _MIN_PROPOSITION_WORDS or key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    def _unit(
        self,
        source: Chunk,
        text: str,
        normalized: str,
        origins: IntArray,
        *,
        truncated: bool,
    ) -> Chunk:
        """Build one proposition unit, with offsets when the span is verbatim.

        Non-verbatim propositions get ``start_char == end_char == 0``. The citation
        layer resolves a quote to a document position as ``chunk.start_char +
        content.find(quote)``, and for rewritten text there is no such position —
        attaching the source chunk's span would manufacture offsets that look
        valid and point at words the proposition does not contain. The true source
        span travels in metadata instead, and the packer restores the correct base
        from ``parent_start_char`` after substituting the source text.
        """
        span = _verbatim_span(source.content, text, normalized, origins)
        metadata = dict(source.metadata)
        metadata["representation"] = "proposition"
        metadata["parent_start_char"] = source.start_char
        metadata["parent_end_char"] = source.end_char
        metadata["source_index"] = source.index
        metadata["verbatim"] = span is not None
        if truncated:
            metadata["propositions_truncated"] = True

        if span is None:
            content, start_char, end_char = text, 0, 0
        else:
            # Prefer the author's exact characters over the model's retyping: it
            # restores `content == document.content[start:end]`, which makes span
            # citation verification work, and it is the wording BM25 will see.
            relative_start, relative_end = span
            content = source.content[relative_start:relative_end]
            start_char = source.start_char + relative_start
            end_char = source.start_char + relative_end

        return Chunk(
            # Deterministic and namespaced: re-ingesting an unchanged chunk
            # regenerates the same ids, and a reordered proposition list does not
            # churn them because position is not part of the key.
            id=stable_uuid("proposition", source.id, content),
            content=content,
            document_id=source.document_id,
            index=source.index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
            parent_id=source.id,
            modality=Modality.PROPOSITION,
            token_count=count_tokens(content),
            tenant_id=source.tenant_id,
        )

    @staticmethod
    def _as_source_unit(source: Chunk, *, reason: str) -> Chunk:
        source.metadata["representation"] = "source"
        source.metadata["dense_x_skipped"] = reason
        return source

    async def _embed(self, units: Sequence[Chunk]) -> None:
        if self.embedder is None or not units:
            return
        vectors = await self.embedder.embed_documents([unit.embed_text for unit in units])
        for unit, vector in zip(units, vectors, strict=True):
            unit.dense = vector


def _verbatim_span(
    content: str, proposition: str, normalized: str, origins: IntArray
) -> tuple[int, int] | None:
    """Offsets of ``proposition`` inside ``content``, or ``None`` if it is a rewrite.

    The test is done in normalized space so that casing and whitespace differences
    — which are not rewrites — do not lose the offsets, and the trailing-punctuation
    variant is tried because the model reliably terminates its sentences whether the
    source did or not.

    The result is re-checked against the original slice before it is trusted. Case
    folding is not always length-preserving, so a match whose boundary falls inside
    an expanded fold can map back one character short; failing that check demotes
    the proposition to a rewrite rather than emitting a span that does not contain
    it.
    """
    for needle in (_normalize(proposition), _normalize(proposition).strip(_TERMINATORS)):
        if not needle:
            continue
        position = normalized.find(needle)
        if position < 0:
            continue
        start = int(origins[position])
        end = int(origins[position + len(needle)])
        if end > start and _normalize(content[start:end]) == needle:
            return start, end
    return None
