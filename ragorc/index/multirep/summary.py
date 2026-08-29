"""Summary indexing: search a summary, generate from the source.

Same premise as the rest of this package — the retrieval unit and the generation
unit are different objects — with a different substitution. Here the searchable
representation is an LLM-written summary of the chunk, and the chunk itself is
kept behind a ``parent_id`` and swapped back in at generation time. Nothing is
lost: the summary is what the query is matched against, the original text is what
the model reads.

When a summary is the better search target
------------------------------------------
When the source material is **long, meandering, or transcript-like**. A dense
vector is a single point, and a 2000-token stretch of a meeting transcript is
about eight things at once, so its embedding lands at the centroid of eight
topics and is sharply similar to none of them. Ask "what did we decide about the
Postgres migration" and the transcript chunk that contains the decision loses to
a marketing page that is *entirely* about Postgres. Summarizing removes the
filler — the greetings, the digressions, the restatements — and the resulting
vector points at one thing. The same argument applies to earnings calls,
interviews, support threads, meandering wiki pages and anything with a high ratio
of connective tissue to content.

When it is worse, and this is the more important half
----------------------------------------------------
When the source is **dense reference text whose exact wording is the searchable
signal**. API documentation, statutes, error messages, configuration keys, SQL
schemas, changelogs. There the user's query often *quotes* the source —
``ERR_CONNECTION_RESET``, ``max_connection_pool_size``, "Article 6(1)(f)" — and a
summary that says "describes connection pool tuning options" has paraphrased away
the one token that would have matched. Sparse and BM25 retrieval, which is where
exact-term matching actually happens, degrades hardest: the summary simply does
not contain the term.

The asymmetry that decides it: summarizing is a **lossy, permanent, per-chunk
paid** transformation. If the summary drops the identifier, no query-time
technique gets it back, and you paid one model call per chunk for the privilege.
So this indexer earns its keep on prose-heavy corpora and quietly costs recall on
reference corpora — which is why ``indexing.summary_index_enabled`` is off by
default and belongs to the corpus, not to the deployment.

Cost and quality notes
----------------------
One LLM call per chunk at ingest, batched concurrently under
``llm.max_concurrency``. :class:`Task.SUMMARIZE` sits on the *balanced* tier
rather than the fast one on purpose (see :mod:`ragorc.llm.router`): the summary
becomes the retrieval target permanently, so a cheap bad summary is a permanently
bad index entry — unlike a cheap bad grade, which only affects one query.

Two things this module deliberately does not do:

* **It does not summarize a chunk that is already short.** Below
  :data:`_SUMMARIZE_FLOOR_TOKENS` the "summary" would be about as long as the
  source, so the call buys a worse search target for real money. Those chunks are
  indexed as themselves. This is also the reason to raise ``indexing.chunk_size``
  when enabling this indexer: at the 512-character default a chunk is ~110 tokens
  and the compression is modest, and the technique pays for itself on the
  1500-token chunks it was designed for.
* **It does not copy the source text into the summary's payload.** One summary per
  chunk is a 1:1 fan-out, so inlining would only double storage rather than
  multiply it eightfold as it would for parent/child — but it is still a copy of
  the corpus in the vector store, and ``expand_parents`` resolves the pointers in
  one batched query at query time.

Late chunking does not compose with this representation, and cannot. A summary is
new text, not a span of any document, so there is no document-wide forward pass to
pool it out of; summaries are embedded on their own. That is a real quality cost
relative to parent-document indexing, and it is the reason to reach for this
technique second.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog

from ragorc.core.concurrency import map_concurrent
from ragorc.core.errors import BudgetExceeded, RagOrcError
from ragorc.core.ids import stable_uuid
from ragorc.core.models import Chunk, Modality, ScoredChunk, Usage
from ragorc.core.protocols import LLM
from ragorc.core.registry import register
from ragorc.core.schemas import SummaryOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.core.tokens import count_tokens, truncate_to_tokens
from ragorc.index.multirep.parent_document import expand_parents
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = ["SummaryIndexer"]

_SUMMARY_COMPRESSION = 3
"""Target ratio of source tokens to summary tokens.

A summary that is not materially shorter than its source has bought nothing: the
same index size, the same context cost, and a paraphrase instead of the author's
words. A third is the point where the prompt can still carry every entity, number
and date — which is what ``summarize_chunk`` asks for — while dropping the
connective tissue that blurred the vector."""

_SUMMARY_MIN_TOKENS = 48
"""Floor on the requested length. Squeezing a long chunk into 20 tokens produces a
topic label, not a summary, and a topic label matches every query about the topic
equally — precisely the dilution this technique exists to remove."""

_SUMMARY_MAX_TOKENS = 320
"""Ceiling. Past this the summary stops being a search key and starts being a
second copy of the chunk."""

_SUMMARIZE_FLOOR_TOKENS = _SUMMARY_MIN_TOKENS * 3 // 2
"""Chunks below this are indexed as themselves.

Derived from ``_SUMMARY_MIN_TOKENS`` rather than chosen separately, because the
two cannot be allowed to drift apart: a source only a few tokens longer than the
shortest summary we would write yields a retyping, not a compression, and the model
call is pure loss. At half again the minimum the worst case is still a ~1.5x
reduction, which is worth one call."""

_JSON_ENVELOPE_TOKENS = 32
"""Slack added to ``max_tokens`` for the structured-output wrapper. The completion
is ``{"summary": "...", "title": "..."}``, so capping ``max_tokens`` at exactly
the summary budget truncates the JSON and fails validation instead of shortening
the prose."""


@register("indexer", "summary")
class SummaryIndexer:
    """Turns chunks into summary-keyed searchable units."""

    name = "summary"

    def __init__(
        self,
        llm: LLM,
        *,
        router: ModelRouter | None = None,
        embedder: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.config = self.settings.indexing
        self.router = router or ModelRouter(self.settings.llm)
        self.embedder = embedder
        self.prompt = get_prompt("summarize_chunk")

    # ------------------------------------------------------------------
    # Building the representation
    # ------------------------------------------------------------------
    async def build(self, chunks: Sequence[Chunk]) -> tuple[list[Chunk], Usage]:
        """Summarize every chunk concurrently and return the searchable units.

        The result is one unit per input chunk, always — a chunk that was skipped
        as too short, or whose summarization failed, comes back as itself rather
        than being dropped. A gap in the index is invisible at query time and
        permanent until the next full re-ingest, so the failure mode is chosen to
        be "this chunk is indexed slightly worse" rather than "this chunk is
        gone".
        """
        if not chunks:
            return [], Usage()

        with timed("summary_index_build", chunks=len(chunks)):
            outcomes = await map_concurrent(
                self._summarize,
                list(chunks),
                limit=max(1, self.settings.llm.max_concurrency),
                return_exceptions=True,
            )

        units: list[Chunk] = []
        usages: list[Usage] = []
        failures = 0
        summarized = 0
        for source, outcome in zip(chunks, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                # Only genuinely unexpected exceptions reach here; the library's
                # own errors are handled inside _summarize.
                failures += 1
                log.warning(
                    "summary_unexpected_failure",
                    chunk_id=source.id,
                    error=str(outcome),
                    error_type=type(outcome).__name__,
                )
                units.append(self._as_source_unit(source, reason="error"))
                continue
            unit, usage = outcome
            usages.append(usage)
            if unit.modality is Modality.SUMMARY:
                summarized += 1
            units.append(unit)

        total = Usage.sum(usages)
        log.info(
            "summary_index_built",
            chunks=len(chunks),
            summarized=summarized,
            passthrough=len(chunks) - summarized,
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
        """Build, embed the summaries and write both representations.

        Only the summaries are embedded and upserted into the vector store. The
        sources go to the docstore unembedded — they are what
        :func:`~ragorc.index.multirep.parent_document.expand_parents` reads back,
        and a vector on them would be a vector nothing ever queries.
        """
        units, usage = await self.build(chunks)
        if not units:
            return units, usage

        await self._embed(units)
        if docstore is not None:
            # Only the chunks something actually points at need persisting; a
            # passthrough unit *is* its source and is already going to the
            # vector store.
            referenced = {unit.parent_id for unit in units if unit.parent_id}
            sources = [c for c in chunks if c.id in referenced]
            if sources:
                await docstore.upsert_chunks(sources)
        if vector_store is not None:
            await vector_store.upsert(units)
        return units, usage

    async def expand(self, chunks: Sequence[ScoredChunk], store: Any) -> list[ScoredChunk]:
        """Swap each retrieved summary back for the text it summarizes."""
        return await expand_parents(chunks, store)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    async def _summarize(self, source: Chunk) -> tuple[Chunk, Usage]:
        budget = self._budget(source)
        if budget is None:
            return self._as_source_unit(source, reason="too_short"), Usage()

        prompt = self.prompt.render(max_tokens=budget, text=source.content)
        try:
            result, usage = await self.llm.structured(
                prompt,
                SummaryOutput,
                system=self.prompt.system,
                model=self.router.model_for(Task.SUMMARIZE),
                stage="summary_index",
                max_tokens=budget + _JSON_ENVELOPE_TOKENS,
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
            # Degrade to the raw text: an unsummarized chunk still retrieves.
            log.warning(
                "summary_failed",
                chunk_id=source.id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return self._as_source_unit(source, reason="llm_error"), Usage()

        text = " ".join(result.summary.split())
        if not text:
            log.warning("summary_empty", chunk_id=source.id)
            return self._as_source_unit(source, reason="empty"), usage

        # The model overruns its instruction routinely; the cap is enforced here
        # because a summary longer than the source turns the saving negative.
        text = truncate_to_tokens(text, budget)
        return self._as_summary_unit(source, text, result.title.strip()), usage

    def _budget(self, source: Chunk) -> int | None:
        """Requested summary length in tokens, or ``None`` to skip the call."""
        tokens = source.token_count or count_tokens(source.content)
        if tokens < _SUMMARIZE_FLOOR_TOKENS:
            return None
        target = tokens // _SUMMARY_COMPRESSION
        return max(_SUMMARY_MIN_TOKENS, min(target, _SUMMARY_MAX_TOKENS))

    def _as_summary_unit(self, source: Chunk, text: str, title: str) -> Chunk:
        """Wrap a summary as a searchable chunk pointing back at its source.

        ``start_char``/``end_char`` are zero, not the source's span. The citation
        layer computes document offsets as ``chunk.start_char + content.find(quote)``
        (:mod:`ragorc.generate.citations`), and a quote found in a *summary* has no
        position in the document at all — carrying the source span here would
        produce offsets that look valid and point at the wrong words. The true
        span travels in metadata, and the packer restores the real base offset
        from ``parent_start_char`` once it has substituted the source text.
        """
        metadata = dict(source.metadata)
        metadata["representation"] = "summary"
        metadata["parent_start_char"] = source.start_char
        metadata["parent_end_char"] = source.end_char
        metadata["source_index"] = source.index
        if title:
            metadata["title"] = title
        return Chunk(
            # Content-derived and namespaced: an edited chunk yields a new summary
            # id so the stale vector is replaced, and a summary can never collide
            # with the chunk it summarizes.
            id=stable_uuid("summary", source.id, text),
            content=text,
            document_id=source.document_id,
            index=source.index,
            metadata=metadata,
            parent_id=source.id,
            modality=Modality.SUMMARY,
            token_count=count_tokens(text),
            tenant_id=source.tenant_id,
        )

    @staticmethod
    def _as_source_unit(source: Chunk, *, reason: str) -> Chunk:
        """Index the chunk as itself, recording why it was not summarized."""
        source.metadata["representation"] = "source"
        source.metadata["summary_skipped"] = reason
        return source

    async def _embed(self, units: Sequence[Chunk]) -> None:
        """Embed the searchable units, whatever they turned out to be.

        ``embed_text`` rather than ``content``, so a contextual prefix that
        survived from the splitter is embedded with the summary — the two
        techniques stack, and the prefix costs nothing here.
        """
        if self.embedder is None or not units:
            return
        vectors = await self.embedder.embed_documents([unit.embed_text for unit in units])
        for unit, vector in zip(units, vectors, strict=True):
            unit.dense = vector
