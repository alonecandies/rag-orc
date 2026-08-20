"""Parent-document indexing: search the sentence, answer from the section.

The premise of this whole package, stated once here because every module in it is
a variation on it: **the retrieval unit and the generation unit do not have to be
the same object.** Search wants a small, single-topic representation — a
256-character child has one subject, so its vector is not the average of five
topics and it either matches the query or it does not. Generation wants the
opposite: the 2048-character parent that carries the definition the sentence
depends on, the table it refers to, and the paragraph that says which release it
applies to. Indexing one object and hoping it serves both roles is a compromise
nobody has to make.

Why the parent split runs first
-------------------------------
The document is split twice, parents first, and the child split runs *inside*
each parent rather than across the document. A child that straddled two parents
would have no single parent to expand to, and picking one of them would hand the
generator text the child never came from. Child offsets therefore come out
relative to the parent and are shifted by the parent's ``start_char``, which
keeps the invariant every later stage depends on — ``document.content[start:end]
== chunk.content`` — true for children and parents alike. That is what keeps late
chunking (ADR-0002) available for the child vectors: the pooler converts a
child's character span into a token span of the *document's* single forward pass,
so the child vector ends up conditioned on the whole document even though the
child text is one sentence.

Parents are split with zero overlap on purpose. Overlapping parents would repeat
the same paragraph twice in the prompt whenever two adjacent parents were both
expanded, and a child sitting in the overlap would have two equally valid
parents with no principled way to choose. Children keep their overlap: they are
the search unit, and the boundary case where the answer straddles a cut is the
one failure retrieval cannot recover from.

Why ``parent_text`` is not written at index time
------------------------------------------------
Only the children are embedded and upserted into the vector store; the parents go
to the docstore (Postgres) with no vectors at all, because nothing ever searches
them. The tempting shortcut is to copy the parent body into every child's
payload so retrieval needs no second round trip. With the default 2048/256 sizes
that stores the entire corpus roughly eight to ten times over inside the vector
store's payload — and a vector store's payload is the expensive place to keep
text, since ``on_disk_payload`` still has to read it back on every hit. So the
child stores a ``parent_id`` and nothing else, and :func:`expand_parents` fetches
the bodies for one result set in a single batched query.

Why expansion runs after ranking, never before
----------------------------------------------
Ordering matters three separate ways, all in the same direction:

1. **Precision.** The child is the precise key. Score the parent instead and the
   dilution the child split exists to remove comes straight back.
2. **Cost.** Expanding before ranking means fetching parents for the whole
   candidate set — ``fetch_k`` per retriever per query variant, easily 200 rows —
   instead of the ``top_k`` that survive. Same query count, an order of magnitude
   more bytes.
3. **Deduplication.** Several matching children of one parent must yield that
   parent exactly once, and the survivor should be the best-scoring child. That
   choice can only be made after the scores exist.

:func:`expand_parents` writes ``metadata["parent_text"]``, which
:mod:`ragorc.context.pack` already substitutes for ``content`` at pack time —
after selection and after reordering, so precision is preserved where it is
earned and breadth is added where it is used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.concurrency import bounded_gather, map_concurrent
from ragorc.core.errors import ConfigError
from ragorc.core.ids import chunk_id, stable_uuid
from ragorc.core.models import Chunk, Document, ScoredChunk
from ragorc.core.protocols import Splitter
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.index.split import build_splitter
from ragorc.validate.schema import DocumentValidator

log = structlog.get_logger(__name__)

__all__ = [
    "ParentDocumentIndex",
    "ParentDocumentIndexer",
    "expand_parents",
    "scoped_settings",
]


def scoped_settings(base: Settings, *, target: int, overlap: int) -> Settings:
    """A deep copy of ``base`` whose chunk sizes are pinned to ``target``.

    The splitters read their sizes from ``settings.indexing``, so re-sizing them
    means handing them different settings rather than different arguments — which
    is also the only way this works for an *injected* splitter whose constructor
    we do not control.

    ``max_chunk_size`` is pinned to ``target`` as well. Leaving the global 2048
    ceiling in place while asking for 256-character children lets
    ``merge_small_spans`` weld a run of short lines into a "child" eight times the
    requested size, which quietly turns the child index back into a parent index.
    """
    scoped = base.model_copy(deep=True)
    size = max(target, 1)
    scoped.indexing.chunk_size = size
    scoped.indexing.max_chunk_size = size
    scoped.indexing.min_chunk_size = min(base.indexing.min_chunk_size, max(size // 4, 1))
    scoped.indexing.chunk_overlap = max(min(overlap, size // 4), 0)
    return scoped


@dataclass(slots=True)
class ParentDocumentIndex:
    """The two halves of the representation, kept apart because they go to
    different stores: ``children`` are embedded and upserted into the vector
    store, ``parents`` are written to the docstore with no vectors."""

    children: list[Chunk] = field(default_factory=list)
    parents: list[Chunk] = field(default_factory=list)

    def extend(self, other: ParentDocumentIndex) -> None:
        self.children.extend(other.children)
        self.parents.extend(other.parents)

    def parents_by_id(self) -> dict[str, Chunk]:
        return {parent.id: parent for parent in self.parents}

    def __len__(self) -> int:
        """Length is the number of *searchable* units — the children."""
        return len(self.children)


@register("indexer", "parent_document")
class ParentDocumentIndexer:
    """Builds the child/parent pair and resolves parents for a result set.

    No LLM is involved, which makes this the cheapest multi-representation option
    by a wide margin: the other three in this package cost one model call per
    chunk at ingest, and this one costs a second pass of a character splitter.
    When parent-document indexing is good enough for a corpus, it is the right
    answer purely on price.
    """

    name = "parent_document"

    def __init__(
        self,
        *,
        parent_splitter: Splitter | None = None,
        child_splitter: Splitter | None = None,
        embedder: Any | None = None,
        validator: DocumentValidator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.indexing
        self.validator = validator or DocumentValidator(self.settings)
        self.embedder = embedder
        # Both levels use the corpus's configured strategy — a markdown corpus
        # wants markdown parents *and* markdown children — differing only in the
        # size settings handed to it. Parents get no overlap (see the module
        # docstring); children keep theirs.
        self.parent_splitter: Splitter = parent_splitter or build_splitter(
            embedder=embedder,
            settings=scoped_settings(
                self.settings, target=self.config.parent_chunk_size, overlap=0
            ),
        )
        self.child_splitter: Splitter = child_splitter or build_splitter(
            embedder=embedder,
            settings=scoped_settings(
                self.settings,
                target=self.config.child_chunk_size,
                overlap=self.config.chunk_overlap,
            ),
        )

    # ------------------------------------------------------------------
    # Building the representation
    # ------------------------------------------------------------------
    async def build(self, document: Document) -> ParentDocumentIndex:
        """Split one validated document into children and parents."""
        return await self._build(self.validator.validate_document(document))

    async def build_many(self, documents: Sequence[Document]) -> ParentDocumentIndex:
        """Split a batch, dropping documents the validator rejects.

        Rejections are collected rather than raised: one unparseable file in a
        10k-document crawl must not abort the crawl, and the validator's report
        already carries the reasons.
        """
        report = self.validator.validate_batch(list(documents))
        parts = await map_concurrent(
            self._build,
            report.accepted,
            limit=max(1, self.config.max_concurrent_documents),
        )
        combined = ParentDocumentIndex()
        for part in parts:
            combined.extend(part)
        return combined

    async def _build(self, document: Document) -> ParentDocumentIndex:
        parents = await self.parent_splitter.split(document)
        if not parents:
            return ParentDocumentIndex()

        # One child split per parent. With the default recursive child splitter
        # this is pure CPU and the gather buys nothing; with an injected semantic
        # child splitter every parent is an embedding batch, and the bound is what
        # stops a 300-parent document from firing 300 batches at once.
        groups = await bounded_gather(
            (self._split_children(document, parent) for parent in parents),
            limit=max(1, self.config.max_concurrent_documents),
        )

        result = ParentDocumentIndex()
        child_index = 0
        for parent_index, (parent, raw_children) in enumerate(zip(parents, groups, strict=True)):
            # A parent id must live in its own namespace. A parent that yielded a
            # single child covering all of it has byte-identical content at the
            # same index, so `chunk_id` would hand both the same id — and the
            # docstore row would then overwrite the vector-store point they share.
            parent_id = stable_uuid("parent_chunk", document.id, parent_index, parent.content)

            children: list[Chunk] = []
            for raw in raw_children:
                content = raw.content
                metadata = dict(raw.metadata)
                metadata["representation"] = "child"
                metadata["parent_index"] = parent_index
                children.append(
                    Chunk(
                        id=chunk_id(document.id, child_index, content),
                        content=content,
                        document_id=document.id,
                        index=child_index,
                        # Offsets are shifted, never recomputed: the child came
                        # out of a slice of the document, so parent.start_char +
                        # relative offset is exact, and the late-chunking pooler
                        # can still find this span in the document's forward pass.
                        start_char=parent.start_char + raw.start_char,
                        end_char=parent.start_char + raw.end_char,
                        metadata=metadata,
                        parent_id=parent_id,
                        modality=raw.modality,
                        contextual_prefix=raw.contextual_prefix,
                        token_count=raw.token_count,
                        tenant_id=document.tenant_id,
                    )
                )
                child_index += 1

            parent_metadata = dict(parent.metadata)
            parent_metadata["representation"] = "parent"
            parent_metadata["child_count"] = len(children)
            result.parents.append(
                Chunk(
                    id=parent_id,
                    content=parent.content,
                    document_id=document.id,
                    index=parent_index,
                    start_char=parent.start_char,
                    end_char=parent.end_char,
                    metadata=parent_metadata,
                    children_ids=tuple(child.id for child in children),
                    modality=parent.modality,
                    token_count=parent.token_count,
                    tenant_id=document.tenant_id,
                )
            )
            result.children.extend(children)

        log.debug(
            "parent_document_built",
            document_id=document.id,
            parents=len(result.parents),
            children=len(result.children),
        )
        return result

    async def _split_children(self, document: Document, parent: Chunk) -> list[Chunk]:
        """Split one parent's text, with the parent itself as the floor.

        The synthetic document is what lets an arbitrary :class:`Splitter` be
        reused unchanged — it plans against ``content`` and returns offsets into
        it, which is exactly what we need to shift. If it returns nothing (a
        parent that is entirely punctuation, say) the parent is emitted as its own
        single child rather than dropped: an unreachable parent is text that was
        silently removed from the index, which is the class of defect nobody
        notices for months.
        """
        sub_document = Document(
            id=document.id,
            content=parent.content,
            metadata=document.metadata,
            source=document.source,
            title=document.title,
            modality=parent.modality,
            tenant_id=document.tenant_id,
        )
        children = await self.child_splitter.split(sub_document)
        if children:
            return children
        log.warning(
            "parent_yielded_no_children",
            document_id=document.id,
            start_char=parent.start_char,
            end_char=parent.end_char,
        )
        return [
            Chunk(
                id=parent.id,
                content=parent.content,
                document_id=document.id,
                start_char=0,
                end_char=len(parent.content),
                metadata=dict(parent.metadata),
                modality=parent.modality,
                token_count=parent.token_count,
            )
        ]

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------
    async def index(
        self,
        documents: Sequence[Document],
        *,
        vector_store: Any | None = None,
        docstore: Any | None = None,
    ) -> ParentDocumentIndex:
        """Validate, split, embed the children and write both halves.

        Two writes, two stores, and only one of them holds vectors. The children
        are validated a second time with :meth:`DocumentValidator.validate_chunks`
        because they are the units that reach the index, and a child too short to
        carry meaning still costs an embedding call and a vector slot while being
        unretrievable. Parents skip that check: they are never searched, and
        dropping one would orphan every child pointing at it.
        """
        report = self.validator.validate_batch(list(documents))
        if not report.accepted:
            return ParentDocumentIndex()

        with timed("parent_document_index", documents=len(report.accepted)):
            parts = await map_concurrent(
                self._prepare,
                report.accepted,
                limit=max(1, self.config.max_concurrent_documents),
            )
        combined = ParentDocumentIndex()
        for part in parts:
            combined.extend(part)
        combined.children = self.validator.validate_chunks(combined.children)

        if docstore is not None and combined.parents:
            await docstore.upsert_chunks(combined.parents)
        if vector_store is not None and combined.children:
            await vector_store.upsert(combined.children)

        log.info(
            "parent_document_indexed",
            documents=len(report.accepted),
            rejected=len(report.rejected),
            parents=len(combined.parents),
            children=len(combined.children),
        )
        return combined

    async def _prepare(self, document: Document) -> ParentDocumentIndex:
        built = await self._build(document)
        await self._embed_children(document, built.children)
        return built

    async def _embed_children(self, document: Document, children: Sequence[Chunk]) -> None:
        """Embed the children — and only the children — in place.

        The late-chunking path is preferred when the injected embedder offers it,
        and the reason is exactly the weakness of small chunks: a 256-character
        child has lost the referents that made it meaningful, and pooling its span
        out of one document-wide forward pass puts them back into the vector
        without putting them into the text. One pass per document, not one per
        child, so it is also the cheaper of the two.
        """
        embedder = self.embedder
        if embedder is None or not children:
            return
        pooled = getattr(embedder, "embed_chunks", None)
        if callable(pooled):
            vectors = await pooled(document.content, [(c.start_char, c.end_char) for c in children])
        else:
            vectors = await embedder.embed_documents([c.embed_text for c in children])
        for chunk, vector in zip(children, vectors, strict=True):
            chunk.dense = vector

    # ------------------------------------------------------------------
    # Query time
    # ------------------------------------------------------------------
    async def expand(self, chunks: Sequence[ScoredChunk], store: Any) -> list[ScoredChunk]:
        """Instance-level alias for :func:`expand_parents`."""
        return await expand_parents(chunks, store)


async def expand_parents(chunks: Sequence[ScoredChunk], store: Any) -> list[ScoredChunk]:
    """Resolve every hit's parent body in ONE batched query.

    Deduplication happens before the fetch, not after: several children of one
    parent collapse to the highest-scoring child, so the batch asks for each
    parent once and the prompt contains it once. The dropped siblings are not
    discarded silently — their ``component_scores`` merge into the survivor and
    their ids land in ``explain["parent_siblings"]``, so "this parent won because
    four of its children matched" stays visible after the fact.

    Only ``metadata["parent_text"]`` is written here. The substitution itself is
    :mod:`ragorc.context.pack`'s job and happens after selection and reordering,
    which is what keeps a large parent from being weighed as if it were the
    precise thing that matched.

    A parent that is missing from the docstore, or a docstore that is down, leaves
    its child untouched rather than dropping it. Losing breadth degrades the
    answer; losing the hit loses the evidence.
    """
    if not chunks:
        return []

    survivors = _dedupe_by_parent(chunks)
    wanted = [
        scored.chunk.parent_id
        for scored in survivors
        if scored.chunk.parent_id and "parent_text" not in scored.chunk.metadata
    ]
    parents = await _fetch_parents(store, list(dict.fromkeys(wanted)))

    resolved = 0
    for rank, scored in enumerate(survivors):
        scored.rank = rank
        parent = parents.get(scored.chunk.parent_id or "")
        if parent is None or not parent.content:
            continue
        meta = scored.chunk.metadata
        meta["parent_text"] = parent.content
        # Provenance for the citation layer: after the packer swaps `content` for
        # the parent body, the chunk's own start/end no longer describe what the
        # generator saw, and these do.
        meta["parent_start_char"] = parent.start_char
        meta["parent_end_char"] = parent.end_char
        scored.explain["representation"] = "parent_document"
        resolved += 1

    log.debug(
        "parents_expanded",
        hits=len(chunks),
        kept=len(survivors),
        requested=len(wanted),
        resolved=resolved,
    )
    return survivors


def _dedupe_by_parent(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Collapse children sharing a parent, keeping the best-scoring one.

    The representative is chosen by score in a first pass and emitted in the
    caller's order in a second, so the result does not depend on whether the
    input happened to arrive sorted — it survives an unsorted candidate set and a
    reranker that rewrote the scores but not the order.
    """
    best: dict[str, ScoredChunk] = {}
    for scored in chunks:
        key = scored.chunk.parent_id or scored.chunk.id
        prior = best.get(key)
        if prior is None or scored.score > prior.score:
            best[key] = scored

    out: list[ScoredChunk] = []
    for scored in chunks:
        key = scored.chunk.parent_id or scored.chunk.id
        winner = best[key]
        if winner is scored:
            out.append(scored)
            continue
        winner.component_scores.update(scored.component_scores)
        winner.explain.setdefault("parent_siblings", []).append(scored.chunk.id)
    return out


async def _fetch_parents(store: Any, ids: Sequence[str]) -> dict[str, Chunk]:
    """One batched read, whichever docstore shape the caller passed.

    ``get_chunks`` is Postgres', ``get`` is the vector store's; both take a list
    of ids and issue a single statement. A store with neither is a configuration
    mistake and says so immediately, because the alternative — silently returning
    no parents — looks exactly like a corpus that has none.
    """
    if not ids:
        return {}
    getter = getattr(store, "get_chunks", None) or getattr(store, "get", None)
    if not callable(getter):
        raise ConfigError(
            "parent expansion needs a store with get_chunks(ids) or get(ids)",
            store=type(store).__name__,
        )
    try:
        rows = await getter(ids)
    except Exception as exc:  # noqa: BLE001 - degrade to unexpanded children
        log.warning("parent_fetch_failed", error=str(exc), error_type=type(exc).__name__)
        return {}
    return {row.id: row for row in rows}
