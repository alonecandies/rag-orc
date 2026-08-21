"""The ingest orchestrator: documents in, vectors out, exactly once.

Every stage this module drives already exists and none of them know about each
other: loaders produce documents, splitters produce boundaries, embedders produce
vectors, stores accept writes. What is left — and what actually decides whether an
ingest is cheap, restartable and correct — is the *order*, the *memory policy* and
the *failure policy*. That is this file.

The checksum skip is the headline
---------------------------------
Before anything is split or embedded, the pipeline asks the relational store for
the stored ``(id, checksum)`` of the documents it is about to write and drops the
ones that did not move. Loaders make this possible by deriving the id from the
*source* and the checksum from the *content* (see :mod:`ragorc.index.loaders`), so
an unchanged file matches and an edited one does not.

The saving is not marginal. A nightly re-ingest of a 100k-document corpus with a
1% change rate costs 1k documents of embedding instead of 100k — two orders of
magnitude — and the probe that achieves it is a handful of indexed primary-key
lookups. It is done in **one batched query per few hundred documents**, never one
per document: at 5 ms per round trip, a per-document probe over 100k documents is
eight minutes of pure latency spent deciding to do nothing.

Backpressure: neither the chunk list nor the document list is materialized
-------------------------------------------------------------------------
Documents are processed ``max_concurrent_documents`` at a time and their chunks are
streamed to the stores in ``batch_size`` batches. Nothing accumulates.

The document list used to, which made the bound below false at the scale it was
quoted for: ingesting a directory loaded every document's text before the first
vector was written, and at 100k documents the document list is the larger of the
two numbers. A directory is now read ``indexing.document_window`` documents at a
time (512, chosen so that anything smaller behaves exactly as it did — one window,
one pass). An explicit list of documents is not windowed: it is already in the
caller's memory, so bounding it here would bound nothing.

The arithmetic is the reason. A 100k-document corpus at ~40 chunks per document is
4M chunks; each carries its text (~2 KB) plus a dense vector (384 float32 = 1.5 KB)
plus a sparse vector, so building the full list before the first write needs tens
of gigabytes and then performs one all-or-nothing write at the end. Streaming
bounds live memory to one window of documents, ``max_concurrent_documents``
documents' worth of chunks and one pending batch — tens of megabytes, independent
of corpus size — and every batch
that lands is durable, so a crash costs one batch instead of the whole run.

Failure policy: two rules, opposite directions
----------------------------------------------
* **A document failure is recorded and the run continues.** One corrupt PDF in
  10k files must not discard the 9,999 that parsed. The document is counted in
  ``failed`` with its exception, and because ids are deterministic a later run
  retries exactly that document and nothing else.
* **A store failure aborts the run.** If nothing can be written, continuing to
  embed spends real money to produce vectors that have nowhere to go. Aborting is
  also safe to retry: every id is content-derived, so a restarted ingest rewrites
  the same rows rather than duplicating them, and a partially written batch is
  simply overwritten.

The same asymmetry applies to the optional stages: they are enrichment, so a
failing stage is disabled for the rest of the run with a loud warning while the
documents themselves still get indexed. Losing a RAPTOR tree is recoverable;
losing the leaf chunks is not.

Order of writes, and why it is not arbitrary
--------------------------------------------
1. **Document rows before chunk rows.** ``ragorc_chunks.document_id`` is a foreign
   key; writing chunks first fails the constraint for every new document.
2. **Purge stale chunks only after the replacement chunks exist in memory.** A
   changed document's old chunk ids are content-derived, so they do not collide
   with the new ones and an upsert alone would leave the previous version in the
   index forever — retrievable, quotable and wrong. The purge is therefore
   mandatory, but it runs per batch and only after the new chunks are in hand, so
   a document whose extraction has started failing keeps its old vectors instead
   of vanishing.
3. **Large runs write inside Qdrant's bulk-load mode**
   (:func:`ragorc.stores.qdrant.collections.bulk_load_mode`). With indexing on,
   the optimizer builds HNSW graphs over segments that are still growing and
   rebuilds them repeatedly; deferring the build makes ingest an append and
   constructs each graph once. Below :data:`BULK_LOAD_MIN_DOCUMENTS` the
   collection-config round trips and the wait-for-green on exit cost more than the
   deferral saves.

The strategy ladder
-------------------
``resolve_strategy`` (ADR-0002) is called **once per run**, not per document, and
what it returns changes which embedding path executes:

* ``LATE`` — one forward pass per document, then mean-pool over each chunk's exact
  ``start_char``/``end_char``. Cheaper *and* better; the offsets come straight from
  the splitter and are never recomputed.
* ``CONTEXTUAL`` — an LLM writes a situating prefix per chunk, then normal
  embedding of ``chunk.embed_text``. One model call per chunk, so it is opt-in.
* ``EARLY`` — batched embedding of each chunk in isolation. The floor.

One consequence is easy to miss and expensive to discover: under ``LATE`` the
vectors come out of the late-chunking backend, whose pooled space and width belong
to *that* model, not to ``embedding.dense_model``. So the same object is wired in
as the store's query-side embedder and the true dimension is measured with one
probe forward pass before the collection is created. Mixing a pooled document
space with a differently-pooled query space is a silent recall collapse, and a
dimension mismatch is an opaque insert error thousands of vectors later.

No cost ledger is installed for the run: ``cost.max_cost_per_query_usd`` is a
per-*query* ceiling and enforcing it over a corpus-sized job would abort a
perfectly legitimate ingest. The bill is reported instead, per stage, in
:class:`IngestReport`.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import time
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import BudgetExceeded, StoreUnavailable
from ragorc.core.ids import content_hash
from ragorc.core.models import Chunk, ChunkingStrategy, Document, Usage
from ragorc.core.protocols import (
    LLM,
    DenseEmbedder,
    LateInteractionEmbedder,
    RelationalStore,
    SparseEmbedder,
    Splitter,
    VectorStore,
)
from ragorc.core.registry import resolve
from ragorc.core.settings import IndexingSettings, Settings, get_settings
from ragorc.core.telemetry import current_ledger, timed
from ragorc.embed.late_chunking import LateChunkingEmbedder, resolve_strategy
from ragorc.index.loaders import DirectoryLoader
from ragorc.index.loaders import load as load_source
from ragorc.index.split import build_splitter
from ragorc.stores.postgres.ddl import chunks_table, documents_table
from ragorc.validate.schema import DocumentValidator

log = structlog.get_logger(__name__)

__all__ = [
    "BULK_LOAD_MIN_DOCUMENTS",
    "IndexStage",
    "IngestPipeline",
    "IngestReport",
    "RelationalIngestStore",
]

T = TypeVar("T")

BULK_LOAD_MIN_DOCUMENTS = 64
"""Documents above which the run writes inside Qdrant's bulk-load mode. Below it,
reading and restoring the collection's ``indexing_threshold`` plus waiting for the
index to go green costs more than the deferred graph build saves."""

_DIMENSION_PROBE = "dimension probe"
"""Text used for the one warm-up forward pass. It also loads the model, so the
first real document does not pay a 0.5-3 s ONNX session load with
``max_concurrent_documents`` tasks queued behind it."""

_ABORTING: tuple[type[BaseException], ...] = (
    StoreUnavailable,
    BudgetExceeded,
    asyncio.CancelledError,
    KeyboardInterrupt,
    SystemExit,
)
"""Exceptions that end the run instead of being charged to one document. A dead
store, an exhausted budget or a cancellation apply to every remaining document, so
recording them per document would produce N copies of one failure."""


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class IngestReport:
    """What one ingest run did. The operational answer to "did it land?".

    ``timings_ms`` sums each stage across documents that ran *concurrently*, so the
    stage totals add up to more than ``total_ms``. That is intended: the ratios
    identify the bottleneck (embedding vs extraction vs writing), where wall-clock
    per stage would only ever report the pipeline's own concurrency back at you.
    """

    documents_in: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_rejected: int = 0
    documents_duplicate: int = 0
    documents_failed: int = 0
    documents_empty: int = 0
    """Documents that survived validation and produced no indexable chunk — every
    chunk was shorter than ``indexing.min_chunk_size`` or carried no word
    characters. It exists because without it these documents were counted
    *nowhere*: a corpus of one-line FAQ or glossary files reported ``indexed: 0,
    rejected: 0, failed: 0`` over an empty index, and the counters did not
    reconcile with ``documents_in``. Deliberately not folded into
    ``documents_skipped``, which means "unchanged since the last ingest" and feeds
    ``skip_rate`` — a healthy signal that must not be inflated by a silent loss."""
    chunks_created: int = 0
    vectors_written: int = 0
    strategy: str = ChunkingStrategy.AUTO.value
    total_ms: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        """Total LLM spend. Non-zero only for the stages that call a model —
        contextual enrichment, RAPTOR summaries, graph extraction."""
        return self.usage.cost_usd

    @property
    def skip_rate(self) -> float:
        """Fraction of accepted documents the checksum comparison skipped. On a
        steady-state corpus this should be close to 1.0; a sudden drop means
        something upstream is rewriting content that did not change."""
        considered = self.documents_skipped + self.documents_indexed + self.documents_failed
        return self.documents_skipped / considered if considered else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "documents_in": self.documents_in,
            "indexed": self.documents_indexed,
            "skipped": self.documents_skipped,
            "rejected": self.documents_rejected,
            "duplicate": self.documents_duplicate,
            "failed": self.documents_failed,
            "empty": self.documents_empty,
            "chunks": self.chunks_created,
            "vectors": self.vectors_written,
            "strategy": self.strategy,
            "skip_rate": round(self.skip_rate, 3),
            "cost_usd": round(self.cost_usd, 6),
            "llm_calls": self.usage.calls,
            "total_ms": round(self.total_ms, 1),
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
        }


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
class RelationalIngestStore(RelationalStore, Protocol):
    """The relational store as the *write* path uses it.

    :class:`~ragorc.core.protocols.RelationalStore` describes the read half: the
    surface Text-to-SQL and full-text search need. Ingest also applies the schema
    and writes documents and chunks, and always has — those four methods are
    called on whatever store is injected here, they were simply never part of the
    declared protocol. Stating them makes the requirement checkable, so a
    read-only implementation is rejected up front instead of by an
    ``AttributeError`` partway through a run that has already embedded a batch.

    The parameters are positional-only because the pipeline only ever passes them
    positionally; pinning the argument *names* would exclude conforming
    implementations for no reason.
    """

    async def ensure_schema(self) -> None: ...

    async def upsert_documents(self, documents: Sequence[Document], /) -> int: ...

    async def upsert_chunks(self, chunks: Sequence[Chunk], /) -> int: ...

    async def delete_document(self, document_id: str, /) -> int: ...


# ---------------------------------------------------------------------------
# Optional stages
# ---------------------------------------------------------------------------
class IndexStage(Protocol):
    """The shape of an optional enrichment stage.

    Multi-representation indexing, RAPTOR and GraphRAG all do the same thing from
    this pipeline's point of view: take a document and its leaf chunks, return the
    chunks to write plus the LLM bill. They are *plugins*, not imports — each needs
    an extra that may not be installed (``ragorc[raptor]``, ``ragorc[graphrag]``)
    and each is off by default — so they are located at run time and a missing one
    degrades the ingest instead of breaking it.
    """

    async def enrich(
        self, document: Document, chunks: list[Chunk]
    ) -> tuple[list[Chunk], Usage]: ...


@dataclass(frozen=True, slots=True)
class _Plugin:
    """How to find and call one sibling stage.

    Several module and attribute names are accepted because these modules are
    developed alongside this one; the first that resolves wins and the rest are
    never touched.
    """

    label: str
    modules: tuple[str, ...]
    factories: tuple[str, ...]
    methods: tuple[str, ...] = ("enrich", "run", "index", "apply")
    extra: str = ""


_CONTEXTUAL = _Plugin(
    label="contextual",
    modules=("ragorc.index.contextual", "ragorc.index.multirep.contextual"),
    factories=("ContextualEnricher", "build_contextual_enricher"),
    methods=("enrich", "contextualize", "run"),
)

_OPTIONAL_STAGES: tuple[_Plugin, ...] = (
    _Plugin(
        label="multirep",
        modules=("ragorc.index.multirep",),
        factories=("MultiRepresentationIndexer", "build_multirep", "MultiRepIndexer"),
    ),
    _Plugin(
        label="raptor",
        modules=("ragorc.index.raptor", "ragorc.index.raptor.tree"),
        factories=("RaptorIndexer", "RaptorBuilder", "build_raptor"),
        extra="raptor",
    ),
)

#: Corpus-wide passes, which a streaming ingest cannot host. Listed here so
#: ``graph.enabled`` produces an instruction instead of silence: entity
#: resolution and community detection are only meaningful over the whole corpus,
#: and this pipeline deliberately holds one document's chunks at a time (see
#: :meth:`IngestPipeline._enrich`). ``GraphBuilder`` also owns its own writes and
#: returns a build report rather than chunks, so there is no shape in which it
#: could be an enrichment stage — it was previously listed as one and silently
#: failed to construct on every run, because nothing passed it a graph store.
_GRAPH_CORPUS_HINT = (
    "graph.enabled requires a corpus-wide second pass, which a streaming ingest "
    "cannot perform: build it after ingest with "
    "GraphBuilder(llm, graph_store, settings=settings).build(chunks) — see "
    "examples/04_graphrag.py and docs/modules/index.md"
)


def _load_plugin(plugin: _Plugin) -> Any | None:
    """Resolve a stage's factory, or ``None`` when it is not present."""
    for module_name in plugin.modules:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for attr in plugin.factories:
            factory = getattr(module, attr, None)
            if factory is not None:
                return factory
    return None


def _construct(factory: Any, **candidates: Any) -> Any:
    """Instantiate a stage, passing only the arguments it declares.

    Stage constructors differ — one needs an ``llm``, one needs the dense
    embedder, one needs the graph store — and this orchestrator has no business
    tracking which. Filtering by signature keeps the call site stable while the
    stages evolve, and a stage that declares ``**kwargs`` gets everything.
    """
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # C-implemented callable: hand it nothing
        return factory()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return factory(**candidates)
    accepted = {
        name: value
        for name, value in candidates.items()
        if name in signature.parameters and value is not None
    }
    return factory(**accepted)


async def _invoke(stage: Any, plugin: _Plugin, **candidates: Any) -> tuple[list[Chunk], Usage]:
    """Call a stage's enrichment method and normalize its return shape."""
    method = next(
        (m for name in plugin.methods if callable(m := getattr(stage, name, None))),
        None,
    )
    if method is None:
        raise AttributeError(
            f"{type(stage).__name__} exposes none of {plugin.methods}; "
            f"cannot run the {plugin.label} stage"
        )
    result = await _construct_call(method, **candidates)
    if isinstance(result, tuple) and len(result) == 2:
        chunks, usage = result
        return list(chunks), usage if isinstance(usage, Usage) else Usage()
    return list(result), Usage()


async def _construct_call(method: Any, **candidates: Any) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return await method(**candidates)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return await method(**candidates)
    accepted = {name: value for name, value in candidates.items() if name in signature.parameters}
    return await method(**accepted)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _windows(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _count_vectors(chunks: Sequence[Chunk]) -> int:
    """Named vectors actually written, not points.

    A chunk with dense + sparse + ColBERT is three vectors, and it is the vector
    count — not the chunk count — that determines index build time and storage.
    """
    total = 0
    for chunk in chunks:
        total += chunk.dense is not None
        total += chunk.sparse is not None and len(chunk.sparse) > 0
        total += chunk.multi is not None
    return total


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------
class IngestPipeline:
    """Loads, validates, splits, embeds and writes documents.

    Every collaborator is injectable and every one has a default. Injection is the
    point rather than a convenience: the retriever and the ingest pipeline should
    share one loaded ONNX session and one Qdrant connection, so a service builds
    the embedders once and hands the same objects to both.

    The stores are built *after* the chunking strategy is resolved, because the
    strategy decides which embedder is the query-side embedder and therefore what
    the collection's vector dimension has to be.
    """

    def __init__(
        self,
        *,
        vector_store: VectorStore | None = None,
        relational_store: RelationalIngestStore | None = None,
        splitter: Splitter | None = None,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
        late_embedder: LateInteractionEmbedder | None = None,
        late_chunker: LateChunkingEmbedder | None = None,
        llm: LLM | None = None,
        validator: DocumentValidator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config: IndexingSettings = self.settings.indexing
        self.vector = vector_store
        self.relational: RelationalIngestStore | None = relational_store
        self.splitter = splitter
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.late_embedder = late_embedder
        self.late_chunker = late_chunker
        self.llm = llm
        self.validator = validator or DocumentValidator(self.settings)
        self._strategy: ChunkingStrategy | None = None
        self._enricher: Any | None = None
        self._stages: list[tuple[_Plugin, Any]] = []
        self._embedding_cache: Any | None = None
        self._owns_stores = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def ingest(self, target: Any) -> IngestReport:
        """Ingest documents or a source, and report what happened.

        ``target`` is a :class:`~ragorc.core.models.Document`, a path to a file or
        directory, or any iterable mixing the two. Paths go through
        :func:`ragorc.index.loaders.load`, which is what supplies the deterministic
        id and the checksum the skip depends on.
        """
        report = IngestReport()
        run_started = time.perf_counter()

        with timed("ingest"):
            strategy: ChunkingStrategy | None = None
            async for documents in self._document_windows(target, report):
                report.documents_in += len(documents)
                accepted = await self._validate(documents, report)
                if not accepted:
                    continue
                # Resolved on the first window that has work, and reused: the
                # strategy is a property of the configuration and the embedder, not
                # of a window, and re-resolving it would re-measure the model's
                # dimension once per window.
                if strategy is None:
                    strategy = await self._prepare(report)
                    report.strategy = strategy.value
                todo, changed = await self._select_changed(accepted, report)
                if todo:
                    await self._run(todo, changed, strategy, report)

            if not report.documents_in:
                log.warning("ingest_nothing_to_do", target=str(target)[:200])
                return report
            if strategy is None:
                report.total_ms = _elapsed_ms(run_started)
                log.warning("ingest_all_rejected", **report.summary())
                return report

        report.total_ms = _elapsed_ms(run_started)
        log.info("ingest_complete", **report.summary())
        return report

    async def close(self) -> None:
        """Release the stores this pipeline created.

        Injected stores are left alone: they belong to the caller and are almost
        certainly shared with the retriever, so closing them here would break the
        query path.
        """
        if not self._owns_stores:
            return
        for store in (self.vector, self.relational):
            if store is not None:
                with contextlib.suppress(Exception):
                    await store.close()

    # ------------------------------------------------------------------
    # a. collect + validate
    # ------------------------------------------------------------------
    async def _document_windows(
        self, target: Any, report: IngestReport
    ) -> AsyncIterator[list[Document]]:
        """Yield the documents to ingest, a window at a time.

        Only a directory streams. Everything else — a single file, a Document, an
        iterable of them — is already in the caller's memory, so windowing it would
        bound nothing and only complicate the report.

        The module's memory policy claims a bound independent of corpus size. That
        was true of the chunk stream and false of the document list: loading a
        directory materialized every document's text before the first vector was
        written, and at 100k documents the document list is the larger of the two
        numbers. `indexing.document_window` is what makes the claim true.
        """
        started = time.perf_counter()
        window = max(1, self.config.document_window)
        root = target if isinstance(target, Path) else None
        if isinstance(target, str):
            root = Path(target)
        if root is not None and root.is_dir():
            loader = DirectoryLoader(tenant_id=self.settings.tenant_id, settings=self.settings)
            async for batch in loader.iter_documents(root, window=window):
                report.timings_ms["load"] = report.timings_ms.get("load", 0.0) + _elapsed_ms(
                    started
                )
                started = time.perf_counter()
                if batch:
                    yield batch
            return
        documents = await self._resolve_target(target)
        report.timings_ms["load"] = _elapsed_ms(started)
        if documents:
            yield documents

    async def _collect(self, target: Any, report: IngestReport) -> list[Document]:
        """Load everything at once. Retained for callers that want one list."""
        started = time.perf_counter()
        documents = await self._resolve_target(target)
        report.timings_ms["load"] = _elapsed_ms(started)
        return documents

    async def _resolve_target(self, target: Any) -> list[Document]:
        if isinstance(target, Document):
            return [target]
        if isinstance(target, (str, Path)):
            return await load_source(
                target, tenant_id=self.settings.tenant_id, settings=self.settings
            )
        if isinstance(target, Iterable):
            items = list(target)
            documents = [item for item in items if isinstance(item, Document)]
            sources = [item for item in items if not isinstance(item, Document)]
            if sources:
                documents.extend(
                    await load_source(
                        sources, tenant_id=self.settings.tenant_id, settings=self.settings
                    )
                )
            return documents
        raise TypeError(f"cannot ingest {type(target).__name__}: expected documents or a source")

    async def _validate(
        self, documents: Sequence[Document], report: IngestReport
    ) -> list[Document]:
        """Validate and de-duplicate, in a thread.

        ``validate_batch`` normalizes control characters and runs a binary-content
        heuristic over every document; across a 10k-document batch that is enough
        pure CPU to stall the loop, and it is the last stage before real I/O
        resumes.
        """
        started = time.perf_counter()
        validation = await asyncio.to_thread(self.validator.validate_batch, list(documents))
        accepted = validation.accepted

        # Anything neither accepted nor rejected was an identical-checksum
        # duplicate of a document already in this batch: two ids, one payload,
        # which would take two result slots and double-assert the same fact.
        report.documents_rejected = len(validation.rejected)
        report.documents_duplicate = len(documents) - len(accepted) - len(validation.rejected)
        report.rejected.extend(validation.rejected)
        report.warnings.extend(validation.warnings)

        for doc in accepted:
            # A hand-built Document may arrive without one, and the skip, the
            # dedupe report and every future re-ingest all key off it.
            if not doc.checksum:
                doc.checksum = content_hash(doc.content)

        report.timings_ms["validate"] = _elapsed_ms(started)
        log.info(
            "documents_validated",
            accepted=len(accepted),
            rejected=report.documents_rejected,
            duplicates=report.documents_duplicate,
        )
        return accepted

    # ------------------------------------------------------------------
    # b. checksum skip
    # ------------------------------------------------------------------
    async def _select_changed(
        self, documents: Sequence[Document], report: IngestReport
    ) -> tuple[list[Document], set[str]]:
        """Split into work and no-ops. Returns ``(to_index, ids_that_changed)``."""
        if not self.config.skip_unchanged or self.relational is None:
            return list(documents), set()

        started = time.perf_counter()
        known = await self._existing_checksums([doc.id for doc in documents])
        todo: list[Document] = []
        changed: set[str] = set()
        for doc in documents:
            prior = known.get(doc.id)
            if prior is None:
                todo.append(doc)
            elif prior == doc.checksum:
                report.documents_skipped += 1
            else:
                changed.add(doc.id)
                todo.append(doc)
        report.timings_ms["checksum"] = _elapsed_ms(started)
        log.info(
            "checksum_skip",
            considered=len(documents),
            unchanged=report.documents_skipped,
            changed=len(changed),
            new=len(todo) - len(changed),
        )
        return todo, changed

    async def _existing_checksums(self, ids: Sequence[str]) -> dict[str, str]:
        """Stored checksums for these ids, in as few round trips as possible.

        One statement per window of ids rather than one per document. The window is
        bounded by ``postgres.max_sql_rows`` because ``execute_readonly`` caps the
        rows it will return — a cap that exists to fence in generated Text-to-SQL
        and is merely an inconvenience here. Even so the probe is ~200x fewer round
        trips than the per-document version.

        No tenant predicate is needed: ``document_id`` is salted with the tenant, so
        an id cannot exist under two tenants.

        **A document only counts as ingested once it has chunks.** The document row
        is written before its chunks because the chunks table has a foreign key to
        it, which means a run that dies between the two — a vector-store error, a
        killed process — leaves the row behind with nothing attached. Matching on
        the row alone then reports every document as unchanged and the retry
        indexes nothing, exiting *successfully* with an empty index. That is the
        worst available outcome: the corpus is missing, the ingest says it is fine,
        and every later query abstains for reasons nobody can trace back to here.
        The join makes the chunk rows the commit marker the schema does not have.
        """
        if self.relational is None or not ids:
            return {}
        table = documents_table(self.settings.postgres).as_string(None)
        chunks = chunks_table(self.settings.postgres).as_string(None)
        # Not injection: both identifiers are rendered by psycopg's Identifier from
        # settings, and the only caller-supplied value (the id list) is bound.
        statement = (
            f"SELECT d.id, d.checksum FROM {table} d "  # noqa: S608
            f"WHERE d.id = ANY(%s) "
            f"AND EXISTS (SELECT 1 FROM {chunks} c WHERE c.document_id = d.id)"
        )
        page = max(1, min(self.config.batch_size, self.settings.postgres.max_sql_rows))
        known: dict[str, str] = {}
        for window in _windows(list(ids), page):
            rows = await self.relational.execute_readonly(
                statement, [list(window)], limit=len(window)
            )
            known.update(
                {str(row["id"]): str(row["checksum"]) for row in rows if row.get("checksum")}
            )
        return known

    # ------------------------------------------------------------------
    # Preparation: strategy, components, stores
    # ------------------------------------------------------------------
    async def _prepare(self, report: IngestReport | None = None) -> ChunkingStrategy:
        """Resolve the strategy once, then build everything it determines."""
        if self._strategy is not None:
            return self._strategy

        started = time.perf_counter()
        chunker = self._late_chunker()
        strategy = await resolve_strategy(self.config.chunking_strategy, chunker, self.settings)

        if strategy is ChunkingStrategy.CONTEXTUAL:
            self._enricher = self._build_enricher()
            if self._enricher is None:
                # The bottom of the ADR-0002 ladder. Degrading loudly beats
                # indexing a whole corpus with a strategy nobody chose.
                log.warning(
                    "contextual_enricher_unavailable",
                    fallback=ChunkingStrategy.EARLY.value,
                    hint="ragorc.index.contextual is not present in this build",
                )
                strategy = ChunkingStrategy.EARLY

        # Under LATE the vectors are pooled by the late-chunking backend, so *it*
        # is the embedder the query side must use and its width is the collection's
        # width. Under EARLY/CONTEXTUAL the configured dense embedder owns both.
        query_side: Any = chunker if strategy is ChunkingStrategy.LATE else self._dense()
        dimension = await self._pin_dimension(query_side, measure=strategy is ChunkingStrategy.LATE)

        self._stages = self._build_stages(report)
        await self._ensure_stores(query_side, dimension)

        self._strategy = strategy
        log.info(
            "ingest_prepared",
            strategy=strategy.value,
            splitter=getattr(self._splitter_for(), "name", "unknown"),
            dense_dimension=dimension,
            sparse=self.sparse_embedder is not None,
            colbert=self.late_embedder is not None,
            stages=[plugin.label for plugin, _ in self._stages],
            prepare_ms=round(_elapsed_ms(started), 1),
        )
        return strategy

    async def _pin_dimension(self, embedder: Any, *, measure: bool) -> int:
        """Establish the true vector width with one real forward pass.

        ``measure`` forces the probe for late chunking: the declared dimension
        describes ``embedding.dense_model``, but the pooled vectors come from the
        late-chunking backend, which may be a different model of a different width.
        Trusting the declaration there produces vectors the collection rejects
        thousands of writes later.
        """
        if not measure:
            warmup = getattr(embedder, "warmup", None)
            if callable(warmup):
                await warmup()
            declared = int(getattr(embedder, "dimension", 0) or 0)
            if declared:
                return declared
        vector = await embedder.embed_query(_DIMENSION_PROBE)
        dimension = int(np.asarray(vector).shape[-1])
        if dimension and getattr(embedder, "dimension", 0) != dimension:
            with contextlib.suppress(AttributeError):
                embedder.dimension = dimension
        return dimension

    async def _ensure_stores(self, query_side: Any, dimension: int) -> None:
        """Create the stores, then their schema. Idempotent, once per pipeline.

        ``postgres.vector_dimension`` is aligned with the measured width for the
        same reason :meth:`Settings.model_post_init` aligns it with the configured
        one: pgvector's column type is fixed at creation and a mismatch surfaces as
        an opaque insert error long after the run that caused it.
        """
        if dimension and self.settings.postgres.vector_dimension != dimension:
            log.info(
                "pgvector_dimension_aligned",
                previous=self.settings.postgres.vector_dimension,
                dimension=dimension,
            )
            self.settings.postgres.vector_dimension = dimension

        if self.vector is None:
            from ragorc.stores.qdrant.store import QdrantStore

            self.vector = QdrantStore(
                self.settings,
                dense_embedder=query_side,
                sparse_embedder=self.sparse_embedder,
                late_embedder=self.late_embedder,
            )
            self._owns_stores = True
        if self.relational is None:
            from ragorc.stores.postgres.store import PostgresStore

            self.relational = PostgresStore(self.settings)
            self._owns_stores = True

        # Concurrently: two independent databases, and a first run pays DDL on
        # both. A failure here aborts before anything has been embedded, which is
        # the cheapest possible moment to discover an unreachable store.
        await bounded_gather(
            [self.vector.ensure_collection(), self.relational.ensure_schema()], limit=2
        )

    # -- component construction -------------------------------------------
    def _cache(self) -> Any:
        """Embedding cache shared by every embedder this pipeline builds.

        Worth wiring even though the checksum skip already avoids unchanged
        documents: a partially edited document re-embeds only the chunks whose text
        actually moved, and boilerplate shared across documents is embedded once.
        """
        if self._embedding_cache is None:
            from ragorc.cache.tiered import build_cache
            from ragorc.embed.cache import EmbeddingCache

            self._embedding_cache = EmbeddingCache(build_cache(self.settings.cache), self.settings)
        return self._embedding_cache

    def _provider_class(self, kind: str, provider: str) -> type:
        """Resolve an embedder class, importing its provider module first.

        The registry only knows a class after the module defining it has been
        imported, and importing every provider eagerly would require every hosted
        SDK to be installed. The module name is derived from the provider name, so
        a missing extra fails with that provider's own ``ImportError``.
        """
        importlib.import_module(f"ragorc.embed.{provider}_provider")
        return resolve(kind, provider)

    def _dense(self) -> DenseEmbedder:
        if self.dense_embedder is None:
            cls = self._provider_class("dense_embedder", self.settings.embedding.provider)
            self.dense_embedder = cls(cache=self._cache(), settings=self.settings)
        return self.dense_embedder

    def _sparse(self) -> SparseEmbedder | None:
        """Sparse vectors, unless hybrid search is off.

        Built here rather than left to the store because the store's job is to
        write what it is given; a chunk that reaches Qdrant without the sparse
        vector its collection declares is not an error, it is a chunk that lexical
        search will never find.
        """
        if self.sparse_embedder is None and self.settings.retrieval.use_sparse:
            cls = self._provider_class("sparse_embedder", "fastembed")
            self.sparse_embedder = cls(cache=self._cache(), settings=self.settings)
        return self.sparse_embedder

    def _colbert(self) -> LateInteractionEmbedder | None:
        if self.late_embedder is None and self.settings.embedding.enable_late_interaction:
            cls = self._provider_class("late_interaction_embedder", "fastembed")
            self.late_embedder = cls(cache=self._cache(), settings=self.settings)
        return self.late_embedder

    def _late_chunker(self) -> LateChunkingEmbedder:
        """The late-chunking pooler, wired to a token-capable embedder if we have
        one. It builds its own backend otherwise (transformers, else FastEmbed)."""
        if self.late_chunker is None:
            self.late_chunker = LateChunkingEmbedder(
                token_embedder=self.late_embedder, settings=self.settings
            )
        return self.late_chunker

    def _splitter_for(self) -> Splitter:
        if self.splitter is None:
            self.splitter = build_splitter(embedder=self._dense(), settings=self.settings)
        return self.splitter

    def _build_enricher(self) -> Any | None:
        factory = _load_plugin(_CONTEXTUAL)
        if factory is None:
            return None
        try:
            return _construct(factory, llm=self.llm, embedder=self._dense(), settings=self.settings)
        except Exception as exc:  # a stage that cannot be built is a stage we skip
            log.warning(
                "contextual_enricher_unbuildable",
                error=str(exc)[:200],
                error_type=type(exc).__name__,
                hint="pass llm=... to IngestPipeline",
            )
            return None

    def _build_stages(self, report: IngestReport | None = None) -> list[tuple[_Plugin, Any]]:
        """Locate the optional stages whose settings flags are on.

        A stage the caller asked for and did not get is recorded on the report,
        not just logged. Both failures below were invisible for exactly that
        reason: an operator who set ``indexing.summary_index_enabled`` saw a
        successful ingest with ``llm_calls=0`` and no indication that the stage
        they configured had been dropped at build time.
        """
        wanted = [plugin for plugin in _OPTIONAL_STAGES if self._stage_enabled(plugin.label)]
        built: list[tuple[_Plugin, Any]] = []
        for plugin in wanted:
            factory = _load_plugin(plugin)
            if factory is None:
                # An extra names an install; no extra means the factory is simply
                # absent from this build, which is a packaging bug rather than
                # something the operator can fix by installing anything.
                hint = (
                    f"pip install 'ragorc[{plugin.extra}]'"
                    if plugin.extra
                    else f"none of {plugin.factories} exists in {plugin.modules[0]}"
                )
                log.warning("index_stage_unavailable", stage=plugin.label, hint=hint)
                if report is not None:
                    report.warnings.append(
                        f"{plugin.label} stage is enabled but unavailable: {hint}"
                    )
                continue
            try:
                built.append(
                    (
                        plugin,
                        _construct(
                            factory,
                            llm=self.llm,
                            embedder=self._dense(),
                            settings=self.settings,
                        ),
                    )
                )
            except Exception as exc:
                log.warning(
                    "index_stage_unbuildable",
                    stage=plugin.label,
                    error=str(exc)[:200],
                    error_type=type(exc).__name__,
                )
                if report is not None:
                    report.warnings.append(
                        f"{plugin.label} stage is enabled but could not be built: {exc}"
                    )
        if self.settings.graph.enabled:
            # Not a failure — a different shape of work. Saying so beats an
            # ingest that reports success while the graph the operator asked for
            # does not exist.
            log.info("graph_requires_second_pass", hint=_GRAPH_CORPUS_HINT)
            if report is not None:
                report.warnings.append(_GRAPH_CORPUS_HINT)
        return built

    def _stage_enabled(self, label: str) -> bool:
        if label == "multirep":
            return (
                self.config.parent_document_enabled
                or self.config.summary_index_enabled
                or self.config.dense_x_enabled
            )
        if label == "raptor":
            return self.config.raptor_enabled
        return False

    # ------------------------------------------------------------------
    # The streaming run
    # ------------------------------------------------------------------
    async def _run(
        self,
        documents: Sequence[Document],
        changed: set[str],
        strategy: ChunkingStrategy,
        report: IngestReport,
    ) -> None:
        """Process documents in windows and stream chunks to the stores.

        Two nested bounds, each doing a different job. The *window* bounds how many
        documents are being split and embedded at once, which bounds peak memory
        and provider concurrency. The *batch* bounds how many chunks are held
        before a write, which bounds the size of the work a crash discards.
        """
        assert self.vector is not None  # noqa: S101 - _prepare built it or raised
        window = max(1, self.config.max_concurrent_documents)
        batch = max(1, self.config.batch_size)
        pending: list[Chunk] = []

        async with contextlib.AsyncExitStack() as stack:
            if len(documents) >= BULK_LOAD_MIN_DOCUMENTS:
                bulk = getattr(self.vector, "bulk_load", None)
                if callable(bulk):
                    await stack.enter_async_context(bulk())

            for group in _windows(list(documents), window):
                ready, chunks = await self._process_window(group, strategy, report)
                if not ready:
                    continue

                # The stale purge happens now — after the replacement chunks exist
                # — so a document whose extraction just started failing keeps the
                # vectors it already had instead of disappearing from the index.
                stale = [doc for doc in ready if doc.id in changed]
                if stale:
                    await self._purge(stale, report)
                await self._write_documents(ready, report)

                pending.extend(chunks)
                while len(pending) >= batch:
                    await self._write_chunks(pending[:batch], report)
                    del pending[:batch]

            if pending:
                await self._write_chunks(pending, report)

    async def _process_window(
        self,
        group: Sequence[Document],
        strategy: ChunkingStrategy,
        report: IngestReport,
    ) -> tuple[list[Document], list[Chunk]]:
        outcomes = await bounded_gather(
            (self._process_document(doc, strategy, report) for doc in group),
            limit=max(1, self.config.max_concurrent_documents),
            return_exceptions=True,
        )
        ready: list[Document] = []
        chunks: list[Chunk] = []
        for doc, outcome in zip(group, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, _ABORTING):
                    raise outcome
                report.documents_failed += 1
                report.failed.append((doc.id, f"{type(outcome).__name__}: {outcome}"))
                log.warning(
                    "document_ingest_failed",
                    document_id=doc.id,
                    source=doc.source,
                    error=str(outcome)[:200],
                    error_type=type(outcome).__name__,
                )
                continue
            if not outcome:
                # Validated as text but produced nothing indexable: a file of
                # headings, a page of table borders, or — the common case — a
                # document shorter than min_chunk_size, whose single chunk
                # `validate_chunks` drops. Not a failure, but not a document
                # either, so it is not counted as indexed.
                #
                # It *is* counted here, and at warning level. Before, this branch
                # incremented nothing and logged at info without a reason, so the
                # run reported success with the document silently absent from the
                # index and `documents_in` larger than every other counter
                # combined.
                report.documents_empty += 1
                report.warnings.append(
                    f"{doc.id} produced no indexable chunks "
                    f"(min_chunk_size={self.config.min_chunk_size}); not indexed"
                )
                log.warning(
                    "document_produced_no_chunks",
                    document_id=doc.id,
                    source=doc.source,
                    chars=len(doc.content),
                    min_chunk_size=self.config.min_chunk_size,
                )
                continue
            ready.append(doc)
            chunks.extend(outcome)
            report.chunks_created += len(outcome)
            report.documents_indexed += 1
        return ready, chunks

    # ------------------------------------------------------------------
    # c-f. per-document: split, embed, enrich
    # ------------------------------------------------------------------
    async def _process_document(
        self, document: Document, strategy: ChunkingStrategy, report: IngestReport
    ) -> list[Chunk]:
        started = time.perf_counter()
        chunks = await self._splitter_for().split(document)
        chunks = self.validator.validate_chunks(chunks)
        if self.config.dedupe_chunks:
            chunks = _dedupe_exact(chunks)
        report.timings_ms["split"] = report.timings_ms.get("split", 0.0) + _elapsed_ms(started)
        if not chunks:
            return []

        await self._embed_dense(document, chunks, strategy, report)
        await self._add_sparse(chunks, report)
        await self._add_colbert(chunks, report)
        await self._enrich(document, chunks, report)
        log.debug(
            "document_indexed",
            document_id=document.id,
            chunks=len(chunks),
            strategy=strategy.value,
            ms=round(_elapsed_ms(started), 1),
        )
        return chunks

    async def _embed_dense(
        self,
        document: Document,
        chunks: list[Chunk],
        strategy: ChunkingStrategy,
        report: IngestReport,
    ) -> None:
        """Fill ``chunk.dense`` by the resolved strategy (ADR-0002)."""
        started = time.perf_counter()
        if strategy is ChunkingStrategy.LATE:
            # The exact splitter offsets, untouched. The pooler converts them to
            # token spans in the document's single forward pass, so an off-by-one
            # here pools the wrong tokens with no error and no log line.
            spans = [(chunk.start_char, chunk.end_char) for chunk in chunks]
            vectors = await self._late_chunker().embed_chunks(document.content, spans)
        else:
            if strategy is ChunkingStrategy.CONTEXTUAL:
                await self._contextualize(document, chunks, report)
            # ``embed_text`` includes the contextual prefix when one was written
            # and is exactly the content otherwise.
            vectors = await self._dense().embed_documents([chunk.embed_text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.dense = vector
        report.timings_ms["embed_dense"] = report.timings_ms.get("embed_dense", 0.0) + _elapsed_ms(
            started
        )

    async def _contextualize(
        self, document: Document, chunks: list[Chunk], report: IngestReport
    ) -> None:
        """Have the enricher write ``contextual_prefix`` on each chunk.

        The prefix is embedded but never shown to the generator (see
        :attr:`Chunk.contextual_prefix`), so a failure here degrades the vectors
        rather than corrupting the answer — which is why it is a warning and not an
        abort. The chunks are still embedded, just without the situating sentence.
        """
        if self._enricher is None:
            return
        started = time.perf_counter()
        try:
            enriched, usage = await _invoke(
                self._enricher, _CONTEXTUAL, document=document, chunks=chunks
            )
        except Exception as exc:
            log.warning(
                "contextual_enrichment_failed",
                document_id=document.id,
                error=str(exc)[:200],
                error_type=type(exc).__name__,
            )
            self._enricher = None  # one broken enricher, not one per document
            report.warnings.append(f"contextual enrichment disabled: {exc}")
            return
        _replace(chunks, enriched)
        self._charge(report, usage, "contextual")
        report.timings_ms["contextual"] = report.timings_ms.get("contextual", 0.0) + _elapsed_ms(
            started
        )

    async def _add_sparse(self, chunks: list[Chunk], report: IngestReport) -> None:
        embedder = self._sparse()
        if embedder is None:
            return
        started = time.perf_counter()
        # ``embed_text`` again, not ``content``: with contextual retrieval the
        # prefix names the entity the chunk only refers to, and BM25 over the
        # contextualized text is where Anthropic measured the larger part of the
        # improvement.
        vectors = await embedder.embed_documents([chunk.embed_text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            chunk.sparse = vector
        report.timings_ms["embed_sparse"] = report.timings_ms.get(
            "embed_sparse", 0.0
        ) + _elapsed_ms(started)

    async def _add_colbert(self, chunks: list[Chunk], report: IngestReport) -> None:
        """Late-interaction matrices, only when the collection declares them.

        A ColBERT matrix is ~100x a dense vector on disk, so this is off by
        default; when it is on it is the multivector Qdrant rescores the fused
        candidates with, server-side.
        """
        embedder = self._colbert()
        if embedder is None:
            return
        started = time.perf_counter()
        matrices = await embedder.embed_documents([chunk.content for chunk in chunks])
        for chunk, matrix in zip(chunks, matrices, strict=True):
            chunk.multi = matrix
        report.timings_ms["embed_colbert"] = report.timings_ms.get(
            "embed_colbert", 0.0
        ) + _elapsed_ms(started)

    async def _enrich(self, document: Document, chunks: list[Chunk], report: IngestReport) -> None:
        """Run the optional stages in order, each behind its settings flag.

        A stage runs over one document's chunks because that is all the streaming
        pipeline is holding — deliberately, since materializing a 4M-chunk corpus
        to cluster it is exactly what the backpressure exists to prevent. For a
        corpus-wide RAPTOR tree or community graph, run that stage as a second pass
        over the store after ingest; per-document trees are what a streaming ingest
        can offer.

        A stage that fails is disabled for the remainder of the run. The leaf
        chunks are already vectorized and writable at this point, and losing an
        enrichment is recoverable where losing the leaves is not.
        """
        if not self._stages:
            return
        survivors: list[tuple[_Plugin, Any]] = []
        for plugin, stage in self._stages:
            started = time.perf_counter()
            try:
                produced, usage = await _invoke(
                    stage,
                    plugin,
                    document=document,
                    chunks=chunks,
                    vector_store=self.vector,
                    relational_store=self.relational,
                )
            except Exception as exc:
                log.warning(
                    "index_stage_failed",
                    stage=plugin.label,
                    document_id=document.id,
                    error=str(exc)[:200],
                    error_type=type(exc).__name__,
                    action="stage disabled for the rest of this run",
                )
                report.warnings.append(f"{plugin.label} stage disabled: {exc}")
                continue
            survivors.append((plugin, stage))
            _replace(chunks, produced)
            self._charge(report, usage, plugin.label)
            report.timings_ms[plugin.label] = report.timings_ms.get(
                plugin.label, 0.0
            ) + _elapsed_ms(started)
        self._stages = survivors

    def _charge(self, report: IngestReport, usage: Usage, stage: str) -> None:
        """Aggregate a stage's bill into the report and the ambient ledger."""
        if usage.calls == 0 and usage.cost_usd == 0.0:
            return
        report.usage = report.usage + usage
        ledger = current_ledger()
        if ledger is not None:
            ledger.record(usage, stage=f"ingest.{stage}")

    # ------------------------------------------------------------------
    # g. writes
    # ------------------------------------------------------------------
    async def _write_documents(self, documents: Sequence[Document], report: IngestReport) -> None:
        """Document rows first: ``chunks.document_id`` is a foreign key."""
        if self.relational is None:
            return
        started = time.perf_counter()
        await self.relational.upsert_documents(documents)
        report.timings_ms["write_documents"] = report.timings_ms.get(
            "write_documents", 0.0
        ) + _elapsed_ms(started)

    async def _write_chunks(self, chunks: Sequence[Chunk], report: IngestReport) -> None:
        """Write one batch to both stores concurrently.

        Concurrently because the two writes are independent — Qdrant takes the
        vectors, Postgres takes the rows plus the pgvector copy — and serializing
        them would make ingest throughput the sum of two latencies instead of the
        max.

        A failure propagates and ends the run. There is no partial-write cleanup to
        do: ids are content-derived, so re-running overwrites the same points and
        rows rather than duplicating them.
        """
        if not chunks or self.vector is None:
            return
        started = time.perf_counter()
        writes = [self.vector.upsert(chunks)]
        if self.relational is not None:
            writes.append(self.relational.upsert_chunks(chunks))
        await bounded_gather(writes, limit=len(writes))
        report.vectors_written += _count_vectors(chunks)
        report.timings_ms["write_chunks"] = report.timings_ms.get(
            "write_chunks", 0.0
        ) + _elapsed_ms(started)
        log.debug("chunk_batch_written", chunks=len(chunks), ms=round(_elapsed_ms(started), 1))

    async def _purge(self, documents: Sequence[Document], report: IngestReport) -> None:
        """Delete a changed document's previous chunks before writing the new ones.

        Mandatory, not hygiene. ``chunk_id`` folds the content in, so an edited
        document's chunks get *new* ids and an upsert leaves the old ones in place —
        still indexed, still retrievable, still citable, and now wrong. Deleting by
        ``document_id`` filter removes a whole batch's worth in one request per
        tenant instead of one per chunk.
        """
        started = time.perf_counter()
        by_tenant: dict[str | None, list[str]] = {}
        for doc in documents:
            by_tenant.setdefault(doc.tenant_id, []).append(doc.id)

        if self.vector is not None:
            for tenant, ids in by_tenant.items():
                await self.vector.delete(filters={"document_id": ids}, tenant_id=tenant)
        if self.relational is not None:
            # Per document: the relational delete reports the row count and takes
            # the chunks and the document row in one transaction. Bounded because
            # a large re-ingest can change thousands of documents at once.
            await bounded_gather(
                (self.relational.delete_document(doc.id) for doc in documents),
                limit=max(1, self.config.max_concurrent_documents),
            )
        report.timings_ms["purge"] = report.timings_ms.get("purge", 0.0) + _elapsed_ms(started)
        log.info("stale_chunks_purged", documents=len(documents))


# ---------------------------------------------------------------------------
# Chunk-list plumbing
# ---------------------------------------------------------------------------
def _replace(chunks: list[Chunk], produced: Sequence[Chunk]) -> None:
    """Swap a stage's output into the caller's list in place.

    In place because the list is the streaming unit: the caller holds it, extends a
    pending batch from it and writes it. Returning a new list from every stage
    would double peak memory for the window, and rebinding would silently drop the
    chunks a stage added.
    """
    if produced is chunks:
        return
    chunks[:] = list(produced)


def _dedupe_exact(chunks: list[Chunk]) -> list[Chunk]:
    """Drop byte-identical chunks within one document, keeping the first.

    Exact duplicates only. ``indexing.dedupe_threshold`` describes a *similarity*
    pass, which belongs at query time (:mod:`ragorc.retrieve.noise`) for two
    reasons: it needs the vectors this stage has not computed yet, and choosing
    which of two near-identical chunks to keep depends on the query that retrieved
    them.
    """
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        digest = content_hash(chunk.content, size=8)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(chunk)
    return out
