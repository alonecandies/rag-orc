"""Bidirectional LangChain interop — the escape hatch that makes ADR-0001 free.

The core of this library deliberately depends on **LangGraph and nothing else from
the LangChain ecosystem** (ADR-0001). That decision is about capability, not
taste: ``ChatOpenAI`` cannot express OpenRouter provider routing, provider-reported
per-call cost, a token bucket and a structured-output repair loop;
``QdrantVectorStore`` cannot express multivectors, quantization search params or
server-side prefetch fusion (ADR-0003); LangChain's splitters and retrievers are
Python loops where this library is vectorized numpy, and they have no hook for
late chunking (ADR-0002). Those are the performance features, so wrapping a wrapper
to reach them would have cost the entire point.

This module is the other half of that decision. A team already invested in LCEL
or LangGraph should be able to adopt one piece of this library — the hybrid
retriever, the graph search, the whole pipeline — **without rewriting their app,
and without this library growing a LangChain dependency in its core.** So the
interop lives here, at the edge, behind the ``[langchain]`` extra, and every
``langchain_core`` import happens inside a function body:

* :class:`RagOrcRetriever` presents any of our retrievers as a LangChain
  ``BaseRetriever``, so it drops into an existing LCEL chain or LangGraph app.
* :func:`from_langchain_retriever` runs the conversion the other way, so a
  retriever a team already owns can be one leg of our
  :class:`~ragorc.retrieve.ensemble.EnsembleRetriever` and be fused with ours.
* :func:`as_runnable` wraps any of our components so it can sit in a pipe.
* :func:`to_langchain_documents` / :func:`from_langchain_documents` are the
  underlying conversion, exposed because the interesting part is *what survives*
  the round trip.

Three details that decide whether that round trip is lossy
----------------------------------------------------------
**Identity.** ``Chunk.id`` is content-derived and identical in Qdrant, Postgres and
Neo4j (:mod:`ragorc.core.ids`), so it is the join key for everything downstream —
citations, dedupe, cache lookups. It goes into ``Document.id`` *and* is mirrored
into ``metadata["chunk_id"]``, because some LangChain components construct new
``Document`` objects from ``page_content`` and ``metadata`` alone and silently drop
the top-level ``id``.

**Score.** LangChain's ``Document`` has no score field, so ours travels in
metadata. Coming back the other way most retrievers supply no score at all: the
only information present is the *order*, so we synthesize a monotone descending
score from the rank and record that we did in ``explain["score"]``. Inventing a
similarity would be worse than admitting we have a rank — the fusion layer treats
rank-only inputs correctly (RRF reads position, not magnitude) and a fabricated
0.9 would make weighted fusion trust a number that carries no information.

**Vectors do not cross.** ``metadata`` has to survive JSON serialization in most
LangChain sinks, and a 384-float array does not belong there. Anything needing the
vector should stay on our side of the boundary; what crosses is text, provenance
and one number.

Async only, like the rest of the library. ``BaseRetriever`` declares a sync
``_get_relevant_documents`` as abstract so it is implemented — to raise. Running
our coroutines from a fresh event loop would be worse than an error: the store
clients cache connections keyed on the *running* loop, so a sync call would hand
out clients bound to a loop that is about to be closed, and that failure surfaces
as an intermittent hang rather than an exception.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import Sequence
from typing import Any

import structlog

from ragorc.core.errors import ConfigError
from ragorc.core.ids import chunk_id as make_chunk_id
from ragorc.core.models import (
    Chunk,
    Modality,
    Query,
    RetrievalSource,
    ScoredChunk,
)
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import new_request_context, timed

log = structlog.get_logger(__name__)

__all__ = [
    "LangChainRetriever",
    # Built at first access by ``__getattr__`` below, because the class subclasses
    # ``langchain_core.retrievers.BaseRetriever`` and so cannot exist before that
    # import has happened. Advertised here so tooling and ``import *`` see it.
    "RagOrcRetriever",  # noqa: F822
    "as_runnable",
    "from_langchain_documents",
    "from_langchain_retriever",
    "to_langchain_documents",
    "to_langchain_retriever",
]

_EXTRA_HINT = "pip install 'ragorc[langchain]'"

_SCORE_KEYS: tuple[str, ...] = ("score", "relevance_score", "_score", "similarity")
"""Metadata keys under which LangChain integrations publish a higher-is-better
score. Checked in order; the first present one wins."""

_DISTANCE_KEYS: tuple[str, ...] = ("distance", "_distance", "l2_distance")
"""Lower-is-better keys. Converted, never used raw — every score in this library
is higher-is-better and mixing the two conventions inverts a ranking silently."""


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _document_class() -> type:
    """``langchain_core.documents.Document``, imported on demand."""
    try:
        from langchain_core.documents import Document as LCDocument
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            f"LangChain interop requires langchain-core. Install it with: {_EXTRA_HINT}"
        ) from exc
    return LCDocument


def _base_retriever_class() -> type:
    """``langchain_core.retrievers.BaseRetriever``, imported on demand."""
    try:
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            f"RagOrcRetriever requires langchain-core. Install it with: {_EXTRA_HINT}"
        ) from exc
    return BaseRetriever


def _runnable_lambda_class() -> type:
    try:
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            f"as_runnable requires langchain-core. Install it with: {_EXTRA_HINT}"
        ) from exc
    return RunnableLambda


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def to_langchain_documents(items: Sequence[ScoredChunk | Chunk]) -> list[Any]:
    """Convert our chunks to LangChain ``Document`` objects.

    Accepts :class:`~ragorc.core.models.ScoredChunk` (the retrieval output) or bare
    :class:`~ragorc.core.models.Chunk` (the ingest unit); a bare chunk has no score,
    so ``metadata["score"]`` is omitted rather than defaulted to a number the caller
    would then treat as a similarity.
    """
    document_cls = _document_class()
    out: list[Any] = []
    for position, item in enumerate(items):
        scored = item if isinstance(item, ScoredChunk) else None
        # Read through ``scored`` rather than ``item``: it is the same object, and
        # it is the one the ``is not None`` test has narrowed.
        chunk = scored.chunk if scored is not None else item
        assert isinstance(chunk, Chunk)

        metadata: dict[str, Any] = dict(chunk.metadata)
        # Structural fields last so a metadata key of the same name cannot shadow
        # the provenance the citation layer resolves against.
        metadata.update(
            {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "level": chunk.level,
                "modality": chunk.modality.value,
            }
        )
        if chunk.parent_id:
            metadata["parent_id"] = chunk.parent_id
        if chunk.tenant_id:
            metadata["tenant_id"] = chunk.tenant_id
        if chunk.token_count is not None:
            metadata["token_count"] = chunk.token_count
        if scored is not None:
            metadata["score"] = float(scored.score)
            metadata["rank"] = int(scored.rank or position)
            metadata["retrieval_source"] = scored.source.value
            if scored.component_scores:
                # Per-retriever contributions are the only way to answer "why did
                # this rank third?" after fusion has flattened the scores.
                metadata["component_scores"] = {
                    k: float(v) for k, v in scored.component_scores.items()
                }
        out.append(document_cls(id=chunk.id, page_content=chunk.content, metadata=metadata))
    return out


def from_langchain_documents(
    documents: Sequence[Any],
    *,
    source: RetrievalSource = RetrievalSource.DENSE,
    document_id: str = "langchain",
) -> list[ScoredChunk]:
    """Convert LangChain ``Document`` objects into scored chunks.

    ``source`` defaults to ``DENSE`` because a LangChain retriever is a vector
    store far more often than anything else, and there is no enum member for
    "someone else's retriever" — :mod:`ragorc.core.models` is the contract, not a
    scratchpad. Pass the honest value when you know it (``RetrievalSource.WEB`` for
    a web retriever, ``BM25`` for a lexical one) because fusion weights and the
    citation footer both read it.

    Ids: a ``Document`` that carries one keeps it, so a round trip through LangChain
    preserves the join key. One that does not gets a deterministic id derived from
    its content, which is what makes a LangChain leg's results dedupe correctly
    against our own instead of appearing as N distinct near-duplicates.
    """
    scores = _resolve_scores(documents)
    out: list[ScoredChunk] = []
    for rank, (document, score) in enumerate(zip(documents, scores, strict=True)):
        content = str(getattr(document, "page_content", "") or "")
        raw_meta = dict(getattr(document, "metadata", None) or {})
        doc_id = str(raw_meta.pop("document_id", None) or raw_meta.get("source") or document_id)
        index = int(raw_meta.pop("index", rank) or 0)
        identifier = str(
            getattr(document, "id", None) or raw_meta.pop("chunk_id", "") or ""
        ) or make_chunk_id(doc_id, index, content)
        raw_meta.pop("chunk_id", None)
        explain: dict[str, Any] = {"adapter": "langchain"}
        if not _has_score(raw_meta):
            explain["score"] = "synthesized from rank: the retriever supplied none"

        component = raw_meta.pop("component_scores", None)
        out.append(
            ScoredChunk(
                chunk=Chunk(
                    id=identifier,
                    content=content,
                    document_id=doc_id,
                    index=index,
                    start_char=int(raw_meta.pop("start_char", 0) or 0),
                    end_char=int(raw_meta.pop("end_char", len(content)) or len(content)),
                    level=int(raw_meta.pop("level", 0) or 0),
                    modality=_modality(raw_meta.pop("modality", None)),
                    parent_id=raw_meta.pop("parent_id", None),
                    tenant_id=raw_meta.pop("tenant_id", None),
                    metadata=raw_meta,
                ),
                score=score,
                source=source,
                rank=rank,
                component_scores=(
                    {k: float(v) for k, v in component.items()}
                    if isinstance(component, dict)
                    else {source.value: score}
                ),
                explain=explain,
            )
        )
    return out


def _modality(raw: Any) -> Modality:
    try:
        return Modality(raw) if raw else Modality.TEXT
    except ValueError:
        return Modality.TEXT


def _has_score(metadata: dict[str, Any]) -> bool:
    return any(key in metadata for key in (*_SCORE_KEYS, *_DISTANCE_KEYS))


def _resolve_scores(documents: Sequence[Any]) -> list[float]:
    """One score per document, on a higher-is-better scale.

    Three cases, and the third is the common one:

    * a higher-is-better key is present — use it;
    * only a distance is present — map it with ``1/(1+d)``, which is monotone
      decreasing and bounded, so the ranking is exactly inverted and nothing is
      claimed about magnitude;
    * nothing is present — synthesize ``1 - i/n`` from the position. Order is the
      only signal the retriever gave us, and this is the shape that preserves it
      without pretending to be a similarity.
    """
    total = len(documents)
    scores: list[float] = []
    for i, document in enumerate(documents):
        metadata = getattr(document, "metadata", None) or {}
        value: float | None = None
        for key in _SCORE_KEYS:
            raw = metadata.get(key)
            if isinstance(raw, (int, float)):
                value = float(raw)
                break
        if value is None:
            for key in _DISTANCE_KEYS:
                raw = metadata.get(key)
                if isinstance(raw, (int, float)):
                    value = 1.0 / (1.0 + max(float(raw), 0.0))
                    break
        if value is None:
            value = 1.0 - (i / total if total else 0.0)
        scores.append(value)
    return scores


# ---------------------------------------------------------------------------
# ragorc -> LangChain
# ---------------------------------------------------------------------------
def _request_context(settings: Settings, label: str) -> Any:
    """Install a trace and a cost ledger for one call across the boundary.

    Not optional. Every ceiling this library enforces — cost, call count, tokens —
    lives on the request ledger, and a component invoked from an LCEL chain has no
    other opportunity to acquire one. Without it a CRAG loop reached through a pipe
    would have no spending bound at all.
    """
    cost = settings.cost
    return new_request_context(
        request_id=f"{label}-{uuid.uuid4().hex[:12]}",
        max_cost_usd=cost.max_cost_per_query_usd if cost.track_costs else None,
        max_calls=cost.max_llm_calls_per_query,
        max_tokens=cost.max_tokens_per_query,
    )


def _as_query(
    value: Any, *, top_k: int | None, filters: dict[str, Any] | None, tenant: str | None
) -> Query:
    """Accept a string, a :class:`Query`, or an LCEL-style dict.

    LCEL chains pass dicts around (``{"question": ..., "chat_history": ...}``), so a
    retriever that only accepts a bare string forces every user to insert an
    ``itemgetter``. Accepting all three shapes here is the difference between
    dropping in and being plumbed in.
    """
    if isinstance(value, Query):
        query = value
    else:
        if isinstance(value, dict):
            text = value.get("question") or value.get("query") or value.get("input") or ""
        else:
            text = value
        query = Query(text=str(text))
    if top_k:
        query.top_k = int(top_k)
    if filters:
        query.filters = {**query.filters, **filters}
    if tenant and not query.tenant_id:
        query.tenant_id = tenant
    return query


async def _retrieve_as_documents(
    retriever: Any,
    value: Any,
    *,
    top_k: int | None,
    filters: dict[str, Any] | None,
    tenant_id: str | None,
    settings: Settings,
    extra: dict[str, Any] | None,
) -> list[Any]:
    query = _as_query(value, top_k=top_k, filters=filters, tenant=tenant_id)
    kwargs: dict[str, Any] = dict(extra or {})
    if filters:
        kwargs.setdefault("filters", filters)
    if query.tenant_id:
        kwargs.setdefault("tenant_id", query.tenant_id)

    with (
        _request_context(settings, "lc-retrieve"),
        timed(
            "adapter.langchain.retrieve",
            retriever=getattr(retriever, "name", type(retriever).__name__),
        ),
    ):
        chunks = await retriever.retrieve(query, top_k=top_k or query.top_k, **kwargs)
    log.debug(
        "langchain_retriever_bridged",
        retriever=getattr(retriever, "name", type(retriever).__name__),
        results=len(chunks),
    )
    return to_langchain_documents(chunks)


@functools.lru_cache(maxsize=1)
def _retriever_class() -> type:
    """Build the ``BaseRetriever`` subclass once, on first use.

    The class *must* be defined after ``langchain_core`` is imported, because it
    subclasses one of its classes — which is exactly what a module-level import
    would have forced on every user of this library. Defining it inside a cached
    factory keeps the import lazy and still gives callers a real, stable class
    object (``lru_cache`` guarantees one identity, so ``isinstance`` works).

    Every field is annotated with a name resolvable from this module's globals:
    ``BaseRetriever`` is a pydantic model and resolves annotations against the
    defining module, not against this function's locals.
    """
    base = _base_retriever_class()

    class RagOrcRetriever(base):  # type: ignore[misc, valid-type]
        """A ragorc retriever presented as a LangChain ``BaseRetriever``.

            from ragorc.adapters.langchain import RagOrcRetriever
            from ragorc.retrieve import HybridRetriever

            lc = RagOrcRetriever(retriever=HybridRetriever(), top_k=8)
            docs = await lc.ainvoke("why is late chunking cheaper?")

        ``retriever`` is typed ``Any`` on purpose:
        :class:`~ragorc.core.protocols.Retriever` is a structural protocol, and
        making pydantic validate against it would run an ``isinstance`` check on
        every construction to learn something the first call would reveal anyway.
        """

        retriever: Any
        """Anything satisfying :class:`~ragorc.core.protocols.Retriever`."""
        top_k: int | None = None
        filters: dict[str, Any] | None = None
        tenant_id: str | None = None
        settings: Any = None
        retrieve_kwargs: dict[str, Any] | None = None
        """Extra keyword arguments forwarded verbatim to ``retrieve`` — the seam
        for retriever-specific options (``route=``, ``use_variants=``) that the
        LangChain interface has nowhere to express."""

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: Any = None, **kwargs: Any
        ) -> list[Any]:
            return await _retrieve_as_documents(
                self.retriever,
                query,
                top_k=self.top_k,
                filters=self.filters,
                tenant_id=self.tenant_id,
                settings=self.settings or get_settings(),
                extra={**(self.retrieve_kwargs or {}), **kwargs},
            )

        def _get_relevant_documents(
            self, query: str, *, run_manager: Any = None, **kwargs: Any
        ) -> list[Any]:
            """Refuse, rather than spin up an event loop.

            ``BaseRetriever`` declares this abstract, so it has to exist. It cannot
            do the work: our store clients are cached per *running* event loop, so
            driving them from a throwaway loop yields clients bound to a loop that
            is about to close — a bug whose symptom is an intermittent hang in an
            unrelated request, not an exception here.
            """
            raise NotImplementedError(
                "ragorc is async-only: use `await retriever.ainvoke(...)` "
                "(or `.abatch` / `astream`) instead of the synchronous API"
            )

    return RagOrcRetriever


def to_langchain_retriever(
    retriever: Any,
    *,
    top_k: int | None = None,
    filters: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    settings: Settings | None = None,
    **retrieve_kwargs: Any,
) -> Any:
    """Wrap one of our retrievers as a LangChain ``BaseRetriever``.

    The functional counterpart of :func:`from_langchain_retriever`, and equivalent
    to constructing :class:`RagOrcRetriever` directly.
    """
    return _retriever_class()(
        retriever=retriever,
        top_k=top_k,
        filters=filters,
        tenant_id=tenant_id,
        settings=settings,
        retrieve_kwargs=retrieve_kwargs or None,
    )


# ---------------------------------------------------------------------------
# LangChain -> ragorc
# ---------------------------------------------------------------------------
class LangChainRetriever:
    """A LangChain retriever presented as one of ours.

    This is what lets an existing retriever — someone's tuned Elasticsearch
    ``BaseRetriever``, a hosted search product's integration, a
    ``ParentDocumentRetriever`` they already trust — become one leg of
    :class:`~ragorc.retrieve.ensemble.EnsembleRetriever` and be fused with hybrid
    search, graph search and the rest. Adoption then costs one line instead of a
    migration.

    ``top_k`` is applied *after* the call, and that asymmetry is deliberate. In
    LangChain the result count is fixed at construction (``search_kwargs={"k": ...}``)
    and there is no portable per-call override, so asking for more than the wrapped
    retriever was built for cannot work and silently returning fewer results would
    misreport recall. Truncating locally is honest: the log line records both
    numbers when they disagree.
    """

    name = "langchain"

    def __init__(
        self,
        retriever: Any,
        *,
        name: str | None = None,
        source: RetrievalSource = RetrievalSource.DENSE,
        settings: Settings | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.retriever = retriever
        self.settings = settings or get_settings()
        self.source = source
        self.config = config
        self.name = name or f"langchain:{type(retriever).__name__}"

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """:class:`~ragorc.core.protocols.Retriever` entry point."""
        limit = int(top_k or query.top_k or self.settings.retrieval.top_k)
        with timed("adapter.langchain.reverse", retriever=self.name):
            documents = await self._call(query.text)
        chunks = from_langchain_documents(documents, source=self.source, document_id=self.name)
        if len(chunks) > limit:
            log.debug(
                "langchain_leg_truncated",
                retriever=self.name,
                returned=len(chunks),
                requested=limit,
                hint="raise the wrapped retriever's own k to lift the ceiling",
            )
            chunks = chunks[:limit]
        return chunks

    async def _call(self, text: str) -> list[Any]:
        """Invoke the wrapped retriever by whichever async surface it has.

        ``ainvoke`` is the modern one and ``BaseRetriever`` implements it for
        sync-only subclasses by dispatching to a thread, so it covers almost
        everything. The two fallbacks exist for duck-typed objects that never
        inherited from ``BaseRetriever`` — and the last one runs in a thread rather
        than inline, because a blocking HTTP call inside a coroutine stalls every
        other leg of the fan-out it was supposed to be overlapping with.
        """
        ainvoke = getattr(self.retriever, "ainvoke", None)
        if callable(ainvoke):
            return list(await ainvoke(text, config=self.config))
        legacy = getattr(self.retriever, "aget_relevant_documents", None)
        if callable(legacy):
            return list(await legacy(text))
        sync = getattr(self.retriever, "invoke", None) or getattr(
            self.retriever, "get_relevant_documents", None
        )
        if callable(sync):
            return list(await asyncio.to_thread(sync, text))
        raise ConfigError(
            "object does not look like a LangChain retriever",
            got=type(self.retriever).__name__,
            expected="ainvoke / aget_relevant_documents / invoke",
        )


def from_langchain_retriever(
    retriever: Any,
    *,
    name: str | None = None,
    source: RetrievalSource = RetrievalSource.DENSE,
    settings: Settings | None = None,
    config: dict[str, Any] | None = None,
) -> LangChainRetriever:
    """Adapt a LangChain retriever so it satisfies our ``Retriever`` protocol.

    from ragorc.adapters.langchain import from_langchain_retriever
    from ragorc.retrieve import EnsembleRetriever, HybridRetriever

    ensemble = EnsembleRetriever(
        {"hybrid": HybridRetriever(), "legacy": from_langchain_retriever(their_retriever)}
    )
    """
    return LangChainRetriever(retriever, name=name, source=source, settings=settings, config=config)


# ---------------------------------------------------------------------------
# LCEL
# ---------------------------------------------------------------------------
_METHODS: tuple[str, ...] = (
    "query",
    "retrieve",
    "translate",
    "route",
    "generate",
    "compress",
    "rerank",
)
"""Method names probed to identify a component, most specific first.

``query`` before ``retrieve`` matters: a pipeline exposes both, and wrapping the
pipeline's retrieval half when the caller asked for the pipeline would silently
turn an answering chain into a search chain."""


def _resolve_method(component: Any, method: str | None) -> str:
    if method is not None:
        if not callable(getattr(component, method, None)):
            raise ConfigError(
                f"{type(component).__name__} has no method {method!r}",
                available=[m for m in _METHODS if callable(getattr(component, m, None))],
            )
        return method
    for candidate in _METHODS:
        if callable(getattr(component, candidate, None)):
            return candidate
    if callable(component):
        return "__call__"
    raise ConfigError(
        "cannot adapt this object to a Runnable",
        got=type(component).__name__,
        hint=f"expected one of {', '.join(_METHODS)}, or a callable",
    )


def as_runnable(
    component: Any,
    *,
    method: str | None = None,
    name: str | None = None,
    settings: Settings | None = None,
    **defaults: Any,
) -> Any:
    """Wrap any ragorc component so it can sit in an LCEL pipe.

    The mapping from our stage shapes onto the single-input/single-output shape
    LCEL wants:

    ============  =========================  ==================================
    method        input                      output
    ============  =========================  ==================================
    ``query``     ``str`` / ``dict``         :class:`~ragorc.core.models.Answer`
    ``retrieve``  ``str`` / ``dict``         ``list[Document]``
    ``translate`` ``str`` / ``dict``         :class:`~ragorc.core.models.Query`
    ``route``     ``str`` / ``dict``         :class:`~ragorc.core.models.RouteDecision`
    ``generate``  ``{"query", "retrieval"}`` :class:`~ragorc.core.models.Answer`
    ``compress``  ``{"query", "chunks"}``    ``list[ScoredChunk]``
    ``rerank``    ``{"query", "documents"}`` ``list[tuple[int, float]]``
    ============  =========================  ==================================

    Retrieval is the one that converts to ``Document``, because that is the type an
    LCEL chain's next link expects. The others return our own objects: an ``Answer``
    carries the groundedness score, the citations and the cost ledger, and flattening
    it to a string at the boundary would throw away exactly the guarantees this
    library exists to provide. Pipe it into ``lambda a: a.text`` when a string is
    what you want — the loss is then a choice, made where it is visible.

    The returned ``Runnable`` is async-only: ``.ainvoke`` / ``.abatch`` /
    ``.astream`` work, and ``.invoke`` raises. Every invocation installs its own
    request context, so cost ceilings and the step trace apply inside a chain
    exactly as they do outside one.
    """
    runnable_lambda = _runnable_lambda_class()
    resolved = _resolve_method(component, method)
    config = settings or get_settings()
    label = name or f"ragorc.{getattr(component, 'name', type(component).__name__)}.{resolved}"

    async def call(value: Any) -> Any:
        if resolved == "retrieve":
            return await _retrieve_as_documents(
                component,
                value,
                top_k=defaults.get("top_k"),
                filters=defaults.get("filters"),
                tenant_id=defaults.get("tenant_id"),
                settings=config,
                extra={k: v for k, v in defaults.items() if k not in _RETRIEVE_KEYS},
            )

        with _request_context(config, "lc-runnable"), timed(f"adapter.{label}"):
            return await _dispatch(component, resolved, value, defaults, config)

    return runnable_lambda(call, name=label)


_RETRIEVE_KEYS = frozenset({"top_k", "filters", "tenant_id"})


async def _dispatch(
    component: Any, method: str, value: Any, defaults: dict[str, Any], settings: Settings
) -> Any:
    """Call one component with whatever shape LCEL handed us.

    The two-argument stages (``generate``, ``compress``, ``rerank``) take a dict,
    because there is no way to squeeze a query *and* a document list through a
    single positional input without one of them being implicit — and an implicit
    argument in a chain is a bug that only shows up as a wrong answer.
    """
    fn = getattr(component, method)

    if method in ("translate", "route"):
        # Both return ``(result, Usage)``: the cost is already on the ledger this
        # call installed, so the chain sees the result and the ledger holds the bill.
        result, _usage = await fn(_as_query(value, top_k=None, filters=None, tenant=None))
        return result

    if method == "query":
        if isinstance(value, dict):
            text = value.get("question") or value.get("query") or value.get("input") or ""
            extra = {k: v for k, v in value.items() if k not in ("question", "query", "input")}
        else:
            text = str(value)
            extra = {}
        return await fn(text, **{**defaults, **extra})

    if method in ("generate", "compress", "rerank"):
        if not isinstance(value, dict):
            raise ConfigError(
                f"{method!r} needs a mapping input",
                expected=sorted(_TWO_ARG_KEYS[method]),
                got=type(value).__name__,
            )
        first, second = _TWO_ARG_KEYS[method]
        if second not in value:
            raise ConfigError(f"{method!r} input is missing {second!r}", got=sorted(value))
        primary = value[first] if first in value else value.get("question", "")
        if method == "generate":
            return await fn(
                _as_query(primary, top_k=None, filters=None, tenant=None), value[second]
            )
        if method == "compress":
            result, _usage = await fn(
                _as_query(primary, top_k=None, filters=None, tenant=None), value[second]
            )
            return result
        return await fn(str(primary), value[second], top_k=defaults.get("top_k"))

    return await fn(value, **defaults)


_TWO_ARG_KEYS: dict[str, tuple[str, str]] = {
    "generate": ("query", "retrieval"),
    "compress": ("query", "chunks"),
    "rerank": ("query", "documents"),
}


# ---------------------------------------------------------------------------
# PEP 562: keep ``RagOrcRetriever`` a name without making the import eager
# ---------------------------------------------------------------------------
def __getattr__(name: str) -> Any:
    if name == "RagOrcRetriever":
        value = _retriever_class()
        globals()[name] = value  # cached: __getattr__ only runs on a miss
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
