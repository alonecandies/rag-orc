"""Late chunking — embed the document once, pool per chunk span (ADR-0002).

The problem it solves
---------------------
Early chunking embeds each chunk in isolation, so a chunk is represented by
whatever it says about itself. Real prose does not work that way. A chunk reading
"Its revenue grew 40% year over year, driven by the enterprise tier" has lost
*who* "it" is: the sentence naming the company is two chunks up. The vector
lands somewhere in generic-financial-language space and no query mentioning the
company will retrieve it. Coreference, section context and topic drift all fail
the same way, and no amount of overlap tuning fixes it because the missing
information is not adjacent — it is upstream.

Late chunking (Günther et al., 2024, "Late Chunking: Contextual Chunk Embeddings
Using Long-Context Embedding Models") inverts the order of two operations:

    early:  split -> embed each chunk           (N forward passes, no context)
    late:   embed whole document -> pool spans  (1 forward pass, full context)

Because the transformer sees the entire document in one pass, every token vector
is conditioned on every other token — the tokens in "Its revenue grew 40%" have
already attended to the company name. Mean-pooling those token vectors over a
chunk's span therefore produces a chunk embedding that *encodes its own context*
while still pointing at that chunk's text.

It is also **cheaper than early chunking**, which is the part people find
surprising. One forward pass over a 4000-token document costs less than 30
forward passes over its 30 chunks, because the chunks re-tokenize and re-embed
the same overlap regions and each pass re-pays the per-call overhead. Attention
is quadratic in sequence length, so the saving narrows on very long inputs, but
at realistic document sizes late chunking is both better and faster.

Two backends
------------
**Backend B (torch, ``ragorc[local]``)** — ``transformers.AutoModel`` with
``output_hidden_states``, taking ``last_hidden_state``. This is true late
chunking: with a long-context encoder (``jinaai/jina-embeddings-v2-base-en``
handles 8192 tokens) a whole document fits in one window, which is the regime the
paper measures. Preferred whenever torch is importable, for a second reason
beyond context length: the backend loads
``settings.embedding.dense_model`` itself, so the pooled vectors come from the
same weights as the rest of the index.

**Backend A (no torch)** — FastEmbed's ``LateInteractionTextEmbedding``, which
already emits one vector per token, plus that model's own tokenizer for
character offsets. Mean-pooling ColBERT token vectors is a legitimate
context-aware dense representation — the tokens are contextualized by the same
attention mechanism — but it lives in **ColBERT space**, not in the space of any
pooled sentence encoder. Which brings up the one rule that governs both
backends:

    Queries must be embedded through *this* object's ``embed_query``.

Mean-pooling a model's token outputs reproduces the model's own sentence
embedding only when that model was trained with mean pooling (jina-v2, E5). BGE
and friends pool the ``[CLS]`` token instead, and ColBERT does not pool at all.
In every case the pooled space is self-consistent and comparable with itself, and
in some cases it is *not* comparable with the provider's ``embed_documents``
output. Routing both sides through this class makes that a non-issue; mixing them
is a silent recall collapse, which is why the factory wires the query side here
too.

Long documents: overlapping macro-windows
-----------------------------------------
When a document exceeds the model's window it is processed in windows of
``max_tokens`` with 25% overlap, and each chunk is pooled from the window that
contains it **most fully**, tie-broken toward the window where it sits closest to
the centre. The overlap is not defensive padding, it is the whole point: a chunk
that straddles a window boundary would otherwise be pooled from tokens that only
ever saw half of their context — reintroducing exactly the truncated-context
failure late chunking exists to remove. With 25% overlap every chunk smaller than
a quarter-window is fully interior to some window, and the centre tie-break
prefers the window that gives it the most context on *both* sides.

Failure mode
------------
If no backend can produce token embeddings, ``embed_chunks`` raises. It never
falls back to per-chunk embedding internally: that would return vectors from a
different space than the rest of the index, under a method whose name promises
otherwise, and the resulting recall loss would be invisible. Callers use
``resolve_strategy`` to pick a strategy up front, and the ladder degrades
LATE -> CONTEXTUAL -> EARLY explicitly and in the logs.
"""

from __future__ import annotations

import importlib.util
import threading
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import EmbeddingError
from ragorc.core.models import ChunkingStrategy, FloatArray
from ragorc.core.settings import Settings, get_settings
from ragorc.embed._runtime import register_shutdown_hook
from ragorc.embed.base import l2_normalize, run_in_thread

log = structlog.get_logger(__name__)

__all__ = ["LateChunkingEmbedder", "resolve_strategy"]

_OVERLAP_FRACTION = 0.25
"""Macro-window overlap. A quarter of the window is enough that any chunk up to
a quarter-window long is fully interior to at least one window, while costing
only ~33% more forward-pass tokens than disjoint windows."""

_SPECIAL_TOKEN_SLACK = 4
"""Room for [CLS]/[SEP]/marker tokens the model adds to every window."""

_TORCH_MODELS: dict[tuple[str, str], tuple[Any, Any, int]] = {}
_TORCH_LOCK = threading.Lock()


def clear_model_cache() -> int:
    """Drop every cached torch model. Returns how many were released.

    Registered as a shutdown hook for the reason in :mod:`ragorc.embed._runtime`:
    a process that exits with native model sessions still alive can abort during
    static destruction rather than exiting cleanly. That module promised to
    release "the cached models"; until this existed it released FastEmbed's and
    left these, so a late-chunking deployment got none of the mitigation.
    """
    with _TORCH_LOCK:
        count = len(_TORCH_MODELS)
        _TORCH_MODELS.clear()
    return count


register_shutdown_hook(clear_model_cache)


class _TokenBackend(Protocol):
    """What late chunking needs from a model: offsets, and per-token vectors."""

    name: str
    max_tokens: int

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every token in ``text``, untruncated."""
        ...

    def token_vectors(self, text: str) -> tuple[FloatArray, list[tuple[int, int]]]:
        """``(n_content_tokens, dim)`` vectors plus each row's character span."""
        ...


# ---------------------------------------------------------------------------
# Backend A: FastEmbed late-interaction (no torch)
# ---------------------------------------------------------------------------
class _FastEmbedBackend:
    """Wraps a late-interaction embedder that can expose token offsets."""

    name = "fastembed-late-interaction"

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.max_tokens = int(getattr(embedder, "max_tokens", 0) or 512)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.embedder.token_offsets(text)

    def token_vectors(self, text: str) -> tuple[FloatArray, list[tuple[int, int]]]:
        return self.embedder.token_vectors_sync(text)


# ---------------------------------------------------------------------------
# Backend B: transformers (torch)
# ---------------------------------------------------------------------------
def _torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load_hf(model_name: str, device: str) -> tuple[Any, Any, int]:
    """Load tokenizer + encoder once per (model, device).

    Guarded by a ``threading.Lock`` because loading happens inside a worker
    thread; a torch encoder is hundreds of MB and seconds of load, so two
    concurrent first calls must not each build one.
    """
    key = (model_name, device)
    cached = _TORCH_MODELS.get(key)
    if cached is not None:
        return cached
    with _TORCH_LOCK:
        cached = _TORCH_MODELS.get(key)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch late chunking needs transformers: pip install 'ragorc[local]'"
            ) from exc

        def _build(trust: bool) -> tuple[Any, Any]:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=trust)
            return tokenizer, model

        try:
            tokenizer, model = _build(False)
        except (ValueError, OSError, KeyError) as exc:
            if "trust_remote_code" not in str(exc):
                raise EmbeddingError(
                    f"could not load {model_name!r} for late chunking",
                    device=device,
                    error=str(exc)[:300],
                ) from exc
            # The long-context models this feature is built for (jina-v2 and
            # relatives) ship their own attention implementation, so the repo
            # executes code at load time. Escalating is the only way to use
            # them; it is logged rather than silent because it is a real trust
            # decision about a third-party repository.
            log.warning(
                "late_chunking_trust_remote_code",
                model=model_name,
                reason="model repository ships custom modeling code",
            )
            tokenizer, model = _build(True)

        model = model.to(device)
        model.eval()
        # `model_max_length` is often a 1e30 sentinel, so the positional limit is
        # the real bound. Getting this wrong either truncates 8192-token models
        # at 512 or feeds a 512-token model 8192 tokens of garbage.
        positional = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        declared = int(getattr(tokenizer, "model_max_length", 0) or 0)
        limit = (
            min(x for x in (positional, declared) if 0 < x < 1_000_000)
            if (0 < positional < 1_000_000 or 0 < declared < 1_000_000)
            else 512
        )
        log.info(
            "late_chunking_model_loaded",
            model=model_name,
            device=device,
            max_tokens=limit,
            torch_version=torch.__version__,
        )
        _TORCH_MODELS[key] = (tokenizer, model, limit)
        return _TORCH_MODELS[key]


class _TorchBackend:
    """``transformers.AutoModel`` last hidden state, one row per token."""

    name = "transformers"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or _torch_device()
        self.tokenizer, self.model, self.max_tokens = _load_hf(model_name, self.device)

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return [tuple(pair) for pair in encoding["offset_mapping"]]

    def token_vectors(self, text: str) -> tuple[FloatArray, list[tuple[int, int]]]:
        import torch

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
            max_length=self.max_tokens,
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        special = encoding.pop("special_tokens_mask")[0].tolist()
        inputs = {key: value.to(self.device) for key, value in encoding.items()}
        with torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output.hidden_states[-1]
        # .float() first: fp16/bf16 weights would otherwise pool in half
        # precision, where a 512-term mean loses real bits.
        matrix = hidden[0].float().cpu().numpy()
        keep = ~np.asarray(special, dtype=bool)
        rows = np.ascontiguousarray(matrix[keep], dtype=np.float32)
        spans = [tuple(offsets[i]) for i in np.flatnonzero(keep).tolist()]
        return rows, spans


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _torch_available() -> bool:
    return _module_present("torch") and _module_present("transformers")


# ---------------------------------------------------------------------------
# The embedder
# ---------------------------------------------------------------------------
class LateChunkingEmbedder:
    """Embeds chunk spans by pooling one document-wide forward pass."""

    def __init__(
        self,
        token_embedder: Any | None = None,
        tokenizer_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.embedding
        self.token_embedder = token_embedder
        self.tokenizer_name = tokenizer_name or self.config.dense_model
        self.model_name = self.tokenizer_name
        self.dimension = self.config.dense_dimension or 0
        self.max_tokens = self.config.max_length
        self._backend: _TokenBackend | None = None
        self._lock = threading.Lock()

    # -- capability --------------------------------------------------------
    @property
    def supports_token_embeddings(self) -> bool:
        """Whether a token-level backend can be built, checked *cheaply*.

        The strategy resolver calls this before ingest starts, so it must not
        download or load a model: presence of the packages, or of a token-capable
        embedder, is the whole test. A backend that then fails to load raises at
        ``embed_chunks`` rather than lying here.

        Note what is deliberately *not* a qualifying condition: fastembed merely
        being installed. That used to return True, because the backend resolver
        would substitute ColBERT for any embedder that could not emit tokens. It
        no longer does — a pooled ColBERT vector is not comparable to a query
        vector from the dense model — so "fastembed is present" says nothing about
        whether late chunking is possible, and answering True on it made the
        resolver choose a strategy that then failed at the vector write.
        """
        if self._backend is not None:
            return True
        if _torch_available():
            return True
        return self._embedder_is_token_capable(self.token_embedder)

    @staticmethod
    def _embedder_is_token_capable(embedder: Any | None) -> bool:
        return embedder is not None and all(
            hasattr(embedder, attr) for attr in ("token_offsets", "token_vectors_sync")
        )

    def backend_name(self) -> str:
        return self._backend.name if self._backend is not None else "unresolved"

    # -- public API --------------------------------------------------------
    async def embed_chunks(
        self, document_text: str, spans: Sequence[tuple[int, int]]
    ) -> list[FloatArray]:
        """Embed each ``(start_char, end_char)`` span of ``document_text``.

        One thread hop covers model load, tokenization, the forward passes and the
        pooling: all of it is blocking CPU (or GPU) work, and splitting it across
        hops would only add scheduling latency.
        """
        span_list = [(int(start), int(end)) for start, end in spans]
        if not span_list:
            return []
        return await run_in_thread(self._embed_chunks_sync, document_text, span_list)

    async def embed_query(self, text: str) -> FloatArray:
        """Pool the whole query through the same backend.

        This is not an optional convenience: the pooled document space is only
        guaranteed to be comparable with itself (see the module docstring), so the
        query side must be produced the same way.
        """
        vectors = await self.embed_chunks(text, [(0, len(text))])
        return vectors[0]

    async def embed_queries(self, texts: Sequence[str]) -> list[FloatArray]:
        return await bounded_gather(
            (self.embed_query(text) for text in texts),
            limit=max(1, self.settings.indexing.max_concurrent_documents),
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[FloatArray]:
        """Whole-text pooling, one document per call.

        Present so this object satisfies :class:`DenseEmbedder` and can be used
        as the query-side embedder; the interesting entry point is
        ``embed_chunks``.
        """
        return await self.embed_queries(texts)

    # -- backend resolution ------------------------------------------------
    def _ensure_backend(self) -> _TokenBackend:
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is None:
                self._backend = self._build_backend()
                self.max_tokens = self._backend.max_tokens
                log.info(
                    "late_chunking_backend",
                    backend=self._backend.name,
                    max_tokens=self._backend.max_tokens,
                )
            return self._backend

    def _build_backend(self) -> _TokenBackend:
        attempts: list[str] = []
        if _torch_available():
            try:
                return _TorchBackend(self.tokenizer_name)
            except (ImportError, EmbeddingError, OSError, ValueError) as exc:
                attempts.append(f"transformers({self.tokenizer_name}): {str(exc)[:200]}")
                log.warning(
                    "late_chunking_torch_backend_failed",
                    model=self.tokenizer_name,
                    error=str(exc)[:200],
                )

        embedder = self.token_embedder
        if not self._embedder_is_token_capable(embedder):
            # No substitution. An earlier version fell back to
            # ``FastEmbedLateInteraction`` here on the reasoning that ColBERT
            # already emits one vector per token — true, and the wrong conclusion.
            #
            # Pooled token vectors are only usable as a chunk vector if they land
            # in the same space as the *query* vector, and the query is embedded by
            # the dense model. ColBERT is a different model in a different space at
            # a different width (128 vs 384 for the default BGE), so the pooled
            # output is not comparable to anything the retriever will ask with.
            # Qdrant rejected it on dimension, which was luck: at equal width it
            # would have been accepted and returned quietly meaningless neighbours.
            #
            # Late chunking needs *one* model emitting both the token vectors and
            # the pooled document vector (Günther et al., 2024). If the dense model
            # cannot do that, late chunking is genuinely unavailable — so raise, and
            # let ``resolve_strategy`` degrade to CONTEXTUAL or EARLY with a log
            # line, rather than indexing a corpus that can never be retrieved.
            if embedder is not None:
                attempts.append(
                    f"{type(embedder).__name__}({getattr(embedder, 'model_name', '?')}): "
                    "exposes no token offsets"
                )
            else:
                attempts.append("no token embedder supplied")
            embedder = None

        if self._embedder_is_token_capable(embedder):
            return _FastEmbedBackend(embedder)

        raise EmbeddingError(
            "no token-embedding backend available: late chunking is impossible",
            tried=attempts,
            hint=(
                "install 'ragorc[local]' for transformers-based late chunking, or "
                "set indexing.chunking_strategy to 'contextual' or 'early'"
            ),
        )

    # -- the pooling core --------------------------------------------------
    def _embed_chunks_sync(self, text: str, spans: list[tuple[int, int]]) -> list[FloatArray]:
        backend = self._ensure_backend()
        doc_offsets = backend.offsets(text)
        if not doc_offsets:
            raise EmbeddingError(
                "document tokenized to zero tokens; nothing to pool",
                chars=len(text),
                backend=backend.name,
            )

        budget = max(1, backend.max_tokens - _SPECIAL_TOKEN_SLACK)
        windows = _plan_windows(doc_offsets, budget, len(text))
        span_array = np.asarray(spans, dtype=np.int64)
        assignment, coverage = _assign_spans(span_array, windows, len(text))
        if coverage.min() < 1.0:
            log.warning(
                "late_chunking_span_exceeds_window",
                worst_coverage=round(float(coverage.min()), 3),
                window_tokens=budget,
                hint="chunk_size is close to the model's context window",
            )

        pooled: list[FloatArray | None] = [None] * len(spans)
        for window_index in range(windows.shape[0]):
            members = np.flatnonzero(assignment == window_index)
            if members.size == 0:
                # Overlapping windows mean some windows own no chunk; skipping
                # them is the saving that makes overlap affordable.
                continue
            start_char = int(windows[window_index, 0])
            end_char = int(windows[window_index, 1])
            matrix, token_spans = backend.token_vectors(text[start_char:end_char])
            if matrix.shape[0] == 0 or not token_spans:
                raise EmbeddingError(
                    "backend returned no token vectors for a window",
                    backend=backend.name,
                    window=(start_char, end_char),
                )
            if not self.dimension:
                self.dimension = int(matrix.shape[1])

            vectors = _pool_spans(matrix, token_spans, start_char, span_array[members])
            for slot, vector in zip(members.tolist(), vectors, strict=True):
                pooled[slot] = vector

        complete = [vector for vector in pooled if vector is not None]
        if len(complete) != len(pooled):  # pragma: no cover - every span gets a window
            raise EmbeddingError(
                "some chunk spans were not covered by any window",
                spans=len(spans),
                missing=len(pooled) - len(complete),
            )
        # One normalization pass over the whole document's chunks.
        return list(l2_normalize(np.stack(complete)))


def _plan_windows(doc_offsets: list[tuple[int, int]], budget: int, total_chars: int) -> np.ndarray:
    """Character ranges of the macro-windows, ``(n_windows, 2)``.

    The first window starts at character 0 and the last ends at ``total_chars``
    even when the tokenizer dropped leading or trailing whitespace, so every
    character of the document belongs to at least one window and no chunk span
    can point outside all of them.
    """
    n_tokens = len(doc_offsets)
    if n_tokens <= budget:
        return np.asarray([[0, total_chars]], dtype=np.int64)

    stride = max(1, int(budget * (1.0 - _OVERLAP_FRACTION)))
    ranges: list[tuple[int, int]] = []
    token_start = 0
    while True:
        token_end = min(token_start + budget, n_tokens)
        first_char = 0 if token_start == 0 else int(doc_offsets[token_start][0])
        last_char = total_chars if token_end >= n_tokens else int(doc_offsets[token_end - 1][1])
        ranges.append((first_char, last_char))
        if token_end >= n_tokens:
            break
        token_start += stride
    return np.asarray(ranges, dtype=np.int64)


def _assign_spans(
    spans: np.ndarray, windows: np.ndarray, total_chars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pick, for every span, the window that contains it most fully.

    Fully vectorized over ``(n_spans, n_windows)``: coverage is the fraction of
    the span's characters inside the window, and ties — the common case, since
    overlapping windows both contain a small chunk entirely — are broken toward
    the window whose centre is closest to the span's centre, which is the window
    that saw the most context on both sides of it.

    The coverage term is rounded and scaled so it strictly dominates the centre
    penalty: a genuinely better-covered window always wins, and the penalty only
    decides between equals.
    """
    span_start = spans[:, 0:1].astype(np.float64)
    span_end = spans[:, 1:2].astype(np.float64)
    win_start = windows[None, :, 0].astype(np.float64)
    win_end = windows[None, :, 1].astype(np.float64)

    overlap = np.clip(np.minimum(span_end, win_end) - np.maximum(span_start, win_start), 0.0, None)
    length = np.maximum(span_end - span_start, 1.0)
    coverage = np.clip(overlap / length, 0.0, 1.0)

    centre_gap = np.abs((span_start + span_end) * 0.5 - (win_start + win_end) * 0.5) / float(
        total_chars + 1
    )
    score = np.round(coverage, 6) * 1e6 - centre_gap

    chosen = np.argmax(score, axis=1)
    return chosen, coverage[np.arange(coverage.shape[0]), chosen]


def _pool_spans(
    matrix: FloatArray,
    token_spans: list[tuple[int, int]],
    offset: int,
    spans: np.ndarray,
) -> list[FloatArray]:
    """Mean-pool ``matrix`` rows over each character span. No Python loop.

    Token spans are non-decreasing, so the token range overlapping a character
    span ``[s, e)`` is found with two ``searchsorted`` calls rather than a scan:
    the first token whose end is past ``s``, up to the first token whose start
    reaches ``e``.

    The means themselves come from a prefix sum: ``cumsum`` once per window makes
    every span's sum two row lookups and a subtraction, so N chunks cost
    ``O(tokens * dim + N * dim)`` instead of ``O(sum of span lengths * dim)``.
    Accumulation stays in float32 — the relative error of a few-thousand-term
    cumulative sum is ~1e-5, which vanishes under the L2 normalization that
    follows.
    """
    starts = (
        np.fromiter((pair[0] for pair in token_spans), dtype=np.int64, count=len(token_spans))
        + offset
    )
    ends = (
        np.fromiter((pair[1] for pair in token_spans), dtype=np.int64, count=len(token_spans))
        + offset
    )

    n_tokens, dim = matrix.shape
    prefix = np.zeros((n_tokens + 1, dim), dtype=np.float32)
    np.cumsum(matrix, axis=0, out=prefix[1:])

    lower = np.searchsorted(ends, spans[:, 0], side="right")
    upper = np.searchsorted(starts, spans[:, 1], side="left")
    lower = np.clip(lower, 0, n_tokens - 1)
    upper = np.clip(upper, 0, n_tokens)

    degenerate = upper <= lower
    if degenerate.any():
        # A span that covers no token at all: whitespace-only, or text the
        # tokenizer dropped. Pooling the nearest single token keeps the chunk
        # retrievable instead of emitting a zero vector that matches everything
        # equally badly.
        log.warning(
            "late_chunking_empty_span",
            count=int(degenerate.sum()),
            hint="span covers no tokens; pooled from the nearest token",
        )
        upper = np.where(degenerate, np.minimum(lower + 1, n_tokens), upper)
        lower = np.where(degenerate, upper - 1, lower)

    counts = (upper - lower).astype(np.float32)
    pooled = (prefix[upper] - prefix[lower]) / counts[:, None]
    return list(pooled)


# ---------------------------------------------------------------------------
# Strategy resolution (the AUTO ladder)
# ---------------------------------------------------------------------------
def _token_embeddings_available(embedder: Any, settings: Settings) -> bool:
    """Whether token-level embeddings are reachable *for this dense embedder*.

    "Reachable" is not enough on its own. Pooling token vectors only yields a
    usable chunk vector if the pooled result lands in the **same space as the
    query embedding** — and queries are embedded by the dense model. A token
    source that is a *different* model produces vectors that are not comparable
    to a query vector at all, however good they are in isolation.

    The concrete failure this prevents: with the FastEmbed default, the only
    token-level model available is ColBERT (128-dim), while the dense model is
    BGE-small (384-dim). Pooling ColBERT tokens gives 128-dim vectors in
    ColBERT space, which Qdrant rejects outright against a 384-dim collection —
    and had the dimensions happened to agree, it would have accepted them and
    returned quietly meaningless neighbours instead, which is far worse.

    True late chunking (Günther et al., 2024) uses **one** model for both: the
    same weights emit the token vectors and the pooled document vector. That is
    what the transformers backend under ``ragorc[local]`` provides.
    """
    flag = getattr(embedder, "supports_token_embeddings", None)
    if isinstance(flag, bool):
        if not flag:
            return False
    elif embedder is not None and all(
        hasattr(embedder, attr) for attr in ("token_offsets", "token_vectors_sync")
    ):
        pass
    elif not LateChunkingEmbedder(embedder, settings=settings).supports_token_embeddings:
        return False

    return _token_source_matches_dense(embedder, settings)


def _token_source_matches_dense(embedder: Any, settings: Settings) -> bool:
    """Reject a token source that is a different model from the dense embedder."""
    dense_name = getattr(embedder, "model_name", None) or settings.embedding.dense_model
    dense_dim = getattr(embedder, "dimension", None)

    probe = LateChunkingEmbedder(embedder, settings=settings)
    source = getattr(probe, "token_embedder", None)
    if source is None or source is embedder:
        # No separate source means the dense model itself supplies the tokens,
        # which is exactly the configuration late chunking is defined for.
        return True

    source_name = getattr(source, "model_name", None)
    source_dim = getattr(source, "dimension", None)

    if source_name and dense_name and source_name == dense_name:
        return True
    if dense_dim is not None and source_dim is not None and dense_dim != source_dim:
        log.warning(
            "late_chunking_unusable",
            reason="token source is a different model from the dense embedder",
            dense_model=dense_name,
            dense_dim=dense_dim,
            token_model=source_name,
            token_dim=source_dim,
            effect="pooled vectors would not be comparable to query vectors",
            hint=(
                "install 'ragorc[local]' and use a model that exposes token "
                "embeddings (e.g. jinaai/jina-embeddings-v2-base-en), or accept "
                "the CONTEXTUAL/EARLY fallback"
            ),
        )
        return False
    # Same dimension but a different model is still a different space; the
    # dimensions agreeing is a coincidence, not compatibility.
    log.warning(
        "late_chunking_unusable",
        reason="token source is a different model from the dense embedder",
        dense_model=dense_name,
        token_model=source_name,
        effect="pooled vectors would be in a different embedding space",
    )
    return False


async def resolve_strategy(
    requested: ChunkingStrategy,
    embedder: Any,
    settings: Settings | None = None,
) -> ChunkingStrategy:
    """Resolve ``AUTO`` (or an impossible request) to a strategy that can run.

    The ladder is LATE -> CONTEXTUAL -> EARLY:

    * **LATE** whenever token embeddings are reachable — best quality, and
      cheaper than the alternatives at index time.
    * **CONTEXTUAL** next, when enabled: an LLM writes a situating sentence per
      chunk before embedding. It recovers most of what late chunking recovers,
      at one LLM call per chunk, which is orders of magnitude more expensive.
    * **EARLY** as the floor. Always available, always the weakest.

    An explicit LATE request that cannot be served is downgraded through the same
    ladder with a warning rather than failing the ingest — but it is never
    downgraded silently, because a corpus indexed with the wrong strategy looks
    fine and retrieves badly.

    The capability probe touches the filesystem (``find_spec``), so it runs in a
    worker thread; this is called once per ingest, not per document.
    """
    resolved = settings or get_settings()
    available = await run_in_thread(_token_embeddings_available, embedder, resolved)

    if requested is not ChunkingStrategy.AUTO:
        if requested is not ChunkingStrategy.LATE or available:
            log.info(
                "chunking_strategy_explicit",
                strategy=requested.value,
                token_embeddings=available,
            )
            return requested
        log.warning(
            "late_chunking_requested_but_unavailable",
            hint="install 'ragorc[local]' for transformers-based late chunking",
        )

    if available:
        chosen, reason = ChunkingStrategy.LATE, "token_embeddings_available"
    elif resolved.indexing.contextual_enabled:
        chosen, reason = ChunkingStrategy.CONTEXTUAL, "contextual_enabled"
    else:
        chosen, reason = ChunkingStrategy.EARLY, "no_better_option"

    log.info(
        "chunking_strategy_resolved",
        requested=requested.value,
        chosen=chosen.value,
        reason=reason,
    )
    return chosen
