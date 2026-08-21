"""sentence-transformers provider (torch, ``ragorc[local]``).

When to reach for this instead of FastEmbed
-------------------------------------------
FastEmbed is the default because ONNX on CPU is smaller, faster to start and
free of a CUDA toolchain. Three things only torch can do, and they are the
reasons this module exists:

* **GPU throughput.** A 4090 embeds ~50x what an 8-core CPU does. On a large
  ingest that is hours versus days.
* **Any model on the Hub.** FastEmbed carries a curated ONNX list; sentence-
  transformers loads anything, including a model you fine-tuned on your own
  corpus (see ``scripts/`` and the ``finetune`` extra), which is usually worth
  more than any off-the-shelf upgrade.
* **Cross-encoders beyond the ONNX set**, including the multilingual and large
  rerankers that have no ONNX export.

Device selection is automatic — ``cuda`` > ``mps`` > ``cpu`` — because the wrong
default here is a 50x performance error rather than a bug you would notice.
Note that ``mps`` (Apple silicon) is fast for inference but has historically
mis-handled some fp16 kernels; models load in fp32 to avoid that class of
silently-wrong output.

Everything runs through ``asyncio.to_thread``: ``encode`` is a blocking torch
call, and torch releases the GIL inside its kernels, so the offload buys real
overlap with the network I/O the rest of the pipeline is doing. Model instances
are cached per ``(model, device)`` in a module-level dict, guarded by a threading
lock, since a torch model is hundreds of MB and loading takes seconds.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.errors import EmbeddingError
from ragorc.core.models import FloatArray
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.embed._runtime import register_shutdown_hook
from ragorc.embed.base import BaseEmbedder, run_in_thread
from ragorc.embed.cache import EmbeddingCache

log = structlog.get_logger(__name__)

__all__ = ["STCrossEncoderReranker", "STEmbedder", "detect_device"]

_MODELS: dict[tuple[str, str, str], Any] = {}
_LOAD_LOCK = threading.Lock()


def clear_model_cache() -> int:
    """Drop every cached model. Returns how many were released.

    Registered as a shutdown hook for the reason in :mod:`ragorc.embed._runtime`:
    a process that exits with native model sessions still alive can abort during
    static destruction. Until this existed only FastEmbed's cache was released,
    so a deployment on this provider got none of the mitigation.
    """
    with _LOAD_LOCK:
        count = len(_MODELS)
        _MODELS.clear()
    return count


register_shutdown_hook(clear_model_cache)

_IMPORT_HINT = "sentence-transformers and torch are required: pip install 'ragorc[local]'"


def detect_device(preferred: str | None = None) -> str:
    """Pick the fastest available device. ``preferred`` short-circuits detection."""
    if preferred:
        return preferred
    try:
        import torch
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _load(kind: str, model_name: str, device: str) -> Any:
    """Load-once cache keyed by (kind, model, device).

    Double-checked locking with a ``threading.Lock``: loading happens inside a
    worker thread, so an asyncio lock would be the wrong primitive, and two
    concurrent first calls must not each allocate a copy of the weights.
    """
    key = (kind, model_name, device)
    model = _MODELS.get(key)
    if model is not None:
        return model
    with _LOAD_LOCK:
        model = _MODELS.get(key)
        if model is not None:
            return model
        try:
            if kind == "encoder":
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name, device=device)
            else:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(model_name, device=device)
        except ImportError as exc:
            raise ImportError(_IMPORT_HINT) from exc
        except (OSError, ValueError) as exc:
            raise EmbeddingError(
                f"could not load sentence-transformers model {model_name!r}",
                kind=kind,
                device=device,
                error=str(exc)[:300],
            ) from exc
        log.info("st_model_loaded", kind=kind, model=model_name, device=device)
        _MODELS[key] = model
        return model


@register("dense_embedder", "sentence_transformers", "st")
class STEmbedder(BaseEmbedder):
    """Dense embeddings via ``SentenceTransformer.encode``."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        cache: EmbeddingCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        resolved = settings or get_settings()
        config = resolved.embedding
        super().__init__(
            model_name=model_name or config.dense_model,
            # Real values are read off the loaded model on first use; the
            # configured values are the pre-load contract with the vector store.
            dimension=config.dense_dimension or 0,
            max_tokens=config.max_length,
            cache=cache,
            settings=resolved,
        )
        self.device = device
        self._resolved_device: str | None = None

    def _model(self) -> Any:
        if self._resolved_device is None:
            self._resolved_device = detect_device(self.device)
        model = _load("encoder", self.model_name, self._resolved_device)
        if not self.dimension:
            self.dimension = int(model.get_sentence_embedding_dimension() or 0)
        limit = int(getattr(model, "max_seq_length", 0) or 0)
        if limit:
            self.max_tokens = limit
        return model

    async def _embed_batch(self, texts: Sequence[str], *, is_query: bool) -> list[FloatArray]:
        # `is_query` is unused: the asymmetry for this provider is expressed
        # through the configured prefixes, which the mixin has already applied.
        return await run_in_thread(self._encode_sync, list(texts))

    def _encode_sync(self, texts: list[str]) -> list[FloatArray]:
        model = self._model()
        matrix = model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            # Let torch normalize on-device: for a GPU batch this avoids a
            # round trip through host memory just to divide by a norm.
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        return list(np.ascontiguousarray(matrix, dtype=np.float32))


@register("reranker", "sentence_transformers", "st_cross_encoder")
class STCrossEncoderReranker:
    """Cross-encoder reranking via ``sentence_transformers.CrossEncoder``.

    Scores are the model's raw output — logits for most ms-marco checkpoints,
    already-sigmoided probabilities for the ones trained with one output neuron
    and a sigmoid head. Either way higher is better and the values are comparable
    within a single query, which is all a reranker needs to be.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding.reranker_model
        self.device = device
        self._resolved_device: str | None = None

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        docs = list(documents)
        if not docs:
            return []
        scores = await run_in_thread(self._score_sync, query, docs)
        limit = min(top_k, len(docs)) if top_k else len(docs)
        if limit < len(docs):
            top = np.argpartition(-scores, limit - 1)[:limit]
            top = top[np.argsort(-scores[top], kind="stable")]
        else:
            top = np.argsort(-scores, kind="stable")
        return [(int(i), float(scores[i])) for i in top]

    def _score_sync(self, query: str, documents: list[str]) -> FloatArray:
        if self._resolved_device is None:
            self._resolved_device = detect_device(self.device)
        model = _load("cross", self.model_name, self._resolved_device)
        raw = model.predict(
            [(query, doc) for doc in documents],
            batch_size=self.settings.retrieval.rerank_batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = np.asarray(raw, dtype=np.float32)
        # Multi-class heads (e.g. NLI-trained rerankers) emit one row per pair;
        # the relevant signal is the last column.
        return scores if scores.ndim == 1 else scores[:, -1]
