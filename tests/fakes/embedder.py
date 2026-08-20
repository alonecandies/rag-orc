"""Embedding doubles — deterministic, no model download."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ragorc.core.models import SparseVector


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little")


class StubEmbedder:
    """Hash-seeded random vectors.

    Deterministic per text, and *uncorrelated* between different texts — which is
    the property a ranking test needs. An embedder that returned, say, a
    character-frequency vector would make every English sentence similar to every
    other, and similarity assertions would pass for the wrong reason.

    ``anchors`` lets a test pin specific texts near each other when it *does* want
    controlled similarity: any text containing an anchor key is nudged toward that
    anchor's direction.
    """

    def __init__(
        self, dimension: int = 32, *, anchors: dict[str, int] | None = None, normalize: bool = True
    ) -> None:
        self.dimension = dimension
        self.model_name = "stub-embedder"
        self.max_tokens = 512
        self.normalize = normalize
        self.anchors = anchors or {}
        self.calls: list[tuple[str, int]] = []

    def _vector(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(_seed(text) % (2**32))
        vector = rng.normal(size=self.dimension).astype(np.float32)
        for key, axis in self.anchors.items():
            if key.lower() in text.lower():
                # Dominate one axis so anchored texts cluster predictably.
                vector[axis % self.dimension] += 6.0
        if self.normalize:
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                vector = vector / norm
        return vector.astype(np.float32)

    async def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        self.calls.append(("embed_documents", len(texts)))
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        self.calls.append(("embed_query", 1))
        return self._vector(text)

    async def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        self.calls.append(("embed_queries", len(texts)))
        return [self._vector(t) for t in texts]

    @property
    def batch_calls(self) -> int:
        """How many round trips were made. A test asserting that N variants were
        embedded in ONE batch checks this, not the number of vectors."""
        return len(self.calls)


class StubSparseEmbedder:
    """Bag-of-words sparse vectors with a stable vocabulary hash."""

    def __init__(self, *, is_lexical: bool = True, vocab_size: int = 4096) -> None:
        self.model_name = "stub-sparse"
        self.is_lexical = is_lexical
        self.vocab_size = vocab_size

    def _vector(self, text: str) -> SparseVector:
        counts: dict[int, float] = {}
        for word in text.lower().split():
            index = _seed(word) % self.vocab_size
            counts[index] = counts.get(index, 0.0) + 1.0
        return SparseVector.from_dict(counts)

    async def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> SparseVector:
        return self._vector(text)


class StubLateInteractionEmbedder:
    """Per-token matrices of shape (n_tokens, dim)."""

    def __init__(self, dimension: int = 8, max_tokens: int = 16) -> None:
        self.dimension = dimension
        self.model_name = "stub-colbert"
        self.max_tokens = max_tokens

    def _matrix(self, text: str) -> np.ndarray:
        tokens = text.split()[: self.max_tokens] or ["empty"]
        rows = []
        for token in tokens:
            rng = np.random.default_rng(_seed(token) % (2**32))
            vector = rng.normal(size=self.dimension).astype(np.float32)
            rows.append(vector / max(float(np.linalg.norm(vector)), 1e-9))
        return np.asarray(rows, dtype=np.float32)

    async def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._matrix(t) for t in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        return self._matrix(text)


class StubReranker:
    """Scores by shared-word overlap, so results are predictable and sensible."""

    def __init__(self) -> None:
        self.model_name = "stub-reranker"
        self.calls: list[tuple[str, int]] = []

    async def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        self.calls.append((query, len(documents)))
        query_words = {w for w in query.lower().split() if len(w) > 2}
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            doc_words = {w for w in doc.lower().split() if len(w) > 2}
            overlap = len(query_words & doc_words)
            scored.append((i, overlap / max(len(query_words), 1)))
        scored.sort(key=lambda t: (-t[1], t[0]))
        return scored[:top_k] if top_k else scored
