"""Semantic routing: choose the prompt by embedding similarity, with no LLM call.

This is the cheapest useful decision in the pipeline. Each candidate prompt is
described by a handful of exemplar questions; those exemplars are embedded once
and cached, and routing a query is then a single matmul against the cached matrix.

The reason that matters: prompt selection has to happen on *every* request, and a
model call for it would add both latency and cost to the critical path for a
decision that embeddings make well. A technical question and a "give me a short
answer" question sit in visibly different regions of embedding space.

Scoring uses max-similarity-per-route rather than the centroid. A route's
exemplars deliberately span several phrasings of its intent, so averaging them
produces a vector that resembles none of them — the classic centroid failure on
multi-modal clusters. The best-matching exemplar is the honest signal.
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog

from ragorc.core.models import FloatArray, Query, RouteDecision, Usage
from ragorc.core.protocols import DenseEmbedder
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["DEFAULT_ROUTES", "SemanticRouter"]

#: prompt name -> exemplar questions. Exemplars should differ in *phrasing*, not
#: just in topic, since that is what the embedding compares.
DEFAULT_ROUTES: dict[str, list[str]] = {
    "answer_technical": [
        "how do I configure the connection pool size",
        "what is the exact syntax for this command",
        "show me the code to initialize the client",
        "which flag enables quantization",
        "what does this error message mean",
        "what are the required environment variables",
    ],
    "answer_concise": [
        "when was it released",
        "who is the maintainer",
        "how many days does it take",
        "what is the default value",
        "is it enabled by default",
        "what is the price",
    ],
    "answer_default": [
        "explain how this works",
        "why was this approach chosen",
        "what are the trade-offs involved",
        "compare these two options",
        "walk me through the process",
        "what should I consider before deciding",
    ],
}


@register("router", "semantic")
class SemanticRouter:
    name = "semantic"

    def __init__(
        self,
        embedder: DenseEmbedder,
        routes: dict[str, list[str]] | None = None,
        settings: Settings | None = None,
        *,
        min_similarity: float = 0.30,
        default_route: str = "answer_default",
    ) -> None:
        self.embedder = embedder
        self.routes = routes or DEFAULT_ROUTES
        self.settings = settings or get_settings()
        self.min_similarity = min_similarity
        self.default_route = default_route
        self._matrix: FloatArray | None = None
        self._owners: list[str] = []
        self._lock = asyncio.Lock()

    async def _ensure_matrix(self) -> tuple[FloatArray, list[str]]:
        """Embed the exemplars once, under a lock.

        The lock matters: without it, concurrent first requests each embed the
        whole exemplar set, which on a cold start is the most expensive thing the
        process does.
        """
        if self._matrix is not None:
            return self._matrix, self._owners
        async with self._lock:
            if self._matrix is not None:
                return self._matrix, self._owners
            texts: list[str] = []
            owners: list[str] = []
            for name, exemplars in self.routes.items():
                for exemplar in exemplars:
                    texts.append(exemplar)
                    owners.append(name)
            vectors = await self.embedder.embed_documents(texts)
            matrix = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            self._matrix = (matrix / np.maximum(norms, 1e-9)).astype(np.float32)
            self._owners = owners
            log.debug("semantic_routes_embedded", routes=len(self.routes), exemplars=len(texts))
        return self._matrix, self._owners

    async def route(self, query: Query) -> tuple[RouteDecision, Usage]:
        matrix, owners = await self._ensure_matrix()
        if matrix.size == 0:
            return RouteDecision(
                stores=(), prompt_name=self.default_route, confidence=0.0, method="semantic"
            ), Usage()

        vector = np.asarray(
            query.dense if query.dense is not None else await self.embedder.embed_query(query.text),
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            vector = vector / norm

        similarities = matrix @ vector  # one matmul for every exemplar

        # Max per route, computed without a Python loop over exemplars.
        best: dict[str, float] = {}
        for name, score in zip(owners, similarities, strict=True):
            value = float(score)
            if value > best.get(name, -1.0):
                best[name] = value

        winner, score = max(best.items(), key=lambda kv: kv[1])
        if score < self.min_similarity:
            log.debug(
                "semantic_route_below_threshold",
                best=winner,
                score=round(score, 3),
                threshold=self.min_similarity,
            )
            winner, score = self.default_route, score

        decision = RouteDecision(
            stores=(),  # this router selects a prompt, not a store
            prompt_name=winner,
            confidence=float(score),
            reasoning=f"nearest exemplar similarity {score:.3f}",
            method="semantic",
        )
        # No LLM call, so no Usage: the point of this router is that it is free.
        return decision, Usage()
