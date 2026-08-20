"""Semantic answer cache.

The largest single cost lever in a production RAG service, and the most dangerous
one. Instead of hashing the question, it embeds it and looks for a *near*
neighbour among previously answered questions — so "what is our refund window"
can serve the answer computed for "how long do refunds take".

A hit skips the **entire pipeline**: no retrieval, no reranking, no synthesis.
That is why its hit rate matters more than the exact-cache's: an exact hit saves
one call, a semantic hit saves twenty.

The danger is symmetrical. Set the threshold too low and you confidently answer a
question nobody asked, which is worse than any cache miss. Default 0.97, and the
docstring on the setting says to be conservative for exactly this reason.

Storage is a small Qdrant collection rather than a bespoke index: it is already a
dependency, already does approximate nearest neighbour well, and gives TTL
expiry through payload filtering.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import orjson
import structlog

from ragorc.core.errors import StoreUnavailable
from ragorc.core.ids import content_hash
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["SemanticCache", "SemanticHit"]


class SemanticHit:
    __slots__ = ("answer", "question", "score", "stored_at")

    def __init__(self, answer: dict[str, Any], question: str, score: float, stored_at: float):
        self.answer = answer
        self.question = question
        self.score = score
        self.stored_at = stored_at


class SemanticCache:
    """Embedding-proximity cache over answered questions."""

    def __init__(
        self,
        embedder: Any,
        client: Any = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder
        self._client = client
        self.collection = self.settings.cache.semantic_collection
        self._ready = False
        self.hits = 0
        self.misses = 0

    async def _ensure(self) -> Any:
        if self._client is None:
            from ragorc.stores.qdrant.client import build_client

            self._client = build_client(self.settings.qdrant)
        if not self._ready:
            from qdrant_client import models

            dim = (
                getattr(self.embedder, "dimension", None) or self.settings.postgres.vector_dimension
            )
            existing = await self._client.collection_exists(self.collection)
            if not existing:
                await self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=int(dim), distance=models.Distance.COSINE
                    ),
                )
            self._ready = True
        return self._client

    async def get(self, question: str, *, tenant_id: str | None = None) -> SemanticHit | None:
        cfg = self.settings.cache
        if not (cfg.enabled and cfg.semantic_enabled):
            return None
        try:
            client = await self._ensure()
            vector = await self.embedder.embed_query(question)
            from qdrant_client import models

            conditions: list[Any] = []
            if tenant_id:
                # Tenant scoping is not optional here: a cache keyed only by
                # question text would leak one tenant's answer to another.
                conditions.append(
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
                )
            cutoff = time.time() - cfg.semantic_ttl_s
            conditions.append(models.FieldCondition(key="ts", range=models.Range(gte=cutoff)))

            result = await client.query_points(
                collection_name=self.collection,
                query=np.asarray(vector, dtype=np.float32).tolist(),
                limit=1,
                score_threshold=cfg.semantic_threshold,
                query_filter=models.Filter(must=conditions),
                with_payload=True,
            )
            points = getattr(result, "points", result) or []
        except Exception as exc:  # noqa: BLE001 - a cache miss, never an outage
            log.warning("semantic_cache_get_failed", error=str(exc)[:200])
            return None

        if not points:
            self.misses += 1
            return None
        point = points[0]
        payload = point.payload or {}
        self.hits += 1
        log.info(
            "semantic_cache_hit",
            score=round(float(point.score), 4),
            threshold=cfg.semantic_threshold,
            cached_question=str(payload.get("question", ""))[:80],
        )
        try:
            answer = orjson.loads(payload.get("answer") or b"{}")
        except orjson.JSONDecodeError:
            return None
        return SemanticHit(
            answer=answer,
            question=str(payload.get("question", "")),
            score=float(point.score),
            stored_at=float(payload.get("ts", 0.0)),
        )

    async def set(
        self,
        question: str,
        answer: dict[str, Any],
        *,
        tenant_id: str | None = None,
    ) -> None:
        cfg = self.settings.cache
        if not (cfg.enabled and cfg.semantic_enabled):
            return
        # Never cache an abstention. It is a statement about the index at one
        # moment, and serving it later hides content that has since been added.
        if answer.get("abstained"):
            return
        try:
            client = await self._ensure()
            vector = await self.embedder.embed_query(question)
            from qdrant_client import models

            point_id = content_hash("semcache", tenant_id or "", question, size=16)
            await client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=int(point_id[:16], 16) % (2**63),
                        vector=np.asarray(vector, dtype=np.float32).tolist(),
                        payload={
                            "question": question,
                            "answer": orjson.dumps(answer).decode(),
                            "ts": time.time(),
                            **({"tenant_id": tenant_id} if tenant_id else {}),
                        },
                    )
                ],
                wait=False,
            )
        except StoreUnavailable:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("semantic_cache_set_failed", error=str(exc)[:200])

    async def clear(self) -> None:
        try:
            client = await self._ensure()
            await client.delete_collection(self.collection)
            self._ready = False
        except Exception as exc:  # noqa: BLE001
            log.warning("semantic_cache_clear_failed", error=str(exc)[:200])

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "tier": "semantic",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "threshold": self.settings.cache.semantic_threshold,
        }
