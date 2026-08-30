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

Staleness is the third danger, and TTL alone does not cover it
--------------------------------------------------------------
An answer is a statement about the index *at the moment it was computed*. This
module already refused to cache an abstention for that reason — "serving one
later hides content that has since been added" — and the same sentence is true
of a positive answer with the sign flipped: serving it later shows content that
has since been corrected or removed. Reproduced against a live stack: a policy
edited from "30 days" to "7 days" and re-ingested was still answered "30 days",
because the ingest purge does not reach here.

TTL bounds that window but cannot close it, and an operator who has just fixed a
document does not want to wait an hour. So each entry records the documents its
answer was built from, and :meth:`SemanticCache.invalidate` drops every entry
that cites one. Without that field there is nothing to invalidate *by*: the old
payload was question, answer, timestamp, tenant and scope, so the only available
removal was :meth:`clear`, which drops the collection for every tenant at once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import orjson
import structlog

from ragorc.core.errors import StoreUnavailable
from ragorc.core.ids import content_hash
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["SemanticCache", "SemanticHit"]


def scope_key(filters: Any | None = None, top_k: int | None = None, *, pipeline: Any = None) -> str:
    """A stable identity for the retrieval scope a question was answered under.

    Two requests with the same text are not the same request. `filters` narrows
    which passages are admissible and `top_k` changes how many are used, so an
    answer produced under one scope is not the answer to the same question asked
    under another — and filters are how a caller restricts themselves to a subset
    they are entitled to see. Keying only on text and tenant served the first
    caller's answer to the second.

    Sorted before hashing so key order in a filter dict cannot split the cache.
    """
    normalized = orjson.dumps(filters or {}, option=orjson.OPT_SORT_KEYS).decode()
    # The pipeline belongs here for the same reason the other two do: ``graphrag``
    # and ``naive`` answer the same question differently on purpose, which is the
    # whole reason a caller names one. Without it, whichever ran first answered
    # for both — so a benchmark comparing two pipelines measured one of them twice.
    # Optional, so callers with no pipeline concept keep their existing keys.
    selected = getattr(pipeline, "value", pipeline) or ""
    return content_hash("semscope", normalized, str(top_k or ""), str(selected), size=16)


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
        self._init_lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def _ensure(self) -> Any:
        """Build the client and the collection, once, even under a cold fan-out.

        There are two awaits between the `_ready` check and the assignment, so two
        concurrent first requests both saw `_ready` false, both called
        `create_collection`, and the loser's write failed with
        `409 Collection already exists` — silently, since `set` degrades. Measured:
        two concurrent cold writes, one point stored.

        A lock rather than a `collection_exists` retry, because the race is the
        whole of the initialization and not just the create.
        """
        async with self._init_lock:
            return await self._ensure_locked()

    async def _ensure_locked(self) -> Any:
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

    async def get(
        self, question: str, *, tenant_id: str | None = None, scope: str | None = None
    ) -> SemanticHit | None:
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
            if scope:
                # Matched exactly, like tenant_id: a near-miss on the question is
                # the point of a semantic cache, a near-miss on the scope is a
                # wrong answer.
                conditions.append(
                    models.FieldCondition(key="scope", match=models.MatchValue(value=scope))
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
        scope: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> None:
        """Store one answer.

        ``document_ids`` is what makes the entry invalidatable — see
        :meth:`invalidate`. It is optional because a caller may not know the
        provenance, but an entry stored without it can only ever be removed by TTL
        or :meth:`clear`, so the pipeline always passes it.
        """
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

            point_id = content_hash("semcache", tenant_id or "", scope or "", question, size=16)
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
                            **({"scope": scope} if scope else {}),
                            # A list, matched with MatchAny on the way out. Stored
                            # even when empty is pointless, so it is omitted — an
                            # answer with no sources cites nothing to invalidate.
                            **({"document_ids": sorted(set(document_ids))} if document_ids else {}),
                        },
                    )
                ],
                wait=False,
            )
        except StoreUnavailable:
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("semantic_cache_set_failed", error=str(exc)[:200])

    async def invalidate(self, document_ids: Sequence[str], *, tenant_id: str | None = None) -> int:
        """Drop every cached answer built from one of these documents.

        Called from the ingest purge, so that correcting a document takes effect on
        the next query rather than at the end of the TTL. Returns the number of
        documents the removal was requested for, not the number of points removed:
        Qdrant's filtered delete does not report a count, and inventing one by
        counting first would double the round trips to produce a number no caller
        uses.

        Entries stored before this field existed, or by a caller that passed no
        provenance, carry no ``document_ids`` and are not matched. They expire by
        TTL as they always did. That is the honest behaviour — a filtered delete
        cannot match a field that is absent — and it is why the ingest path passes
        provenance unconditionally.
        """
        ids = sorted({d for d in document_ids if d})
        if not ids or not (self.settings.cache.enabled and self.settings.cache.semantic_enabled):
            return 0
        try:
            client = await self._ensure()
            from qdrant_client import models

            conditions: list[Any] = [
                models.FieldCondition(key="document_ids", match=models.MatchAny(any=ids))
            ]
            if tenant_id:
                # Scoped for the same reason `get` is: without it one tenant's
                # re-ingest would evict another tenant's answers. Harmless to
                # correctness, but it is their cache hit rate being spent.
                conditions.append(
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant_id))
                )
            await client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(filter=models.Filter(must=conditions)),
                wait=False,
            )
        except StoreUnavailable:
            return 0
        except Exception as exc:  # noqa: BLE001
            # Never fail an ingest because the answer cache could not be reached.
            # The cost of not invalidating is a stale answer for up to one TTL; the
            # cost of raising is a failed ingest, which is worse and is the thing
            # the operator was in the middle of fixing.
            log.warning(
                "semantic_cache_invalidate_failed", error=str(exc)[:200], documents=len(ids)
            )
            return 0
        log.info("semantic_cache_invalidated", documents=len(ids), tenant_id=tenant_id)
        return len(ids)

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
