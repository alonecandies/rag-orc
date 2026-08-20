"""Store doubles — in-memory, with the behaviours that matter for correctness.

These are not trivial dicts. Each reproduces the specific store behaviour that
tests need to assert on:

* ``FakeVectorStore`` does real cosine ranking over the stub vectors and honours
  filters, so a tenant-isolation test actually verifies isolation rather than
  verifying that a mock was called.
* ``FakeRelationalStore`` records the SQL it was handed, which is how a test
  proves the guard ran *before* execution.
* ``FakeGraphStore`` maintains an adjacency map so neighbour expansion and path
  finding produce real graph results — a mock returning a canned path cannot
  catch an off-by-one in hop counting.
* ``FakeCache`` counts hits and misses, so caching claims are testable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from ragorc.core.models import (
    Chunk,
    Community,
    Entity,
    GraphPath,
    Query,
    Relation,
    RetrievalSource,
    ScoredChunk,
)


def _matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Minimal filter dialect: eq, in, nin, ne, gte/gt/lte/lt."""
    if not filters:
        return True
    for key, condition in filters.items():
        if key in ("$and", "$or", "$not"):
            if key == "$and":
                if not all(_matches(payload, c) for c in condition):
                    return False
            elif key == "$or":
                if not any(_matches(payload, c) for c in condition):
                    return False
            elif _matches(payload, condition):
                return False
            continue
        value = payload.get(key)
        if isinstance(condition, dict):
            for op, operand in condition.items():
                if op in ("eq",) and value != operand:
                    return False
                if op == "ne" and value == operand:
                    return False
                if op == "in" and value not in operand:
                    return False
                if op == "nin" and value in operand:
                    return False
                if op == "gte" and not (value is not None and value >= operand):
                    return False
                if op == "gt" and not (value is not None and value > operand):
                    return False
                if op == "lte" and not (value is not None and value <= operand):
                    return False
                if op == "lt" and not (value is not None and value < operand):
                    return False
        elif value != condition:
            return False
    return True


class FakeVectorStore:
    """In-memory vector store with real cosine ranking."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.searches: list[dict[str, Any]] = []
        self.upserts = 0
        self.collection_created = False

    async def ensure_collection(self, *, recreate: bool = False) -> None:
        self.collection_created = True
        if recreate:
            self.chunks.clear()

    async def upsert(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        self.upserts += 1
        return len(chunks)

    async def search(
        self,
        query: Query,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScoredChunk]:
        self.searches.append(
            {"text": query.text, "filters": filters or query.filters, "top_k": top_k, **kwargs}
        )
        effective = filters if filters is not None else query.filters
        candidates = [c for c in self.chunks.values() if _matches(c.payload(), effective)]

        if query.dense is not None:
            q = np.asarray(query.dense, dtype=np.float32)
            q = q / max(float(np.linalg.norm(q)), 1e-9)
            scored: list[ScoredChunk] = []
            for chunk in candidates:
                if chunk.dense is None:
                    continue
                v = np.asarray(chunk.dense, dtype=np.float32)
                v = v / max(float(np.linalg.norm(v)), 1e-9)
                scored.append(
                    ScoredChunk(chunk=chunk, score=float(q @ v), source=RetrievalSource.DENSE)
                )
        else:
            # No query vector: fall back to lexical overlap so tests that do not
            # set up embeddings still get a sensible ordering.
            words = {w for w in query.text.lower().split() if len(w) > 2}
            scored = [
                ScoredChunk(
                    chunk=chunk,
                    score=len(words & set(chunk.content.lower().split())) / max(len(words), 1),
                    source=RetrievalSource.DENSE,
                )
                for chunk in candidates
            ]

        scored.sort(key=lambda s: s.score, reverse=True)
        limit = top_k or query.top_k or 10
        out = scored[:limit]
        for rank, item in enumerate(out):
            item.rank = rank
            item.component_scores = {"dense": item.score}
        return out

    async def get(self, ids: Sequence[str]) -> list[Chunk]:
        return [self.chunks[i] for i in ids if i in self.chunks]

    async def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> int:
        if ids is None:
            count = len(self.chunks)
            self.chunks.clear()
            return count
        removed = 0
        for i in ids:
            if self.chunks.pop(i, None) is not None:
                removed += 1
        return removed

    async def count(self, **kwargs: Any) -> int:
        return len(self.chunks)

    async def close(self) -> None:
        return None


class FakeRelationalStore:
    """Records executed SQL; returns canned rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, schema: str = "") -> None:
        self.rows = rows if rows is not None else [{"count": 3}]
        self._schema = schema or (
            "TABLE customers(id BIGINT PK, name TEXT, country TEXT, segment TEXT, arr_usd NUMERIC)\n"
            "TABLE orders(id BIGINT PK, customer_id BIGINT FK->customers.id, total_usd NUMERIC, status TEXT)"
        )
        self.executed: list[str] = []
        self.schema_refreshes = 0

    async def execute_readonly(
        self, sql: str, params: Sequence[Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        self.executed.append(sql)
        return self.rows[:limit]

    async def schema_summary(self, *, refresh: bool = False) -> str:
        if refresh:
            self.schema_refreshes += 1
        return self._schema

    async def fulltext_search(
        self, query: str, *, top_k: int = 10, **kwargs: Any
    ) -> list[ScoredChunk]:
        words = {w for w in query.lower().split() if len(w) > 2}
        out: list[ScoredChunk] = []
        for i, row in enumerate(self.rows[:top_k]):
            content = " ".join(str(v) for v in row.values())
            score = len(words & set(content.lower().split())) / max(len(words), 1)
            out.append(
                ScoredChunk(
                    chunk=Chunk(id=f"pg-{i}", content=content),
                    score=score,
                    source=RetrievalSource.FULLTEXT,
                    rank=i,
                )
            )
        return out

    async def close(self) -> None:
        return None


class FakeGraphStore:
    """In-memory graph with real neighbour expansion and BFS paths."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self.communities_list: list[Community] = []
        self.executed: list[str] = []
        self.chunk_links: dict[str, set[str]] = defaultdict(set)

    async def execute_readonly(
        self, cypher: str, params: dict[str, Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        self.executed.append(cypher)
        return [{"name": e.name, "type": e.type} for e in list(self.entities.values())[:limit]]

    async def schema_summary(self, *, refresh: bool = False) -> str:
        labels = sorted({e.type for e in self.entities.values()}) or ["Entity"]
        types = sorted({r.type for r in self.relations}) or ["RELATED_TO"]
        return f"Labels: {labels}\nRelationships: {types}\nProperties: name, description"

    async def upsert_entities(self, entities: Sequence[Entity]) -> int:
        for entity in entities:
            key = entity.key
            existing = self.entities.get(key)
            if existing is None:
                self.entities[key] = entity
            else:
                existing.aliases = tuple(set(existing.aliases) | set(entity.aliases))
                existing.source_chunk_ids = tuple(
                    set(existing.source_chunk_ids) | set(entity.source_chunk_ids)
                )
        return len(entities)

    async def upsert_relations(self, relations: Sequence[Relation]) -> int:
        for relation in relations:
            match = next((r for r in self.relations if r.key == relation.key), None)
            if match is None:
                self.relations.append(relation)
            else:
                match.weight += relation.weight
            self._adjacency[relation.source.casefold()].add(relation.target.casefold())
            self._adjacency[relation.target.casefold()].add(relation.source.casefold())
        return len(relations)

    async def link_chunks(self, entity_name: str, chunk_ids: Sequence[str]) -> None:
        self.chunk_links[entity_name.casefold()].update(chunk_ids)

    async def neighbors(
        self, names: Sequence[str], *, hops: int = 1, limit: int = 50
    ) -> tuple[list[Entity], list[Relation]]:
        frontier = {n.casefold() for n in names}
        seen = set(frontier)
        for _ in range(max(hops, 0)):
            nxt: set[str] = set()
            for node in frontier:
                nxt |= self._adjacency.get(node, set())
            frontier = nxt - seen
            seen |= frontier
            if not frontier:
                break
        entities = [self.entities[k] for k in seen if k in self.entities][:limit]
        relations = [
            r for r in self.relations if r.source.casefold() in seen and r.target.casefold() in seen
        ][:limit]
        return entities, relations

    async def paths(
        self, start: Sequence[str], end: Sequence[str], *, max_hops: int = 3, limit: int = 10
    ) -> list[GraphPath]:
        targets = {n.casefold() for n in end}
        results: list[GraphPath] = []
        for origin in start:
            queue: list[list[str]] = [[origin.casefold()]]
            while queue and len(results) < limit:
                path = queue.pop(0)
                if len(path) - 1 > max_hops:
                    continue
                node = path[-1]
                if node in targets and len(path) > 1:
                    relations = tuple(
                        r
                        for a, b in pairwise(path)
                        for r in self.relations
                        if {r.source.casefold(), r.target.casefold()} == {a, b}
                    )
                    weight = sum(r.weight for r in relations) or 1.0
                    results.append(
                        GraphPath(
                            nodes=tuple(
                                self.entities[n].name if n in self.entities else n for n in path
                            ),
                            relations=relations,
                            score=weight / max(len(path) - 1, 1),
                        )
                    )
                    continue
                for neighbour in sorted(self._adjacency.get(node, set())):
                    if neighbour not in path:
                        queue.append([*path, neighbour])
        results.sort(key=lambda p: p.score, reverse=True)
        return results[:limit]

    async def communities(self, *, level: int | None = None) -> list[Community]:
        if level is None:
            return list(self.communities_list)
        return [c for c in self.communities_list if c.level == level]

    async def close(self) -> None:
        return None


class FakeCache:
    """Dict cache that counts hits and misses."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> bytes | None:
        value = self.store.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    async def set(self, key: str, value: bytes, *, ttl: float | None = None) -> None:
        self.store[key] = value

    async def delete(self, key: str) -> None:
        self.store.pop(key, None)

    async def clear(self, prefix: str | None = None) -> int:
        if prefix is None:
            count = len(self.store)
            self.store.clear()
            return count
        keys = [k for k in self.store if k.startswith(prefix)]
        for k in keys:
            del self.store[k]
        return len(keys)
