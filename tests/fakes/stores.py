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
* ``FakeDocumentStore`` enforces the chunks-to-documents foreign key and its
  cascade, so a test can tell a correct ingest write *order* from an incorrect
  one — which a double that accepts anything cannot.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from itertools import pairwise
from typing import Any

import numpy as np
from psycopg.errors import ForeignKeyViolation

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
        elif isinstance(condition, (list, tuple, set, frozenset)):
            # A bare sequence means any-of, which is what the real store does:
            # `_condition` in `stores/qdrant/filters.py` turns it into `MatchAny`.
            # Treating it as equality here made `filters={"document_id": [...]}` —
            # the shape the purge and the delete both use — match nothing, so a
            # test could watch a delete remove zero rows and call it correct.
            if value not in condition:
                return False
        elif value != condition:
            return False
    return True


class FakeVectorStore:
    """In-memory vector store with real cosine ranking."""

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.get_calls: list[tuple[list[str], bool]] = []
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

    async def get(
        self,
        ids: Sequence[str],
        *,
        with_vectors: bool = False,
        tenant_id: str | None = None,
    ) -> list[Chunk]:
        """Chunk bodies by id, modelling Qdrant's ``with_vectors`` and its scope.

        The flag is honoured rather than accepted-and-ignored, because ignoring it
        makes a whole class of bug untestable: production code that stops asking
        for vectors gets them anyway from the fake, so a change that silently
        deletes a similarity term against real Qdrant leaves every test green.
        Asked without vectors, a real point comes back without them.

        ``tenant_id`` is enforced for the same reason, and it is the newer of the
        two lessons: a by-id read that accepted a tenant and ignored it would let
        the cross-tenant leak this parameter exists to close reappear with every
        test still passing.
        """
        self.get_calls.append((list(ids), with_vectors))
        found = [self.chunks[i] for i in ids if i in self.chunks]
        if tenant_id is not None:
            found = [c for c in found if c.tenant_id == tenant_id]
        if with_vectors:
            return found
        return [replace(chunk, dense=None, sparse=None, multi=None) for chunk in found]

    async def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        **kwargs: Any,
    ) -> int:
        """Delete by id or by filter, honouring the tenant either way.

        The real store routes a bare id list through a filter when isolation is on
        — ``HasIdCondition`` ANDed with the tenant — precisely so one tenant cannot
        delete another's point by guessing a deterministic id. A double that
        ignored ``tenant_id`` would let that argument be dropped at a call site
        with every test still green, which is the shape of the round-nine leaks.
        """
        del kwargs
        if ids is None and filters is None:
            if tenant_id:
                doomed = [k for k, c in self.chunks.items() if c.tenant_id == tenant_id]
            else:
                doomed = list(self.chunks)
        else:
            candidates = list(ids) if ids is not None else list(self.chunks)
            doomed = [
                cid
                for cid in candidates
                if (chunk := self.chunks.get(cid)) is not None
                and (not tenant_id or chunk.tenant_id == tenant_id)
                and _matches(chunk.payload(), filters)
            ]
        for cid in doomed:
            self.chunks.pop(cid, None)
        return len(doomed)

    async def scroll(
        self,
        *,
        filters: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[Chunk]:
        """Iterate the collection, filtered the same way ``search`` is.

        Modelled because the delete path reads chunk ids through it — the graph's
        only handle on a document — so a fake without it would leave that read
        untested at exactly the point it matters.
        """
        del kwargs, with_vectors
        for chunk in list(self.chunks.values()):
            if tenant_id and chunk.tenant_id != tenant_id:
                continue
            if _matches(chunk.payload(), filters):
                yield chunk

    async def count(self, **kwargs: Any) -> int:
        return len(self.chunks)

    async def health(self) -> dict[str, Any]:
        """Reachability, with no tenant in the question.

        Modelled separately from ``count`` rather than delegating to it, because
        the distinction is the point: the real store's ``count`` applies the
        tenant filter and fails closed, which is why a health probe must not be a
        count. A fake that answered both from one code path could not tell the two
        apart.
        """
        return {"collection": "fake", "points": len(self.chunks), "status": "green"}

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


class FakeDocumentStore:
    """The two ingest tables, including the constraint between them.

    ``chunks.document_id REFERENCES documents (id) ON DELETE CASCADE`` is the
    reason the ingest pipeline has a write *order* at all, so a double that
    accepts chunks for documents it has never seen cannot tell a correct order
    from an incorrect one. That is how writes the real schema rejects outright —
    and a purge that cascaded away rows written moments earlier — passed the whole
    unit suite.

    A falsy ``document_id`` stores a NULL FK and is not checked, matching
    ``PostgresStore.upsert_chunks``: a RAPTOR summary and a synthetic proposition
    have no owning document row.
    """

    def __init__(self, *, checksums: dict[str, str] | None = None) -> None:
        self.documents: dict[str, Any] = {}
        self.chunks: dict[str, Any] = {}
        self.calls: list[str] = []
        self._seeded_checksums = dict(checksums or {})
        self.schema_ensured = 0

    async def ensure_schema(self) -> None:
        self.schema_ensured += 1

    async def upsert_documents(self, documents: Sequence[Any]) -> int:
        self.calls.append("upsert_documents")
        for doc in documents:
            self.documents[doc.id] = doc
        return len(documents)

    async def upsert_chunks(self, chunks: Sequence[Any]) -> int:
        self.calls.append("upsert_chunks")
        for chunk in chunks:
            owner = getattr(chunk, "document_id", None)
            if owner and owner not in self.documents:
                raise ForeignKeyViolation(
                    f'insert or update on table "chunks" violates foreign key '
                    f'constraint "chunks_document_id_fkey": key (document_id)=({owner}) '
                    f'is not present in table "documents"'
                )
            self.chunks[chunk.id] = chunk
        return len(chunks)

    async def delete_document(self, document_id: str, *, tenant_id: str | None = None) -> int:
        """Delete the row and cascade to its chunks, as the FK declares.

        Tenant-scoped like the real store, so a test can see a foreign delete match
        nothing rather than succeed. A double that ignores the scope would let the
        argument be dropped at the call site without a single test noticing, which
        is the failure mode that produced the round-nine leaks.
        """
        self.calls.append("delete_document")
        owner = self.documents.get(document_id)
        if tenant_id and owner is not None and getattr(owner, "tenant_id", None) != tenant_id:
            return 0
        self.documents.pop(document_id, None)
        cascaded = [
            cid
            for cid, c in self.chunks.items()
            if getattr(c, "document_id", None) == document_id
            and (not tenant_id or getattr(c, "tenant_id", None) == tenant_id)
        ]
        for cid in cascaded:
            del self.chunks[cid]
        return len(cascaded)

    async def execute_readonly(
        self, sql: str, params: Sequence[Any] | None = None, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Answers the checksum probe from what has actually been written.

        The probe joins documents to chunks on purpose — a document counts as
        ingested only once it has chunks — so this mirrors the join rather than
        returning every seeded row.
        """
        self.calls.append("execute_readonly")
        with_chunks = {getattr(c, "document_id", None) for c in self.chunks.values()}
        out: list[dict[str, Any]] = []
        for doc_id, doc in self.documents.items():
            if doc_id not in with_chunks:
                continue
            checksum = getattr(doc, "checksum", None) or self._seeded_checksums.get(doc_id)
            if checksum:
                out.append({"id": doc_id, "checksum": checksum})
        return out[:limit]

    async def close(self) -> None:
        return None
