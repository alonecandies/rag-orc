"""Datastore bindings: Qdrant (vector), Postgres (relational), Neo4j (graph).

Imported lazily per store, because each pulls a driver and constructing a client
opens connections. A project that only uses the vector store should not pay for
the Postgres and Neo4j drivers on import.
"""

from __future__ import annotations

__all__ = ["Neo4jStore", "PostgresStore", "QdrantStore"]


def __getattr__(name: str):  # noqa: ANN202
    if name == "QdrantStore":
        from ragorc.stores.qdrant.store import QdrantStore

        return QdrantStore
    if name == "PostgresStore":
        from ragorc.stores.postgres.store import PostgresStore

        return PostgresStore
    if name == "Neo4jStore":
        from ragorc.stores.neo4j.store import Neo4jStore

        return Neo4jStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
