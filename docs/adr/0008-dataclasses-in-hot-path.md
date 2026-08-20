# ADR-0008: Dataclasses in the hot path, pydantic at the boundaries

**Status:** accepted · **Date:** 2026-08-19

## Context

pydantic v2 is excellent and its validator is Rust. It is still not free: model
construction validates every field. The objects in `core/models.py` are created
in bulk — millions of `Chunk`s during an ingest, thousands of `ScoredChunk`s per
query — and their fields come from *our own code*, not from user input, so
validation is re-checking invariants we already hold.

## Decision

Split by trust boundary.

**`@dataclass(slots=True)`** for `Document`, `Chunk`, `ScoredChunk`, `Query`,
`Entity`, `Relation`, `Community`, `Usage`, `Answer`. Roughly 2-3x faster to
instantiate, and `__slots__` removes the per-instance `__dict__` for ~40% less
memory per object — which is what decides whether a 1M-chunk ingest fits in RAM.

**pydantic `BaseModel`** wherever data crosses a trust boundary and validation
is the point:

- `core/settings.py` — environment input, needs coercion and defaults.
- `core/schemas.py` — LLM output. Validation here *is* the feature: it is what
  makes `structured()` reliable, and field descriptions become part of the
  prompt.
- `server/` — HTTP request and response bodies.

Vectors are `numpy` arrays, never `list[float]`: a 1024-dim float32 array is
4 KiB against ~40 KiB for the equivalent Python list.

## Consequences

- Classes holding arrays are declared `eq=False` with an explicit `__hash__` on
  id. numpy's elementwise `==` makes a generated `__eq__` raise on truthiness
  testing, and identity-by-id is the semantics we actually want.
- Dataclasses do not coerce, so constructors are the enforcement point: `Chunk`
  requires the caller to have already produced correct types. Ingest input passes
  through `validate/schema.py` first, which is where coercion belongs.
- Serialization is explicit (`payload()` / `from_payload()`) rather than
  automatic. That is a feature for the vector store, where the payload shape is a
  storage contract that should not silently change when a field is added.
