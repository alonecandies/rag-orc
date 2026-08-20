# ADR-0009: Floors are the versions we verified; majors are capped

**Status:** accepted · **Date:** 2026-08-20

## Context

The base dependency list originally declared floors like `sqlglot>=25.24`,
`langgraph>=0.2.45` and `numpy>=1.26`. A fresh resolve produced sqlglot **30.17**,
langgraph **1.2.11** and numpy **2.5.2** — up to five major versions above the
declared minimum.

That is not permissiveness, it is an untested claim. `pip install ragorc` was free
to resolve sqlglot 25 with code written and tested against sqlglot 30, and nobody
had ever run that combination. The declared range described a compatibility
surface we had not verified and could not support.

Two of the gaps were worse than merely untested:

* **`sqlglot` is a security boundary.** The SQL guard
  (`ragorc/security/sql_guard.py`) matches on `sqlglot.exp` node classes —
  `exp.Insert`, `exp.Delete`, `exp.Into`, `exp.Command`. A rename or a
  re-parenting in a new major does not raise an `ImportError`; it produces a
  guard that silently stops matching, which is a hole in the allowlist rather
  than a crash. This is the one dependency where an unbounded range is a
  vulnerability.
* **`langgraph>=0.2.45` permitted the 0.x line.** The pipelines are written
  against the 1.x `StateGraph` and reducer API. A resolve to 0.2 would fail at
  import or, worse, at the first concurrent state write.

## Decision

**The floor is the version the test suite passes against. Majors are capped.**

| Dependency | Range | Why the cap |
|---|---|---|
`sqlglot` | `>=30.0,<31` | AST node names are a security boundary — see above |
`langgraph` | `>=1.0,<2` | the graphs use the 1.x StateGraph/reducer API |
`neo4j` | `>=6.0,<7` | the store is written against the 6.x driver surface |
`numpy` | `>=2.0,<3` | numpy 2 changed scalar promotion; the vectorized scoring is tested on 2.x |
`pydantic` | `>=2.9,<3` | v1 and v2 are different libraries |
`psycopg` | `>=3.3,<4` | binary + pipeline mode behaviour |
`qdrant-client` | `>=1.19,<2` | the client only guarantees compatibility within one minor of the server |
`fastembed` | `>=0.8,<1` | pre-1.0, so a minor bump can move the API |
others | floor + next major | ordinary caution |

`qdrant-client` carries an additional coupling: its range must stay in step with
the image tag in `docker-compose.yml`, because the client refuses to guarantee
compatibility when the server's minor drifts by more than one — and says so at
runtime before anything actually breaks.

A CI check asserts that the installed environment satisfies every declared range,
so a floor and a lockfile cannot drift apart again unnoticed.

## Consequences

- **The declared range is a claim we have tested**, which is the only kind worth
  making.
- Upgrading a capped dependency is a deliberate act: raise the cap, run the suite,
  and for `sqlglot` specifically re-run the adversarial corpus in
  `tests/unit/test_security_guards.py`, because that is what would catch a guard
  that stopped matching.
- Callers with a conflicting pin are blocked rather than silently given an
  untested combination. That is the intended trade for a library whose failure
  mode includes "the SQL guard no longer guards".
