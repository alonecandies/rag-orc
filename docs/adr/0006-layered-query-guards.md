# ADR-0006: Generated SQL and Cypher are guarded in three independent layers

**Status:** accepted · **Date:** 2026-08-19

## Context

Text-to-SQL and Text-to-Cypher take untrusted natural language, have a language
model turn it into code, and then execute that code against a database. Stated
plainly, the feature is an arbitrary-query primitive driven by user input.

Single-layer defences fail predictably. A substring blocklist misses
`WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x`, and it produces false
positives on `WHERE note = 'DELETE ME'`. Relying on the model's cooperation is
not a control at all.

## Decision

Three layers that fail independently, so a bug in one is not a breach:

**Layer 1 — parse, then decide.** SQL is validated on a `sqlglot` AST: statement
type must be a read, and no write/DDL/utility node may appear at *any* depth. A
statement type is a node type in the tree, so it cannot be disguised by
whitespace, comments, casing or nesting. Additional AST-level rules: table
allowlist, join ceiling, function blocklist (`pg_read_file`, `dblink`,
`pg_sleep`, …), system-schema ban, mandatory `LIMIT` (injected if absent, clamped
if too large).

Cypher has no maintained Python parser, so validation is lexical but
*normalized first*: string literals and comments are blanked before scanning, so
content cannot trigger a false positive or hide a keyword. Cypher-specific rules:
variable-length patterns must have an upper bound (`*1..3`, never `*`), the hop
count is capped, and procedures are **allowlisted** rather than blocklisted —
the set of dangerous procedures is open-ended, the set we need is small.

**Layer 2 — execute read-only.** Every generated statement runs in a transaction
opened with `SET TRANSACTION READ ONLY`, under a server-side `statement_timeout`,
with a row cap.

**Layer 3 — a role that cannot write.** Postgres connections for generated SQL
use `ragorc_ro`, which holds `SELECT` only and has `default_transaction_read_only`
set on the role itself.

Beyond the query guards, retrieved **documents** are treated as untrusted input:
scanned for prompt injection, NFKC-normalized with invisible/bidi characters
stripped, and structurally isolated in delimiters that their own content cannot
break out of.

## Consequences

- A `GuardrailViolation` is never retried. Retrying a blocked statement produces
  the same block; the pipeline instead reports the refusal or falls back to
  another store.
- The guards are unit-tested against a concrete attack corpus (statement
  stacking, CTE-wrapped DML, `SELECT ... INTO`, system catalogs, SSRF via
  `apoc.load.json`, `LOAD CSV` file reads, unbounded traversal). A guard without
  adversarial tests is a guess.
- Prompt-injection detection deliberately defaults to `sanitize`, not `block`:
  the patterns have false positives (a security wiki page *about* prompt
  injection matches all of them), and structural isolation is the load-bearing
  defence anyway.
