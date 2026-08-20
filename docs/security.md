# Security

The threat model, the controls, and what is deliberately *not* claimed.

## Threat model

RAG systems have four attack surfaces, and the second and third are the ones that
get overlooked.

| Surface | Threat | Control |
|---|---|---|
**User query** | injection, resource exhaustion, PII in logs | validation, injection scan, rate limits, redaction |
**Generated SQL/Cypher** | arbitrary query → data exfiltration or destruction | AST validation, read-only transaction, `SELECT`-only role |
**Retrieved documents** | indirect prompt injection through indexed content | normalization, injection scan, structural isolation |
**Multi-tenant data** | one tenant's query returning another's documents | fail-closed tenant scoping |

## 1. Generated queries are the highest-severity surface

Text-to-SQL takes untrusted natural language, has a model turn it into code, and
executes it. Stated plainly, the feature is an arbitrary-query primitive.

**Three independent layers** ([ADR-0006](adr/0006-layered-query-guards.md)), so a
bug in one is not a breach:

### Layer 1 — parse, then decide

SQL is validated on a `sqlglot` **AST**. A statement type is a node type in the
tree, so it cannot be disguised by whitespace, casing, comments or nesting. Blocked
anywhere at any depth: `Insert`, `Update`, `Delete`, `Drop`, `Create`, `Alter`,
`Grant`, `Command` (COPY/VACUUM/SET), `Merge`, `Into`, `Lock`.

This is why AST validation is not optional:

```sql
-- A substring blocklist for a leading DELETE misses this entirely:
WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x
-- And this is a write wearing a SELECT's clothes:
SELECT * FROM orders INTO new_table
```

Also enforced: table allowlist, join ceiling, function blocklist
(`pg_read_file`, `dblink`, `pg_sleep`, `lo_import`, …), system-schema ban
(`pg_catalog`, `information_schema`), mandatory `LIMIT` (injected if absent,
clamped if too large), and rejection of NUL bytes and bidi control characters
(parser/executor differentials).

Cypher has no maintained Python parser, so validation is lexical — but
**normalized first**: string literals and comments are blanked before scanning, so
content can neither trigger a false positive nor hide a keyword.

```cypher
-- Correctly ALLOWED (a literal, not a clause):
MATCH (n) WHERE n.name = 'DELETE ME' RETURN n
-- Correctly BLOCKED (whitespace does not hide it):
MATCH (n) DETACH  DELETE n RETURN 1
-- Correctly BLOCKED (unbounded traversal is an outage):
MATCH (a)-[*]-(b) RETURN a, b
-- Correctly BLOCKED (SSRF):
CALL apoc.load.json('http://169.254.169.254/') YIELD value RETURN value
```

Procedures are **allowlisted**, not blocklisted: the set of dangerous procedures
is open-ended, the set we need is small.

### Layer 2 — read-only execution

Every generated statement runs inside `SET TRANSACTION READ ONLY` with a
server-side `statement_timeout` and a row cap.

### Layer 3 — a role that cannot write

```sql
CREATE ROLE ragorc_ro LOGIN PASSWORD '...';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ragorc_ro;
ALTER ROLE ragorc_ro SET default_transaction_read_only = on;
```

Set `RAGORC_POSTGRES__READONLY_DSN` to use it. If layers 1 and 2 both fail, the
connection still cannot mutate anything.

## 2. Indirect prompt injection through retrieved documents

Everyone guards the user input. Almost nobody guards the *documents* — yet those
are attacker-controllable in any system indexing uploads, scraped pages, emails,
tickets or wiki edits, and they land in the same prompt with the same authority as
your instructions.

Three defences:

1. **Normalization** — NFKC (collapses full-width and homoglyph variants) plus
   removal of zero-width, bidi-override and Unicode-tag characters. These are
   invisible to a human reviewer and to most logs, but the tokenizer sees them,
   which makes them the preferred carrier for hidden instructions.
2. **Detection** — patterns for instruction override, role hijack, prompt
   exfiltration, delimiter spoofing, tool abuse, markdown-image exfiltration
   channels, encoded payloads. Risk combines as a noisy-OR, so several weak
   signals outweigh one, and nothing saturates on a single heuristic.
3. **Structural isolation** — the load-bearing defence. Retrieved text is wrapped
   in delimiters whose closing tag is escaped inside the payload, so content
   cannot terminate its own container.

The default action is `sanitize`, not `block`. The patterns have real false
positives — a security wiki page *about* prompt injection matches all of them —
and isolation is what actually holds. Sanitizing keeps the document available as
evidence while defanging it.

```bash
RAGORC_SECURITY__INJECTION_ACTION=sanitize   # sanitize | block | flag
```

## 3. Multi-tenancy fails closed

```bash
RAGORC_SECURITY__ENFORCE_TENANT_ISOLATION=true
```

With this on, a query carrying no `tenant_id` is **rejected**, not silently
executed against every tenant's data. `scope_filter()` is the only sanctioned way
to build a store filter, and it raises on a conflicting `tenant_id` — an attempt
to override the scope is either a bug or an attack.

Qdrant additionally gets a payload index with `is_tenant=True`, which co-locates
each tenant's vectors on disk so the filter is cheap rather than a filtered scan.

### Generated SQL and Cypher cannot be scoped, so they refuse

Tenant isolation covers the vector store **by construction** — every filter is
built through `scope_filter()`, so a query cannot reach Qdrant unscoped.

It cannot cover *generated* SQL and Cypher the same way, and this is worth being
precise about because the gap was real and silent: `tenant_id` reached the
Text-to-SQL path only to *stamp the resulting chunk*, never to filter the query.
With `enforce_tenant_isolation=true`, the vector leg was correctly scoped while
the relational leg read every tenant's rows.

The reason it cannot be fixed by rewriting is structural. A generated statement
runs against **your** schema, and this library cannot know which column of
`orders` carries a tenant, whether the concept exists there at all, or — across
joins, CTEs, subqueries and set operations — where the predicate would have to go
to be correct. A rewriter that placed it subtly wrong would provide the
*appearance* of isolation, which is worse than refusing.

So those legs fail closed, and you declare how isolation is actually enforced:

```bash
RAGORC_SECURITY__GENERATED_QUERY_ISOLATION=reject    # default: refuse the legs
RAGORC_SECURITY__GENERATED_QUERY_ISOLATION=rls       # PostgreSQL row-level security
RAGORC_SECURITY__GENERATED_QUERY_ISOLATION=database  # database-per-tenant
RAGORC_SECURITY__GENERATED_QUERY_ISOLATION=trusted   # single-tenant, or handled upstream
```

**`rls` is the right answer** if you need multi-tenant Text-to-SQL. Row-level
security is enforced by PostgreSQL on every statement — including ones this
library never sees — which no amount of query rewriting can match:

```sql
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orders
  USING (tenant_id = current_setting('app.tenant_id', true));
```

Set `app.tenant_id` on the connection, and every generated query is scoped by the
database whether or not the model cooperated.

Single-tenant deployments are unaffected: with `enforce_tenant_isolation=false`
there is nothing to enforce and nothing is refused.

## 4. PII

Regex detection with checksum validation where the format allows it — a 16-digit
string is only a card number if it passes Luhn, and an IBAN only if mod-97 holds.
Validation is what keeps the false-positive rate low enough to enable redaction by
default.

Detectors: EMAIL, PHONE, CREDIT_CARD (Luhn), SSN, IBAN (mod-97), IP, AWS keys,
private keys, JWTs. Actions: `redact`, `hash` (stable, so entities stay joinable
after redaction), `flag`.

## 5. Secrets and logs

- All credentials are `SecretStr`; `Settings.summary()` is redaction-safe.
- The log processor redacts any key containing `api_key`, `password`, `token`,
  `secret`, `authorization`, `dsn`.
- `observability.log_prompts` is **off by default** — prompts contain retrieved
  customer data.
- The audit log records decisions, not content: generated statements, guard
  verdicts, injection signals, PII actions, cost.

## Hardening checklist for production

- [ ] `RAGORC_ENVIRONMENT=prod` (forces the guards on)
- [ ] `RAGORC_POSTGRES__READONLY_DSN` set to the `ragorc_ro` role
- [ ] `RAGORC_SERVER__API_KEYS` set — empty means open
- [ ] `RAGORC_SECURITY__ENFORCE_TENANT_ISOLATION=true` for multi-tenant data
- [ ] `RAGORC_LLM__DATA_COLLECTION=deny` so no provider trains on your documents
- [ ] `RAGORC_COST__MAX_COST_PER_QUERY_USD` set
- [ ] `postgres.allowed_tables` populated — an empty allowlist means every
      readable table
- [ ] TLS on all three stores; none of them are safe on a public network
- [ ] Neo4j password changed from the compose default
- [ ] `security.audit_log_path` set and shipped somewhere durable

## What is not claimed

- **Injection detection is heuristic.** It raises the cost of an attack; it does
  not eliminate the class. Structural isolation and least privilege are the real
  controls.
- **PII detection is regex-based.** It will miss names, addresses and free-text
  identifiers. Use Presidio or a dedicated NER model if you need coverage rather
  than best effort.
- **The guards do not sandbox the model.** They constrain what its output is
  permitted to *do*.
