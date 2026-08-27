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
   in delimiters, and every spelling of the fence tag inside the payload is
   escaped — case, stray whitespace, and the *opening* tag as well as the
   closing one, since escaping only the closing tag stops break-out and leaves
   boundary forgery. A passage's provenance line (`source: …`) is rendered
   **inside** the fence, because it comes from `metadata` on the ingest request
   and is therefore attacker-controlled; only the citation number `[n]`, which
   the packer generates by counting, sits above it.

Where the scan runs, and where it does not:

| text | scanned | by |
|---|---|---|
| the user's question | yes | `validate/input.py`, stricter — a *question* has no legitimate reason to contain a role switch |
| an ingested document | yes, at ingest | `validate/schema.py` |
| a web search result | yes, at retrieval | `retrieve/web.py` |

Documents are scanned at **ingest** rather than at retrieval: a document is
written once and read many times, so the cost is paid per document, and `block`
can only mean anything at the door. A blocked document is reported in
`IngestReport.rejected` and the rest of the batch still indexes — one poisoned
file in a bulk upload must not reject the other 9 999.

The default action is `sanitize`, not `block`. The patterns have real false
positives — a security wiki page *about* prompt injection matches all of them —
and isolation is what actually holds. Sanitizing keeps the document available as
evidence while defanging it. Normalization applies either way, so a zero-width
space inside a word never reaches the index; that is a retrieval fix as much as a
security one, since it silently breaks every lexical match on that word.

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

### The knowledge graph cannot be scoped, so it refuses

```bash
RAGORC_SECURITY__GRAPH_TENANT_ISOLATION=reject    # default
RAGORC_SECURITY__GRAPH_TENANT_ISOLATION=trusted   # one graph per tenant
```

Nothing in Neo4j carries a tenant. Entities merge on `name`, communities on a
membership hash, chunk links on a chunk id — none namespaced — so two tenants
writing about the same company converge on one node and a traversal from it
reaches both. With `reject` (the default) local, global, DRIFT and bridge search
all refuse while `enforce_tenant_isolation` is on, `auto` will not select
`graphrag`, and `/health` says so.

`trusted` asserts that the graph holds one tenant's data — a Neo4j instance or
database per tenant. It is an explicit assertion because the consequence of it
being untrue is that entity names, descriptions and relationships cross tenants
in the verbalized subgraph, *stamped with the querying tenant's id*.

There is deliberately no `rls`-equivalent. Making the graph multi-tenant means
entity identity becomes `(tenant, name)`, which changes every MERGE, every
traversal predicate and the fulltext index, and needs a migration for graphs
already built. A filter added at query time would provide the appearance of
isolation.

By-id chunk reads (`QdrantStore.get`, `PostgresStore.get_chunks`) are scoped
independently, so a foreign chunk id resolves to nothing even under `trusted`.
An id is not a filter, and those two reads were the one unscoped door.

### A retriever this library does not own

```bash
RAGORC_SECURITY__FOREIGN_RETRIEVER_TENANT_ISOLATION=filter
```

`from_langchain_retriever` makes someone else's retriever one leg of an
ensemble. That leg queries its own store, so no filter of ours reaches it, and
the `tenant_id` on what comes back is read out of the foreign document's own
metadata — the retriever's claim, not a fact. With isolation on and this left at
`reject`, an ensemble holding one adapted leg answered a query scoped to
`globex` with a chunk whose own `tenant_id` was `acme`.

`reject` (the default) refuses the leg. `filter` passes the tenant and filters
down in `config["metadata"]`, so a retriever capable of scoping itself can, and
then keeps only the chunks that **declare** the querying tenant. An unlabelled
chunk is dropped rather than stamped: an absent label is not a match, and
stamping it would forge the provenance the paragraph above exists to prevent.
`trusted` asserts the wrapped retriever holds one tenant's data and skips the
filter — necessary, because a genuinely single-tenant retriever's documents
usually carry no tenant label at all, and filtering would drop all of them.

Unlike the graph there is a middle ground here because the decision needs no
schema we do not control: a label that is present can be checked, and one that
is absent can be dropped.

### Naming a tenant is not the same as owning one

`tenant_id` is a field on the request **body**, so on its own the setting above
enforces that a request *names* a tenant — not that the caller is entitled to it.
Bind each key to its tenant:

```bash
RAGORC_SERVER__API_KEYS='["key-acme","key-globex"]'
RAGORC_SERVER__API_KEY_TENANTS='{"key-acme":"acme","key-globex":"globex"}'
```

A bound key may omit `tenant_id` (its own is used) and is refused if it names
another (`rule=tenant_not_owned`, HTTP 400 — deliberately not 403, which would
confirm to a caller probing ids which tenants exist). Reads and writes are both
bound, since a caller able to file documents under another tenant's id is a
caller able to have them answered back to that tenant.

A key **absent** from the map stays unrestricted, so an existing deployment
keeps working across the upgrade. That leaves the door open by default, which is
why `/health` says so out loud whenever isolation is on and nothing is bound:

> `security.enforce_tenant_isolation is on but server.api_key_tenants is empty:
> any authenticated caller can read any tenant by naming it`

That configuration is self-consistent — every query names a tenant, every store
filter carries it — which is exactly why it has to be stated rather than
inferred.

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

## 4. Server-side paths are opt-in, and the opt-in is an environment variable

Two endpoints accept a path for the *service* to open: `POST /ingest` (`paths[]`)
and `POST /eval` (`dataset`). Both compose badly with the read-back channel —
`/query` returns chunk text verbatim in `QueryResponse.chunks[]`, and an eval
response quotes the dataset — so an unconfined path is an arbitrary local file
read with a way to get the bytes back out, reachable by anyone who can open the
port (`server.api_keys` is empty by default).

Both are confined to `RAGORC_INGEST_ROOTS`, an OS-path-separated list:

```bash
export RAGORC_INGEST_ROOTS=/srv/corpus:/srv/datasets
```

**An empty allowlist refuses every server-side path** rather than allowing every
path, and unset is the default every deployment gets. Inline `text` and multipart
uploads carry their own bytes and are unaffected, which is why refusing costs the
common deployment nothing.

It is an environment variable rather than a setting on purpose: it belongs with
`ReadOnlyPaths=` in a unit file, not in the RAG configuration tree. A confinement
boundary a request body could reach — settings are echoed by `/health`, and
`Settings` is constructible from any dict — is not a boundary.

Both paths are also size-bounded, because both are read whole into a process that
is serving queries at the same time.

## 5. PII

Regex detection with checksum validation where the format allows it — a 16-digit
string is only a card number if it passes Luhn, and an IBAN only if mod-97 holds.
Validation is what keeps the false-positive rate low enough to enable redaction by
default.

Detectors: EMAIL, PHONE, CREDIT_CARD (Luhn), SSN, IBAN (mod-97), IP, AWS keys,
private keys, JWTs. Actions: `redact`, `hash` (stable, so entities stay joinable
after redaction), `flag`.

## 6. Secrets and logs

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
- [ ] `RAGORC_SERVER__API_KEY_TENANTS` set — without it, isolation only checks
      that a tenant was *named*, and any authenticated caller can name any tenant
- [ ] `RAGORC_SECURITY__GRAPH_TENANT_ISOLATION` left at `reject` unless you run
      one graph per tenant — the graph stores no tenant and cannot be filtered
- [ ] `RAGORC_SERVER__MAX_BODY_BYTES` sized for your largest upload — it is
      enforced on the wire, so it bounds a chunked request that declares no
      `Content-Length`
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

## What the answer is not allowed to carry out

Two checks run on every answer, and both **remove** rather than report:

* **Personal data.** `enable_pii_redaction` used to scrub the inbound question
  and nothing else. The question is written by the caller, who already knows
  what is in it; the answer is assembled from retrieved documents, which is
  where the corpus's personal data lives. It now applies to both.
* **Prompt scaffolding.** A model echoing `</untrusted_document>` gets the
  delimiter stripped. Not abstained on — the echo is usually a document being
  quoted, and refusing would turn a formatting artifact into an outage — but
  removed, which closes the case where the echo is an attempt to forge a fence
  in the next turn's context.

`answer.metadata["validation"]` reports `pii_redacted` and `scaffold_leak`, so a
caller can tell that an answer was rewritten.

Both run on `/query/stream` too. Groundedness genuinely cannot: it needs the
whole answer, and you cannot un-emit a token. These are regexes over emitted
text, so the only thing they need is a 96-character held-back tail so a pattern
split across two deltas is still seen whole. The bound is real — a construction
longer than the window can still straddle it, and the complete-answer path is
the one to use when that matters.
