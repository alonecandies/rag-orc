# `ragorc.construct` — natural language into SQL, Cypher and filters

Three constructors, one contract: **the artifact is validated before it exists as a
runnable string, and execution is a separate call.** That separation is the security
boundary — the only object carrying executable SQL or Cypher is the guard's own
output, so nothing can reach a datastore without having passed a guard.

Related: [ADR-0006 — layered query guards](../adr/0006-layered-query-guards.md) ·
[`ragorc.security`](security.md).

## Key classes

```python
TextToSQLConstructor(llm, store: RelationalStore | None = None, *,
                     guard=None, audit=None, router=None, settings=None)
    name = "text_to_sql"; target = "sql"
    async construct(query, store=None, **kw) -> tuple[SQLValidation, Usage]
    async construct_and_execute(query, store=None) -> tuple[list[dict], SQLValidation, Usage]
    def to_chunks(rows, ...) -> list[ScoredChunk]

TextToCypherConstructor(llm, store: GraphStore | None = None, *,
                        guard=None, audit=None, router=None, settings=None)
    name = "text_to_cypher"; target = "cypher"
    async construct(query, store=None, **kw) -> tuple[CypherValidation, Usage]
    async construct_and_execute(query, store=None) -> tuple[list[dict], CypherValidation, Usage]

SelfQueryConstructor(llm, attributes: Sequence[AttributeInfo] = (), *, router=None, settings=None)
    name = "self_query"; target = "filter"
    async construct(query, **kw) -> tuple[SelfQueryResult, Usage]
    async apply(query, **kw) -> tuple[Query, SelfQueryResult, Usage]

AttributeInfo(name, type="string", description="", examples=())
SelfQueryResult(query_text, filters, dropped, coerced, combinator)   # .has_filters, .report()
build_constructor(name, llm, store=None, **kw)
```

## Why `construct_and_execute` runs `validation.sql`, never the model's output

The guard does not only accept or reject — it *rewrites*: it injects or clamps a
`LIMIT`, normalizes the statement, and returns that. Executing the raw generation
because it "passed validation" throws away the rewrite, which is where the row cap
lives. Both constructors also state the ceiling in the prompt, so the common case
validates without a rewrite warning and the query the model wrote is the query that
runs.

Both retry once on a guard rejection, with the violation fed back to the model, and
raise `ConstructionError` if the repair is also rejected. `GuardrailViolation`
itself is never retried — a blocked statement is still blocked.

## Why `AttributeInfo.examples` matters

Self-query extracts metadata constraints out of a question ("refunded orders from
Germany last quarter" → filter plus a smaller semantic query). Given three real
values per field, a model stops inventing plausible ones — the difference between
`status eq "Delivered"` and `status eq "delivered"`, only one of which matches
anything. With no attributes at all the constructor passes the query through
untouched rather than asking the model to invent fields.

## Usage

```python
from ragorc.construct import AttributeInfo, SelfQueryConstructor, TextToSQLConstructor
from ragorc.core.settings import Settings

# Refused by default. A generated query is only bounded by the guard, which says
# nothing about *whose* rows it reads, so it will not run until the deployment
# declares how tenants are isolated. See docs/security.md.
settings = Settings(security={"generated_query_isolation": "rls"})

sql = TextToSQLConstructor(llm, postgres, settings=settings)
rows, validation, usage = await sql.construct_and_execute(Query(text="ARR by segment"))
print(validation.sql, validation.tables, validation.has_limit)

selfq = SelfQueryConstructor(
    llm,
    [
        AttributeInfo("segment", "string", "customer size", ("enterprise", "mid_market", "smb")),
        AttributeInfo("country", "string", "ISO-2 country", ("US", "GB", "DE")),
    ],
)
narrowed, result, usage = await selfq.apply(Query(text="enterprise accounts in the UK"))
print(narrowed.text, narrowed.filters, result.dropped)
```

`result.dropped` is surfaced, not swallowed: an empty result set caused by a
constraint the constructor rejected is otherwise undiagnosable.

## Settings

| Setting | Effect |
|---|---|
`security.enable_sql_guard` · `sql_allow_statements` · `sql_forbid_functions` · `sql_max_joins` · `sql_require_limit` | the SQL AST guard |
`security.enable_cypher_guard` · `cypher_forbid_keywords` · `cypher_explain_dryrun` | the Cypher guard |
`postgres.allowed_tables` · `max_sql_rows` · `statement_timeout_ms` · `readonly_dsn` | execution fence |
`neo4j.max_cypher_rows` · `query_timeout_s` | Cypher row and wall-clock caps |
`graph.multihop_max_path_length` | doubles as the guard's `max_hops` ceiling |
`llm.fast_model` | all three constructors run on `Task.SELF_QUERY` (the fast tier) |
`llm.strong_model` | the SQL repair attempt escalates to it — a second failure is worth one expensive call |
`cache.cache_schema` | schema introspection is cached, not re-read per query |
`security.audit_log_enabled` | every generated statement is recorded, accepted or blocked |
