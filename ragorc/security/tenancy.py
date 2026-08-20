"""Multi-tenant isolation.

The failure mode this prevents is the worst one a RAG system has: answering
tenant A's question with tenant B's documents. It is easy to hit, because the
default behaviour of every vector store is to search *everything*, and a missing
filter is silent — the query still returns plausible results.

So isolation is enforced by construction:

* ``require_tenant`` refuses a query with no tenant when isolation is on. Failing
  closed is the only safe default; a missing tenant must never mean "all".
* ``scope_filter`` injects the tenant predicate into every store filter, and is
  the *only* sanctioned way to build one.
* Qdrant additionally gets a payload index with ``is_tenant=True`` so the filter
  is cheap (co-located storage) rather than a filtered scan.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragorc.core.errors import GuardrailViolation
from ragorc.core.settings import SecuritySettings, Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = [
    "require_generated_query_isolation",
    "require_tenant",
    "scope_cypher_where",
    "scope_filter",
    "scope_sql_where",
]


def require_tenant(tenant_id: str | None, settings: SecuritySettings | None = None) -> str | None:
    s = settings or get_settings().security
    if not s.enforce_tenant_isolation:
        return tenant_id
    if not tenant_id:
        raise GuardrailViolation(
            "tenant_id is required when tenant isolation is enabled",
            rule="tenant_required",
            hint="pass tenant_id on the Query, or set security.enforce_tenant_isolation=false",
        )
    return tenant_id


def scope_filter(
    filters: dict[str, Any] | None, tenant_id: str | None, settings: SecuritySettings | None = None
) -> dict[str, Any]:
    """Return a filter dict with the tenant predicate applied."""
    tenant = require_tenant(tenant_id, settings)
    out = dict(filters or {})
    if tenant:
        existing = out.get("tenant_id")
        if existing is not None and existing != tenant:
            # A caller trying to override the scope is either a bug or an attack.
            raise GuardrailViolation(
                "conflicting tenant_id in filter",
                rule="tenant_conflict",
                requested=str(existing),
                actual=tenant,
            )
        out["tenant_id"] = tenant
    return out


def scope_sql_where(tenant_id: str | None, *, table_alias: str = "") -> tuple[str, list[Any]]:
    """SQL predicate plus params. Parameterized, never interpolated."""
    tenant = require_tenant(tenant_id)
    if not tenant:
        return "TRUE", []
    prefix = f"{table_alias}." if table_alias else ""
    return f"{prefix}tenant_id = %s", [tenant]


def scope_cypher_where(tenant_id: str | None, *, var: str = "n") -> tuple[str, dict[str, Any]]:
    tenant = require_tenant(tenant_id)
    if not tenant:
        return "true", {}
    return f"{var}.tenant_id = $tenant_id", {"tenant_id": tenant}


def require_generated_query_isolation(target: str, settings: Settings | None = None) -> None:
    """Refuse a generated-SQL/Cypher leg that cannot honour tenant isolation.

    Called before any generated statement executes. The asymmetry it closes was
    real and silent: ``tenant_id`` reached the SQL path only to *stamp the
    resulting chunk*, never to filter the query, so with isolation switched on the
    vector leg was scoped and the relational leg read every tenant's rows.

    Refusing is the honest default. This library cannot inject a correct tenant
    predicate into arbitrary generated SQL — across joins, CTEs and set operations
    the right placement is a schema question — and a rewriter that gets it subtly
    wrong would provide the appearance of isolation, which is worse than a refusal
    an operator can see and configure away.
    """
    from ragorc.core.settings import get_settings as _get_settings

    resolved = settings or _get_settings()
    if not resolved.security.enforce_tenant_isolation:
        return
    mode = resolved.security.generated_query_isolation
    if mode != "reject":
        log.debug("generated_query_isolation", target=target, mode=mode)
        return
    raise GuardrailViolation(
        f"generated {target} is disabled while tenant isolation is enforced",
        rule="generated_query_isolation",
        hint=(
            "this library cannot scope generated SQL/Cypher to a tenant; enable "
            "PostgreSQL row-level security and set "
            "security.generated_query_isolation='rls', or 'database' for a "
            "database-per-tenant deployment, or 'trusted' for single-tenant data"
        ),
    )
