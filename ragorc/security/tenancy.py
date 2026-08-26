"""Multi-tenant isolation.

The failure mode this prevents is the worst one a RAG system has: answering
tenant A's question with tenant B's documents. It is easy to hit, because the
default behaviour of every vector store is to search *everything*, and a missing
filter is silent — the query still returns plausible results.

So isolation is enforced by construction:

* ``require_tenant`` refuses a query with no tenant when isolation is on. Failing
  closed is the only safe default; a missing tenant must never mean "all".
* ``resolve_tenant`` refuses a tenant the *caller* does not own. ``require_tenant``
  only checks that a tenant was named, and ``tenant_id`` arrives in the request
  **body** — so on its own it stops an unscoped query and not a cross-tenant one:
  an authenticated caller could read any tenant by naming it. The binding comes
  from ``server.api_key_tenants``, keyed by the same principal the audit log
  records.
* ``scope_filter`` injects the tenant predicate into every store filter, and is
  the *only* sanctioned way to build one.
* Qdrant additionally gets a payload index with ``is_tenant=True`` so the filter
  is cheap (co-located storage) rather than a filtered scan.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import structlog

from ragorc.core.errors import GuardrailViolation
from ragorc.core.ids import content_hash
from ragorc.core.settings import SecuritySettings, Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = [
    "graph_legs_refused",
    "principal_for_key",
    "require_generated_query_isolation",
    "require_graph_tenant_isolation",
    "require_tenant",
    "resolve_tenant",
    "scope_cypher_where",
    "scope_filter",
    "scope_sql_where",
    "tenant_bindings",
]

ANONYMOUS = "anonymous"
"""The principal an unauthenticated deployment reports. Never bindable: with no
credential there is no identity to bind, and treating it as one would let the
absence of a key select a tenant."""


def principal_for_key(key: str) -> str:
    """The audit-log identity of an API key: a hash prefix, never the key.

    Defined here rather than inline at the authentication dependency because two
    call sites now have to agree on it — the dependency that mints a principal
    per request, and :func:`tenant_bindings`, which has to look one up from the
    configured key. A format string copied into both is a format string that
    drifts, and the failure would be silent: every binding would simply stop
    matching and every key would go back to being unrestricted.
    """
    return f"key:{content_hash(key, size=6)}"


def tenant_bindings(settings: Settings | None = None) -> dict[str, str]:
    """``principal -> tenant`` for every key ``server.api_key_tenants`` binds.

    Built once at service construction. Doing it per request would hash every
    configured key on every call for a lookup that cannot change without a
    restart.
    """
    s = settings or get_settings()
    return {
        principal_for_key(key): tenant for key, tenant in s.server.api_key_tenants.items() if key
    }


def resolve_tenant(
    requested: str | None,
    *,
    principal: str,
    bindings: Mapping[str, str],
    settings: SecuritySettings | None = None,
) -> str | None:
    """The tenant this request may actually use.

    Three cases, and the middle one is the hole this closes:

    * the principal is **bound** and named nothing — it gets its own tenant, so a
      single-tenant client never has to repeat what its credential already says;
    * the principal is **bound** and named someone else's — refused. This is the
      cross-tenant read that ``require_tenant`` waved through, because naming a
      tenant was the whole of its check;
    * the principal is **unbound** — unchanged behaviour, so a deployment that
      has not configured bindings keeps working. It is the operator's decision to
      make, and :func:`unbound_principals_warning` is what makes it a visible one.
    """
    bound = bindings.get(principal)
    if bound is None:
        return require_tenant(requested, settings)
    if requested is not None and requested != bound:
        # Deliberately does not say which tenant the principal owns, and the
        # error maps to 400 rather than 403: both would confirm to a caller
        # probing tenant ids which ones exist.
        log.warning("tenant_not_owned", principal=principal, requested=requested)
        raise GuardrailViolation(
            "tenant_id is not the tenant this credential is bound to",
            rule="tenant_not_owned",
            hint="omit tenant_id and the credential's own tenant is used",
        )
    return require_tenant(bound, settings)


def unbound_principals_warning(settings: Settings | None = None) -> str | None:
    """The warning for "isolation is on, but nothing binds a caller to a tenant".

    Surfaced on ``/health`` next to the unauthenticated-service warning. This
    configuration is *self-consistent* — every query names a tenant and every
    store filter carries it — which is exactly why it needs saying out loud: it
    looks isolated and reads across tenants on request.
    """
    s = settings or get_settings()
    if not s.security.enforce_tenant_isolation or not s.server.api_keys:
        return None
    if s.server.api_key_tenants:
        return None
    return (
        "security.enforce_tenant_isolation is on but server.api_key_tenants is "
        "empty: any authenticated caller can read any tenant by naming it"
    )


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


def graph_legs_refused(settings: Settings | None = None) -> bool:
    """Would a graph leg be refused right now?

    The predicate behind :func:`require_graph_tenant_isolation`, exposed so
    callers that need to *decide* rather than *enforce* — pipeline auto-selection,
    the ``/health`` warning — can ask without catching an exception. Control flow
    through a guard's exception is how a guard ends up being caught and ignored.
    """
    from ragorc.core.settings import get_settings as _get_settings

    resolved = settings or _get_settings()
    return (
        resolved.security.enforce_tenant_isolation
        and resolved.security.graph_tenant_isolation == "reject"
    )


def graph_isolation_warning(settings: Settings | None = None) -> str | None:
    """The ``/health`` line for "the graph is configured but cannot be used"."""
    resolved = settings or _settings_for_warning(settings)
    if not resolved.graph.enabled or not graph_legs_refused(resolved):
        return None
    return (
        "graph.enabled is on but every graph leg is refused: tenant isolation is "
        "enforced and security.graph_tenant_isolation='reject' (the graph stores "
        "no tenant, so a traversal cannot be scoped)"
    )


def _settings_for_warning(settings: Settings | None) -> Settings:
    from ragorc.core.settings import get_settings as _get_settings

    return settings or _get_settings()


def require_graph_tenant_isolation(target: str, settings: Settings | None = None) -> None:
    """Refuse a knowledge-graph leg that cannot honour tenant isolation.

    The same asymmetry :func:`require_generated_query_isolation` closes, one path
    over, and it survived that fix because the graph *retrieval* legs run
    parameterized Cypher rather than generated Cypher — so the guard written for
    generated statements never covered them.

    Nothing in Neo4j carries a tenant. Entities merge on ``name``, communities on
    a membership hash, chunk links on a chunk id; none is namespaced. Two tenants
    writing about the same company converge on one node, and a traversal from it
    reaches both. Reproduced end to end: a query scoped to one tenant returned
    another tenant's chunk body verbatim, and the verbalized subgraph carried a
    third party's entity descriptions while being *stamped* with the querying
    tenant's id — which is worse than an unlabelled leak, because the label says
    the content is yours.

    Refusing is the honest default, for the reason the sibling guard gives:
    scoping this properly is a schema change (entity identity becomes
    ``(tenant, name)``, with a migration for existing graphs), and a filter
    bolted on at query time would provide the appearance of isolation.
    """
    from ragorc.core.settings import get_settings as _get_settings

    resolved = settings or _get_settings()
    if not resolved.security.enforce_tenant_isolation:
        return
    if resolved.security.graph_tenant_isolation != "reject":
        log.debug(
            "graph_tenant_isolation", target=target, mode=resolved.security.graph_tenant_isolation
        )
        return
    raise GuardrailViolation(
        f"graph {target} is disabled while tenant isolation is enforced",
        rule="graph_tenant_isolation",
        hint=(
            "the knowledge graph stores no tenant, so a traversal cannot be scoped; "
            "run one graph per tenant and set security.graph_tenant_isolation='trusted', "
            "or leave the graph legs off"
        ),
    )


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
