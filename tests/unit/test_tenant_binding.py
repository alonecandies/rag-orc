"""Who a ``tenant_id`` has to belong to.

``security.enforce_tenant_isolation`` reads as the switch that makes this
library multi-tenant, and :mod:`ragorc.security.tenancy` opens by naming the
failure it prevents: "answering tenant A's question with tenant B's documents".
What it actually enforced was that a request *names* a tenant. ``tenant_id`` is
a field on the request **body** (``schemas.py``), so naming one is something the
caller does, and any authenticated caller could read any tenant by naming it.

These tests are at the HTTP layer on purpose. The binding is a chain — a key is
hashed into a principal by the auth dependency, the same key is hashed into a
map key at construction, and the two are compared per request — and every link
is individually plausible while the chain is broken. In particular, the
principal used to be minted from an inline format string at one end and would
have been looked up through a copy of it at the other; the tests that would
catch that drift are the ones that go through the real dependency.

``service_dependency`` documents itself as the seam this needs, so that is what
they override.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.errors import GuardrailViolation
from ragorc.core.settings import Settings
from ragorc.index.pipeline import IngestReport
from ragorc.security.tenancy import (
    principal_for_key,
    resolve_tenant,
    tenant_bindings,
    unbound_principals_warning,
)
from ragorc.server.app import RagService, create_app, service_dependency
from ragorc.server.schemas import IngestRequest, QueryRequest

BOUND_KEY = "bound-key-acme"
FREE_KEY = "unbound-key"


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "security": {"enforce_tenant_isolation": True},
        "cache": {"enabled": False},
        "llm": {"api_key": "k"},
        "server": {
            "api_keys": [BOUND_KEY, FREE_KEY],
            "api_key_tenants": {BOUND_KEY: "acme"},
        },
    }
    base.update(overrides)
    return Settings(**base)


class RecordingEngine:
    """Records what it was asked to ingest. The write itself is not under test."""

    def __init__(self) -> None:
        self.targets: list[Any] = []

    async def ingest(self, targets: Any) -> IngestReport:
        self.targets = list(targets)
        return IngestReport(documents_in=len(self.targets), documents_indexed=len(self.targets))


@pytest.fixture
def service() -> RagService:
    svc = RagService(_settings())
    svc.engine = RecordingEngine()
    return svc


# ---------------------------------------------------------------------------
# The resolution itself
# ---------------------------------------------------------------------------
def test_bound_principal_may_omit_its_tenant(service: RagService) -> None:
    """A credential that names one tenant should not have to repeat it."""
    query, _warnings = service.prepare(
        QueryRequest(question="what is the refund window?"),
        principal=principal_for_key(BOUND_KEY),
    )
    assert query.tenant_id == "acme"
    assert query.filters["tenant_id"] == "acme"


def test_bound_principal_may_name_its_own_tenant(service: RagService) -> None:
    query, _warnings = service.prepare(
        QueryRequest(question="what is the refund window?", tenant_id="acme"),
        principal=principal_for_key(BOUND_KEY),
    )
    assert query.tenant_id == "acme"


def test_bound_principal_cannot_read_another_tenant(service: RagService) -> None:
    """The whole point. This request used to be answered."""
    with pytest.raises(GuardrailViolation) as caught:
        service.prepare(
            QueryRequest(question="what is the refund window?", tenant_id="globex"),
            principal=principal_for_key(BOUND_KEY),
        )
    assert caught.value.rule == "tenant_not_owned"
    assert "globex" not in str(caught.value), "the refusal must not confirm the tenant id"


def test_unbound_principal_keeps_working(service: RagService) -> None:
    """Bindings are opt-in, and a key without one behaves as it always did.

    Asserted rather than assumed: a deployment that has not configured
    ``api_key_tenants`` must not start refusing its own traffic on upgrade. The
    cost of that compatibility is stated out loud by
    :func:`unbound_principals_warning`.
    """
    query, _warnings = service.prepare(
        QueryRequest(question="q", tenant_id="anything-at-all"),
        principal=principal_for_key(FREE_KEY),
    )
    assert query.tenant_id == "anything-at-all"


async def test_ingest_is_bound_too(service: RagService) -> None:
    """Writes need the binding as much as reads.

    Unbound, a caller can file documents under another tenant's id — and the
    next thing that happens is that tenant being answered from them.
    """
    with pytest.raises(GuardrailViolation):
        await service.ingest(
            IngestRequest(text="hello", tenant_id="globex"),
            principal=principal_for_key(BOUND_KEY),
        )

    response = await service.ingest(
        IngestRequest(text="hello"), principal=principal_for_key(BOUND_KEY)
    )
    assert response.documents_in == 1


# ---------------------------------------------------------------------------
# Through the real app, because the drift risk is between the links
# ---------------------------------------------------------------------------
@pytest.fixture
def client(service: RagService) -> Any:
    from fastapi.testclient import TestClient

    app = create_app(service.settings)
    app.dependency_overrides[service_dependency] = lambda: service
    with TestClient(app) as test_client:
        yield test_client


def test_http_refuses_a_cross_tenant_ingest(client: Any) -> None:
    """End to end: the principal the auth dependency mints must be the principal
    the bindings are keyed by. Nothing else in this file would catch them
    disagreeing, and if they did every key would silently go back to being
    unrestricted."""
    response = client.post(
        "/ingest",
        json={"text": "hello", "tenant_id": "globex"},
        headers={"X-API-Key": BOUND_KEY},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"]["rule"] == "tenant_not_owned"


def test_http_accepts_the_credentials_own_tenant(client: Any) -> None:
    response = client.post(
        "/ingest",
        json={"text": "hello", "tenant_id": "acme"},
        headers={"X-API-Key": BOUND_KEY},
    )
    assert response.status_code == 200, response.text


def test_http_still_requires_a_key(client: Any) -> None:
    assert client.post("/ingest", json={"text": "hello"}).status_code == 401


# ---------------------------------------------------------------------------
# Saying so out loud
# ---------------------------------------------------------------------------
def test_isolation_without_bindings_is_reported_as_a_warning() -> None:
    """The configuration that looks isolated and is not.

    Every query names a tenant, every store filter carries it, and any caller
    can still read any tenant. It is self-consistent, which is why it has to be
    said rather than inferred.
    """
    warning = unbound_principals_warning(
        Settings(
            security={"enforce_tenant_isolation": True},
            server={"api_keys": ["k"]},
            llm={"api_key": "k"},
        )
    )
    assert warning is not None and "any authenticated caller" in warning


@pytest.mark.parametrize(
    ("isolation", "keys", "bindings"),
    [
        pytest.param(False, ["k"], {}, id="isolation-off"),
        pytest.param(True, [], {}, id="unauthenticated-has-no-identity-to-bind"),
        pytest.param(True, ["k"], {"k": "acme"}, id="bound"),
    ],
)
def test_no_warning_when_there_is_nothing_to_warn_about(
    isolation: bool, keys: list[str], bindings: dict[str, str]
) -> None:
    assert (
        unbound_principals_warning(
            Settings(
                security={"enforce_tenant_isolation": isolation},
                server={"api_keys": keys, "api_key_tenants": bindings},
                llm={"api_key": "k"},
            )
        )
        is None
    )


def test_a_binding_for_an_unusable_key_is_rejected_at_load() -> None:
    """It fails open in the most misleading way available: the config reads as
    restricted while the key actually in use is unbound."""
    with pytest.raises(ValueError, match="can never authenticate"):
        Settings(
            llm={"api_key": "k"},
            server={"api_keys": ["real"], "api_key_tenants": {"typo": "acme"}},
        )


def test_anonymous_is_never_bindable() -> None:
    """With no credential there is no identity, and the absence of a key must
    not be able to select a tenant."""
    settings = _settings()
    assert "anonymous" not in tenant_bindings(settings)
    with pytest.raises(GuardrailViolation, match="tenant_id is required"):
        resolve_tenant(
            None,
            principal="anonymous",
            bindings=tenant_bindings(settings),
            settings=settings.security,
        )


# ---------------------------------------------------------------------------
# The health probe is not a tenant-scoped read
# ---------------------------------------------------------------------------
async def test_health_is_not_refused_by_tenant_isolation() -> None:
    """A multi-tenant deployment's own health check used to take it out of
    rotation.

    `_probe_qdrant` called `vector.count(exact=False)`, and `count` applies the
    tenant filter through `with_tenant`, which fails closed. With
    `enforce_tenant_isolation` on and no service-wide `tenant_id` — the normal
    shape when the tenant arrives per request — the probe raised
    `GuardrailViolation`, so `/health` reported Qdrant `unavailable` and the
    service `degraded` while Qdrant was fine. `/health` documents that an
    orchestrator should key off the individual store entries.

    Failing closed is correct for a read: a query with no tenant must never mean
    "search everything". A health check is not a read of anyone's data.
    """
    from ragorc.core.errors import GuardrailViolation as _GV

    service = RagService(_settings())

    class Refusing:
        """Counts as the real store does: refuses an unscoped count, answers a
        health probe."""

        def __init__(self) -> None:
            self.counts = 0

        async def count(self, **kwargs: object) -> int:
            self.counts += 1
            raise _GV("tenant_id is required", rule="enforce_tenant_isolation")

        async def health(self) -> dict[str, object]:
            return {"collection": "c", "points": 7, "status": "green"}

    class Counting:
        async def count(self, **kwargs: object) -> int:
            return 3

    vector = Refusing()
    service.linear.vector = vector  # type: ignore[assignment]
    service.linear.relational = Counting()  # type: ignore[assignment]

    health = await service.health()

    by_name = {store.name: store for store in health.stores}
    assert by_name["qdrant"].status == "ok", by_name["qdrant"].error
    assert by_name["qdrant"].detail["points"] == 7
    assert health.status == "ok", "a healthy service must not report itself degraded"
    assert vector.counts == 0, "the probe must not go through the tenant-scoped count"
