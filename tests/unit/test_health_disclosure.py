"""What an unauthenticated caller learns from `/health`.

Every one of the twelve data routes returns 401 when `server.api_keys` is set —
probed, all twelve. `/health` stays open on purpose, and its docstring gave three
reasons. Two hold: a probe behind a key cannot be scraped by the load balancer
that needs it, and the route must never take a parameter. The third did not:

    "every field it returns is already redacted by Settings.summary"

`summary()` strips the DSN's password and keeps its host, port and database, so an
anonymous caller got:

    "stores": {"qdrant": "http://localhost:6333",
               "postgres": "db.internal:5433/prod",   # the measured leak
               "neo4j": "bolt://localhost:7687"}

A fine line for a log file; the wrong one for an open endpoint. Liveness and
per-store status are what a probe needs.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ragorc.core.settings import Settings, get_settings

# Deliberately unmistakable as a fixture. `.invalid` is the reserved TLD for
# exactly this, and the credential parts say what they are: the test needs a host
# to look for and a password to not find, and nothing else. A plausible-looking
# DSN here would be entirely invented and still trip a secret scanner on a public
# repository — a cost with no benefit, and a reviewer's time spent clearing it.
_DSN = "postgresql://PLACEHOLDER_USER:PLACEHOLDER_PW@db.example.invalid:5433/prod"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Any:
    from ragorc.server.app import create_app

    monkeypatch.setenv("RAGORC_SERVER__API_KEYS", '["secret-key"]')
    monkeypatch.setenv("RAGORC_LLM__API_KEY", "k")
    monkeypatch.setenv("RAGORC_CACHE__ENABLED", "false")
    monkeypatch.setenv("RAGORC_POSTGRES__DSN", _DSN)
    get_settings.cache_clear()
    yield create_app()
    get_settings.cache_clear()


def _health(app: Any, headers: dict[str, str]) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health", headers=headers)
        assert response.status_code == 200, response.text
        return dict(response.json())


@pytest.mark.parametrize(
    ("label", "headers"),
    [("anonymous", {}), ("wrong key", {"X-API-Key": "wrong"})],
)
def test_an_unauthenticated_probe_gets_liveness_without_topology(
    app: Any, label: str, headers: dict[str, str]
) -> None:
    body = _health(app, headers)

    assert body["status"], "a probe still needs a verdict"
    assert body["stores"], "a probe still needs per-store status"
    assert "stores" not in (body.get("settings") or {}), f"{label} was handed the topology"
    assert "db.example.invalid" not in json.dumps(body), f"{label} learned a backing host"


def test_an_authenticated_operator_gets_the_configuration(app: Any) -> None:
    """The summary is what an operator debugging a deployment reads. Hiding it
    from them to hide it from a stranger would be the wrong trade."""
    body = _health(app, {"X-API-Key": "secret-key"})

    stores = (body.get("settings") or {}).get("stores")
    assert stores, "the operator lost the summary"
    assert stores["postgres"] == "db.example.invalid:5433/prod"


def test_a_credential_is_never_disclosed_either_way(app: Any) -> None:
    """The property `summary()` already had, which must survive the change."""
    for headers in ({}, {"X-API-Key": "secret-key"}):
        body = json.dumps(_health(app, headers))
        assert "PLACEHOLDER_PW" not in body
        assert "PLACEHOLDER_USER" not in body


def test_an_open_deployment_hides_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no keys configured the service is open by the operator's choice, and
    withholding the summary from everyone would be theatre."""
    from ragorc.server.app import create_app

    monkeypatch.setenv("RAGORC_SERVER__API_KEYS", "[]")
    monkeypatch.setenv("RAGORC_LLM__API_KEY", "k")
    monkeypatch.setenv("RAGORC_CACHE__ENABLED", "false")
    get_settings.cache_clear()
    try:
        body = _health(create_app(), {})
        assert "stores" in (body.get("settings") or {})
    finally:
        get_settings.cache_clear()


def test_the_route_still_takes_no_parameters() -> None:
    """The docstring's own constraint, and the right one: an unauthenticated
    endpoint that takes input is an unauthenticated endpoint that does work.
    The disclosure decision comes from a header, not a query parameter."""
    import inspect

    from ragorc.server import app as app_module
    from ragorc.server.app import create_app  # noqa: F401

    source = inspect.getsource(app_module.create_app)
    start = source.index('@app.get("/health"')
    signature = source[start : source.index("-> HealthResponse", start)]
    assert "Query(" not in signature and "Body(" not in signature
    assert "Depends(presented_key)" in signature


def test_summary_topology_is_opt_out_not_opt_in() -> None:
    """Logs and the CLI read `summary()` and should keep the stores block; only
    the open endpoint asks for less. A default of `topology=False` would silently
    strip it from every other reader."""
    settings = Settings(llm={"api_key": "k"})
    assert "stores" in settings.summary()
    assert "stores" not in settings.summary(topology=False)


# ---------------------------------------------------------------------------
# What the audit log records when redaction is on
# ---------------------------------------------------------------------------
def test_the_audit_log_records_the_redacted_question(monkeypatch: pytest.MonkeyPatch) -> None:
    """`RagService.prepare` audited `request.question` — the raw body field —
    while the validator's redacted output, two lines up, was what every downstream
    stage and every store filter actually used. So the response told the caller
    "PII redacted from query: EMAIL, CREDIT_CARD" while the audit file recorded the
    address and the card number verbatim. An audit log that copies customer data
    is what `enable_pii_redaction` exists to prevent.
    """
    import inspect

    from ragorc.server.app import RagService

    source = inspect.getsource(RagService.prepare)
    assert "question=query.text" in source, "the audit line reads the raw request body again"
    assert "question=request.question" not in source


def test_the_redacted_text_is_what_the_validator_produced() -> None:
    """Behaviour under the setting, so the identifier swap above is anchored to a
    real difference rather than to a spelling."""
    from ragorc.core.settings import Settings
    from ragorc.validate.input import QueryValidator

    settings = Settings(
        llm={"api_key": "k"},
        security={"enable_pii_redaction": True, "enforce_tenant_isolation": False},
    )
    validated = QueryValidator(settings).validate("email alice@corp.example about the refund")

    assert "alice@corp.example" not in validated.query.text
    assert "REDACTED" in validated.query.text


# ---------------------------------------------------------------------------
# The property, not the field
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("store_uri", ["bolt://neo4j-secret.internal.example:7999"])
def test_no_unauthenticated_response_names_a_backing_host(
    monkeypatch: pytest.MonkeyPatch, store_uri: str
) -> None:
    """Withholding `settings.stores` was fixing the field I was looking at. The
    same information flowed through `stores[].error` — driver exceptions carry the
    address verbatim (`Failed to DNS resolve address neo4j.internal:7999`) — and
    through `stores[].detail`, whose first key on the Neo4j *success* path is the
    address, so a healthy deployment disclosed it unconditionally.

    The property is that no unauthenticated response names a backing host. It is
    asserted over the whole body, so a probe that grows a new field is covered.
    """
    import json as _json

    from fastapi.testclient import TestClient

    from ragorc.server.app import create_app

    monkeypatch.setenv("RAGORC_SERVER__API_KEYS", '["secret-key"]')
    monkeypatch.setenv("RAGORC_LLM__API_KEY", "k")
    monkeypatch.setenv("RAGORC_CACHE__ENABLED", "false")
    monkeypatch.setenv("RAGORC_GRAPH__ENABLED", "true")
    monkeypatch.setenv("RAGORC_SECURITY__ENFORCE_TENANT_ISOLATION", "false")
    monkeypatch.setenv("RAGORC_NEO4J__URI", store_uri)
    get_settings.cache_clear()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            anonymous = _json.dumps(client.get("/health").json())
            operator = _json.dumps(
                client.get("/health", headers={"X-API-Key": "secret-key"}).json()
            )
    finally:
        get_settings.cache_clear()

    for token in ("neo4j-secret.internal.example", "7999"):
        assert token not in anonymous, f"an anonymous caller learned {token!r}"
    # The operator keeps the diagnostic: hiding it from them to hide it from a
    # stranger would be the wrong trade, and a redacted probe is unactionable.
    assert "neo4j-secret.internal.example" in operator


def test_an_anonymous_probe_still_gets_a_usable_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redacting must not turn the endpoint into a constant. A load balancer keys
    off the per-store status, and that is what stays."""
    from fastapi.testclient import TestClient

    from ragorc.server.app import create_app

    monkeypatch.setenv("RAGORC_SERVER__API_KEYS", '["secret-key"]')
    monkeypatch.setenv("RAGORC_LLM__API_KEY", "k")
    monkeypatch.setenv("RAGORC_CACHE__ENABLED", "false")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as client:
            body = client.get("/health").json()
    finally:
        get_settings.cache_clear()

    assert body["status"]
    assert body["stores"], "a probe with no per-store entries tells a balancer nothing"
    for store in body["stores"]:
        assert store["name"] and store["status"]
