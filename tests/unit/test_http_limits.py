"""The transport ceilings, on the routes that have them and the ones that did not.

Round eight audited the SQL and Cypher guards. Nobody had audited the transport,
and three of its four limits covered a subset of the routes they name:

* `max_body_bytes` checked the *declared* `Content-Length`. A chunked request
  declares none, so a 64 MiB body was buffered in full and answered 200 on
  `DELETE /documents` — and buffered in full on `/query` for a caller with no
  credential, which puts the bypass in front of authentication.
* `request_timeout_s` was a deadline inside the `/query` handler. `GET` and
  `DELETE /documents` were added later and ran unbounded: a 5-second store
  answered in full against a 0.5-second ceiling.
* CORS `allow_methods` omitted DELETE, so `server.cors_origins` could not express
  "this origin may delete" — the method list refused it, not the origin list.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from ragorc.core.settings import get_settings


def _app(monkeypatch: pytest.MonkeyPatch, **env: str) -> Any:
    from ragorc.server.app import create_app

    monkeypatch.setenv("RAGORC_LLM__API_KEY", "k")
    monkeypatch.setenv("RAGORC_CACHE__ENABLED", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    return create_app()


def _chunked(total: int) -> Iterator[bytes]:
    """A body with no ``Content-Length``, which is the whole point."""
    sent = 0
    while sent < total:
        block = b"x" * min(65536, total - sent)
        sent += len(block)
        yield block


# ---------------------------------------------------------------------------
# max_body_bytes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("method", "path"),
    [("POST", "/query"), ("POST", "/query/stream"), ("DELETE", "/documents"), ("POST", "/eval")],
)
def test_an_undeclared_oversized_body_is_refused(
    monkeypatch: pytest.MonkeyPatch, method: str, path: str
) -> None:
    from fastapi.testclient import TestClient

    app = _app(monkeypatch, RAGORC_SERVER__MAX_BODY_BYTES="4096")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.request(
            method, path, content=_chunked(2 * 1024 * 1024),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413, f"{method} {path} answered {response.status_code}"
    assert "max_body_bytes" in response.text


def test_the_ceiling_holds_before_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    """The part that makes it worth an attacker's time: with keys configured the
    route returns 401, and buffering the body to discover that is the cost."""
    from fastapi.testclient import TestClient

    app = _app(
        monkeypatch, RAGORC_SERVER__MAX_BODY_BYTES="4096",
        RAGORC_SERVER__API_KEYS='["secret-key"]',
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/query", content=_chunked(2 * 1024 * 1024),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413, "an unauthenticated caller still got the body buffered"


def test_a_declared_oversized_body_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The check that already worked must survive the one that was added."""
    from fastapi.testclient import TestClient

    app = _app(monkeypatch, RAGORC_SERVER__MAX_BODY_BYTES="4096")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/query", content=b"x" * (8 * 1024))

    assert response.status_code == 413


def test_a_body_within_the_ceiling_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A limit that rejects everything is not a limit."""
    from fastapi.testclient import TestClient

    app = _app(monkeypatch, RAGORC_SERVER__MAX_BODY_BYTES="65536")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/query", json={"question": "hi", "tenant_id": "acme"})

    assert response.status_code != 413, response.text


def test_a_route_with_no_body_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    app = _app(monkeypatch, RAGORC_SERVER__MAX_BODY_BYTES="16")
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# request_timeout_s
# ---------------------------------------------------------------------------
def test_every_short_route_is_bounded() -> None:
    """`ingest` and `eval` are exempt by design — long-running by nature, and a
    120-second cap would abort them mid-run. The two `documents` routes are point
    operations, were added after that reasoning was written, and had no bound.
    """
    import ast
    import inspect
    import textwrap

    from ragorc.server import app as app_module

    source = textwrap.dedent(inspect.getsource(app_module.create_app))
    tree = ast.parse(source)

    bounded: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if any("asyncio.timeout" in ast.unparse(n) for n in ast.walk(node)):
            bounded.add(node.name)

    for handler in ("query", "list_documents", "delete_documents"):
        assert handler in bounded, f"{handler} runs with no upper bound"
    for exempt in ("ingest", "evaluate"):
        assert exempt not in bounded, f"{exempt} is long-running and must stay untimed"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
def test_a_browser_may_preflight_every_route_that_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The method list is not the access-control decision — the origin list is.
    A method missing here refuses an allowed origin before the route is consulted.
    """
    from fastapi.testclient import TestClient

    origin = "https://app.example.com"
    app = _app(monkeypatch, RAGORC_SERVER__CORS_ORIGINS=f'["{origin}"]')
    with TestClient(app, raise_server_exceptions=False) as client:
        for method in ("GET", "POST", "DELETE"):
            response = client.options(
                "/documents",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": method,
                },
            )
            assert response.status_code == 200, f"{method} preflight: {response.text}"


def test_an_unlisted_origin_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Widening the methods must not widen the origins."""
    from fastapi.testclient import TestClient

    app = _app(monkeypatch, RAGORC_SERVER__CORS_ORIGINS='["https://app.example.com"]')
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.options(
            "/documents",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "DELETE",
            },
        )
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}
