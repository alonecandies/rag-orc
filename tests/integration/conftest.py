"""Environment for the *integration* suite.

The unit suite neutralizes the ambient ``RAGORC_*`` configuration and pydantic's
``env_file`` so a clean clone behaves like a developer's machine
(``tests/unit/conftest.py``). These tests need the opposite: their ``settings``
fixture reads the deployment's configuration for the compose stack's ports and
credentials, and its docstring records why — with hardcoded defaults they once
connected to whatever answered on 5432, which on a machine running an unrelated
Postgres means creating tables in someone else's database.

Restored **per test, and reverted afterwards**, rather than at import time. A
module-level restoration is global state, and in a combined ``pytest tests`` run
it put the developer's configuration back under every unit test that ran after
the integration conftest was imported — re-creating the exact contamination the
unit conftest exists to remove, only order-dependently. Whichever suite runs
first, both now get what they promise.
"""

from __future__ import annotations

import pathlib

import pytest

_ENV_FILE = pathlib.Path(__file__).resolve().parents[2] / ".env"

try:  # present only when the unit suite was collected in the same run
    from tests.unit.conftest import STASHED_ENV
except ImportError:  # pragma: no cover - integration-only run
    STASHED_ENV = {}


@pytest.fixture(autouse=True)
def _deployment_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambient ``RAGORC_*`` configuration, for the duration of one test.

    Covers a deployment configured by exported variables rather than by a file;
    the file itself is read explicitly by whoever needs it (see
    :func:`deployment_env_file`), which is not global state.
    """
    for key, value in STASHED_ENV.items():
        monkeypatch.setenv(key, value)


def deployment_env_file() -> str | None:
    """The ``.env`` the CLI and the service read, passed explicitly.

    ``Settings(_env_file=...)`` rather than ``Settings.model_config["env_file"]``
    so that reading it here cannot change what a unit test constructed elsewhere
    in the same run sees.
    """
    return str(_ENV_FILE) if _ENV_FILE.exists() else None
