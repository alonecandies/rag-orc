"""Fixtures and environment for the *unit* suite.

The promise this file enforces is the one in ``docs/internal/OPEN-ITEMS.md``
§11b: the unit suite runs "with no network, no containers, no API keys and no
model downloads". It also, silently, needed an untracked file.

``Settings.model_config`` sets ``env_file=".env"``, resolved against the *current
working directory*, so running pytest from the repo root layered a local
deployment's configuration under every test that did not override the field.
Measured on a clean worktree: 0 failures with ``.env`` present, 11 without —
because ``RAGORC_SECURITY__ENFORCE_TENANT_ISOLATION`` turned the library's
fail-closed default off, and eleven tests queried with no tenant and passed. This
is a public repository; ``git clone && pytest`` is the first thing a contributor
runs.

Scoped to this directory rather than to ``tests/`` because the *integration*
suite legitimately needs that configuration: its ``settings`` fixture reads
``Settings()`` for the compose stack's ports and credentials, and says why —
hardcoded defaults once made it connect to whatever answered on 5432. What is
cleared here is therefore stashed, and ``tests/integration/conftest.py`` restores
it per test — reverting afterwards, so a combined run does not put the
developer's configuration back under the unit tests that follow.
"""

from __future__ import annotations

import os

from ragorc.core.settings import Settings

#: Everything removed from the ambient environment, so the integration suite can
#: restore it whichever order the two conftests happen to be imported in.
STASHED_ENV: dict[str, str] = {}

for _key in [k for k in os.environ if k.startswith("RAGORC_")]:
    STASHED_ENV[_key] = os.environ.pop(_key)

# Both halves are needed: clearing the variables alone leaves the file, and
# clearing the file alone leaves an exported variable.
Settings.model_config["env_file"] = None
