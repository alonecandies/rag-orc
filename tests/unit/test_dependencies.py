"""The declared dependency ranges must describe the environment we test in.

A floor far below the installed version is not permissiveness — it is an untested
claim. `sqlglot>=25.24` with a suite that only ever runs on sqlglot 30 tells an
installer that 25 works, and nobody has checked.

For one dependency this is a security property rather than a hygiene one: the SQL
guard matches on `sqlglot.exp` node *classes*, so a rename in a new major produces
a guard that silently stops matching instead of failing loudly. See
[ADR-0009](../../docs/adr/0009-dependency-pinning.md).
"""

from __future__ import annotations

import pathlib
import tomllib
from importlib.metadata import PackageNotFoundError, version

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

PYPROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"

#: Extras installed in the development environment. Ones that are not installed
#: are skipped rather than failed — `local` pulls torch, and requiring it to run
#: the test suite would defeat ADR-0004.
CHECKED_EXTRAS = (
    "server",
    "langchain",
    "redis",
    "raptor",
    "graphrag",
    "loaders",
    "otel",
    "web",
    "dev",
)

#: Dependencies whose upper bound is load-bearing, with the reason. Asserted
#: individually so a well-meaning "loosen the pins" change has to argue with a
#: named justification rather than a lint rule.
MUST_BE_CAPPED = {
    "sqlglot": "the SQL guard matches on sqlglot.exp node classes — a rename is a silent hole in the allowlist",
    "langgraph": "the pipelines use the 1.x StateGraph and reducer API",
    "pydantic": "v1 and v2 are different libraries",
    "numpy": "numpy 2 changed scalar promotion; the vectorized scoring is tested on 2.x",
    "qdrant-client": "the client only guarantees compatibility within one minor of the server",
    "neo4j": "the store is written against the 6.x driver surface",
}


def declared() -> list[Requirement]:
    data = tomllib.loads(PYPROJECT.read_text())
    raw = list(data["project"]["dependencies"])
    optional = data["project"].get("optional-dependencies", {})
    for extra in CHECKED_EXTRAS:
        raw += optional.get(extra, [])
    out = []
    for item in raw:
        req = Requirement(item)
        # `ragorc[...]` self-references and platform-gated entries are not
        # third-party version claims.
        if req.name == "ragorc" or (req.marker and not req.marker.evaluate()):
            continue
        out.append(req)
    return out


@pytest.mark.parametrize("req", declared(), ids=lambda r: r.name)
def test_installed_version_satisfies_the_declared_range(req: Requirement) -> None:
    """The environment the suite passes in must be one the metadata permits.

    Catches both directions: a floor raised above what is installed (the suite is
    lying about what it tested) and an installed version above a cap (the cap is
    lying about what is supported).
    """
    try:
        installed = Version(version(req.name))
    except PackageNotFoundError:
        pytest.skip(f"{req.name} is not installed in this environment")
    assert installed in req.specifier, (
        f"{req.name} {installed} is installed but pyproject declares "
        f"{req.specifier} — the declared range does not describe what we test"
    )


@pytest.mark.parametrize(("name", "reason"), sorted(MUST_BE_CAPPED.items()))
def test_load_bearing_dependencies_have_an_upper_bound(name: str, reason: str) -> None:
    """An unbounded major on these is a defect, not a preference."""
    req = next((r for r in declared() if r.name == name), None)
    assert req is not None, f"{name} is no longer declared — was it removed deliberately?"
    upper = [s for s in req.specifier if s.operator in ("<", "<=", "==", "~=")]
    assert upper, f"{name} needs an upper bound: {reason}"


def test_floors_are_close_to_what_is_installed() -> None:
    """A floor several majors below the tested version is an untested claim.

    Reported in one assertion listing every offender, because fixing them one
    parametrized failure at a time is how half of them get missed.
    """
    stale: list[str] = []
    for req in declared():
        try:
            installed = Version(version(req.name))
        except PackageNotFoundError:
            continue
        floors = [Version(s.version) for s in req.specifier if s.operator in (">=", "==", "~=")]
        if not floors:
            continue
        floor = max(floors)
        if installed.major - floor.major >= 2:
            stale.append(f"{req.name}: floor {floor} but tested on {installed}")
    assert not stale, "declared floors are far below the tested versions:\n  " + "\n  ".join(stale)


def test_qdrant_client_range_matches_the_compose_image() -> None:
    """The client refuses to guarantee compatibility across a minor gap, and warns
    at runtime before anything actually breaks — so the two are one decision."""
    import re

    compose = (PYPROJECT.parent / "docker-compose.yml").read_text()
    match = re.search(r"image:\s*qdrant/qdrant:v(\d+)\.(\d+)", compose)
    assert match, "could not find the qdrant image tag in docker-compose.yml"
    image = Version(f"{match.group(1)}.{match.group(2)}")

    req = next(r for r in declared() if r.name == "qdrant-client")
    floors = [Version(s.version) for s in req.specifier if s.operator == ">="]
    assert floors, "qdrant-client has no floor"
    floor = max(floors)

    assert floor.major == image.major, (
        f"qdrant-client floor {floor} and image v{image} differ in major version"
    )
    assert abs(floor.minor - image.minor) <= 1, (
        f"qdrant-client floor {floor} and image v{image} differ by more than one minor; "
        "the client will warn about incompatibility at runtime"
    )
