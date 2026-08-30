"""What a command-line flag actually reaches.

`ragorc delete --collection scratch <id>` deleted from the *default* collection
and printed a report indistinguishable from success:

    told --collection r16_b, emptied r16_a (2 -> 0 points)
    documents found 1 / vectors removed 2

`_apply` publishes overrides as environment variables, and pydantic-settings only
reads `RAGORC_`-prefixed names. Six call sites spelled the key as the variable —
`_settings(RAGORC_QDRANT__COLLECTION=collection)` — and two spelled it as their
own typer parameter, `_settings(collection=collection, tenant=tenant)`. Those two
wrote `os.environ["collection"]`, which nothing reads.

The command whose docstring says "the blast radius is kept equal to what was
typed", with a blast radius equal to whatever the environment said.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib

import pytest

from ragorc import cli
from ragorc.core.settings import Settings


def test_an_override_key_that_is_not_a_settings_variable_is_refused() -> None:
    """Loudly, because silence is what the defect was made of. A key pydantic
    never reads is a flag that reaches nothing, and the command runs against the
    ambient configuration while reporting on what it did there."""
    with pytest.raises(ValueError, match="not a settings variable"):
        cli._apply({"collection": "scratch"})


def test_a_settings_variable_is_published(monkeypatch: pytest.MonkeyPatch) -> None:
    # `setenv` first so monkeypatch owns the key and restores it at teardown:
    # `_apply` writes `os.environ` directly, which monkeypatch cannot see, and a
    # leaked RAGORC_* variable breaks the unit suite's no-ambient-config promise
    # for every test that runs after this one.
    monkeypatch.setenv("RAGORC_QDRANT__COLLECTION", "placeholder")
    cli._apply({"RAGORC_QDRANT__COLLECTION": "scratch"})
    assert os.environ["RAGORC_QDRANT__COLLECTION"] == "scratch"


def test_none_still_means_not_supplied(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flag left off the command line must leave the environment alone — the
    guard must not turn "not supplied" into an error."""
    monkeypatch.setenv("RAGORC_QDRANT__COLLECTION", "from_the_environment")
    cli._apply({"RAGORC_QDRANT__COLLECTION": None})
    assert os.environ["RAGORC_QDRANT__COLLECTION"] == "from_the_environment"


def test_the_prefix_is_read_from_the_settings_model() -> None:
    """Retyping it here would let a rename leave the guard checking for a prefix
    nothing uses — which is the same class of defect one level up."""
    assert Settings.model_config["env_prefix"] == cli._ENV_PREFIX


# ---------------------------------------------------------------------------
# The invariant, over every call site
# ---------------------------------------------------------------------------
def test_every_settings_override_in_the_cli_names_a_real_variable() -> None:
    """Asserted over the whole module, not over the two commands that were wrong.

    Eight call sites, six correct and two not, and the two that were wrong were
    the two added last. A ninth would be written the same way and would be found
    the same way: by hand, rounds later, after it had destroyed something.
    """
    source = pathlib.Path(inspect.getfile(cli)).read_text()
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id in {"_settings", "_apply"}):
            continue
        for keyword in node.keywords:
            if keyword.arg is None:  # **overrides, checked at runtime by _apply
                continue
            if not keyword.arg.startswith(cli._ENV_PREFIX):
                offenders.append((node.lineno, keyword.arg))

    assert not offenders, (
        "these override keys are typer parameter names, not settings variables, "
        f"so the flags they carry reach nothing: {offenders}"
    )


def test_the_scoping_flags_reach_the_settings_the_stores_are_built_from() -> None:
    """The two commands that were wrong, by name, because they are the two that
    destroy or enumerate data. `delete` binds its Qdrant store to
    `settings.qdrant.collection` at service build; the flag has to be *in* the
    settings by then, not passed alongside them.
    """
    source = pathlib.Path(inspect.getfile(cli)).read_text()
    tree = ast.parse(source)
    wanted = {"delete", "documents"}
    seen: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in wanted:
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "_settings"
                ):
                    seen[node.name] = {k.arg for k in call.keywords if k.arg}
    assert seen.keys() == wanted, f"could not find both commands: {sorted(seen)}"
    for name, keys in seen.items():
        assert "RAGORC_QDRANT__COLLECTION" in keys, f"{name} does not scope its collection: {keys}"


# ---------------------------------------------------------------------------
# Exit status is a property of the outcome, not of the rendering
# ---------------------------------------------------------------------------
def test_the_delete_exit_check_is_shared_by_both_renderings() -> None:
    """`--json` returned from inside its branch, above both checks below it, so
    the same failing delete exited 0 with the flag and 1 without — and a pipeline
    that parses the JSON is exactly the caller that cannot afford to miss it."""
    import ast
    import textwrap

    source = textwrap.dedent(inspect.getsource(cli.delete))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_delete_exit"
    ]
    assert len(calls) == 2, f"both renderings must reach the exit check, found {len(calls)}"


@pytest.mark.parametrize(
    ("complete", "deleted", "code"),
    [(True, True, None), (True, False, 1), (False, True, 1), (False, False, 1)],
)
def test_the_exit_check_itself(complete: bool, deleted: bool, code: int | None) -> None:
    import typer

    report = type("R", (), {"complete": complete, "deleted": deleted})()
    if code is None:
        cli._delete_exit(report)  # must not raise
    else:
        with pytest.raises(typer.Exit) as caught:
            cli._delete_exit(report)
        assert caught.value.exit_code == code


def test_graph_build_reaches_its_failure_check_in_json_mode() -> None:
    """A build where every extraction failed exited 3 in text mode and 0 with
    `--json`, emitting a body with no failure field at all."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(cli.graph_build)))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "as_json"
    ]
    assert len(branches) == 1, "expected one --json branch"
    body = "\n".join(ast.unparse(stmt) for stmt in branches[0].body)
    assert "raise typer.Exit(3)" in body, (
        f"the json branch returns without reaching the failure check: {body}"
    )
