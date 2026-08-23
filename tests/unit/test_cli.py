"""The CLI's wiring.

Not the behaviour of each command — that lives in the modules they call — but the
part that only breaks in the CLI: a command that is registered but cannot be
invoked, an option that does not exist, a callback that raises on import. This
file existed for none of it, which is how `ragorc.cli` came to be missing its
`__main__` guard and every `make ask` was a silent no-op.

`--help` is the probe because it exercises registration, the option parser and the
module import without touching a store.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from ragorc.cli import app

runner = CliRunner()

COMMANDS = [
    "init",
    "ingest",
    "query",
    "eval",
    "bench",
    "serve",
    "inspect",
    "alias-swap",
]


def test_the_top_level_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for name in COMMANDS:
        assert name in result.output, f"{name} is missing from the top-level help"
    assert "graph" in result.output, "the graph sub-app is not registered"


@pytest.mark.parametrize("name", COMMANDS)
def test_each_command_can_be_invoked(name: str) -> None:
    """A command whose module fails to import, or whose options do not parse, only
    shows up when something actually invokes it."""
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0, f"{name} --help failed: {result.output}"
    assert "Usage" in result.output


def test_graph_build_is_reachable_and_documents_its_second_pass() -> None:
    """Graph construction is a second pass on purpose — `graph.enabled` does not
    build it during ingest, because resolution and community detection are only
    meaningful over the whole corpus. The command is how that pass is run, so it
    has to say so."""
    result = runner.invoke(app, ["graph", "build", "--help"])
    assert result.exit_code == 0, result.output
    assert "--limit" in result.output
    assert "--tenant" in result.output
    assert "second pass" in result.output.replace("\n", " ")


def test_ingest_offers_the_force_flag_the_reindex_needs() -> None:
    """The documented zero-downtime reindex builds into a new collection, which the
    checksum skip cannot see. Without this flag that procedure silently produces an
    empty index."""
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0, result.output
    assert "--force" in result.output


def test_make_bench_actually_has_something_to_benchmark() -> None:
    """`make bench` ran `ragorc bench` with no arguments at all.

    `bench` hard-fails with `EXIT_CONFIG` when it has nothing to time, so the
    target documented in the Makefile help and in docs/performance.md always
    exited 2 without measuring anything. The recipe now names a question file, and
    that file has to exist and hold questions once comments are stripped —
    otherwise this is the same bug with a longer command line.
    """
    import re
    from pathlib import Path

    recipe = next(
        line
        for line in Path("Makefile").read_text().splitlines()
        if line.startswith("\t") and "bench" in line
    )
    assert "$(Q)" in recipe, "make bench Q=... must be able to time one question"

    match = re.search(r"--queries ([^\s)]+)", recipe)
    assert match, f"the default must give bench something to time: {recipe}"
    questions = Path(match.group(1))
    assert questions.is_file(), f"{questions} is named by the Makefile and does not exist"

    usable = [
        stripped
        for line in questions.read_text().splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]
    assert len(usable) >= 5, f"only {len(usable)} usable questions in {questions}"


def test_bench_treats_hash_lines_in_a_question_file_as_comments() -> None:
    """`--queries` read every non-blank line verbatim, so a file's own header was
    benchmarked as a question. The eval dataset loader has always skipped `#`."""
    from pathlib import Path

    questions = Path("examples/eval/bench-questions.txt")
    assert questions.read_text().lstrip().startswith("#"), (
        "the fixture only tests anything while the shipped file has a header"
    )
    result = runner.invoke(app, ["bench", "--queries", str(questions), "--help"])
    assert result.exit_code == 0, result.output


def test_the_batched_ingest_report_carries_every_field_the_table_prints() -> None:
    """`ragorc ingest` merges one report per batch, and the merge enumerates the
    fields it carries by name — so a field added to `IngestReport` is silently
    dropped until someone remembers this list.

    Two already had been: `documents_empty`, which exists precisely so the counters
    reconcile with `documents_in`, and `points_in_store`, which the table printed
    as `None` on a run whose log showed the real number.
    """
    from ragorc.cli import _merge_report, _report_table
    from ragorc.index.pipeline import IngestReport

    def batch(**kwargs: object) -> IngestReport:
        report = IngestReport()
        for key, value in kwargs.items():
            setattr(report, key, value)
        return report

    merged = IngestReport()
    _merge_report(
        merged,
        batch(
            documents_in=3,
            documents_indexed=2,
            documents_empty=1,
            chunks_created=5,
            vectors_written=10,
            points_in_store=5,
        ),
    )
    _merge_report(
        merged,
        batch(
            documents_in=2,
            documents_indexed=2,
            chunks_created=4,
            vectors_written=8,
            points_in_store=9,
        ),
    )

    assert merged.documents_empty == 1, "the counter that makes the report reconcile"
    accounted = (
        merged.documents_indexed
        + merged.documents_skipped
        + merged.documents_rejected
        + merged.documents_duplicate
        + merged.documents_failed
        + merged.documents_empty
    )
    assert accounted == merged.documents_in == 5, merged.summary()
    # The collection's size, not a per-batch delta: summing would report 14 points
    # in a collection holding 9.
    assert merged.points_in_store == 9

    # Every key the table asks for must exist in the merged summary, or rendering
    # raises KeyError on a real run.
    rendered = _report_table(merged)
    assert rendered.row_count >= 12


def test_an_error_panel_does_not_eat_the_extra_it_tells_you_to_install() -> None:
    """Rich treats `[...]` as a style tag, so `pip install 'ragorc[server]'`
    renders as `pip install 'ragorc'` — advice to install the package you already
    have, printed at the exact moment you are stuck.

    Worst on the generic `ImportError` branch: every optional import in this
    library raises with its own pip hint, and `_run` prints that message through a
    panel, so the extra is stripped out of the one line that would have fixed it.
    """
    import io

    import typer
    from rich.console import Console

    from ragorc import cli

    captured = Console(file=io.StringIO(), width=100)
    original = cli.err
    cli.err = captured  # type: ignore[assignment]
    try:
        with pytest.raises(typer.Exit):
            cli._fail(
                "missing dependency",
                "RedisCache requires the redis extra: pip install 'ragorc[redis]'",
                "pip install 'ragorc[server]'",
                2,
            )
    finally:
        cli.err = original  # type: ignore[assignment]

    out = captured.file.getvalue()  # type: ignore[attr-defined]
    assert "ragorc[redis]" in out, f"the message lost its extra:\n{out}"
    assert "ragorc[server]" in out, f"the hint lost its extra:\n{out}"


def test_dynamic_text_in_the_console_survives_square_brackets() -> None:
    """The same hazard wherever a filename, an error or model output is
    interpolated into a Rich string: a path like `notes[2024].md` loses its
    year, and an error quoting a list loses the list."""
    import io

    from rich.console import Console

    from ragorc.cli import _plain

    captured = Console(file=io.StringIO(), width=100)
    captured.print(
        f"[yellow]unreadable[/yellow] {_plain('notes[2024].md')}: {_plain('bad [utf-8]')}"
    )
    out = captured.file.getvalue()

    assert "notes[2024].md" in out, out
    assert "[utf-8]" in out, out
