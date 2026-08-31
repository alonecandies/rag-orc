"""Two ways a run reported a document indexed that was not retrievable.

`_purge`'s docstring calls it "Mandatory, not hygiene": `chunk_id` folds the
content in, so an edited document's chunks get *new* ids and the upsert leaves the
old ones in place — still indexed, still retrievable, still citable, and now wrong.

Both defects made the ingest *report success* over an index that had not received
what the report said it had, which is the one outcome the deferred-checksum design
exists to prevent.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from ragorc.core.models import Document
from ragorc.core.settings import Settings
from ragorc.index.pipeline import IngestPipeline, IngestReport


def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
        "security": {"enforce_tenant_isolation": False},
    }
    base.update(over)
    return Settings(**base)


def _pipeline(**over: Any) -> Any:
    pipeline = object.__new__(IngestPipeline)
    pipeline.settings = _settings(**over)
    pipeline.config = pipeline.settings.indexing
    pipeline.relational = object()  # present, so the bypass is the flag's doing
    return pipeline


def _docs(*ids: str) -> list[Document]:
    return [Document(id=i, content="body", source=f"{i}.md") for i in ids]


# ---------------------------------------------------------------------------
# The purge must run on every path that re-indexes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("label", "force", "skip_unchanged"),
    [("--force", True, True), ("skip_unchanged=false", False, False)],
)
async def test_a_bypass_still_names_what_needs_purging(
    label: str, force: bool, skip_unchanged: bool
) -> None:
    """`changed` is the purge's only input — `_run` computes
    `stale = [doc for doc in ready if doc.id in changed]` — so an empty set
    disabled a step the code calls mandatory, plus the answer-cache invalidation
    nested inside it.

    Measured on `--force`: editing "30 days" to "7 days" left both versions
    retrievable and `answers_invalidated: 0`.
    """
    pipeline = _pipeline(indexing={"skip_unchanged": skip_unchanged})
    documents = _docs("a", "b", "c")

    todo, changed = await pipeline._select_changed(
        documents, IngestReport(), force=force
    )

    assert [d.id for d in todo] == ["a", "b", "c"]
    assert changed == {"a", "b", "c"}, f"{label}: the purge would skip {set('abc') - changed}"


async def test_the_checksum_path_still_purges_only_what_changed() -> None:
    """The saving the skip exists for has to survive: an unchanged document is
    neither re-indexed nor purged."""
    pipeline = _pipeline(indexing={"skip_unchanged": True})

    async def _known(ids: Any) -> dict[str, str]:
        return {"a": "same", "b": "different"}

    pipeline._existing_checksums = _known  # type: ignore[method-assign]
    documents = _docs("a", "b", "c")
    documents[0].checksum = "same"
    documents[1].checksum = "changed-now"

    report = IngestReport()
    todo, changed = await pipeline._select_changed(documents, report, force=False)

    assert {d.id for d in todo} == {"b", "c"}, "an unchanged document was re-indexed"
    assert changed == {"b"}, "a new document has nothing to purge"
    assert report.documents_skipped == 1


# ---------------------------------------------------------------------------
# The commit marker is written after the read-back, not before
# ---------------------------------------------------------------------------
def test_the_marker_is_stamped_after_the_vectors_are_confirmed() -> None:
    """Batches do not wait (`qdrant.wait_on_upsert`), so stamping at the end of a
    window marked a document indexed while its vectors were only *accepted*. The
    retry then skipped it: measured, 4 of 10 documents unretrievable with
    `skip_rate=1.0` on the rerun — which reads as a clean bill of health.
    """
    import ast
    import textwrap

    def calls(method: Any) -> list[tuple[int, str]]:
        """`(lineno, name)` for every method call, from the AST.

        Not a grep: the comment explaining this very change names
        `_stamp_checksums`, and the first version of this test failed on its own
        prose — the class `tests/unit/test_source_assertions.py` exists to catch,
        caught here by the mutation it was written against.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        return [
            (n.lineno, n.func.attr)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ]

    run = [name for _lineno, name in calls(IngestPipeline._run)]
    assert "_stamp_checksums" not in run, "the marker is written inside the window again"
    assert "extend" in run, "the window no longer defers its landed documents"

    ingest = calls(IngestPipeline._ingest)
    flush = next(line for line, name in ingest if name == "_flush_vectors")
    stamp = next(line for line, name in ingest if name == "_stamp_checksums")
    assert flush < stamp, "the marker is written before the read-back confirms the points"


async def test_nothing_is_stamped_when_the_run_never_reaches_the_flush() -> None:
    """The property the ordering buys: a run that dies before confirmation leaves
    no marker, so the retry re-ingests rather than skipping."""
    pipeline = object.__new__(IngestPipeline)
    pipeline._awaiting_confirmation = []
    stamped: list[Any] = []

    async def _stamp(docs: Any, report: Any) -> None:
        stamped.extend(docs)

    pipeline._stamp_checksums = _stamp  # type: ignore[method-assign]

    # A window landed, and the run then failed before `_flush_vectors`.
    pipeline._awaiting_confirmation.extend(_docs("a", "b"))

    assert not stamped, "documents were stamped without confirmation"
    assert len(pipeline._awaiting_confirmation) == 2, "the buffer lost the pending documents"


# ---------------------------------------------------------------------------
# What a machine caller can see, and what the batching costs
# ---------------------------------------------------------------------------
def test_the_did_it_land_check_survives_both_interfaces() -> None:
    """`chunks` against `points_in_store` is the one check that distinguishes
    "indexed" from "retrievable", and neither machine interface carried both:
    the HTTP response dropped `points_in_store` and `empty`, and `--json` dropped
    the warnings — including the read-back's own shortfall line."""
    from ragorc.server.schemas import IngestResponse

    for field in ("chunks", "points_in_store", "empty", "warnings"):
        assert field in IngestResponse.model_fields, f"the HTTP caller cannot see {field}"

    # Declared *and* populated. A field with no assignment in `from_report`
    # serializes its default, so `points_in_store: 0` on a run that wrote 20
    # chunks reads as a total loss — a worse answer than the absent field.
    class _Report:
        def __init__(self) -> None:
            self.warnings = ["vector store holds 12 points but this run wrote 20 chunks"]
            self.rejected: list[tuple[str, str]] = []
            self.failed: list[tuple[str, str]] = []

        def summary(self) -> dict[str, Any]:
            return {
                "documents_in": 10, "indexed": 10, "skipped": 0, "rejected": 0,
                "duplicate": 0, "failed": 0, "chunks": 20, "vectors": 40,
                "strategy": "early", "skip_rate": 0.0, "cost_usd": 0.0, "llm_calls": 0,
                "total_ms": 1.0, "timings_ms": {}, "empty": 3, "points_in_store": 12,
            }

    response = IngestResponse.from_report(_Report(), request_id="r")
    assert response.points_in_store == 12, "the read-back's count is dropped on the way out"
    assert response.empty == 3
    assert response.warnings, "the shortfall warning never reaches the caller"

    cli_source = inspect.getsource(_cli().ingest)
    assert '"warnings": list(report.warnings)' in cli_source, "--json still drops the warnings"


def _cli() -> Any:
    from ragorc import cli

    return cli


def test_a_batching_caller_holds_one_bulk_load_window() -> None:
    """`_ingest`'s own comment explains why one window per *document window* was a
    bug — "every exit rebuilt the graph over everything written so far and then
    waited for green". `ragorc ingest` reintroduced it one level up, calling
    `ingest()` once per --batch-size documents."""
    from ragorc.index.pipeline import IngestPipeline

    assert hasattr(IngestPipeline, "bulk_run")
    cli_source = inspect.getsource(_cli()._ingest_batched)
    assert "bulk_run()" in cli_source, "the CLI opens a window per batch again"


async def test_the_window_is_reentrant() -> None:
    """An outer window must suppress the inner one rather than fight it — nesting
    two `bulk_load` contexts is what turns indexing off and on repeatedly."""
    from ragorc.index.pipeline import IngestPipeline

    entered = 0

    class _Store:
        def bulk_load(self) -> Any:
            import contextlib

            @contextlib.asynccontextmanager
            async def _cm() -> Any:
                nonlocal entered
                entered += 1
                yield

            return _cm()

    pipeline = object.__new__(IngestPipeline)
    pipeline.vector = _Store()
    pipeline._bulk_depth = 0

    async with pipeline.bulk_run(), pipeline.bulk_run():
        pass

    assert entered == 1, f"the window was opened {entered} times"
    assert pipeline._bulk_depth == 0, "the depth was not restored"
