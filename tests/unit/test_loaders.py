"""Where untrusted files enter the system.

``ragorc/index/loaders.py`` is 1400 lines and had no dedicated test file. Six
defects came out of the first pass over it, and they share a shape with the rest
of this codebase: each one is a promise the module docstring makes that the code
does not keep.

The two that matter most both end with *an empty corpus and no error* — the
failure you do not notice, because an ingest that reports success over nothing
looks exactly like an ingest of an empty directory.
"""

from __future__ import annotations

import asyncio
import csv
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from ragorc.index.loaders import CSVLoader, DirectoryLoader, JSONLoader


def _corpus(root: Path, n: int = 3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"doc{i}.md").write_text(f"# Document {i}\nRefunds take 30 days.\n")
    return root


# ---------------------------------------------------------------------------
# One bad file must not take the corpus with it
# ---------------------------------------------------------------------------
async def test_a_corrupt_file_does_not_abort_the_directory(tmp_path: Path) -> None:
    """The class docstring says this in bold, and names .docx and .pdf as the
    cases. The catch tuple was ``(OSError, ValueError, TypeError, ImportError,
    ValidationFailed)``; python-docx raises ``PackageNotFoundError`` (Exception ->
    OpcError) and PyMuPDF raises ``FileDataError`` (Exception -> RuntimeError), so
    neither was caught. One renamed file returned zero documents out of twenty and
    recorded zero failures.
    """
    root = _corpus(tmp_path / "corpus")
    (root / "renamed.docx").write_bytes(b"this is not a docx at all")

    loader = DirectoryLoader()
    documents = await loader.load(root)

    assert len(documents) == 3, "the good files were lost with the bad one"
    assert loader.failures, "the failure was swallowed instead of recorded"
    path, reason = loader.failures[0]
    assert path.endswith("renamed.docx")
    assert "PackageNotFoundError" in reason, reason


async def test_a_corrupt_pdf_is_recorded_the_same_way(tmp_path: Path) -> None:
    """The second exception the docstring names. Separate test because it comes
    from a different library with a different base class — ``FileDataError``
    derives from ``RuntimeError``, which the old tuple also missed."""
    pytest.importorskip("pymupdf")
    root = _corpus(tmp_path / "corpus")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 but not really")

    loader = DirectoryLoader()
    documents = await loader.load(root)

    assert len(documents) == 3
    assert any("broken.pdf" in p for p, _ in loader.failures)


async def test_cancellation_is_not_swallowed_as_a_file_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cost of catching broadly, paid for explicitly. A per-file handler that
    also ate ``CancelledError`` would absorb a shutdown once per remaining file and
    turn Ctrl-C into a long wait.

    The failure is injected into the *file loader*, not into ``_load_one``. An
    earlier version of this test overrode ``_load_one`` itself, which meant the
    handler under test never ran — the mutation that removes ``CancelledError``
    from the re-raise survived, because the test was exercising its own subclass.
    """
    root = _corpus(tmp_path / "corpus", n=1)

    class Cancelling:
        name = "cancelling"

        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def load(self, source: Any, **kwargs: Any) -> list[Any]:
            raise asyncio.CancelledError

    monkeypatch.setattr("ragorc.index.loaders.loader_for", lambda suffix: Cancelling)

    with pytest.raises(asyncio.CancelledError):
        await DirectoryLoader().load(root)


async def test_memory_exhaustion_aborts_rather_than_being_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-raise that is actually load bearing.

    ``CancelledError``, ``KeyboardInterrupt`` and ``SystemExit`` derive from
    ``BaseException`` and escape a broad ``except Exception`` on their own — the
    mutation removing them from the tuple changes nothing, correctly. ``MemoryError``
    does not: it is an ``Exception``, so without the explicit re-raise it would be
    logged as one file's problem and the walk would carry on into the next file to
    fail the same way.
    """
    root = _corpus(tmp_path / "corpus", n=3)

    class Exhausted:
        name = "exhausted"

        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def load(self, source: Any, **kwargs: Any) -> list[Any]:
            raise MemoryError("cannot allocate")

    monkeypatch.setattr("ragorc.index.loaders.loader_for", lambda suffix: Exhausted)

    loader = DirectoryLoader()
    with pytest.raises(MemoryError):
        await loader.load(root)
    assert loader.failures == [], "memory exhaustion was recorded as a per-file failure"


# ---------------------------------------------------------------------------
# The skip list prunes the walk, it does not veto the root
# ---------------------------------------------------------------------------
async def test_a_root_under_a_skipped_directory_name_still_loads(tmp_path: Path) -> None:
    """``_admit`` tested ``path.parts`` — the *absolute* path — so a corpus at
    ~/projects/build/corpus matched ``build`` above the root the caller named and
    admitted nothing, reporting ``directory_empty`` for a directory full of
    documents. Fifteen names do this, including env, venv, dist, target and
    .cache.
    """
    root = _corpus(tmp_path / "build" / "corpus")

    documents = await DirectoryLoader().load(root)

    assert len(documents) == 3, "the ancestor's name vetoed the root the caller chose"


@pytest.mark.parametrize("name", ["node_modules", ".git", "__pycache__", "dist"])
async def test_the_skip_list_still_prunes_below_the_root(tmp_path: Path, name: str) -> None:
    """The behaviour that must survive the fix: pruning what the walk *discovers*
    is the point of the list, and it is what keeps a walk from taking minutes."""
    root = _corpus(tmp_path / "corpus")
    junk = root / name
    junk.mkdir()
    (junk / "junk.md").write_text("# not corpus text")

    documents = await DirectoryLoader().load(root)

    assert len(documents) == 3
    assert not any(name in d.source for d in documents)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
async def test_a_staged_upload_keeps_its_identity_across_requests(tmp_path: Path) -> None:
    """Every upload staged into a fresh temporary directory, and identity came
    from the absolute path, so re-uploading the same file minted a new document id
    every time: the checksum skip could never fire and the store accumulated one
    copy per upload, forever."""
    ids, sources = [], []
    for attempt in ("first", "second"):
        staging = tmp_path / f"ragorc-upload-{attempt}"
        staging.mkdir()
        (staging / "handbook.md").write_text("# Handbook\nRefunds take 30 days.")
        documents = await DirectoryLoader(source_root=staging).load(staging)
        ids.append(documents[0].id)
        sources.append(documents[0].source)

    assert sources == ["handbook.md", "handbook.md"]
    assert len(set(ids)) == 1, f"the same file uploaded twice got two ids: {ids}"


async def test_an_ordinary_ingest_still_identifies_by_absolute_path(tmp_path: Path) -> None:
    """The blast radius of the fix, pinned. Relabelling every directory ingest
    would change every existing document's id and re-ingest the whole corpus
    once — which a bug fix may not do — so ``source_root`` is opt-in."""
    root = _corpus(tmp_path / "corpus", n=1)

    documents = await DirectoryLoader().load(root)

    assert documents[0].source == str(root / "doc0.md")


async def test_the_same_file_in_two_staging_dirs_differs_by_name_only(tmp_path: Path) -> None:
    """Sibling files must still be distinguishable — identity is relative, not
    discarded."""
    staging = tmp_path / "staged"
    staging.mkdir()
    (staging / "a.md").write_text("# A")
    (staging / "b.md").write_text("# B")

    documents = await DirectoryLoader(source_root=staging).load(staging)

    assert len({d.id for d in documents}) == 2
    assert {d.source for d in documents} == {"a.md", "b.md"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
async def test_a_newline_inside_a_quoted_cell_survives(tmp_path: Path) -> None:
    """``text.splitlines()`` strips the terminator, so ``csv.reader`` joined the
    fragments of a continued quoted field with nothing between them — the newline
    was not preserved, it was deleted, gluing "para one" to "para two"."""
    path = tmp_path / "multi.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "body"])
        writer.writerow(["1", "para one\npara two"])

    documents = await CSVLoader(mode="row").load(path)

    assert "para one\npara two" in documents[0].content
    assert "onepara" not in documents[0].content


async def test_a_cell_containing_an_exotic_line_break_is_not_torn_in_two(
    tmp_path: Path,
) -> None:
    """``splitlines()`` also breaks on \\x0b, \\x0c, \\x1c-\\x1e, \\x85, U+2028 and
    U+2029, none of which CSV treats as a record separator. A cell holding one was
    silently split into two rows."""
    path = tmp_path / "exotic.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "body"])
        writer.writerow(["1", "before\u2028after"])
        writer.writerow(["2", "plain"])

    documents = await CSVLoader(mode="row").load(path)

    assert len(documents) == 2, f"a cell was torn into extra rows: {len(documents)}"


async def test_json_records_are_built_off_the_event_loop(tmp_path: Path) -> None:
    """CONTRACTS rule 1, and the module docstring's "one ``asyncio.to_thread`` hop
    covering open, read, decode and parse together".

    Read and parse were in threads and the build was not — and the build is the
    expensive part: 9 ms, 24 ms and 505 ms respectively on a 16 MB file. Measured
    as a stall rather than a duration, because what matters is that other requests
    keep being served.
    """
    path = tmp_path / "records.json"
    path.write_text(json.dumps([{"id": str(i), "content": "x" * 200} for i in range(20000)]))

    threads: set[str] = set()

    class Watching(JSONLoader):
        def _record_to_document(self, *args: Any, **kwargs: Any) -> Any:
            threads.add(threading.current_thread().name)
            return super()._record_to_document(*args, **kwargs)

    documents = await Watching().load(path)

    assert len(documents) == 20000
    # Asserted structurally, not as a duration. A timing threshold has to be loose
    # enough not to flake and is then loose enough to pass with the work back on
    # the loop — which is exactly what happened: the first version of this test
    # allowed 150 ms and 20k records built in less than that, so the mutation
    # putting the build back on the loop survived.
    assert threads and "MainThread" not in threads, (
        f"documents were built on the event loop thread: {threads}"
    )
