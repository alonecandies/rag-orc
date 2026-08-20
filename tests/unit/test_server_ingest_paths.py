"""What ``POST /ingest`` is allowed to read off the server's filesystem.

The endpoint takes three kinds of input and only one of them is dangerous.
Inline ``text`` and multipart uploads carry their own bytes. A ``paths`` entry
does not: it asks the *service* to open a file, and every chunk retrieved later
comes back through ``QueryResponse.chunks[]`` verbatim. With no confinement that
composition is an arbitrary local file read with a read-back channel, reachable
by anyone who can open the port (``server.api_keys`` is empty by default).

So these tests pin the boundary rather than the plumbing: an unset allowlist
refuses every server-side path, a configured allowlist admits exactly what is
under it — resolved, so ``..`` and symlinks cannot walk out — and both of the
transports that carry their own bytes keep working regardless. The service is
driven directly with a recording engine: none of this needs a store, and the
question being asked is which paths reach the ingester, not what it does with
them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from ragorc.core.errors import ValidationFailed
from ragorc.core.settings import Settings
from ragorc.index.pipeline import IngestReport
from ragorc.server.app import (
    _INGEST_ROOTS_ENV,
    RagService,
    _ingest_roots,
    _resolve_paths,
    _staged_uploads,
)
from ragorc.server.schemas import IngestRequest


@pytest.fixture
def settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "k"},
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A directory shaped like a deployment's document root."""
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "doc.md").write_text("# Handbook\n\nRefunds take 14 days.\n")
    (tmp_path / "secret.md").write_text("not for the corpus")
    return root


class RecordingEngine:
    """An ingester that records its targets. Ingest is not what is under test."""

    def __init__(self) -> None:
        self.targets: list[Any] = []

    async def ingest(self, targets: Any) -> IngestReport:
        self.targets = list(targets)
        return IngestReport(documents_in=len(self.targets), documents_indexed=len(self.targets))


def _service(settings: Settings) -> tuple[RagService, RecordingEngine]:
    service = RagService(settings)
    engine = RecordingEngine()
    service.engine = engine
    return service, engine


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------
def test_no_configured_root_refuses_every_server_side_path(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default has to be closed.

    Before the allowlist existed, ``{"paths": ["/etc"]}`` was indexed and read
    back out through ``/query``. An operator who has not named a root has not
    consented to the service reading anything, so a path that exists and is
    perfectly readable is still refused.
    """
    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    assert _ingest_roots() == []

    for probe in (Path("/etc"), Path("/etc/hosts"), corpus / "sub" / "doc.md"):
        with pytest.raises(ValidationFailed) as caught:
            _resolve_paths([str(probe)], roots=_ingest_roots())
        assert _INGEST_ROOTS_ENV in str(caught.value), "the refusal must say how to allow it"


def test_paths_under_a_configured_root_are_accepted(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    assert _resolve_paths([str(corpus / "sub" / "doc.md")], roots=_ingest_roots()) == [
        (corpus / "sub" / "doc.md").resolve()
    ]
    # The root itself is inside the root: ingesting a whole directory is the
    # normal case, and `is_relative_to` is true of a path against itself.
    assert _resolve_paths([str(corpus)], roots=_ingest_roots()) == [corpus.resolve()]


def test_several_roots_can_be_configured(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One deployment can have a corpus and a drop box. ``os.pathsep``, like
    ``PATH``, because that is the separator an operator already knows."""
    other = tmp_path / "dropbox"
    other.mkdir()
    (other / "note.txt").write_text("dropped")
    monkeypatch.setenv(_INGEST_ROOTS_ENV, os.pathsep.join([str(corpus), str(other)]))

    resolved = _resolve_paths(
        [str(corpus / "sub" / "doc.md"), str(other / "note.txt")], roots=_ingest_roots()
    )
    assert resolved == [(corpus / "sub" / "doc.md").resolve(), (other / "note.txt").resolve()]


def test_traversal_out_of_the_root_is_rejected(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``root/sub/../../secret.md`` is a path *inside* the root only by spelling.
    Comparing before resolving is what makes a prefix check decorative."""
    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    escape = corpus / "sub" / ".." / ".." / "secret.md"
    assert escape.exists(), "the traversal target must exist, or existence would refuse it"
    with pytest.raises(ValidationFailed):
        _resolve_paths([str(escape)], roots=_ingest_roots())


def test_a_symlink_out_of_the_root_is_rejected(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink is the traversal a prefix check cannot see: the link lives under
    the root and the bytes do not."""
    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    link = corpus / "outside.md"
    link.symlink_to(tmp_path / "secret.md")
    with pytest.raises(ValidationFailed):
        _resolve_paths([str(link)], roots=_ingest_roots())


def test_a_path_that_does_not_exist_still_fails_that_way(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confinement first, existence second — a typo inside the root must still
    report the typo rather than the boundary."""
    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    with pytest.raises(ValidationFailed) as caught:
        _resolve_paths([str(corpus / "typo.md")], roots=_ingest_roots())
    assert "does not exist" in str(caught.value)


# ---------------------------------------------------------------------------
# Through the service: what the endpoint actually does
# ---------------------------------------------------------------------------
async def test_service_refuses_a_caller_supplied_path_by_default(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    service, engine = _service(settings)
    with pytest.raises(ValidationFailed):
        await service.ingest(IngestRequest(paths=[str(corpus / "sub" / "doc.md")]))
    assert engine.targets == [], "nothing may reach the ingester once a path is refused"


async def test_service_ingests_a_path_inside_a_configured_root(
    settings: Settings, corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    service, engine = _service(settings)
    response = await service.ingest(IngestRequest(paths=[str(corpus)]))
    assert engine.targets == [corpus.resolve()]
    assert response.documents_in == 1


async def test_inline_text_needs_no_configured_root(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confinement must not cost the transport that carries its own bytes:
    inline text never touches the filesystem, so it is unaffected by the roots."""
    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    service, engine = _service(settings)
    await service.ingest(IngestRequest(text="Refunds take 14 days.", source="policy"))
    assert [getattr(doc, "content", None) for doc in engine.targets] == ["Refunds take 14 days."]


async def test_uploads_still_work_and_only_through_their_staging_root(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The upload transport asks for a path — its own temporary directory.

    That is the one path input no caller chose, so it is passed to the service as
    ``staged_root`` instead of being allowlisted globally. The second half of this
    test is the part that matters: the same request is refused without it, so the
    upload path is open because it was granted, not because the check leaks.
    """
    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    service, engine = _service(settings)

    async with _staged_uploads(
        _FakeRequest([_FakeUpload("doc.md", b"# Handbook\n")]), settings
    ) as (
        body,
        staged_root,
    ):
        assert body.paths == [str(staged_root)]
        await service.ingest(body, staged_root=staged_root)
        assert engine.targets == [staged_root.resolve()]

        with pytest.raises(ValidationFailed):
            await service.ingest(body)


# ---------------------------------------------------------------------------
# Multipart doubles. Starlette's form API, only the three members used.
# ---------------------------------------------------------------------------
class _FakeUpload:
    def __init__(self, filename: str, payload: bytes) -> None:
        self.filename = filename
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


class _FakeForm:
    def __init__(self, files: list[_FakeUpload]) -> None:
        self._files = files
        self.closed = False

    def getlist(self, key: str) -> list[_FakeUpload]:
        return self._files if key == "files" else []

    def get(self, key: str, default: Any = None) -> Any:
        return default

    async def close(self) -> None:
        self.closed = True


class _FakeRequest:
    def __init__(self, files: list[_FakeUpload]) -> None:
        self._form = _FakeForm(files)

    async def form(self) -> _FakeForm:
        return self._form
