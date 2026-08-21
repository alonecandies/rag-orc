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
from typing import Any, ClassVar

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


def test_error_detail_redacts_a_provider_body_quoted_under_a_harmless_key() -> None:
    """The scrubber drops credential-shaped *keys*; this is the other half.

    A provider 402 arrives as prose under `body`, carrying a key-management URL
    and the operator's account id. Nothing about the key name says "secret", so
    without value-level redaction the whole thing becomes the response body.
    """
    from ragorc.server.app import _safe_detail

    out = _safe_detail(
        {
            "body": (
                "Insufficient credit. Visit "
                "https://openrouter.ai/workspaces/default/keys/0f1e2d3c4b5a69788796a5b4c3d2e1f0"
            ),
            "candidates": ['{"user_id":"user_EXAMPLEaccountEXAMPLE00000"}'],
            "api_key": "sk-or-v1-dropped-by-the-key-rule",
            "sql": "SELECT id FROM orders",
        }
    )
    assert "0f1e2d3c4b5a69788796a5b4c3d2e1f0" not in out["body"]
    assert "Insufficient credit" in out["body"], "the actionable part must survive"
    assert "user_EXAMPLEaccountEXAMPLE00000" not in out["candidates"][0]
    assert "api_key" not in out, "credential-shaped keys are still dropped whole"
    assert out["sql"] == "SELECT id FROM orders", "ordinary detail is untouched"


async def test_provider_body_is_redacted_where_it_is_captured() -> None:
    """Redacting at the capture site is what makes every sink safe at once."""
    from ragorc.llm.openrouter import OpenRouterLLM

    detail = OpenRouterLLM._provider_detail(
        '{"error":"see https://openrouter.ai/workspaces/default/keys/deadbeefcafe1234",'
        '"user_id":"user_EXAMPLEaccountEXAMPLE00000"}'
    )
    assert "deadbeefcafe1234" not in detail
    assert "user_EXAMPLEaccountEXAMPLE00000" not in detail


# ---------------------------------------------------------------------------
# The other two doors into the filesystem and into memory
# ---------------------------------------------------------------------------
async def test_eval_dataset_is_confined_to_the_same_roots_as_ingest(
    settings: Settings, corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`/eval`'s `dataset` is a caller-supplied server-side path too.

    It was read with `Path(location).expanduser()` and no confinement at all,
    while an eval response quotes the dataset's contents back — the same
    arbitrary-read-with-a-read-back-channel that `/ingest` was fixed for, through
    a different door.
    """
    from ragorc.server.app import load_eval_items

    outside = tmp_path / "outside.jsonl"
    outside.write_text('{"question": "q", "expected_answer": "a"}\n')
    inside = corpus / "dataset.jsonl"
    inside.write_text('{"question": "q", "expected_answer": "a"}\n')

    class _Req:
        def __init__(self, dataset: str) -> None:
            self.dataset = dataset
            self.items: list[Any] = []

    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    assert len(await load_eval_items(_Req(str(inside)))) == 1, "a confined path must still work"

    with pytest.raises(ValidationFailed) as exc:
        await load_eval_items(_Req(str(outside)))
    assert "outside" in str(exc.value).lower() or "root" in str(exc.value).lower()

    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    with pytest.raises(ValidationFailed):
        await load_eval_items(_Req(str(inside)))


async def test_an_oversized_eval_dataset_is_refused_before_it_is_read(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is read whole into a process that is also serving queries."""
    from ragorc.server import app as app_module

    big = corpus / "big.jsonl"
    big.write_text('{"question": "q", "expected_answer": "a"}\n' * 200)
    monkeypatch.setenv(_INGEST_ROOTS_ENV, str(corpus))
    monkeypatch.setattr(app_module, "_MAX_EVAL_DATASET_BYTES", 64)

    class _Req:
        dataset = str(big)
        items: ClassVar[list[Any]] = []

    with pytest.raises(ValidationFailed) as exc:
        await app_module.load_eval_items(_Req())
    assert "too large" in str(exc.value)


async def test_a_chunked_body_is_rejected_before_it_is_all_in_memory() -> None:
    """`Content-Length` is a claim, and a chunked request makes none.

    The middleware's early check keys off that header, so it never fires here,
    and the only remaining limit ran on a body already materialized by
    `await request.body()`. The read now stops at the ceiling instead.
    """
    from ragorc.server.app import _read_bounded

    sent: list[int] = []

    class _Streaming:
        async def stream(self):  # noqa: ANN202
            for _ in range(100):
                sent.append(1)
                yield b"x" * 1024

    with pytest.raises(ValidationFailed) as exc:
        await _read_bounded(_Streaming(), 4096)
    assert exc.value.detail.get("limit_bytes") == 4096
    assert len(sent) < 100, f"stopped after {len(sent)} chunks, not the whole body"


async def test_a_body_within_the_bound_is_returned_whole() -> None:
    from ragorc.server.app import _read_bounded

    class _Streaming:
        async def stream(self):  # noqa: ANN202
            yield b'{"a":'
            yield b"1}"

    assert await _read_bounded(_Streaming(), 4096) == b'{"a":1}'
