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

    async with _staged_uploads(_multipart_request([("doc.md", b"# Handbook\n")]), settings) as (
        body,
        staged_root,
    ):
        assert body.paths == [str(staged_root)]
        await service.ingest(body, staged_root=staged_root)
        assert engine.targets == [staged_root.resolve()]

        with pytest.raises(ValidationFailed):
            await service.ingest(body)


# ---------------------------------------------------------------------------
# A real multipart request
# ---------------------------------------------------------------------------
# Not a double. The doubles this replaces implemented ``form()`` directly and
# returned canned uploads, which meant the test never reached Starlette's parser
# — and the byte ceiling is enforced *around* that parser, on the ASGI receive
# channel, so a double that hands back a finished form cannot observe it. It
# would also have passed with the cap deleted.
_BOUNDARY = "ragorc-test-boundary"


def _multipart_body(files: list[tuple[str, bytes]]) -> bytes:
    parts = []
    for filename, payload in files:
        parts.append(
            f"--{_BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n".encode()
            + payload
            + b"\r\n"
        )
    parts.append(f"--{_BOUNDARY}--\r\n".encode())
    return b"".join(parts)


def _multipart_request(files: list[tuple[str, bytes]], *, chunk: int = 8192) -> Any:
    """A real ``starlette.requests.Request`` over a real multipart body.

    Deliberately sends **no** ``Content-Length``. That is the shape the
    middleware's ceiling cannot see — it reads the declared length and a chunked
    request declares nothing — so it is the shape the wire cap has to handle, and
    building it any other way would test the easy case.
    """
    from starlette.requests import Request

    body = _multipart_body(files)
    chunks = [body[i : i + chunk] for i in range(0, len(body), chunk)] or [b""]
    total = len(chunks)
    served = []

    async def receive() -> Any:
        if chunks:
            served.append(1)
            return {"type": "http.request", "body": chunks.pop(0), "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/ingest",
        "headers": [(b"content-type", f"multipart/form-data; boundary={_BOUNDARY}".encode())],
    }
    built = Request(scope, receive)
    # How much of the body the server actually pulled off the wire. The whole
    # point of a streaming cap is that this stays small, and it is the only
    # observable that separates "refused while it streams" from "refused after
    # the body landed" — both of which raise, and both of which say
    # ``max_body_bytes``.
    built.chunks_served = served  # type: ignore[attr-defined]
    built.chunks_total = total  # type: ignore[attr-defined]
    return built


async def test_an_upload_is_refused_while_it_streams_not_after_it_lands(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling has to bite before the bytes are somewhere.

    Three layers let this through. The middleware reads ``Content-Length``, which
    a chunked upload never sends. Starlette's own ``max_part_size`` applies only
    where ``file is None``, so it bounds a form *field* and not an uploaded file.
    And ``_staged_uploads`` then called ``await value.read()`` — the whole part
    into memory — and compared the running total to the budget afterwards, so the
    check ran after the allocation it existed to prevent.
    """
    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    tiny = Settings(**{**settings.model_dump(), "server": {"max_body_bytes": 4096}})
    request = _multipart_request([("big.md", b"x" * 200_000)], chunk=1024)

    with pytest.raises(ValidationFailed, match=r"^upload exceeds server\.max_body_bytes"):
        async with _staged_uploads(request, tiny):
            pass  # pragma: no cover - the body never gets this far

    # The assertion that distinguishes the two caps. Both raise and both name
    # ``max_body_bytes``, so a test that only checks for a raise passes with the
    # wire cap deleted — the per-part check picks it up instead, after the entire
    # body has been parsed and spooled. Refusing *while it streams* means the
    # remaining ~195 chunks were never asked for.
    assert request.chunks_total > 190, "the body must be large enough for this to mean something"
    assert len(request.chunks_served) <= 8, (
        f"read {len(request.chunks_served)} of {request.chunks_total} chunks before refusing; "
        "the ceiling is being applied after the body landed, not while it arrives"
    )


async def test_an_upload_within_the_ceiling_still_lands(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: the cap must not reject what it is sized to allow."""
    monkeypatch.delenv(_INGEST_ROOTS_ENV, raising=False)
    payload = b"y" * 50_000
    async with _staged_uploads(_multipart_request([("ok.md", payload)], chunk=1024), settings) as (
        _body,
        staged_root,
    ):
        assert (staged_root / "ok.md").read_bytes() == payload


async def test_a_part_is_copied_in_bounded_chunks(tmp_path: Path) -> None:
    """``_copy_part`` is the second layer, and it reads a bounded window.

    The wire cap necessarily trips first when both are set to
    ``max_body_bytes`` — the envelope is always larger than the parts it carries
    — so this drives the helper directly rather than through a request that can
    never reach it.
    """
    from ragorc.server.app import _copy_part

    class _Part:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload
            self.reads: list[int] = []
            self._offset = 0

        async def read(self, size: int = -1) -> bytes:
            self.reads.append(size)
            window = self._payload[self._offset : self._offset + size]
            self._offset += len(window)
            return window

    part = _Part(b"z" * (3 << 20))
    written = await _copy_part(part, tmp_path / "out.bin", budget=1 << 30, written=0)

    assert written == 3 << 20
    assert (tmp_path / "out.bin").stat().st_size == 3 << 20
    assert max(part.reads) == 1 << 20, "the whole part must never be requested at once"

    with pytest.raises(ValidationFailed, match="max_body_bytes"):
        await _copy_part(_Part(b"z" * 4096), tmp_path / "capped.bin", budget=100, written=0)


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


# ---------------------------------------------------------------------------
# The CLI is not a remote caller
# ---------------------------------------------------------------------------
async def test_a_local_eval_dataset_loads_without_configuring_ingest_roots(tmp_path) -> None:  # noqa: ANN001
    """`ragorc eval <file>` — the exact command in `make eval`, the README and
    docs/modules/eval.md — failed with "ingesting a server-side path is disabled".

    The allowlist exists because `POST /eval` takes a *caller-supplied* path and
    quotes the dataset back, which made it an arbitrary file read. That reasoning
    does not transfer to a CLI: the path came from the operator's own shell, and
    they can already read their own files. Confining it there turned a security
    fix for the HTTP surface into a broken headline command.
    """
    from ragorc.server.app import load_eval_items
    from ragorc.server.schemas import EvalRequest

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id": "q1", "question": "how long do refunds take?"}\n')
    request = EvalRequest(dataset=str(dataset))

    cases = await load_eval_items(request, roots=[tmp_path])

    assert len(cases) == 1
    assert cases[0].question == "how long do refunds take?"


async def test_the_http_surface_still_refuses_an_unconfined_dataset(tmp_path) -> None:  # noqa: ANN001
    """The confinement must survive for the caller it was written for: with no
    roots configured, a server-side path is still refused."""
    from ragorc.core.errors import ValidationFailed
    from ragorc.server.app import load_eval_items
    from ragorc.server.schemas import EvalRequest

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id": "q1", "question": "anything"}\n')

    with pytest.raises(ValidationFailed) as caught:
        await load_eval_items(EvalRequest(dataset=str(dataset)), roots=[])
    assert "server-side path" in str(caught.value)


def test_the_eval_command_trusts_the_path_its_operator_typed(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """The loader's `roots` is an argument with a safe default, so a CLI that
    forgets to pass it silently gets the server's policy back. Asserted here for
    the same reason the packer's `shares` wiring is."""
    from typer.testing import CliRunner

    from ragorc import cli

    dataset = tmp_path / "questions.jsonl"
    dataset.write_text('{"id": "q1", "question": "anything"}\n')
    seen: dict[str, object] = {}

    async def _recording(request, *, roots=None):  # noqa: ANN001, ANN202
        seen["roots"] = roots
        raise RuntimeError("stop here — the wiring is what is under test")

    monkeypatch.setattr(cli, "load_eval_items", _recording)
    CliRunner().invoke(cli.app, ["eval", str(dataset)])

    roots = seen.get("roots")
    assert roots, f"the CLI passed no roots, so the server policy applies: {roots!r}"
    assert tmp_path.resolve() in [p.resolve() for p in roots]  # type: ignore[union-attr]
