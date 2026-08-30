"""Settings that report a feature as configured and change nothing.

`indexing.raptor_collapse_tree` was removed in round fifteen and an invariant
added — scoped to `raptor_*`. Widening the same AST walk to all 268 settings
found `retrieval.hybrid_enabled` in seconds: the library's headline switch, listed
in `docs/modules/retrieve.md` beside `use_dense`/`use_sparse`/`use_fulltext` under
"which legs run", reported in `/health`, and read by nothing:

    hybrid_enabled=True  legs_that_ran=['hybrid'] hits=4
    hybrid_enabled=False legs_that_ran=['hybrid'] hits=4

A guard written to describe the bug just fixed is usually narrower than the bug's
class.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest

from ragorc.core.settings import Settings

#: Fields whose only readers are dict *values* — a request body or a pool kwarg,
#: not a report. The AST walk cannot tell those apart from a `describe()` entry,
#: so they are named here with the reader that makes each behavioural.
_DICT_VALUED_READERS = {
    "allow_fallbacks": "openrouter.py provider preferences body",
    "require_parameters": "openrouter.py provider preferences body",
    "data_collection": "openrouter.py provider preferences body",
    "prepare_threshold": "postgres/pool.py connection kwargs",
}


def _leaf_fields() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name, field in Settings.model_fields.items():
        annotation = field.annotation
        if hasattr(annotation, "model_fields"):
            for leaf in annotation.model_fields:
                out.setdefault(leaf, set()).add(name)
    return out


def _behavioural_readers(fields: set[str]) -> set[str]:
    """Every settings field read somewhere that is not a log call or a dict value."""
    found: set[str] = set()

    class Reader(ast.NodeVisitor):
        def __init__(self) -> None:
            self.reporting = 0

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            is_log = (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "log"
            )
            self.reporting += is_log
            self.generic_visit(node)
            self.reporting -= is_log

        def visit_Dict(self, node: ast.Dict) -> None:
            for key in node.keys:
                if key is not None:
                    self.visit(key)
            for value in node.values:
                plain = isinstance(value, ast.Attribute | ast.List | ast.Tuple | ast.Set)
                self.reporting += plain
                self.visit(value)
                self.reporting -= plain

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if not self.reporting and node.attr in fields:
                found.add(node.attr)
            self.generic_visit(node)

    root = pathlib.Path(__file__).resolve().parents[2] / "ragorc"
    for path in root.rglob("*.py"):
        if path.name == "settings.py":
            continue
        Reader().visit(ast.parse(path.read_text()))
    return found


def test_no_setting_is_reported_without_being_read() -> None:
    """The invariant, across every section — not just the one that last broke."""
    fields = _leaf_fields()
    behavioural = _behavioural_readers(set(fields))
    inert = sorted(
        f for f in fields if f not in behavioural and f not in _DICT_VALUED_READERS
    )
    assert not inert, (
        "these settings have no reader outside a log line or a describe() dict, "
        f"so they report a feature as configured and change nothing: {inert}"
    )


@pytest.mark.parametrize("field", sorted(_DICT_VALUED_READERS))
def test_the_allowlist_entries_still_have_their_reader(field: str) -> None:
    """An allowlist that outlives its justification is how an inert setting hides.
    Each entry names the dict that reads it; this checks the name is still true."""
    import re

    root = pathlib.Path(__file__).resolve().parents[2] / "ragorc"
    pattern = re.compile(rf'["\']{field}["\']\s*:\s*[\w.]*\b{field}\b')
    hits = [p for p in root.rglob("*.py") if p.name != "settings.py" and pattern.search(p.read_text())]
    assert hits, f"{field} is allowlisted as a dict value and is no longer read as one"


# ---------------------------------------------------------------------------
# hybrid_enabled, specifically
# ---------------------------------------------------------------------------
def _settings(**over: Any) -> Settings:
    base: dict[str, Any] = {
        "llm": {"api_key": "k"},
        "cache": {"enabled": False},
        "embedding": {"dense_dimension": 32},
        "security": {"enforce_tenant_isolation": False},
    }
    base.update(over)
    return Settings(**base)


class _RecordingClient:
    """A Qdrant client that records which named vectors a search asked for."""

    def __init__(self, *names: str) -> None:
        self.names = names
        self.requests: list[dict[str, Any]] = []

    async def get_collection(self, name: str) -> Any:
        vectors = {n: object() for n in self.names if n != "sparse"}
        sparse = {n: object() for n in self.names if n == "sparse"}
        params = type("P", (), {"vectors": vectors, "sparse_vectors": sparse})()
        return type("I", (), {"config": type("C", (), {"params": params})()})()

    async def query_points(self, **kw: Any) -> Any:
        self.requests.append(kw)
        return type("R", (), {"points": []})()

    def used(self) -> set[str]:
        """Every named vector the searches actually named."""
        names: set[str] = set()
        for request in self.requests:
            if request.get("using"):
                names.add(request["using"])
            for branch in request.get("prefetch") or []:
                if getattr(branch, "using", None):
                    names.add(branch.using)
        return names


def _retriever(client: Any, **over: Any) -> Any:
    from ragorc.retrieve.hybrid import HybridRetriever
    from ragorc.stores.qdrant.store import QdrantStore
    from tests.fakes import StubEmbedder, StubSparseEmbedder

    settings = _settings(retrieval=over)
    store = QdrantStore(
        settings, dense_embedder=StubEmbedder(32), sparse_embedder=StubSparseEmbedder()
    )
    store._client = client
    return HybridRetriever(store, settings=settings), settings


@pytest.mark.parametrize(
    ("hybrid", "expect_sparse"), [(True, True), (False, False)], ids=["on", "off"]
)
async def test_hybrid_enabled_decides_which_legs_run(hybrid: bool, expect_sparse: bool) -> None:
    """The documented meaning: hybrid off means one leg, not a fusion of several.

    Asserted on the named vectors the searches actually requested, because that
    is the observable difference — the flag used to change nothing at all.
    """
    from ragorc.core.models import Query
    from ragorc.stores.qdrant.collections import DENSE_VECTOR, SPARSE_VECTOR

    client = _RecordingClient(DENSE_VECTOR, SPARSE_VECTOR)
    retriever, _ = _retriever(
        client, hybrid_enabled=hybrid, use_sparse=True, server_side_fusion=False
    )

    await retriever.retrieve_detailed(Query(text="q"), top_k=3)

    used = client.used()
    assert DENSE_VECTOR in used, f"the dense leg must always run: {sorted(used)}"
    assert (SPARSE_VECTOR in used) is expect_sparse, f"hybrid={hybrid} used {sorted(used)}"


async def test_an_explicit_override_still_wins() -> None:
    """`hybrid_enabled` narrows the *defaults*, the same precedence the three
    finer flags have. A per-call `use_sparse=True` must not be overruled by a
    deployment-level switch."""
    from ragorc.core.models import Query
    from ragorc.stores.qdrant.collections import DENSE_VECTOR, SPARSE_VECTOR

    client = _RecordingClient(DENSE_VECTOR, SPARSE_VECTOR)
    retriever, _ = _retriever(
        client, hybrid_enabled=False, use_sparse=True, server_side_fusion=False
    )

    await retriever.retrieve_detailed(Query(text="q"), top_k=3, use_sparse=True)

    assert SPARSE_VECTOR in client.used(), "the explicit override was ignored"


# ---------------------------------------------------------------------------
# log_prompts, specifically
# ---------------------------------------------------------------------------
def _audit(**over: Any) -> Any:
    from ragorc.security.audit import AuditLog

    return AuditLog(_settings(observability=over, security={"audit_log_enabled": True}))


def _records(monkeypatch: pytest.MonkeyPatch, audit: Any, call: Any) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(audit, "record", lambda event: seen.append(event.to_record()))
    call(audit)
    return seen


@pytest.mark.parametrize("enabled", [False, True])
def test_log_prompts_decides_whether_the_question_is_recorded(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    """`model_post_init` forces this off in prod, so it had a writer and no
    reader: an operator who turned it on for an incident got the same
    metadata-only lines and no way to tell it had done nothing.
    docs/security.md documents it as a control."""
    audit = _audit(log_prompts=enabled)
    records = _records(
        monkeypatch,
        audit,
        lambda a: a.query(tenant_id="acme", principal="p", question="who approves refunds?"),
    )

    assert records[0]["query_length"] == len("who approves refunds?"), "metadata is unconditional"
    assert ("question" in records[0]) is enabled, records[0]
    if enabled:
        assert records[0]["question"] == "who approves refunds?"


@pytest.mark.parametrize("enabled", [False, True])
def test_log_prompts_decides_whether_the_answer_is_recorded(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    audit = _audit(log_prompts=enabled)
    records = _records(
        monkeypatch,
        audit,
        lambda a: a.answered(
            tenant_id="acme", cost_usd=0.01, chunks=3, grounded=True, answer="the CFO does."
        ),
    )

    assert records[0]["chunks"] == 3
    assert ("answer" in records[0]) is enabled, records[0]


def test_recorded_text_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An audit line is meant to be greppable and shippable; an unbounded answer
    would put a whole context window on one line."""
    from ragorc.security.audit import MAX_LOGGED_CHARS

    audit = _audit(log_prompts=True)
    records = _records(
        monkeypatch,
        audit,
        lambda a: a.answered(
            tenant_id=None, cost_usd=0.0, chunks=0, grounded=True, answer="x" * 99_999
        ),
    )
    assert len(records[0]["answer"]) == MAX_LOGGED_CHARS


def test_the_call_sites_hand_over_the_text_rather_than_its_length() -> None:
    """The policy lives in the log, not in eight callers. A caller that passed
    `len(question)` had already decided, which is why the setting had nowhere to
    be read."""
    import inspect
    import pathlib as _p

    from ragorc.pipeline import builder
    from ragorc.server import app

    for module in (builder, app):
        source = _p.Path(inspect.getfile(module)).read_text()
        assert "length=len(" not in source, f"{module.__name__} still decides for the audit log"


def test_health_reports_the_predicate_not_its_narrowest_flag() -> None:
    """`/health` said `late_interaction: false` while a ColBERT embedder was
    loaded and the stage was running — the last reader of
    `enable_late_interaction`, left behind when the three wirings moved to
    `late_interaction_needed`. A features dict that reports configuration rather
    than capability is how two of these were wrong at once."""
    from ragorc.pipeline.builder import RAGPipeline

    settings = _settings(retrieval={"reranker": "colbert"})
    pipeline = RAGPipeline(settings=settings, llm=object())

    features = pipeline.describe()["features"]

    assert features["late_interaction"] is True, (
        "health reports the stage off while the deployment builds and runs it"
    )
    assert settings.embedding.enable_late_interaction is False, (
        "the narrow flag is still off — which is exactly why reporting it lied"
    )
