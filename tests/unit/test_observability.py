"""What a dashboard would key on, and whether it is measuring the right thing.

Never audited. Three of the four settings that shape observability described
something the code did not do:

* `slow_query_ms` was compared against the *sum of the trace steps*. Steps nest, so
  a parent's time is counted again in each child, and concurrent legs are added
  rather than overlapped — a fast fan-out could exceed the threshold, and a
  genuinely slow query with tracing off measured zero and never fired.
* `/metrics` counted only the success path, so an outage read as *traffic
  stopped* — which is what a healthy quiet period looks like — and a streamed
  deployment reported nothing at all.
* `trace_enabled` is a *privacy* control (a step trace records what each stage did
  with the retrieved passages and is attached to every Answer). Five wirings
  honoured it and the LangChain adapter did not.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# slow_query_ms
# ---------------------------------------------------------------------------
def test_the_slow_threshold_measures_the_query() -> None:
    """Not the sum of its steps. Asserted over the AST, because the comment
    explaining the change necessarily mentions the thing it replaced."""
    from ragorc.pipeline.builder import RAGPipeline

    tree = ast.parse(textwrap.dedent(inspect.getsource(RAGPipeline._finish)))
    assigned = [
        ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "elapsed_ms" for t in node.targets)
    ]
    assert assigned, "no elapsed_ms is computed"
    expression = assigned[0]
    assert "current_trace" not in expression, f"still summing the trace: {expression}"
    assert "started" in expression, expression


def test_the_threshold_is_reached_with_tracing_off() -> None:
    """The step sum is empty when tracing is off, so the warning could not fire on
    the deployments most likely to want it."""
    from ragorc.pipeline.builder import RAGPipeline

    source = inspect.getsource(RAGPipeline._finish)
    assert "started" in inspect.signature(RAGPipeline._finish).parameters
    assert "slow_query_ms" in source


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------
class _Counter:
    def __init__(self) -> None:
        self.counted: list[dict[str, Any]] = []

    def labels(self, **kw: Any) -> Any:
        self.counted.append(kw)
        return self

    def inc(self) -> None:
        pass


class _Hist:
    def __init__(self) -> None:
        self.values: list[float] = []

    def observe(self, value: float) -> None:
        self.values.append(value)


class _Metrics:
    def __init__(self) -> None:
        self.queries = _Counter()
        self.latency = _Hist()
        self.cost = _Hist()
        self.groundedness = _Hist()


@pytest.mark.parametrize("outcome", ["failed", "timeout", "cancelled"])
def test_a_query_that_produced_no_response_is_still_counted(outcome: str) -> None:
    """An outage that stops the counter is indistinguishable from a quiet night,
    and it is the opposite of what an alert should fire on."""
    from ragorc.server.app import PipelineName, _record_outcome

    metrics = _Metrics()
    _record_outcome(metrics, PipelineName.NAIVE, outcome, 0.25)

    assert metrics.queries.counted == [{"pipeline": "naive", "outcome": outcome}]
    assert metrics.latency.values == [0.25], "how long it took before failing is the signal"
    assert not metrics.cost.values, "there is no answer to attribute a cost to"
    assert not metrics.groundedness.values


def test_both_query_endpoints_record_an_outcome() -> None:
    """`/query/stream` reached no metric at all, so an SSE-only deployment
    reported zero traffic while serving every request."""
    from ragorc.server import app as app_module

    source = inspect.getsource(app_module.create_app)
    for handler in ("query(", "query_stream("):
        start = source.index(f"async def {handler}")
        body = source[start : start + 4000]
        assert "_record_outcome(" in body, f"{handler} records no outcome"


def test_the_stream_records_every_terminal_state() -> None:
    """Including a client hanging up: a deployment where most streams are
    abandoned is one an operator wants to see, and it is invisible in the
    answered count."""
    from ragorc.server import app as app_module

    source = inspect.getsource(app_module.create_app)
    start = source.index("async def query_stream(")
    body = source[start : start + 4000]
    for state in ('"cancelled"', '"failed"', '"streamed"'):
        assert state in body, f"the stream never reports {state}"
    assert "finally:" in body, "the outcome is not recorded on every path out"


# ---------------------------------------------------------------------------
# trace_enabled
# ---------------------------------------------------------------------------
def test_every_wiring_honours_trace_enabled() -> None:
    """A privacy control that five of six wirings respect is a privacy control an
    operator cannot rely on."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2] / "ragorc"
    callers = [
        path
        for path in sorted(root.rglob("*.py"))
        if "new_request_context(" in path.read_text() and path.name != "telemetry.py"
    ]
    assert len(callers) >= 4, f"the scan found too few callers: {[p.name for p in callers]}"

    for path in callers:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "new_request_context"
            ):
                continue
            names = {kw.arg for kw in node.keywords}
            assert "trace" in names, (
                f"{path.name}:{node.lineno} opens a context without honouring trace_enabled"
            )
