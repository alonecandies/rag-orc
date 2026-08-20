"""The harness: run a dataset through a pipeline, then decide whether a change helped.

Two jobs, and the second is the one that is usually got wrong.

Running the dataset
-------------------
A case is a question plus optional labels, and one run of it produces more than a
score: the ranked chunk ids (for the retrieval metrics), the answer text (for the
judged metrics), and the operational facts nobody labels for — latency, cost, LLM
calls, cache hits, whether the pipeline abstained. Those last ones are collected
unconditionally because they are the columns that regress first and the only ones
that need no ground truth at all. A dataset with no labels whatsoever still tells
you that p95 doubled.

The pipeline is reached through an adapter (:func:`as_answer_fn`) rather than an
import. This module deliberately does not import :mod:`ragorc.pipeline`: the
harness is equally valid over a bare retriever, a single generator, a LangGraph
compiled elsewhere or a stub in a test, and an import would both couple the two
packages and make an offline unit test of the harness impossible.

Errors are not zeros
--------------------
A case whose pipeline call raised is recorded as an **error** and excluded from
every quality aggregate. Scoring a crashed run as recall 0.0 conflates "the
retriever missed" with "the process fell over", which are opposite findings with
opposite fixes, and it lets an infrastructure flake masquerade as a quality
regression. ``errors`` is reported next to the metrics so a run that half failed
cannot be read as a run that scored badly.

The same argument applies to the undefined-metric handling inherited from
:mod:`ragorc.eval.retrieval_metrics`: an unlabelled case contributes ``nan``, not
zero, and the retrieval block is omitted entirely when no case carried labels.

Deciding whether a change helped
--------------------------------
Two means are not a comparison. Retrieval metrics over a few hundred questions
have a standard error of a few points, so "recall@10 went from 0.71 to 0.74" is
consistent with both a real improvement and pure noise, and shipping on that
number is how configurations drift. :func:`paired_bootstrap` resamples the
*per-case differences*, which is what the per-query vectors in
:class:`~ragorc.eval.retrieval_metrics.RetrievalReport` exist for.

Pairing is what makes it work at this sample size. Question difficulty dominates
the variance — some questions are hard for every configuration — and the paired
difference cancels it, leaving only the effect of the change. Comparing two
independent means throws that cancellation away and needs several times as many
questions to see the same effect. Pairing is on case *id*, which is derived from
the question text, so it survives a regenerated or re-ordered dataset.

Latency is reported as percentiles, not a mean: the mean hides the tail, and the
tail is what users experience. Quality scores are reported as means, where the
distribution is bounded and the average is the quantity of interest.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import structlog

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ConfigError
from ragorc.core.models import Answer, FloatArray, Usage
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import Timer
from ragorc.eval.answer_metrics import ALL_METRICS, AnswerMetrics, Scorecard, cheap_baseline
from ragorc.eval.dataset import EvalCase, EvalDataset
from ragorc.eval.retrieval_metrics import DEFAULT_KS, RetrievalReport, evaluate_retrieval

log = structlog.get_logger(__name__)

__all__ = [
    "AnswerFn",
    "BootstrapResult",
    "CaseResult",
    "Comparison",
    "EvalRunner",
    "RunReport",
    "as_answer_fn",
    "compare_runs",
    "paired_bootstrap",
]

AnswerFn = Callable[[str], Awaitable[Answer]]
"""What the runner needs from a pipeline: a question in, an :class:`Answer` out."""

DEFAULT_CONCURRENCY = 4
"""Cases in flight. Low on purpose — each one is a full pipeline run that itself
fans out to several stores and LLM calls, so the real concurrency is this times
whatever the pipeline does internally, and the useful ceiling is the LLM
provider's rate limit rather than the CPU."""

BOOTSTRAP_ITERATIONS = 2_000
"""Resamples per comparison. 2,000 is where the percentile interval stops moving
in the third decimal for the few-hundred-case datasets this harness targets;
10,000 costs five times as much and changes no decision."""

_BOOTSTRAP_BLOCK = 256
"""Resamples drawn per block. The whole ``(iterations, n)`` index matrix would be
tens of megabytes on a large dataset for no gain, so it is drawn in blocks."""

_OPERATIONAL_SERIES = ("latency_ms", "cost_usd", "llm_calls")
"""Per-case series that need no labels, and are therefore comparable on any
dataset. Exposed for pairing so an A/B can bootstrap latency and cost with the
same machinery it uses for quality."""


# ---------------------------------------------------------------------------
# Reaching a pipeline without importing one
# ---------------------------------------------------------------------------
@runtime_checkable
class Answerer(Protocol):
    """Anything with an async ``answer(question)``. The pipeline facade shape."""

    async def answer(self, question: str) -> Answer: ...


@runtime_checkable
class Invokable(Protocol):
    """A compiled LangGraph: ``ainvoke(state)`` returning a state mapping."""

    async def ainvoke(self, state: Any) -> Any: ...


def _document_key(chunk: Any) -> str:
    """Identify the source document of a retrieved chunk.

    Falls back through the fields a chunk can carry, most specific first. The
    ``source`` metadata is checked before ``document_id`` on purpose: a
    hand-written eval case names a *file* ("02-expenses-policy.md"), which is what
    a loader puts in ``source``, while ``document_id`` is a content-derived uuid
    the author could not have known. Comparing against the uuid would score every
    case zero and look like a retrieval failure.
    """
    metadata = getattr(chunk, "metadata", None) or {}
    for key in ("source_document", "source", "path", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    return str(getattr(chunk, "document_id", "") or "")


def as_answer_fn(target: Any, *, state: Mapping[str, Any] | None = None) -> AnswerFn:
    """Adapt a pipeline, a graph or a plain coroutine function to :data:`AnswerFn`.

    Three shapes are accepted, checked in this order:

    1. an object with ``ainvoke`` — a compiled LangGraph. Called with
       ``{"question": ..., **state}`` and the ``answer`` channel is read out of the
       result, which is the key :class:`~ragorc.pipeline.state.RAGState` defines.
    2. an object with ``answer`` — a pipeline facade.
    3. a callable — ``await target(question)``.

    Per-run options (``tenant_id``, ``top_k``) are passed only through ``state``,
    for the graph shape where the channel names are part of a published contract.
    For the other two the caller supplies its own closure: guessing keyword names
    that happen to match, and raising ``TypeError`` from inside the harness when
    they do not, is worse than making the binding explicit at the call site.
    """
    extra = dict(state or {})

    if isinstance(target, Invokable):

        async def _from_graph(question: str) -> Answer:
            result = await target.ainvoke({"question": question, **extra})
            return _coerce_answer(result)

        return _from_graph

    if isinstance(target, Answerer):

        async def _from_facade(question: str) -> Answer:
            return _coerce_answer(await target.answer(question))

        return _from_facade

    if callable(target):

        async def _from_callable(question: str) -> Answer:
            return _coerce_answer(await target(question))

        return _from_callable

    raise ConfigError(
        f"cannot evaluate {type(target).__name__}: it has no ainvoke(), no answer() "
        "and is not callable",
        hint="pass a compiled graph, a pipeline, or an async function question -> Answer",
    )


def _coerce_answer(result: Any) -> Answer:
    """Accept an :class:`Answer` or a state mapping that carries one."""
    if isinstance(result, Answer):
        return result
    if isinstance(result, Mapping):
        answer = result.get("answer")
        if isinstance(answer, Answer):
            return answer
        raise ConfigError(
            "pipeline returned a state with no `answer`",
            hint="the graph must write an Answer to its `answer` channel",
            keys=sorted(str(k) for k in result)[:12],
        )
    raise ConfigError(
        f"pipeline returned {type(result).__name__}, not an Answer",
        hint="return ragorc.core.models.Answer, or a state mapping containing one",
    )


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CaseResult:
    """What one case produced, kept whole so an aggregate can be re-derived.

    The :class:`Answer` is retained rather than reduced to numbers: the reason a
    case scored badly is in its text, its citations and its trace, and a harness
    that discards them forces the whole run to be repeated to answer "why".
    """

    case: EvalCase
    answer: Answer | None = None
    latency_ms: float = 0.0
    scorecard: Scorecard | None = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.error == "" and self.answer is not None

    @property
    def retrieved_ids(self) -> tuple[str, ...]:
        """Ranked chunk ids the generator saw, in rank order."""
        if self.answer is None:
            return ()
        return tuple(scored.chunk.id for scored in self.answer.chunks)

    @property
    def retrieved_documents(self) -> tuple[str, ...]:
        """The same ranking collapsed to documents, first occurrence wins.

        Deduplicated on the way through, so three chunks from one document count
        as that document appearing once at its best rank. Without that, recall at
        document level could exceed 1.0 for exactly the reason
        :func:`~ragorc.eval.retrieval_metrics.relevance_matrix` guards against.
        """
        if self.answer is None:
            return ()
        seen: dict[str, None] = {}
        for scored in self.answer.chunks:
            key = _document_key(scored.chunk)
            if key:
                seen.setdefault(key, None)
        return tuple(seen)

    @property
    def expected_documents(self) -> tuple[str, ...]:
        """Document-level ground truth from the case metadata.

        This exists because a **hand-written** eval set cannot carry
        ``expected_chunk_ids``: chunk ids are derived from chunk *content* (see
        :mod:`ragorc.core.ids`), so they are unknowable until after an ingest, and
        they change whenever chunking settings change. Naming the source document
        is something a human author can do, and it is stable.

        Document-level recall is a weaker signal than chunk-level — it cannot tell
        you whether the right *passage* was ranked first — but it answers the
        question you actually ask first when quality drops: did retrieval even
        find the right document? Grading nothing at all, which is what happens
        without this, answers neither.
        """
        meta = self.case.metadata or {}
        raw = (
            meta.get("expected_documents")
            or meta.get("source_documents")
            or meta.get("source_document")
            or meta.get("document_id")
            or ()
        )
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(item) for item in raw if item)

    @property
    def usage(self) -> Usage:
        """The pipeline's bill plus the judges'. Both are real spend on a run."""
        pipeline = self.answer.usage if self.answer is not None else Usage()
        judges = self.scorecard.usage if self.scorecard is not None else Usage()
        return pipeline + judges

    def to_dict(self) -> dict[str, Any]:
        answer = self.answer
        return {
            "id": self.case.id,
            "question": self.case.question,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "retrieved": list(self.retrieved_ids),
            "expected_chunk_ids": list(self.case.expected_chunk_ids),
            "retrieved_documents": list(self.retrieved_documents),
            "expected_documents": list(self.expected_documents),
            "answer": answer.text if answer is not None else "",
            "abstained": bool(answer.abstained) if answer is not None else None,
            "grounded": bool(answer.grounded) if answer is not None else None,
            "groundedness": round(answer.groundedness, 4) if answer is not None else None,
            "confidence": round(answer.confidence, 4) if answer is not None else None,
            "citations": len(answer.citations) if answer is not None else 0,
            "cost_usd": round(self.usage.cost_usd, 6),
            "llm_calls": self.usage.calls,
            "scores": self.scorecard.to_dict() if self.scorecard is not None else {},
        }


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RunReport:
    """Every case's result, the metrics over them, and the pairing keys.

    The per-case vectors are kept, not just their means: ``answer_scores`` and
    ``retrieval.per_query`` hold one value per *scored* case, index-aligned with
    ``scored_ids``. That alignment is the whole point — it is what lets
    :meth:`series` hand :func:`paired_bootstrap` two runs keyed by case id instead
    of two means, and a mean cannot say whether a difference is noise.
    """

    name: str = "run"
    dataset: str = ""
    results: list[CaseResult] = field(default_factory=list)
    retrieval: RetrievalReport | None = None
    document_retrieval: RetrievalReport | None = None
    """Retrieval graded at document granularity. Populated when cases name a
    source document, which is the only retrieval label a hand-written dataset can
    carry — see :attr:`CaseResult.expected_documents`."""
    answer_scores: dict[str, FloatArray] = field(default_factory=dict)
    scored_ids: tuple[str, ...] = ()
    ks: tuple[int, ...] = DEFAULT_KS

    # -- slices ------------------------------------------------------------
    @property
    def ok_results(self) -> list[CaseResult]:
        return [r for r in self.results if r.ok]

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def usage(self) -> Usage:
        return Usage.sum(r.usage for r in self.results)

    # -- aggregates --------------------------------------------------------
    def operational(self) -> dict[str, float]:
        """Metrics that need no labels: rates, latency percentiles, spend.

        Rates are over the cases that ran. Dividing by the dataset size instead
        would let a run with failures report a lower abstention rate than the same
        pipeline with none, which is the wrong sign.
        """
        ok = self.ok_results
        answers = [r.answer for r in ok if r.answer is not None]
        n = len(answers)
        latencies = np.asarray([r.latency_ms for r in ok], dtype=np.float64)
        usage = self.usage
        answered = [a for a in answers if not a.abstained]
        out: dict[str, float] = {
            "items": float(len(self.results)),
            "scored": float(n),
            "errors": float(self.n_errors),
            "abstain_rate": _rate(sum(1 for a in answers if a.abstained), n),
            "grounded_rate": _rate(sum(1 for a in answers if a.grounded), n),
            "groundedness_mean": _mean([a.groundedness for a in answers]),
            "confidence_mean": _mean([a.confidence for a in answers]),
            # Over answers that actually asserted something: an abstention has
            # nothing to cite, and counting it as an uncited answer would make
            # correct abstention look like a citation failure.
            "citation_coverage": _rate(sum(1 for a in answered if a.citations), len(answered)),
            "latency_p50_ms": _percentile(latencies, 50.0),
            "latency_p95_ms": _percentile(latencies, 95.0),
            "cost_usd_total": round(usage.cost_usd, 6),
            "cost_usd_per_query": round(usage.cost_usd / n, 6) if n else 0.0,
            "llm_calls_total": float(usage.calls),
            "cache_hit_rate": _rate(usage.cached, usage.calls),
        }
        return out

    def retrieval_metrics(self) -> dict[str, float]:
        """Empty when no case carried chunk labels.

        Empty rather than zeroed: an unmeasurable recall and a zero recall are
        different findings, and a caller that cannot tell them apart will report
        the second when it has the first.
        """
        if self.retrieval is None or self.retrieval.n_labelled == 0:
            return {}
        return {name: round(value, 4) for name, value in self.retrieval.mean().items()}

    def document_retrieval_metrics(self) -> dict[str, float]:
        """The same metrics at document granularity, prefixed ``doc_``.

        Prefixed rather than merged, because the two are not interchangeable: a
        ``recall@10`` of 1.0 at document level with 0.4 at chunk level is a real
        and useful finding (the right file is always found, the right passage
        often is not), and a single key would hide it.
        """
        report = self.document_retrieval
        if report is None or report.n_labelled == 0:
            return {}
        return {f"doc_{name}": round(value, 4) for name, value in report.mean().items()}

    def answer_metrics(self) -> dict[str, float]:
        """Judged/lexical answer scores, averaged over the cases each applied to."""
        out: dict[str, float] = {}
        for name, values in self.answer_scores.items():
            if values.size == 0 or bool(np.all(np.isnan(values))):
                continue
            out[name] = round(float(np.nanmean(values)), 4)
        return out

    def metrics(self) -> dict[str, float]:
        return {
            **self.operational(),
            **self.retrieval_metrics(),
            **self.document_retrieval_metrics(),
            **self.answer_metrics(),
        }

    # -- pairing -----------------------------------------------------------
    def series(self, metric: str) -> dict[str, float]:
        """``case id -> value`` for one metric, for a paired comparison.

        Covers the retrieval metrics, the answer metrics and the operational
        series, because "did latency get worse" deserves the same significance
        test as "did recall get better" and there is no reason to bootstrap them
        with different code.
        """
        if metric in _OPERATIONAL_SERIES:
            return {r.case.id: _operational_value(r, metric) for r in self.results if r.ok}
        if self.retrieval is not None and metric in self.retrieval.per_query:
            values = self.retrieval.per_query[metric]
            return dict(zip(self.scored_ids, (float(v) for v in values), strict=True))
        if metric in self.answer_scores:
            values = self.answer_scores[metric]
            return dict(zip(self.scored_ids, (float(v) for v in values), strict=True))
        raise KeyError(f"no series named {metric!r}; have {sorted(self.series_names())}")

    def series_names(self) -> list[str]:
        names = list(_OPERATIONAL_SERIES)
        if self.retrieval is not None:
            names.extend(self.retrieval.per_query)
        names.extend(self.answer_scores)
        return names

    # -- rendering ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dataset": self.dataset,
            # The ks are part of the result, not of the caller's memory: a results
            # file that does not say which cutoffs it reported cannot be compared
            # with one that used different ones.
            "ks": list(self.ks),
            "operational": self.operational(),
            "retrieval": self.retrieval_metrics(),
            "answer": self.answer_metrics(),
            "retrieval_detail": self.retrieval.to_dict() if self.retrieval is not None else {},
            "errors": [{"id": r.case.id, "error": r.error} for r in self.results if not r.ok],
        }

    def to_markdown(self) -> str:
        ops = self.operational()
        lines = [
            f"### {self.name} — {int(ops['items'])} case(s), {int(ops['errors'])} error(s)",
            "",
            f"latency p50 **{ops['latency_p50_ms']:.0f}ms** · "
            f"p95 **{ops['latency_p95_ms']:.0f}ms** · "
            f"${ops['cost_usd_total']:.4f} total, "
            f"${ops['cost_usd_per_query']:.4f}/query · "
            f"{int(ops['llm_calls_total'])} call(s), "
            f"{ops['cache_hit_rate']:.0%} cached",
            "",
            f"abstained {ops['abstain_rate']:.0%} · grounded {ops['grounded_rate']:.0%} "
            f"(mean {ops['groundedness_mean']:.3f}) · "
            f"citations on {ops['citation_coverage']:.0%} of answers",
        ]
        retrieval = self.retrieval_metrics()
        if retrieval and self.retrieval is not None:
            lines += ["", self.retrieval.to_markdown()]
        else:
            lines += ["", "_no chunk labels: retrieval metrics not computed_"]
        answers = self.answer_metrics()
        if answers:
            lines += ["", "| answer metric | score |", "|---|---|"]
            lines += [f"| {name} | {value:.3f} |" for name, value in sorted(answers.items())]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------
class EvalRunner:
    """Runs a dataset against one pipeline and scores what comes back.

    The judged metrics are optional. Without ``metrics=``, cases that carry a
    reference answer still get the deterministic lexical baseline — free, offline,
    zero variance — which is enough for the CI regression check that an LLM-judged
    suite is too slow and too noisy to serve.

    No per-case cost ledger is installed. The ceilings in
    :class:`~ragorc.core.telemetry.CostLedger` are per-*query* limits sized for the
    request path; an eval case is a query *plus* its judges, so enforcing the
    request-path ceiling here would abort legitimate cases and turn a budget knob
    into flaky measurements. The run's bound is the size of the dataset, and the
    bill is aggregated from the ``Usage`` every call already returns.
    """

    name = "eval_runner"

    def __init__(
        self,
        target: Any,
        settings: Settings | None = None,
        *,
        metrics: AnswerMetrics | None = None,
        metric_names: Sequence[str] = ALL_METRICS,
        ks: Sequence[int] = DEFAULT_KS,
        concurrency: int = DEFAULT_CONCURRENCY,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.answer_fn = as_answer_fn(target, state=state)
        self.metrics = metrics
        self.metric_names = tuple(metric_names)
        self.ks = tuple(sorted({int(k) for k in ks if int(k) > 0})) or DEFAULT_KS
        self.concurrency = max(int(concurrency), 1)

    async def run(
        self,
        dataset: EvalDataset | Sequence[EvalCase],
        *,
        name: str = "",
        limit: int | None = None,
    ) -> RunReport:
        """Score every case, then aggregate.

        Cases run concurrently and independently: one failing case is recorded and
        the rest of the run continues, because a harness that aborts on the first
        error is a harness that never finishes on a dataset large enough to matter.
        """
        cases, source = _as_cases(dataset)
        if limit is not None:
            cases = cases[:limit]
        if not cases:
            log.warning("eval_run_empty", dataset=source)
            return RunReport(name=name or source or "run", dataset=source, ks=self.ks)

        outcomes = await bounded_gather(
            (self._run_case(case) for case in cases),
            limit=self.concurrency,
            return_exceptions=True,
        )
        results: list[CaseResult] = []
        for case, outcome in zip(cases, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                # _run_case already traps its own failures, so reaching here means
                # the harness itself broke on this case. Still not fatal to the run.
                results.append(CaseResult(case=case, error=_describe(outcome)))
            else:
                results.append(outcome)

        report = self._aggregate(results, name=name or source or "run", source=source)
        log.info(
            "eval_run_finished",
            run=report.name,
            dataset=source,
            metrics=report.metrics(),
        )
        return report

    # -- one case ----------------------------------------------------------
    async def _run_case(self, case: EvalCase) -> CaseResult:
        timer = Timer(f"eval:{case.id}")
        try:
            answer = await self.answer_fn(case.question)
        except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
            timer.stop()
            log.warning("eval_case_failed", case_id=case.id, error=str(exc)[:200])
            return CaseResult(case=case, latency_ms=timer.elapsed_ms, error=_describe(exc))
        latency_ms = timer.stop()

        scorecard: Scorecard | None = None
        try:
            scorecard = await self._score(case, answer)
        except Exception as exc:  # noqa: BLE001 - a judge failure is not a case failure
            log.warning("eval_case_scoring_failed", case_id=case.id, error=str(exc)[:200])
        return CaseResult(case=case, answer=answer, latency_ms=latency_ms, scorecard=scorecard)

    async def _score(self, case: EvalCase, answer: Answer) -> Scorecard | None:
        if self.metrics is not None:
            return await self.metrics.evaluate(
                case.question,
                answer.text,
                answer.chunks,
                reference=case.expected_answer,
                metrics=self.metric_names,
            )
        if not case.has_reference or "lexical_overlap" not in self.metric_names:
            return None
        card = Scorecard()
        card.add(await cheap_baseline(answer.text, case.expected_answer))
        return card

    # -- aggregation -------------------------------------------------------
    def _aggregate(self, results: list[CaseResult], *, name: str, source: str) -> RunReport:
        scored = [r for r in results if r.ok]
        retrieval: RetrievalReport | None = None
        document_retrieval: RetrievalReport | None = None
        if scored:
            retrieval = evaluate_retrieval(
                [list(r.retrieved_ids) for r in scored],
                [r.case.relevant_ids for r in scored],
                ks=self.ks,
            )
            # Graded separately rather than as a fallback: a dataset can carry
            # both, and when it does, the two answer different questions — chunk
            # level says whether the right passage ranked well, document level
            # says whether the right file was found at all.
            document_labelled = [r for r in scored if r.expected_documents]
            if document_labelled:
                document_retrieval = evaluate_retrieval(
                    [list(r.retrieved_documents) for r in document_labelled],
                    [set(r.expected_documents) for r in document_labelled],
                    ks=self.ks,
                )
                log.info(
                    "document_level_retrieval_graded",
                    cases=len(document_labelled),
                    of=len(scored),
                    reason="cases carry a source document but no chunk labels",
                )

        # One column per metric over the scored cases, nan where the metric did not
        # apply to that case — which keeps the vectors aligned with `scored_ids` so
        # a comparison can pair them, and keeps a skipped metric out of the mean.
        names: list[str] = []
        for result in scored:
            if result.scorecard is None:
                continue
            names.extend(n for n in result.scorecard.scores if n not in names)
        answer_scores = {
            metric: np.asarray(
                [_scorecard_value(r.scorecard, metric) for r in scored], dtype=np.float32
            )
            for metric in names
        }

        return RunReport(
            name=name,
            dataset=source,
            results=results,
            retrieval=retrieval,
            document_retrieval=document_retrieval,
            answer_scores=answer_scores,
            scored_ids=tuple(r.case.id for r in scored),
            ks=self.ks,
        )


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------
#: Substrings identifying metrics where a *lower* value is the better outcome.
#: Matched as substrings rather than exact names so the operational series
#: ("latency_ms_p95", "cost_usd") and any future prefix are covered without a
#: second list to keep in step.
_LOWER_IS_BETTER = (
    "latency",
    "cost",
    "usd",
    "tokens",
    "duration",
    "error_rate",
    "n_errors",
)


def lower_is_better(metric: str) -> bool:
    """Whether a decrease in ``metric`` is an improvement.

    Abstention rate is deliberately absent. A rise can mean retrieval regressed
    *or* that the guardrails correctly started declining questions the corpus
    cannot answer, and those are opposite conclusions — so it is reported as a
    change and left to a human, rather than labelled better or worse.
    """
    name = metric.lower()
    return any(token in name for token in _LOWER_IS_BETTER)


@dataclass(slots=True)
class BootstrapResult:
    """A paired difference with an interval around it.

    ``verdict`` is derived from the interval, not from the point estimate: an
    improvement whose interval straddles zero has not been demonstrated, and the
    only honest report of it is "inconclusive, run more cases".
    """

    metric: str
    n_pairs: int
    baseline: float = float("nan")
    candidate: float = float("nan")
    difference: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    p_value: float = float("nan")
    confidence: float = 0.95

    @property
    def significant(self) -> bool:
        if not np.isfinite(self.ci_low) or not np.isfinite(self.ci_high):
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def verdict(self) -> str:
        """Direction of the change, oriented by what the metric measures.

        Not every metric improves by going up. Latency, cost and token count are
        better when they fall, so a bare ``difference > 0`` reports a slower and
        more expensive pipeline as an improvement — which is precisely backwards
        for the two numbers most likely to decide whether a change ships.
        """
        if self.n_pairs == 0:
            return "no_pairs"
        if not self.significant:
            return "inconclusive"
        improved = self.difference < 0 if lower_is_better(self.metric) else self.difference > 0
        return "better" if improved else "worse"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "pairs": self.n_pairs,
            "baseline": _round_or_none(self.baseline),
            "candidate": _round_or_none(self.candidate),
            "difference": _round_or_none(self.difference),
            "ci": [_round_or_none(self.ci_low), _round_or_none(self.ci_high)],
            "p_value": _round_or_none(self.p_value),
            "verdict": self.verdict,
        }


def paired_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    metric: str = "metric",
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap the mean per-case difference between two runs of the same dataset.

    Both arguments are ``case id -> value``, so only the cases *both* runs scored
    are compared and the pairing survives a reordered or partially failed run.
    Pairs where either side is ``nan`` are dropped: the metric was undefined for
    that case in at least one run, and an undefined value has no difference.

    The interval is the percentile interval of the resampled mean difference, and
    the p-value is its two-sided achieved significance level. Seeded, because an
    unseeded significance test gives a different answer every time it is run and
    invites re-rolling until the number is the desired one.
    """
    shared = [cid for cid in baseline if cid in candidate]
    pairs = np.asarray(
        [(baseline[cid], candidate[cid]) for cid in shared], dtype=np.float64
    ).reshape(-1, 2)
    usable = ~np.isnan(pairs).any(axis=1)
    pairs = pairs[usable]
    n = int(pairs.shape[0])
    if n == 0:
        return BootstrapResult(metric=metric, n_pairs=0, confidence=confidence)

    base_values, cand_values = pairs[:, 0], pairs[:, 1]
    diffs = cand_values - base_values
    observed = float(diffs.mean())
    result = BootstrapResult(
        metric=metric,
        n_pairs=n,
        baseline=float(base_values.mean()),
        candidate=float(cand_values.mean()),
        difference=observed,
        confidence=confidence,
    )
    if n == 1:
        # One pair says nothing about variance, and a bootstrap over a single point
        # can only redraw that point. The difference is *observed*, not
        # demonstrated, so the interval is left undefined and the verdict comes out
        # inconclusive — the alternative would call every one-case A/B significant.
        result.p_value = 1.0
        return result
    if float(diffs.std()) == 0.0:
        # Every pair moved by exactly the same amount: there is no within-sample
        # variation to resample, so every bootstrap mean equals the observed
        # difference and the interval is legitimately degenerate.
        result.ci_low = result.ci_high = observed
        result.p_value = 0.0 if observed != 0.0 else 1.0
        return result

    rng = np.random.default_rng(seed)
    means = np.empty(max(int(iterations), 1), dtype=np.float64)
    filled = 0
    while filled < means.size:
        block = min(_BOOTSTRAP_BLOCK, means.size - filled)
        idx = rng.integers(0, n, size=(block, n))
        means[filled : filled + block] = diffs[idx].mean(axis=1)
        filled += block

    tail = (1.0 - confidence) / 2.0
    result.ci_low = float(np.quantile(means, tail))
    result.ci_high = float(np.quantile(means, 1.0 - tail))
    # Two-sided achieved significance level: how often the resampled difference
    # lands on the other side of zero from the observed one.
    share_le = float(np.mean(means <= 0.0))
    share_ge = float(np.mean(means >= 0.0))
    result.p_value = float(min(1.0, 2.0 * min(share_le, share_ge)))
    return result


@dataclass(slots=True)
class Comparison:
    """One baseline against one candidate, metric by metric."""

    baseline: str
    candidate: str
    results: dict[str, BootstrapResult] = field(default_factory=dict)

    def __getitem__(self, metric: str) -> BootstrapResult:
        return self.results[metric]

    def wins(self) -> list[str]:
        return [m for m, r in self.results.items() if r.verdict == "better"]

    def losses(self) -> list[str]:
        return [m for m, r in self.results.items() if r.verdict == "worse"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "wins": self.wins(),
            "losses": self.losses(),
            "metrics": {m: r.to_dict() for m, r in self.results.items()},
        }

    def to_markdown(self) -> str:
        lines = [
            f"### {self.candidate} vs {self.baseline}",
            "",
            "| metric | baseline | candidate | Δ | 95% CI | p | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for metric, r in self.results.items():
            lines.append(
                f"| {metric} | {r.baseline:.3f} | {r.candidate:.3f} | {r.difference:+.3f} | "
                f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.p_value:.3f} | {r.verdict} |"
            )
        lines += [
            "",
            "A verdict of `inconclusive` means the interval includes zero — the "
            "difference is consistent with noise at this sample size, not that the "
            "two configurations are equal.",
        ]
        return "\n".join(lines)


def compare_runs(
    baseline: RunReport,
    candidate: RunReport,
    *,
    metrics: Sequence[str] | None = None,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Paired A/B over every metric both runs produced.

    Defaults to the metrics the two runs *share*: comparing a metric only one side
    computed is not a comparison, and silently reporting it against nothing is how
    a missing judge turns into an apparent regression.
    """
    if metrics is None:
        shared = [m for m in baseline.series_names() if m in set(candidate.series_names())]
    else:
        shared = list(metrics)

    results: dict[str, BootstrapResult] = {}
    for metric in shared:
        results[metric] = paired_bootstrap(
            baseline.series(metric),
            candidate.series(metric),
            metric=metric,
            iterations=iterations,
            confidence=confidence,
            seed=seed,
        )
    comparison = Comparison(baseline=baseline.name, candidate=candidate.name, results=results)
    log.info(
        "eval_compare_finished",
        baseline=baseline.name,
        candidate=candidate.name,
        wins=comparison.wins(),
        losses=comparison.losses(),
    )
    return comparison


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_cases(dataset: EvalDataset | Sequence[EvalCase]) -> tuple[list[EvalCase], str]:
    if isinstance(dataset, EvalDataset):
        return list(dataset.cases), dataset.name
    return list(dataset), ""


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:300]


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _mean(values: Sequence[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def _percentile(values: np.ndarray, q: float) -> float:
    """Latencies are accumulated in float64: a millisecond timing summed and
    quantiled in float32 loses resolution exactly where the tail is."""
    return round(float(np.percentile(values, q)), 2) if values.size else 0.0


def _round_or_none(value: float) -> float | None:
    return None if not np.isfinite(value) else round(float(value), 4)


def _scorecard_value(card: Scorecard | None, metric: str) -> float:
    if card is None or metric not in card.scores:
        return float("nan")
    return card.scores[metric].score


def _operational_value(result: CaseResult, metric: str) -> float:
    if metric == "latency_ms":
        return result.latency_ms
    if metric == "cost_usd":
        return result.usage.cost_usd
    return float(result.usage.calls)
