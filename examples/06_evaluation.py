#!/usr/bin/env python
"""Evaluation — synthesize a dataset, run it, and A/B two configurations properly.

What this demonstrates
    How to decide a configuration question with a number instead of a demo. Three parts,
    in the order the numbers depend on each other:

    1. **A labelled dataset from an unlabelled corpus.**
       :class:`~ragorc.eval.dataset.SyntheticQuestionGenerator` writes questions *from* a
       chunk and records that chunk's id as the retrieval ground truth, then writes a
       vocabulary-controlled **paraphrase** of each question that inherits the same
       label. Both halves are needed: the originals share wording with their source
       chunk, so a lexical retriever looks better on them than it is, and the paraphrases
       are the control that measures what happens when a user does not use your words.

    2. **A free retrieval A/B.** The recall *ceiling* needs no generation at all — only
       the ranked chunk ids — so the first comparison runs a retrieval-only answer
       function and costs nothing beyond the embeddings. That is the cheapest useful
       experiment in this file and the one to run first: if ``recall@fetch_k`` is low, no
       reranker, compressor or prompt will get the pipeline past it.

    3. **A judged answer A/B** over the hand-written cases in
       ``examples/eval/questions.jsonl``, including the two deliberately unanswerable
       ones. Abstention is a success state, so a harness that cannot score it is
       measuring the wrong thing — the abstention table at the end is the check that both
       configurations decline when they should.

    The comparison itself is a **paired bootstrap**, not two means. The pairing (by case
    id) removes the between-question variance that otherwise dominates the estimate, so a
    two-point difference on a few dozen cases can be told apart from noise rather than
    reported as an improvement.

    The two configurations differ in exactly one dimension each time, because an A/B over
    two changes at once tells you nothing about either.

Services needed
    * Qdrant on ``localhost:6333`` / ``:6334``  —  ``docker compose up -d qdrant``
    * ``RAGORC_LLM__API_KEY``                   —  https://openrouter.ai/keys

Cost
    Question generation is one balanced-tier call per source chunk plus one per
    paraphrase; the judged A/B is one synthesis plus a handful of grader calls per case
    per configuration. With the constants below (``GENERATE_FROM=6``, ``JUDGE_CASES=4``)
    the whole run is roughly 60-80 calls — a few tens of cents. Raise them for a real
    experiment; a few hundred cases is where the intervals get tight.

Run
    python examples/06_evaluation.py
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ragorc.cache.tiered import build_cache
from ragorc.core.models import Answer, Query, RetrievalResult
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging, new_request_context
from ragorc.embed.cache import EmbeddingCache
from ragorc.embed.fastembed_provider import FastEmbedDense, FastEmbedSparse
from ragorc.eval.answer_metrics import AnswerMetrics
from ragorc.eval.dataset import EvalDataset, SyntheticQuestionGenerator
from ragorc.eval.runner import EvalRunner, compare_runs
from ragorc.generate.answer import AnswerGenerator
from ragorc.index.loaders import load
from ragorc.index.split import build_splitter
from ragorc.llm.openrouter import OpenRouterLLM
from ragorc.llm.router import ModelTier
from ragorc.retrieve.hybrid import HybridRetriever
from ragorc.retrieve.rerank import build_reranker
from ragorc.stores.qdrant.store import QdrantStore

CORPUS = Path(__file__).parent / "corpus"
QUESTIONS_FILE = Path(__file__).parent / "eval" / "questions.jsonl"
COLLECTION = "ragorc_eval_demo"

GENERATE_FROM = 6
"""Source chunks for the synthetic set. Each yields one question plus one paraphrase.

Deliberately small so the example is affordable. A real experiment wants a few hundred
cases: the bootstrap interval narrows with the square root of the case count, and at a
dozen cases almost nothing is significant."""

JUDGE_CASES = 4
"""Hand-written cases scored by the LLM judge. Each one costs a synthesis plus several
grader calls, *per configuration*, so this is the constant that dominates the bill."""


# ---------------------------------------------------------------------------
# Preflight (repeated per example so each file stands alone)
# ---------------------------------------------------------------------------
def die(*lines: str) -> None:
    print("\n".join(("", *lines, "")), file=sys.stderr)
    raise SystemExit(1)


def endpoint(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlsplit(url if "//" in url else f"//{url}")
    return parsed.hostname or "localhost", parsed.port or default_port


async def reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout_s)
    except (OSError, TimeoutError):
        return False
    writer.close()
    with contextlib.suppress(Exception):
        await writer.wait_closed()
    return True


async def preflight(settings: Settings) -> None:
    problems: list[str] = []
    if not CORPUS.is_dir():
        problems.append(f"The example corpus is missing: {CORPUS}")
    if not QUESTIONS_FILE.is_file():
        problems.append(f"The hand-written eval set is missing: {QUESTIONS_FILE}")
    if not settings.llm.api_key.get_secret_value():
        problems.append(
            "RAGORC_LLM__API_KEY is not set. Get a key at https://openrouter.ai/keys, "
            "then:\n      export RAGORC_LLM__API_KEY=sk-or-v1-...\n"
            "    or copy .env.example to .env and fill it in."
        )
    host, rest_port = endpoint(settings.qdrant.url, 6333)
    port = settings.qdrant.grpc_port if settings.qdrant.prefer_grpc else rest_port
    if not await reachable(host, port):
        problems.append(
            f"Qdrant is not reachable at {host}:{port}.\n      docker compose up -d qdrant"
        )
    if problems:
        die("Cannot run this example:", *(f"\n  - {problem}" for problem in problems))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def configure() -> Settings:
    settings = get_settings().model_copy(deep=True)
    settings.qdrant.collection = COLLECTION
    settings.generation.prompt_name = "answer_default"
    settings.generation.max_answer_tokens = 350
    settings.retrieval.fetch_k = 30
    return settings


async def index(settings: Settings) -> tuple[QdrantStore, Any, list[Any]]:
    cache = EmbeddingCache(build_cache(settings.cache), settings)
    dense = FastEmbedDense(cache=cache, settings=settings)
    sparse = FastEmbedSparse(cache=cache, settings=settings)
    await dense.warmup()

    documents = await load(CORPUS, settings=settings)
    splitter = build_splitter(embedder=dense, settings=settings)
    chunks = await splitter.split_many(documents)
    # The splitter carries no document metadata into its chunks, so the readable name
    # is stamped on here. Worth doing beyond this example's output: the context packer
    # prints ``metadata["source"]`` as each passage's provenance line, and a model that
    # can see where a passage came from writes better attributions than one shown a UUID.
    names = {
        document.id: Path(str(document.source or document.title or "")).name
        for document in documents
    }
    for chunk in chunks:
        chunk.metadata["source"] = names.get(chunk.document_id) or chunk.document_id

    texts = [chunk.embed_text for chunk in chunks]
    dense_vectors, sparse_vectors = await asyncio.gather(
        dense.embed_documents(texts), sparse.embed_documents(texts)
    )
    for chunk, dense_vector, sparse_vector in zip(
        chunks, dense_vectors, sparse_vectors, strict=True
    ):
        chunk.dense = dense_vector
        chunk.sparse = sparse_vector

    store = QdrantStore(settings, dense_embedder=dense, sparse_embedder=sparse)
    await store.ensure_collection(recreate=True)
    await store.upsert(chunks)
    print(f"indexed {len(documents)} documents -> {len(chunks)} chunks in '{COLLECTION}'")
    return store, dense, chunks


# ---------------------------------------------------------------------------
# The two things under test
# ---------------------------------------------------------------------------
def retrieval_only(
    retriever: HybridRetriever, settings: Settings, *, use_sparse: bool, rerank: Any | None
) -> Any:
    """An answer function that retrieves and does **not** generate.

    Legitimate rather than a shortcut: retrieval metrics read only
    ``Answer.chunks``, so a synthesis call would add cost and latency to a
    measurement that cannot use it. The returned ``Answer`` is honest about it —
    empty text, ``abstained=False`` because no decision was made — and
    ``operational()`` will report a zero answer rate, which is exactly right.
    """

    async def answer_fn(question: str) -> Answer:
        query = Query(text=question, top_k=settings.retrieval.top_k)
        result = await retriever.retrieve_detailed(
            query, use_dense=True, use_sparse=use_sparse, use_variants=False
        )
        chunks = result.chunks
        if rerank is not None:
            chunks = await rerank.rerank_chunks(query, chunks, top_k=settings.retrieval.top_k)
        else:
            chunks = chunks[: settings.retrieval.top_k]
        return Answer(text="", chunks=chunks)

    return answer_fn


def full_answer(
    retriever: HybridRetriever,
    generator: AnswerGenerator,
    settings: Settings,
    *,
    use_sparse: bool,
    rerank: Any | None,
) -> Any:
    """The real thing: retrieve, rerank, generate, verify, maybe abstain.

    Each case gets its own request context, so the per-case ledger and trace end up on
    the ``Answer`` the harness keeps — which is what makes ``cost_usd_per_query`` in the
    report a measurement rather than an estimate.
    """
    cost = settings.cost

    async def answer_fn(question: str) -> Answer:
        with new_request_context(
            request_id=f"eval-{abs(hash(question)) % 10**8}",
            max_cost_usd=cost.max_cost_per_query_usd,
            max_calls=cost.max_llm_calls_per_query,
            max_tokens=cost.max_tokens_per_query,
        ):
            query = Query(text=question, top_k=settings.retrieval.top_k)
            result = await retriever.retrieve_detailed(
                query, use_dense=True, use_sparse=use_sparse, use_variants=False
            )
            chunks = result.chunks
            if rerank is not None:
                chunks = await rerank.rerank_chunks(query, chunks, top_k=settings.retrieval.top_k)
            return await generator.generate(query, RetrievalResult(chunks=chunks))

    return answer_fn


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(report: Any, *, keys: tuple[str, ...]) -> None:
    metrics = report.metrics()
    shown = {k: metrics[k] for k in keys if k in metrics}
    print(f"\n  {report.name}")
    for key, value in shown.items():
        print(f"    {key:22} {value:>10.4f}")
    if report.n_errors:
        print(f"    {'errors':22} {report.n_errors:>10}")


def print_abstentions(name: str, report: Any, dataset: EvalDataset) -> None:
    """Abstention is scored as its own outcome, never blended into a mean.

    A case tagged ``unanswerable`` is *supposed* to produce a refusal, so counting it in
    the same average as an answerable one makes correct behaviour look like a failure.
    """
    expected = {case.id for case in dataset.cases if case.metadata.get("answerable") is False}
    if not expected:
        return
    print(f"\n  {name} — abstention on the unanswerable cases")
    for result in report.results:
        if result.case.id not in expected:
            continue
        answer = result.answer
        if answer is None:
            print(f"    {result.case.id}: run failed — {result.error}")
            continue
        verdict = "correct (abstained)" if answer.abstained else "WRONG (answered anyway)"
        print(f"    {result.case.id}: {verdict}")
        if not answer.abstained:
            print(f"      {answer.text.strip()[:110]!r}")


# ---------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------
async def synthesize(settings: Settings, llm: Any, chunks: list[Any]) -> EvalDataset:
    generator = SyntheticQuestionGenerator(
        llm,
        settings,
        questions_per_chunk=1,
        paraphrase=True,
        # Pinned rather than left to the router. A bad *grade* costs one noisy
        # measurement; a bad *label* is permanently wrong and biases every future
        # experiment against this dataset, so this is the one place to pay up.
        model=settings.llm.model,
    )
    dataset, report = await generator.generate(
        chunks, limit=GENERATE_FROM, seed=0, name="synthetic"
    )
    summary = report.summary()
    print(
        f"\nsynthesized {len(dataset)} case(s) from {summary['chunks_used']} chunk(s): "
        f"{summary['questions']} original + {summary['paraphrases']} paraphrase(s)"
    )
    rejected = summary["rejected"]
    print(
        f"  rejected {rejected['unanswerable']} unanswerable, "
        f"{rejected['referential']} referential, {rejected['duplicate']} duplicate"
    )
    print(f"  {summary['llm_calls']} call(s), ${summary['cost_usd']:.4f}")
    return dataset


async def retrieval_ab(
    settings: Settings, retriever: HybridRetriever, dataset: EvalDataset
) -> None:
    print("\n" + "=" * 78)
    print("A/B 1 — dense only vs hybrid + rerank, retrieval metrics, no generation")
    print("=" * 78)

    reranker = build_reranker("cross_encoder", settings=settings)
    # Two different vocabularies, deliberately. `keys` names *aggregates* for the summary
    # tables; `paired` names the per-case *series* a bootstrap can pair by case id —
    # `latency_p50_ms` is a percentile over a run and has no per-case value to pair.
    keys = ("recall@10", "recall@30", "ndcg@10", "mrr", "latency_p50_ms", "cost_usd_total")
    paired = ("recall@10", "recall@30", "ndcg@10", "mrr", "latency_ms")

    # metrics=None: no judge. Every number below is either a rank metric over the
    # dataset's own labels or an operational one, and all of them are free.
    baseline = await EvalRunner(
        retrieval_only(retriever, settings, use_sparse=False, rerank=None),
        settings,
        metrics=None,
        metric_names=(),
        ks=(1, 3, 10, 30),
    ).run(dataset, name="dense only")
    candidate = await EvalRunner(
        retrieval_only(retriever, settings, use_sparse=True, rerank=reranker),
        settings,
        metrics=None,
        metric_names=(),
        ks=(1, 3, 10, 30),
    ).run(dataset, name="hybrid + rerank")

    print_report(baseline, keys=keys)
    print_report(candidate, keys=keys)
    # Named explicitly. Left to itself the comparison covers every metric both runs
    # produced — every k of four rank families — and a table that long buries the four
    # numbers the experiment was about.
    print("\n" + compare_runs(baseline, candidate, metrics=paired, seed=0).to_markdown())

    # The slice that matters most, and the one a single mean hides: the originals share
    # vocabulary with their source chunk and the paraphrases do not, so a configuration
    # that only works on the originals is a configuration that only works on your words.
    for slice_name, subset in (
        ("originals", dataset.slice(paraphrases=False)),
        ("paraphrases", dataset.slice(paraphrases=True)),
    ):
        if not len(subset):
            continue
        run = await EvalRunner(
            retrieval_only(retriever, settings, use_sparse=True, rerank=reranker),
            settings,
            metrics=None,
            metric_names=(),
            ks=(1, 3, 10, 30),
        ).run(subset, name=f"hybrid + rerank [{slice_name}]")
        print_report(run, keys=("recall@10", "ndcg@10", "mrr"))


async def answer_ab(
    settings: Settings,
    retriever: HybridRetriever,
    generator: AnswerGenerator,
    judge: AnswerMetrics,
    dataset: EvalDataset,
) -> None:
    print("\n" + "=" * 78)
    print("A/B 2 — the same two configurations, judged answer quality")
    print("=" * 78)

    reranker = build_reranker("cross_encoder", settings=settings)
    # Cheap and reference-free metrics only. `answer_correctness` and `context_recall`
    # need ground truth and are the expensive pair; add them once the run size justifies
    # the bill.
    metric_names = ("faithfulness", "answer_relevance", "lexical_overlap")
    keys = (
        "faithfulness",
        "answer_relevance",
        "lexical_overlap",
        "abstain_rate",
        "grounded_rate",
        "citation_coverage",
        "cost_usd_per_query",
    )

    baseline = await EvalRunner(
        full_answer(retriever, generator, settings, use_sparse=False, rerank=None),
        settings,
        metrics=judge,
        metric_names=metric_names,
        concurrency=2,
    ).run(dataset, name="dense only", limit=JUDGE_CASES)
    candidate = await EvalRunner(
        full_answer(retriever, generator, settings, use_sparse=True, rerank=reranker),
        settings,
        metrics=judge,
        metric_names=metric_names,
        concurrency=2,
    ).run(dataset, name="hybrid + rerank", limit=JUDGE_CASES)

    print_report(baseline, keys=keys)
    print_report(candidate, keys=keys)
    print(
        "\n"
        + compare_runs(
            baseline, candidate, metrics=(*metric_names, "latency_ms", "cost_usd"), seed=0
        ).to_markdown()
    )


async def abstention_check(
    settings: Settings,
    retriever: HybridRetriever,
    generator: AnswerGenerator,
    dataset: EvalDataset,
) -> None:
    unanswerable = EvalDataset(
        cases=[c for c in dataset.cases if c.metadata.get("answerable") is False],
        name="unanswerable",
    )
    if not len(unanswerable):
        return
    print("\n" + "=" * 78)
    print("The abstention cases — the outcome a mean would hide")
    print("=" * 78)
    reranker = build_reranker("cross_encoder", settings=settings)
    report = await EvalRunner(
        full_answer(retriever, generator, settings, use_sparse=True, rerank=reranker),
        settings,
        metrics=None,
        metric_names=(),
        concurrency=2,
    ).run(unanswerable, name="hybrid + rerank")
    print_abstentions("hybrid + rerank", report, unanswerable)


async def main() -> None:
    settings = configure()
    configure_logging(settings.observability.log_level, json_logs=False)
    await preflight(settings)

    store, dense, chunks = await index(settings)
    llm = OpenRouterLLM(settings.llm)
    try:
        retriever = HybridRetriever(store, settings=settings)
        generator = AnswerGenerator(llm, settings)
        judge = AnswerMetrics(
            llm,
            settings,
            embedder=dense,
            # Pinned explicitly: changing the judge between runs silently re-scales every
            # metric, and the diff then looks like a pipeline regression.
            model=settings.llm.model,
        )
        print(f"judge model pinned to {settings.llm.model!r} ({ModelTier.BALANCED.value} tier)")

        synthetic = await synthesize(settings, llm, chunks)
        if len(synthetic):
            await retrieval_ab(settings, retriever, synthetic)
        else:
            print("no synthetic cases were produced; skipping the retrieval A/B")

        handwritten = await EvalDataset.load(QUESTIONS_FILE)
        print(f"\nloaded {len(handwritten)} hand-written case(s) from {QUESTIONS_FILE.name}")
        answerable = EvalDataset(
            cases=[c for c in handwritten.cases if c.metadata.get("answerable") is not False],
            name="answerable",
        )
        await answer_ab(settings, retriever, generator, judge, answerable)
        await abstention_check(settings, retriever, generator, handwritten)
    finally:
        await llm.aclose()
        await store.close()

    print(
        "\nRead the bootstrap verdicts, not the means. A metric whose interval spans zero\n"
        "did not move: with a few dozen cases most differences are noise, which is the\n"
        "finding, and shipping on one is how a pipeline gets slower for nothing."
    )


if __name__ == "__main__":
    asyncio.run(main())
