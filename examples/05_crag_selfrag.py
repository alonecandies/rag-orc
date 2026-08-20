#!/usr/bin/env python
"""CRAG and Self-RAG — the two feedback loops, and the abstention at the end.

What this demonstrates
    A RAG query is not a chain, it is a state machine with two loops, and they grade
    different things:

    * **CRAG** grades the *retrieved documents*. Its verdict is ``CORRECT`` /
      ``AMBIGUOUS`` / ``INCORRECT``, and on anything but CORRECT it rewrites the question
      for a search engine and falls back to the web. It also does the step most
      implementations skip — the **knowledge strip**: an otherwise-relevant document is
      cut into strips, the strips are graded, and only the relevant ones are reassembled.
      ``strip_retention`` in the report below is how you tell whether refinement is
      earning its calls on your corpus.
    * **Self-RAG** grades the *generated answer*, for support (ISSUP) and usefulness
      (ISUSE), **concurrently** — they are independent judgements about the same text, so
      serializing them doubles the verification latency for nothing. On failure it
      rewrites for the specific failure that occurred: an ungrounded answer means the
      model outran its evidence, an unuseful one means the evidence was about the wrong
      thing, and those need different rewrites.

    Two questions are asked. The first the corpus can answer, as a control. The second it
    cannot — and the point of the example is what the system does about it: grade the
    documents INCORRECT, try the web, and if nothing supports an answer, **abstain**.

    Abstention is a success state. A system that always answers cannot signal inadequate
    evidence, so its worst outputs are indistinguishable from its best. Both loops here
    terminate in a refusal rather than in the least-bad ungrounded attempt.

Services needed
    * Qdrant on ``localhost:6333`` / ``:6334``  —  ``docker compose up -d qdrant``
    * ``RAGORC_LLM__API_KEY``                   —  https://openrouter.ai/keys
    * Optional: the ``[web]`` extra for a real web fallback. Without it the run still
      works — the null web retriever reports a step that returned nothing, which is a
      legitimate outcome and is printed as such.

Cost
    Grading is high-volume and cheap by construction (``fast_model``, ADR-0005): expect
    roughly 15-30 calls and a few cents for the whole run. The per-question spend is
    printed so the loops' cost is visible rather than assumed.

Run
    python examples/05_crag_selfrag.py
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
from ragorc.generate.answer import AnswerGenerator
from ragorc.generate.self_rag import SelfRAG
from ragorc.index.loaders import load
from ragorc.index.split import build_splitter
from ragorc.llm.openrouter import OpenRouterLLM
from ragorc.retrieve.crag import CorrectiveRAG
from ragorc.retrieve.hybrid import HybridRetriever
from ragorc.retrieve.web import make_web_retriever
from ragorc.stores.qdrant.store import QdrantStore

CORPUS = Path(__file__).parent / "corpus"
COLLECTION = "ragorc_crag_demo"

QUESTIONS: list[tuple[str, str]] = [
    (
        "the corpus answers this — the control",
        "What is the SEV-1 response time on the Gold support plan?",
    ),
    (
        "the corpus does not contain this, and every noun in it does",
        "How many weeks of paid parental leave does Halcyon Data offer?",
    ),
]


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
    settings.generation.max_answer_tokens = 400

    rs = settings.retrieval
    rs.crag_enabled = True
    rs.crag_grade_top_k = 4
    rs.crag_web_fallback = True

    gen = settings.generation
    gen.self_rag_enabled = True
    gen.self_rag_max_retries = 1
    # Both loops are only as good as the grader under them, and the abstention gate is
    # what turns a grade into a decision. Turning it off here would make the second
    # question produce a confident wrong answer, which is the whole failure being shown.
    gen.allow_abstention = True
    gen.check_groundedness = True
    return settings


async def index(settings: Settings) -> QdrantStore:
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
    return store


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_crag(report: dict[str, Any]) -> None:
    print(
        f"  CRAG verdict   : {report['action']}  "
        f"({report['relevant']}/{report['graded']} documents relevant, "
        f"threshold {report['threshold']}, max confidence {report['max_confidence']})"
    )
    if report["ungraded"]:
        print(
            f"  ungraded       : {report['ungraded']} (ranked below crag_grade_top_k, "
            "so never judged — marked, not silently mixed in)"
        )
    if report["strips_total"]:
        print(
            f"  knowledge strip: kept {report['strips_kept']}/{report['strips_total']} strips "
            f"(retention {report['strip_retention']}) across "
            f"{report['refined_documents']} document(s)"
        )
    if report["rewritten"]:
        print("  rewrote the query for the web (a corpus query and a web query differ)")
    print(f"  web results    : {report['web_results']}")
    if report["errors"]:
        print(f"  degraded       : {report['errors']}")
    print(f"  grader calls   : {report['grader_calls']}")


def print_answer(answer: Answer, ledger: Any) -> None:
    print(f"\n  {answer.text.strip()}\n")
    if answer.abstained:
        print(
            f"  ABSTAINED at gate '{answer.metadata.get('abstain_gate')}' — {answer.abstain_reason}"
        )
        rejected = answer.metadata.get("rejected_answer")
        if rejected:
            # Retained rather than discarded: it is the most useful artifact for
            # diagnosing why the pipeline declined, and a caller must opt in to see it.
            print(f"  rejected draft : {rejected.strip()[:140]!r}")
    print(
        f"  grounded={answer.grounded} ({answer.groundedness:.2f})  "
        f"confidence={answer.confidence:.2f}  citations={len(answer.citations)}"
    )
    unsupported = answer.metadata.get("unsupported_claims") or []
    contradicted = answer.metadata.get("contradicted_claims") or []
    for claim in contradicted[:2]:
        print(f"  CONTRADICTED   : {claim[:110]!r}")
    for claim in unsupported[:2]:
        print(f"  unsupported    : {claim[:110]!r}")
    bill = ledger.report()
    print(f"  cost ${bill['total_cost_usd']:.5f} over {bill['calls']} call(s)")
    for stage, spend in sorted(bill["by_stage"].items()):
        print(f"    {stage:26} {spend['calls']:>2} call(s)  ${spend['cost_usd']:.5f}")


# ---------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------
async def run_question(
    settings: Settings,
    crag: CorrectiveRAG,
    generator: AnswerGenerator,
    self_rag: SelfRAG,
    why: str,
    question: str,
) -> None:
    print("\n" + "=" * 78)
    print(f"Q: {question}\n   ({why})")
    print("=" * 78)

    # The loops take plain callables rather than objects, which is what keeps them
    # composable — and testable with stubs. ``retrieve`` here is CRAG, so Self-RAG's
    # retries re-enter the *graded* retriever rather than the raw one.
    async def retrieve(query: Query) -> RetrievalResult:
        result, _usage = await crag.run(query, top_k=settings.retrieval.top_k)
        crag_report = query.metadata.get("crag")
        if crag_report:
            print_crag(crag_report)
        return result

    async def answer_fn(query: Query, retrieval: RetrievalResult) -> Answer:
        return await generator.generate(query, retrieval)

    cost = settings.cost
    with new_request_context(
        request_id="crag-selfrag",
        max_cost_usd=cost.max_cost_per_query_usd,
        max_calls=cost.max_llm_calls_per_query,
        max_tokens=cost.max_tokens_per_query,
    ) as (_trace, ledger):
        query = Query(text=question, top_k=settings.retrieval.top_k)
        result = await self_rag.run(query, retrieve, answer_fn)

    trail = result.report()
    print(
        f"\n  Self-RAG       : {trail['iterations']} iteration(s), "
        f"accepted at {trail['accepted_at']}, abstained={trail['abstained']}"
    )
    for attempt in trail["trail"]:
        print(
            f"    iteration {attempt['i']}: grounded={attempt['grounded']} "
            f"useful={attempt['useful']} -> {attempt['verdict']}"
        )
    print_answer(result.answer, ledger)


async def main() -> None:
    settings = configure()
    configure_logging(settings.observability.log_level, json_logs=False)
    await preflight(settings)

    store = await index(settings)
    llm = OpenRouterLLM(settings.llm)
    web = make_web_retriever(settings)
    print(
        f"web fallback: provider={settings.retrieval.web_search_provider!r} "
        f"enabled={getattr(web, 'enabled', True)}"
    )
    try:
        hybrid = HybridRetriever(store, settings=settings)
        crag = CorrectiveRAG(hybrid, llm, settings, web=web)
        generator = AnswerGenerator(llm, settings)
        self_rag = SelfRAG(llm, settings)
        for why, question in QUESTIONS:
            await run_question(settings, crag, generator, self_rag, why, question)
    finally:
        await llm.aclose()
        await store.close()

    print(
        "\nThe second question is the one to read. Every noun in it exists in the corpus\n"
        "and the fact does not, which is exactly the case a similarity-only retriever\n"
        "scores highest. CRAG grades those documents, Self-RAG grades the answer written\n"
        "from them, and the run ends in a refusal instead of an invented number."
    )


if __name__ == "__main__":
    asyncio.run(main())
