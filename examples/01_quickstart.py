#!/usr/bin/env python
"""Quickstart — the shortest honest path from text to a cited, costed answer.

What this demonstrates
    The five stages every query goes through, wired by hand so each one is visible:
    split → embed → index (Qdrant) → hybrid retrieve + rerank → generate. The answer
    comes back with resolvable citations, a groundedness score and an itemized bill,
    because :class:`~ragorc.generate.answer.AnswerGenerator` produces all three or
    abstains — none of them is an extra a caller has to remember to ask for.

    The components are wired directly rather than through ``build_pipeline()`` for one
    reason: the full pipeline also writes document rows to Postgres, and this example is
    meant to run with **only Qdrant up**. Once all three stores are running,
    ``examples/03_multi_store.py`` shows the one-line version.

Services needed
    * Qdrant on ``localhost:6333`` / ``:6334``  —  ``docker compose up -d qdrant``
    * ``RAGORC_LLM__API_KEY``                   —  https://openrouter.ai/keys

    No Postgres, no Neo4j. The first run downloads two small ONNX models (~90 MB).

Run
    python examples/01_quickstart.py
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ragorc.cache.tiered import build_cache
from ragorc.core.ids import content_hash, document_id
from ragorc.core.models import Document, Query, RetrievalResult
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging, new_request_context
from ragorc.embed.cache import EmbeddingCache
from ragorc.embed.fastembed_provider import FastEmbedDense, FastEmbedSparse
from ragorc.generate.answer import AnswerGenerator
from ragorc.index.split import build_splitter
from ragorc.llm.openrouter import OpenRouterLLM
from ragorc.retrieve.hybrid import HybridRetriever
from ragorc.retrieve.rerank import build_reranker
from ragorc.stores.qdrant.store import QdrantStore

COLLECTION = "ragorc_quickstart"

QUESTION = "How much does the Graph Add-on cost, and what does it need in order to run?"

# Deliberately three short documents rather than a corpus: the subject here is the
# pipeline's shape, not retrieval quality. Note the second one — its pricing sentence
# starts with "It", and the antecedent is two paragraphs above. That is the case late
# chunking exists for, and examples/02_hybrid_search.py measures it on the real corpus.
DOCUMENTS: list[tuple[str, str]] = [
    (
        "retrieval-engine.md",
        "# Retrieval Engine\n\n"
        "Retrieval Engine indexes a customer's documents and answers questions over "
        "them. A single query runs dense semantic search and BM25 lexical search in one "
        "round trip against the same index, fuses the two rankings, and reranks the "
        "survivors with a cross-encoder before generating a cited answer.\n\n"
        "It costs $1,200 per month per environment. An environment is one isolated "
        "index with its own credentials. Document storage and query volume are "
        "included; embedding throughput is billed separately as Embedding Credits.\n\n"
        "The current release is 3.4.\n",
    ),
    (
        "graph-addon.md",
        "# Graph Add-on\n\n"
        "The Graph Add-on extends Retrieval Engine with entity and relationship "
        "search. Where the base product ranks passages, the add-on builds a graph of "
        "the entities those passages mention and answers a question by traversing it.\n\n"
        "Ingest extracts entities and typed relationships per chunk, resolves the "
        "mentions of one thing into one node, detects communities over the resolved "
        "graph, and writes a summary per community.\n\n"
        "It is billed at $450 per month per seat, and it cannot be bought on its own — "
        "the environment must already be running version 3.0 or later.\n",
    ),
    (
        "support-plans.md",
        "# Support Plans\n\n"
        "Four tiers. Bronze is included with every subscription and responds by the "
        "next business day. Silver costs $300 per month and responds to a SEV-1 within "
        "8 business hours. Gold is priced per contract, adds 24x5 coverage and responds "
        "to a SEV-1 within 2 hours. Platinum is priced per contract, adds 24x7 coverage "
        "and responds to a SEV-1 within 30 minutes.\n\n"
        "Response means a named engineer has acknowledged the ticket and started work. "
        "It is not a resolution commitment, and we do not sell one.\n",
    ),
]


# ---------------------------------------------------------------------------
# Preflight
#
# Repeated in every example on purpose: each file is meant to be readable and
# copy-pastable on its own, and a shared helper would make the first thing a reader
# meets an import they have to go and look up.
# ---------------------------------------------------------------------------
def die(*lines: str) -> None:
    print("\n".join(("", *lines, "")), file=sys.stderr)
    raise SystemExit(1)


def endpoint(url: str, default_port: int) -> tuple[str, int]:
    parsed = urlsplit(url if "//" in url else f"//{url}")
    return parsed.hostname or "localhost", parsed.port or default_port


async def reachable(host: str, port: int, timeout_s: float = 2.0) -> bool:
    """A TCP connect, not a client handshake.

    The point is to fail with one clear sentence *before* any driver is constructed: a
    Qdrant client built against a dead port raises a transport error several frames deep,
    which is precisely the traceback this check exists to replace.
    """
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
    if not settings.llm.api_key.get_secret_value():
        problems.append(
            "RAGORC_LLM__API_KEY is not set. Get a key at https://openrouter.ai/keys, "
            "then:\n      export RAGORC_LLM__API_KEY=sk-or-v1-...\n"
            "    or copy .env.example to .env and fill it in."
        )
    host, port = endpoint(settings.qdrant.url, 6333)
    if not await reachable(host, port):
        problems.append(
            f"Qdrant is not reachable at {host}:{port}.\n      docker compose up -d qdrant"
        )
    if problems:
        die("Cannot run this example:", *(f"\n  - {problem}" for problem in problems))


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Bundle:
    """The objects the ingest side and the query side must *share*.

    Sharing the embedders is not a saving, it is a correctness requirement: the two
    paths have to embed into the same space, and two independently constructed embedders
    with different asymmetric prefixes silently index one space and search another.
    """

    settings: Settings
    dense: Any
    sparse: Any
    store: QdrantStore
    llm: Any

    async def aclose(self) -> None:
        await self.llm.aclose()
        await self.store.close()


def configure() -> Settings:
    """A private copy of the settings, so the demo cannot disturb a real index."""
    settings = get_settings().model_copy(deep=True)
    settings.qdrant.collection = COLLECTION
    # ``generation.prompt_name`` defaults to the shorthand "default", which the pipeline
    # layer expands to a registered prompt name. Driving the generator directly means
    # expanding it here.
    settings.generation.prompt_name = "answer_default"
    return settings


async def build(settings: Settings) -> Bundle:
    cache = EmbeddingCache(build_cache(settings.cache), settings)
    dense = FastEmbedDense(cache=cache, settings=settings)
    sparse = FastEmbedSparse(cache=cache, settings=settings)
    # Loads the ONNX session and pins the true vector width. Before the collection is
    # created, because a collection's dimension is fixed at creation and a mismatch
    # surfaces as an opaque insert error thousands of vectors later.
    await dense.warmup()
    store = QdrantStore(settings, dense_embedder=dense, sparse_embedder=sparse)
    return Bundle(settings, dense, sparse, store, OpenRouterLLM(settings.llm))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
async def index(bundle: Bundle) -> int:
    documents = [
        Document(
            id=document_id(name, text),
            content=text,
            source=name,
            title=name,
            checksum=content_hash(text),
        )
        for name, text in DOCUMENTS
    ]
    splitter = build_splitter(embedder=bundle.dense, settings=bundle.settings)
    chunks = await splitter.split_many(documents)

    # The splitter carries no document metadata into its chunks, so the readable name is
    # stamped on here. Worth doing beyond this example's output: the context packer prints
    # ``metadata["source"]`` as each passage's provenance line, and a model that can see
    # where a passage came from writes better attributions than one shown a UUID.
    names = {document.id: document.source or "" for document in documents}
    for chunk in chunks:
        chunk.metadata["source"] = names.get(chunk.document_id) or chunk.document_id

    # Splitters return boundaries and never vectors (ADR-0002), so embedding is a
    # separate step — which is exactly what makes late chunking possible at all. Both
    # modalities are embedded concurrently: they are independent forward passes.
    texts = [chunk.embed_text for chunk in chunks]
    dense_vectors, sparse_vectors = await asyncio.gather(
        bundle.dense.embed_documents(texts), bundle.sparse.embed_documents(texts)
    )
    for chunk, dense_vector, sparse_vector in zip(
        chunks, dense_vectors, sparse_vectors, strict=True
    ):
        chunk.dense = dense_vector
        chunk.sparse = sparse_vector

    # recreate=True so re-running the quickstart is idempotent in the obvious way. A
    # real ingest never does this: IngestPipeline upserts by content-derived id and
    # purges only the chunks of documents whose checksum actually moved.
    await bundle.store.ensure_collection(recreate=True)
    written = await bundle.store.upsert(chunks)
    print(f"indexed {len(documents)} documents -> {written} chunks in '{COLLECTION}'")
    return written


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
async def ask(bundle: Bundle, question: str) -> None:
    settings = bundle.settings
    retriever = HybridRetriever(bundle.store, settings=settings)
    reranker = build_reranker(settings=settings)
    generator = AnswerGenerator(bundle.llm, settings)

    # Every request runs inside a context. That is what populates the step trace and
    # enforces the cost ceilings *before* each model call rather than after the money is
    # spent — without one there is no ledger, and a loop has no spending bound at all.
    cost = settings.cost
    with new_request_context(
        request_id="quickstart",
        max_cost_usd=cost.max_cost_per_query_usd,
        max_calls=cost.max_llm_calls_per_query,
        max_tokens=cost.max_tokens_per_query,
    ) as (trace, ledger):
        query = Query(text=question, top_k=settings.retrieval.top_k)
        # fetch_k candidates, not top_k: recall is bought here and precision is the
        # reranker's job. A passage this stage misses can never be recovered.
        result = await retriever.retrieve_detailed(query)
        print(
            f"\nretrieved {result.total_candidates} candidates from {sorted(result.per_store)}"
            f" -> kept {len(result.chunks)}"
        )
        top = await reranker.rerank_chunks(query, result.chunks, top_k=settings.retrieval.top_k)
        answer = await generator.generate(query, RetrievalResult(chunks=top))

    report(question, answer, trace, ledger)


def report(question: str, answer: Any, trace: Any, ledger: Any) -> None:
    print(f"\nQ: {question}\n")
    print(answer.text)

    if answer.abstained:
        # Not a failure. A system that always answers cannot signal weak evidence, so
        # its worst outputs look exactly like its best.
        print(f"\nABSTAINED — {answer.abstain_reason}")
    print(
        f"\ngrounded={answer.grounded} ({answer.groundedness:.2f})  "
        f"confidence={answer.confidence:.2f}"
    )

    sources = {c.id: c.chunk.metadata.get("source") or c.chunk.document_id for c in answer.chunks}
    if answer.citations:
        print("\ncitations")
        for citation in answer.citations:
            label = sources.get(citation.chunk_id, "?")
            print(f"  {label:22} support={citation.support:.2f}  {citation.quote[:64]!r}")

    # Two views of the same spend. ``answer.usage`` is what this answer cost;
    # ``ledger.report()`` itemizes the whole request by stage, which is the view that
    # answers "where did the money go" when a pipeline makes twenty calls.
    print(
        f"\nanswer cost ${answer.usage.cost_usd:.6f} over {answer.usage.calls} call(s), "
        f"{answer.usage.total_tokens} tokens"
    )
    bill = ledger.report()
    print(
        f"request total ${bill['total_cost_usd']:.6f} over {bill['calls']} call(s), "
        f"{bill['total_tokens']} tokens, {bill['cached_calls']} served from cache"
    )
    for stage, spend in bill["by_stage"].items():
        print(f"  {stage:26} {spend['calls']:>2} call(s)  ${spend['cost_usd']:.6f}")
    print("\ntrace")
    for step in trace:
        print(f"  {step.name:26} {step.duration_ms:8.1f} ms")


async def main() -> None:
    settings = configure()
    configure_logging(settings.observability.log_level, json_logs=False)
    await preflight(settings)
    bundle = await build(settings)
    try:
        await index(bundle)
        await ask(bundle, QUESTION)
    finally:
        await bundle.aclose()


if __name__ == "__main__":
    asyncio.run(main())
