#!/usr/bin/env python
"""GraphRAG — build the graph, then search it three ways.

What this demonstrates
    The two halves of GraphRAG and why they are separate packages.

    **Construction** (:mod:`ragorc.index.graph`): extract entities and typed relations
    per chunk, *resolve* the mentions of one thing into one node, detect communities over
    the resolved graph, and write a summary per community. The resolution step is the one
    that decides whether any of it works — without it "Contoso", "Contoso Ltd" and
    "CONTOSO LTD" are three unconnected nodes and traversal stops finding anything, which
    the report below makes visible as ``merged_entities``.

    **Search** (:mod:`ragorc.retrieve.graph`): the same graph answers three shapes of
    question through three different entry points, and this example runs all three on
    the same questions so the difference is visible rather than described.

    ===========  =========================  ===============================================
    mode         entry point                the question it is for
    ===========  =========================  ===============================================
    ``local``    entities named in the      specific, entity-anchored: "who owns X?"
                 question
    ``global``   community summaries        corpus-wide and aggregate: "what kinds of ...?"
    ``drift``    vector hits, then graph    descriptive questions that name nothing
                 expansion around them
    ===========  =========================  ===============================================

    Global search is map-reduce, and only the **map** half is a retriever: N cheap
    parallel calls asking what each community contributes. The reduce is *generation* —
    one call that synthesizes the partials — so it belongs to the generator, which
    already owns citations, groundedness and abstention. That is why the global mode
    below reads its prompt from ``graphrag.GLOBAL_REDUCE_PROMPT``.

Cost
    Graph construction is **one LLM call per chunk** plus one per community report, so it
    is the most expensive thing in this directory: expect roughly $0.10-$0.40 on the
    default models for this ten-document corpus, and re-runs cost the same because the
    graph is rebuilt. ``graph.max_gleanings`` is set to 0 here to halve the extraction
    calls; production leaves it at 1. The actual spend is printed.

Services needed
    * Qdrant **and** Neo4j        —  ``docker compose up -d qdrant neo4j``
    * ``RAGORC_LLM__API_KEY``     —  https://openrouter.ai/keys

    No Postgres. Community detection uses Leiden via ``ragorc[graphrag]``; without that
    extra it degrades to a NetworkX partitioner and says so in the log.

Run
    python examples/04_graphrag.py
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ragorc.cache.tiered import build_cache
from ragorc.core.models import Chunk, Query, RetrievalResult
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging, new_request_context
from ragorc.embed.cache import EmbeddingCache
from ragorc.embed.fastembed_provider import FastEmbedDense, FastEmbedSparse
from ragorc.generate.answer import AnswerGenerator
from ragorc.index.graph.build import GraphBuilder
from ragorc.index.loaders import load
from ragorc.index.split import build_splitter
from ragorc.llm.openrouter import OpenRouterLLM
from ragorc.pipeline.graphs import graphrag
from ragorc.retrieve.graph import GraphDriftRetriever, GraphGlobalRetriever, GraphLocalRetriever
from ragorc.stores.neo4j.store import Neo4jStore
from ragorc.stores.qdrant.store import QdrantStore

CORPUS = Path(__file__).parent / "corpus"
COLLECTION = "ragorc_graphrag_demo"

QUESTIONS: list[tuple[str, str]] = [
    (
        "entity-anchored — names a thing the graph has a node for",
        "Who is the on-call primary for the Graph Service, and what escalates past them?",
    ),
    (
        "corpus-wide — the answer is a property of the whole set",
        "What does Halcyon Data sell, and who owns each product?",
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
    qdrant_host, rest_port = endpoint(settings.qdrant.url, 6333)
    checks = [
        (
            "Qdrant",
            qdrant_host,
            settings.qdrant.grpc_port if settings.qdrant.prefer_grpc else rest_port,
            "docker compose up -d qdrant",
        ),
        ("Neo4j", *endpoint(settings.neo4j.uri, 7687), "docker compose up -d neo4j"),
    ]
    results = await asyncio.gather(*(reachable(host, port) for _, host, port, _ in checks))
    for (name, host, port, hint), ok in zip(checks, results, strict=True):
        if not ok:
            problems.append(f"{name} is not reachable at {host}:{port}.\n      {hint}")
    if problems:
        die("Cannot run this example:", *(f"\n  - {problem}" for problem in problems))


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def configure() -> Settings:
    settings = get_settings().model_copy(deep=True)
    settings.qdrant.collection = COLLECTION
    settings.generation.prompt_name = "answer_default"
    # Shorter answers so three modes fit on one screen; this is presentation, not policy.
    settings.generation.max_answer_tokens = 400

    graph = settings.graph
    graph.enabled = True
    graph.resolve_entities = True
    graph.detect_communities = True
    graph.summarize_communities = True
    # Each gleaning pass is another call per chunk and finds progressively fewer
    # entities. Production keeps 1; a demo does not need to pay for it twice.
    graph.max_gleanings = 0
    graph.min_community_size = 2
    graph.max_community_levels = 2
    return settings


async def index(settings: Settings) -> tuple[QdrantStore, Any, list[Chunk]]:
    """Index the corpus into Qdrant and return the chunks the graph is built from.

    The graph and the vector index are built from the *same* chunk objects on purpose:
    entity ``source_chunk_ids`` then point at ids that exist in Qdrant, which is what
    lets local search resolve a traversal back into citable prose. Two separate splits
    would produce two id spaces and a graph that can name entities but never quote them.
    """
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


async def build_graph(settings: Settings, llm: Any, dense: Any, chunks: list[Chunk]) -> Neo4jStore:
    graph = Neo4jStore(settings=settings)
    applied = await graph.ensure_schema()
    print(f"neo4j schema ready ({len(applied)} constraint(s)/index(es))")

    print(
        f"\nextracting a graph from {len(chunks)} chunks — one model call each, "
        f"plus one per community report"
    )
    report = await GraphBuilder(llm, graph, embedder=dense, settings=settings).build(chunks)
    summary = report.summary()
    print(
        f"  entities {summary['entities']} (merged {summary['merged_entities']}), "
        f"relations {summary['relations']}, communities {summary['communities']} "
        f"across {summary['community_levels']} level(s)"
    )
    print(
        f"  wrote {summary['entities_written']} entities, "
        f"{summary['relations_written']} relations, "
        f"{summary['chunk_links_written']} chunk links, "
        f"{summary['communities_written']} community summaries"
    )
    print(
        f"  {summary['llm_calls']} call(s), {summary['tokens']} tokens, "
        f"${summary['cost_usd']:.4f}, {summary['total_ms']:.0f} ms"
    )
    if summary["dangling_relations"]:
        # An edge whose endpoint did not survive resolution. Reported rather than
        # silently dropped: a high count means extraction is naming things
        # inconsistently, which is a prompt problem, not a graph problem.
        print(f"  dropped {summary['dangling_relations']} relation(s) with a missing endpoint")
    return graph


# ---------------------------------------------------------------------------
# The three search modes
# ---------------------------------------------------------------------------
async def compare(
    settings: Settings,
    generator: AnswerGenerator,
    modes: dict[str, Any],
    why: str,
    question: str,
) -> None:
    print("\n" + "=" * 78)
    print(f"Q: {question}\n   ({why})")
    print("=" * 78)

    for mode, retriever in modes.items():
        cost = settings.cost
        with new_request_context(
            request_id=f"graphrag-{mode}",
            max_cost_usd=cost.max_cost_per_query_usd,
            max_calls=cost.max_llm_calls_per_query,
            max_tokens=cost.max_tokens_per_query,
        ) as (_trace, ledger):
            query = Query(text=question, top_k=settings.retrieval.top_k)
            chunks = await retriever.retrieve(query)
            # Global search's evidence is community *summaries*, so the reduce step
            # needs the wording written for partial answers rather than for passages.
            # graphrag registered that prompt with the generator's slot names at import.
            prompt = graphrag.GLOBAL_REDUCE_PROMPT if mode == "global" else None
            answer = await generator.generate(
                query, RetrievalResult(chunks=chunks), prompt_name=prompt
            )

        print(f"\n--- {mode} ({len(chunks)} evidence chunk(s))")
        if not chunks:
            # An expected outcome for local search on a question naming nothing the
            # graph knows, which is exactly the gap DRIFT exists to cover.
            print("    no evidence: this mode found no entry point into the graph")
        for scored in chunks[:3]:
            print(f"    [{scored.source.value:>12}] {scored.score:.3f} {scored.content[:88]!r}")
        print(f"\n    {answer.text.strip()}")
        if answer.abstained:
            print(f"    ABSTAINED — {answer.abstain_reason}")
        bill = ledger.report()
        print(
            f"\n    grounded={answer.grounded} ({answer.groundedness:.2f})  "
            f"cost ${bill['total_cost_usd']:.5f} over {bill['calls']} call(s)"
        )


async def main() -> None:
    settings = configure()
    configure_logging(settings.observability.log_level, json_logs=False)
    await preflight(settings)

    vector, dense, chunks = await index(settings)
    llm = OpenRouterLLM(settings.llm)
    graph: Neo4jStore | None = None
    try:
        graph = await build_graph(settings, llm, dense, chunks)

        # All three modes share one Neo4j store and one Qdrant store: local search needs
        # the chunk bodies to quote, and DRIFT needs the vectors for its seed search.
        local = GraphLocalRetriever(graph, vector, settings=settings)
        modes: dict[str, Any] = {
            "local": local,
            "global": GraphGlobalRetriever(llm, graph, settings=settings),
            "drift": GraphDriftRetriever(vector, graph, local=local, settings=settings),
        }
        generator = AnswerGenerator(llm, settings)
        for why, question in QUESTIONS:
            await compare(settings, generator, modes, why, question)
    finally:
        await llm.aclose()
        await vector.close()
        if graph is not None:
            await graph.close()

    print(
        "\nRead the evidence lines, not just the answers: local cites the chunks an\n"
        "entity traversal reached, global cites community summaries (so it can answer a\n"
        "question about the whole corpus that no single passage states), and DRIFT is\n"
        "the one that still works when the question names nothing at all."
    )


if __name__ == "__main__":
    asyncio.run(main())
