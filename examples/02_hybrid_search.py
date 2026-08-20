#!/usr/bin/env python
"""Hybrid search — four rankings of the same query, side by side.

What this demonstrates
    Why hybrid retrieval is not a tuning knob. The same three questions are answered
    four ways over one index — **dense only**, **sparse (BM25) only**, **hybrid** (both,
    fused server-side by Qdrant), and **hybrid + cross-encoder rerank** — and the
    rankings are printed next to each other so the disagreement is visible rather than
    asserted.

    The three questions are chosen because each one is won by a different mechanism:

    1. a question whose vocabulary matches the document — both legs find it, and the
       interesting column is what *reranking* does to the order;
    2. a paraphrase whose answering sentence shares almost no words with it (and starts
       with a pronoun) — dense finds it, BM25 cannot;
    3. a rare proper noun — BM25 finds it exactly, dense smears it across neighbours,
       because an embedding of a token the model never saw is an embedding of
       "something like that".

    No LLM calls are made. This example measures retrieval, and adding a synthesis call
    per configuration would only add cost and latency to a comparison that does not
    need one.

Services needed
    * Qdrant on ``localhost:6333`` / ``:6334``  —  ``docker compose up -d qdrant``

    No Postgres, no Neo4j, and **no API key** — everything here runs on local ONNX
    models. The first run downloads three of them (~120 MB).

Run
    python examples/02_hybrid_search.py
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ragorc.cache.tiered import build_cache
from ragorc.core.models import Query, ScoredChunk
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging
from ragorc.embed.cache import EmbeddingCache
from ragorc.embed.fastembed_provider import FastEmbedDense, FastEmbedSparse
from ragorc.index.loaders import load
from ragorc.index.split import build_splitter
from ragorc.retrieve.hybrid import HybridRetriever
from ragorc.retrieve.rerank import build_reranker
from ragorc.stores.qdrant.store import QdrantStore

CORPUS = Path(__file__).parent / "corpus"
COLLECTION = "ragorc_hybrid_demo"
SHOW = 5
"""Rows printed per column. Five is enough to see the disagreement without turning the
table into the whole candidate list."""

QUESTIONS: list[tuple[str, str]] = [
    (
        "vocabulary matches the document",
        "What is the SEV-1 response time on the Platinum support plan?",
    ),
    (
        "paraphrase; the answering sentence starts with a pronoun",
        "how much does the graph product cost for each named user",
    ),
    (
        "rare proper noun",
        "Voyager portal",
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
    # The port the *client* will use, which is the gRPC one by default. Checking 6333
    # while the client dials 6334 is a preflight that passes and then fails anyway.
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
    settings.retrieval.fetch_k = 20
    # The noise filter's relative cutoff keeps a different number of rows per
    # configuration, which is correct in production and ruins a side-by-side table: the
    # columns would have different lengths for reasons unrelated to ranking. Turned off
    # here so every column shows the same depth.
    settings.retrieval.relative_score_cutoff = None
    settings.retrieval.mmr_enabled = False
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
    print(f"indexed {len(documents)} documents -> {len(chunks)} chunks in '{COLLECTION}'\n")
    return store


# ---------------------------------------------------------------------------
# The four rankings
# ---------------------------------------------------------------------------
def label(scored: ScoredChunk) -> str:
    """A short, stable name for a chunk: ``document#index``.

    Stable across configurations, which is the whole point — the table is only
    readable if the same passage prints the same way in every column.
    """
    meta = scored.chunk.metadata
    source = meta.get("source") or meta.get("title") or scored.chunk.document_id
    return f"{Path(str(source)).stem}#{scored.chunk.index}"


async def rankings(
    retriever: HybridRetriever, reranker: Any, settings: Settings, question: str
) -> dict[str, list[ScoredChunk]]:
    """Run the four configurations against one question.

    Each leg is asked for a *single*-query search (``use_variants=False``): variant
    fan-out is a translation-layer feature and would change all four columns equally
    while making the comparison harder to read.

    The dense-only and sparse-only legs force the client-side path, because server-side
    fusion needs two branches to fuse — with one modality enabled there is nothing to
    fuse and the single-leg client path is equivalent.
    """
    common: dict[str, Any] = {"use_variants": False}
    query = Query(text=question, top_k=settings.retrieval.top_k)

    dense_only, sparse_only, hybrid = await asyncio.gather(
        retriever.retrieve_detailed(query, use_dense=True, use_sparse=False, **common),
        retriever.retrieve_detailed(query, use_dense=False, use_sparse=True, **common),
        retriever.retrieve_detailed(query, use_dense=True, use_sparse=True, **common),
    )
    # Reranking is a *reordering* of the hybrid candidates, not a fifth retrieval: that
    # is the division of labour the whole package is built on — this stage buys recall,
    # the cross-encoder buys precision, and it can only reorder what it was handed.
    reranked = await reranker.rerank_chunks(query, list(hybrid.chunks), top_k=SHOW)

    return {
        "dense": dense_only.chunks,
        "sparse (bm25)": sparse_only.chunks,
        f"hybrid ({settings.retrieval.fusion.value})": hybrid.chunks,
        "hybrid + rerank": reranked,
    }


def print_table(question: str, why: str, columns: dict[str, list[ScoredChunk]]) -> None:
    names = list(columns)
    width = 34

    print(f"Q: {question}")
    print(f"   ({why})\n")
    print("  # " + "".join(f"{name:<{width}}" for name in names))
    print("  --" + "-" * (width * len(names)))
    for rank in range(SHOW):
        cells: list[str] = []
        for name in names:
            hits = columns[name]
            if rank < len(hits):
                cells.append(f"{label(hits[rank])[:24]:<24} {hits[rank].score:8.3f}")
            else:
                cells.append("")
        print(f"  {rank + 1} " + "".join(f"{cell:<{width}}" for cell in cells))

    # The most informative line in the whole example: what one leg found and the other
    # never saw. A passage missing from a column cannot be recovered downstream, which
    # is why running one leg alone sets a ceiling nothing later can lift.
    seen = {name: {label(c) for c in hits[:SHOW]} for name, hits in columns.items()}
    dense_only = seen["dense"] - seen["sparse (bm25)"]
    sparse_only = seen["sparse (bm25)"] - seen["dense"]
    print(f"\n  only dense found : {', '.join(sorted(dense_only)) or '-'}")
    print(f"  only sparse found: {', '.join(sorted(sparse_only)) or '-'}")
    fused = next(name for name in names if name.startswith("hybrid ("))
    recovered = (dense_only | sparse_only) & seen[fused]
    print(f"  fusion kept both : {', '.join(sorted(recovered)) or '-'}")
    print()


async def main() -> None:
    settings = configure()
    configure_logging(settings.observability.log_level, json_logs=False)
    await preflight(settings)

    store = await index(settings)
    retriever = HybridRetriever(store, settings=settings)
    # Local ONNX cross-encoder, so this whole example still needs no API key.
    reranker = build_reranker("cross_encoder", settings=settings)
    try:
        for why, question in QUESTIONS:
            print_table(question, why, await rankings(retriever, reranker, settings, question))
    finally:
        await store.close()

    print(
        "Read the last three lines of each block first. Wherever 'only dense' or 'only\n"
        "sparse' is non-empty, running that leg alone would have set a recall ceiling\n"
        "no reranker could lift — and the hybrid column is where both survive."
    )


if __name__ == "__main__":
    asyncio.run(main())
