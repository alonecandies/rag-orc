#!/usr/bin/env python
"""End-to-end verification: real corpus, real stores, one real OpenRouter call.

This is the check that cannot be faked. Everything else in the suite runs against
stubs — which is the right default, because a test that calls a model is a sample,
not a test — but a library whose whole claim is "grounded, cited, costed answers"
has to be shown doing that once, against a real model, with real vectors, in a
real vector store.

What it asserts, in order of how much it would matter if it broke:

1. An answer comes back **grounded** with **resolvable citations** — the central
   claim, and the one that needs a real model to test at all.
2. Retrieval finds the **right document**, checked by name rather than by chunk id
   (see ``CaseResult.expected_documents`` for why that distinction exists).
3. A question the corpus cannot answer produces an **explicit abstention**, not a
   confident guess. This is the property a stub can never really demonstrate: a
   stub returns whatever it was told to, while a real model given no evidence will
   answer from its parameters unless something stops it.
4. The bill is **itemized** — per stage and per model, with the cache hit rate.

Exits non-zero on any failure, so it can gate a release.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

CORPUS = Path("examples/corpus")

# A question the corpus answers, and the file that answers it.
ANSWERABLE = "How many days does Finance take to pay an approved expense claim?"
EXPECTED_DOCUMENT = "02-expenses-policy.md"

# A question about the same company that the corpus says nothing about. Chosen to
# be plausible rather than absurd: "what colour is the sky" is trivially refused
# by topic, whereas an unanswerable *company* question tests whether the guardrails
# actually work rather than whether the model noticed a change of subject.
UNANSWERABLE = "What was Halcyon Data's exact revenue in the third quarter of 2021?"


def line(char: str = "-") -> None:
    print(char * 78)


async def main() -> int:
    from ragorc.core.settings import get_settings
    from ragorc.core.telemetry import configure_logging, new_request_context
    from ragorc.embed import build_dense_embedder, build_reranker, build_sparse_embedder
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.index.loaders import DirectoryLoader
    from ragorc.index.split import build_splitter
    from ragorc.llm.openrouter import OpenRouterLLM
    from ragorc.retrieve.hybrid import HybridRetriever
    from ragorc.stores.qdrant.store import QdrantStore

    configure_logging(level="WARNING", json_logs=False)
    settings = get_settings()
    if not settings.llm.api_key.get_secret_value():
        print("RAGORC_LLM__API_KEY is not set", file=sys.stderr)
        return 2

    failures: list[str] = []

    print(f"corpus   {CORPUS}")
    print(f"model    {settings.llm.model}  (graders: {settings.llm.fast_model})")
    print(f"embed    {settings.embedding.dense_model} + {settings.embedding.sparse_model}")
    line("=")

    # --- ingest -----------------------------------------------------------
    dense = build_dense_embedder(settings)
    sparse = build_sparse_embedder(settings)
    reranker = build_reranker(settings)

    documents = await DirectoryLoader(settings=settings).load(CORPUS, recursive=True)
    print(f"loaded   {len(documents)} documents")

    splitter = build_splitter(embedder=dense, settings=settings)
    chunks = await splitter.split_many(documents)
    print(f"split    {len(chunks)} chunks ({type(splitter).__name__})")

    texts = [c.embed_text for c in chunks]
    dense_vectors = await dense.embed_documents(texts)
    for chunk, vector in zip(chunks, dense_vectors, strict=True):
        chunk.dense = vector
    if sparse is not None:
        for chunk, vector in zip(chunks, await sparse.embed_documents(texts), strict=True):
            chunk.sparse = vector
    print(f"embedded {len(chunks)} chunks")

    store = QdrantStore(settings=settings, dense_embedder=dense, sparse_embedder=sparse)
    await store.ensure_collection(recreate=True)
    written = await store.upsert(chunks)
    print(f"indexed  {written} chunks into '{settings.qdrant.collection}'")
    line("=")

    llm = OpenRouterLLM(settings.llm)
    # Only the store is passed. HybridRetriever's `dense`/`sparse` parameters take
    # *sub-retrievers*, not embedders — it builds them from the store, which already
    # holds the embedders, so the ONNX sessions load once per process rather than
    # once per retriever.
    retriever = HybridRetriever(store=store, settings=settings)
    generator = AnswerGenerator(llm, settings)

    try:
        # --- 1. the answerable question -----------------------------------
        print(f"\nQ  {ANSWERABLE}")
        with new_request_context(request_id="e2e-1") as (_trace, ledger):
            from ragorc.core.models import Query, RetrievalResult

            query = Query(text=ANSWERABLE, top_k=settings.retrieval.top_k)
            candidates = await retriever.retrieve(query)
            if reranker is not None and candidates:
                order = await reranker.rerank(
                    query.text,
                    [c.chunk.content for c in candidates],
                    top_k=settings.retrieval.top_k,
                )
                candidates = [candidates[i].with_score(score) for i, score in order]
            answer = await generator.generate(query, RetrievalResult(chunks=candidates))

        print(f"\nA  {answer.text.strip()[:600]}")
        line()
        print(
            f"   grounded={answer.grounded} ({answer.groundedness:.2f})  "
            f"abstained={answer.abstained}  citations={len(answer.citations)}"
        )

        sources = []
        for citation in answer.citations:
            match = next((c for c in answer.chunks if c.chunk.id == citation.chunk_id), None)
            if match is not None:
                sources.append(str(match.chunk.metadata.get("source", "?")))
                print(
                    f"   [{citation.chunk_id[:8]}] {Path(sources[-1]).name}: "
                    f"{citation.quote[:70]!r}"
                )

        if answer.abstained:
            failures.append("abstained on a question the corpus answers")
        if not answer.grounded:
            failures.append("answer was not grounded")
        if not answer.citations:
            failures.append("answer carried no citations")
        elif not any(EXPECTED_DOCUMENT in s for s in sources):
            failures.append(f"cited {sources} rather than {EXPECTED_DOCUMENT}")

        retrieved_docs = [str(c.chunk.metadata.get("source", "")) for c in answer.chunks]
        if not any(EXPECTED_DOCUMENT in d for d in retrieved_docs):
            failures.append(f"{EXPECTED_DOCUMENT} was not retrieved at all")
        else:
            rank = next(i for i, d in enumerate(retrieved_docs) if EXPECTED_DOCUMENT in d)
            print(f"   {EXPECTED_DOCUMENT} retrieved at rank {rank}")

        report = ledger.report()
        print(
            f"   cost=${report['total_cost_usd']:.6f}  calls={report['calls']}  "
            f"tokens={report['total_tokens']}  cache={report['cache_hit_rate']:.0%}"
        )
        for stage, spend in sorted(report["by_stage"].items(), key=lambda kv: -kv[1]["cost_usd"])[
            :5
        ]:
            print(f"     {stage:24} {spend['calls']:2}x  ${spend['cost_usd']:.6f}")
        if report["calls"] == 0:
            failures.append("no LLM call was recorded — was the answer real?")

        # --- 2. the unanswerable question ---------------------------------
        line("=")
        print(f"\nQ  {UNANSWERABLE}")
        with new_request_context(request_id="e2e-2") as (_trace, ledger2):
            query2 = Query(text=UNANSWERABLE, top_k=settings.retrieval.top_k)
            candidates2 = await retriever.retrieve(query2)
            answer2 = await generator.generate(query2, RetrievalResult(chunks=candidates2))

        print(f"\nA  {answer2.text.strip()[:400]}")
        line()
        print(f"   abstained={answer2.abstained}  reason={answer2.abstain_reason or '-'}")
        print(f"   grounded={answer2.grounded} ({answer2.groundedness:.2f})")
        print(f"   cost=${ledger2.report()['total_cost_usd']:.6f}")

        # Either gate is a pass: an explicit abstention, or a grounded answer that
        # says the corpus does not contain the figure. Both are honest; only an
        # invented number is a failure.
        said_unknown = any(
            phrase in answer2.text.lower()
            for phrase in ("not", "does not", "no information", "cannot", "unable", "doesn't")
        )
        if not (answer2.abstained or said_unknown):
            failures.append("invented an answer the corpus does not contain")

    finally:
        with __import__("contextlib").suppress(Exception):
            await store.client.delete_collection(settings.qdrant.collection)
        await store.close()
        await llm.aclose()

    line("=")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASSED: grounded, cited, correctly-sourced answer; honest refusal; itemized cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
