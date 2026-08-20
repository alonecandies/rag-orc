#!/usr/bin/env python
"""Multi-store routing — one question, three databases, one ranked list.

What this demonstrates
    The architecture the whole library is shaped around. Vector, relational and graph
    are not three interchangeable indexes; they answer three different *kinds* of
    question, and something has to decide which of them a question needs. This example
    shows that decision and its consequences:

    * the :class:`~ragorc.core.models.RouteDecision` — which stores, with what
      confidence, and the router's stated reasoning;
    * the **concurrent** fan-out over the routed stores, with per-store result counts
      and per-store latency, because serial fan-out makes a query's latency the *sum*
      of its backends instead of the maximum;
    * every degradation, by name, in ``RetrievalResult.errors`` — a store that failed, a
      store nobody configured, a store whose circuit breaker is open. A dead backend
      costs recall, never the query.

    Three questions are asked, one per store shape: an aggregate over business data
    (Postgres, via guarded Text-to-SQL), a relationship between two named things
    (Neo4j), and an explanatory question about the documents (Qdrant). Then one question
    goes through the full pipeline so the answer, the trace and the bill are visible.

    An empty Neo4j is the *expected* state here — nothing has extracted a graph yet — and
    the point of printing the diagnostics is that you can see the leg was consulted and
    returned nothing, rather than wondering why the graph never contributed. Run
    ``examples/04_graphrag.py`` first if you want that leg to have something to find.

Services needed
    * Qdrant, Postgres **and** Neo4j  —  ``docker compose up -d`` (all three)
    * ``RAGORC_LLM__API_KEY``         —  https://openrouter.ai/keys

    Postgres also needs the demo business schema, which ``docker compose`` loads from
    ``scripts/init/postgres`` on first start — that is what Text-to-SQL queries.

Run
    python examples/03_multi_store.py
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ragorc import build_pipeline
from ragorc.core.models import Query, RouteDecision
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging

CORPUS = Path(__file__).parent / "corpus"
COLLECTION = "ragorc_multistore_demo"

QUESTIONS: list[tuple[str, str]] = [
    (
        "business data — an aggregate over rows, not passages",
        "Which three customers have the highest ARR, and what segment is each one in?",
    ),
    (
        "a relationship between two named things",
        "How is Contoso Ltd connected to the Graph Service?",
    ),
    (
        "an explanation that lives in the documents",
        "Why does the Graph Add-on exist, and what two question shapes does it cover?",
    ),
]

FULL_PIPELINE_QUESTION = "What SEV-1 response time is Fabrikam GmbH entitled to, and why?"


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
        ),
        ("Postgres", *endpoint(settings.postgres.dsn, 5432)),
        ("Neo4j", *endpoint(settings.neo4j.uri, 7687)),
    ]
    # Checked concurrently: three sequential two-second timeouts is six seconds of
    # nothing before the first useful line of output.
    results = await asyncio.gather(*(reachable(host, port) for _, host, port in checks))
    for (name, host, port), ok in zip(checks, results, strict=True):
        if not ok:
            problems.append(
                f"{name} is not reachable at {host}:{port}.\n      docker compose up -d"
            )
    if problems:
        die("Cannot run this example:", *(f"\n  - {problem}" for problem in problems))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_route(decision: RouteDecision) -> None:
    stores = ", ".join(store.value for store in decision.stores) or "(none)"
    print(f"  route    : {stores}   [{decision.method}, confidence {decision.confidence:.2f}]")
    if decision.prompt_name:
        print(f"  prompt   : {decision.prompt_name}")
    if decision.reasoning:
        print(f"  reasoning: {decision.reasoning.strip()[:160]}")


def print_diagnostics(result: Any) -> None:
    """The per-store table is the reason ``retrieve_detailed`` exists.

    Once fusion has flattened several score scales into one number, "which store
    contributed this, how long did it take, and what failed" is unrecoverable — so the
    fan-out records it as it goes rather than reconstructing it afterwards.
    """
    print(f"  candidates: {result.total_candidates} -> returned {len(result.chunks)}")
    for store, chunks in sorted(result.per_store.items()):
        took = result.timings_ms.get(store, 0.0)
        error = result.errors.get(store)
        state = f"FAILED: {error}" if error else f"{len(chunks)} result(s)"
        print(f"    {store:12} {took:8.1f} ms  {state}")
    total = result.timings_ms.get("total")
    if total:
        per_store = [v for k, v in result.timings_ms.items() if k not in ("total", "fuse")]
        # The headline number of the whole example. Wall clock tracks the *slowest*
        # store, not the sum of them: serial fan-out would have cost the sum, and that
        # difference is what makes a three-store architecture a feature and not a tax.
        print(
            f"    {'wall clock':12} {total:8.1f} ms  "
            f"(slowest leg {max(per_store, default=0.0):.1f} ms, "
            f"sum of legs {sum(per_store):.1f} ms)"
        )
    if result.errors:
        print("    degraded, not failed: the query continued with the stores that answered")
    for scored in result.chunks[:3]:
        source = scored.chunk.metadata.get("source") or scored.chunk.document_id
        print(
            f"      [{scored.source.value:>12}] {scored.score:.3f} "
            f"{Path(str(source)).name}: {scored.content.strip()[:70]!r}"
        )


# ---------------------------------------------------------------------------
# The example
# ---------------------------------------------------------------------------
async def route_and_fan_out(rag: Any, why: str, question: str) -> None:
    print(f"\nQ: {question}\n   ({why})")
    query = Query(text=question, top_k=rag.settings.retrieval.top_k)

    decision, usage = await rag.router.route(query)
    print_route(decision)
    if usage.calls:
        print(f"  routing cost: ${usage.cost_usd:.6f} over {usage.calls} call(s)")

    result = await rag.retriever.retrieve_detailed(query, route=decision)
    print_diagnostics(result)


async def full_pipeline(rag: Any, question: str) -> None:
    print("\n" + "=" * 78)
    print(f"the whole pipeline, one call: rag.query({question!r})")
    print("=" * 78)

    answer = await rag.query(question)
    print(f"\n{answer.text}\n")
    if answer.abstained:
        print(f"ABSTAINED — {answer.abstain_reason}\n")
    if answer.route is not None:
        print_route(answer.route)
    print(
        f"  grounded={answer.grounded} ({answer.groundedness:.2f})  "
        f"confidence={answer.confidence:.2f}  citations={len(answer.citations)}"
    )
    print(
        f"  cost ${answer.usage.cost_usd:.6f} over {answer.usage.calls} call(s), "
        f"{answer.usage.total_tokens} tokens"
    )
    print("  trace")
    for step in answer.trace:
        print(f"    {step.name:28} {step.duration_ms:8.1f} ms")


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.observability.log_level, json_logs=False)
    await preflight(settings)

    # Overrides use the same ``__`` nesting as the environment, and apply to this
    # process only. ``use_fulltext`` puts Postgres in *two* roles at once — a full-text
    # leg inside hybrid search and the Text-to-SQL execution target — which is worth
    # seeing, because they are different capabilities of one connection.
    rag = await build_pipeline(
        qdrant__collection=COLLECTION,
        retrieval__use_fulltext=True,
    )
    try:
        print(f"pipeline '{rag.select_graph('auto')}' ready; ingesting {CORPUS}")
        report = await rag.ingest(CORPUS)
        print(f"  {report.summary()}\n")

        for why, question in QUESTIONS:
            await route_and_fan_out(rag, why, question)

        await full_pipeline(rag, FULL_PIPELINE_QUESTION)
    finally:
        await rag.aclose()


if __name__ == "__main__":
    asyncio.run(main())
