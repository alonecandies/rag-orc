"""The command line.

Four properties shape every command here, and they are the reason this file is
not a thin shell around the service.

**Configuration flows through the environment, not through a parallel channel.**
:class:`~ragorc.core.settings.Settings` is populated from ``RAGORC_*`` variables
with ``__`` for nesting, and it is a process-wide ``lru_cache`` singleton that a
dozen components reach for when nobody handed them one. So a flag like
``--collection`` sets ``RAGORC_QDRANT__COLLECTION`` and then clears the cache,
rather than constructing a modified ``Settings`` and hoping every collaborator
was passed it explicitly. One mechanism, and it reaches the code that never
received an argument — which is precisely the code a mutated copy would miss.

**A dead store is a message, not a traceback.** Every command funnels through
:func:`_run`, which turns each error class into the sentence an operator can act
on: which store, which variable to set, which container to start. A stack trace
tells someone running ``ragorc ingest ./docs`` nothing they can use, and it buries
the one line that would have.

**Exit codes distinguish "wrong" from "broken".** ``0`` succeeded, ``1`` the
operation failed, ``2`` the configuration is wrong. That split is what lets a
Makefile or a CI job react: a ``2`` will never succeed on retry, a ``1`` might.

**Each command owns one event loop.** ``asyncio.run`` per command, with
:func:`~ragorc.core.concurrency.install_uvloop` called before it — a CLI process
does one job and exits, so there is nothing to gain from a shared loop and a real
cost to a loop that outlives the work in it. ``serve`` is the single exception,
and it is documented where it happens: uvicorn owns the loop it runs the app in.

The components themselves come from :class:`~ragorc.server.app.RagService`, the
same object the HTTP service builds. Not for convenience — so that ``ragorc
query`` and ``POST /query`` cannot answer differently. A CLI with its own
composition is a second implementation whose divergence is discovered in
production.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import math
import os
import pathlib
import statistics
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from pathlib import Path
from typing import Any, NoReturn, TypeVar

import orjson
import structlog
import typer
from rich.console import Console
from rich.markup import escape as _escape_markup
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ragorc.core.concurrency import install_uvloop
from ragorc.core.errors import (
    BudgetExceeded,
    ConfigError,
    GuardrailViolation,
    RagOrcError,
    RateLimited,
    StoreUnavailable,
    ValidationFailed,
)
from ragorc.core.models import Query
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import configure_logging
from ragorc.security.tenancy import scope_filter
from ragorc.server.app import RagService, load_eval_items
from ragorc.server.schemas import (
    EvalRequest,
    PipelineName,
    QueryRequest,
)

log = structlog.get_logger(__name__)

__all__ = ["app"]

T = TypeVar("T")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIG = 2
"""Three outcomes, because two of them need different reactions. A ``1`` is worth
retrying — a store was down, a provider timed out. A ``2`` never is: the settings
are wrong and the next run will fail identically. A caller that cannot tell them
apart either retries forever or gives up too early."""

console = Console()
err = Console(stderr=True)
"""One rule, applied without exception: **results on stdout, everything else on
stderr**.

Progress bars, spinners, discovery counts and warnings all go to ``err``. Not
tidiness — ``--json`` output is meant to be piped, and a progress bar written to
stdout corrupts the document the next process in the pipe is parsing. Keeping the
split unconditional means ``--json`` needs no special case, which is what stops
the next command added here from getting it wrong."""

_STORE_HINTS = {
    "qdrant": "start Qdrant (docker compose up -d qdrant) or set RAGORC_QDRANT__URL",
    "postgres": "start Postgres (docker compose up -d postgres) or set RAGORC_POSTGRES__DSN",
    "neo4j": "start Neo4j (docker compose up -d neo4j) or set RAGORC_NEO4J__URI",
}
"""What to actually do, per store. The exception already says *what* failed; the
only thing missing from a useful error message is *what next*."""

_INGEST_BATCH_DOCUMENTS = 64
"""Documents handed to the pipeline per batch.

The pipeline streams internally and never materializes its chunk list, so this is
not about its memory — it is the resolution of the progress bar. Too large and the
bar jumps in visible steps; too small and the checksum-skip probe degenerates
towards one round trip per document, which is the thing that probe exists to
avoid."""


# ---------------------------------------------------------------------------
# Settings, from flags, through the environment
# ---------------------------------------------------------------------------
def _env_value(value: Any) -> str:
    """Render a flag value the way pydantic-settings will read it back."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return orjson.dumps(value).decode()
    return str(value)


_ENV_PREFIX = str(Settings.model_config.get("env_prefix") or "RAGORC_")
"""Read from the settings model rather than retyped, so a rename cannot leave
this guard checking for a prefix nothing uses."""


def _apply(overrides: dict[str, Any]) -> None:
    """Publish flag values as ``RAGORC_*`` environment variables.

    ``None`` means "not supplied" and is skipped, so a flag left off the command
    line leaves whatever the environment or ``.env`` already said. That is the
    behaviour anyone expects and the reason every override option here defaults
    to ``None`` rather than to the setting's own default: a default copied into a
    flag silently overrides the environment every time the command runs.
    """
    for key, value in overrides.items():
        if not key.startswith(_ENV_PREFIX):
            # A key that is not an environment variable name is silently ignored
            # by pydantic-settings, so the flag it came from would reach nothing
            # and the command would run against whatever the environment said.
            #
            # Two commands spelled these as their *typer parameter* names —
            # `_settings(collection=collection, tenant=tenant)` — while the other
            # six spelled them as the variable. `delete --collection scratch`
            # therefore deleted from the default collection and printed
            # `documents found 1 / vectors removed 2`: a report identical to a
            # correct delete, from a command whose own docstring says "the blast
            # radius is kept equal to what was typed".
            #
            # Raising rather than warning, because this is a programming error in
            # a call site inside this module and every one of them is reachable
            # from a test.
            raise ValueError(
                f"override key {key!r} is not a settings variable; "
                f"use the {_ENV_PREFIX}* name pydantic-settings reads "
                f"(e.g. {_ENV_PREFIX}QDRANT__COLLECTION)"
            )
        if value is not None:
            os.environ[key] = _env_value(value)


def _settings(**overrides: Any) -> Settings:
    """Resolve settings after applying overrides, or exit 2 explaining why."""
    _apply(overrides)
    # The singleton may already have been built — by an earlier command in the
    # same process, or by importing a module that resolved its own defaults — and
    # it would not see anything set above.
    get_settings.cache_clear()
    try:
        return get_settings()
    except Exception as exc:  # noqa: BLE001 - pydantic's error, rendered for a human
        err.print(
            Panel(
                f"[red]{type(exc).__name__}[/red]\n\n{exc}",
                title="configuration is invalid",
                subtitle="check your RAGORC_* variables and .env",
                border_style="red",
            )
        )
        raise typer.Exit(EXIT_CONFIG) from exc


@contextlib.asynccontextmanager
async def _service(settings: Settings) -> AsyncIterator[RagService]:
    """Build the service, hand it over, and close it however the body ends.

    The same object the HTTP service builds, for the same reason it builds it once:
    the ONNX model load and the connection pools are the expensive part, and a CLI
    command that leaked them would hold a Postgres connection until the process
    happened to exit.
    """
    service = RagService(settings)
    try:
        await service.build()
    except BaseException:
        # A partially built service still owns whatever it opened before failing.
        await service.aclose()
        raise
    try:
        yield service
    finally:
        await service.aclose()


# ---------------------------------------------------------------------------
# Error presentation
# ---------------------------------------------------------------------------
def _plain(text: object) -> str:
    """Escape text that is *data*, before it reaches Rich's markup parser.

    Rich reads ``[...]`` as a style tag, so an unescaped
    ``pip install 'ragorc[server]'`` renders as ``pip install 'ragorc'`` — advice
    to install the package the operator already has, delivered at the moment they
    are stuck. Every optional import in this library raises with its own pip hint,
    and those messages are printed straight into a panel, so the extra was being
    eaten from the one line that would have fixed the problem.

    The same hazard applies to any interpolated filename, error or model output:
    ``notes[2024].md`` loses its year. Style tags this module writes itself stay
    outside the escaped part.
    """
    return _escape_markup(str(text))


def _fail(title: str, message: str, hint: str, code: int) -> NoReturn:
    """Report and exit. Declared ``NoReturn`` so callers need no unreachable branch."""
    err.print(
        Panel(
            _plain(message),
            title=f"[red]{_plain(title)}[/red]",
            subtitle=_plain(hint),
            border_style="red",
        )
    )
    raise typer.Exit(code)


def _run(factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run one command's coroutine, translating every failure into a message.

    The translation table is the point of this function. Each branch answers the
    question the operator is actually asking — "what do I do now?" — which the
    exception knows and a traceback obscures. Anything unrecognized is reported
    with its type and message only, plus the flag that produces the traceback for
    someone who wants it; printing one by default trains people to ignore the
    line above it, which is the line that mattered.
    """
    install_uvloop()
    try:
        return asyncio.run(factory())
    except typer.Exit:
        # The command already reported and chose its code. This has to come first:
        # click's Exit derives from RuntimeError, so a later `except Exception`
        # would swallow a deliberate exit and relabel it as an unexpected failure.
        raise
    except KeyboardInterrupt:
        err.print("[yellow]interrupted[/yellow]")
        raise typer.Exit(EXIT_ERROR) from None
    except StoreUnavailable as exc:
        _fail(
            f"{exc.store} is unreachable",
            str(exc.message),
            _STORE_HINTS.get(exc.store, "check the store's configuration and that it is running"),
            EXIT_ERROR,
        )
    except GuardrailViolation as exc:
        _fail(
            "blocked by a guardrail",
            f"{exc.message}\n\nrule: [bold]{exc.rule or 'unspecified'}[/bold]",
            str(exc.detail.get("hint") or "this is a policy decision, not a bug"),
            EXIT_ERROR,
        )
    except ValidationFailed as exc:
        _fail("invalid input", str(exc), "fix the input and run again", EXIT_ERROR)
    except BudgetExceeded as exc:
        _fail(
            "cost budget exhausted",
            str(exc),
            "raise RAGORC_COST__MAX_COST_PER_QUERY_USD, or inspect the trace to see which "
            "stage is looping",
            EXIT_ERROR,
        )
    except RateLimited as exc:
        _fail("rate limited", str(exc), "retry, or lower the concurrency", EXIT_ERROR)
    except ConfigError as exc:
        _fail(
            "misconfigured",
            str(exc.message),
            str(exc.detail.get("hint") or "check your settings"),
            EXIT_CONFIG,
        )
    except ImportError as exc:
        # Every optional import in this library raises with its own pip hint, so
        # the message is already the instruction.
        _fail("missing dependency", str(exc), "install the extra named above", EXIT_CONFIG)
    except RagOrcError as exc:
        _fail(type(exc).__name__, str(exc), "see the log above for the failing stage", EXIT_ERROR)
    except Exception as exc:  # noqa: BLE001 - the CLI's last line of defence
        log.debug("command_failed", error=str(exc), exc_info=True)
        _fail(
            type(exc).__name__,
            str(exc)[:600] or "no message",
            "re-run with --log-level DEBUG for the full traceback",
            EXIT_ERROR,
        )


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
app = typer.Typer(
    name="ragorc",
    help="Retrieval-augmented generation over Qdrant, Postgres and Neo4j.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="DEBUG, INFO, WARNING, ERROR.", metavar="LEVEL"
    ),
    json_logs: bool = typer.Option(
        False,
        "--json-logs/--text-logs",
        help="JSON log lines instead of the console renderer.",
    ),
) -> None:
    """Configure logging before anything else runs.

    Text logs by default, inverting the library's own JSON default. The library
    ships JSON because its logs are shipped to a collector; this command's reader
    is a person at a terminal, and JSON is the wrong format for that audience.
    ``--json-logs`` restores it for a CLI invocation inside a container.

    :func:`configure_logging` is idempotent and configures the *first* caller's
    choice, which is why it happens here — in the callback Typer runs before every
    command — rather than in each command, where the first import to log anything
    would have already fixed the format.
    """
    configure_logging(level=log_level, json_logs=json_logs, redact=True)
    install_uvloop()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
graph_app = typer.Typer(no_args_is_help=True, help="Build and inspect the entity graph.")
app.add_typer(graph_app, name="graph")


@graph_app.command("build")
def graph_build(
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Qdrant collection to read chunks from."
    ),
    tenant: str | None = typer.Option(None, "--tenant", "-t", help="Restrict to one tenant."),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="Stop after this many chunks. For a trial run."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Build the entity graph over an already-indexed corpus.

    This is a second pass on purpose, and it is why `graph.enabled` does not build
    the graph during ingest. Entity resolution and community detection are only
    meaningful over the whole corpus, while ingest deliberately holds one window of
    documents at a time — a graph built per window would resolve entities against a
    fraction of their mentions and detect communities inside an arbitrary slice.

    So: ingest first, then run this. It reads the chunks back from the collection
    rather than re-reading your source documents, so it costs no loading or
    embedding, only the extraction calls.

    Unlike ingest, this does hold the corpus: extraction is per chunk, but
    resolution, community detection and the write are global, which is what makes
    the result a graph rather than a pile of per-window graphs. `--limit` bounds a
    trial run.
    """
    settings = _settings(RAGORC_QDRANT__COLLECTION=collection)

    async def run() -> Any:
        from ragorc.index.graph.build import GraphBuilder

        async with _service(settings) as service:
            engine = service.linear
            store = engine.graph_store()
            await store.ensure_schema()

            chunks: list[Any] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=err,
            ) as progress:
                task = progress.add_task("reading chunks", total=limit)
                async for chunk in engine.vector.scroll(tenant_id=tenant, limit=limit):
                    chunks.append(chunk)
                    progress.update(task, advance=1)

            if not chunks:
                _fail(
                    "Nothing to build from",
                    f"collection {settings.qdrant.collection!r} returned no chunks",
                    "run `ragorc ingest` first, or check --collection and --tenant",
                    2,
                )

            # `embedder=` is not optional in practice. Without it
            # `EntityResolver._merge_by_embedding` returns `{}` on its first line,
            # so resolution stage 3 — the only stage that can merge "Meta" with
            # "Facebook" or survive a transliteration — never runs, `Entity.embedding`
            # is never populated, and `graph.resolution_threshold` is an inert knob
            # that `docs/operations.md` tells operators to lower when graph search
            # comes back empty. The engine has held a built dense embedder since
            # `build()`; the examples pass it and this command did not.
            builder = GraphBuilder(engine.llm, store, embedder=engine.dense, settings=settings)
            return await builder.build(chunks)

    report = _run(run)
    if not settings.graph.enabled:
        # This command writes to Neo4j regardless of the flag — deliberately, since
        # the flag gates *querying* — but `delete` uses the presence of a graph to
        # decide whether to tear one down, and in a fresh process that is the same
        # flag. So a graph built here and left un-flagged is a graph no delete will
        # ever reach. Said where the trap is set rather than where it springs.
        err.print(
            "[yellow]graph.enabled is off[/yellow] — this graph was written, but "
            "`ragorc delete` will not remove from it. Set RAGORC_GRAPH__ENABLED=true "
            "for the pipeline to query it and for deletes to reach it."
        )
    failed = int(report.stages.get("extract", {}).get("chunks_failed", 0) or 0)
    # Computed before the branch, so both renderings reach the same verdict below.
    # `--json` used to return here, above the "no entities extracted" check, so a
    # build where every extraction failed exited 3 without the flag and 0 with it.
    empty = bool(report.chunks_used and not report.entities)
    if as_json:
        console.print_json(data=report.summary())
        if empty:
            raise typer.Exit(3)
        return
    table = Table(title="graph build", box=None)
    table.add_column("stage")
    table.add_column("count", justify="right")
    for label, value in (
        ("chunks read", report.chunks_in),
        ("chunks used", report.chunks_used),
        ("chunks failed", failed),
        ("entities", report.entities),
        ("entities merged", report.merged_entities),
        ("relations", report.relations),
        ("relations dropped", report.dangling_relations),
        ("communities", report.communities),
        ("entities written", report.entities_written),
        ("relations written", report.relations_written),
    ):
        table.add_row(label, f"{value:,}")
    console.print(table)
    console.print(
        f"[dim]cost ${report.usage.cost_usd:.4f} across {report.usage.calls} call(s)[/dim]"
    )

    # A clean table over an empty graph is the wrong thing to print. Extraction
    # failing on every chunk — an exhausted key, a model that stopped answering —
    # otherwise reads as "your corpus has no entities in it".
    if empty:
        _fail(
            "No entities extracted",
            f"{report.chunks_used} chunk(s) were read and none produced an entity"
            + (f"; extraction failed on {failed} of them" if failed else ""),
            "check the log above for the per-chunk error — an exhausted API key and "
            "a corpus with no entities look identical in the counts",
            3,
        )


@app.command()
def init(
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Qdrant collection to create."
    ),
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop and rebuild the Qdrant collection. Destroys its vectors."
    ),
    drop: bool = typer.Option(
        False, "--drop", help="Drop and rebuild the Postgres tables. Destroys their rows."
    ),
    graph: bool | None = typer.Option(
        None, "--graph/--no-graph", help="Create the Neo4j indexes. Defaults to graph.enabled."
    ),
) -> None:
    """Create the Qdrant collection, the Postgres schema and the Neo4j indexes.

    Idempotent, and safe to run on every deploy: each store's schema step creates
    only what is missing. ``--recreate`` and ``--drop`` are the exceptions and they
    say so — they exist for the reindex flow in docs/operations.md, where changing
    an embedding model means the old vectors are not comparable to the new ones and
    a rebuild is the only correct move.

    The stores are prepared **sequentially**, not concurrently. Concurrency would
    save a second and cost the thing that matters here: with three DDL steps in
    flight, a failure gives you three half-finished stores and no clear statement
    of which one to fix.
    """
    settings = _settings(RAGORC_QDRANT__COLLECTION=collection)
    want_graph = settings.graph.enabled if graph is None else graph

    async def run() -> None:
        async with _service(settings) as service:
            engine = service.linear
            steps: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
                (
                    f"qdrant: collection {settings.qdrant.collection!r}",
                    lambda: engine.vector.ensure_collection(recreate=recreate),
                ),
                (
                    f"postgres: schema {settings.postgres.schema_name!r}",
                    lambda: engine.relational.ensure_schema(drop=drop),
                ),
            ]
            if want_graph:
                steps.append(("neo4j: constraints and indexes", engine.graph_store().ensure_schema))

            table = Table(title="initialized", box=None, pad_edge=False)
            table.add_column("store")
            table.add_column("result")
            for label, step in steps:
                with err.status(f"[cyan]{label}"):
                    result = await step()
                detail = f"{len(result)} statement(s)" if isinstance(result, list) else "ready"
                table.add_row(label, f"[green]{detail}[/green]")
            console.print(table)
            if not want_graph:
                err.print(
                    "[dim]neo4j skipped (graph.enabled is false; pass --graph to create it)[/dim]"
                )

    _run(run)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    paths: list[Path] = typer.Argument(
        ..., help="Files or directories to index.", exists=True, resolve_path=True
    ),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Walk directories."),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        help="Chunking strategy: auto, late, contextual, early.",
        metavar="STRATEGY",
    ),
    graph: bool = typer.Option(
        False, "--graph", help="Extract entities and relations into Neo4j. One LLM call per chunk."
    ),
    raptor: bool = typer.Option(
        False, "--raptor", help="Build the RAPTOR summary tree. Needs ragorc[raptor]."
    ),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Qdrant collection."),
    tenant: str | None = typer.Option(None, "--tenant", "-t", help="Tenant to write under."),
    include: list[str] = typer.Option([], "--include", help="Glob to include, repeatable."),
    exclude: list[str] = typer.Option([], "--exclude", help="Glob to exclude, repeatable."),
    batch_size: int = typer.Option(
        _INGEST_BATCH_DOCUMENTS, "--batch-size", min=1, help="Documents per pipeline call."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help=(
            "Reindex documents the checksum skip would drop. Needed when building "
            "into a new collection: the skip is a question about Postgres and does "
            "not know which collection you are writing."
        ),
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Load, split, embed and write documents.

    Documents are discovered first, then fed to the pipeline in batches so the
    progress bar counts something real. That ordering is a deliberate trade: it
    holds the loaded document text in memory, which the pipeline's own
    ``ingest(path)`` never does, in exchange for a bar that knows its total. For a
    corpus large enough that the text does not fit, ingest by path through the
    service instead — ``POST /ingest`` with ``paths`` — and watch the log.

    The optional stages are flags rather than settings lookups because they decide
    what gets *loaded*: ``--raptor`` pulls in UMAP and scikit-learn, ``--graph``
    turns on one LLM call per chunk. Both are opt-in per corpus, not per
    deployment.
    """
    settings = _settings(
        RAGORC_QDRANT__COLLECTION=collection,
        RAGORC_TENANT_ID=tenant,
        RAGORC_INDEXING__CHUNKING_STRATEGY=strategy,
        RAGORC_GRAPH__ENABLED=graph or None,
        RAGORC_INDEXING__RAPTOR_ENABLED=raptor or None,
    )

    async def run() -> None:
        documents = await _discover(
            paths, settings, recursive=recursive, include=include, exclude=exclude, tenant=tenant
        )
        if not documents:
            err.print("[yellow]nothing to ingest[/yellow]")
            raise typer.Exit(EXIT_OK)

        async with _service(settings) as service:
            report = await _ingest_batched(service, documents, batch_size, force=force)

        if as_json:
            # `summary()` is the counters. The warnings are where the read-back
            # says the vector store holds fewer points than this run wrote — the
            # one line that distinguishes "indexed" from "retrievable" — and a
            # machine caller had no way to see it.
            _print_json(
                {
                    **report.summary(),
                    "warnings": list(report.warnings),
                    "rejections": [[doc, why] for doc, why in report.rejected],
                    "failures": [[doc, why] for doc, why in report.failed],
                }
            )
            return
        console.print(_report_table(report))
        if report.rejected or report.failed:
            _print_problems(report)

    _run(run)


@app.command()
def documents(
    source: str | None = typer.Option(None, "--source", "-s", help="Substring of the path."),
    tenant: str | None = typer.Option(None, "--tenant", "-t", help="Tenant to list."),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Qdrant collection."),
    limit: int = typer.Option(100, "--limit", "-n", min=1, help="Documents to list."),
    as_json: bool = typer.Option(False, "--json", help="Print as JSON."),
) -> None:
    """List what is indexed: document id, source and chunk count.

    Exists because `ragorc delete` had nothing to name. Its help pointed at
    `ragorc query --json` for an id, and that needs a working LLM — so a
    deployment out of model credit could not find out what it had indexed, let
    alone remove any of it. `inspect` reports counts, not ids.

    `--source` matches a substring of the path, which is how you get from "I
    deleted this file" to the id that removes it.
    """
    settings = _settings(
        RAGORC_QDRANT__COLLECTION=collection, RAGORC_TENANT_ID=tenant
    )

    async def run() -> None:
        async with _service(settings) as service:
            rows = await service.engine.documents(tenant_id=tenant, source=source, limit=limit)

        if as_json:
            _print_json(rows)
            return
        if not rows:
            err.print("[yellow]nothing indexed matches[/yellow]")
            return
        table = Table(title="documents", box=None, pad_edge=False)
        table.add_column("document id")
        table.add_column("chunks", justify="right")
        table.add_column("source")
        for row in rows:
            table.add_row(str(row["document_id"]), str(row["chunks"]), str(row["source"]))
        console.print(table)

    _run(run)


def _delete_exit(report: Any) -> None:
    """Exit non-zero for the two outcomes a caller must not read as success.

    Shared by both rendering modes, because it is the *outcome* that decides.
    `--json` returned before these checks, so the same failing delete exited 0
    with the flag and 1 without it — and a pipeline that parses the JSON is
    exactly the caller that cannot afford to miss it.
    """
    if not report.complete or not report.deleted:
        raise typer.Exit(EXIT_ERROR)


@app.command()
def delete(
    document_ids: list[str] = typer.Argument(..., help="Document ids to remove."),
    tenant: str | None = typer.Option(None, "--tenant", "-t", help="Tenant that owns them."),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Qdrant collection."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
    as_json: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Remove documents from Qdrant, Postgres, Neo4j and the answer cache.

    Until this existed there was no supported way to take a document out of the
    index: the only deletion anywhere was the stale-purge inside re-ingest, which
    replaces a document and cannot remove one.

    Ids, not globs. A filtered delete is one typo away from emptying an index and
    there is nothing behind it — no tombstone, no undo — so the blast radius is
    kept equal to what was typed. Get the ids from `ragorc documents`, which needs
    no model credit — this used to point at `ragorc query --json`, which does.

    Confirms first unless `--yes`, because this is the one command in the CLI that
    destroys data.
    """
    settings = _settings(
        RAGORC_QDRANT__COLLECTION=collection, RAGORC_TENANT_ID=tenant
    )
    ids = [i for i in document_ids if i]
    if not ids:
        err.print("[yellow]no document ids given[/yellow]")
        raise typer.Exit(EXIT_OK)

    if not yes:
        listed = ", ".join(ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")
        typer.confirm(
            f"Permanently delete {len(ids)} document(s) — {listed} — from every store?",
            abort=True,
        )

    async def run() -> None:
        async with _service(settings) as service:
            report = await service.engine.delete(ids, tenant_id=tenant)

        # The exit status is a property of the outcome, not of the rendering.
        # `--json` used to `return` from inside this branch, above both checks
        # below, so a delete that matched nothing exited 0 with the flag and 1
        # without it — and CI that reads the JSON saw success.
        if as_json:
            _print_json(
                {
                    "requested": report.documents,
                    "found": report.found,
                    "deleted": report.deleted,
                    "vectors": report.vectors,
                    "rows": report.rows,
                    "entities": report.entities,
                    "communities": report.communities,
                    "answers_invalidated": report.answers_invalidated,
                    "complete": report.complete,
                    "errors": report.errors,
                    "skipped": report.skipped,
                }
            )
            _delete_exit(report)
            return

        # Two columns, because half these numbers are requests and half are
        # results. One column headed "removed" printed `documents 1` for an id
        # that never existed, and `cached answers 1` for an invalidation whose own
        # primitive documents its return as "the number of documents the removal
        # was requested for, not the number of points removed". The honest
        # statement was there and was lost at every layer above it.
        table = Table(title="delete", box=None, pad_edge=False)
        table.add_column("")
        table.add_column("count", justify="right")
        table.add_row("documents requested", str(report.documents))
        table.add_row("documents found", str(report.found))
        table.add_row("vectors removed", str(report.vectors))
        table.add_row("rows removed", str(report.rows))
        table.add_row("orphaned entities removed", str(report.entities))
        table.add_row("empty communities removed", str(report.communities))
        table.add_row("cache invalidated for", f"{report.answers_invalidated} document(s)")
        console.print(table)

        for store, why in report.skipped.items():
            # A store that was not consulted is not a store that answered. Silence
            # here is how `complete: true` came to mean "except the one I did not
            # ask", which for a compliance delete is the wrong kind of quiet.
            err.print(f"[yellow]{store} not consulted[/yellow]: {why}")

        if report.complete and not report.deleted:
            missing = report.documents - report.found
            err.print(
                f"[yellow]{missing} of {report.documents} id(s) matched nothing[/yellow] — "
                "unknown id, already deleted, or owned by another tenant"
            )
        elif not report.complete:
            # Not an exception: the stores that succeeded are already done, and a
            # caller needs to know which ones to retry rather than being told the
            # whole thing failed.
            for store, message in report.errors.items():
                err.print(f"[red]{store}[/red]: {message}")
            err.print("[yellow]the document may still be retrievable — retry[/yellow]")
        _delete_exit(report)

    _run(run)


async def _discover(
    paths: Sequence[Path],
    settings: Settings,
    *,
    recursive: bool,
    include: Sequence[str],
    exclude: Sequence[str],
    tenant: str | None,
) -> list[Any]:
    """Load documents from every path, reusing the loaders' own dispatch.

    :class:`~ragorc.index.loaders.DirectoryLoader` is used for directories rather
    than a local walk: it already prunes binaries by suffix, sniffs unknown
    extensions, enforces the size ceiling, and — the part worth not
    reimplementing — records a per-file failure and keeps going, so one truncated
    PDF does not abort a ten-thousand-file corpus.
    """
    from ragorc.index.loaders import DirectoryLoader, load

    documents: list[Any] = []
    failures: list[tuple[str, str]] = []
    with err.status("[cyan]discovering documents"):
        for path in paths:
            if path.is_dir():
                loader = DirectoryLoader(
                    include=include or None,
                    exclude=exclude or None,
                    recursive=recursive,
                    tenant_id=tenant,
                    settings=settings,
                )
                documents.extend(await loader.load(path))
                failures.extend(loader.failures)
            else:
                documents.extend(await load(path, tenant_id=tenant, settings=settings))
    err.print(f"discovered [bold]{len(documents)}[/bold] document(s)")
    for source, reason in failures[:10]:
        err.print(f"[yellow]unreadable[/yellow] {_plain(source)}: {_plain(reason)}")
    if len(failures) > 10:
        err.print(f"[yellow]... and {len(failures) - 10} more unreadable file(s)[/yellow]")
    return documents


async def _ingest_batched(
    service: RagService, documents: Sequence[Any], batch: int, *, force: bool = False
) -> Any:
    """Ingest in batches, advancing a bar, and merge the reports into one."""
    from ragorc.index.pipeline import IngestReport

    merged = IngestReport()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    ) as progress:
        task = progress.add_task("indexing", total=len(documents))
        # One bulk-load window across every batch. Each `ingest` call opens its
        # own otherwise, so a 640-document directory turned HNSW construction off
        # and back on ten times and rebuilt the graph over everything written so
        # far at each exit — the repeated rebuild the single-window design removed,
        # reinstated by the batching one level up.
        async with service.linear.ingest_pipeline.bulk_run():
            for start in range(0, len(documents), batch):
                window = list(documents[start : start + batch])
                report = await service.linear.ingest(window, force=force)
                _merge_report(merged, report)
                progress.update(
                    task,
                    advance=len(window),
                    description=f"indexing · {merged.chunks_created} chunks · "
                    f"{merged.documents_skipped} unchanged",
                )
    return merged


def _merge_report(into: Any, other: Any) -> None:
    """Accumulate one batch's report into the run's total.

    Counters add. Timings add across batches for the same reason the pipeline adds
    them across concurrent documents: the ratios identify the bottleneck, which is
    what these numbers are read for, and wall-clock per stage would only report
    the batching back at you.
    """
    for field in (
        "documents_in",
        "documents_indexed",
        "documents_skipped",
        "documents_rejected",
        "documents_duplicate",
        "documents_failed",
        "documents_empty",
        "chunks_created",
        "vectors_written",
        "total_ms",
    ):
        setattr(into, field, getattr(into, field) + getattr(other, field))
    for stage, value in other.timings_ms.items():
        into.timings_ms[stage] = into.timings_ms.get(stage, 0.0) + value
    into.usage = into.usage + other.usage
    into.strategy = other.strategy or into.strategy
    # Not a sum: it is the size of the whole collection, read back after each
    # batch's flush, so the last batch's answer is the run's answer. Summing it
    # would multiply the collection by the number of batches.
    if other.points_in_store is not None:
        into.points_in_store = other.points_in_store
    into.rejected.extend(other.rejected)
    into.failed.extend(other.failed)
    into.warnings.extend(w for w in other.warnings if w not in into.warnings)


def _citation_label(citation: Any) -> str:
    """The most human-followable identifier a citation carries.

    Prefers the source path, falls back to the document id, and shortens a bare
    UUID rather than spending 36 columns on something nobody can act on.
    """
    source = getattr(citation, "source", None)
    if source:
        # The basename, not the absolute path. A citation column is ~30 characters
        # and truncates from the right, so a full path shows
        # "/Users/maddie/Desktop/Work/AI/ra…" — every character of which is the
        # part that does not identify the document, with the filename cut off.
        text = str(source)
        return pathlib.PurePath(text).name if ("/" in text or "\\" in text) else text
    identifier = str(citation.document_id or citation.chunk_id or "")
    if len(identifier) == 36 and identifier.count("-") == 4:
        return identifier[:8] + "\u2026"
    return identifier


def _report_table(report: Any) -> Table:
    summary = report.summary()
    table = Table(title="ingest", box=None, pad_edge=False)
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right")
    for key in (
        "documents_in",
        "indexed",
        "skipped",
        "rejected",
        "duplicate",
        "failed",
        "empty",
        "chunks",
        "vectors",
        # What the store says it holds, beside what this run sent. The operator's
        # cross-check for "did it land?", and the reason it is printed rather than
        # only logged: the two stores share no transaction.
        "points_in_store",
        "strategy",
        "skip_rate",
        "llm_calls",
        "cost_usd",
        "total_ms",
    ):
        table.add_row(key, str(summary[key]))
    return table


def _print_problems(report: Any) -> None:
    for label, rows in (("rejected", report.rejected), ("failed", report.failed)):
        for source, reason in list(rows)[:10]:
            err.print(f"[yellow]{_plain(label)}[/yellow] {_plain(source)}: {_plain(reason)}")
        if len(rows) > 10:
            err.print(f"[yellow]... and {len(rows) - 10} more {label}[/yellow]")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------
@app.command()
def query(
    question: str = typer.Argument(..., help="The question to answer."),
    pipeline: PipelineName = typer.Option(
        PipelineName.AUTO, "--pipeline", "-p", help="Which graph to run."
    ),
    top_k: int | None = typer.Option(None, "--top-k", "-k", min=1, help="Chunks to retrieve."),
    stream: bool = typer.Option(False, "--stream", help="Print tokens as they arrive."),
    as_json: bool = typer.Option(False, "--json", help="Print the whole response as JSON."),
    show_trace: bool = typer.Option(False, "--show-trace", help="Print the per-stage trace."),
    tenant: str | None = typer.Option(None, "--tenant", "-t", help="Tenant to search within."),
    filters: list[str] = typer.Option(
        [], "--filter", "-f", help="Metadata predicate, key=value. Repeatable.", metavar="K=V"
    ),
) -> None:
    """Answer a question and show what it was based on and what it cost.

    ``--stream`` prints tokens as they are produced and then stops: a streamed
    answer has not been groundedness-checked, because that judgement needs the
    finished text, and printing a verification footer under text that was never
    verified would be a lie told in a convenient place. Without ``--stream`` the
    answer arrives whole, with its citations and its groundedness score.
    """
    settings = _settings(RAGORC_TENANT_ID=tenant)
    request = QueryRequest(
        question=question,
        pipeline=pipeline,
        top_k=top_k,
        tenant_id=tenant or settings.tenant_id,
        filters=_parse_filters(filters),
    )

    async def run() -> None:
        async with _service(settings) as service:
            if stream:
                await _stream_answer(service, request)
                return
            started = time.perf_counter()
            response = await service.query(request, principal="cli")
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            if as_json:
                _print_json(response.model_dump(mode="json"))
                return
            _print_answer(response, elapsed_ms=elapsed_ms, show_trace=show_trace)

    _run(run)


async def _stream_answer(service: RagService, request: QueryRequest) -> None:
    """Print the token stream, then its footer.

    The events are consumed rather than the deltas alone because the terminal
    ``done`` event is where the bill and the unverified disclaimer live, and an
    ``error`` event is the only way a failure can be reported once the 200 is
    already on the wire.
    """
    async for event, data in service.stream(request, principal="cli"):
        if event == "token":
            console.print(data, end="", markup=False, highlight=False)
        elif event == "warning":
            err.print(f"[yellow]{_plain(data)}[/yellow]")
        elif event == "done":
            console.print()
            summary = orjson.loads(data)
            cost = summary.get("cost", {})
            console.print(
                f"[dim]{summary.get('tokens_emitted', 0)} deltas · "
                f"${cost.get('total_cost_usd', 0.0):.6f} · "
                f"{cost.get('calls', 0)} call(s) · [yellow]unverified[/yellow][/dim]"
            )
        elif event == "error":
            payload = orjson.loads(data)
            _fail(
                payload.get("error", "StreamError"),
                str(payload.get("message", "")),
                "the stream had already begun, so the failure arrives as an event",
                EXIT_ERROR,
            )


def _print_answer(response: Any, *, elapsed_ms: float, show_trace: bool) -> None:
    title = "[yellow]abstained[/yellow]" if response.abstained else "answer"
    console.print(Panel(response.answer, title=title, border_style="cyan", highlight=False))

    if response.abstained and response.abstain_reason:
        console.print(f"[dim]reason: {_plain(response.abstain_reason)}[/dim]")

    if response.citations:
        table = Table(title="citations", box=None, pad_edge=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("document")
        table.add_column("support", justify="right")
        table.add_column("quote")
        for index, citation in enumerate(response.citations, start=1):
            # `source` first: it is the filename a reader can actually open.
            # `document_id` and `chunk_id` are content-derived UUIDs — stable and
            # useful for joins, useless in a citation, because a reader cannot
            # follow "679babc7-d94c-..." back to anything.
            table.add_row(
                str(index),
                _citation_label(citation),
                f"{citation.support:.2f}",
                (citation.quote or "").replace("\n", " ")[:100],
            )
        console.print(table)
    elif not response.abstained:
        err.print("[yellow]no citations: the answer is unattributed[/yellow]")

    if show_trace and response.trace:
        table = Table(title="trace", box=None, pad_edge=False)
        table.add_column("stage")
        table.add_column("ms", justify="right")
        table.add_column("detail", style="dim")
        for step in response.trace:
            table.add_row(
                step.name,
                f"{step.duration_ms:.1f}",
                ", ".join(f"{k}={v}" for k, v in list(step.detail.items())[:4]),
            )
        console.print(table)

    for warning in response.warnings:
        err.print(f"[yellow]{_plain(warning)}[/yellow]")

    cost = response.metadata.get("cost", {}) if isinstance(response.metadata, dict) else {}
    console.print(
        "[dim]"
        f"{response.pipeline.value} · {len(response.chunks)} chunk(s) · "
        f"grounded {response.groundedness:.2f} · confidence {response.confidence:.2f} · "
        f"${response.usage.cost_usd:.6f} over {response.usage.calls} call(s) · "
        f"{response.usage.total_tokens} tokens · {elapsed_ms:.0f} ms"
        + (f" · cache hit ({cost.get('cache_hit_rate', 0)})" if response.cached else "")
        + "[/dim]"
    )


def _parse_filters(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse ``key=value`` flags, reading each value as JSON when it is JSON.

    So ``--filter year=2024`` filters on the number, ``--filter tag=beta`` on the
    string, and ``--filter 'ids=[1,2]'`` on the list. Without the JSON attempt
    every metadata filter would be a string comparison against a numeric payload
    field and would silently match nothing.
    """
    out: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key.strip():
            _fail(
                "bad filter",
                f"{pair!r} is not key=value",
                "example: --filter year=2024",
                EXIT_CONFIG,
            )
        try:
            out[key.strip()] = orjson.loads(raw)
        except orjson.JSONDecodeError:
            out[key.strip()] = raw
    return out


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
@app.command(name="eval")
def evaluate(
    dataset: Path = typer.Argument(
        ..., help="JSONL or JSON dataset of eval cases.", exists=True, resolve_path=True
    ),
    compare: str | None = typer.Option(
        None,
        "--compare",
        help=(
            "A pipeline name to A/B against (paired bootstrap), or a path to a JSON file of "
            "RAGORC_* overrides to score as a second configuration."
        ),
        metavar="PIPELINE|CONFIG",
    ),
    pipeline: PipelineName = typer.Option(
        PipelineName.AUTO, "--pipeline", "-p", help="Baseline pipeline."
    ),
    top_k: int | None = typer.Option(None, "--top-k", "-k", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Score only the first N cases."),
    concurrency: int = typer.Option(4, "--concurrency", min=1, max=32),
    tenant: str | None = typer.Option(None, "--tenant", "-t"),
    as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
) -> None:
    """Score a dataset, optionally against a second pipeline or configuration.

    Two comparison modes, and the difference matters:

    * ``--compare <pipeline>`` scores both pipelines over the same cases in the
      same process against the same index, and reports a **paired bootstrap** —
      per-case differences resampled, so the answer is "this is a real change" or
      "this is noise at this sample size" rather than two means to eyeball.
    * ``--compare <config.json>`` scores a second *configuration*: the file's
      ``RAGORC_*`` overrides are applied and the service is rebuilt. That is how a
      different embedding model, ``fetch_k`` or reranker gets measured — things a
      pipeline name cannot express — and it is reported as side-by-side numbers,
      because the two runs are separate aggregations and the harness pairs
      per-case series within a run.
    """
    base = _settings(RAGORC_TENANT_ID=tenant)
    compare_pipeline = _as_pipeline(compare)
    request = EvalRequest(
        dataset=str(dataset),
        pipeline=pipeline,
        compare=[compare_pipeline] if compare_pipeline else [],
        top_k=top_k,
        limit=limit,
        concurrency=concurrency,
        tenant_id=tenant or base.tenant_id,
    )

    # The operator typed this path, so it is trusted the way any argument to a
    # local command is. `load_eval_items` defaults to the server's ingest
    # allowlist — right for a caller-supplied path over HTTP, and it confined
    # `make eval` out of existence here. Resolved out here rather than in the
    # coroutine: it is a syscall, and this is request setup, not loop work.
    dataset_roots = [Path(dataset).expanduser().resolve().parent]

    async def run() -> None:
        # Validate the dataset before building anything: a typo in the file costs
        # nothing to find and an ONNX model load to find late.
        cases = await load_eval_items(request, roots=dataset_roots)
        err.print(f"loaded [bold]{len(cases)}[/bold] case(s) from {dataset}")

        async with _service(base) as service:
            baseline = await service.evaluate(request, dataset_roots=dataset_roots)

        variant = None
        if compare and compare_pipeline is None:
            overrides = _load_overrides(Path(compare))
            settings = _settings(**overrides)
            err.print(f"[cyan]variant[/cyan] {compare}: {', '.join(sorted(overrides))}")
            async with _service(settings) as service:
                variant = await service.evaluate(request, dataset_roots=dataset_roots)

        if as_json:
            _print_json(
                {
                    "baseline": baseline.model_dump(mode="json"),
                    **({"variant": variant.model_dump(mode="json")} if variant else {}),
                }
            )
            return

        rows = [(row.pipeline.value, row) for row in baseline.results]
        if variant is not None:
            rows.extend((f"{row.pipeline.value} (variant)", row) for row in variant.results)
        for table in _metrics_tables(rows):
            console.print(table)
        for comparison in baseline.comparisons:
            table, hidden = _comparison_table(comparison)
            console.print(table)
            if hidden:
                err.print(f"[dim]{hidden} inconclusive metric(s) hidden; --json has all[/dim]")
        for warning in [*baseline.warnings, *(variant.warnings if variant else [])]:
            err.print(f"[yellow]{_plain(warning)}[/yellow]")

    _run(run)


def _as_pipeline(value: str | None) -> PipelineName | None:
    if not value:
        return None
    try:
        return PipelineName(value)
    except ValueError:
        return None


def _load_overrides(path: Path) -> dict[str, Any]:
    """Read a JSON file of ``RAGORC_*`` overrides.

    Environment variable names rather than a nested structure, because that is the
    configuration surface this library actually has: one flat namespace with ``__``
    for nesting, documented on :class:`~ragorc.core.settings.Settings`. Inventing a
    second, prettier schema here would mean two things to keep in sync and one of
    them undocumented.
    """
    if not path.is_file():
        _fail(
            "no such comparison",
            f"{path} is neither a pipeline name nor a readable file",
            f"pipelines: {', '.join(p.value for p in PipelineName)}",
            EXIT_CONFIG,
        )
    try:
        payload = orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError as exc:
        _fail("invalid comparison config", str(exc), "the file must be a JSON object", EXIT_CONFIG)
    if not isinstance(payload, dict) or not payload:
        _fail(
            "invalid comparison config",
            "expected a non-empty JSON object of RAGORC_* overrides",
            'example: {"RAGORC_RETRIEVAL__FETCH_K": 100}',
            EXIT_CONFIG,
        )
    unknown = [key for key in payload if not str(key).startswith("RAGORC_")]
    if unknown:
        _fail(
            "invalid comparison config",
            f"these keys are not settings variables: {', '.join(map(str, unknown))}",
            "every key must start with RAGORC_ (see Settings)",
            EXIT_CONFIG,
        )
    return dict(payload)


_HEADLINE_RETRIEVAL = (
    "recall@10",
    "recall@50",
    "ndcg@10",
    "mrr",
    # The same four at document granularity. A dataset labelled by source
    # document rather than chunk id — which the shipped one is — produces only
    # these, and without a column for them the table showed nothing at all.
    "doc_recall@10",
    "doc_ndcg@10",
    "doc_mrr",
)
"""The four retrieval numbers worth a terminal column.

The harness reports every metric at every *k* — six ``k``s times four families
plus MRR and MAP is thirty-odd columns, which a terminal renders as ellipses and
nobody reads. These four are the ones that answer different questions, which is
why they are the ones kept:

``recall@10``  what the generator actually saw.
``recall@50``  the *ceiling* ``fetch_k`` set. A gap between the two is a reranking
               problem; a low ``recall@50`` is a retrieval problem, and no
               reranker can fix it.
``ndcg@10``    ranking quality inside that window, not just presence.
``mrr``        how far down the first hit was, over the whole returned list.

``--json`` carries all of them; this is the reading, not the record."""


def _metrics_tables(rows: Sequence[tuple[str, Any]]) -> list[Table]:
    """Two tables: what the run *did*, and how *good* it was.

    Split rather than combined because one table with both is fifteen columns, and
    a terminal renders fifteen columns as ellipses. The split is also the natural
    one — the first table needs no labelled data and is always present, the second
    exists only for a dataset that carries labels.
    """
    operational = Table(title="eval · run", box=None, pad_edge=False)
    operational.add_column("run")
    for column in ("cases", "errors", "abstain", "grounded", "cites", "p50 ms", "p95 ms", "$/q"):
        operational.add_column(column, justify="right")
    for label, row in rows:
        operational.add_row(
            label,
            str(row.items),
            str(row.errors),
            f"{row.abstain_rate:.2f}",
            f"{row.groundedness_mean:.3f}",
            f"{row.citation_coverage:.2f}",
            f"{row.latency_p50_ms:.0f}",
            f"{row.latency_p95_ms:.0f}",
            f"{row.cost_usd_per_query:.5f}",
        )

    available = {key for _label, row in rows for key in row.retrieval}
    columns = [key for key in _HEADLINE_RETRIEVAL if key in available]
    columns += sorted({key for _label, row in rows for key in row.answer})
    if not columns:
        return [operational]

    quality = Table(
        title="eval · quality",
        box=None,
        pad_edge=False,
        caption="labelled cases only — --json carries every metric at every k",
    )
    quality.add_column("run")
    quality.add_column("labelled", justify="right")
    for column in columns:
        quality.add_column(column, justify="right")
    for label, row in rows:
        values = {**row.retrieval, **row.answer}
        quality.add_row(
            label,
            str(row.labelled),
            *[f"{values[key]:.3f}" if key in values else "-" for key in columns],
        )
    return [operational, quality]


_EMPTY = "—"
"""Shown where a statistic is undefined, which is not the same as zero."""


def _number(value: object, *, sign: bool = False) -> str:
    """Format a statistic, or mark it undefined.

    `paired_bootstrap` returns `None` for every field when the two runs share no
    comparable case. Formatting that with `:.3f` raised
    `TypeError: unsupported format string passed to NoneType.__format__` and took
    down the whole `--compare` render, including the metrics that did compare.
    Substituting 0.0 would be worse: it reads as a measured no-difference.
    """
    if value is None or not isinstance(value, (int, float)):
        return _EMPTY
    return f"{value:+.3f}" if sign else f"{value:.3f}"


def _comparison_table(comparison: dict[str, Any]) -> tuple[Table, int]:
    """Render an A/B, showing everything that moved plus the headline metrics.

    Filtering on the verdict rather than on a metric list is what keeps this table
    honest at a readable size: a metric with a conclusive verdict is by definition
    the news, and one that is inconclusive and not a headline is a row that says
    "no information" thirty times. The count of what was hidden is returned so the
    caller can say so rather than silently dropping it.
    """
    metrics: dict[str, Any] = comparison.get("metrics") or {}
    headline = (*_HEADLINE_RETRIEVAL, "latency_ms", "cost_usd", "llm_calls")
    shown = {
        name
        for name, result in metrics.items()
        if result.get("verdict") != "inconclusive" or name in headline
    }
    table = Table(
        title=f"{comparison.get('candidate')} vs {comparison.get('baseline')}",
        box=None,
        pad_edge=False,
        caption=(
            "inconclusive means the interval includes zero — consistent with noise "
            "at this sample size, not equal"
        ),
    )
    table.add_column("metric")
    for column in ("baseline", "candidate", "delta"):
        table.add_column(column, justify="right")
    table.add_column("95% CI", justify="right", no_wrap=True)
    table.add_column("p", justify="right")
    table.add_column("verdict")
    colours = {"better": "green", "worse": "red"}
    for name, result in metrics.items():
        if name not in shown:
            continue
        verdict = str(result.get("verdict", ""))
        colour = colours.get(verdict, "dim")
        # Every statistic is `None` when the two runs share no scored case, and
        # the interval is a `ci` *pair* — not `ci_low`/`ci_high`, which never
        # existed, so `.get(..., 0.0)` pinned this column to "+0.000 to +0.000"
        # on every row. That is the one number that says whether a difference is
        # real, so a hard zero there is worse than no column at all.
        low, high = (result.get("ci") or (None, None))[:2]
        table.add_row(
            name,
            _number(result.get("baseline")),
            _number(result.get("candidate")),
            _number(result.get("difference"), sign=True),
            _EMPTY if low is None or high is None else f"{low:+.3f} to {high:+.3f}",
            _number(result.get("p_value")),
            f"[{colour}]{_plain(verdict)}[/{colour}]",
        )
    return table, len(metrics) - len(shown)


# ---------------------------------------------------------------------------
# bench
# ---------------------------------------------------------------------------
_BENCH_VARIANTS: tuple[tuple[str, dict[str, Any], bool, bool], ...] = (
    ("dense", {"use_dense": True, "use_sparse": False}, False, False),
    ("sparse", {"use_dense": False, "use_sparse": True}, False, False),
    ("hybrid", {"use_dense": True, "use_sparse": True}, False, False),
    ("hybrid+rerank", {"use_dense": True, "use_sparse": True}, True, False),
    ("hybrid+compress", {"use_dense": True, "use_sparse": True}, True, True),
)
"""The five configurations worth timing, cumulative left to right so each column
shows what the previous one cost. No LLM call appears in any of them: retrieval is
the part these flags change, and folding synthesis latency in would bury a 20 ms
difference under a two-second provider round trip."""


@app.command()
def bench(
    questions: list[str] = typer.Argument(None, help="Questions to time."),
    runs: int = typer.Option(3, "--runs", "-n", min=1, help="Repetitions per question."),
    queries: Path | None = typer.Option(
        None, "--queries", help="File of questions, one per line.", exists=True
    ),
    top_k: int | None = typer.Option(None, "--top-k", "-k", min=1),
    tenant: str | None = typer.Option(None, "--tenant", "-t"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Time the retrieval configurations against your own corpus.

    Latency only, and free: no synthesis call is made, so this can be run in a
    loop while tuning ``fetch_k`` or ``hnsw_ef_search``. What it deliberately does
    not measure is *quality* — recall and nDCG need labelled data, which is what
    ``ragorc eval`` is for. Every default in this library is a reasonable prior,
    not a substitute for measuring both halves on your corpus.

    The first run of each variant is discarded when more than one is requested.
    It pays for the ONNX warm-up, the connection handshake and a cold page cache,
    and including it makes every variant look like whichever one ran first.
    """
    settings = _settings(RAGORC_TENANT_ID=tenant)
    prompts = list(questions or [])
    if queries is not None:
        # `#` starts a comment, as it does in the eval dataset loader. Without
        # this the file's own header is benchmarked as a question.
        prompts.extend(
            stripped
            for line in queries.read_text().splitlines()
            if (stripped := line.strip()) and not stripped.startswith("#")
        )
    if not prompts:
        _fail(
            "nothing to benchmark",
            "no questions given",
            "pass questions as arguments or --queries FILE",
            EXIT_CONFIG,
        )

    async def run() -> None:
        async with _service(settings) as service:
            results = await _bench(
                service, prompts, settings, runs=runs, top_k=top_k, tenant=tenant
            )
        if as_json:
            _print_json(results)
            return
        table = Table(
            title=f"retrieval latency · {len(prompts)} question(s) x {runs} run(s)",
            box=None,
            pad_edge=False,
        )
        table.add_column("variant")
        for column in ("p50 ms", "p95 ms", "mean ms", "results", "errors"):
            table.add_column(column, justify="right")
        for row in results:
            table.add_row(
                row["variant"],
                f"{row['p50_ms']:.1f}",
                f"{row['p95_ms']:.1f}",
                f"{row['mean_ms']:.1f}",
                f"{row['mean_results']:.1f}",
                str(row["errors"]),
            )
        console.print(table)
        err.print("[dim]latency only — run `ragorc eval` for recall and nDCG[/dim]")

    _run(run)


async def _bench(
    service: RagService,
    prompts: Sequence[str],
    settings: Settings,
    *,
    runs: int,
    top_k: int | None,
    tenant: str | None,
) -> list[dict[str, Any]]:
    from ragorc.retrieve import build_compressor

    engine = service.linear
    limit = int(top_k or settings.retrieval.top_k)
    scope = scope_filter(None, tenant or settings.tenant_id, settings.security)
    compressor = build_compressor("embedding_filter", embedder=engine.dense, settings=settings)

    out: list[dict[str, Any]] = []
    total = len(_BENCH_VARIANTS) * len(prompts) * runs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err,
    ) as progress:
        task = progress.add_task("benchmarking", total=total)
        for name, legs, rerank, compress in _BENCH_VARIANTS:
            samples: list[float] = []
            counts: list[int] = []
            errors = 0
            for text in prompts:
                for attempt in range(runs):
                    query = Query(
                        text=text,
                        top_k=limit,
                        tenant_id=scope.get("tenant_id"),
                        filters=dict(scope),
                    )
                    started = time.perf_counter()
                    try:
                        # `vector_leg`, not `hybrid`: that is what every route
                        # retrieves through, and on a multi-representation index
                        # the two differ by the docstore round trip that resolves
                        # a derived unit back to its source — measured at 19.7 ms
                        # against 34.8 ms on a 48-chunk toy corpus, a gap that
                        # grows with the docstore. A benchmark of a path nothing
                        # serves is a number with no referent.
                        #
                        # `fetch_k` too, because `retrieve_detailed` returns
                        # `max(fetch_k, top_k)` candidates: below the default 50,
                        # `--top-k` moved a floor and left the measured width at
                        # 50, so `-k 10` and `-k 20` benchmarked the same work.
                        result = await engine.vector_leg.retrieve_detailed(
                            query, top_k=limit, fetch_k=limit, use_variants=False, **legs
                        )
                        chunks = result.chunks
                        if rerank and chunks:
                            chunks = await engine.reranker.rerank_chunks(query, chunks)
                        if compress and chunks:
                            chunks, _usage = await compressor.compress(query, chunks)
                    except RagOrcError as exc:
                        errors += 1
                        log.warning("bench_leg_failed", variant=name, error=str(exc)[:200])
                        progress.advance(task)
                        continue
                    elapsed = (time.perf_counter() - started) * 1000.0
                    progress.advance(task)
                    # Discard the warm-up pass, but only when there is another to
                    # keep: with --runs 1 the cold number is all there is, and
                    # reporting nothing would be worse than reporting it.
                    if attempt == 0 and runs > 1:
                        continue
                    samples.append(elapsed)
                    counts.append(len(chunks))
            out.append(
                {
                    "variant": name,
                    "samples": len(samples),
                    "p50_ms": _percentile(samples, 50),
                    "p95_ms": _percentile(samples, 95),
                    "mean_ms": round(statistics.fmean(samples), 2) if samples else 0.0,
                    "mean_results": round(statistics.fmean(counts), 2) if counts else 0.0,
                    "errors": errors,
                }
            )
    return out


def _percentile(values: Sequence[float], pct: int) -> float:
    """Nearest-rank percentile: the ``ceil(pct/100 * n)``-th smallest sample.

    Not interpolated: with the tens of samples a benchmark run produces, an
    interpolated p95 reports a latency no query actually had, and the nearest
    observed sample is a real measurement.

    The ceiling is what makes the rank the *defined* one rather than a rounded
    guess. ``round`` breaks ties to even, so ``round(n/2)`` for n=5, 9, 13 …
    rounds a rank of x.5 *down* and reports the sample below the median as p50 —
    off by one for every fourth sample count, which in a table of five variants
    is indistinguishable from one of them being faster.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return round(ordered[index], 2)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------
@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", help="Interface to bind."),
    port: int | None = typer.Option(None, "--port", "-p", min=1, max=65_535),
    reload: bool = typer.Option(False, "--reload", help="Restart on source changes. Development."),
    workers: int | None = typer.Option(
        None, "--workers", "-w", min=1, help="Worker processes. Incompatible with --reload."
    ),
) -> None:
    """Run the HTTP service.

    The only command that does not call ``asyncio.run``: uvicorn creates and owns
    the loop it serves the app in, and starting one here would leave it competing
    with uvicorn's.

    The app is passed as the import string ``ragorc.server.app:app`` rather than as
    an object, because both ``--reload`` and ``--workers`` need to *re-import* it in
    a fresh process. Handing uvicorn a live object silently disables both.

    Two warnings are worth having before this is exposed: with no ``api_keys`` the
    service is unauthenticated (logged at startup), and with more than one worker
    and no Redis the semantic cache hit rate divides by the worker count, because
    each process warms its own memory tier.
    """
    settings = _settings(
        RAGORC_SERVER__HOST=host,
        RAGORC_SERVER__PORT=port,
        RAGORC_SERVER__WORKERS=workers,
    )
    try:
        uvicorn = importlib.import_module("uvicorn")
    except ImportError:
        _fail(
            "missing dependency",
            "uvicorn is required to serve",
            "pip install 'ragorc[server]'",
            EXIT_CONFIG,
        )

    if reload and (settings.server.workers or 1) > 1:
        _fail(
            "conflicting options",
            "--reload and --workers cannot be combined",
            "reload runs a single process by design",
            EXIT_CONFIG,
        )
    if settings.server.workers > 1 and not settings.cache.redis_url:
        err.print(
            "[yellow]multiple workers without cache.redis_url: each process warms its own "
            "cache, so the effective semantic hit rate divides by the worker count[/yellow]"
        )
    if not settings.server.api_keys:
        err.print("[yellow]server.api_keys is empty: every endpoint is unauthenticated[/yellow]")

    err.print(
        f"serving [bold]http://{settings.server.host}:{settings.server.port}[/bold] "
        f"({settings.environment}, {settings.server.workers} worker(s))"
    )
    install_uvloop()
    uvicorn.run(
        "ragorc.server.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=reload,
        workers=None if reload else settings.server.workers,
        # The library configured structlog already; letting uvicorn install its
        # own dictConfig would replace the formatter and split the request log
        # across two formats.
        log_config=None,
        timeout_keep_alive=30,
    )


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------
@app.command(name="inspect")
def inspect_config(
    as_json: bool = typer.Option(False, "--json"),
    components: bool = typer.Option(
        True, "--components/--no-components", help="List registered components."
    ),
    stores: bool = typer.Option(True, "--stores/--no-stores", help="Probe the stores."),
) -> None:
    """Show the resolved configuration, the registered components and store stats.

    The settings block comes from :meth:`Settings.summary`, which redacts by
    construction: it reports whether an API key is present, never its value, and
    reduces each DSN to its host. That belongs to the producer rather than to this
    command, because the output of ``ragorc inspect`` is the thing people paste
    into tickets.

    Listing components requires *importing* the modules that register them —
    decorators run at class-definition time — so ``--no-components`` is the fast
    path when only the settings are wanted.
    """
    settings = _settings()

    async def run() -> None:
        payload: dict[str, Any] = {"settings": settings.summary()}
        if components:
            payload["components"] = _components()
        if stores:
            async with _service(settings) as service:
                health = await service.health()
            payload["health"] = health.model_dump(mode="json")

        if as_json:
            _print_json(payload)
            return

        console.print(Panel(_render(settings.summary()), title="settings", border_style="cyan"))
        if components:
            table = Table(title="registered components", box=None, pad_edge=False)
            table.add_column("kind")
            table.add_column("names", style="dim")
            for kind, names in payload["components"].items():
                table.add_row(kind, ", ".join(names))
            console.print(table)
        if stores:
            console.print(_store_table(payload["health"]))
            for warning in payload["health"]["warnings"]:
                err.print(f"[yellow]{_plain(warning)}[/yellow]")

    _run(run)


def _components() -> dict[str, list[str]]:
    """Populate and read the component registry.

    ``@register`` runs when a class is *defined*, so the registry only knows what
    has been imported. Loading the modules first is what makes this list the truth
    rather than "whatever happened to be imported already" — and it is why a
    mistyped component name in configuration should be caught at startup, not on
    the one request that used it.
    """
    from ragorc.core.registry import available
    from ragorc.retrieve import load_all

    load_all()
    for module in (
        "ragorc.index",
        "ragorc.translate",
        "ragorc.route",
        "ragorc.construct",
        "ragorc.generate",
        f"ragorc.embed.{get_settings().embedding.provider}_provider",
    ):
        with contextlib.suppress(ImportError):
            importlib.import_module(module)
    return available()


def _store_table(health: dict[str, Any]) -> Table:
    table = Table(title=f"stores · {health['status']}", box=None, pad_edge=False)
    for column in ("store", "status", "ms", "detail"):
        table.add_column(column, justify="right" if column == "ms" else "left")
    for store in health["stores"]:
        colour = "green" if store["status"] == "ok" else "red"
        table.add_row(
            store["name"],
            f"[{colour}]{store['status']}[/{colour}]",
            f"{store['latency_ms']:.0f}",
            store["error"] or ", ".join(f"{k}={v}" for k, v in store["detail"].items()),
        )
    return table


def _render(payload: Any, indent: int = 0) -> str:
    lines: list[str] = []
    pad = "  " * indent
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, dict):
                lines.append(f"{pad}[bold]{key}[/bold]")
                lines.append(_render(value, indent + 1))
            else:
                lines.append(f"{pad}{key} = {value}")
    else:
        lines.append(f"{pad}{payload}")
    return "\n".join(line for line in lines if line)


# ---------------------------------------------------------------------------
# alias-swap
# ---------------------------------------------------------------------------
@app.command(name="alias-swap")
def alias_swap(
    alias: str = typer.Argument(..., help="Alias the application reads through."),
    collection: str = typer.Argument(..., help="Collection the alias should point at."),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Wait for the new collection to finish indexing first."
    ),
    timeout: float = typer.Option(300.0, "--timeout", help="Seconds to wait for green."),
) -> None:
    """Atomically repoint an alias — the cutover step of a zero-downtime reindex.

        ragorc init --collection ragorc_v2
        ragorc ingest ./corpus --collection ragorc_v2
        ragorc alias-swap ragorc ragorc_v2

    Two things make this safe, and both are the reason not to do it by hand.

    The delete and the create are submitted as **one** alias operation batch, which
    Qdrant applies atomically, so readers never observe a window where the alias
    resolves to nothing.

    The wait comes *before* the swap. A collection still building its HNSW index
    answers queries — slowly, by brute force — so swapping first and indexing after
    is a latency incident rather than an outage, and therefore the kind that gets
    diagnosed as something else. ``--no-wait`` exists for a collection you already
    know is green.

    The previous target is printed and left in place: rollback is one more
    ``alias-swap`` back to it, not a re-ingest.
    """
    settings = _settings()

    async def run() -> None:
        from ragorc.stores.qdrant.client import build_client, close_all_clients
        from ragorc.stores.qdrant.collections import swap_alias, wait_for_green

        client = build_client(settings)
        try:
            if not await client.collection_exists(collection):
                _fail(
                    "no such collection",
                    f"{collection!r} does not exist in Qdrant",
                    "run `ragorc init --collection` and ingest into it first",
                    EXIT_CONFIG,
                )
            if wait:
                with err.status(f"[cyan]waiting for {collection} to report green"):
                    green = await wait_for_green(client, collection, timeout_s=timeout)
                if not green:
                    err.print(
                        f"[yellow]{collection} is still indexing after {timeout:.0f}s; "
                        "swapping anyway would serve brute-force searches[/yellow]"
                    )
                    raise typer.Exit(EXIT_ERROR)
            previous = await swap_alias(client, alias, collection)
            console.print(
                Panel(
                    f"alias [bold]{alias}[/bold] -> [green]{collection}[/green]\n"
                    f"previous: {previous or '[dim](none)[/dim]'}",
                    title="alias swapped",
                    subtitle=(
                        f"roll back with: ragorc alias-swap {alias} {previous}"
                        if previous
                        else "nothing to roll back to"
                    ),
                    border_style="cyan",
                )
            )
        finally:
            await close_all_clients()

    _run(run)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _print_json(payload: Any) -> None:
    """Write JSON to stdout, unstyled.

    ``print`` through the file object rather than ``console.print``: rich would
    wrap long lines and inject ANSI, and this output exists to be piped into
    something that parses it.
    """
    console.file.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode())
    console.file.write("\n")
    console.file.flush()


if __name__ == "__main__":  # pragma: no cover - exercised by `python -m ragorc.cli`
    # Without this, `python -m ragorc.cli query ...` imports the module, defines
    # the commands and exits 0 having done nothing — a silent no-op, which is the
    # worst possible failure for a CLI because it looks like success. The
    # `ragorc` console script from pyproject calls `app()` directly and is
    # unaffected, which is exactly why the gap went unnoticed.
    app()
