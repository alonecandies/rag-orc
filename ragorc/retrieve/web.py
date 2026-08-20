"""The web fallback for corrective retrieval.

Why this module exists at all
----------------------------
CRAG's INCORRECT branch concludes that the corpus does not contain the answer.
That conclusion is only *actionable* if there is somewhere else to look, so the
web search here is not a feature bolted onto the retriever — it is the second
half of the CRAG decision. Without it the pipeline can detect that retrieval
failed and then do nothing about it.

Two backends sit behind one interface because the choice is a deployment
decision, not a code decision. Tavily is a search API built for LLM
consumption: it returns a cleaned snippet plus a per-result relevance score, and
it bills per call. DDGS scrapes public engines: free, no scores, ordered results
only, and rate-limited by whoever it is scraping. Provider ``"none"`` yields
:class:`NullWebRetriever`, which retrieves nothing *successfully* — CRAG then
degrades to "use what the corpus gave us" rather than raising. That is why the
null object exists instead of an ``if self.web is not None`` at every call site:
the absence of a web provider is a configuration, not an error condition.

Why every single result is scanned for injection
-----------------------------------------------
All retrieved content is untrusted (see :mod:`ragorc.security.injection`). Web
content is *adversarially* untrusted, and the difference is worth spelling out
because it changes what the defence has to cover:

* A corpus document is there because someone in your organization decided to
  index it. A web result is there because a page ranked for a query — and the
  query that reaches this module is CRAG's *rewrite* of a question the corpus
  could not answer, i.e. a long-tail phrasing that is cheap to rank for.
  "Publish a page that wins an obscure query, put instructions in it" is a
  practical attack, not a theoretical one, and the fallback path is precisely
  where it lands.
* The text is a snippet the engine chose out of a page nobody reviewed, and it
  arrives in the generator prompt with exactly the same authority as your own
  documents.
* Titles are attacker-controlled too, and titles do **not** travel inside the
  isolation wrapper: the context packer renders ``[n] | source: <title/url>`` on
  the passage header line, outside ``<untrusted_document>``. A title containing a
  newline could therefore forge a passage boundary and smuggle text into the
  trusted part of the prompt. Titles are whitespace-flattened here for that
  reason, not for tidiness.
* A URL from a result can be rendered as a link in the final answer, which makes
  it an exfiltration channel. Anything that is not ``http(s)`` is dropped rather
  than sanitized — there is no legitimate ``javascript:`` search result.

So each result is normalized, scanned, and sanitized according to
``security.injection_action``; if the configured action is ``block``, the
offending result is *dropped* and the query continues. One hostile page must not
fail a request that four other results could answer, and raising here would hand
an attacker a denial-of-service primitive: publish a page matching the pattern
list, kill every query that retrieves it.

Why the backends run in a worker thread
---------------------------------------
Both libraries are synchronous, and ``tavily-python`` uses ``requests``. Calling
either from the event loop stalls every other in-flight retrieval for the length
of an HTTP round trip to a third party. They therefore run through
``asyncio.to_thread`` — including their imports, which touch the filesystem on
first use — with the library's *own* timeout set alongside ours.  Both deadlines
are needed: ``asyncio.wait_for`` can only stop us *waiting*, it cannot cancel a
thread, so without the library-level timeout a hung request would occupy an
executor thread until the process exits.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from ragorc.core.errors import ConfigError, GuardrailViolation, StoreUnavailable
from ragorc.core.ids import chunk_id, document_id
from ragorc.core.models import Chunk, Modality, Query, RetrievalSource, ScoredChunk
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.security.injection import InjectionScanner

log = structlog.get_logger(__name__)

__all__ = [
    "NullWebRetriever",
    "WebResult",
    "WebSearchRetriever",
    "make_web_retriever",
]

_EXTRA_HINT = "pip install 'ragorc[web]'"

_TIMEOUT_SLACK = 1.5
"""How much longer our deadline is than the library's own.

The library's timeout is the real bound — only it can abort its socket and free
the worker thread. Ours is the backstop for a backend that ignores the parameter,
so it has to be *later*: equal deadlines race, and losing that race throws away a
response that arrived in time. Multiplicative rather than a fixed grace period so
the backstop stays proportionate whether the per-store budget is 10 seconds or
100 milliseconds.
"""

_MIN_RANK_SCORE = 0.5
"""Floor for rank-derived scores.

DDGS returns an ordered list and no scores, so a score has to be synthesized.
Only the *order* it encodes is real, and the range is kept narrow and inside
[0, 1] deliberately: these numbers sit next to cosine similarities once CRAG
merges web and corpus results, and a synthetic score that decayed to 0.05 would
claim a confidence about the fifth web hit that nobody measured.
"""


@dataclass(slots=True)
class WebResult:
    """One search hit, before it becomes a :class:`Chunk`."""

    title: str
    url: str
    content: str
    score: float | None = None
    """Provider-supplied relevance, higher-is-better, where the provider has one
    (Tavily). ``None`` means rank order is the only signal the backend gave us."""


def _require(module: str) -> None:
    """Fail at construction time rather than mid-query when an extra is missing.

    ``find_spec`` resolves the module without importing it, so this costs a
    filesystem stat and does not pull a search library into a process that will
    never search.
    """
    if importlib.util.find_spec(module) is None:
        raise ImportError(
            f"{module} is required for the '{module}' web search provider: {_EXTRA_HINT}"
        )


def _flatten(text: str) -> str:
    """Collapse all whitespace to single spaces.

    Applied to titles because the title is rendered on the passage *header* line,
    outside the untrusted-content wrapper. A newline there would let a title
    forge a second passage header.
    """
    return " ".join(text.split())


def _safe_url(raw: str) -> str:
    """Keep http(s) only. Returns ``""`` for anything else, which drops the result."""
    url = _flatten(raw)
    return url if url.lower().startswith(("http://", "https://")) else ""


# ---------------------------------------------------------------------------
# Backends — synchronous by nature, always called through a thread
# ---------------------------------------------------------------------------
class _TavilyBackend:
    """Tavily. Paid, scored, snippet-cleaned."""

    name = "tavily"

    def __init__(self, api_key: str) -> None:
        _require("tavily")
        if not api_key:
            raise ConfigError(
                "TAVILY_API_KEY is required for web_search_provider='tavily'",
                hint="export TAVILY_API_KEY=... or pass api_key=",
            )
        self._api_key = api_key

    def search(self, query: str, *, max_results: int, timeout_s: float) -> list[WebResult]:
        """Blocking call. Runs inside :func:`asyncio.to_thread`.

        A fresh client per call on purpose: ``TavilyClient`` wraps a
        ``requests.Session``, which is not safe to share across threads, and the
        web fallback fires at most once per query so a pooled connection would be
        cold anyway. ``search_depth="basic"`` is one Tavily credit against
        ``advanced``'s two, and the extra depth buys page *content* we do not use
        — CRAG re-grades the snippet either way.
        """
        # Imported here, inside the thread: the first import walks the
        # filesystem, which is blocking I/O the event loop should never do.
        from tavily import TavilyClient

        payload = TavilyClient(api_key=self._api_key).search(
            query,
            max_results=max_results,
            search_depth="basic",
            timeout=timeout_s,
        )
        rows = payload.get("results") or []
        out: list[WebResult] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            score = row.get("score")
            out.append(
                WebResult(
                    title=str(row.get("title") or ""),
                    url=str(row.get("url") or ""),
                    content=str(row.get("content") or row.get("raw_content") or ""),
                    # Tavily's score is already a similarity in [0, 1]: higher is
                    # better, no distance conversion needed.
                    score=float(score) if isinstance(score, int | float) else None,
                )
            )
        return out


class _DDGSBackend:
    """DDGS. Free, unscored, scraped — and rate-limited by third parties."""

    name = "ddgs"

    def __init__(self) -> None:
        _require("ddgs")

    def search(self, query: str, *, max_results: int, timeout_s: float) -> list[WebResult]:
        """Blocking call. Runs inside :func:`asyncio.to_thread`.

        ``timeout`` is per-engine and integral in ddgs, so it is floored at one
        second: passing 0 would disable the library's own deadline and leave the
        executor thread unbounded.
        """
        from ddgs import DDGS  # imported in-thread: see _TavilyBackend.search

        with DDGS(timeout=max(int(timeout_s), 1)) as client:
            rows = client.text(query, max_results=max_results)
        out: list[WebResult] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            out.append(
                WebResult(
                    title=str(row.get("title") or ""),
                    # ddgs names the link "href"; keep "url" as a fallback so a
                    # future key rename degrades to a dropped result, not a crash.
                    url=str(row.get("href") or row.get("url") or ""),
                    content=str(row.get("body") or ""),
                    score=None,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------
@register("retriever", "web")
class WebSearchRetriever:
    """Search the open web and return sanitized, scored chunks."""

    name = "web"
    enabled = True
    """Lets CRAG skip the query-rewrite call when the fallback cannot do anything
    with the result. See :class:`NullWebRetriever`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or self.settings.retrieval.web_search_provider
        self.scanner = InjectionScanner(self.settings.security)
        self._backend = self._make_backend(api_key)

    def _make_backend(self, api_key: str | None) -> _TavilyBackend | _DDGSBackend:
        if self.provider == "tavily":
            # The key is not in Settings because it belongs to the search vendor,
            # not to this library's configuration surface.
            return _TavilyBackend(api_key or os.environ.get("TAVILY_API_KEY", ""))
        if self.provider == "ddgs":
            return _DDGSBackend()
        raise ConfigError(
            f"unsupported web search provider {self.provider!r}",
            supported=["tavily", "ddgs", "none"],
            hint="provider 'none' is served by NullWebRetriever via make_web_retriever()",
        )

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        """Search for ``query.text`` and return at most ``top_k`` chunks.

        Raises :class:`StoreUnavailable` on any backend failure, which is what
        makes the web a *degradable* source: the caller records the error and
        answers from whatever else it has.
        """
        # ``top_k=0`` is honoured rather than folded into the default by an
        # ``or``: a caller asking for no web results is asking for no search, and
        # billing them for one would be the wrong reading of an explicit zero.
        limit = int(self.settings.retrieval.web_search_results if top_k is None else top_k)
        if limit <= 0 or not query.text.strip():
            return []
        results = await self._search(query.text, limit)
        return self._to_chunks(results, tenant_id=query.tenant_id)

    async def _search(self, text: str, limit: int) -> list[WebResult]:
        timeout = self.settings.retrieval.per_store_timeout_s
        try:
            rows = await asyncio.wait_for(
                asyncio.to_thread(self._backend.search, text, max_results=limit, timeout_s=timeout),
                timeout=timeout * _TIMEOUT_SLACK,
            )
        except TimeoutError as exc:
            # The thread is *not* cancelled by this — it runs to completion and
            # its result is discarded. The library-level timeout above is what
            # actually bounds the thread's lifetime.
            raise StoreUnavailable(
                "web", f"{self._backend.name} search timed out", timeout_s=timeout
            ) from exc
        except Exception as exc:
            # Each backend raises its own exception family (and ddgs surfaces
            # HTTP 202/429 rate limiting as an exception). Catching broadly and
            # translating keeps the vendor hierarchies out of the call sites.
            raise StoreUnavailable(
                "web", f"{self._backend.name} search failed: {exc}", provider=self._backend.name
            ) from exc

        log.debug("web_search", provider=self._backend.name, requested=limit, returned=len(rows))
        return rows[:limit]

    def _to_chunks(
        self, results: Sequence[WebResult], *, tenant_id: str | None
    ) -> list[ScoredChunk]:
        """Scan, sanitize and score. Order is preserved; dropped results close the gap."""
        kept: list[WebResult] = []
        blocked = 0
        for result in results:
            url = _safe_url(result.url)
            if not url:
                # Web evidence without a resolvable source is not citable, and an
                # uncitable passage is worse than one fewer passage.
                log.debug("web_result_dropped", reason="unusable_url", raw=result.url[:120])
                continue
            body = self._scan(result.content, url)
            if body is None:
                blocked += 1
                continue
            if not body.strip():
                continue
            # A blocked *title* costs only the title: the snippet is the evidence,
            # and the URL still identifies the source for a citation.
            title = _flatten(self._scan(result.title, url) or "")
            kept.append(WebResult(title=title, url=url, content=body, score=result.score))

        if blocked:
            log.warning("web_results_blocked", blocked=blocked, kept=len(kept))
        if not kept:
            return []

        scores = self._scores(kept)
        out: list[ScoredChunk] = []
        for rank, (result, score) in enumerate(zip(kept, scores, strict=True)):
            doc_id = document_id(result.url, tenant_id=tenant_id)
            # index=0, not the rank: one chunk per page, so the id is a function
            # of (url, snippet) alone. The same page found again by a differently
            # ranked query then collapses in the dedupe pass instead of arriving
            # as a second, identical passage.
            chunk = Chunk(
                id=chunk_id(doc_id, 0, result.content),
                content=result.content,
                document_id=doc_id,
                index=0,
                end_char=len(result.content),
                modality=Modality.TEXT,
                metadata={
                    "source": result.url or result.title,
                    "url": result.url,
                    "title": result.title,
                    "provider": self._backend.name,
                    "web": True,
                },
                tenant_id=tenant_id,
            )
            out.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(score),
                    source=RetrievalSource.WEB,
                    rank=rank,
                    component_scores={"web": float(score)},
                    explain={"provider": self._backend.name, "url": result.url},
                )
            )
        return out

    def _scan(self, text: str, url: str) -> str | None:
        """Run one field through the injection scanner.

        ``None`` means "drop this result": the configured action is ``block`` and
        the scanner raised. Dropping is deliberate — see the module docstring on
        why raising would be a denial-of-service primitive.
        """
        if not text:
            return ""
        try:
            scan = self.scanner.scan(text, source=f"web:{url}")
        except GuardrailViolation as exc:
            log.warning("web_result_blocked", url=url[:200], detail=str(exc)[:200])
            return None
        return scan.clean_text

    @staticmethod
    def _scores(results: Sequence[WebResult]) -> np.ndarray:
        """One score per result, higher-is-better, in [0, 1].

        Provider scores are used when *every* result has one; a partially scored
        list is scored by rank instead, because mixing a measured relevance with
        a synthesized one puts two different scales in the same ranking.
        """
        provided = [r.score for r in results]
        if provided and all(s is not None for s in provided):
            return np.clip(np.asarray(provided, dtype=np.float64), 0.0, 1.0)
        return np.linspace(1.0, _MIN_RANK_SCORE, num=len(results), dtype=np.float64)


@register("retriever", "web_none")
class NullWebRetriever:
    """The web provider that succeeds at retrieving nothing.

    ``web_search_provider="none"`` has to produce *something* that satisfies
    :class:`~ragorc.core.protocols.Retriever`, otherwise every consumer needs a
    None-check and CRAG's INCORRECT branch has two shapes instead of one. It
    reports ``name = "web"`` so telemetry shows a web step that returned no
    results, rather than a step that mysteriously does not exist, and
    ``enabled = False`` so CRAG can skip the query-rewrite LLM call whose only
    consumer would be this no-op.
    """

    name = "web"
    enabled = False

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kwargs: Any
    ) -> list[ScoredChunk]:
        return []


def make_web_retriever(
    settings: Settings | None = None, *, provider: str | None = None
) -> WebSearchRetriever | NullWebRetriever:
    """Build the configured web retriever, or the null one when disabled.

    Resolution happens here rather than in :class:`WebSearchRetriever` so that
    "no web search" costs nothing at import time: a deployment with
    ``web_search_provider="none"`` never needs the ``[web]`` extra installed.
    """
    settings = settings or get_settings()
    chosen = provider or settings.retrieval.web_search_provider
    if chosen == "none":
        return NullWebRetriever(settings)
    return WebSearchRetriever(settings, provider=chosen)
