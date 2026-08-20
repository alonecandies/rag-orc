"""In-process BM25 over a list of chunks — the offline fallback.

Production BM25 in this library does not live here. It lives *inside Qdrant* as a
sparse vector with the ``Modifier.IDF`` modifier, so lexical and semantic search
are one query against one engine and the IDF term is computed from live corpus
statistics (see :mod:`ragorc.retrieve.sparse` and
``docs/adr/0003-server-side-fusion.md``). That is the implementation to use when
there is a server.

This class exists for the three situations where there is not one:

* **Offline mode.** A notebook, a CLI run over a directory, an air-gapped
  evaluation — no Qdrant, no ONNX model download, no network.
* **Tests.** A deterministic lexical retriever with zero external dependencies,
  which is what lets the fusion, reranking and pipeline test suites assert on
  exact rankings.
* **Small corpora.** Below a few thousand chunks, an in-memory scan is faster
  than a round trip, and the operational cost of a vector database is not worth
  paying for a hundred documents.

What it is and is not
---------------------
It is textbook Okapi BM25: ``sum over query terms of idf(t) * tf * (k1 + 1) /
(tf + k1 * (1 - b + b * |d| / avgdl))`` with ``idf(t) = ln(1 + (N - df + 0.5) /
(df + 0.5))``. The query-term-frequency factor of the full BM25 formula is
omitted deliberately: it only matters for long queries, and Robertson's own
recommendation is to drop it for the short queries a search system actually sees.

It is *not* equivalent to the Qdrant path. There is no stemming (that would mean
a tokenizer dependency), no stopword list beyond what IDF suppresses naturally,
and no term hashing. Results will differ from the server-side implementation, and
that difference is the reason this is a fallback rather than an alternative.

Why the shape of the index is what it is
----------------------------------------
Scoring must never loop over documents. A corpus of 50k chunks scored in Python
is tens of milliseconds *per query term*; the same arithmetic in numpy is
microseconds. So the corpus is compiled once into a CSR-style inverted index —
``postings_ptr`` (per-term slice bounds), ``postings_doc`` and ``postings_tf``
(flat parallel arrays) — and a query becomes: gather the postings of the query's
terms, evaluate the BM25 expression over the gathered array in one pass, and
scatter-add into a per-document score vector with ``np.bincount``.

The gather itself is vectorized too, with the standard ragged-range trick
(``repeat`` of each slice start plus a within-run offset), so *all* query terms
of *all* query variants are scored in a single arithmetic pass over a single
array. The only Python-level iteration in the whole class is tokenization at
build time, and that runs in a worker thread so it never stalls the event loop.

``np.bincount`` rather than ``np.add.at``: both scatter-add, but bincount is a
tight C loop while ``add.at`` goes through the much slower unbuffered ufunc path
— measurably several times faster on posting lists of any real size.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import structlog

from ragorc.core.errors import ValidationFailed
from ragorc.core.models import Chunk, Query, RetrievalSource, ScoredChunk
from ragorc.core.registry import register
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import timed
from ragorc.retrieve.fusion import reciprocal_rank_fusion
from ragorc.security.tenancy import scope_filter

log = structlog.get_logger(__name__)

__all__ = ["BM25_B", "BM25_K1", "InMemoryBM25Retriever"]

BM25_K1 = 1.5
"""Term-frequency saturation. Above ~1.5 a term appearing ten times counts
almost the same as five times, which is the behaviour that makes BM25 robust to
keyword stuffing and to long documents that repeat themselves."""

BM25_B = 0.75
"""Length normalization strength. 0 ignores document length entirely (long
documents win by accumulating matches), 1 divides fully by relative length
(short documents win). 0.75 is the value from the original TREC experiments and
has survived thirty years of re-tuning attempts."""

_TOKEN = re.compile(r"\w+")

_FILTER_ATTRS = ("id", "document_id", "parent_id", "level", "tenant_id")
"""Filter keys that map onto a :class:`Chunk` attribute. Everything else is
looked up in ``metadata``, mirroring the Qdrant filter dialect."""


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _chunk_value(chunk: Chunk, key: str) -> Any:
    """Resolve a filter key against a chunk the way the payload would.

    ``modality`` is compared as its string value because that is what the store
    payload carries and therefore what a filter written against the Qdrant
    dialect will contain.
    """
    if key == "modality":
        return chunk.modality.value
    if key in _FILTER_ATTRS:
        return getattr(chunk, key)
    return chunk.metadata.get(key)


def _matches(chunk: Chunk, key: str, spec: Any) -> bool:
    """One filter predicate against one chunk.

    Unknown operators raise rather than being ignored, for the same reason they
    raise in :mod:`ragorc.stores.qdrant.filters`: silently dropping a predicate
    returns *more* documents than the caller asked for, and in a filtered or
    multi-tenant corpus that is a data leak wearing the costume of a recall win.
    """
    value = _chunk_value(chunk, key)
    if isinstance(spec, dict):
        for op, operand in spec.items():
            if op in ("$eq", "eq"):
                if value != operand:
                    return False
            elif op in ("$ne", "ne"):
                if value == operand:
                    return False
            elif op in ("$in", "in"):
                if value not in operand:
                    return False
            elif op in ("$nin", "nin"):
                if value in operand:
                    return False
            else:
                raise ValidationFailed(
                    f"unsupported filter operator {op!r} for in-memory BM25",
                    field=key,
                    supported=["$eq", "$ne", "$in", "$nin"],
                )
        return True
    if isinstance(spec, list | tuple | set):
        return value in spec
    return value == spec


@register("retriever", "bm25", "lexical")
class InMemoryBM25Retriever:
    """Dependency-free BM25 over an in-memory chunk list."""

    name = "bm25"

    def __init__(
        self,
        chunks: Sequence[Chunk] | None = None,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        use_embed_text: bool = False,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.k1 = float(k1)
        self.b = float(b)
        self.use_embed_text = use_embed_text
        """Index ``Chunk.embed_text`` (contextual prefix included) rather than
        ``content``. Off by default: the prefix is an LLM-written summary, and
        indexing it makes lexical hits land on words the document never used."""

        self._chunks: list[Chunk] = list(chunks or ())
        self._vocab: dict[str, int] = {}
        self._postings_ptr: np.ndarray = np.zeros(1, dtype=np.int64)
        self._postings_doc: np.ndarray = np.zeros(0, dtype=np.int64)
        self._postings_tf: np.ndarray = np.zeros(0, dtype=np.float64)
        self._idf: np.ndarray = np.zeros(0, dtype=np.float64)
        self._doc_len: np.ndarray = np.zeros(0, dtype=np.float64)
        self._tenants: np.ndarray = np.zeros(0, dtype=object)
        self._avgdl = 1.0
        self._built = False
        # Created on first use: a lock constructed before the loop exists is fine
        # in 3.10+, but the store layer uses this pattern and consistency is worth
        # more than the two saved lines.
        self._lock: asyncio.Lock | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_lexical(self) -> bool:
        return True

    def add(self, chunks: Sequence[Chunk]) -> int:
        """Extend the corpus. Invalidates the index; the next query rebuilds it.

        Rebuilding wholesale rather than incrementally is the right trade here:
        every document frequency and the average document length change with an
        insert, so a correct incremental update touches the whole ``idf`` vector
        anyway, and this class is scoped to corpora where a full rebuild is
        milliseconds.
        """
        if not chunks:
            return len(self._chunks)
        self._chunks.extend(chunks)
        self._built = False
        return len(self._chunks)

    # -- index -------------------------------------------------------------
    async def ensure_index(self) -> None:
        """Build the index if needed, exactly once, off the event loop."""
        if self._built:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._built:
                # Lost the race: another coroutine built it while we waited.
                return
            with timed("bm25.build", documents=len(self._chunks)):
                await asyncio.to_thread(self._build)

    def _build(self) -> None:
        """Tokenize the corpus and compile the CSR inverted index.

        CPU-bound and Python-heavy (the regex and the ``Counter`` per document),
        which is exactly why the caller runs it in a thread: on a 50k-chunk corpus
        this is seconds, and seconds on the event loop is every concurrent request
        stalled.
        """
        chunks = self._chunks
        n_docs = len(chunks)
        vocab: dict[str, int] = {}
        doc_ids: list[int] = []
        term_ids: list[int] = []
        term_freqs: list[float] = []
        lengths = np.zeros(n_docs, dtype=np.float64)

        for di, chunk in enumerate(chunks):
            text = chunk.embed_text if self.use_embed_text else chunk.content
            counts = Counter(_tokenize(text))
            lengths[di] = float(sum(counts.values()))
            for term, freq in counts.items():
                ti = vocab.get(term)
                if ti is None:
                    ti = len(vocab)
                    vocab[term] = ti
                doc_ids.append(di)
                term_ids.append(ti)
                term_freqs.append(float(freq))

        n_terms = len(vocab)
        terms = np.asarray(term_ids, dtype=np.int64)
        docs = np.asarray(doc_ids, dtype=np.int64)
        freqs = np.asarray(term_freqs, dtype=np.float64)

        # Group postings by term so each term's postings are one contiguous slice.
        order = np.argsort(terms, kind="stable")
        # One posting per (document, term) pair, so a term's posting count *is* its
        # document frequency — no separate df pass.
        df = np.bincount(terms, minlength=n_terms).astype(np.float64) if n_terms else np.zeros(0)
        ptr = np.zeros(n_terms + 1, dtype=np.int64)
        if n_terms:
            np.cumsum(df.astype(np.int64), out=ptr[1:])

        self._vocab = vocab
        self._postings_ptr = ptr
        self._postings_doc = docs[order]
        self._postings_tf = freqs[order]
        # Lucene's smoothed IDF: always positive, so a term present in every
        # document contributes almost nothing instead of a negative score that
        # would *penalize* documents for containing a query term.
        self._idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)) if n_terms else np.zeros(0)
        self._doc_len = lengths
        self._tenants = np.asarray([c.tenant_id for c in chunks], dtype=object)
        self._avgdl = float(lengths.mean()) if n_docs and lengths.sum() else 1.0
        self._built = True
        log.debug(
            "bm25_index_built",
            documents=n_docs,
            terms=n_terms,
            postings=int(docs.size),
            avgdl=round(self._avgdl, 2),
        )

    # -- scoring -----------------------------------------------------------
    def _term_ids(self, text: str) -> np.ndarray:
        """Query terms as vocabulary ids, de-duplicated, unknown terms dropped.

        De-duplication is what drops the query-term-frequency factor: a term
        repeated in the query would otherwise score twice, which for a two-word
        query is a 2x weight nobody asked for.
        """
        seen: dict[str, None] = dict.fromkeys(_tokenize(text))
        ids = [self._vocab[term] for term in seen if term in self._vocab]
        return np.asarray(ids, dtype=np.int64)

    def _score_matrix(self, term_lists: Sequence[np.ndarray]) -> np.ndarray:
        """BM25 scores for every (variant, document) pair in one pass.

        Shape ``(n_variants, n_docs)``. The ragged gather is the interesting part:
        ``starts`` and ``counts`` describe one contiguous posting slice per query
        term, and ``repeat(starts, counts) + (arange(total) - repeat(offsets,
        counts))`` expands all of those slices into one flat index array without a
        Python loop. Everything downstream — the IDF lookup, the length
        normalization, the saturation — is then a single vectorized expression
        over that array, for all terms of all variants simultaneously.
        """
        n_docs = len(self._chunks)
        n_variants = len(term_lists)
        empty = np.zeros((n_variants, n_docs), dtype=np.float64)
        if not n_docs or not n_variants:
            return empty

        sizes = np.asarray([len(t) for t in term_lists], dtype=np.int64)
        if not sizes.sum():
            return empty
        tids = np.concatenate([t for t in term_lists if t.size])
        vids = np.repeat(np.arange(n_variants, dtype=np.int64), sizes)

        starts = self._postings_ptr[tids]
        counts = self._postings_ptr[tids + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return empty

        run_start = np.repeat(np.cumsum(counts) - counts, counts)
        gather = np.repeat(starts, counts) + (np.arange(total, dtype=np.int64) - run_start)

        docs = self._postings_doc[gather]
        tf = self._postings_tf[gather]
        idf = np.repeat(self._idf[tids], counts)
        variant = np.repeat(vids, counts)

        denom = tf + self.k1 * (1.0 - self.b + self.b * self._doc_len[docs] / self._avgdl)
        contrib = idf * tf * (self.k1 + 1.0) / denom

        # Flatten (variant, doc) into one axis so a single bincount does the
        # scatter-add for every variant at once.
        flat = np.bincount(variant * n_docs + docs, weights=contrib, minlength=n_variants * n_docs)
        return flat.reshape(n_variants, n_docs)

    def _mask(self, filters: dict[str, Any]) -> np.ndarray | None:
        """Boolean keep-mask for the filter, or ``None`` when nothing is filtered.

        The tenant predicate — the one that is always on in a multi-tenant
        deployment — is a vectorized comparison against a precomputed object
        array. The remaining keys go through a Python pass because their values
        are arbitrary objects; it only runs when such a filter is present, and
        this class is the small-corpus path by definition.
        """
        if not filters:
            return None
        mask: np.ndarray | None = None
        rest = dict(filters)
        tenant = rest.get("tenant_id")
        if tenant is not None and not isinstance(tenant, dict | list | tuple | set):
            mask = self._tenants == tenant
            del rest["tenant_id"]
        if rest:
            extra = np.fromiter(
                (all(_matches(c, k, v) for k, v in rest.items()) for c in self._chunks),
                dtype=bool,
                count=len(self._chunks),
            )
            mask = extra if mask is None else (mask & extra)
        return mask

    # -- retrieval ---------------------------------------------------------
    async def retrieve(
        self, query: Query, *, top_k: int | None = None, **kw: Any
    ) -> list[ScoredChunk]:
        """Lexical retrieval over the in-memory corpus.

        Keyword arguments: ``filters``, ``tenant_id``, ``use_variants``.

        Scores are raw BM25: unbounded, corpus-dependent, higher-is-better. They
        are *not* comparable with a cosine similarity, which is why the hybrid
        retriever fuses this leg by rank rather than by score.
        """
        k = int(top_k or query.top_k or self.settings.retrieval.top_k)
        tenant = kw.get("tenant_id") or query.tenant_id or self.settings.tenant_id
        filters = dict(query.filters)
        if kw.get("filters"):
            filters.update(kw["filters"])
        filters = scope_filter(filters, tenant, self.settings.security)

        await self.ensure_index()
        if not self._chunks:
            return []

        texts = list(query.all_texts) if kw.get("use_variants", True) else [query.text]
        term_lists = [self._term_ids(text) for text in texts]

        with timed("retrieve.bm25", documents=len(self._chunks), variants=len(texts)):
            scores = await asyncio.to_thread(self._score_matrix, term_lists)
        mask = self._mask(filters)

        per_variant: dict[str, list[ScoredChunk]] = {}
        for vi, row in enumerate(scores):
            hits = self._top(row, mask, k)
            if hits:
                per_variant["bm25" if vi == 0 else f"bm25_v{vi}"] = hits

        if not per_variant:
            return []
        if len(per_variant) == 1:
            # Single list: keep raw BM25 scores rather than replacing them with an
            # RRF value that carries no magnitude information.
            return next(iter(per_variant.values()))
        return reciprocal_rank_fusion(per_variant, self.settings.retrieval.rrf_k, top_k=k)

    def _top(self, scores: np.ndarray, mask: np.ndarray | None, k: int) -> list[ScoredChunk]:
        """Top-k by score, using ``argpartition`` rather than a full sort.

        Partition is O(n) against O(n log n), which is the difference that matters
        when n is the corpus rather than the candidate set: only the k winners then
        get sorted. Zero-scoring documents share no term with the query and are
        dropped before ranking so they cannot pad the result to k.
        """
        candidates = (
            np.flatnonzero(scores > 0.0) if mask is None else np.flatnonzero((scores > 0.0) & mask)
        )
        if candidates.size == 0:
            return []
        values = scores[candidates]
        if candidates.size > k:
            keep = np.argpartition(-values, k - 1)[:k]
            candidates, values = candidates[keep], values[keep]
        order = np.argsort(-values, kind="stable")
        out: list[ScoredChunk] = []
        for rank, position in enumerate(order.tolist()):
            di = int(candidates[position])
            score = float(values[position])
            out.append(
                ScoredChunk(
                    chunk=self._chunks[di],
                    score=score,
                    source=RetrievalSource.BM25,
                    rank=rank,
                    component_scores={"bm25": score},
                    explain={"store": "memory", "k1": self.k1, "b": self.b},
                )
            )
        return out
