"""Provider selection for the embedding layer.

Why a factory rather than direct construction
---------------------------------------------
Four *kinds* of model (dense, sparse, late-interaction, reranker) times five
*providers* is twenty constructors a caller would otherwise have to know about,
and the useful combinations are constrained in ways that are easy to get wrong:

* **Only FastEmbed offers all four kinds.** The hosted providers return pooled
  dense vectors and nothing else — asking OpenAI for a sparse vector is not a
  configuration mistake to be discovered at query time, it is a category error
  that should be named at build time.
* **Only a local, token-level model can support late chunking** (ADR-0002), so
  the choice of provider silently decides the chunking strategy the ingest
  pipeline is able to use. The factory makes that consequence visible in a log
  line instead of leaving it to be inferred.
* **The cache belongs to the provider, not the caller.** Every embedder shares one
  :class:`~ragorc.embed.cache.EmbeddingCache` over the same tiered backend, so a
  chunk embedded by the splitter is not re-embedded by the indexer. Wiring that
  per call site is how it gets forgotten in one of them.

So this module answers one question per kind — *given the configuration, which
concrete class, wired to what* — and answers it in exactly one place.

Fallback policy: a missing **optional** dependency for a *hosted* provider is a
hard error, because the operator explicitly asked for that provider and silently
using a different model would corrupt an index with mismatched vectors. A missing
model for an *auxiliary* capability (a reranker, late interaction) degrades to
``None`` with a warning, because retrieval without reranking is worse but still
correct, and refusing to start would be the greater harm.
"""

from __future__ import annotations

from typing import Any

import structlog

from ragorc.core.errors import ConfigError
from ragorc.core.protocols import (
    Cache,
    DenseEmbedder,
    LateInteractionEmbedder,
    Reranker,
    SparseEmbedder,
)
from ragorc.core.settings import Settings, get_settings
from ragorc.embed.cache import EmbeddingCache

log = structlog.get_logger(__name__)

__all__ = [
    "PROVIDERS",
    "build_dense_embedder",
    "build_embedding_cache",
    "build_late_chunking_embedder",
    "build_late_interaction_embedder",
    "build_reranker",
    "build_sparse_embedder",
    "supports_late_chunking",
]

#: What each provider can actually produce. Consulted before construction so an
#: impossible combination is reported as configuration, not as a runtime failure.
PROVIDERS: dict[str, frozenset[str]] = {
    "fastembed": frozenset({"dense", "sparse", "late", "rerank"}),
    "openai": frozenset({"dense"}),
    "voyage": frozenset({"dense"}),
    "cohere": frozenset({"dense", "rerank"}),
    "sentence_transformers": frozenset({"dense", "rerank"}),
}

#: Providers whose dense embeddings are computed locally and can therefore expose
#: per-token vectors — the precondition for late chunking (ADR-0002).
_LOCAL_PROVIDERS = frozenset({"fastembed", "sentence_transformers"})


def _resolved(settings: Settings | None) -> Settings:
    return settings or get_settings()


def _provider(settings: Settings) -> str:
    name = settings.embedding.provider
    if name not in PROVIDERS:
        raise ConfigError(
            f"unknown embedding provider {name!r}",
            known=sorted(PROVIDERS),
        )
    return name


def _require(provider: str, capability: str) -> None:
    if capability not in PROVIDERS[provider]:
        raise ConfigError(
            f"the {provider!r} embedding provider cannot produce {capability!r} vectors",
            provider=provider,
            capability=capability,
            supports=sorted(PROVIDERS[provider]),
            hint=(
                "set embedding.provider='fastembed' for sparse and late-interaction "
                "vectors, or disable the feature that needs them"
            ),
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def build_embedding_cache(
    backend: Cache | None = None, settings: Settings | None = None
) -> EmbeddingCache | None:
    """Wrap a cache backend for embedding use, or return ``None`` when disabled.

    Returns ``None`` rather than a no-op cache so the embedders can skip the
    key-derivation work entirely: hashing every input to look it up in a cache
    that never hits is pure overhead on a large ingest.
    """
    resolved = _resolved(settings)
    if not (resolved.cache.enabled and resolved.embedding.cache_embeddings):
        return None
    if backend is None:
        from ragorc.cache.tiered import build_cache

        backend = build_cache(resolved.cache)
    return EmbeddingCache(backend, resolved)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------
def build_dense_embedder(
    settings: Settings | None = None,
    *,
    cache: EmbeddingCache | None = None,
    cache_backend: Cache | None = None,
    model_name: str | None = None,
) -> DenseEmbedder:
    """Construct the configured dense embedder.

    This is the one embedder every configuration needs, so an import failure here
    is fatal rather than degraded: without dense vectors there is no vector
    retrieval to degrade *to*.
    """
    resolved = _resolved(settings)
    provider = _provider(resolved)
    _require(provider, "dense")
    embedding_cache = cache if cache is not None else build_embedding_cache(cache_backend, resolved)

    if provider == "fastembed":
        from ragorc.embed.fastembed_provider import FastEmbedDense

        embedder: DenseEmbedder = FastEmbedDense(
            model_name, cache=embedding_cache, settings=resolved
        )
    elif provider == "openai":
        from ragorc.embed.openai_provider import OpenAIEmbedder

        embedder = OpenAIEmbedder(model_name, cache=embedding_cache, settings=resolved)
    elif provider == "voyage":
        from ragorc.embed.voyage_provider import VoyageEmbedder

        embedder = VoyageEmbedder(model_name, cache=embedding_cache, settings=resolved)
    elif provider == "cohere":
        from ragorc.embed.cohere_provider import CohereEmbedder

        embedder = CohereEmbedder(model_name, cache=embedding_cache, settings=resolved)
    else:
        from ragorc.embed.sentence_transformers_provider import STEmbedder

        embedder = STEmbedder(model_name, cache=embedding_cache, settings=resolved)

    log.info(
        "dense_embedder_built",
        provider=provider,
        model=getattr(embedder, "model_name", model_name),
        dimension=getattr(embedder, "dimension", None),
        cached=embedding_cache is not None,
        # Whether the *provider* could ever support late chunking. Deliberately
        # not the same question as `supports_late_chunking()`, which additionally
        # checks that this model exposes token vectors — that check builds an
        # embedder, and doing it here would recurse.
        provider_is_local=provider in _LOCAL_PROVIDERS,
    )
    return embedder


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------
def build_sparse_embedder(
    settings: Settings | None = None,
    *,
    cache: EmbeddingCache | None = None,
    cache_backend: Cache | None = None,
    model_name: str | None = None,
) -> SparseEmbedder | None:
    """Construct the sparse embedder, or ``None`` when hybrid search is off.

    Returns ``None`` rather than raising when ``retrieval.use_sparse`` is false, so
    a dense-only deployment needs no special-casing at the call site.
    """
    resolved = _resolved(settings)
    if not resolved.retrieval.use_sparse:
        return None
    provider = _provider(resolved)
    _require(provider, "sparse")

    from ragorc.embed.fastembed_provider import FastEmbedSparse

    config = resolved.embedding
    chosen = model_name or (config.splade_model if config.use_splade else config.sparse_model)
    embedder = FastEmbedSparse(
        chosen,
        cache=cache if cache is not None else build_embedding_cache(cache_backend, resolved),
        settings=resolved,
    )
    log.info(
        "sparse_embedder_built",
        model=getattr(embedder, "model_name", chosen),
        # Lexical models need Qdrant's IDF modifier on the sparse vector field, so
        # the collection config depends on this flag being right.
        lexical=getattr(embedder, "is_lexical", None),
    )
    return embedder


# ---------------------------------------------------------------------------
# Late interaction (ColBERT)
# ---------------------------------------------------------------------------
def build_late_interaction_embedder(
    settings: Settings | None = None,
    *,
    cache: EmbeddingCache | None = None,
    cache_backend: Cache | None = None,
    model_name: str | None = None,
    required: bool = False,
) -> LateInteractionEmbedder | None:
    """Construct the ColBERT embedder, or ``None`` when it is not enabled.

    ``required=True`` turns an unavailable model into an error. Use it when the
    Qdrant collection already has a multivector field: writing chunks without
    ColBERT vectors into a collection whose reranking stage expects them produces
    silently worse results rather than a visible failure.
    """
    resolved = _resolved(settings)
    config = resolved.embedding
    if not (config.enable_late_interaction or resolved.retrieval.colbert_rerank or required):
        return None
    provider = _provider(resolved)
    if "late" not in PROVIDERS[provider]:
        if required:
            _require(provider, "late")
        log.warning(
            "late_interaction_unavailable",
            provider=provider,
            effect="ColBERT reranking disabled",
            hint="set embedding.provider='fastembed' to enable it",
        )
        return None

    from ragorc.embed.fastembed_provider import FastEmbedLateInteraction

    embedder = FastEmbedLateInteraction(
        model_name or config.late_interaction_model,
        cache=cache if cache is not None else build_embedding_cache(cache_backend, resolved),
        settings=resolved,
    )
    log.info(
        "late_interaction_embedder_built",
        model=getattr(embedder, "model_name", None),
        dimension=getattr(embedder, "dimension", None),
    )
    return embedder


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
def build_reranker(
    settings: Settings | None = None,
    *,
    model_name: str | None = None,
    required: bool = False,
) -> Reranker | None:
    """Construct the cross-encoder reranker, or ``None`` when unavailable.

    Degrades to ``None`` rather than raising: retrieval without reranking is less
    precise but still correct, so a missing reranker model should not stop a
    service from starting. ``required=True`` overrides that for callers that would
    rather fail loudly.
    """
    resolved = _resolved(settings)
    if not resolved.retrieval.rerank_enabled and not required:
        return None
    provider = _provider(resolved)

    try:
        if provider == "cohere":
            from ragorc.embed.cohere_provider import CohereReranker

            reranker: Reranker = CohereReranker(model_name, settings=resolved)
        elif provider == "sentence_transformers":
            from ragorc.embed.sentence_transformers_provider import STCrossEncoderReranker

            reranker = STCrossEncoderReranker(model_name, settings=resolved)
        else:
            # FastEmbed's ONNX cross-encoder is the default for every provider that
            # has no reranker of its own: it needs no API key and no torch, so a
            # hosted dense provider still gets local reranking.
            from ragorc.embed.fastembed_provider import FastEmbedReranker

            reranker = FastEmbedReranker(model_name, settings=resolved)
    except ImportError as exc:
        if required:
            raise
        log.warning(
            "reranker_unavailable",
            provider=provider,
            error=str(exc)[:160],
            effect="results are returned in first-stage order",
        )
        return None

    log.info("reranker_built", provider=provider, model=getattr(reranker, "model_name", None))
    return reranker


# ---------------------------------------------------------------------------
# Late chunking
# ---------------------------------------------------------------------------
def supports_late_chunking(settings: Settings | None = None) -> bool:
    """Whether this configuration can actually use late chunking.

    Late chunking needs **one model** that emits both the per-token vectors and
    the pooled document vector, because the pooled result has to land in the same
    space as the query embedding (Günther et al., 2024). Two conditions follow:

    * the dense model must expose token-level output, which in practice means the
      transformers backend under ``ragorc[local]`` — FastEmbed's ``TextEmbedding``
      returns pooled vectors only; and
    * it must be the *same* model that embeds queries.

    An earlier version returned ``True`` for any "local" provider, on the theory
    that a local model can always be opened up. That was wrong in the way that
    matters: it reported capability the resolver then declined, so a caller
    printing "late chunking: enabled" alongside a corpus indexed EARLY had no way
    to notice. This now agrees with
    :func:`ragorc.embed.late_chunking.resolve_strategy`, which remains the
    authority.
    """
    resolved = _resolved(settings)
    if _provider(resolved) not in _LOCAL_PROVIDERS:
        return False
    # The probe is a package-presence check, not a model load, so it is cheap
    # enough to call before an ingest reports its plan.
    from ragorc.embed.late_chunking import LateChunkingEmbedder

    probe = LateChunkingEmbedder(
        build_dense_embedder(resolved),
        tokenizer_name=resolved.embedding.dense_model,
        settings=resolved,
    )
    return bool(probe.supports_token_embeddings)


def build_late_chunking_embedder(
    settings: Settings | None = None,
    *,
    token_embedder: Any | None = None,
    cache: EmbeddingCache | None = None,
    cache_backend: Cache | None = None,
) -> Any:
    """Construct the late-chunking embedder.

    The token source defaults to the **dense embedder itself**, and that is not an
    incidental choice — it is what late chunking means. Pooling token vectors only
    produces a usable chunk vector if the result lands in the same space as the
    *query* vector, and queries are embedded by the dense model. The paper
    (Günther et al., 2024) uses one model for both: the same weights emit the
    token vectors and the pooled document vector.

    An earlier version of this function substituted the ColBERT late-interaction
    model when the dense model could not emit tokens, on the reasoning that
    ColBERT already returns one vector per token. That is true and irrelevant:
    ColBERT is a different model in a different space at a different width (128
    vs 384), so the pooled vectors were not comparable to anything the retriever
    would ever ask with. Qdrant rejected them outright on dimension — which was
    lucky, because at equal width it would have accepted them and returned
    quietly meaningless neighbours instead.

    So when the dense model cannot expose token embeddings, late chunking is
    genuinely unavailable and ``resolve_strategy`` degrades to CONTEXTUAL or
    EARLY. Install ``ragorc[local]`` and use a model that exposes token output
    (``jinaai/jina-embeddings-v2-base-en`` is the reference choice) to get it.
    """
    resolved = _resolved(settings)
    from ragorc.embed.late_chunking import LateChunkingEmbedder

    source = token_embedder
    if source is None:
        # The dense embedder, not a substitute. LateChunkingEmbedder probes it for
        # token-level output and reports `supports_token_embeddings` accordingly.
        source = build_dense_embedder(resolved, cache=cache, cache_backend=cache_backend)
    return LateChunkingEmbedder(
        source, tokenizer_name=resolved.embedding.dense_model, settings=resolved
    )
