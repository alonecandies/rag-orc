"""Embedding provider selection.

The factory's job is to turn a configuration into the right concrete class *and*
to name the impossible combinations at build time rather than at query time. The
tests that matter here are the negative ones: asking a hosted provider for a
sparse vector must fail with an explanation, not succeed and quietly produce a
dense-only index that hybrid search cannot use.

No model is loaded anywhere in this file — construction is asserted, inference is
not, so the suite stays offline and fast.
"""

from __future__ import annotations

import pytest

from ragorc.core.errors import ConfigError
from ragorc.core.settings import Settings
from ragorc.embed import (
    PROVIDERS,
    build_dense_embedder,
    build_embedding_cache,
    build_late_interaction_embedder,
    build_reranker,
    build_sparse_embedder,
    supports_late_chunking,
)


def settings_for(provider: str, **overrides) -> Settings:
    embedding = {"provider": provider, **overrides.pop("embedding", {})}
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        embedding=embedding,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Capability table
# ---------------------------------------------------------------------------
def test_only_fastembed_covers_every_kind() -> None:
    """The reason FastEmbed is the default: it is the only provider that can
    produce all four representations hybrid search and ColBERT need."""
    assert PROVIDERS["fastembed"] == {"dense", "sparse", "late", "rerank"}
    for hosted in ("openai", "voyage"):
        assert PROVIDERS[hosted] == {"dense"}


def test_settings_reject_an_unknown_provider() -> None:
    """First line of defence: the ``Literal`` on the field means an unknown
    provider name cannot be constructed at all."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        settings_for("pinecone-magic")


def test_factory_rejects_an_unknown_provider() -> None:
    """Second line of defence, for a Settings object built by other means (a
    fixture, a deserialized snapshot, a monkeypatched field): the factory names
    the problem instead of raising a KeyError from a dict lookup."""
    from ragorc.embed.factory import _provider

    settings = settings_for("fastembed")
    object.__setattr__(settings.embedding, "provider", "pinecone-magic")
    with pytest.raises(ConfigError, match="unknown embedding provider"):
        _provider(settings)


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------
def test_dense_embedder_is_built_for_the_default_provider() -> None:
    embedder = build_dense_embedder(settings_for("fastembed"))
    assert embedder.dimension == 384
    assert "bge-small" in embedder.model_name


def test_dense_model_name_override() -> None:
    embedder = build_dense_embedder(settings_for("fastembed"), model_name="BAAI/bge-base-en-v1.5")
    assert embedder.model_name == "BAAI/bge-base-en-v1.5"


# ---------------------------------------------------------------------------
# Sparse — the negative case is the important one
# ---------------------------------------------------------------------------
def test_sparse_from_a_hosted_provider_is_a_configuration_error() -> None:
    """A hosted provider returns pooled dense vectors only. Discovering that at
    query time would mean an index already written without sparse vectors."""
    with pytest.raises(ConfigError) as exc:
        build_sparse_embedder(settings_for("openai"))
    assert "sparse" in str(exc.value)
    assert "fastembed" in str(exc.value.detail.get("hint", ""))


def test_sparse_returns_none_when_hybrid_is_disabled() -> None:
    """Dense-only deployments need no special-casing at the call site."""
    settings = settings_for("fastembed", retrieval={"use_sparse": False})
    assert build_sparse_embedder(settings) is None


def test_sparse_defaults_to_bm25_and_is_lexical() -> None:
    """Lexical models require Qdrant's IDF modifier on the sparse field, so the
    collection configuration depends on this flag."""
    embedder = build_sparse_embedder(settings_for("fastembed"))
    assert embedder is not None
    assert embedder.is_lexical is True
    assert "bm25" in embedder.model_name.lower()


def test_splade_is_selected_when_requested() -> None:
    settings = settings_for("fastembed", embedding={"use_splade": True})
    embedder = build_sparse_embedder(settings)
    assert embedder is not None
    assert "splade" in embedder.model_name.lower()
    assert embedder.is_lexical is False, "a learned sparse model is not lexical"


# ---------------------------------------------------------------------------
# Late interaction
# ---------------------------------------------------------------------------
def test_late_interaction_is_off_by_default() -> None:
    """ColBERT multivectors are ~100x a dense vector, so they are opt-in."""
    assert build_late_interaction_embedder(settings_for("fastembed")) is None


def test_late_interaction_is_built_when_enabled() -> None:
    settings = settings_for("fastembed", embedding={"enable_late_interaction": True})
    embedder = build_late_interaction_embedder(settings)
    assert embedder is not None
    assert "colbert" in embedder.model_name.lower()


def test_late_interaction_degrades_on_a_hosted_provider() -> None:
    """Unavailable, not fatal: reranking is an enhancement. But `required=True`
    must raise, because writing chunks without ColBERT vectors into a collection
    whose rerank stage expects them degrades silently."""
    settings = settings_for("openai", embedding={"enable_late_interaction": True})
    assert build_late_interaction_embedder(settings) is None
    with pytest.raises(ConfigError):
        build_late_interaction_embedder(settings, required=True)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
def test_reranker_is_built_by_default() -> None:
    reranker = build_reranker(settings_for("fastembed"))
    assert reranker is not None
    assert "ms-marco" in reranker.model_name


def test_reranker_returns_none_when_disabled() -> None:
    settings = settings_for("fastembed", retrieval={"rerank_enabled": False})
    assert build_reranker(settings) is None


def test_hosted_dense_provider_still_gets_local_reranking() -> None:
    """FastEmbed's ONNX cross-encoder needs no API key and no torch, so a hosted
    dense provider is not left without a reranker."""
    reranker = build_reranker(settings_for("openai"))
    assert reranker is not None
    assert "ms-marco" in reranker.model_name


# ---------------------------------------------------------------------------
# Late chunking capability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "provider", ["fastembed", "openai", "voyage", "cohere", "sentence_transformers"]
)
def test_late_chunking_capability_is_never_claimed_without_token_output(provider: str) -> None:
    """Capability is a property of the *model*, not of the provider being local.

    This test previously asserted `fastembed -> True` on the theory that a local
    provider can always be opened up for token output. It cannot: FastEmbed's
    ``TextEmbedding`` returns pooled vectors only, so the claim was false for the
    default configuration — the one almost everybody runs.

    What must hold for every provider is weaker and actually true: a claim of
    support implies the dense model exposes token vectors, and a hosted provider
    can never claim it.
    """
    settings = settings_for(provider)
    claimed = supports_late_chunking(settings)

    if provider in ("openai", "voyage", "cohere"):
        assert claimed is False, "a hosted provider returns pooled vectors only"
        return

    if claimed:
        # Only legitimate via the transformers backend, which needs torch.
        from ragorc.embed.late_chunking import _torch_available

        assert _torch_available(), (
            f"{provider} claims late-chunking support without a token-capable backend installed"
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_is_none_when_disabled() -> None:
    """``None`` rather than a no-op object, so embedders skip key derivation
    entirely — hashing every input for a cache that never hits is pure overhead."""
    assert build_embedding_cache(settings=settings_for("fastembed")) is None


def test_cache_is_built_when_enabled() -> None:
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": True, "redis_url": ""},
        embedding={"provider": "fastembed", "cache_embeddings": True},
    )
    assert build_embedding_cache(settings=settings) is not None


def test_cache_respects_the_embedding_specific_switch() -> None:
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": True},
        embedding={"provider": "fastembed", "cache_embeddings": False},
    )
    assert build_embedding_cache(settings=settings) is None


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------
def test_importing_the_package_loads_no_model() -> None:
    """`import ragorc.embed` must not construct an ONNX session or require the
    optional provider SDKs."""
    import subprocess
    import sys

    code = (
        "import sys, ragorc.embed; "
        "print([m for m in ('onnxruntime','fastembed','openai','voyageai','cohere','torch') "
        "if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]", f"import pulled {out}"


def test_lazy_attributes_resolve_and_unknown_ones_raise() -> None:
    import ragorc.embed as embed

    assert embed.FastEmbedDense.__name__ == "FastEmbedDense"
    assert embed.LateChunkingEmbedder.__name__ == "LateChunkingEmbedder"
    with pytest.raises(AttributeError):
        _ = embed.NoSuchEmbedder


# ---------------------------------------------------------------------------
# Late chunking: the capability check and the resolver must agree
# ---------------------------------------------------------------------------
async def test_capability_check_agrees_with_the_resolver() -> None:
    """These two must never disagree.

    ``supports_late_chunking()`` is what a caller prints before an ingest;
    ``resolve_strategy()`` is what the ingest actually does. When they diverged,
    the CLI reported "late chunking: enabled" while the corpus was indexed EARLY,
    and nothing in the output said so.

    Asserted across every provider, because the previous version answered from a
    provider allow-list and was wrong for exactly the default configuration.
    """
    from ragorc.core.models import ChunkingStrategy
    from ragorc.embed.late_chunking import resolve_strategy

    for provider in ("fastembed", "openai", "voyage", "cohere"):
        settings = settings_for(provider)
        claimed = supports_late_chunking(settings)
        # Hosted providers cannot build a dense embedder without a key, so only
        # the local one is resolved for real; the claim is still checked.
        if provider != "fastembed":
            assert claimed is False, f"{provider} cannot support late chunking"
            continue
        resolved = await resolve_strategy(
            ChunkingStrategy.AUTO, build_dense_embedder(settings), settings
        )
        assert claimed is (resolved is ChunkingStrategy.LATE), (
            f"{provider}: supports_late_chunking()={claimed} but AUTO resolved to {resolved.value}"
        )


async def test_default_install_resolves_to_early_not_late() -> None:
    """Pins the documented behaviour of the zero-dependency default.

    FastEmbed's ``TextEmbedding`` returns pooled vectors only, so there is no
    token source for the dense model and late chunking is genuinely unavailable.
    ADR-0002 states this; the test keeps the statement true.
    """
    from ragorc.core.models import ChunkingStrategy
    from ragorc.embed.late_chunking import resolve_strategy

    settings = settings_for("fastembed")
    resolved = await resolve_strategy(
        ChunkingStrategy.AUTO, build_dense_embedder(settings), settings
    )
    assert resolved is ChunkingStrategy.EARLY


async def test_late_chunking_never_substitutes_another_model() -> None:
    """The bug this guards against, and it was a live one.

    ColBERT emits per-token vectors, so it looks like a valid token source. It is
    not: it is a different model in a different space at 128 dims against the
    dense model's 384. Pooling it produced vectors incomparable to any query
    vector — caught only because Qdrant rejected the dimension. At equal width it
    would have been accepted and returned meaningless neighbours indefinitely.
    """
    from ragorc.embed import build_late_chunking_embedder

    settings = settings_for("fastembed")
    embedder = build_late_chunking_embedder(settings)
    source = embedder.token_embedder
    assert source is not None
    assert source.model_name == settings.embedding.dense_model, (
        f"token source is {source.model_name!r}, not the dense model "
        f"{settings.embedding.dense_model!r} — pooled vectors would be in a "
        "different embedding space"
    )
    assert "colbert" not in source.model_name.lower()
