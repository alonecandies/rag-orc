"""Token counting and budgeting.

Counting is the foundation of both cost control and overflow handling, so it
has to be cheap. ``tiktoken`` is a Rust BPE; the encoder is cached per model
and the count is a single FFI call.

For models without a public tokenizer (most OpenRouter models are not OpenAI
models) we fall back to ``o200k_base``, which is within a few percent for
Llama/Mistral/Claude-family text — accurate enough for budgeting, and we
reconcile against the provider's reported usage after each call anyway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import structlog

__all__ = [
    "TokenBudget",
    "count_tokens",
    "count_tokens_batch",
    "load_encoder",
    "truncate_to_tokens",
]

_DEFAULT_ENCODING = "o200k_base"

log = structlog.get_logger(__name__)

_ENCODERS: dict[str | None, Any] = {}
"""Successfully loaded encoders, by model.

A plain dict rather than ``functools.lru_cache`` because only *successes* may be
cached. ``lru_cache`` memoizes the ``None`` a failed load returns, so one
transient network blip at first use pinned the four-chars-per-token estimate for
the rest of the process — silently, and permanently, even after the network came
back. ``count_tokens``'s docstring framed that estimate as "if tiktoken is
absent", so the symptom read as a missing dependency rather than a cached
failure."""

_LOAD_FAILURES = 0
"""How many times the encoder failed to load. Only used to log the first one:
a per-call warning on a hot path is its own outage."""


def load_encoder(model: str | None = None) -> Any:
    """The tokenizer for ``model``, or ``None`` if it cannot be loaded.

    ``tiktoken.get_encoding`` downloads a ~1.6 MB BPE file over a **synchronous,
    blocking** socket the first time it is asked. Every caller of
    :func:`count_tokens` in this library is reached from ``async def`` request
    code with no ``to_thread`` hop — the context budgeter, the packer, the
    generator — so on a cold cache the first query stalled the whole event loop,
    and with it every other in-flight request including the health probe.

    :meth:`~ragorc.pipeline.builder.RAGPipeline._warmup` now loads it in a thread
    at startup, which is where the docstring said the one-time costs get paid.
    This function stays synchronous because that is what a token count is; what
    changed is that a failure is no longer permanent.
    """
    global _LOAD_FAILURES
    if model in _ENCODERS:
        return _ENCODERS[model]
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is a base dependency
        return None
    encoder = None
    if model:
        try:
            encoder = tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            encoder = None
    if encoder is None:
        try:
            encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)
        except Exception as exc:  # noqa: BLE001 - fall back to the estimate
            _LOAD_FAILURES += 1
            if _LOAD_FAILURES == 1:
                log.warning(
                    "tokenizer_unavailable",
                    error=str(exc)[:200],
                    effect="token counts fall back to a 4-chars-per-token estimate",
                    retried="on every call until it succeeds",
                )
            return None
    _ENCODERS[model] = encoder
    return encoder


def _encoder(model: str | None = None):  # noqa: ANN202
    """Backwards-compatible alias for :func:`load_encoder`."""
    return load_encoder(model)


def count_tokens(text: str, model: str | None = None) -> int:
    """Token count, with a 4-chars-per-token estimate if tiktoken is absent."""
    if not text:
        return 0
    enc = _encoder(model)
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text, disallowed_special=()))


def count_tokens_batch(texts: Sequence[str], model: str | None = None) -> list[int]:
    """Batch count. ``encode_batch`` releases the GIL and runs in parallel in
    Rust, so this is markedly faster than a Python loop over ``count_tokens``."""
    if not texts:
        return []
    enc = _encoder(model)
    if enc is None:
        return [max(1, len(t) // 4) for t in texts]
    return [len(ids) for ids in enc.encode_batch(list(texts), disallowed_special=())]


def truncate_to_tokens(text: str, limit: int, model: str | None = None) -> str:
    """Hard truncate at a token boundary (not a character boundary, which can
    split a multi-byte token and produce mojibake)."""
    if limit <= 0:
        return ""
    enc = _encoder(model)
    if enc is None:
        return text[: limit * 4]
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= limit:
        return text
    return enc.decode(ids[:limit])


@dataclass(slots=True)
class TokenBudget:
    """Explicit allocation of a model's context window.

    Reserving space for the answer *before* packing context is what prevents
    the classic failure where retrieved documents fill the window and the model
    has no room left to reply.
    """

    total: int
    reserved_output: int = 1024
    reserved_system: int = 0
    reserved_query: int = 0
    safety_margin: float = 0.05
    """Absorbs tokenizer drift between our estimate and the provider's count."""

    @property
    def available_context(self) -> int:
        usable = int(self.total * (1.0 - self.safety_margin))
        remaining = usable - self.reserved_output - self.reserved_system - self.reserved_query
        return max(remaining, 0)

    def fits(self, tokens: int) -> bool:
        return tokens <= self.available_context

    def split(self, shares: dict[str, float]) -> dict[str, int]:
        """Divide the context window between sources (e.g. 60% vector, 25%
        graph, 15% sql). Shares are normalized, so they need not sum to 1."""
        if not shares:
            return {}
        total_share = sum(shares.values()) or 1.0
        available = self.available_context
        return {k: int(available * (v / total_share)) for k, v in shares.items()}
