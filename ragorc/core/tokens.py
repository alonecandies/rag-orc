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

import functools
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["TokenBudget", "count_tokens", "count_tokens_batch", "truncate_to_tokens"]

_DEFAULT_ENCODING = "o200k_base"


@functools.lru_cache(maxsize=32)
def _encoder(model: str | None = None):  # noqa: ANN202
    try:
        import tiktoken
    except ImportError:  # pragma: no cover - tiktoken is a base dependency
        return None
    if model:
        try:
            return tiktoken.encoding_for_model(model)
        except (KeyError, ValueError):
            pass
    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception:  # pragma: no cover
        return None


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
