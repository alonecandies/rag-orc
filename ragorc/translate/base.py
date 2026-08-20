"""Query translation: shared machinery.

The premise of this whole layer is that **the user's phrasing is one sample from
the space of ways their question could be asked**, and the document that answers
it was written in a different sample. Dense retrieval fails at exactly that gap.

Which means the value of a query variant is its *vocabulary distance* from the
original, not its existence. Three paraphrases that share the same nouns retrieve
the same documents and buy nothing but latency and tokens. So the cleaning here is
not cosmetic: it drops variants that are too close to the original to change the
result set, and that filter is what keeps a 4x fan-out from being 4x waste.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

import structlog

from ragorc.core.models import Query, Usage
from ragorc.core.protocols import LLM, QueryTranslator
from ragorc.core.settings import Settings, get_settings
from ragorc.llm.router import ModelRouter

log = structlog.get_logger(__name__)

__all__ = ["BaseTranslator", "CompositeTranslator", "clean_variants"]

#: Leading enumeration a model adds despite the schema: "1. ", "2) ", "- ", "* ".
_ENUMERATION = re.compile(r"^\s*(?:\d{1,2}[.)\]]|[-*•])\s+")
# Escaped rather than literal so the smart quotes this strips do not themselves
# trip the ambiguous-character lint.
_QUOTED = re.compile("^[\"\u201c\u2018'](.*)[\"\u201d\u2019']$", re.DOTALL)


def clean_variants(
    variants: Sequence[str],
    *,
    original: str,
    max_variants: int = 8,
    min_distance: float = 0.12,
    min_length: int = 8,
) -> list[str]:
    """Normalize and prune generated query variants.

    ``min_distance`` is the load-bearing parameter: a variant whose similarity to
    the original (or to an already-kept variant) exceeds ``1 - min_distance``
    would retrieve substantially the same documents, so keeping it costs a
    retrieval round and returns duplicates that the fusion step then has to
    collapse. Dropping it early is strictly better.
    """
    kept: list[str] = []
    baseline = original.strip().lower()
    for raw in variants:
        if not raw:
            continue
        text = _ENUMERATION.sub("", raw).strip()
        if match := _QUOTED.match(text):
            text = match.group(1).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) < min_length:
            continue
        lowered = text.lower()
        if lowered == baseline:
            continue
        if SequenceMatcher(None, lowered, baseline).ratio() > (1.0 - min_distance):
            log.debug("variant_dropped", reason="too_similar_to_original", variant=text[:60])
            continue
        if any(
            SequenceMatcher(None, lowered, existing.lower()).ratio() > (1.0 - min_distance)
            for existing in kept
        ):
            log.debug("variant_dropped", reason="duplicate_of_variant", variant=text[:60])
            continue
        kept.append(text)
        if len(kept) >= max_variants:
            break
    return kept


class BaseTranslator:
    """Common construction and variant bookkeeping for translators."""

    name = "base"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        """Produce an enriched query. Every subclass overrides this.

        Declared here so the base advertises the contract — a subclass that omits
        it is now a type error and a clear message, where before it was an
        ``AttributeError`` raised mid-request from inside a composite.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement translate()")

    @staticmethod
    def _extend(query: Query, new_variants: Sequence[str], **metadata: object) -> Query:
        """Return a copy of ``query`` with variants appended.

        A copy, not a mutation: the pipeline keeps the pre-translation query for
        grading and for the trace, and mutating in place would silently rewrite
        history for every stage that already holds a reference.
        """
        merged = clean_variants(
            [*query.variants, *new_variants],
            original=query.original or query.text,
        )
        return Query(
            text=query.text,
            original=query.original,
            variants=tuple(merged),
            hypothetical=query.hypothetical,
            filters=dict(query.filters),
            top_k=query.top_k,
            dense=query.dense,
            sparse=query.sparse,
            multi=query.multi,
            tenant_id=query.tenant_id,
            metadata={**query.metadata, **metadata},
        )


class CompositeTranslator:
    """Runs translators in sequence, accumulating variants and cost.

    Sequential rather than parallel on purpose: step-back and decomposition are
    meant to see the original question, while a later translator may legitimately
    want the variants an earlier one produced (HyDE over a step-back question, for
    instance). Order is the caller's choice and it matters.
    """

    name = "composite"

    def __init__(self, translators: Sequence[QueryTranslator], *, max_variants: int = 8) -> None:
        self.translators = list(translators)
        self.max_variants = max_variants

    async def translate(self, query: Query) -> tuple[Query, Usage]:
        usages: list[Usage] = []
        current = query
        for translator in self.translators:
            try:
                current, usage = await translator.translate(current)
            except Exception as exc:  # noqa: BLE001
                # Translation is an enhancement, not a requirement. A failed
                # expansion must degrade to the original query, never fail the
                # request: the un-expanded query still retrieves something.
                log.warning(
                    "translator_failed",
                    translator=getattr(translator, "name", type(translator).__name__),
                    error=str(exc)[:200],
                )
                continue
            usages.append(usage)
        if len(current.variants) > self.max_variants:
            current = Query(
                text=current.text,
                original=current.original,
                variants=current.variants[: self.max_variants],
                hypothetical=current.hypothetical,
                filters=current.filters,
                top_k=current.top_k,
                dense=current.dense,
                sparse=current.sparse,
                multi=current.multi,
                tenant_id=current.tenant_id,
                metadata=current.metadata,
            )
        log.info(
            "translated",
            translators=[getattr(t, "name", "?") for t in self.translators],
            variants=len(current.variants),
        )
        return current, Usage.sum(usages)
