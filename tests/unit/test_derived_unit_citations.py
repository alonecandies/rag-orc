"""Citations for text that exists nowhere in the document it names.

Three units in this library carry generated text: a RAPTOR cluster summary, an
LLM chunk summary, and a proposition the model rewrote rather than quoted. All
three set ``start_char = 0`` deliberately, and both multi-representation indexers
explain why in a docstring — then name ``parent_start_char`` as the key that
restores the real base. They wrote ``source_start_char``, which nothing read, and
the citation layer never asked whether an offset existed at all.

The result was a citation naming the real document, at a span pointing at
unrelated text, quoting a sentence no human wrote:

    [RAPTOR-SUMMARY] doc=d1 chunk=4017826a start=0 end=82
    quote: 'Overall the vendor offers generous money-back arrangements...'

A citation with no offset is a weaker citation. A citation with a *wrong* offset
is a false one, and it is the class of output the span validator exists to make
checkable.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Chunk, Modality, ScoredChunk
from ragorc.generate.citations import extract_citations

_DOCUMENT = (
    "Refunds are available for thirty days after delivery. After that window the "
    "item is yours. Shipping costs are not refundable in either case."
)
_SUMMARY = "The vendor offers generous money-back arrangements with no fixed deadline."


def _cite(chunk: Chunk, claim: str, **explain: Any) -> Any:
    scored = ScoredChunk(chunk=chunk, score=0.9)
    scored.explain.update(explain)
    found = extract_citations(f"{claim} [1].", [scored])
    assert found, "the marker did not resolve"
    return found[0]


# ---------------------------------------------------------------------------
# Generated text has no position
# ---------------------------------------------------------------------------
def test_a_raptor_summary_cites_without_an_offset() -> None:
    """`level > 0` is a cluster summary over many chunks. There is no single span
    of any document that it came from, and `start_char=0` is a placeholder the
    indexer sets on purpose — not a claim that the summary begins the file."""
    summary = Chunk(id="r1", content=_SUMMARY, document_id="d1", level=1, start_char=0)

    citation = _cite(summary, "The vendor offers generous money-back arrangements")

    assert citation.document_id == "d1"
    assert citation.quote, "the quote is still the summary's own sentence"
    assert citation.start_char is None, f"claimed document span at {citation.start_char}"
    assert citation.end_char is None


def test_an_llm_summary_cites_without_an_offset() -> None:
    summary = Chunk(
        id="s1", content=_SUMMARY, document_id="d1", modality=Modality.SUMMARY, start_char=0
    )

    citation = _cite(summary, "The vendor offers generous money-back arrangements")

    assert citation.start_char is None


def test_a_rewritten_proposition_cites_without_an_offset() -> None:
    """Dense-X marks `verbatim=False` when it could not find the model's sentence
    in the source. That flag existed and had no reader."""
    proposition = Chunk(
        id="p1",
        content="Customers may return items within one month.",
        document_id="d1",
        modality=Modality.PROPOSITION,
        start_char=0,
        end_char=0,
        metadata={"verbatim": False},
    )

    citation = _cite(proposition, "Customers may return items within one month")

    assert citation.start_char is None


def test_a_verbatim_proposition_keeps_its_offset() -> None:
    """The half a blanket rule would break. Dense-X restores the author's exact
    characters when it finds them, precisely so the span is checkable — throwing
    that away would be the same loss in the other direction."""
    text = "Refunds are available for thirty days after delivery."
    proposition = Chunk(
        id="p2",
        content=text,
        document_id="d1",
        modality=Modality.PROPOSITION,
        start_char=0,
        end_char=len(text),
        metadata={"verbatim": True},
    )

    citation = _cite(proposition, "Refunds are available for thirty days")

    assert citation.start_char == 0
    assert _DOCUMENT[citation.start_char : citation.end_char] == text


def test_an_ordinary_chunk_is_unaffected() -> None:
    """The overwhelmingly common case must keep its spans."""
    chunk = Chunk(id="c1", content=_DOCUMENT, document_id="d1", start_char=100)

    citation = _cite(chunk, "Shipping costs are not refundable")

    assert citation.start_char is not None and citation.start_char >= 100


# ---------------------------------------------------------------------------
# The key the indexers promise is the key they write
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module", ["ragorc.index.multirep.summary", "ragorc.index.multirep.dense_x"]
)
def test_the_indexers_write_the_key_their_docstrings_name(module: str) -> None:
    """Both said "the packer restores the real base offset from
    ``parent_start_char``" and both wrote ``source_start_char`` — a key with no
    reader in the library. Two vocabularies for one value is how a promise stays
    true in prose and false in code."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    assert "source_start_char" not in source, "the key nothing reads is still written"
    assert 'metadata["parent_start_char"]' in source


@pytest.mark.parametrize(
    ("module", "cls"),
    [
        ("ragorc.index.multirep.summary", "SummaryIndexer"),
        ("ragorc.index.multirep.dense_x", "PropositionIndexer"),
    ],
)
def test_the_written_offset_is_the_sources_offset(module: str, cls: str) -> None:
    """The value, not the key. A grep for the name is satisfied by writing zero,
    and a base offset of zero is exactly the false span this file exists to stop.
    """
    import importlib

    from ragorc.core.settings import Settings

    module_obj = importlib.import_module(module)
    indexer = getattr(module_obj, cls)(object(), settings=Settings(llm={"api_key": "k"}))
    source = Chunk(id="c1", content=_DOCUMENT, document_id="d1", start_char=137, end_char=200)

    if cls == "SummaryIndexer":
        unit = indexer._as_summary_unit(source, _SUMMARY, "Refunds")
    else:
        import numpy as np

        # `_unit` is the proposition builder; `_as_source_unit` returns the source
        # itself and carries no derived offset to check.
        unit = indexer._unit(
            source,
            "Customers may return items.",
            "",
            np.zeros(0, dtype=np.int32),
            truncated=False,
        )

    assert unit.metadata["parent_start_char"] == 137, unit.metadata
    assert unit.metadata["parent_end_char"] == 200


def test_a_window_expanded_derived_unit_keeps_its_window_base() -> None:
    """The regression the rename introduced, and the reason the base is keyed on
    the expansion rather than on a fixed priority.

    A summary built from a sentence-window chunk inherits ``window_start`` from its
    source *and* gains ``parent_start_char``. The packer substitutes
    ``parent_text or window_text``; `_base_offset` read ``parent_start_char``
    first. When only the window text was present the two halves disagreed about
    which span the prompt held, and the citation moved by a whole window —
    measured (88, 126), slicing to 'ing is never refundable. Closing notes'.
    """
    from ragorc.context.pack import ContextPacker
    from ragorc.core.settings import Settings

    document = (
        "Opening remarks and preamble go here first. Refunds are available for thirty days. "
        "Shipping is never refundable. Closing notes follow at the very end of the page."
    )
    sentence = "Refunds are available for thirty days."
    unit = Chunk(
        id="s1",
        content=sentence,
        document_id="d1",
        modality=Modality.SUMMARY,
        parent_id="p1",
        metadata={
            "window_text": document,
            "window_start": 0,
            "parent_start_char": document.index(sentence),
        },
    )
    settings = Settings(
        llm={"api_key": "k"},
        security={"enforce_tenant_isolation": False},
        retrieval={"parent_expansion": True, "dedupe_enabled": False},
    )
    packed = ContextPacker(settings).build(
        [ScoredChunk(chunk=unit, score=0.9)], budget=400, isolate=False
    )
    assert packed.chunks[0].explain["expansion"] == "window", "the packer widened with the window"

    citation = extract_citations("Refunds are available for thirty days [1].", packed.chunks)[0]
    assert document[citation.start_char : citation.end_char] == sentence, (
        f"({citation.start_char}, {citation.end_char}) slices to "
        f"{document[citation.start_char : citation.end_char]!r}"
    )


def test_a_parent_expanded_unit_uses_the_parent_base() -> None:
    """The other branch of the same decision."""
    from ragorc.context.pack import ContextPacker
    from ragorc.core.settings import Settings

    document = "Preamble here. " + _DOCUMENT
    unit = Chunk(
        id="s1",
        content=_SUMMARY,
        document_id="d1",
        modality=Modality.SUMMARY,
        parent_id="p1",
        metadata={"parent_text": _DOCUMENT, "parent_start_char": len("Preamble here. ")},
    )
    settings = Settings(
        llm={"api_key": "k"},
        security={"enforce_tenant_isolation": False},
        retrieval={"parent_expansion": True, "dedupe_enabled": False},
    )
    packed = ContextPacker(settings).build(
        [ScoredChunk(chunk=unit, score=0.9)], budget=400, isolate=False
    )
    assert packed.chunks[0].explain["expansion"] == "parent"

    citation = extract_citations("Shipping costs are not refundable [1].", packed.chunks)[0]
    assert citation.quote.strip() in document[citation.start_char : citation.end_char]


def test_only_expansion_may_re_base_a_citation() -> None:
    """`parent_start_char` describes the *parent*, so it is the right base only
    once the packer has actually substituted the parent's text. Reading it off an
    unexpanded chunk re-bases the quote by a whole span — which is the same defect
    the key was introduced to fix, pointed the other way."""
    child = Chunk(
        id="c1",
        content="Shipping costs are not refundable in either case.",
        document_id="d1",
        parent_id="p1",
        start_char=95,
        metadata={"parent_start_char": 0, "parent_text": _DOCUMENT},
    )

    unexpanded = _cite(child, "Shipping costs are not refundable")
    assert unexpanded.start_char == 95

    expanded_chunk = Chunk(
        id="c1",
        content=_DOCUMENT,
        document_id="d1",
        parent_id="p1",
        start_char=95,
        metadata={"parent_start_char": 0, "parent_text": _DOCUMENT},
    )
    expanded = _cite(expanded_chunk, "Shipping costs are not refundable", expanded=True)
    assert expanded.start_char == _DOCUMENT.index("Shipping costs are not refundable")
