"""Building the graph: what gets merged, and what runs at all.

The retrieval side of GraphRAG was audited in rounds 8-10. This is the write
side, and it had three defects that share the shape of everything else here —
the guard exists, the stage exists, and neither is reached.

Over-merging is the failure mode that matters. A fragmented graph is
disappointing; a graph that has fused two different companies into one node
answers questions confidently and wrongly, and nothing in the pipeline can
detect it afterwards.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragorc.core.models import Entity, Relation
from ragorc.core.settings import Settings
from ragorc.index.graph.resolve import EntityResolver, normalized_form


def _settings() -> Settings:
    return Settings(llm={"api_key": "k"}, embedding={"dense_dimension": 32})


# ---------------------------------------------------------------------------
# Normalization must not delete a distinguishing word
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("The Acme Corporation, Inc.", "acme", id="english-article-and-suffix"),
        pytest.param("ACME Corp", "acme", id="suffix-only"),
        pytest.param("The Times", "times", id="english-article-alone"),
        pytest.param("El Salvador", "el salvador", id="spanish-article-is-part-of-the-name"),
        pytest.param("Los Angeles", "los angeles", id="spanish-plural-article"),
        pytest.param("La Paz", "la paz", id="french-or-spanish-la"),
        pytest.param("Der Spiegel", "der spiegel", id="german-article"),
        pytest.param("Le Monde", "le monde", id="french-article"),
    ],
)
def test_only_english_articles_are_stripped(name: str, expected: str) -> None:
    """An English leading "the" is usually droppable from a proper noun; a
    Romance or German article usually *is* the proper noun's first word. The list
    used to include `le la les el los der die das`, which deleted the first word
    of every such name."""
    assert normalized_form(name) == expected


async def test_two_places_that_differ_only_by_an_article_stay_apart() -> None:
    """The reproduction. Stage 2 unions on `(type, normalized_form)`
    *unconditionally* — `_cluster_conflict` is reached only from stage 3 — so
    these merged with no threshold and no veto, and the graph then asserted that
    a Central American country is the capital of a Brazilian state.

    Stage 3 scored this pair at 0.8167 against the 0.92 threshold: the guard that
    exists would have refused the merge it never saw.
    """
    entities = [
        Entity(
            name="El Salvador",
            type="LOCATION",
            description="A country in Central America.",
            source_chunk_ids=("c1",),
        ),
        Entity(
            name="Salvador",
            type="LOCATION",
            description="A coastal city in Bahia, Brazil.",
            source_chunk_ids=("c2",),
        ),
        Entity(name="Bahia", type="LOCATION", description="A Brazilian state.",
               source_chunk_ids=("c2",)),
    ]
    relations = [Relation("Salvador", "Bahia", "CAPITAL_OF", weight=1.0)]

    report = await EntityResolver(None, _settings()).resolve(entities, relations)

    assert len(report.entities) == 3, "two distinct places were fused into one node"
    assert report.normalized_merges == 0
    by_name = {e.name: e for e in report.entities}
    assert by_name["El Salvador"].description == "A country in Central America."
    assert report.relations[0].source == "Salvador", "the edge was re-pointed at the wrong node"


async def test_the_merge_the_normalizer_exists_for_still_happens() -> None:
    """The behaviour that must survive the narrowing: legal suffixes and an
    English article are still what let two spellings of one company meet."""
    entities = [
        Entity(name="The Acme Corporation, Inc.", type="ORG", description="A widget maker.",
               source_chunk_ids=("c1",)),
        Entity(name="ACME Corp", type="ORG", description="Makes widgets.",
               source_chunk_ids=("c2",)),
    ]

    report = await EntityResolver(None, _settings()).resolve(entities, [])

    assert len(report.entities) == 1
    assert report.normalized_merges == 1


# ---------------------------------------------------------------------------
# Stage 3 has to be reachable from the shipped command
# ---------------------------------------------------------------------------
def test_the_cli_hands_the_graph_builder_an_embedder() -> None:
    """`EntityResolver._merge_by_embedding` returns `{}` on its first line when
    the embedder is None, so resolution stage 3 — the only stage that can merge
    "Meta" with "Facebook" — never ran on the only shipped path. The engine has
    held a built dense embedder since `build()`; the example passes it and the
    command did not, which also made `graph.resolution_threshold` an inert knob
    that `docs/operations.md` tells operators to lower.
    """
    import inspect

    from ragorc import cli

    source = inspect.getsource(cli)
    construction = [line for line in source.splitlines() if "GraphBuilder(" in line]
    assert construction, "the graph build command no longer constructs a GraphBuilder"
    assert all("embedder=" in line for line in construction), construction


async def test_a_resolver_without_an_embedder_skips_stage_three_silently() -> None:
    """Why the above matters: the omission produces no error and no warning, just
    a report whose embedding counters are all zero."""
    entities = [
        Entity(name="Facebook", type="ORG", description="A social network.",
               source_chunk_ids=("c1",)),
        Entity(name="FB", type="ORG", description="A social network.",
               source_chunk_ids=("c2",)),
    ]

    report = await EntityResolver(None, _settings()).resolve(entities, [])

    assert len(report.entities) == 2
    assert (report.embedded, report.compared_pairs, report.embedding_merges) == (0, 0, 0)


# ---------------------------------------------------------------------------
# A second pass must not cost the first
# ---------------------------------------------------------------------------
async def test_a_failing_gleaning_pass_keeps_what_pass_one_found() -> None:
    """Gleaning exists to find what pass 1 missed. Losing what pass 1 found is
    strictly worse than not gleaning at all — and that is what happened: the
    exception propagated out of `_extract_chunk`, whose caller treats any
    exception as "this chunk failed", so the chunk left the graph entirely and
    the first call's usage went with it, under-reporting spend already paid.
    """
    from ragorc.core.models import Chunk
    from ragorc.core.schemas import ExtractionOutput
    from ragorc.index.graph.extract import EntityExtractor

    class FlakyLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def structured(self, prompt: str, schema: Any, **kw: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                from ragorc.core.models import Usage

                return (
                    ExtractionOutput.model_validate(
                        {
                            "entities": [
                                {"name": "Acme", "type": "ORG", "description": "A widget maker."}
                            ],
                            "relations": [],
                        }
                    ),
                    Usage(model="stub", calls=1),
                )
            raise TimeoutError("gleaning timed out")

    settings = Settings(
        llm={"api_key": "k"},
        embedding={"dense_dimension": 32},
        graph={"enabled": True, "max_gleanings": 2},
    )
    llm = FlakyLLM()
    extractor = EntityExtractor(llm, settings=settings)
    chunk = Chunk(id="c1", content="Acme makes widgets in Leeds.", document_id="d")

    result, usage = await extractor.extract([chunk])

    assert [e.name for e in result.entities] == ["Acme"], "pass 1's extraction was discarded"
    assert result.chunks_failed == 0, "a gleaning failure marked the whole chunk failed"
    assert result.gleaning_failures == 1
    assert usage.calls >= 1, "the first call was paid for and must be reported"
