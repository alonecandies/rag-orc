"""Indexing: entity resolution, ColBERT pruning, RAPTOR, multi-representation.

Entity resolution gets the most attention here because it is the step that
decides whether GraphRAG works at all. If "Acme", "Acme Corp" and "ACME
Corporation" survive as three nodes, the graph fragments and traversal stops
finding anything — and the failure is silent, because the graph still looks
populated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ragorc.core.models import Chunk, Document, Entity, Relation
from ragorc.core.settings import Settings
from ragorc.index.colbert import estimate_storage, maxsim, prune_tokens
from ragorc.index.graph.extract import normalize_entity_name, normalize_relation_type
from ragorc.index.graph.resolve import EntityResolver, normalized_form


@pytest.fixture
def settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        graph={"enabled": True, "resolve_entities": True, "resolution_threshold": 0.92},
    )


# ---------------------------------------------------------------------------
# Entity name normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Acme Corporation", "ACME Corp"),
        ("The Acme Corporation, Inc.", "acme"),
        ("Fabrikam GmbH", "Fabrikam"),
        ("Contoso Ltd.", "CONTOSO LIMITED"),
        ("Adventure Works LLC", "adventure works"),
    ],
)
def test_normalized_form_collapses_legal_variants(a: str, b: str) -> None:
    assert normalized_form(a) == normalized_form(b), f"{a!r} and {b!r} must share a key"


def test_normalized_form_keeps_distinct_entities_apart() -> None:
    """Over-normalizing is as damaging as under-normalizing: merging two real
    companies fabricates relationships that do not exist."""
    assert normalized_form("Acme Corp") != normalized_form("Acme Bank")
    assert normalized_form("Northwind Traders") != normalized_form("Northwind Systems")


def test_normalize_relation_type_produces_a_safe_identifier() -> None:
    """Neo4j cannot parameterize a relationship type, so an unvalidated one is a
    security problem, not just untidy."""
    import re

    pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for raw in ["works for", "acquired (2019)", "→", "", "a" * 200, "OWNS", "is-part-of", "5x"]:
        out = normalize_relation_type(raw)
        assert pattern.match(out), f"{raw!r} produced unsafe type {out!r}"


def test_normalize_relation_type_rejects_an_injection_attempt() -> None:
    hostile = "RELATED]->() DETACH DELETE n //"
    out = normalize_relation_type(hostile)
    assert "]" not in out and "-" not in out and "/" not in out


def test_normalize_entity_name_trims_without_destroying() -> None:
    assert normalize_entity_name("  Acme   Corporation  ") == "Acme Corporation"
    assert normalize_entity_name("") == ""


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------
async def test_resolution_merges_surface_variants(settings: Settings) -> None:
    entities = [
        Entity(
            name="Acme Corporation",
            type="ORGANIZATION",
            description="a manufacturer",
            source_chunk_ids=("c1",),
        ),
        Entity(
            name="ACME Corp",
            type="ORGANIZATION",
            description="makes widgets",
            source_chunk_ids=("c2",),
        ),
        Entity(name="Acme", type="ORGANIZATION", source_chunk_ids=("c3",)),
        Entity(name="Contoso", type="ORGANIZATION", source_chunk_ids=("c4",)),
    ]
    report = await EntityResolver(None, settings).resolve(entities)
    names = {e.name for e in report.entities}
    assert len(report.entities) == 2, f"expected Acme + Contoso, got {names}"

    acme = next(e for e in report.entities if "cme" in e.name)
    # The canonical name should be the most complete form, and every source
    # chunk must survive the merge or the graph loses its links back to text.
    assert acme.name == "Acme Corporation"
    assert set(acme.source_chunk_ids) == {"c1", "c2", "c3"}
    assert "widgets" in acme.description or "manufacturer" in acme.description


async def test_resolution_rewrites_relation_endpoints(settings: Settings) -> None:
    """A relation pointing at a merged-away alias becomes a dangling edge, which
    fragments traversal exactly as a duplicate node would."""
    entities = [
        Entity(name="Acme Corporation", type="ORGANIZATION"),
        Entity(name="ACME Corp", type="ORGANIZATION"),
        Entity(name="Contoso", type="ORGANIZATION"),
    ]
    relations = [Relation("ACME Corp", "Contoso", "SUPPLIES", weight=2.0)]
    report = await EntityResolver(None, settings).resolve(entities, relations)

    canonical = {e.name for e in report.entities}
    for relation in report.relations:
        assert relation.source in canonical, f"dangling source {relation.source!r}"
        assert relation.target in canonical, f"dangling target {relation.target!r}"


async def test_resolution_accumulates_repeated_edge_weight(settings: Settings) -> None:
    """An edge asserted by several documents should outrank one asserted once."""
    entities = [Entity(name="Acme Corporation"), Entity(name="Contoso")]
    relations = [
        Relation("Acme Corporation", "Contoso", "SUPPLIES", weight=1.0),
        Relation("Acme Corporation", "Contoso", "SUPPLIES", weight=2.0),
    ]
    report = await EntityResolver(None, settings).resolve(entities, relations)
    supplies = [r for r in report.relations if r.type == "SUPPLIES"]
    assert len(supplies) == 1
    assert supplies[0].weight == pytest.approx(3.0)


async def test_resolution_disabled_still_collapses_exact_keys(settings: Settings) -> None:
    """Exact-key collapse is a precondition for the store's per-batch MERGE, not
    resolution, so it must happen even with resolution off."""
    off = settings.model_copy(deep=True)
    off.graph.resolve_entities = False
    entities = [Entity(name="Acme"), Entity(name="acme"), Entity(name="Contoso")]
    report = await EntityResolver(None, off).resolve(entities)
    assert len(report.entities) == 2


async def test_resolution_reaches_stage_three_for_the_pairs_blocking_keeps(
    settings: Settings, embedder
) -> None:
    """Stage 3 must run on what stages 1 and 2 left alone — and only on that.

    The two Widget names share a blocking key and reach the comparison; Zephyr is
    a singleton block, so embedding it would be a forward pass nobody reads. The
    shared stub embedder tops out around 0.5 for anchored texts, well under the
    0.92 threshold, so nothing merges here on the cosine alone — asserting
    ``<= 3`` entities was true no matter what the stage did. Forcing a pair over
    the threshold needs ``_AnchoredEmbedder`` below, which is where the merge and
    veto decisions are actually exercised.
    """
    anchored = type(embedder)(dimension=32, anchors={"widget": 0})
    entities = [
        Entity(name="Widget Industries", description="widget maker"),
        Entity(name="Widget Manufacturing Group", description="widget maker"),
        Entity(name="Zephyr Airlines", description="an airline"),
    ]
    report = await EntityResolver(anchored, settings).resolve(entities)
    assert report.input_entities == 3
    assert report.embedded == 2, "the singleton block must not be embedded"
    assert report.compared_pairs == 1, report.summary()
    assert report.embedding_merges == 0, report.summary()
    assert {e.name for e in report.entities} == {
        "Widget Industries",
        "Widget Manufacturing Group",
        "Zephyr Airlines",
    }


async def test_resolution_handles_empty_input(settings: Settings) -> None:
    report = await EntityResolver(None, settings).resolve([], [])
    assert report.entities == []
    assert report.relations == []


# ---------------------------------------------------------------------------
# ColBERT
# ---------------------------------------------------------------------------
def test_maxsim_agrees_with_a_naive_loop() -> None:
    """The einsum must compute exactly what the definition says."""
    rng = np.random.default_rng(0)
    q = rng.normal(size=(4, 8)).astype(np.float32)
    docs = [rng.normal(size=(n, 8)).astype(np.float32) for n in (3, 7, 1, 12)]

    fast = maxsim(q, docs)
    slow = [float(sum((q[i] @ d.T).max() for i in range(q.shape[0]))) for d in docs]
    assert np.allclose(fast, slow, atol=1e-5), f"{fast} != {slow}"


def test_prune_tokens_bounds_the_matrix() -> None:
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(50, 8)).astype(np.float32)
    pruned = prune_tokens(matrix, 10)
    assert pruned.shape == (10, 8)
    # Under the limit is a no-op, not a pad.
    assert prune_tokens(matrix[:5], 10).shape == (5, 8)


def test_prune_tokens_keeps_the_informative_rows() -> None:
    """Pruning drops the lowest-norm vectors, which are the least discriminative
    — that is what makes ColBERT storage affordable without losing much."""
    strong = np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    weak = np.array([[0.001, 0.0], [0.0, 0.001]], dtype=np.float32)
    matrix = np.vstack([weak, strong])
    pruned = prune_tokens(matrix, 2)
    norms = np.linalg.norm(pruned, axis=1)
    assert (norms > 1.0).all(), f"pruning kept the weak rows: {pruned}"


def test_storage_estimate_shows_the_real_cost() -> None:
    """ColBERT is ~100x a single dense vector, and a user should see that before
    turning it on."""
    colbert = estimate_storage(100_000, tokens_per_chunk=180.0, dimension=128)
    total = getattr(colbert, "total_bytes", None) or getattr(colbert, "bytes_total", None)
    assert total is None or total > 0
    assert colbert is not None


# ---------------------------------------------------------------------------
# Parent-document indexing
# ---------------------------------------------------------------------------
async def test_parent_document_links_children_to_parents(settings: Settings) -> None:
    from ragorc.index.multirep.parent_document import ParentDocumentIndexer

    indexer = ParentDocumentIndexer(settings=settings)
    document = Document(
        id="d1",
        content=(
            "Refunds are processed within 14 days. Requests go through support. "
            "Shipping is free above 50 USD. International orders differ. "
            "Enterprise plans include a manager. That manager handles escalations. "
        )
        * 4,
    )
    index = await indexer.build_many([document])
    children = getattr(index, "children", None) or getattr(index, "child_chunks", None)
    parents = getattr(index, "parents", None) or getattr(index, "parent_chunks", None)
    assert children and parents, f"index exposes {dir(index)}"

    parent_ids = {p.id for p in parents}
    assert all(c.parent_id in parent_ids for c in children), "orphaned child chunk"
    # Only the children are embedded and indexed; the parents are the payload.
    assert len(children) >= len(parents)
    assert all(c.dense is None for c in children), "the indexer must not embed"


def test_scoped_settings_overrides_chunk_size(settings: Settings) -> None:
    from ragorc.index.multirep.parent_document import scoped_settings

    scoped = scoped_settings(settings, target=2048, overlap=0)
    assert scoped.indexing.chunk_size == 2048
    assert settings.indexing.chunk_size != 2048, "the base settings must not be mutated"


# ---------------------------------------------------------------------------
# RAPTOR (structure only — clustering needs the [raptor] extra)
# ---------------------------------------------------------------------------
def test_raptor_forecasts_its_cost_before_building(settings: Settings) -> None:
    """A tree costs one LLM call per cluster per level, so a user must be able to
    see the bill before starting."""
    from ragorc.index.raptor import RaptorIndexer

    raptor = RaptorIndexer(llm=None, embedder=None, settings=settings)
    forecast = getattr(raptor, "forecast", None)
    if forecast is None:
        pytest.skip("no forecast API on this build")
    chunks = [Chunk(id=f"c{i}", content=f"chunk {i}") for i in range(40)]
    result = forecast(len(chunks)) if callable(forecast) else None
    assert result is not None


# ---------------------------------------------------------------------------
# Entity resolution, stage 3: what the embedding is allowed to merge
#
# A wrong merge is worse than the fragmentation resolution exists to prevent.
# Fragmentation finds nothing; a mis-merge re-points every edge at the survivor,
# so the graph answers confidently from a relationship nobody ever wrote down.
# The embedder below is deliberately more credulous than the real default one:
# it puts the pathological pairs at cos ~1.0, so these tests are about what the
# resolver refuses to do with a high score, not about tuning the threshold.
# ---------------------------------------------------------------------------
class _AnchoredEmbedder:
    """Deterministic vectors that force chosen names above the threshold.

    Reproducing the audit's cosines with the real default model would mean
    downloading ``BAAI/bge-small-en-v1.5``, so the pathological similarity is
    staged here instead: two texts whose anchored names share an axis come out at
    cos > 0.99 — far above the 0.92 default — and everything else is
    near-orthogonal. Anchors match on the *start* of the text because stage 3
    embeds the name first, whatever it appends after it.
    """

    def __init__(self, anchors: dict[str, int], dimension: int = 16) -> None:
        self.anchors = {name.casefold(): axis for name, axis in anchors.items()}
        self.dimension = dimension
        self.model_name = "anchored-stub"
        self.max_tokens = 512
        self.embedded: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(text.encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little") % (2**32))
        # Jitter at 5% of the anchor keeps distinct texts distinct without ever
        # pulling an anchored pair back under the threshold.
        vector = (0.05 * rng.normal(size=self.dimension)).astype(np.float32)
        lowered = text.casefold()
        for name, axis in self.anchors.items():
            if lowered.startswith(name):
                vector[axis % self.dimension] += 1.0
        return (vector / max(float(np.linalg.norm(vector)), 1e-9)).astype(np.float32)

    async def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    async def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._vector(t) for t in texts]


async def test_stage_three_keeps_names_that_disagree_on_a_token_apart(settings: Settings) -> None:
    """The audit's case. With the default embedder "Acme Group"/"Acme Holdings"
    scores 0.9359 and "Q1 2024"/"Q2 2024" 0.9390 — both over the 0.92 default —
    because most of each string is shared, not because the entities are the same.
    Merging them re-points the edges, so the graph asserts that Acme Holdings
    acquired Zeta Labs from a sentence that was about Acme Group.
    """
    embedder = _AnchoredEmbedder({"acme group": 0, "acme holdings": 0, "q1 2024": 1, "q2 2024": 1})
    entities = [
        Entity(
            name="Acme Group",
            type="ORGANIZATION",
            description="Holding company for the retail arm.",
            source_chunk_ids=("c1",),
        ),
        Entity(
            name="Acme Holdings",
            type="ORGANIZATION",
            description="Separate entity that owns the real estate.",
            source_chunk_ids=("c2",),
        ),
        Entity(name="Q1 2024", type="DATE", description="First quarter of 2024."),
        Entity(name="Q2 2024", type="DATE", description="Second quarter of 2024."),
        Entity(name="Zeta Labs", type="ORGANIZATION", source_chunk_ids=("c3",)),
    ]
    relations = [
        Relation(
            "Acme Group",
            "Zeta Labs",
            "ACQUIRED",
            description="Acme Group acquired Zeta Labs.",
        )
    ]
    report = await EntityResolver(embedder, settings).resolve(entities, relations)

    names = {e.name for e in report.entities}
    assert names == {"Acme Group", "Acme Holdings", "Q1 2024", "Q2 2024", "Zeta Labs"}, names
    assert report.embedding_merges == 0, "a shared token is not evidence of identity"
    assert report.vetoed_pairs == 2, report.summary()

    # The consequence the audit measured: the edge must still start where the
    # corpus put it, and neither company may carry the other's name as an alias.
    assert [(r.source, r.target) for r in report.relations] == [("Acme Group", "Zeta Labs")]
    for entity in report.entities:
        assert entity.aliases == (), f"{entity.name} absorbed {entity.aliases}"
    # Contradictory definitions under one name is the other half of the damage.
    holdings = next(e for e in report.entities if e.name == "Acme Holdings")
    assert "retail arm" not in holdings.description


async def test_stage_three_embeds_the_description_not_just_the_name(
    settings: Settings,
) -> None:
    """A cosine over two bare names cannot see what makes the entities different,
    so the threshold was being asked to separate them on evidence that never
    mentioned the distinction. What distinguishes them lives in the description.
    """
    embedder = _AnchoredEmbedder({"acme group": 0, "acme holdings": 0})
    entities = [
        Entity(
            name="Acme Group",
            type="ORGANIZATION",
            description="Holding company for the retail arm.",
        ),
        Entity(
            name="Acme Holdings",
            type="ORGANIZATION",
            description="Separate entity that owns the real estate.",
        ),
    ]
    await EntityResolver(embedder, settings).resolve(entities)

    assert len(embedder.embedded) == 2, embedder.embedded
    assert all(text not in {"Acme Group", "Acme Holdings"} for text in embedder.embedded), (
        f"stage 3 still compares bare names: {embedder.embedded}"
    )
    assert any("retail arm" in text for text in embedder.embedded), embedder.embedded
    assert any("owns the real estate" in text for text in embedder.embedded), embedder.embedded
    # The name has to stay in front: it is still most of the signal, and the
    # description is the tiebreaker rather than the whole comparison.
    assert {text.split(":")[0] for text in embedder.embedded} == {"Acme Group", "Acme Holdings"}


async def test_stage_three_still_merges_a_variant_that_shares_no_token(
    settings: Settings,
) -> None:
    """Over-splitting is the failure this stage exists to prevent, so the guard
    must be a veto on *disagreement*, not on merging.

    Two shapes have to survive it: the legal-form variant stage 2 collapses
    ("Acme Corp"/"Acme Corporation"), and the pair with no token in common
    ("Facebook"/"FB") — nothing lexical to disagree about, which is exactly the
    case stage 3 was added for.
    """
    embedder = _AnchoredEmbedder({"facebook": 0, "fb": 0})
    entities = [
        Entity(
            name="Acme Corp",
            type="ORGANIZATION",
            description="Widget maker.",
            source_chunk_ids=("c1",),
        ),
        Entity(
            name="Acme Corporation",
            type="ORGANIZATION",
            description="Widget maker.",
            source_chunk_ids=("c2",),
        ),
        Entity(name="Facebook", type="ORGANIZATION", description="The social network."),
        Entity(name="FB", type="ORGANIZATION", description="The social network."),
    ]
    report = await EntityResolver(embedder, settings).resolve(entities)

    names = {e.name for e in report.entities}
    assert names == {"Acme Corporation", "Facebook"}, names
    assert report.embedding_merges == 1, report.summary()
    assert report.normalized_merges == 1, report.summary()
    facebook = next(e for e in report.entities if e.name == "Facebook")
    assert facebook.aliases == ("FB",)
    acme = next(e for e in report.entities if e.name == "Acme Corporation")
    assert set(acme.source_chunk_ids) == {"c1", "c2"}


async def test_veto_is_not_bypassed_by_a_chain_through_a_shorter_name(
    settings: Settings,
) -> None:
    """Union-find is transitive, so refusing only the direct pair is not enough.
    A bare "Acme" is lexically compatible with both qualified names — each is just
    "Acme" plus a qualifier — so merging it into both would reunite Group and
    Holdings by the back door, with no pair having been accepted.
    """
    embedder = _AnchoredEmbedder({"acme": 0})
    entities = [
        Entity(name="Acme", type="ORGANIZATION", description="The Acme family of companies."),
        Entity(
            name="Acme Group",
            type="ORGANIZATION",
            description="Holding company for the retail arm.",
        ),
        Entity(
            name="Acme Holdings",
            type="ORGANIZATION",
            description="Separate entity that owns the real estate.",
        ),
    ]
    report = await EntityResolver(embedder, settings).resolve(entities)

    clusters = [{e.name, *e.aliases} for e in report.entities]
    assert not any({"Acme Group", "Acme Holdings"} <= cluster for cluster in clusters), clusters
    # The bare name may join either one — that ambiguity is in the data — but the
    # two qualified names must not end up as one node.
    assert len(report.entities) == 2, [e.name for e in report.entities]


async def test_embedding_merge_is_logged_with_both_names_and_the_score(
    settings: Settings,
) -> None:
    """``embedding_merges: 2`` does not say which two nodes became one, which is
    what made the audit's bad merge invisible after the fact."""
    from structlog.testing import capture_logs

    embedder = _AnchoredEmbedder({"facebook": 0, "fb": 0})
    entities = [
        Entity(name="Facebook", type="ORGANIZATION"),
        Entity(name="FB", type="ORGANIZATION"),
    ]
    with capture_logs() as logs:
        report = await EntityResolver(embedder, settings).resolve(entities)

    assert report.embedding_merges == 1
    merged = [event for event in logs if event["event"] == "graph_entity_merged"]
    assert len(merged) == 1, logs
    assert {merged[0]["left"], merged[0]["right"]} == {"Facebook", "FB"}
    assert merged[0]["score"] >= settings.graph.resolution_threshold


# ---------------------------------------------------------------------------
# Ingest accounting: a document that produces no chunk
#
# `validate_chunks` drops any chunk under `indexing.min_chunk_size` (64), and a
# document shorter than that has exactly one chunk — so the whole document
# disappears. The audit ingested six short files and got
# `documents_in: 6, indexed: 2, skipped: 0, rejected: 0, failed: 0`: four
# documents unaccounted for, exit 0, empty index. These tests hold the two halves
# of the fix — a counter that makes `documents_in` reconcile, and a log line at a
# level an operator actually sees.
# ---------------------------------------------------------------------------
class _WriteOnlyRelational:
    """The write half of `RelationalStore`, in memory.

    `IngestPipeline` constructs a real `PostgresStore` when `relational_store` is
    None, so the leg has to be injected even for a test that only cares about the
    counters. `tests.fakes.FakeRelationalStore` is read-side only (text-to-SQL and
    full-text search), and this test must not change a fake other suites depend on.
    """

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []

    async def ensure_schema(self) -> None:
        return None

    async def upsert_documents(self, documents: Sequence[Document]) -> int:
        self.documents.extend(documents)
        return len(documents)

    async def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    async def delete_document(self, document_id: str) -> int:
        return 0

    async def close(self) -> None:
        return None


@pytest.fixture
def ingest_settings() -> Settings:
    """Offline ingest: no tenant isolation, no cache, no checksum skip (which
    would need the read side of a relational store), recursive splitter so the
    chunk boundaries are a function of size alone."""
    return Settings(
        environment="dev",
        security={"enforce_tenant_isolation": False},
        cache={"enabled": False},
        llm={"api_key": "test-key"},
        embedding={"dense_dimension": 32},
        indexing={"splitter": "recursive", "skip_unchanged": False},
        # No sparse leg: the ingest pipeline would build the real FastEmbed BM25
        # model from settings, and a unit test must not load an ONNX session (or
        # reach a model host on a cold machine) to count documents.
        retrieval={"use_sparse": False},
    )


def test_validate_chunks_warns_with_the_documents_it_dropped(ingest_settings: Settings) -> None:
    """At debug level this was invisible, and it never said which document lost
    its text — the two properties that made the silent drop unobservable."""
    from structlog.testing import capture_logs

    from ragorc.validate.schema import DocumentValidator

    minimum = ingest_settings.indexing.min_chunk_size
    chunks = [
        Chunk(id="c0", content="too short", document_id="faq-42"),
        Chunk(id="c1", content="!!! ???", document_id="faq-42"),
        Chunk(id="c2", content="x" * (minimum + 10), document_id="handbook"),
    ]
    with capture_logs() as logs:
        kept = DocumentValidator(ingest_settings).validate_chunks(chunks)

    assert [c.id for c in kept] == ["c2"]
    dropped = [event for event in logs if event["event"] == "chunks_dropped"]
    assert len(dropped) == 1, logs
    assert dropped[0]["log_level"] == "warning", "a dropped document must be visible at INFO"
    assert dropped[0]["count"] == 2
    assert dropped[0]["documents"] == ["faq-42"]
    assert dropped[0]["minimum"] == minimum


async def test_ingest_counts_documents_that_produced_no_chunks(
    ingest_settings: Settings, embedder
) -> None:
    """The report must reconcile: every document in is indexed, skipped,
    rejected, duplicated, failed or empty. Before the `documents_empty` counter,
    a short document was none of those and the run still reported success."""
    from ragorc.index.pipeline import IngestPipeline
    from tests.fakes import FakeVectorStore

    minimum = ingest_settings.indexing.min_chunk_size
    short = Document(id="faq-1", content="Yes.", source="faq-1.md")
    real = Document(
        id="handbook",
        content="Expense reports are filed in the portal within 30 days. " * 6,
        source="handbook.md",
    )
    assert len(short.content) < minimum, "the fixture must be under the floor to be a repro"

    pipeline = IngestPipeline(
        settings=ingest_settings,
        dense_embedder=embedder,
        vector_store=FakeVectorStore(),
        relational_store=_WriteOnlyRelational(),
    )
    report = await pipeline.ingest([short, real])

    assert report.documents_in == 2
    assert report.documents_indexed == 1
    assert report.documents_empty == 1, "the dropped document must be counted somewhere"
    assert report.summary()["empty"] == 1, "and it must reach the printed report"
    accounted = (
        report.documents_indexed
        + report.documents_skipped
        + report.documents_rejected
        + report.documents_duplicate
        + report.documents_failed
        + report.documents_empty
    )
    assert accounted == report.documents_in, report.summary()
    assert any("faq-1" in warning for warning in report.warnings), report.warnings
    # An empty document is not a failure and must not be reported as one: the
    # ingest of the other document really did succeed.
    assert report.documents_failed == 0
    assert report.chunks_created >= 1


# ---------------------------------------------------------------------------
# An ingest stage the operator asked for and did not get
# ---------------------------------------------------------------------------
def _stub_pipeline(**settings_kwargs):  # noqa: ANN003, ANN202
    from ragorc.index.pipeline import IngestPipeline
    from tests.fakes import StubEmbedder, StubLLM

    settings_kwargs.setdefault("security", {"enforce_tenant_isolation": False})
    return IngestPipeline(
        llm=StubLLM(), dense_embedder=StubEmbedder(), settings=Settings(**settings_kwargs)
    )


def test_an_enabled_stage_that_cannot_load_is_reported_not_just_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure mode that hid two unreachable features.

    A stage is resolved by factory name, and when no such factory exists it is
    dropped — which used to be a log line the caller never saw: the ingest
    returned success with `llm_calls=0` and an empty warnings list, so nothing
    distinguished "summarised every chunk" from "silently did not run".

    Driven with a deliberately unresolvable stage rather than a real one, so the
    test keeps testing the reporting after the real stages are fixed.
    """
    from ragorc.index import pipeline as pipeline_module
    from ragorc.index.pipeline import IngestReport, _Plugin

    ghost = _Plugin(
        label="multirep",
        modules=("ragorc.index.multirep",),
        factories=("NoSuchFactory", "AlsoMissing"),
    )
    monkeypatch.setattr(pipeline_module, "_OPTIONAL_STAGES", (ghost,))

    pipeline = _stub_pipeline(indexing={"summary_index_enabled": True})
    report = IngestReport()
    built = pipeline._build_stages(report)

    assert built == []
    assert any("multirep" in w and "unavailable" in w for w in report.warnings), report.warnings
    # The hint must name the real cause. "pip install ragorc[...]" would send the
    # operator to fix an install that is not the problem.
    assert any("exists in ragorc.index.multirep" in w for w in report.warnings)


def test_graph_enabled_asks_for_a_second_pass_instead_of_failing_to_build() -> None:
    """Graph construction is corpus-wide; a streaming ingest holds one document.

    It used to be listed as a per-document enrichment stage, where it could never
    work: `GraphBuilder` needs a graph store nothing passed it, exposes only
    `build()`, and returns a build report rather than chunks. Every run logged a
    TypeError and continued. The flag now produces an instruction.
    """
    from ragorc.index.pipeline import _OPTIONAL_STAGES, IngestReport

    assert "graph" not in [p.label for p in _OPTIONAL_STAGES], (
        "graph cannot be a per-document stage: it owns its writes and returns a report"
    )

    pipeline = _stub_pipeline(graph={"enabled": True})
    report = IngestReport()
    built = pipeline._build_stages(report)

    assert built == [], "nothing should be built for a corpus-wide pass"
    assert any("second pass" in w for w in report.warnings), report.warnings
    assert any("GraphBuilder" in w for w in report.warnings), "the warning must name the API"
    assert not any("could not be built" in w for w in report.warnings), (
        "this is a different shape of work, not a build failure"
    )


def test_stages_that_do_work_still_build_and_report_nothing() -> None:
    """The check must not cry wolf: raptor is present and constructible."""
    from ragorc.index.pipeline import IngestReport

    pipeline = _stub_pipeline(indexing={"raptor_enabled": True})
    report = IngestReport()
    built = pipeline._build_stages(report)

    assert [plugin.label for plugin, _ in built] == ["raptor"]
    assert report.warnings == []


async def test_a_directory_is_ingested_a_window_at_a_time(tmp_path) -> None:  # noqa: ANN001
    """The module's memory policy claims a bound independent of corpus size.

    That was true of the chunk stream and false of the document list: loading a
    directory materialized every document's text before the first vector was
    written, and at 100k documents the document list is the larger of the two.
    """
    for i in range(7):
        (tmp_path / f"doc{i}.md").write_text(f"# Document {i}\n\nSome policy text here.\n")

    pipeline = _stub_pipeline(indexing={"document_window": 2})
    seen: list[int] = []
    original = pipeline._validate

    async def _spy(documents, report):  # noqa: ANN001, ANN202
        seen.append(len(documents))
        return await original(documents, report)

    pipeline._validate = _spy  # type: ignore[method-assign]
    await pipeline.ingest(tmp_path)

    assert sum(seen) == 7, f"every document must still be ingested, saw {seen}"
    assert max(seen) <= 2, f"never more than one window resident, saw {seen}"
    assert len(seen) == 4, f"7 documents in windows of 2 is four passes, saw {seen}"


async def test_a_small_directory_still_takes_one_pass(tmp_path) -> None:  # noqa: ANN001
    """The default window is large enough that ordinary corpora are unaffected —
    the streaming is for the case that needs it, not a behaviour change for
    everyone."""
    for i in range(3):
        (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}\n\ntext\n")

    pipeline = _stub_pipeline()
    seen: list[int] = []
    original = pipeline._validate

    async def _spy(documents, report):  # noqa: ANN001, ANN202
        seen.append(len(documents))
        return await original(documents, report)

    pipeline._validate = _spy  # type: ignore[method-assign]
    report = await pipeline.ingest(tmp_path)

    assert seen == [3], f"one window for a small corpus, saw {seen}"
    assert report.documents_in == 3


async def test_an_empty_directory_still_reports_nothing_to_do(tmp_path) -> None:  # noqa: ANN001
    """The early-return outcomes must survive the rewrite of the loop."""
    report = await _stub_pipeline().ingest(tmp_path)
    assert report.documents_in == 0
    assert report.strategy == "auto", "the strategy is never resolved when there is no work"


# ---------------------------------------------------------------------------
# Multi-representation indexing as an ingest stage
# ---------------------------------------------------------------------------
async def test_multirep_returns_its_units_and_writes_none_of_them() -> None:
    """Write ownership is the whole design question, and it has one answer.

    `_process_document` embeds the leaf chunks, calls the stage, and then writes
    whatever comes back. An indexer that also upserted would write the same ids
    twice and race the pipeline's own vectors, so the façade withholds
    `vector_store` — the indexers build and embed, and the pipeline writes.
    """
    from ragorc.core.models import Chunk, Document
    from ragorc.index.multirep import MultiRepresentationIndexer
    from tests.fakes import StubEmbedder, StubLLM

    class _Recording:
        """Records both write channels so the test can tell them apart."""

        def __init__(self) -> None:
            self.vector_upserts: list[Any] = []
            self.docstore_writes: list[Any] = []

        async def upsert(self, chunks: Any) -> None:
            self.vector_upserts.append(list(chunks))

        async def upsert_chunks(self, chunks: Any) -> None:
            self.docstore_writes.append(list(chunks))

    store = _Recording()
    document = Document(id="d1", content="Refunds take five days. " * 40)
    chunks = [
        Chunk(
            id=f"c{i}",
            content="Refunds are processed within five business days. " * 40,
            document_id="d1",
        )
        for i in range(3)
    ]

    stage = MultiRepresentationIndexer(
        StubLLM(),
        embedder=StubEmbedder(),
        settings=Settings(
            indexing={"summary_index_enabled": True},
            security={"enforce_tenant_isolation": False},
        ),
    )
    produced, usage = await stage.enrich(document, chunks, relational_store=store)

    assert produced, "the stage must hand back the units it built"
    assert usage.calls, "summarising costs model calls and they must be billed"
    assert store.vector_upserts == [], "a vector write here duplicates the pipeline's"
    # The docstore write is not a duplicate: the derived unit replaces its source
    # in the vector store, and expand_parents reads that source back at query
    # time. Without it, retrieval returns units whose sources do not exist.
    assert store.docstore_writes, "the sources must still be persisted for expansion"


async def test_multirep_is_a_no_op_when_no_flag_is_set() -> None:
    """It is registered unconditionally, so it must cost nothing when unused."""
    from ragorc.core.models import Chunk, Document
    from ragorc.index.multirep import MultiRepresentationIndexer
    from tests.fakes import StubEmbedder, StubLLM

    llm = StubLLM()
    chunks = [Chunk(id="c1", content="text", document_id="d1")]
    stage = MultiRepresentationIndexer(
        llm,
        embedder=StubEmbedder(),
        settings=Settings(security={"enforce_tenant_isolation": False}),
    )
    produced, usage = await stage.enrich(Document(id="d1", content="text"), chunks)

    assert [c.id for c in produced] == ["c1"], "the chunks must pass through untouched"
    assert usage.calls == 0
    assert llm.calls == []


async def test_parent_document_says_so_rather_than_half_running() -> None:
    """It re-splits the document, so running it as an enrichment would index every
    document twice — once as the pipeline's chunks, once as its own children."""
    from ragorc.core.models import Chunk, Document
    from ragorc.index.multirep import MultiRepresentationIndexer
    from tests.fakes import StubEmbedder, StubLLM

    llm = StubLLM()
    chunks = [Chunk(id="c1", content="text", document_id="d1")]
    stage = MultiRepresentationIndexer(
        llm,
        embedder=StubEmbedder(),
        settings=Settings(
            indexing={"parent_document_enabled": True},
            security={"enforce_tenant_isolation": False},
        ),
    )
    produced, usage = await stage.enrich(Document(id="d1", content="text"), chunks)

    assert [c.id for c in produced] == ["c1"], "nothing is re-chunked here"
    assert usage.calls == 0, "and nothing is spent pretending to"
