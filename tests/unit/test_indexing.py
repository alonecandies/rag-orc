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


async def test_parent_document_replaces_the_split_and_persists_the_parents(
    tmp_path,  # noqa: ANN001
) -> None:
    """Parent-document indexing is a chunking mode, not an enrichment.

    It splits the document twice — parents with no overlap, then children inside
    each parent — and the child is the retrieval unit while the parent is the
    generation unit. Running it as an enrichment beside the normal split would
    index every document twice, so it replaces the split instead: children go to
    the vector store, parents to the docstore alone, because nothing searches a
    parent and the default 2048/256 sizes would store the corpus eight to ten
    times over in the payload.

    The parents are *queued* during processing and written by the flush, because
    `chunks.document_id` is a foreign key to a row `_run` writes a step later —
    see `_DeferredDocstore`. That is asserted here too: it is the ordering, not
    just the content, that the real schema enforces.
    """
    from ragorc.core.models import Document

    persisted: list[list[Any]] = []

    class _Docstore:
        async def upsert_chunks(self, chunks: Any) -> None:
            persisted.append(list(chunks))

        async def ensure_schema(self) -> None:
            return None

    pipeline = _stub_pipeline(indexing={"parent_document_enabled": True})
    pipeline.relational = _Docstore()  # type: ignore[assignment]

    document = Document(
        id="d1",
        content="\n\n".join(f"Section {i}. " + "Refund policy detail. " * 40 for i in range(3)),
    )
    from ragorc.core.models import ChunkingStrategy
    from ragorc.index.pipeline import IngestReport

    report = IngestReport()
    children = await pipeline._process_document(document, ChunkingStrategy.EARLY, report)

    assert children, "the children are what gets indexed"
    assert all(c.parent_id for c in children), "every child must point at its parent"
    assert persisted == [], (
        "nothing may reach the chunks table while the document row does not exist"
    )
    assert pipeline._deferred, "the parents must be queued for the flush"

    await pipeline._flush_deferred(report)

    assert persisted, "the parents must be persisted for query-time expansion"
    parent_ids = {p.id for batch in persisted for p in batch}
    assert {c.parent_id for c in children} <= parent_ids, (
        "every child's parent must be among the persisted parents"
    )
    assert all(p.dense is None for batch in persisted for p in batch), (
        "a parent carries no vector: nothing ever searches one"
    )
    assert pipeline._deferred == [], "a flushed buffer must not write the same rows twice"


# ---------------------------------------------------------------------------
# The collection is declared from the embedders, so they must exist first
# ---------------------------------------------------------------------------
def _stub_providers(monkeypatch: pytest.MonkeyPatch, pipeline: Any, *, colbert_dim: int) -> None:
    """Route the pipeline's lazy provider lookups to stubs.

    Without this the pipeline resolves `fastembed` for real, which loads an ONNX
    session and may reach a model host — neither belongs in a unit test.
    """
    from tests.fakes import StubLateInteractionEmbedder, StubSparseEmbedder

    class _Late(StubLateInteractionEmbedder):
        def __init__(self, **_: Any) -> None:
            super().__init__(dimension=colbert_dim, max_tokens=40)

    class _Sparse(StubSparseEmbedder):
        def __init__(self, **_: Any) -> None:
            super().__init__(is_lexical=True)

    classes = {"sparse_embedder": _Sparse, "late_interaction_embedder": _Late}
    monkeypatch.setattr(pipeline, "_provider_class", lambda kind, provider: classes[kind])


async def test_the_collection_is_declared_from_the_real_embedders_not_a_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_prepare` creates the collection, so every embedder that shapes it must
    already be built when it runs.

    The sparse and ColBERT embedders were built lazily, on first use — which is
    inside `_process_document`, *after* the collection exists. So with
    `enable_late_interaction` on and no hand-injected embedder, `_colbert_dim()`
    saw `None` and fell back to ColBERTv2's 128. The configured model decides the
    real width, and `answerdotai/answerai-colbert-small-v1` is 96: the collection
    was then declared 96 wide short of what every upsert would send, and Qdrant
    rejected the whole batch. `_pin_dimension` already forces a real forward pass
    for the dense vector for exactly this reason; the other two named vectors need
    the same discipline.

    `is_lexical` rides along: it decides the IDF modifier, and read off `None` it
    is guessed from `use_splade` instead of from the provider that will actually
    produce the vectors.
    """
    from ragorc.index.pipeline import IngestPipeline
    from tests.fakes import StubEmbedder

    seen: dict[str, Any] = {}

    async def spy(self: Any, query_side: Any, dimension: int) -> None:
        seen["late"] = self.late_embedder
        seen["sparse"] = self.sparse_embedder

    monkeypatch.setattr(IngestPipeline, "_ensure_stores", spy)
    pipeline = IngestPipeline(
        dense_embedder=StubEmbedder(dimension=32),
        settings=Settings(
            security={"enforce_tenant_isolation": False},
            cache={"enabled": False},
            embedding={"dense_dimension": 32, "enable_late_interaction": True},
            indexing={"splitter": "recursive", "skip_unchanged": False},
            retrieval={"use_sparse": True},
        ),
    )
    _stub_providers(monkeypatch, pipeline, colbert_dim=96)

    await pipeline._prepare()

    assert seen["sparse"] is not None, "the sparse leg decides the IDF modifier"
    assert seen["late"] is not None, "the ColBERT leg decides the multivector width"
    assert seen["late"].dimension == 96

    # The consequence, at the seam that suffered it. This is the value the
    # collection is created with; 128 here is a collection no upsert can satisfy.
    from ragorc.stores.qdrant.store import QdrantStore

    guessed = QdrantStore(pipeline.settings, dense_embedder=StubEmbedder(dimension=32))
    assert guessed._colbert_dim() == 128, "the fallback guess, for contrast"
    sized = QdrantStore(
        pipeline.settings,
        dense_embedder=StubEmbedder(dimension=32),
        late_embedder=seen["late"],
    )
    assert sized._colbert_dim() == 96, "what the pipeline must hand it"


# ---------------------------------------------------------------------------
# The ingest path must use the ColBERT indexer, not re-implement it
# ---------------------------------------------------------------------------
async def test_ingest_colbert_vectors_are_pruned_and_carry_the_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_add_colbert` re-implemented `ColBERTIndexer.index` in five lines.

    What the re-implementation dropped: token pruning, so
    `late_interaction_max_tokens` bounded nothing and a field already ~100x a
    dense vector grew without limit; batching, so one request carried a whole
    window; the dimension check, which turns a server-side rejection of the entire
    upsert into a message naming both widths; and `embed_text`, so ColBERT alone
    among the three vectors indexed the chunk without its contextual prefix.
    """
    import numpy as np

    from ragorc.core.models import ChunkingStrategy, Document
    from ragorc.index.pipeline import IngestPipeline, IngestReport
    from tests.fakes import FakeVectorStore, StubEmbedder

    cap = 6

    class _RecordingLate:
        """One token row per whitespace-separated word, so length is predictable."""

        dimension = 8
        model_name = "stub/colbert"

        def __init__(self, **_: Any) -> None:
            self.seen: list[str] = []

        async def embed_documents(self, texts: Any) -> list[Any]:
            self.seen.extend(texts)
            return [
                np.ones((max(1, len(t.split())), self.dimension), dtype=np.float32) for t in texts
            ]

    late = _RecordingLate()
    pipeline = IngestPipeline(
        dense_embedder=StubEmbedder(dimension=32),
        late_embedder=late,
        vector_store=FakeVectorStore(),
        settings=Settings(
            security={"enforce_tenant_isolation": False},
            cache={"enabled": False},
            embedding={
                "dense_dimension": 32,
                "enable_late_interaction": True,
                "late_interaction_max_tokens": cap,
            },
            indexing={"splitter": "recursive", "skip_unchanged": False},
            retrieval={"use_sparse": False},
        ),
    )

    document = Document(id="d1", content="Refunds are processed within five days. " * 40)
    chunks = await pipeline._process_document(document, ChunkingStrategy.EARLY, IngestReport())

    assert chunks, "the fixture must produce chunks to have anything to assert on"
    assert all(c.multi is not None for c in chunks), "late interaction is on"
    widths = {int(c.multi.shape[0]) for c in chunks if c.multi is not None}
    assert widths and max(widths) <= cap, (
        f"every matrix must be pruned to late_interaction_max_tokens={cap}, saw {sorted(widths)}"
    )

    # And the prefix reaches it. Set one by hand: the enricher is an LLM call, and
    # the property under test is which attribute the indexer reads.
    marker = "SITUATED IN THE REFUNDS POLICY"
    chunk = chunks[0]
    chunk.multi = None
    chunk.contextual_prefix = marker
    late.seen.clear()
    await pipeline._add_colbert([chunk], IngestReport())

    assert late.seen, "the indexer must have re-embedded the chunk it had no matrix for"
    assert any(marker in text for text in late.seen), (
        "ColBERT must index embed_text like dense and sparse do, not bare content"
    )


async def test_a_chunk_that_already_has_a_matrix_is_not_re_embedded() -> None:
    """The resume property `skip_unchanged` gives the document level, at the chunk
    level: a re-run of an interrupted ingest must not re-pay for the matrices it
    already has. The inline version re-embedded every chunk unconditionally."""
    import numpy as np

    from ragorc.core.models import Chunk
    from ragorc.index.pipeline import IngestPipeline, IngestReport
    from tests.fakes import StubEmbedder

    class _Counting:
        dimension = 8
        model_name = "stub/colbert"

        def __init__(self) -> None:
            self.batches = 0

        async def embed_documents(self, texts: Any) -> list[Any]:
            self.batches += 1
            return [np.ones((2, self.dimension), dtype=np.float32) for _ in texts]

    late = _Counting()
    pipeline = IngestPipeline(
        dense_embedder=StubEmbedder(dimension=32),
        late_embedder=late,
        settings=Settings(
            security={"enforce_tenant_isolation": False},
            cache={"enabled": False},
            embedding={"dense_dimension": 32, "enable_late_interaction": True},
            retrieval={"use_sparse": False},
        ),
    )
    done = Chunk(id="c1", content="already indexed", document_id="d1")
    done.multi = np.ones((2, 8), dtype=np.float32)
    todo = Chunk(id="c2", content="not yet indexed", document_id="d1")

    await pipeline._add_colbert([done, todo], IngestReport())

    assert late.batches == 1, "one batch for the one chunk that needed it"
    assert todo.multi is not None

    late.batches = 0
    await pipeline._add_colbert([done, todo], IngestReport())
    assert late.batches == 0, "nothing left to do must cost nothing"


# ---------------------------------------------------------------------------
# Derived units are searched the same way leaf chunks are
# ---------------------------------------------------------------------------
async def test_derived_units_carry_every_vector_the_collection_declares() -> None:
    """A summary unit replaces its source as the retrieval target, so it has to be
    reachable by every leg the collection declares.

    Sparse and ColBERT ran before the enrichment stage that creates these units,
    so they carried the dense vector their indexer computed and nothing else: on a
    hybrid collection they were findable by vector search and invisible to BM25 —
    silently half-indexed, which is worse than absent because the recall gap has
    no symptom.
    """
    from ragorc.core.models import ChunkingStrategy, Document
    from ragorc.index.pipeline import IngestPipeline, IngestReport
    from tests.fakes import FakeVectorStore, StubEmbedder, StubLLM, StubSparseEmbedder

    class _Docstore:
        async def upsert_chunks(self, chunks: Any) -> None: ...

    pipeline = IngestPipeline(
        llm=StubLLM(),
        dense_embedder=StubEmbedder(dimension=32),
        sparse_embedder=StubSparseEmbedder(),
        vector_store=FakeVectorStore(),
        relational_store=_Docstore(),
        settings=Settings(
            security={"enforce_tenant_isolation": False},
            cache={"enabled": False},
            embedding={"dense_dimension": 32},
            indexing={
                "splitter": "recursive",
                "skip_unchanged": False,
                "summary_index_enabled": True,
            },
            retrieval={"use_sparse": True},
        ),
    )
    pipeline._stages = pipeline._build_stages(IngestReport())
    assert pipeline._stages, "the multirep stage must be loadable for this to test anything"

    document = Document(id="d1", content="Refunds are processed within five business days. " * 60)
    chunks = await pipeline._process_document(document, ChunkingStrategy.EARLY, IngestReport())

    derived = [c for c in chunks if c.metadata.get("representation") == "summary"]
    assert derived, "the stage must have produced summary units to assert on"
    assert all(c.dense is not None for c in derived), "the indexer computes this one"
    assert all(c.sparse is not None and len(c.sparse) for c in derived), (
        "a derived unit invisible to BM25 is half-indexed on a hybrid collection"
    )


# ---------------------------------------------------------------------------
# Nothing may reach the chunks table before its document row
# ---------------------------------------------------------------------------
def _fk_pipeline(store: Any, **indexing: Any) -> Any:
    from ragorc.index.pipeline import IngestPipeline
    from tests.fakes import FakeVectorStore, StubEmbedder, StubLLM

    return IngestPipeline(
        llm=StubLLM(),
        dense_embedder=StubEmbedder(dimension=32),
        vector_store=FakeVectorStore(),
        relational_store=store,
        settings=Settings(
            security={"enforce_tenant_isolation": False},
            cache={"enabled": False},
            embedding={"dense_dimension": 32},
            indexing={"splitter": "recursive", "skip_unchanged": False, **indexing},
            retrieval={"use_sparse": False},
        ),
    )


async def test_parents_are_not_written_before_the_document_row_exists() -> None:
    """The docstore write ran inside `_process_document`, and the document row is
    written a step later in `_run`.

    So on a fresh corpus every parent insert referenced a row that did not exist:
    `ForeignKeyViolation`, caught as a per-document failure, and a 10k-document
    ingest returned `indexed: 0, failed: 10000` after paying for every embedding.
    No unit test saw it because the doubles had no foreign key.
    """
    from tests.fakes import FakeDocumentStore

    store = FakeDocumentStore()
    pipeline = _fk_pipeline(store, parent_document_enabled=True)
    document = Document(
        id="handbook",
        content="\n\n".join(f"Section {i}. " + "Refund policy detail. " * 40 for i in range(3)),
    )

    report = await pipeline.ingest([document])

    assert report.documents_failed == 0, report.warnings
    assert report.documents_indexed == 1
    assert store.documents, "the document row must exist"
    parents = [c for c in store.chunks.values() if c.dense is None]
    assert parents, "the parents must survive in the docstore for expansion to find them"


async def test_the_stale_purge_does_not_cascade_away_the_parents_just_written() -> None:
    """Second half of the same ordering bug, on a corpus that already exists.

    The purge deliberately runs *after* the replacement chunks are built, so a
    document whose extraction just started failing keeps the vectors it had. But
    `delete_document` cascades to every chunk with that `document_id`, and the
    parents had already been written during processing — so re-ingesting a changed
    document deleted its parents and left the children pointing at nothing.
    """
    from tests.fakes import FakeDocumentStore

    store = FakeDocumentStore()
    pipeline = _fk_pipeline(store, parent_document_enabled=True, skip_unchanged=True)
    body = "\n\n".join(f"Section {i}. " + "Refund policy detail. " * 40 for i in range(3))

    first = await pipeline.ingest([Document(id="handbook", content=body)])
    assert first.documents_indexed == 1, first.warnings

    edited = await pipeline.ingest([Document(id="handbook", content=body + "\n\nAddendum. " * 30)])

    assert edited.documents_indexed == 1, edited.warnings
    parents = [c for c in store.chunks.values() if c.dense is None]
    assert parents, "the purge cascade must not outlive the parents it precedes"
    children = [c for c in store.chunks.values() if c.parent_id]
    parent_ids = {p.id for p in parents}
    assert children, "the fixture must produce children to have anything to check"
    assert {c.parent_id for c in children} <= parent_ids, (
        "every child must still resolve to a parent that exists"
    )


async def test_summary_sources_are_not_written_before_the_document_row() -> None:
    """The multirep stage persists the sources it replaces, from inside `_enrich`
    — also before the document row. The violation was swallowed one level up as
    "stage disabled for the rest of this run", so `summary_index_enabled` turned
    itself off on the first document and the run still reported success."""
    from tests.fakes import FakeDocumentStore

    store = FakeDocumentStore()
    pipeline = _fk_pipeline(store, summary_index_enabled=True)
    document = Document(id="d1", content="Refunds are processed within five business days. " * 60)

    report = await pipeline.ingest([document])

    assert report.documents_failed == 0, report.warnings
    assert not any("disabled" in w for w in report.warnings), report.warnings
    assert store.chunks, "the summarised sources must be persisted for expansion"


# ---------------------------------------------------------------------------
# Report counters under a windowed ingest
# ---------------------------------------------------------------------------
async def test_windowed_ingest_accumulates_the_validation_counters(tmp_path: Any) -> None:
    """`_validate` runs once per document window, and three of its report fields
    assigned where every sibling accumulates.

    So a directory ingest reported the *last* window's rejects and duplicates
    while `report.rejected` carried all of them: the count and the list
    contradicted each other, and `documents_in` stopped reconciling with the sum
    of the outcome counters — the one invariant the report exists to hold.
    """
    from ragorc.validate.schema import DocumentValidator
    from tests.fakes import FakeDocumentStore

    # Two windows, each holding one document validation rejects and one it
    # accepts. The size limit is the deterministic rejection: the binary heuristic
    # runs *after* control characters are substituted out, so a file of NUL bytes
    # reaches it looking like ordinary words.
    sentence = "Refunds are processed within five business days. "
    for i in range(4):
        (tmp_path / f"doc-{i}.md").write_text(sentence * (20 if i % 2 else 60))

    store = FakeDocumentStore()
    pipeline = _fk_pipeline(store)
    pipeline.settings.indexing.document_window = 2
    pipeline.validator = DocumentValidator(pipeline.settings, max_bytes=1500)

    report = await pipeline.ingest(tmp_path)

    assert report.documents_in == 4
    assert len(report.rejected) == 2, report.rejected
    assert report.documents_rejected == len(report.rejected), (
        "the count and the list are the same fact and must agree across windows"
    )
    accounted = (
        report.documents_indexed
        + report.documents_skipped
        + report.documents_rejected
        + report.documents_duplicate
        + report.documents_failed
        + report.documents_empty
    )
    assert accounted == report.documents_in, report.summary()


# ---------------------------------------------------------------------------
# "Did it land?" must be answered by the store, not by the sender
# ---------------------------------------------------------------------------
async def test_ingest_flushes_the_vector_store_and_reports_what_it_holds() -> None:
    """`qdrant.wait_on_upsert` is False, so an upsert returns once the write is
    accepted — not once its points are searchable.

    Two docstrings promised "one final waiting flush" and no such call existed
    anywhere, so the run reported the vectors it *sent*. That is the dangerous
    number to report, because the commit marker for "this document is ingested"
    is the Postgres chunk rows: a collection that never finished applying its
    points looked exactly like a good run, and the next run skipped those
    documents as already done.

    Run with a sparse leg on purpose: a chunk is one *point* carrying two named
    vectors there, so the read-back and `vectors_written` differ by construction.
    Comparing those two numbers is what the first version of this check did, and
    it reported a shortfall on every healthy hybrid run.
    """
    from tests.fakes import FakeDocumentStore, FakeVectorStore, StubSparseEmbedder

    class _Flushing(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.flushes = 0

        async def flush(self, *, timeout_s: float = 300.0) -> int:
            self.flushes += 1
            return len(self.chunks)

    store = _Flushing()
    pipeline = _fk_pipeline(FakeDocumentStore())
    pipeline.vector = store
    pipeline.sparse_embedder = StubSparseEmbedder()
    pipeline.settings.retrieval.use_sparse = True
    document = Document(id="d1", content="Refunds are processed within five business days. " * 60)

    report = await pipeline.ingest([document])

    assert store.flushes == 1, "exactly one barrier per run, not one per batch"
    assert report.points_in_store == len(store.chunks)
    assert report.points_in_store == report.chunks_created, (
        "a chunk is one point; into an empty collection the counts must agree"
    )
    assert report.vectors_written > report.chunks_created, (
        "the fixture must be a multi-vector collection, or the check below is vacuous"
    )
    assert not [w for w in report.warnings if "holds" in w], (
        "vectors_written counts named vectors and points_in_store counts points; "
        f"comparing them warns on every healthy hybrid run: {report.warnings}"
    )
    assert report.summary()["points_in_store"] == report.points_in_store, (
        "the read-back is the operator's cross-check and has to reach the report"
    )


async def test_a_flush_that_fails_is_reported_not_swallowed() -> None:
    """The flush *is* the check, so losing it silently defeats its purpose. It is
    not fatal either — the writes were accepted; only the confirmation failed."""
    from tests.fakes import FakeDocumentStore, FakeVectorStore

    class _Unflushable(FakeVectorStore):
        async def flush(self, *, timeout_s: float = 300.0) -> int:
            raise TimeoutError("collection never went green")

    pipeline = _fk_pipeline(FakeDocumentStore())
    pipeline.vector = _Unflushable()
    document = Document(id="d1", content="Refunds are processed within five business days. " * 60)

    report = await pipeline.ingest([document])

    assert report.documents_indexed == 1, "a failed confirmation is not a failed ingest"
    assert report.points_in_store is None, "unknown must not read as zero"
    assert any("could not confirm" in w for w in report.warnings), report.warnings


async def test_bulk_load_is_entered_once_for_the_whole_run_not_once_per_window() -> None:
    """Bulk-load mode turns HNSW construction off, and its *exit* builds every
    graph once over segments whose final size is known — the single biggest ingest
    speedup available.

    It was entered inside `_run`, which is called once per document window, so a
    directory ingest toggled indexing off and back on once per `document_window`
    documents and every exit rebuilt the graph over everything written so far and
    then waited for green. That is precisely the repeated rebuilding the mode
    exists to prevent, put on a schedule.
    """
    import contextlib

    from tests.fakes import FakeDocumentStore, FakeVectorStore

    class _BulkLoading(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.entries = 0

        @contextlib.asynccontextmanager
        async def bulk_load(self) -> Any:
            self.entries += 1
            yield

        async def flush(self, *, timeout_s: float = 300.0) -> int:
            return len(self.chunks)

    store = _BulkLoading()
    pipeline = _fk_pipeline(FakeDocumentStore())
    pipeline.vector = store
    # Three windows' worth of documents, each window over the bulk-load floor.
    pipeline.settings.indexing.document_window = 70
    body = "Refunds are processed within five business days. " * 12
    documents = [Document(id=f"d{i}", content=f"Document {i}. {body}") for i in range(210)]

    report = await pipeline.ingest(documents)

    assert report.documents_indexed == 210, report.warnings
    assert store.entries == 1, (
        f"one bulk-load window for the run, not one per document window (saw {store.entries})"
    )


async def test_a_small_ingest_does_not_pay_for_bulk_load_mode() -> None:
    """Turning indexing off costs two round trips and a wait-for-green on exit,
    which is not worth paying to insert a handful of points."""
    import contextlib

    from ragorc.index.pipeline import BULK_LOAD_MIN_DOCUMENTS
    from tests.fakes import FakeDocumentStore, FakeVectorStore

    class _BulkLoading(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.entries = 0

        @contextlib.asynccontextmanager
        async def bulk_load(self) -> Any:
            self.entries += 1
            yield

    store = _BulkLoading()
    pipeline = _fk_pipeline(FakeDocumentStore())
    pipeline.vector = store
    body = "Refunds are processed within five business days. " * 12
    few = [Document(id=f"d{i}", content=f"Document {i}. {body}") for i in range(4)]
    assert len(few) < BULK_LOAD_MIN_DOCUMENTS

    await pipeline.ingest(few)

    assert store.entries == 0
