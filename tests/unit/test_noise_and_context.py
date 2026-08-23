"""Retrieval noise handling and context management."""

from __future__ import annotations

import numpy as np
import pytest

from ragorc.context.budget import ContextBudgeter
from ragorc.context.pack import ContextPacker, reorder_lost_in_middle
from ragorc.core.models import Chunk, ScoredChunk
from ragorc.core.settings import Settings
from ragorc.retrieve.noise import (
    NoiseFilter,
    mmr_select,
    normalize_scores,
    simhash,
)


def sc(cid: str, text: str, score: float, vector: list[float] | None = None) -> ScoredChunk:
    chunk = Chunk(id=cid, content=text)
    if vector is not None:
        chunk.dense = np.asarray(vector, dtype=np.float32)
    return ScoredChunk(chunk=chunk, score=score)


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------
def test_simhash_is_stable_under_cosmetic_edits() -> None:
    a = simhash("The cat sat on the mat quietly")
    b = simhash("The cat sat on the  mat quietly!")
    assert bin(a ^ b).count("1") <= 4, "whitespace and punctuation must barely move the hash"


def test_simhash_separates_unrelated_text() -> None:
    a = simhash("The cat sat on the mat")
    b = simhash("Quantum error correction thresholds in surface codes")
    assert bin(a ^ b).count("1") > 15


def test_simhash_handles_empty_and_short() -> None:
    assert simhash("") == 0
    assert simhash("hi") != 0


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------
def test_normalize_minmax_spans_unit_interval() -> None:
    out = normalize_scores([sc("a", "x", 10.0), sc("b", "y", 5.0), sc("c", "z", 0.0)])
    assert [round(c.score, 3) for c in out] == [1.0, 0.5, 0.0]


def test_normalize_keeps_raw_score_for_audit() -> None:
    out = normalize_scores([sc("a", "x", 42.0), sc("b", "y", 1.0)])
    assert any("raw_" in k for k in out[0].component_scores)


def test_normalize_handles_identical_scores() -> None:
    out = normalize_scores([sc("a", "x", 3.0), sc("b", "y", 3.0)])
    assert all(c.score == 1.0 for c in out), "a zero-width range must not divide by zero"


# ---------------------------------------------------------------------------
# Dedupe / thresholding
# ---------------------------------------------------------------------------
@pytest.fixture
def noise_settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={
            "top_k": 10,
            "relative_score_cutoff": 0.35,
            "dedupe_enabled": True,
            "near_dupe_threshold": 0.9,
            "mmr_enabled": False,
        },
    )


def test_exact_and_normalized_duplicates_collapse(noise_settings: Settings) -> None:
    """Both duplicates go, but not at the same stage — which stage matters.

    ``c`` is the same text under a new id, so exact dedupe collapses it. ``b``
    differs by punctuation, and *exact* dedupe deliberately no longer claims that
    is the same passage (it cannot tell ``a == b`` from ``a != b`` if it ignores
    operators); the word-shingled near-duplicate stage removes it instead, which
    is the stage that is allowed to be fuzzy.
    """
    items = [
        sc("a", "Refunds are processed within 14 days.", 0.90),
        sc("b", "Refunds  are processed within 14 days!", 0.88),  # punctuation only
        sc("c", "Refunds are processed within 14 days.", 0.70),  # identical text, new id
        sc("d", "Shipping takes 3-5 business days.", 0.61),
    ]
    kept, report = NoiseFilter(noise_settings).apply(items)
    assert [k.chunk.id for k in kept] == ["a", "d"]
    assert report.exact_duplicates == 1
    assert report.near_duplicates == 1


def test_duplicate_merges_provenance(noise_settings: Settings) -> None:
    """'found by both dense and sparse' must survive deduplication."""
    first = sc("a", "same text here", 0.9)
    first.component_scores = {"dense": 0.9}
    second = sc("b", "same text here", 0.8)
    second.component_scores = {"sparse": 0.8}
    kept, _ = NoiseFilter(noise_settings).apply([first, second])
    assert len(kept) == 1
    assert set(kept[0].component_scores) == {"dense", "sparse"}


def test_relative_cutoff_adapts_to_the_query(noise_settings: Settings) -> None:
    """A 0.14 hit is noise beside a 0.42 top hit, and fine beside a 0.20 one."""
    strong = [sc("a", "alpha text", 0.42), sc("b", "beta text", 0.14)]
    kept, report = NoiseFilter(noise_settings).apply(strong)
    assert [k.chunk.id for k in kept] == ["a"]
    assert report.below_threshold == 1

    flat = [sc("a", "alpha text", 0.20), sc("b", "beta text", 0.14)]
    kept2, _ = NoiseFilter(noise_settings).apply(flat)
    assert len(kept2) == 2, "the same absolute score survives a flatter distribution"


def test_near_dupe_uses_vectors_when_present(noise_settings: Settings) -> None:
    items = [
        sc("a", "completely different words alpha", 0.9, [1.0, 0.0, 0.0]),
        sc("b", "totally unrelated tokens beta", 0.8, [0.99, 0.14, 0.0]),  # cos ~0.99
        sc("c", "third distinct entry gamma", 0.7, [0.0, 1.0, 0.0]),
    ]
    kept, report = NoiseFilter(noise_settings).apply(items)
    assert report.near_duplicates == 1
    assert {k.chunk.id for k in kept} == {"a", "c"}


def test_noise_filter_assigns_contiguous_ranks(noise_settings: Settings) -> None:
    items = [sc(str(i), f"unique text number {i}", 0.9 - i * 0.05) for i in range(5)]
    kept, _ = NoiseFilter(noise_settings).apply(items)
    assert [k.rank for k in kept] == list(range(len(kept)))


# ---------------------------------------------------------------------------
# MMR
# ---------------------------------------------------------------------------
def test_mmr_surfaces_a_second_topic() -> None:
    items = [sc(f"t{i}", f"topic one variant {i}", 0.9 - 0.01 * i, [1.0, 0.0]) for i in range(3)]
    items.append(sc("other", "a different topic entirely", 0.5, [0.0, 1.0]))
    picked = mmr_select(items, k=2, lambda_mult=0.5)
    assert {p.chunk.id for p in picked} == {"t0", "other"}


def test_mmr_pure_relevance_when_lambda_is_one() -> None:
    items = [sc(f"t{i}", f"text {i}", 0.9 - 0.01 * i, [1.0, 0.0]) for i in range(3)]
    items.append(sc("other", "different", 0.5, [0.0, 1.0]))
    picked = mmr_select(items, k=2, lambda_mult=1.0)
    assert {p.chunk.id for p in picked} == {"t0", "t1"}


def test_mmr_falls_back_without_vectors() -> None:
    items = [sc("a", "x", 0.9), sc("b", "y", 0.8), sc("c", "z", 0.7)]
    picked = mmr_select(items, k=2)
    assert [p.chunk.id for p in picked] == ["a", "b"], "no vectors means no diversity signal"


# ---------------------------------------------------------------------------
# Lost in the middle
# ---------------------------------------------------------------------------
def test_reorder_puts_best_at_both_ends() -> None:
    items = [sc(str(i), f"text {i}", 1.0 - i * 0.1) for i in range(6)]
    order = [c.chunk.id for c in reorder_lost_in_middle(items)]
    assert order == ["0", "2", "4", "5", "3", "1"]
    assert order[0] == "0", "rank 0 must be first"
    assert order[-1] == "1", "rank 1 must be last"


def test_reorder_is_a_noop_for_short_lists() -> None:
    items = [sc("a", "x", 0.9), sc("b", "y", 0.8)]
    assert [c.chunk.id for c in reorder_lost_in_middle(items)] == ["a", "b"]


# ---------------------------------------------------------------------------
# Budget & packing
# ---------------------------------------------------------------------------
def test_budget_reserves_output_and_splits_by_share() -> None:
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        llm={"context_window": 10_000},
        generation={"reserved_output_tokens": 1_000, "max_answer_tokens": 500},
    )
    plan = ContextBudgeter(settings).plan(system_prompt="s" * 400, question="q" * 40)
    assert plan.budget.available_context < 9_000
    assert plan.per_source["vector"] > plan.per_source["web"]


def test_budget_strategy_escalates_with_overflow() -> None:
    """The threshold is a ratio, so the fixture is sized from the real budget
    rather than from an assumption about the tokenizer."""
    settings = Settings(security={"enforce_tenant_isolation": False}, llm={"context_window": 2_000})
    budgeter = ContextBudgeter(settings)
    plan = budgeter.plan()
    available = plan.budget.available_context

    def chunks_totalling(target_tokens: int, n: int) -> list[ScoredChunk]:
        per_chunk = max(target_tokens // n, 1)
        return [sc(str(i), "word " * per_chunk, 0.9) for i in range(n)]

    assert budgeter.decide_strategy(chunks_totalling(available // 2, 2), plan) == "fit"
    # Modest overflow (<=1.6x) drops the weak tail; larger overflow compresses,
    # because the tail is then too much evidence to discard.
    assert budgeter.decide_strategy(chunks_totalling(int(available * 1.3), 3), plan) == "truncate"
    assert budgeter.decide_strategy(chunks_totalling(int(available * 4), 8), plan) == "summarize"


def test_packer_selects_by_relevance_per_token() -> None:
    """A big mediocre chunk must not displace several small strong ones."""
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"reorder_lost_in_middle": False, "parent_expansion": False},
    )
    packer = ContextPacker(settings)
    items = [
        sc("top", "the top hit " * 10, 0.95),
        sc("fat", "bulky filler text " * 200, 0.81),
        sc("lean1", "short and strong one", 0.79),
        sc("lean2", "short and strong two", 0.78),
    ]
    result = packer.build(items, budget=200, isolate=False)
    ids = {c.chunk.id for c in result.chunks}
    assert "top" in ids, "the top hit is always admitted"
    assert "fat" not in ids, "the bulky low-density chunk must be skipped"
    assert {"lean1", "lean2"} <= ids


def test_packer_truncates_rather_than_returning_nothing() -> None:
    settings = Settings(security={"enforce_tenant_isolation": False})
    result = ContextPacker(settings).build(
        [sc("big", "word " * 500, 0.99)], budget=50, isolate=False
    )
    assert len(result.chunks) == 1
    assert result.truncated == 1


def test_packer_numbers_passages_for_citations() -> None:
    settings = Settings(security={"enforce_tenant_isolation": False})
    items = [sc("a", "alpha content", 0.9), sc("b", "beta content", 0.8)]
    result = ContextPacker(settings).build(items, budget=5_000, isolate=False)
    assert "[1]" in result.text and "[2]" in result.text


def test_packer_isolates_untrusted_content() -> None:
    settings = Settings(security={"enforce_tenant_isolation": False})
    result = ContextPacker(settings).build([sc("a", "body text", 0.9)], budget=5_000, isolate=True)
    assert "<untrusted_document" in result.text


def test_packer_expands_parent_once_per_parent() -> None:
    """Several matching children of one parent must yield the parent a single time."""
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"parent_expansion": True, "reorder_lost_in_middle": False},
    )
    children = []
    for i in range(3):
        chunk = Chunk(
            id=f"child{i}",
            content=f"child slice {i}",
            parent_id="p1",
            metadata={"parent_text": "the full parent document body"},
        )
        children.append(ScoredChunk(chunk=chunk, score=0.9 - i * 0.1))
    result = ContextPacker(settings).build(children, budget=5_000, isolate=False)
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk.content == "the full parent document body"


# ---------------------------------------------------------------------------
# The relative cutoff must not be applied to rank-fusion scores
# ---------------------------------------------------------------------------
def test_relative_cutoff_is_skipped_for_rank_fusion_scores() -> None:
    """The worst bug this codebase had, pinned.

    Qdrant's server-side RRF uses k=2, so its scores are 1.0, 0.667, 0.4,
    0.286… A `relative_score_cutoff` of 0.35 — a fraction of the top score,
    which is only meaningful on a similarity scale — therefore truncated at
    **rank 2 for every query ever run**. The default retriever handed the
    reranker 3 candidates instead of 50 and threw away the relevant evidence
    doing it.

    Our own client-side RRF uses k=60, clustering scores near 0.028, where the
    same cutoff is silently inert. That is why it looked like a server-side quirk
    rather than a category error.
    """
    from ragorc.core.models import RetrievalSource

    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"top_k": 50, "relative_score_cutoff": 0.35, "dedupe_enabled": False},
    )
    # Exactly Qdrant's RRF k=2 progression.
    fused = [
        ScoredChunk(
            chunk=Chunk(id=f"c{i}", content=f"distinct passage number {i} about topic {i}"),
            score=1.0 / (2 + i) * 2,
            source=RetrievalSource.FUSED,
        )
        for i in range(10)
    ]
    kept, report = NoiseFilter(settings).apply(fused, top_k=50)
    assert len(kept) == 10, (
        f"rank-fusion scores were cut to {len(kept)}; the relative cutoff must not "
        "apply to a scale where 'a fraction of the top score' is meaningless"
    )
    assert report.below_threshold == 0


def test_relative_cutoff_still_applies_to_similarity_scores() -> None:
    """The guard must not disable the filter where it is correct and useful."""
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"top_k": 50, "relative_score_cutoff": 0.35, "dedupe_enabled": False},
    )
    similarity = [
        sc("strong", "the passage that answers the question", 0.42),
        sc("weak", "an unrelated passage about something else", 0.14),
    ]
    kept, report = NoiseFilter(settings).apply(similarity, top_k=50)
    assert [k.chunk.id for k in kept] == ["strong"]
    assert report.below_threshold == 1


def test_one_fused_entry_disables_the_cutoff_for_the_batch() -> None:
    """Conservative by design: mixing a rank score into a similarity comparison
    is the error being guarded, so a single fused entry stands the filter down.
    Extra candidates reaching the reranker is the cheap direction to err in —
    that stage exists to reject them."""
    from ragorc.core.models import RetrievalSource

    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"top_k": 50, "relative_score_cutoff": 0.35, "dedupe_enabled": False},
    )
    mixed = [
        sc("a", "first distinct passage here", 1.0),
        sc("b", "second distinct passage here", 0.10),
    ]
    mixed[0].source = RetrievalSource.FUSED
    kept, _ = NoiseFilter(settings).apply(mixed, top_k=50)
    assert len(kept) == 2


def test_packer_density_survives_negative_reranker_scores() -> None:
    """Cross-encoder scores are logits and routinely negative.

    A live `Xenova/ms-marco-MiniLM-L-6-v2` run scores `[8.42, -11.29, -11.31]`.
    Dividing a negative score by token count inverts the ordering — a bigger
    denominator moves a negative *closer to zero* — so raw density rewards longer
    chunks. Measured before the fix: a 401-token chunk at -5.00 was selected over
    a 31-token chunk at -0.50, a passage 10x worse and 13x longer.

    The pre-existing density test used positive scores only, which is why the
    suite missed this.
    """
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"reorder_lost_in_middle": False, "parent_expansion": False},
    )

    def chunk(cid: str, score: float, words: int) -> ScoredChunk:
        return ScoredChunk(chunk=Chunk(id=cid, content=("w " * words).strip()), score=score)

    negative = [chunk("top", 8.4, 20), chunk("good", -0.50, 30), chunk("bad", -5.00, 400)]
    ids = {
        c.chunk.id
        for c in ContextPacker(settings).build(negative, budget=440, isolate=False).chunks
    }
    assert "good" in ids, f"selected {ids}: the better, shorter chunk must win"
    assert "bad" not in ids

    # The ordering must be unchanged for positive scores — the lift is conditional,
    # not a change of policy.
    positive = [chunk("top", 9.0, 20), chunk("good", 0.50, 30), chunk("bad", 0.10, 400)]
    ids = {
        c.chunk.id
        for c in ContextPacker(settings).build(positive, budget=440, isolate=False).chunks
    }
    assert "good" in ids and "bad" not in ids

    # Which is why the lift is conditional rather than unconditional. Subtracting
    # the batch minimum from an already-positive batch would pin the weakest
    # candidate at density 0 and hand its budget to the bulky chunk — inverting the
    # example the module docstring argues the whole design from: a 900-token chunk
    # at 0.81 must not displace several 250-token chunks at 0.79. Only the tightly
    # clustered scores that fusion and cosine produce show the difference, so the
    # spread-out fixture above cannot stand in for this one.
    clustered = [
        chunk("top", 0.95, 10),
        chunk("fat", 0.81, 600),
        chunk("lean1", 0.79, 150),
        chunk("lean2", 0.79, 150),
        chunk("lean3", 0.79, 150),
    ]
    ids = {
        c.chunk.id
        for c in ContextPacker(settings).build(clustered, budget=630, isolate=False).chunks
    }
    assert "fat" not in ids, f"selected {ids}: relevance per token, on the raw ratio"
    assert {"lean1", "lean2", "lean3"} <= ids


def test_citation_offsets_follow_the_expanded_text() -> None:
    """After the packer swaps in a wider span, the chunk's own offsets no longer
    describe what the generator saw. Adding a within-quote offset to them lands on
    unrelated text — measured (90, 111) slicing to 'unts apply in Q4 only'."""
    from ragorc.generate.citations import extract_citations

    document = "Discounts apply in Q4 only. The fee is 3 percent. Refunds take 14 days."
    # A window chunk: content is the wide span, start_char still points at the child.
    chunk = Chunk(
        id="c1",
        content=document,
        document_id="d1",
        start_char=27,
        end_char=48,
        metadata={"window_start": 0, "source": "policy.md"},
    )
    citations = extract_citations(
        "The fee is 3 percent [1].", [ScoredChunk(chunk=chunk, score=0.9)]
    )
    assert citations
    citation = citations[0]
    assert citation.start_char is not None
    sliced = document[citation.start_char : citation.end_char]
    assert citation.quote.strip() in sliced or sliced.strip() in citation.quote, (
        f"offsets ({citation.start_char}, {citation.end_char}) slice to {sliced!r}, "
        f"which is not the quote {citation.quote!r}"
    )


def test_packer_budgets_the_scaffolding_it_renders() -> None:
    """`_select` priced body text only, while `_render` spends 20-25 tokens a passage
    on the numbered header and the <untrusted_document> wrapper.

    Measured before the fix on an 8192-token window: `available_context=6572` came back
    as a 9563-token pack (45% over), and prompt + `reserved_output` then no longer fit
    the window at all. Small windows are where it bites — at the 128k default the same
    bug is a harmless ~4.5%, which is why nothing caught it. It bites in `truncate`,
    the strategy whose whole job is making the context fit.
    """
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"reorder_lost_in_middle": False, "parent_expansion": False},
    )
    packer = ContextPacker(settings)
    passages = [
        ScoredChunk(
            chunk=Chunk(
                id=f"c{i}",
                content=("token " * 50).strip(),
                document_id="d1",
                metadata={"source": "policy.md"},
            ),
            score=0.9 - i * 0.01,
            rank=i,
        )
        for i in range(20)
    ]
    budget = 600

    wrapped = packer.build(passages, budget=budget, isolate=True)
    assert wrapped.tokens <= budget, (
        f"{len(wrapped.chunks)} passages rendered to {wrapped.tokens} tokens against a "
        f"{budget}-token budget: the header and the wrapper are read by the model too"
    )
    # The charge must be the measured one, not a pessimistic constant — leaving a
    # fifth of the window empty would trade one bug for another.
    assert wrapped.tokens > budget * 0.8
    assert len(wrapped.chunks) >= 6

    # `isolate=False` emits no wrapper, so the same chunks cost a few tokens each
    # instead of ~23 and more of them fit. A hardcoded per-passage constant would get
    # this direction wrong.
    plain = packer.build(passages, budget=budget, isolate=False)
    assert plain.tokens <= budget
    assert len(plain.chunks) > len(wrapped.chunks)

    # The unconditionally admitted top hit is clipped to fit *including* its own
    # scaffolding: budget=40 rendered 49 tokens before the fix.
    only = packer.build([sc("big", "word " * 500, 0.99)], budget=40)
    assert only.truncated == 1
    assert only.tokens <= 40


def test_packer_ships_the_top_hit_when_the_budget_is_all_scaffolding() -> None:
    """Charging for scaffolding raised the floor below which nothing fit at all.

    The unconditional admission of rank 0 clips it to `remaining` minus its own
    overhead. Once that overhead became the *measured* header and wrapper instead
    of a hardcoded 8, every budget under ~23 tokens clipped rank 0 to the empty
    string — so it was dropped, `truncated` still reported 0, and `remaining` was
    left whole for a weaker chunk to spend. Measured with the charge in place and
    the fallback removed: `budget=8 -> selected=[] dropped=2`, an empty pack,
    which `generate/answer.py:133` turns into an abstention. Pricing the
    scaffolding honestly widened a hole the audit found at the old constant.

    A sub-25-token window is unusable either way; what this pins is the direction
    it fails in. Shipping rank 0 clipped and over budget beats dropping the best
    evidence and answering from whatever was cheap enough to fit beneath it.
    """
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"reorder_lost_in_middle": False, "parent_expansion": False},
    )
    packer = ContextPacker(settings)
    items = [sc("best", "word " * 500, 0.99), sc("weak", "tiny", 0.10)]
    # Sized from the real scaffolding rather than a guess about the tokenizer:
    # `isolate=True` is the default and costs ~16-23 tokens before any body.
    scaffold = packer._scaffold_tokens(items[:1], isolate=True)[0]
    assert scaffold > 1, "a passage that costs nothing to frame would make this vacuous"

    for budget in range(1, scaffold + 1):
        result = packer.build(items, budget=budget, isolate=True)
        ids = [c.chunk.id for c in result.chunks]
        assert ids == ["best"], f"budget={budget} packed {ids} instead of the top hit"
        assert result.chunks[0].chunk.content, f"budget={budget} admitted an empty body"
        assert result.truncated == 1, f"budget={budget} clipped rank 0 without reporting it"


def test_packer_does_not_mutate_the_callers_chunks() -> None:
    """`with_score` shares the underlying `Chunk`, so assigning `scored.chunk.content`
    rewrote the object the caller still holds.

    Measured before the fix: after `build(budget=40)` the caller's 801-token chunk was
    permanently 32 tokens, and a second `build(budget=100_000)` still saw the clipped
    text — packing was not idempotent, which bites on the `generate()`-then-`stream()`
    path where the same chunks are packed twice. The content-derived `chunk.id` also
    stopped describing the body it names.

    What comes *back* must still carry the packed text: citations resolve `[n]` against
    what the model actually saw. The distinction is the point of this test.
    """
    from ragorc.core.tokens import count_tokens

    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"reorder_lost_in_middle": False, "parent_expansion": False},
    )
    original = "word " * 800
    caller = ScoredChunk(chunk=Chunk(id="big", content=original), score=0.9)

    clipped = ContextPacker(settings).build([caller], budget=40, isolate=False)
    assert caller.chunk.content == original, "the caller's chunk was clipped in place"
    assert "truncated" not in caller.explain, "the caller's explain dict was written to"
    assert clipped.truncated == 1
    packed = clipped.chunks[0]
    assert packed.chunk.content != original, "the pack must carry what the model saw"
    assert packed.explain["truncated"] is True
    assert packed.chunk.id == caller.chunk.id, "citations and the rerank cache key on it"
    assert packed.chunk.token_count == count_tokens(packed.chunk.content)

    # Idempotent: packing the same chunks again with room to spare returns full bodies.
    roomy = ContextPacker(settings).build([caller], budget=100_000, isolate=False)
    assert roomy.truncated == 0
    assert roomy.chunks[0].chunk.content == original

    # Same for the expansion path, which substituted the parent body in place.
    expanding = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"parent_expansion": True, "reorder_lost_in_middle": False},
    )
    parent_text = "CHAPTER 4 - FEES. The fee is 3 percent. Exceptions apply in Q4."
    child = Chunk(
        id="kid",
        content="The fee is 3 percent.",
        parent_id="p1",
        metadata={"parent_text": parent_text},
    )
    held = ScoredChunk(chunk=child, score=0.9)
    expanded = ContextPacker(expanding).build([held], budget=5_000, isolate=False)
    assert child.content == "The fee is 3 percent.", "expansion rewrote the caller's chunk"
    assert expanded.chunks[0].chunk.content == parent_text
    assert expanded.chunks[0].chunk is not child


def test_exact_dedupe_keeps_operator_only_differences() -> None:
    """Punctuation is content, not formatting.

    The key was built from ``\\w+`` tokens, so every operator vanished from it:
    ``>=`` and ``<=``, ``==`` and ``!=``, ``C++`` and ``C`` all hashed alike and
    one of each pair was dropped as an exact duplicate. On code, config or
    threshold documentation the dropped copy is the one that *contradicts* the
    survivor, which is the conflict this module says it wants to surface rather
    than hide. Case and whitespace normalization must survive the fix.
    """
    pairs = [
        ("The guard fires when latency >= 500 ms.", "The guard fires when latency <= 500 ms."),
        ("assert version == expected", "assert version != expected"),
        ("Written in C++.", "Written in C."),
        ("!!!", "###"),
    ]
    for left, right in pairs:
        kept, removed = NoiseFilter._dedupe_exact([sc("p", left, 0.9), sc("q", right, 0.8)])
        assert removed == 0, f"{left!r} and {right!r} were treated as the same passage"
        assert [k.chunk.id for k in kept] == ["p", "q"]

    same, removed = NoiseFilter._dedupe_exact(
        [sc("p", "Refunds  take 14 DAYS", 0.9), sc("q", "refunds take 14 days", 0.8)]
    )
    assert removed == 1, "case and whitespace must still normalize"
    assert [k.chunk.id for k in same] == ["p"]


def test_noise_filter_keeps_contradicting_operators_end_to_end(
    noise_settings: Settings,
) -> None:
    """The same case through ``apply``, with vectors so the near stage uses them.

    Without dense vectors the SimHash fallback is word-shingled and would collapse
    these two anyway — that stage is deliberately fuzzy and is left alone. What is
    pinned here is that the *exact* stage no longer decides an operator flip is the
    same text, so the disagreement survives to the reranker.
    """
    items = [
        sc("ge", "The guard fires when latency >= 500 ms.", 0.90, [1.0, 0.0, 0.0]),
        sc("le", "The guard fires when latency <= 500 ms.", 0.88, [0.0, 1.0, 0.0]),
    ]
    kept, report = NoiseFilter(noise_settings).apply(items)
    assert [k.chunk.id for k in kept] == ["ge", "le"]
    assert report.exact_duplicates == 0


# ---------------------------------------------------------------------------
# SimHash must see operators but ignore cosmetics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("assert x == 0", "assert x != 0"),
        ("if count > 0: ok", "if count < 0: ok"),
        ("when total >= limit", "when total <= limit"),
        ("value = a + b", "value = a - b"),
    ],
)
def test_simhash_separates_operator_only_differences(left: str, right: str) -> None:
    """The residual left by the exact-dedupe fix, one layer down.

    SimHash shingled over *words*, so `==` and `!=` produced an identical hash and
    collapsed at Hamming distance 0 — which no `near_dupe_threshold` could
    separate, because there is no threshold below zero. For a corpus of code or
    configuration the distinguishing line was silently dropped.
    """
    from ragorc.retrieve.noise import simhash

    distance = bin(simhash(left) ^ simhash(right)).count("1")
    assert distance > 8, (
        f"{left!r} and {right!r} hash within {distance} bits; an operator "
        "difference must be visible to the near-duplicate stage"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("The cat sat on the mat.", "The cat sat on the mat!"),
        ("x  ==  0", "x==0"),
        ("one, two, three here", "one two three here"),
        ('he said "hi" today', "he said hi today"),
    ],
)
def test_simhash_still_ignores_cosmetic_differences(left: str, right: str) -> None:
    """The other half of the contract, and the reason the token set is curated
    rather than "any punctuation": collapsing templated near-duplicates that
    differ only in spacing or sentence punctuation is what this stage is for."""
    from ragorc.retrieve.noise import simhash

    assert simhash(left) == simhash(right), f"{left!r} and {right!r} must hash alike"


def test_operator_only_chunks_survive_the_whole_filter() -> None:
    """End to end, with defaults: neither dedupe stage may drop either line."""
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        retrieval={"top_k": 10, "dedupe_enabled": True, "relative_score_cutoff": None},
    )
    items = [
        sc("eq", "assert x == 0", 0.9),
        sc("ne", "assert x != 0", 0.8),
    ]
    kept, report = NoiseFilter(settings).apply(items)
    assert {k.chunk.id for k in kept} == {"eq", "ne"}
    assert report.exact_duplicates == 0
    assert report.near_duplicates == 0


async def test_summarizer_clip_does_not_destroy_the_caller_s_chunk() -> None:
    """`_clip` truncated `scored.chunk.content` in place and returned the same
    object, so the map-failure path rewrote a chunk the caller still holds.

    The same aliasing hazard as `GraphLocalRetriever._annotate`: a `ScoredChunk`
    is shared with the retrieval result, the cache and every other consumer of
    that result set, and none of them asked to have their copy shortened.
    """
    from ragorc.context.summarize import ContextSummarizer
    from ragorc.core.models import Chunk, RetrievalSource, ScoredChunk

    original = "Refunds are processed within five business days. " * 40
    scored = ScoredChunk(
        chunk=Chunk(id="c1", content=original),
        score=0.9,
        source=RetrievalSource.DENSE,
        rank=0,
    )

    clipped = ContextSummarizer._clip(scored, 16)

    assert scored.chunk.content == original, "the caller's chunk must be untouched"
    assert clipped.chunk.content != original, "the returned copy must be the clipped one"
    assert len(clipped.chunk.content) < len(original)
    assert clipped.explain.get("truncated") is True
    assert "truncated" not in scored.explain, "nor may the caller's explain be written"


def test_the_overflow_decision_prices_the_scaffolding_the_packer_will_add() -> None:
    """`decide_strategy` compared body tokens against the window, but the packer
    charges each passage its header, wrapper and separator too — 3 tokens for a
    bare numbered passage, up to ~23 for a wrapped one with provenance.

    So a set that overflows once framed was reported as "fit", and the packer
    then silently dropped the tail the budgeter had decided to keep. The gap
    scales with the number of passages, which is exactly when it matters.
    """
    from ragorc.context.budget import ContextBudgeter
    from ragorc.context.pack import ContextPacker
    from ragorc.core.models import Chunk, RetrievalSource, ScoredChunk
    from ragorc.core.settings import Settings
    from ragorc.core.tokens import count_tokens

    settings = Settings(security={"enforce_tenant_isolation": False})
    packer, budgeter = ContextPacker(settings), ContextBudgeter(settings)

    chunks = [
        ScoredChunk(
            chunk=Chunk(id=f"c{i}", content="Refund policy detail. " * 8, document_id="d1"),
            score=1.0 - i / 100,
            source=RetrievalSource.DENSE,
            rank=i,
        )
        for i in range(12)
    ]
    bodies = sum(count_tokens(c.chunk.content) for c in chunks)
    overhead = packer.overhead(chunks)
    assert sum(overhead) > 0, "the fixture must actually carry scaffolding"

    # A window whose *available context* is just big enough for the bodies and
    # nothing more. `available_context` also subtracts the output reserve and a
    # safety margin, so the window is searched for rather than assumed.
    window = settings.generation.reserved_output_tokens + bodies
    while budgeter.plan(window=window).budget.available_context < bodies:
        window += 8
    plan = budgeter.plan(window=window)
    assert bodies <= plan.budget.available_context < bodies + sum(overhead), (
        "fixture: the bodies must fit alone and the framing must be what tips it over"
    )

    assert budgeter.decide_strategy(chunks, plan, overhead=overhead) != "fit", (
        "the framing is part of what has to fit"
    )
