"""Entity resolution: making one node out of many mentions of one thing.

Without this stage the graph fragments. "Acme", "Acme Corp" and "ACME
Corporation" become three nodes, each holding a third of the edges, and every
one of them is too sparsely connected to be useful: neighbourhood expansion from
"Acme" never reaches the acquisition that was asserted against "Acme Corp",
community detection splits one company across three communities, and a question
about Acme matches whichever fragment happens to share the question's wording.
Fragmentation does not degrade traversal gracefully — it stops traversal from
finding anything, and it does so silently, because every individual node looks
fine.

Three stages, cheapest first, each one only seeing what the previous could not
merge:

1. **Exact case-folded key.** ``ids.entity_id`` already case-folds, so bucketing
   on it is a dict insert per mention. This catches the overwhelming majority of
   duplicates, because the extraction prompt asks for consistent naming and
   mostly gets it.
2. **Normalized form.** Strip legal suffixes (Inc, Ltd, GmbH, Corp, LLC, SA, BV,
   PLC and their long forms), punctuation and leading articles, then compare. A
   string comparison, no model involved, and it catches the largest remaining
   class: the same organization written with and without its legal form.
3. **Embedding similarity** above ``graph.resolution_threshold``. The expensive
   one, and the only one that can merge "Meta" with "Facebook" or catch a
   transliteration difference.

**What stage 3 compares, and what it refuses.** The vector is over *name and
description* (``_resolution_text``), never the bare name, and a pair that clears
the threshold is still refused when the two normalized names disagree on a token
(``_token_conflict``). Both guards exist for the same reason: a cosine between
two short names is dominated by the part of the string they share, so name-only
similarity says "these strings look alike", which is not the question. On that
evidence alone stage 3 merged "Acme Group" with "Acme Holdings" (0.9359 under the
default bge-small-en-v1.5) and "Q1 2024" with "Q2 2024" (0.9390) — and because
``_rewrite_relations`` re-points every edge at the survivor, the graph then
asserted an acquisition about a company that never made one. A merge that is
wrong is worse than the fragmentation this whole module exists to prevent: a
fragmented graph finds nothing, a mis-merged one answers confidently from a
relationship nobody wrote down.

**Why stage 3 is blocked.** The naive form is an all-pairs comparison: 50 000
entities is 1.25 billion pairs, and materializing the similarity matrix alone is
10 GB at float32. Blocking partitions the candidates by ``(type, first
characters of the normalized form)`` and only compares within a block, which
turns the cost from n² into the sum of the blocks' squares — typically two to
three orders of magnitude less, because entity names spread across types and
initial letters. It is not lossless: an acronym and its expansion ("IBM" /
"International Business Machines") land in different blocks and will never be
compared. That is an acceptable miss, because the cosine between an acronym and
its expansion rarely clears 0.92 anyway — the pair that blocking loses is a pair
the threshold would have rejected.

All the comparisons inside a block are still done as **one** vectorized
operation over the whole candidate set: the block structure is expressed as
index arrays, and every surviving pair's similarity comes out of a single
``einsum`` over rows of the L2-normalized entity-embedding matrix. There is no
Python loop over pairs anywhere, and no per-block matmul either.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ragorc.core.ids import entity_id
from ragorc.core.models import Entity, FloatArray, Relation
from ragorc.core.protocols import DenseEmbedder
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["EntityResolver", "ResolutionReport", "normalized_form"]

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")

#: Legal forms that identify the *same* organization written two ways. Long and
#: short forms both listed because punctuation has already been stripped by the
#: time this set is consulted ("Inc." has become "inc").
_LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "llp",
        "lp",
        "plc",
        "gmbh",
        "mbh",
        "ag",
        "kg",
        "kgaa",
        "sa",
        "sas",
        "sarl",
        "spa",
        "srl",
        "bv",
        "nv",
        "ab",
        "as",
        "asa",
        "oy",
        "oyj",
        "aps",
        "pty",
        "pte",
        "kk",
    }
)
#: Deliberately NOT in the suffix set: "Group", "Holdings", "Partners",
#: "Ventures", "Foundation". They distinguish real, different entities — "Acme
#: Group" and "Acme Holdings" are separate companies — so stripping them would
#: merge two nodes that must stay apart. Stage 3 can still merge them when they
#: genuinely are the same thing, at the cost of one similarity comparison.

_ARTICLES = frozenset({"the", "a", "an", "le", "la", "les", "el", "los", "der", "die", "das"})

#: Types that carry no information, so they lose a vote when picking the merged
#: entity's type.
_GENERIC_TYPES = frozenset({"", "entity", "concept", "other", "unknown", "misc"})

#: Blocks larger than this are refined with a longer prefix. 2 000 members is
#: ~2M pairs, which is a 16 MB index array and a few milliseconds of einsum —
#: past that the quadratic term starts to dominate the whole resolve stage.
_MAX_BLOCK = 2_000
#: Longest prefix used for blocking. Beyond three characters the blocks stop
#: catching real variants: "Acme" and "ACME Corporation" share three initial
#: characters after normalization, a fourth starts splitting true duplicates.
_MAX_PREFIX = 3
#: How much description goes into the string stage 3 embeds. Enough to carry the
#: clause that distinguishes two similarly-named entities ("holding company for
#: the retail arm" vs "owns the real estate"), short enough that the name still
#: dominates the vector — descriptions here are prompt-sized, and a 2 000
#: character definition would make every entity of a type look alike instead.
_RESOLUTION_DESCRIPTION_CHARS = 240


def normalized_form(name: str) -> str:
    """Comparison key for stage 2: no punctuation, no legal form, no article.

    ``"The Acme Corporation, Inc."`` and ``"ACME Corp"`` both reduce to
    ``"acme"``. Ampersands become ``and`` *before* punctuation is stripped,
    because otherwise ``AT&T`` reduces to ``"at t"`` while ``AT and T`` reduces
    to ``"at and t"`` and the two forms stop matching.

    A name that reduces to nothing (all punctuation, or a bare article) keeps
    its case-folded original: collapsing every such name to the empty string
    would merge them all into one node.
    """
    text = (name or "").casefold().replace("&", " and ")
    tokens = _WHITESPACE.sub(" ", _PUNCT.sub(" ", text)).split()
    while tokens and tokens[0] in _ARTICLES:
        tokens.pop(0)
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens) or (name or "").strip().casefold()


@dataclass(slots=True)
class ResolutionReport:
    """Resolved graph plus the accounting of how each merge was decided."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    canonical: dict[str, str] = field(default_factory=dict)
    """Case-folded mention -> canonical name. Lets a caller rewrite anything
    else that referenced the pre-resolution names."""
    input_entities: int = 0
    input_relations: int = 0
    exact_merges: int = 0
    normalized_merges: int = 0
    embedding_merges: int = 0
    embedded: int = 0
    compared_pairs: int = 0
    vetoed_pairs: int = 0
    """Pairs that cleared ``resolution_threshold`` and were refused anyway
    because the two names disagree on a token. Counted separately from
    ``compared_pairs`` so a threshold that has drifted too low is visible as a
    ratio rather than as a silently different graph."""
    self_loops_dropped: int = 0
    dangling_dropped: int = 0

    @property
    def merged(self) -> int:
        return self.exact_merges + self.normalized_merges + self.embedding_merges

    def summary(self) -> dict[str, Any]:
        return {
            "input_entities": self.input_entities,
            "entities": len(self.entities),
            "input_relations": self.input_relations,
            "relations": len(self.relations),
            "exact_merges": self.exact_merges,
            "normalized_merges": self.normalized_merges,
            "embedding_merges": self.embedding_merges,
            "embedded": self.embedded,
            "compared_pairs": self.compared_pairs,
            "vetoed_pairs": self.vetoed_pairs,
            "self_loops_dropped": self.self_loops_dropped,
            "dangling_dropped": self.dangling_dropped,
        }


class _UnionFind:
    """Path-halving union-find over integer group ids.

    Merges arrive as unordered pairs from three independent stages, and a pair
    can chain ("Acme" ~ "Acme Corp" ~ "ACME Corporation"). Union-find closes
    those chains in near-constant time; resolving them by repeated dict rewrites
    would be quadratic in the chain length.
    """

    __slots__ = ("parent",)

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, a: int, b: int) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        # Lower index wins, which keeps the outcome independent of the order
        # the pairs arrived in — the same input graph must resolve identically
        # on every run or the whole ingest stops being idempotent.
        if root_a > root_b:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        return True


class EntityResolver:
    """Deduplicates entity mentions and rewrites the edges to match.

    ``embedder`` is optional by design: stages 1 and 2 need no model at all, and
    a deployment that has not configured an embedder still gets the bulk of the
    benefit rather than an error.
    """

    def __init__(
        self,
        embedder: DenseEmbedder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.embedder = embedder
        self.settings = settings or get_settings()
        self.cfg = self.settings.graph
        #: Descriptions become prompt payload for the community report, and the
        #: report is capped at ``community_summary_max_tokens``. A description
        #: longer than the report that consumes it is bandwidth with no reader,
        #: so the cap is derived from that budget (~4 characters per token).
        self._max_description_chars = max(400, 4 * self.cfg.community_summary_max_tokens)

    async def resolve(
        self, entities: Sequence[Entity], relations: Sequence[Relation] = ()
    ) -> ResolutionReport:
        """Merge mentions, then rewrite and merge the edges between them."""
        report = ResolutionReport(input_entities=len(entities), input_relations=len(relations))
        if not entities:
            report.relations = list(relations)
            return report

        groups, report.exact_merges = self._exact_groups(entities)
        if not self.cfg.resolve_entities:
            # Exact-key collapse is not resolution, it is a precondition for the
            # store's per-batch merge, so it happens even when resolution is off.
            resolved = [self._merge_group(group, embeddings={}) for group in groups]
            report.entities = resolved
            report.canonical = self._canonical_map(groups, resolved)
            report.relations = self._rewrite_relations(relations, report)
            self._assign_degree(report.entities, report.relations)
            log.info("graph_resolved", resolution="exact_only", **report.summary())
            return report

        merges = _UnionFind(len(groups))
        report.normalized_merges = self._merge_by_normalized_form(groups, merges)
        embeddings = await self._merge_by_embedding(groups, merges, report)

        clusters = self._clusters(groups, merges)
        report.entities = [
            self._merge_group(cluster, embeddings=embeddings) for cluster in clusters
        ]
        report.canonical = self._canonical_map(clusters, report.entities)
        report.relations = self._rewrite_relations(relations, report)
        self._assign_degree(report.entities, report.relations)
        log.info(
            "graph_resolved",
            resolution="full",
            threshold=self.cfg.resolution_threshold,
            **report.summary(),
        )
        return report

    # -- stage 1: exact case-folded key ------------------------------------
    def _exact_groups(self, entities: Sequence[Entity]) -> tuple[list[list[Entity]], int]:
        """Bucket mentions on ``ids.entity_id``, which case-folds for us.

        Insertion order of first appearance is preserved so that everything
        downstream — group indices, cluster ids, the canonical map — is a pure
        function of the input order and nothing depends on dict reordering
        between runs.
        """
        buckets: dict[str, list[Entity]] = {}
        tenant = self.settings.tenant_id
        for entity in entities:
            name = entity.name.strip()
            if not name:
                continue
            buckets.setdefault(entity_id(name, tenant), []).append(entity)
        groups = list(buckets.values())
        merged = sum(len(group) - 1 for group in groups)
        return groups, merged

    # -- stage 2: normalized form ------------------------------------------
    def _merge_by_normalized_form(
        self, groups: Sequence[Sequence[Entity]], merges: _UnionFind
    ) -> int:
        """Union groups whose representative names share a normalized form."""
        seen: dict[tuple[str, str], int] = {}
        count = 0
        for index, group in enumerate(groups):
            name = _representative(group).name
            # Type is part of the key: a PERSON called "Ford" and an
            # ORGANIZATION called "Ford Motor Company" normalize close together
            # and are not the same node.
            key = (_group_type(group), normalized_form(name))
            first = seen.setdefault(key, index)
            if first != index and merges.union(first, index):
                count += 1
        return count

    # -- stage 3: embedding similarity -------------------------------------
    async def _merge_by_embedding(
        self,
        groups: Sequence[Sequence[Entity]],
        merges: _UnionFind,
        report: ResolutionReport,
    ) -> dict[str, FloatArray]:
        """Compare blocked candidates in one vectorized pass.

        Only groups that share a block with at least one other group are
        embedded. Singleton blocks cannot produce a pair, so embedding them
        would be a forward pass whose result is never read — and singletons are
        the majority of any real graph's entities.
        """
        if self.embedder is None or len(groups) < 2:
            return {}

        blocks = _blocks(groups)
        candidates = sorted({index for block in blocks for index in block})
        if not candidates:
            return {}

        names = [_representative(groups[index]).name for index in candidates]
        # Name *and description*, not the name alone. See _resolution_text: a
        # cosine over two bare short names measures string overlap, and the
        # threshold was merging separate entities on it.
        vectors = await self.embedder.embed_documents(
            [_resolution_text(groups[index]) for index in candidates]
        )
        if not vectors:
            return {}
        report.embedded = len(candidates)

        matrix = np.asarray(np.stack(vectors), dtype=np.float32)
        position = {index: slot for slot, index in enumerate(candidates)}
        left, right = await asyncio.to_thread(_pair_indices, blocks, position)
        report.compared_pairs = int(left.size)
        if left.size:
            similar = await asyncio.to_thread(
                _similar_pairs, matrix, left, right, float(self.cfg.resolution_threshold)
            )
            self._union_similar(groups, merges, report, candidates, similar)

        # Keyed by name so the merge step is a dict hit rather than a scan over
        # every embedded candidate for every surviving entity. The vector is the
        # one stage 3 compared — description included — which is also the better
        # payload for the Neo4j entity index, since questions are phrased against
        # what an entity *is*, not against its name in isolation.
        return dict(zip(names, (matrix[position[index]] for index in candidates), strict=True))

    @staticmethod
    def _union_similar(
        groups: Sequence[Sequence[Entity]],
        merges: _UnionFind,
        report: ResolutionReport,
        candidates: Sequence[int],
        similar: Sequence[tuple[int, int, float]],
    ) -> None:
        """Union the pairs the embedding accepted, minus the lexical vetoes.

        Enriching the embedded text moves most look-alike pairs below the
        threshold, but not the ones whose descriptions are as parallel as their
        names: "First quarter of 2024." and "Second quarter of 2024." differ by
        the same single word the names do. So a pair that *disagrees* on a token
        is refused outright, no matter what the cosine says. Disagreement is
        narrower than difference — see ``_token_conflict``; "Facebook"/"FB" and
        "Meta"/"Meta Platforms" are still decided by the embedding alone, because
        those are the pairs this stage exists for.

        The check is cluster-against-cluster, not pair-against-pair, because the
        union is transitive: with a bare "Acme" in the corpus too, "Acme" merging
        with both "Acme Group" and "Acme Holdings" would reunite exactly the two
        names the veto just kept apart.

        This is a Python loop, unlike the rest of stage 3, and deliberately: it
        only ever walks pairs that already cleared the threshold — a handful,
        against the millions the einsum scored — so vectorizing it would cost
        clarity to save nothing.
        """
        if not similar:
            # Nothing cleared the threshold, so skip the per-group tokenization:
            # on a corpus where stage 3 finds nothing it is the only cost left.
            return

        tokens = [
            frozenset(normalized_form(_representative(group).name).split()) for group in groups
        ]
        members: dict[int, list[int]] = {}
        for index in range(len(groups)):
            members.setdefault(merges.find(index), []).append(index)

        for left, right, score in similar:
            a, b = candidates[left], candidates[right]
            root_a, root_b = merges.find(a), merges.find(b)
            if root_a == root_b:
                continue
            conflict = _cluster_conflict(members[root_a], members[root_b], tokens)
            if conflict is not None:
                blocker_a, blocker_b, differing = conflict
                report.vetoed_pairs += 1
                # Debug, not info: above a loose threshold near-misses are the
                # common case, and the info stream should carry the decisions
                # that changed the graph rather than the ones that did not.
                log.debug(
                    "graph_entity_merge_vetoed",
                    left=_representative(groups[a]).name,
                    right=_representative(groups[b]).name,
                    score=round(score, 4),
                    conflict=sorted(differing),
                    transitive=(blocker_a, blocker_b) != (a, b),
                )
                continue
            if not merges.union(a, b):
                continue
            report.embedding_merges += 1
            root = merges.find(a)
            members[root].extend(members.pop(root_b if root == root_a else root_a))
            # Every merge, by name and score, at info. Counts alone made a wrong
            # merge unreviewable after the fact: the graph had one node where the
            # corpus had two and nothing said which two.
            log.info(
                "graph_entity_merged",
                stage="embedding",
                left=_representative(groups[a]).name,
                right=_representative(groups[b]).name,
                score=round(score, 4),
            )

    # -- merging -----------------------------------------------------------
    @staticmethod
    def _clusters(groups: Sequence[Sequence[Entity]], merges: _UnionFind) -> list[list[Entity]]:
        """Flatten the union-find into one mention list per surviving entity."""
        by_root: dict[int, list[Entity]] = {}
        for index, group in enumerate(groups):
            by_root.setdefault(merges.find(index), []).extend(group)
        return [by_root[root] for root in sorted(by_root)]

    def _merge_group(self, group: Sequence[Entity], *, embeddings: dict[str, FloatArray]) -> Entity:
        """Collapse a cluster of mentions into one canonical entity.

        The canonical name is the longest surface form: "Acme Corporation" tells
        a reader (and a full-text index) more than "Acme", and the shorter forms
        survive as aliases so a question phrased either way still matches. Ties
        break lexicographically, because a tie broken by iteration order would
        make the graph depend on which chunk finished extracting first.
        """
        canonical = max(group, key=lambda e: (len(e.name), e.name))
        aliases: dict[str, None] = {}
        chunk_ids: dict[str, None] = {}
        descriptions: list[str] = []
        length = 0
        for entity in group:
            if entity.name != canonical.name:
                aliases.setdefault(entity.name, None)
            for alias in entity.aliases:
                if alias and alias != canonical.name:
                    aliases.setdefault(alias, None)
            for chunk_id in entity.source_chunk_ids:
                if chunk_id:
                    chunk_ids.setdefault(chunk_id, None)
            fragment = entity.description.strip()
            if fragment and fragment not in descriptions and length < self._max_description_chars:
                descriptions.append(fragment)
                length += len(fragment) + 1

        metadata: dict[str, Any] = {}
        for entity in group:
            metadata.update(entity.metadata)
        if len(group) > 1:
            metadata["merged_mentions"] = len(group)

        return Entity(
            name=canonical.name,
            type=_group_type(group),
            description="\n".join(descriptions),
            aliases=tuple(aliases),
            source_chunk_ids=tuple(chunk_ids),
            community_id=None,
            # Attached only when Neo4j will actually index it. The vector was
            # computed for resolution either way, but 384 floats per entity is
            # real bandwidth and disk, and with ``create_vector_index`` off
            # nothing ever reads the property back.
            embedding=(
                embeddings.get(canonical.name) if self.settings.neo4j.create_vector_index else None
            ),
            metadata=metadata,
        )

    @staticmethod
    def _canonical_map(
        clusters: Sequence[Sequence[Entity]], resolved: Sequence[Entity]
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for cluster, entity in zip(clusters, resolved, strict=True):
            for mention in cluster:
                mapping[mention.name.casefold()] = entity.name
                for alias in mention.aliases:
                    if alias:
                        mapping.setdefault(alias.casefold(), entity.name)
        return mapping

    # -- edges -------------------------------------------------------------
    def _rewrite_relations(
        self, relations: Sequence[Relation], report: ResolutionReport
    ) -> list[Relation]:
        """Point every edge at canonical names and merge the duplicates.

        This is the half of resolution that actually fixes traversal: merging
        the nodes without rewriting the edges leaves the edges attached to names
        that no longer exist, which is a worse graph than the fragmented one.

        Weights *sum* here, unlike inside a chunk: two different chunks
        asserting the same relationship is corroboration, and an edge many
        documents agree on should outrank one a single sentence claimed.
        """
        if not relations:
            return []
        canonical = report.canonical
        merged: dict[tuple[str, str, str], Relation] = {}
        for relation in relations:
            source = canonical.get(relation.source.casefold(), relation.source)
            target = canonical.get(relation.target.casefold(), relation.target)
            if not source or not target:
                report.dangling_dropped += 1
                continue
            if source.casefold() == target.casefold():
                # Two merged mentions were the endpoints of this edge, so the
                # edge is now a self-loop. It asserted something real about two
                # names, but as one node it says nothing and it would inflate
                # the degree that community ranking reads.
                report.self_loops_dropped += 1
                continue
            key = (source.casefold(), relation.type, target.casefold())
            existing = merged.get(key)
            if existing is None:
                merged[key] = Relation(
                    source=source,
                    target=target,
                    type=relation.type,
                    description=relation.description,
                    weight=float(relation.weight),
                    source_chunk_ids=tuple(relation.source_chunk_ids),
                    metadata=dict(relation.metadata),
                )
                continue
            existing.weight += float(relation.weight)
            fragment = relation.description.strip()
            if fragment and fragment not in existing.description:
                existing.description = (
                    f"{existing.description}\n{fragment}" if existing.description else fragment
                )
            existing.source_chunk_ids = _union(existing.source_chunk_ids, relation.source_chunk_ids)
        return list(merged.values())

    @staticmethod
    def _assign_degree(entities: Sequence[Entity], relations: Sequence[Relation]) -> None:
        """Fill ``Entity.degree`` from the resolved edge set.

        Degree is read twice downstream — community ranking and the truncation
        priority in the community summarizer — and computing it here means
        neither has to walk the edge list again. Counted on the merged edges, so
        an edge asserted by ten chunks counts once: degree is about the graph's
        shape, not about how often it was mentioned.
        """
        counts: dict[str, int] = {}
        for relation in relations:
            counts[relation.source.casefold()] = counts.get(relation.source.casefold(), 0) + 1
            counts[relation.target.casefold()] = counts.get(relation.target.casefold(), 0) + 1
        for entity in entities:
            entity.degree = counts.get(entity.key, 0)


# ---------------------------------------------------------------------------
# Blocking and similarity (CPU-bound; the numpy halves run in a thread)
# ---------------------------------------------------------------------------
def _representative(group: Sequence[Entity]) -> Entity:
    """The mention that stands for a group: longest name, lexicographic tie."""
    return max(group, key=lambda e: (len(e.name), e.name))


def _resolution_text(group: Sequence[Entity]) -> str:
    """The string stage 3 embeds: the name plus a bounded description.

    Embedding the bare name is what made this stage merge different companies.
    Two short names that share a token sit at 0.93-0.94 under any general-purpose
    encoder — "Acme Group"/"Acme Holdings" is 0.9359 with the default
    bge-small-en-v1.5 — because most of the string is identical, so the threshold
    was being asked to tell two entities apart on evidence that never mentioned
    what distinguishes them. The description is where "holding company for the
    retail arm" and "separate entity that owns the real estate" actually diverge,
    so it has to be inside the vector.

    The *type* is deliberately left out, even though it is part of the blocking
    key. Blocking guarantees that every pair reaching the comparison already has
    the identical type, so appending it would add the same tokens to both sides
    of every cosine — discriminating nothing and lifting every score toward the
    threshold, which is the opposite of what this function is for.

    Descriptions from every mention in the group, not just the representative's:
    the longest *name* is frequently the mention with the thinnest description,
    and dropping the others would put the group's whole distinguishing content
    outside the comparison.
    """
    fragments: dict[str, None] = {}
    for entity in group:
        fragment = entity.description.strip()
        if fragment:
            fragments.setdefault(fragment, None)
    name = _representative(group).name
    description = " ".join(fragments)[:_RESOLUTION_DESCRIPTION_CHARS]
    return f"{name}: {description}" if description else name


def _token_conflict(left: frozenset[str], right: frozenset[str]) -> frozenset[str]:
    """Tokens two normalized names *disagree* on, empty when they merely differ.

    An empty result is not a claim that the two are the same thing — it means the
    names carry no objection and the embedding decides alone. Three cases:

    * No shared token ("meta"/"facebook", "facebook"/"fb"): nothing to disagree
      about, and these are precisely the pairs stage 3 exists to catch.
    * One side is the other plus qualification ("meta"/"meta platforms"): an
      extension is how one thing gets written at two levels of detail.
    * Both sides contribute a token the other lacks *while* sharing one
      ("acme group"/"acme holdings", "q1 2024"/"q2 2024"): the shared token is
      what pulls the cosine up and the differing ones are the entire content of
      the distinction. Refused — this is the conflict.

    Deliberately not a curated list of qualifier words ("Group", "Holdings",
    "Ventures", …): the same shape shows up in quarters, report years, product
    generations and building names, and a blocklist would have to be extended
    once per corpus. The structural test needs no vocabulary.
    """
    if not left & right:
        return frozenset()
    only_left, only_right = left - right, right - left
    if not only_left or not only_right:
        return frozenset()
    return only_left | only_right


def _cluster_conflict(
    left: Sequence[int], right: Sequence[int], tokens: Sequence[frozenset[str]]
) -> tuple[int, int, frozenset[str]] | None:
    """First conflicting cross-cluster pair, or ``None`` if the merge is clean.

    Quadratic in the cluster sizes, which is fine: clusters at this point hold a
    handful of surface forms of one entity, and the alternative — checking only
    the two candidates — lets a third name chain the conflicting pair together.
    """
    for i in left:
        for j in right:
            differing = _token_conflict(tokens[i], tokens[j])
            if differing:
                return i, j, differing
    return None


def _group_type(group: Sequence[Entity]) -> str:
    """Majority type, with generic labels losing to specific ones.

    A mention typed ``CONCEPT`` by one chunk and ``ORGANIZATION`` by three
    should end up an organization; ties break on the type string so the outcome
    does not depend on extraction order.
    """
    votes: dict[str, int] = {}
    for entity in group:
        entity_type = (entity.type or "").strip()
        if not entity_type:
            continue
        weight = 1 if entity_type.casefold() in _GENERIC_TYPES else 1000
        votes[entity_type] = votes.get(entity_type, 0) + weight
    if not votes:
        return "Entity"
    return max(sorted(votes), key=lambda t: votes[t])


def _blocks(groups: Sequence[Sequence[Entity]]) -> list[list[int]]:
    """Partition group indices into comparison blocks.

    Key is ``(type, normalized-form prefix)``. A block that grows past
    ``_MAX_BLOCK`` is refined with a longer prefix instead of being compared
    quadratically — the refinement trades a few missed cross-prefix merges for a
    bounded worst case, which matters because a corpus about one industry
    genuinely does put thousands of ORGANIZATION names under the same letter.
    """
    keys = [(_group_type(group), normalized_form(_representative(group).name)) for group in groups]
    pending: list[tuple[int, list[int]]] = [(1, list(range(len(groups))))]
    out: list[list[int]] = []
    while pending:
        prefix_len, members = pending.pop()
        buckets: dict[tuple[str, str], list[int]] = {}
        for index in members:
            entity_type, normalized = keys[index]
            buckets.setdefault((entity_type, normalized[:prefix_len]), []).append(index)
        for bucket in buckets.values():
            if len(bucket) < 2:
                continue
            if len(bucket) > _MAX_BLOCK and prefix_len < _MAX_PREFIX:
                pending.append((prefix_len + 1, bucket))
                continue
            if len(bucket) > _MAX_BLOCK:
                log.warning(
                    "graph_resolution_block_oversized",
                    size=len(bucket),
                    prefix_len=prefix_len,
                    pairs=len(bucket) * (len(bucket) - 1) // 2,
                )
            out.append(bucket)
    return out


def _pair_indices(
    blocks: Sequence[Sequence[int]], position: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten every within-block pair into two row-index arrays.

    Building the pair list per block and concatenating is what lets the actual
    similarity be one operation over the whole candidate set instead of one
    matmul per block. ``triu_indices`` gives the upper triangle only, so no pair
    is computed twice and no self-pair is computed at all.
    """
    lefts: list[np.ndarray] = []
    rights: list[np.ndarray] = []
    for block in blocks:
        rows = np.fromiter((position[index] for index in block), dtype=np.int64, count=len(block))
        upper_i, upper_j = np.triu_indices(len(block), k=1)
        lefts.append(rows[upper_i])
        rights.append(rows[upper_j])
    if not lefts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    return np.concatenate(lefts), np.concatenate(rights)


def _similar_pairs(
    matrix: FloatArray, left: np.ndarray, right: np.ndarray, threshold: float
) -> list[tuple[int, int, float]]:
    """Cosine similarity for every candidate pair, in one pass.

    The embedders in this library L2-normalize by default, which makes cosine a
    plain dot product; the norms are divided out anyway so a provider that does
    not normalize still gets a correct cosine rather than a silently inflated
    one. ``einsum`` over gathered rows is the whole computation — one C loop, no
    (n, n) matrix ever allocated, so memory is O(pairs) rather than O(n²).

    The score travels with the pair because every merge is logged with it: a
    merge at 0.921 and one at 0.998 are different claims, and without the number
    a reviewer cannot tell which of the two produced a bad node.
    """
    norms = np.linalg.norm(matrix, axis=1)
    np.maximum(norms, 1e-12, out=norms)
    unit = matrix / norms[:, None]
    scores = np.einsum("ij,ij->i", unit[left], unit[right])
    keep = np.flatnonzero(scores >= threshold)
    if keep.size == 0:
        return []
    # Strongest pairs first: union-find is order-independent for the final
    # partition, but merging the most confident pairs first keeps the logged
    # sample interpretable when someone audits a bad merge.
    order = keep[np.argsort(-scores[keep], kind="stable")]
    return [(int(left[i]), int(right[i]), float(scores[i])) for i in order]


def _union(*groups: Iterable[str]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for group in groups:
        for value in group:
            if value:
                seen.setdefault(value, None)
    return tuple(seen)
