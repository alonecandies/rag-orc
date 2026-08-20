"""Entity and relation extraction — the step that decides whether the graph works.

GraphRAG's retrieval quality is bounded by this module, not by the traversal
that reads it. A missed entity is a node that no query can reach; an
inconsistently named entity is two nodes where there should be one, and the
edges that should have met at it point at different places instead. Everything
below exists to protect against those two failures.

**Why one call per chunk, not one call per batch.** ``extraction_batch_size``
bounds *concurrency*, not the number of chunks per prompt. Packing several
chunks into one request would be cheaper per token, and it is the wrong trade:
the extraction has to record *which chunk* asserted each entity — that is what
``source_chunk_ids`` is, and it is what later lets local search walk from a
matched entity back to citable text. A merged prompt loses the attribution, and
extraction recall measurably drops when a model is asked to enumerate entities
across several unrelated passages at once.

**Why gleaning, and why the default is one pass.** A single extraction pass
misses entities: the model satisfices, stops at the salient ones and ignores the
tail. Asking again — "here is what you found, what did you miss?" — recovers
part of that tail. But each pass re-sends the whole chunk plus the growing
already-found list, so it costs *more* than the first pass while returning
strictly fewer entities: the first gleaning typically adds 10-20%, a second adds
a few percent, a third is noise and hallucination. One pass is where the curve
stops paying, which is why ``graph.max_gleanings`` defaults to 1. A pass that
returns nothing new short-circuits the rest: the same text cannot yield more on
a later attempt.

**Why endpoints are validated against the extracted entity set.** A relation
naming an entity that was not extracted creates a phantom node when the store
merges it — the Neo4j writer merges relationship endpoints on purpose, so a
typo'd or hallucinated endpoint becomes a real, empty, description-less node.
Those nodes fragment traversal: neighbourhood expansion walks into them and
finds nothing, community detection sees spurious bridges, and the entity never
matches a question because it has no text to match on. Dropping the edge loses
one assertion; keeping it corrupts the graph.

**Why relationship types are normalized and validated here.** Cypher cannot
parameterize a relationship type — ``-[:$type]->`` does not exist — so the type
is interpolated into the query text by the store. An LLM-authored type string
therefore sits on an injection path, and the store's guard raises
:class:`~ragorc.core.errors.GuardrailViolation` on anything outside
``^[A-Z][A-Z0-9_]*$``. Raising there would fail a whole ingest batch over one
malformed label, so the repair happens here: normalize what can be normalized
(spaces and hyphens to underscores, uppercase, illegal characters removed) and
fall back to ``RELATED_TO`` for the rest. The edge is real information; its
label is not worth an aborted ingest.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.concurrency import map_concurrent
from ragorc.core.models import Chunk, Entity, Relation, Usage
from ragorc.core.protocols import LLM
from ragorc.core.schemas import ExtractionOutput
from ragorc.core.settings import Settings, get_settings
from ragorc.core.tokens import TokenBudget, truncate_to_tokens
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task

log = structlog.get_logger(__name__)

__all__ = [
    "EntityExtractor",
    "GraphExtraction",
    "normalize_entity_name",
    "normalize_relation_type",
]

#: Mirrors ``ragorc.stores.neo4j.schema._REL_TYPE_RE``. Duplicated rather than
#: imported so extraction does not depend on a store implementation, and kept
#: identical because the store's guard is the thing that actually rejects.
_REL_TYPE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_REL_SEPARATORS = re.compile(r"[\s\-/.,]+")
_REL_ILLEGAL = re.compile(r"[^A-Z0-9_]+")
_REL_RUNS = re.compile(r"_{2,}")
_FALLBACK_REL_TYPE = "RELATED_TO"

_WHITESPACE = re.compile(r"\s+")
#: Quote characters a model wraps names in. Typographic variants included
#: because "smart quotes" survive copy-paste into source documents and a name
#: wrapped in one is a different node from the same name without it.
_WRAPPING_QUOTES = "\"'`\u201c\u201d\u2018\u2019\u00ab\u00bb"

#: Fraction of the model's usable context an extraction prompt may occupy. The
#: rest is headroom for the schema, the system block and the reply. Chunks are
#: ~512 tokens by default, so this only ever binds on abnormally large chunks —
#: and truncating one is much better than a hard provider rejection mid-ingest.
_CHUNK_SHARE = 0.5
#: Share of the extraction budget the "already extracted" list may take in a
#: gleaning pass. The chunk text is what the model reads; the existing list is
#: only there to suppress repeats, so it yields when space is tight.
_EXISTING_SHARE = 0.3


def normalize_entity_name(raw: str) -> str:
    """Canonical surface form of an extracted name.

    Whitespace collapse and quote stripping are not cosmetic: they are what
    makes the relation-endpoint lookup below agree with the entity table when a
    model writes ``"Acme  Corporation"`` in one field and ``Acme Corporation``
    in another. Casing is *preserved* — the most complete spelling is what the
    graph shows a reader — and case-insensitivity is handled by
    :attr:`~ragorc.core.models.Entity.key`.
    """
    return _WHITESPACE.sub(" ", (raw or "").strip().strip(_WRAPPING_QUOTES)).strip()


def normalize_relation_type(raw: str) -> str:
    """Coerce a model-authored type into ``^[A-Z][A-Z0-9_]*$``, or fall back.

    ``"works for"`` becomes ``WORKS_FOR``; ``"acquired (2019)"`` becomes
    ``ACQUIRED_2019``; ``"→"`` becomes ``RELATED_TO``. Truncation happens before
    the pattern check so a long-but-legal type keeps its meaning instead of
    being discarded for length.
    """
    candidate = _REL_SEPARATORS.sub("_", (raw or "").strip()).upper()
    candidate = _REL_RUNS.sub("_", _REL_ILLEGAL.sub("", candidate)).strip("_")
    candidate = candidate[:64].strip("_")
    return candidate if _REL_TYPE_RE.match(candidate) else _FALLBACK_REL_TYPE


@dataclass(slots=True)
class GraphExtraction:
    """Mention-level output: one entity per (chunk, name), not yet resolved.

    Cross-chunk deduplication is deliberately *not* done here. It needs the
    normalization and embedding machinery in :mod:`ragorc.index.graph.resolve`,
    and doing half of it here would hide which stage decided that two mentions
    were the same thing.
    """

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    chunks_processed: int = 0
    chunks_failed: int = 0
    llm_calls: int = 0
    gleaning_calls: int = 0
    dangling_dropped: int = 0
    types_coerced: int = 0
    types_unknown: int = 0

    def merge(self, other: GraphExtraction) -> None:
        self.entities.extend(other.entities)
        self.relations.extend(other.relations)
        self.chunks_processed += other.chunks_processed
        self.chunks_failed += other.chunks_failed
        self.llm_calls += other.llm_calls
        self.gleaning_calls += other.gleaning_calls
        self.dangling_dropped += other.dangling_dropped
        self.types_coerced += other.types_coerced
        self.types_unknown += other.types_unknown

    def summary(self) -> dict[str, Any]:
        return {
            "entity_mentions": len(self.entities),
            "relations": len(self.relations),
            "chunks_processed": self.chunks_processed,
            "chunks_failed": self.chunks_failed,
            "llm_calls": self.llm_calls,
            "gleaning_calls": self.gleaning_calls,
            "dangling_dropped": self.dangling_dropped,
            "types_coerced": self.types_coerced,
            "types_unknown": self.types_unknown,
        }


class EntityExtractor:
    """Chunks in, entity/relation mentions out, one LLM call per chunk per pass."""

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.cfg = self.settings.graph
        self.router = router or ModelRouter(self.settings.llm)
        self._extract_prompt = get_prompt("extract_graph")
        self._gleaning_prompt = get_prompt("extract_gleaning")
        # ``entity_types`` lives in the *system* block of ``extract_graph``, and
        # ``Prompt.render`` only formats the template — so the system block is
        # rendered here, once per extractor, instead of once per call.
        self._entity_types = [t.strip().upper() for t in self.cfg.entity_types if t.strip()]
        self._allowed_types = {t: t for t in self._entity_types}
        self._extract_system = self._extract_prompt.system.format(
            entity_types=", ".join(self._entity_types) or "any salient entity"
        )
        self._fallback_type = (
            "CONCEPT"
            if "CONCEPT" in self._allowed_types
            else (self._entity_types[-1] if self._entity_types else "CONCEPT")
        )

        budget = TokenBudget(
            total=self.settings.llm.context_window,
            reserved_output=self.settings.llm.max_tokens,
            reserved_system=len(self._extract_system) // 4,
        )
        self._chunk_budget = max(int(budget.available_context * _CHUNK_SHARE), 256)
        self._existing_budget = max(int(self._chunk_budget * _EXISTING_SHARE), 128)

    # -- public API --------------------------------------------------------
    async def extract(self, chunks: Sequence[Chunk]) -> tuple[GraphExtraction, Usage]:
        """Extract over every chunk concurrently, bounded by ``extraction_batch_size``.

        One chunk's failure is absorbed, not propagated: a 20 000-chunk ingest
        must not be lost to a single provider hiccup or an unparseable reply.
        The failure count comes back in the report so the caller can decide
        whether the loss is acceptable.
        """
        result = GraphExtraction()
        if not chunks or not self.cfg.extract_entities:
            return result, Usage()

        outcomes = await map_concurrent(
            self._extract_chunk,
            list(chunks),
            limit=max(1, self.cfg.extraction_batch_size),
            return_exceptions=True,
        )

        usages: list[Usage] = []
        for chunk, outcome in zip(chunks, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                result.chunks_failed += 1
                log.warning(
                    "graph_extract_chunk_failed",
                    chunk_id=chunk.id,
                    error=str(outcome)[:200],
                    error_type=type(outcome).__name__,
                )
                continue
            partial, usage = outcome
            result.merge(partial)
            usages.append(usage)

        total = Usage.sum(usages)
        log.info(
            "graph_extracted",
            **result.summary(),
            model=self.router.model_for(Task.EXTRACT_GRAPH),
            cost_usd=round(total.cost_usd, 6),
        )
        return result, total

    async def extract_chunk(self, chunk: Chunk) -> tuple[GraphExtraction, Usage]:
        """Extract from a single chunk, gleanings included. Public for callers
        that stream chunks rather than materializing the whole list."""
        return await self._extract_chunk(chunk)

    # -- internals ---------------------------------------------------------
    async def _extract_chunk(self, chunk: Chunk) -> tuple[GraphExtraction, Usage]:
        text = truncate_to_tokens(chunk.content, self._chunk_budget)
        model = self.router.model_for(Task.EXTRACT_GRAPH)
        usages: list[Usage] = []
        result = GraphExtraction(chunks_processed=1)

        #: name -> (canonical name, type, description parts). Keyed on the
        #: case-folded name so a gleaning pass that re-spells an entity adds to
        #: it instead of creating a sibling node.
        entities: dict[str, Entity] = {}
        relations: dict[tuple[str, str, str], Relation] = {}

        first, usage = await self.llm.structured(
            self._extract_prompt.render(text=text),
            ExtractionOutput,
            system=self._extract_system,
            model=model,
            stage="extract_graph",
        )
        usages.append(usage)
        result.llm_calls += 1
        self._absorb(first, chunk, entities, relations, result)

        for pass_index in range(max(0, self.cfg.max_gleanings)):
            before = len(entities) + len(relations)
            existing = truncate_to_tokens(
                self._render_existing(entities, relations), self._existing_budget
            )
            gleaned, usage = await self.llm.structured(
                self._gleaning_prompt.render(text=text, existing=existing),
                ExtractionOutput,
                system=self._gleaning_prompt.system,
                model=model,
                stage="extract_gleaning",
            )
            usages.append(usage)
            result.llm_calls += 1
            result.gleaning_calls += 1
            self._absorb(gleaned, chunk, entities, relations, result)
            if len(entities) + len(relations) == before:
                # Nothing new. The text is exhausted for this model, and every
                # further pass costs a full prompt to confirm that again.
                log.debug(
                    "graph_gleaning_exhausted",
                    chunk_id=chunk.id,
                    pass_index=pass_index,
                    entities=len(entities),
                )
                break

        result.entities = list(entities.values())
        result.relations = self._prune_dangling(chunk, entities, relations, result)
        return result, Usage.sum(usages)

    def _absorb(
        self,
        output: ExtractionOutput,
        chunk: Chunk,
        entities: dict[str, Entity],
        relations: dict[tuple[str, str, str], Relation],
        result: GraphExtraction,
    ) -> None:
        """Fold one pass's output into the chunk-level accumulators."""
        for entity_out in output.entities:
            name = normalize_entity_name(entity_out.name)
            if not name:
                continue
            entity_type, unknown = self._coerce_type(entity_out.type)
            result.types_unknown += unknown
            entity_key = name.casefold()
            existing_entity = entities.get(entity_key)
            if existing_entity is None:
                entities[entity_key] = Entity(
                    name=name,
                    type=entity_type,
                    description=(entity_out.description or "").strip(),
                    source_chunk_ids=(chunk.id,) if chunk.id else (),
                    metadata=self._provenance(chunk),
                )
                continue
            # A gleaning pass may spell the same entity more completely, or add
            # a description the first pass omitted. Keep the longer name so the
            # graph shows the most complete form; append genuinely new text.
            if len(name) > len(existing_entity.name):
                existing_entity.name = name
            description = (entity_out.description or "").strip()
            if description and description not in existing_entity.description:
                existing_entity.description = (
                    f"{existing_entity.description}\n{description}"
                    if existing_entity.description
                    else description
                )

        for relation_out in output.relations:
            source = normalize_entity_name(relation_out.source)
            target = normalize_entity_name(relation_out.target)
            if not source or not target or source.casefold() == target.casefold():
                # A self-loop carries no traversal information and inflates
                # degree centrality, which downstream ranking reads.
                continue
            raw_type = (relation_out.type or _FALLBACK_REL_TYPE).strip()
            rel_type = normalize_relation_type(raw_type)
            if rel_type != raw_type.upper():
                result.types_coerced += 1
            relation_key = (source.casefold(), rel_type, target.casefold())
            existing_relation = relations.get(relation_key)
            if existing_relation is None:
                relations[relation_key] = Relation(
                    source=source,
                    target=target,
                    type=rel_type,
                    description=(relation_out.description or "").strip(),
                    # The schema's 0-10 salience is per-chunk evidence strength.
                    # Cross-chunk accumulation happens in resolution and in the
                    # store, so the mention keeps its own number unscaled.
                    weight=float(relation_out.weight),
                    source_chunk_ids=(chunk.id,) if chunk.id else (),
                    metadata=self._provenance(chunk),
                )
                continue
            # Same edge asserted twice inside one chunk (initial pass plus a
            # gleaning). That is a repeat, not corroboration: take the stronger
            # salience rather than summing it.
            existing_relation.weight = max(existing_relation.weight, float(relation_out.weight))
            description = (relation_out.description or "").strip()
            if description and description not in existing_relation.description:
                existing_relation.description = (
                    f"{existing_relation.description}\n{description}"
                    if existing_relation.description
                    else description
                )

    def _prune_dangling(
        self,
        chunk: Chunk,
        entities: dict[str, Entity],
        relations: dict[tuple[str, str, str], Relation],
        result: GraphExtraction,
    ) -> list[Relation]:
        """Drop edges whose endpoints were never extracted; snap the rest.

        Endpoints that *do* match are rewritten to the entity's canonical
        spelling. Without that rewrite an edge asserted as ``Acme`` against an
        entity extracted as ``Acme Corporation`` would merge a second node in
        the store — the exact fragmentation this validation exists to prevent.
        """
        kept: list[Relation] = []
        dropped: list[tuple[str, str, str]] = []
        for relation in relations.values():
            source = entities.get(relation.source.casefold())
            target = entities.get(relation.target.casefold())
            if source is None or target is None:
                dropped.append(
                    (
                        relation.source,
                        relation.target,
                        "source" if source is None else "target",
                    )
                )
                continue
            relation.source = source.name
            relation.target = target.name
            kept.append(relation)
        if dropped:
            result.dangling_dropped += len(dropped)
            log.warning(
                "graph_dangling_relations_dropped",
                chunk_id=chunk.id,
                dropped=len(dropped),
                kept=len(kept),
                sample=dropped[:3],
                reason="endpoint_not_in_extracted_entities",
            )
        return kept

    def _coerce_type(self, raw: str) -> tuple[str, int]:
        """Map a model-authored type onto ``graph.entity_types``.

        An out-of-vocabulary type is coerced, not rejected: the entity itself is
        real information and losing a node over a label mismatch is a worse
        outcome than filing it under the generic type.
        """
        candidate = _WHITESPACE.sub("_", (raw or "").strip()).upper()
        known = self._allowed_types.get(candidate)
        if known is not None:
            return known, 0
        if not self._entity_types:
            return candidate or self._fallback_type, 0
        return self._fallback_type, 1

    @staticmethod
    def _provenance(chunk: Chunk) -> dict[str, Any]:
        """Document provenance on the mention.

        Kept minimal on purpose: this dict is serialized to JSON on the node,
        and copying a chunk's whole metadata onto every entity it mentions
        multiplies the graph's storage by the metadata size.
        """
        meta: dict[str, Any] = {}
        if chunk.document_id:
            meta["document_id"] = chunk.document_id
        if chunk.tenant_id:
            meta["tenant_id"] = chunk.tenant_id
        return meta

    @staticmethod
    def _render_existing(
        entities: dict[str, Entity], relations: dict[tuple[str, str, str], Relation]
    ) -> str:
        """Compact rendering of what has been found, for the gleaning prompt.

        Names and types only. Descriptions would triple the prompt for no gain:
        the list exists to tell the model what *not* to repeat, and a name is
        enough to recognize a repeat.
        """
        lines = [f"- {entity.name} ({entity.type})" for entity in entities.values()]
        lines.extend(
            f"- {relation.source} -[{relation.type}]-> {relation.target}"
            for relation in relations.values()
        )
        return "\n".join(lines) if lines else "(nothing yet)"
