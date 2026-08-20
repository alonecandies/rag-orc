"""Text-to-Cypher: natural language into a guarded, executed graph query.

Why a graph leg at all
----------------------
Some answers are not in any document because they are *relationships between*
documents. "How is Northwind connected to Contoso" may have no passage stating it:
Northwind's supplier is mentioned in one file, that supplier's parent company in
another, and the connection exists only as a path. A vector search over either
file retrieves neither answer, because neither file contains one.

Cypher expresses that traversal directly, which is the justification for asking a
model to write it — and, as with SQL, for fencing what it writes.

Why the guard is lexical here, and why that is still sound
----------------------------------------------------------
There is no maintained Python Cypher parser, so
:class:`~ragorc.security.cypher_guard.CypherGuard` cannot validate an AST the way
the SQL guard does. It compensates by **normalizing before scanning**: string
literals and comments are blanked out, so a keyword inside a value can neither
trigger a false rejection nor hide a real clause. That single step is what
separates a usable lexical guard from the naive substring check that both rejects
``WHERE n.note = 'DELETE ME'`` and misses ``DETACH  DELETE``.

Two hazards get dedicated treatment because they have no SQL analogue:

* **Unbounded traversal.** ``MATCH (a)-[*]-(b)`` walks the whole graph. It is not
  a slow query, it is an outage, and the guard requires an upper bound.
* **Procedures.** ``CALL apoc.load.json('http://169.254.169.254/…')`` is
  server-side request forgery from inside the database. Procedures are
  allowlisted rather than blocklisted, because the harmful set is open-ended and
  the set we need is a dozen names.

The optional ``EXPLAIN`` dry run adds a check we cannot perform ourselves: Neo4j
compiles and plans the query without touching data, which catches syntax the
lexical guard cannot see.

Why graph rows need verbalizing
-------------------------------
A Neo4j driver returns ``Node`` and ``Relationship`` objects. Handed to a model as
a repr, they are unreadable noise; handed to ``orjson``, they raise. ``to_chunks``
renders them as ``A -[WORKS_FOR]-> B`` sentences, which is both what the model can
use and what a citation can quote.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import structlog

from ragorc.core.errors import ConstructionError, GuardrailViolation
from ragorc.core.ids import content_hash, stable_uuid
from ragorc.core.models import Chunk, Modality, Query, RetrievalSource, ScoredChunk, Usage
from ragorc.core.protocols import LLM, GraphStore
from ragorc.core.registry import register
from ragorc.core.schemas import CypherQuery
from ragorc.core.settings import Settings, get_settings
from ragorc.llm.prompts import get_prompt
from ragorc.llm.router import ModelRouter, Task
from ragorc.security.audit import AuditLog
from ragorc.security.cypher_guard import CypherGuard, CypherValidation
from ragorc.security.tenancy import require_generated_query_isolation

log = structlog.get_logger(__name__)

__all__ = ["TextToCypherConstructor"]

#: A graph query either matches a pattern or it does not — there is no graded
#: relevance to report. A fixed, deliberately-below-1.0 score marks these results
#: as exact-but-not-ranked, which is the signal cross-store fusion needs.
_EXACT_MATCH_SCORE = 0.95
_MAX_VERBALIZED_ROWS = 40


@register("constructor", "text_to_cypher", "cypher")
class TextToCypherConstructor:
    """Builds, validates and (optionally) runs read-only Cypher for a question."""

    name = "text_to_cypher"
    target = "cypher"

    def __init__(
        self,
        llm: LLM,
        store: GraphStore | None = None,
        *,
        guard: CypherGuard | None = None,
        audit: AuditLog | None = None,
        router: ModelRouter | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.store = store
        self.max_rows = self.settings.neo4j.max_cypher_rows
        self.guard = guard or CypherGuard(
            self.settings.security,
            max_rows=self.max_rows,
            max_hops=self.settings.graph.multihop_max_path_length,
        )
        self.audit = audit or AuditLog(self.settings)
        self.router = router or ModelRouter(self.settings.llm)
        self.prompt = get_prompt("text_to_cypher")
        # The guard appends or clamps LIMIT regardless; stating the ceiling in the
        # prompt means the query the model wrote is the query that runs.
        self.system = self.prompt.system.format(max_rows=self.max_rows)

    # -- construction ------------------------------------------------------
    async def construct(
        self, query: Query, store: GraphStore | None = None, **kwargs: Any
    ) -> tuple[CypherValidation, Usage]:
        """Return validated Cypher plus the token cost of producing it."""
        target = self._require_store(store)
        schema = await target.schema_summary()
        model = self.router.model_for(Task.SELF_QUERY)

        generated, usage = await self._ask(schema, query.text, model)
        try:
            validated = self.guard.validate(generated.cypher)
        except GuardrailViolation as first:
            self.audit.generated_query("cypher", generated.cypher, allowed=False, rule=first.rule)
            log.info("cypher_rejected", rule=first.rule, attempt=1)
            repaired, repair_usage = await self._repair(
                schema, query.text, generated.cypher, str(first), model
            )
            usage = usage + repair_usage
            try:
                validated = self.guard.validate(repaired.cypher)
            except GuardrailViolation as second:
                self.audit.generated_query(
                    "cypher", repaired.cypher, allowed=False, rule=second.rule
                )
                raise ConstructionError(
                    "could not produce a permitted Cypher query",
                    rule=second.rule,
                    detail_message=str(second),
                    question=query.text[:200],
                ) from second
            generated = repaired

        self.audit.generated_query("cypher", validated.cypher, allowed=True)

        # A planning dry run costs one round trip and no data access, and catches
        # the syntax errors a lexical guard structurally cannot.
        if self.settings.security.cypher_explain_dryrun:
            try:
                await self.guard.explain(target, validated.cypher)
            except GuardrailViolation as exc:
                raise ConstructionError(
                    "generated Cypher failed to plan",
                    rule=exc.rule,
                    cypher=validated.cypher[:300],
                ) from exc

        validated.metadata.update(
            {
                "explanation": generated.explanation,
                "labels_used": list(generated.labels_used),
            }
        )
        log.info(
            "cypher_constructed",
            hops=validated.max_hops,
            procedures=validated.procedures,
            warnings=validated.warnings,
        )
        return validated, usage

    async def _ask(self, schema: str, question: str, model: str) -> tuple[CypherQuery, Usage]:
        return await self.llm.structured(
            self.prompt.render(schema=schema, question=question, max_rows=self.max_rows),
            CypherQuery,
            system=self.system,
            model=model,
            stage="text_to_cypher",
        )

    async def _repair(
        self, schema: str, question: str, rejected: str, reason: str, model: str
    ) -> tuple[CypherQuery, Usage]:
        """One repair attempt, reusing the same system block.

        Keeping the system prompt byte-identical is what lets the provider's
        prompt cache hit on the second call — the schema is the expensive part of
        this prompt, and it has not changed.
        """
        feedback = (
            f"{question}\n\n"
            f"Your previous query was REJECTED by the security guard:\n{rejected}\n\n"
            f"Reason: {reason}\n\n"
            "Write a corrected read-only query. If the rejection was for a forbidden "
            "clause or procedure, do not attempt it again by another route."
        )
        return await self.llm.structured(
            self.prompt.render(schema=schema, question=feedback, max_rows=self.max_rows),
            CypherQuery,
            system=self.system,
            model=model,
            stage="text_to_cypher_repair",
        )

    # -- execution ---------------------------------------------------------
    async def construct_and_execute(
        self, query: Query, store: GraphStore | None = None, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], CypherValidation, Usage]:
        # Checked before the model is even asked. This library cannot scope a
        # generated statement to a tenant — where a predicate belongs across
        # joins and CTEs is a schema question — so with tenant isolation on, this
        # leg refuses unless the operator has declared how isolation is actually
        # enforced. The gap this closes was silent: tenant_id reached here only to
        # stamp the resulting chunk, never to filter the query, so the vector leg
        # was scoped and the cypher leg read every tenant's rows.
        require_generated_query_isolation("Cypher", self.settings)
        target = self._require_store(store)
        validated, usage = await self.construct(query, target, **kwargs)
        rows = await target.execute_readonly(validated.cypher, limit=self.max_rows)
        log.info("cypher_executed", rows=len(rows))
        return rows, validated, usage

    def _require_store(self, store: GraphStore | None) -> GraphStore:
        target = store or self.store
        if target is None:
            raise ConstructionError(
                "text-to-Cypher needs a GraphStore",
                hint="pass store= to the constructor or to construct()",
            )
        return target

    # -- rendering ---------------------------------------------------------
    def to_chunks(
        self,
        rows: Sequence[dict[str, Any]],
        cypher: str,
        *,
        tenant_id: str | None = None,
        score: float = _EXACT_MATCH_SCORE,
    ) -> list[ScoredChunk]:
        """Render graph rows as retrieval evidence.

        One chunk for the whole result, for the same reason SQL rows are: a path
        is only meaningful beside the other paths, and splitting them lets
        reranking silently drop half of an answer.

        Rendering rules, chosen for what a model reads reliably:

        * A row containing a path becomes ``A -[TYPE]-> B -[TYPE]-> C``. Arrows
          carry direction, which matters — "Acme acquired Beta" and "Beta acquired
          Acme" are different facts, and a bare node list loses that.
        * A row of scalars becomes ``key: value`` lines, which survive reordering
          by the context packer and can each be quoted independently.
        * Nodes render as ``Label(name)`` with their scalar properties, never as a
          driver repr — ``<Node element_id='4:...'>`` is noise the model has to
          discard, and internal element ids are meaningless outside the database.
        """
        if not rows:
            log.info("cypher_no_rows", cypher=cypher[:200])
            return []

        lines: list[str] = []
        for row in rows[:_MAX_VERBALIZED_ROWS]:
            rendered = _render_row(row)
            if rendered:
                lines.append(rendered)
        if not lines:
            return []
        if len(rows) > _MAX_VERBALIZED_ROWS:
            # State the truncation rather than hiding it: a model told it is
            # seeing part of a result set hedges, where one that is not asserts
            # completeness it cannot know.
            lines.append(f"… {len(rows) - _MAX_VERBALIZED_ROWS} further rows not shown")

        content = "\n".join(lines)
        chunk_key = content_hash(cypher, content)
        chunk = Chunk(
            id=stable_uuid("cypher", tenant_id or "", chunk_key),
            content=content,
            document_id="graph://cypher",
            modality=Modality.TABLE,
            tenant_id=tenant_id,
            metadata={
                "source": "neo4j",
                "cypher": cypher,
                "row_count": len(rows),
                "truncated": len(rows) > _MAX_VERBALIZED_ROWS,
            },
        )
        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                source=RetrievalSource.CYPHER,
                rank=0,
                component_scores={"cypher": score},
                explain={"rows": len(rows)},
            )
        ]


# ---------------------------------------------------------------------------
# Row rendering
# ---------------------------------------------------------------------------
def _render_row(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in row.items():
        rendered = _render_value(value)
        if not rendered:
            continue
        # A verbalized path is already a sentence; prefixing it with the return
        # alias ("p: A -[X]-> B") only adds noise.
        parts.append(rendered if _is_path_like(value) else f"{key}: {rendered}")
    return " | ".join(parts)


def _is_path_like(value: Any) -> bool:
    return hasattr(value, "relationships") or hasattr(value, "nodes")


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if _is_path_like(value):
        return _render_path(value)
    if _looks_like_node(value):
        return _render_node(value)
    if _looks_like_relationship(value):
        return _render_relationship(value)
    if isinstance(value, (list, tuple, set)):
        rendered = [_render_value(v) for v in value]
        return ", ".join(r for r in rendered if r)
    if isinstance(value, dict):
        inner = ", ".join(f"{k}={_scalar(v)}" for k, v in value.items() if v is not None)
        return f"{{{inner}}}"
    return _scalar(value)


def _looks_like_node(value: Any) -> bool:
    return hasattr(value, "labels") and hasattr(value, "items")


def _looks_like_relationship(value: Any) -> bool:
    return hasattr(value, "type") and hasattr(value, "start_node")


def _render_node(node: Any) -> str:
    labels = ":".join(sorted(getattr(node, "labels", ()) or ())) or "Node"
    properties = dict(node.items()) if hasattr(node, "items") else {}
    name = (
        properties.pop("name", None) or properties.pop("title", None) or properties.pop("id", None)
    )
    # Embeddings and other vector properties are large and useless in a prompt.
    scalars = {
        k: v
        for k, v in properties.items()
        if v is not None and not isinstance(v, (list, tuple, dict, bytes))
    }
    head = f"{labels}({_scalar(name)})" if name is not None else labels
    if not scalars:
        return head
    inner = ", ".join(f"{k}={_scalar(v)}" for k, v in list(scalars.items())[:6])
    return f"{head}[{inner}]"


def _render_relationship(relationship: Any) -> str:
    rel_type = getattr(relationship, "type", "RELATED_TO")
    start = getattr(relationship, "start_node", None)
    end = getattr(relationship, "end_node", None)
    if start is not None and end is not None:
        return f"{_node_name(start)} -[{rel_type}]-> {_node_name(end)}"
    return f"-[{rel_type}]->"


def _render_path(path: Any) -> str:
    nodes = list(getattr(path, "nodes", ()) or ())
    relationships = list(getattr(path, "relationships", ()) or ())
    if not nodes:
        return ""
    if not relationships:
        return " -> ".join(_node_name(n) for n in nodes)

    parts: list[str] = [_node_name(nodes[0])]
    for i, relationship in enumerate(relationships):
        rel_type = getattr(relationship, "type", "RELATED_TO")
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None
        # Direction is recovered from the relationship's own endpoints, because a
        # path can traverse an edge against its direction and the arrow must
        # reflect the assertion, not the walk.
        start = getattr(relationship, "start_node", None)
        forward = start is None or _node_name(start) == parts[-1]
        arrow = f"-[{rel_type}]->" if forward else f"<-[{rel_type}]-"
        parts.append(arrow)
        if nxt is not None:
            parts.append(_node_name(nxt))
    return " ".join(parts)


def _node_name(node: Any) -> str:
    if hasattr(node, "items"):
        properties = dict(node.items())
        for key in ("name", "title", "id"):
            if properties.get(key) is not None:
                return _scalar(properties[key])
    labels = ":".join(sorted(getattr(node, "labels", ()) or ())) or "Node"
    return labels


def _scalar(value: Any) -> str:
    """Render a leaf value as something a model can transcribe without error."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    text = str(value)
    if hasattr(value, "iso_format"):  # neo4j temporal types
        text = value.iso_format()
    return " ".join(text.split())
