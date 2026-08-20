"""Structured-output schemas.

Every non-generative LLM call in this library is schema-constrained. That is a
deliberate architectural stance: a router that returns prose has to be parsed
with a regex, and that regex is where your pipeline breaks at 3am. These models
are passed to ``LLM.structured()``, which sends them as a JSON Schema in
``response_format`` and validates the reply with pydantic's Rust core.

Field descriptions are not documentation — they are *part of the prompt*. The
schema is serialized into the request, so a well-worded ``description`` steers
the model as effectively as an instruction in the prompt body.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "AnswerWithCitations",
    "ClaimList",
    "ClaimVerdict",
    "CommunityReport",
    "CompressedExcerpt",
    "CypherQuery",
    "DecompositionOutput",
    "EntityOut",
    "ExtractionOutput",
    "GroundednessGrade",
    "HyDEOutput",
    "MetadataFilterOutput",
    "MultiQueryOutput",
    "PropositionOutput",
    "RankOrder",
    "RelationOut",
    "RelevanceGrade",
    "RouteOutput",
    "SQLQuery",
    "StepBackOutput",
    "SufficiencyCheck",
    "SummaryOutput",
    "UtilityGrade",
]


# ---------------------------------------------------------------------------
# Query translation
# ---------------------------------------------------------------------------
class MultiQueryOutput(BaseModel):
    """Multi-Query / RAG-Fusion: several phrasings of one question.

    Distinct *vocabulary* is what matters. Three paraphrases that share the
    same nouns retrieve the same documents and buy nothing.
    """

    queries: list[str] = Field(
        description=(
            "Alternative phrasings of the user's question, each from a different "
            "angle and using different vocabulary. Do not include the original."
        ),
        min_length=1,
        max_length=8,
    )


class StepBackOutput(BaseModel):
    """Step-back prompting: ask the more general question first.

    "Which team did Messi play for in 2005?" steps back to "What is Messi's
    club history?" — the general question retrieves the passage that contains
    the specific answer, which the specific question often misses.
    """

    step_back_question: str = Field(
        description="A more general, higher-level question whose answer provides "
        "the background needed to answer the original question."
    )
    reasoning: str = Field(default="", description="Why this generalization helps.")


class DecompositionOutput(BaseModel):
    """Break a compound question into independently answerable sub-questions."""

    sub_questions: list[str] = Field(
        description=(
            "Ordered sub-questions. Each must be answerable on its own from "
            "documents; later ones may depend on earlier answers."
        ),
        min_length=1,
        max_length=8,
    )
    is_decomposable: bool = Field(
        default=True,
        description="False if the question is already atomic and should not be split.",
    )


class HyDEOutput(BaseModel):
    """HyDE: a hypothetical answer document, embedded instead of the question.

    Works because a fake answer lives in the same vector neighbourhood as real
    answers, whereas a question lives in question-space. Factual accuracy of
    the hypothetical document is irrelevant — only its style and terminology
    are used.
    """

    document: str = Field(
        description="A concise, confident passage that reads as if excerpted from "
        "a document answering the question. Plausible prose, no hedging."
    )


class RewriteOutput(BaseModel):
    """RRR / Self-RAG query rewriting after a failed retrieval."""

    rewritten_query: str = Field(description="An improved search query.")
    reasoning: str = Field(default="", description="What was wrong with the previous query.")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class RouteOutput(BaseModel):
    """Logical routing: which datastore(s) can answer this?"""

    datastores: list[Literal["vector", "relational", "graph", "web", "none"]] = Field(
        description=(
            "Datastores to query. 'relational' for aggregations, counts, filters "
            "over structured fields; 'graph' for questions about relationships, "
            "paths or multi-entity connections; 'vector' for semantic/conceptual "
            "questions over prose; 'web' for current events beyond the corpus; "
            "'none' when the question needs no retrieval at all."
        ),
        min_length=1,
    )
    prompt_name: str | None = Field(
        default=None, description="Named prompt template best suited to this question."
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------
class SQLQuery(BaseModel):
    """Text-to-SQL output. Validated by the SQL guard before it ever executes."""

    sql: str = Field(
        description="A single read-only PostgreSQL SELECT statement. No DDL, no DML, "
        "no semicolon-separated statements. Use only the listed tables and columns."
    )
    explanation: str = Field(default="", description="What the query returns.")
    tables_used: list[str] = Field(default_factory=list)


class CypherQuery(BaseModel):
    """Text-to-Cypher output. Validated by the Cypher guard before executing."""

    cypher: str = Field(
        description="A single read-only Cypher MATCH/RETURN query. No CREATE, MERGE, "
        "SET, DELETE or REMOVE. Always include a LIMIT."
    )
    explanation: str = Field(default="")
    labels_used: list[str] = Field(default_factory=list)


class FilterCondition(BaseModel):
    field: str
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "range"]
    value: object


class MetadataFilterOutput(BaseModel):
    """Self-query retriever: split a question into a semantic part and a
    structured metadata filter.

    "papers about diffusion models after 2022 by Sohl-Dickstein" becomes
    query="diffusion models" plus two filters — which is the difference between
    a filtered search and hoping the embedding encoded a date.
    """

    query: str = Field(description="The semantic part of the question, filters removed.")
    conditions: list[FilterCondition] = Field(
        default_factory=list, description="Structured filters implied by the question."
    )
    combinator: Literal["and", "or"] = "and"


# ---------------------------------------------------------------------------
# Grading — CRAG, Self-RAG, noise handling
# ---------------------------------------------------------------------------
class RelevanceGrade(BaseModel):
    """Per-document relevance. The CRAG gate."""

    relevant: bool = Field(
        description="True if the document contains information that "
        "helps answer the question, even partially."
    )
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the judgement.")
    reason: str = Field(default="", max_length=300)


class GroundednessGrade(BaseModel):
    """Self-RAG's ISSUP token: is the answer supported by the context?"""

    grounded: bool = Field(
        description="True only if EVERY factual claim in the answer is supported by "
        "the provided context. Plausible-but-unsupported statements are not grounded."
    )
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims not supported by the context."
    )


class UtilityGrade(BaseModel):
    """Self-RAG's ISUSE token: does the answer actually answer the question?"""

    useful: bool = Field(description="True if the answer addresses what was asked.")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    missing: str = Field(default="", description="What the answer fails to address.")


class SufficiencyCheck(BaseModel):
    """Multi-hop early exit. Most questions need one hop; asking prevents
    paying for three."""

    sufficient: bool = Field(
        description="True if the evidence so far is enough to answer completely."
    )
    missing_information: str = Field(
        default="", description="What is still needed, phrased as a search query."
    )
    next_entities: list[str] = Field(
        default_factory=list, description="Entities worth expanding in the graph next."
    )


# ---------------------------------------------------------------------------
# Reranking & compression
# ---------------------------------------------------------------------------
class RankOrder(BaseModel):
    """RankGPT listwise output: permutation of the input passages."""

    order: list[int] = Field(
        description="Passage numbers from most to least relevant. Every input "
        "passage appears exactly once."
    )


class CompressedExcerpt(BaseModel):
    """Contextual compression: extract only the relevant spans, verbatim."""

    excerpt: str = Field(
        description="Verbatim sentences from the document that are relevant to the "
        "question. Empty string if nothing is relevant. Never paraphrase."
    )
    relevant: bool = True


# ---------------------------------------------------------------------------
# Indexing helpers
# ---------------------------------------------------------------------------
class SummaryOutput(BaseModel):
    summary: str = Field(
        description="A dense, self-contained summary preserving named entities, numbers and dates."
    )
    title: str = Field(default="")


class PropositionOutput(BaseModel):
    """Dense-X: decompose a passage into atomic, context-independent facts.

    Each proposition must survive being read alone, so pronouns are resolved
    into their referents. This is what makes proposition indexing precise.
    """

    propositions: list[str] = Field(
        description="Atomic factual statements. Each is a complete sentence with all "
        "pronouns replaced by the entities they refer to, understandable in isolation.",
        max_length=30,
    )


class ContextualPrefix(BaseModel):
    """Anthropic-style contextual retrieval blurb."""

    context: str = Field(
        description="One or two sentences situating this chunk within the overall "
        "document, for search purposes. Do not summarize the chunk itself."
    )


# ---------------------------------------------------------------------------
# GraphRAG extraction
# ---------------------------------------------------------------------------
class EntityOut(BaseModel):
    name: str = Field(description="Canonical name, as written in the text.")
    type: str = Field(default="CONCEPT", description="One of the allowed entity types.")
    description: str = Field(default="", description="What this entity is, per this text only.")


class RelationOut(BaseModel):
    source: str = Field(description="Name of the source entity, exactly as extracted.")
    target: str = Field(description="Name of the target entity, exactly as extracted.")
    type: str = Field(
        default="RELATED_TO", description="Relationship type in SCREAMING_SNAKE_CASE."
    )
    description: str = Field(default="", description="How they are related.")
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Strength/salience 0-10.")


class ExtractionOutput(BaseModel):
    entities: list[EntityOut] = Field(default_factory=list, max_length=60)
    relations: list[RelationOut] = Field(default_factory=list, max_length=100)


class CommunityReport(BaseModel):
    """A GraphRAG community summary — the unit of global search."""

    title: str = Field(description="Short name for what this cluster of entities is about.")
    summary: str = Field(
        description="Self-contained report on the community's entities, "
        "their relationships and their significance."
    )
    rating: float = Field(
        default=5.0, ge=0.0, le=10.0, description="Importance of this community, 0-10."
    )
    findings: list[str] = Field(default_factory=list, max_length=10)


class MapAnswer(BaseModel):
    """Map step of global search: answer from one community, with a score so
    the reduce step can rank contributions."""

    answer: str = Field(description="Partial answer from this community, or empty if irrelevant.")
    score: float = Field(default=0.0, ge=0.0, le=10.0, description="Helpfulness 0-10.")


# ---------------------------------------------------------------------------
# Generation & verification
# ---------------------------------------------------------------------------
class CitedStatement(BaseModel):
    text: str
    source_ids: list[int] = Field(
        default_factory=list, description="Numbers of the context passages supporting this."
    )


class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The answer, grounded strictly in the provided context.")
    statements: list[CitedStatement] = Field(
        default_factory=list, description="The answer split into statements with their sources."
    )
    sufficient: bool = Field(
        default=True, description="False if the context did not contain enough to answer."
    )


class ClaimList(BaseModel):
    """Atomic claim decomposition for fine-grained hallucination checking."""

    claims: list[str] = Field(
        description="Atomic, independently verifiable factual claims from the answer. "
        "Exclude hedges, opinions and restatements of the question.",
        max_length=40,
    )


class ClaimVerdict(BaseModel):
    """NLI-style verdict for one claim against the evidence."""

    verdict: Literal["supported", "contradicted", "not_enough_info"]
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quote: str = Field(default="", description="Verbatim supporting span, if any.")
