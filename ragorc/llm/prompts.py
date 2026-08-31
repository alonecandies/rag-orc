"""The prompt library.

Prompts are treated as versioned assets, not string literals scattered through
the code, for three reasons: they are the highest-leverage tuning surface in the
system, they must be diffable in review, and the semantic router selects between
them *by name* at runtime.

Each prompt separates ``system`` (stable, cacheable, sent with prompt-cache
hints) from ``template`` (per-request). That split is what makes provider prompt
caching effective — a static system block plus a small variable user block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PROMPTS",
    "Prompt",
    "get_prompt",
    "register_prompt",
    "resolve_prompt_name",
]


@dataclass(slots=True, frozen=True)
class Prompt:
    name: str
    template: str
    system: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, **kwargs: Any) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            raise KeyError(f"prompt {self.name!r} is missing variable {exc}") from exc


PROMPTS: dict[str, Prompt] = {}


def register_prompt(prompt: Prompt) -> Prompt:
    PROMPTS[prompt.name] = prompt
    return prompt


def resolve_prompt_name(name: str | None) -> str | None:
    """Accept both a prompt's registered name and its bare form.

    ``generation.prompt_name`` defaults to ``"default"`` while the library
    registers it as ``"answer_default"`` — the same shorthand a user will type when
    they ask for ``"concise"`` or ``"technical"``.

    Defined here, beside :data:`PROMPTS`, because it was defined in
    ``pipeline.nodes`` and therefore reached one of the two wirings that consume a
    routed prompt name: the RAGPipeline node resolved it, and
    ``AnswerGenerator.generate``/``stream`` — the path the HTTP engine takes — did
    not. Two spellings of one predicate is the shape half this library's defects
    take, so :func:`get_prompt` now applies it and there is nothing left to
    remember at a call site.
    """
    if not name:
        return None
    if name in PROMPTS:
        return name
    prefixed = f"answer_{name}"
    return prefixed if prefixed in PROMPTS else name


def get_prompt(name: str) -> Prompt:
    resolved = resolve_prompt_name(name) or name
    try:
        return PROMPTS[resolved]
    except KeyError:
        raise KeyError(f"unknown prompt {name!r}; known: {sorted(PROMPTS)}") from None


def _p(**kwargs: Any) -> None:
    register_prompt(Prompt(**kwargs))


# ===========================================================================
# QUERY TRANSLATION
# ===========================================================================
_p(
    name="multi_query",
    tags=("translate",),
    description="Multi-Query / RAG-Fusion query expansion.",
    system=(
        "You expand search queries for a retrieval system. Vector search fails when "
        "the question's wording does not match the document's wording, so your job is "
        "to cover the vocabulary the answer is likely written in.\n\n"
        "Rules:\n"
        "- Each query must use DIFFERENT terminology from the others. Synonym swaps "
        "on the same sentence are worthless; vary the framing.\n"
        "- Cover different angles: definitional, causal, procedural, comparative.\n"
        "- Keep every query self-contained. Never write 'it' or 'that'.\n"
        "- Preserve proper nouns, identifiers, versions and numbers exactly."
    ),
    template=(
        "Generate {n} alternative search queries for this question.\n\nQuestion: {question}\n"
    ),
)

_p(
    name="step_back",
    tags=("translate",),
    description="Step-back prompting: generalize before retrieving.",
    system=(
        "You convert specific questions into the more general question that must be "
        "answered first. Retrieval for a narrow question often misses the passage that "
        "contains the answer, because that passage discusses the broader topic.\n\n"
        "Example:\n"
        "  Specific: 'Which team did Messi play for in 2005?'\n"
        "  Step-back: 'What is the club career history of Lionel Messi?'\n\n"
        "The step-back question must be broad enough to retrieve context, but still "
        "about the same subject. Do not make it so generic it loses the topic."
    ),
    template="Question: {question}\n\nProduce the step-back question.",
)

_p(
    name="decomposition",
    tags=("translate",),
    description="Break a compound question into ordered sub-questions.",
    system=(
        "You decompose complex questions into smaller sub-questions that a document "
        "retrieval system can answer one at a time.\n\n"
        "Rules:\n"
        "- Order them so that answering them in sequence builds toward the final answer.\n"
        "- Each sub-question must be independently searchable and self-contained: "
        "repeat the entity names instead of using pronouns.\n"
        "- Do NOT decompose a question that is already atomic. Set is_decomposable=false "
        "and return the question unchanged.\n"
        "- Comparisons decompose into one sub-question per item plus the comparison itself."
    ),
    template="Question: {question}\n\nDecompose into at most {max_sub} sub-questions.",
)

_p(
    name="hyde",
    tags=("translate",),
    description="HyDE: generate a hypothetical answer document to embed.",
    system=(
        "You write a passage that looks exactly like an excerpt from a document which "
        "answers the user's question. This passage is never shown to a user — it is "
        "embedded and used as the search vector, because a hypothetical answer sits "
        "nearer to real answers in embedding space than a question does.\n\n"
        "Therefore:\n"
        "- Write in the declarative, confident register of reference material.\n"
        "- Use the domain's technical vocabulary densely.\n"
        "- Do NOT hedge, do NOT say 'I don't know', do NOT mention the question.\n"
        "- Factual accuracy is irrelevant. Plausible terminology is what matters.\n"
        "- 3-6 sentences."
    ),
    template="Question: {question}\n\nWrite the passage.",
)

_p(
    name="rewrite_query",
    tags=("translate", "generate"),
    description="RRR / Self-RAG query rewriting after a retrieval failure.",
    system=(
        "You rewrite failed search queries. The previous query retrieved nothing "
        "useful, so identify why — wrong vocabulary, too narrow, too broad, ambiguous, "
        "or containing terms unlikely to appear in documents — and fix that.\n"
        "Return a query optimized for document retrieval, not for conversation."
    ),
    template=(
        "Original question: {question}\n"
        "Previous query: {previous}\n"
        "Retrieved (unhelpful): {retrieved}\n\n"
        "Write a better search query."
    ),
)


# ===========================================================================
# ROUTING
# ===========================================================================
_p(
    name="logical_route",
    tags=("route",),
    description="Choose which datastore(s) can answer the question.",
    system=(
        "You route questions to the datastore that can actually answer them.\n\n"
        "relational (PostgreSQL) — structured records. Choose it for counts, sums, "
        "averages, rankings, date ranges, filters on known columns, 'how many', "
        "'top N', 'between X and Y'.\n"
        "graph (Neo4j) — entities and their relationships. Choose it for connections, "
        "paths, influence, 'who works with', 'how is A related to B', questions "
        "needing several hops between entities.\n"
        "vector (Qdrant) — unstructured prose. Choose it for concepts, explanations, "
        "definitions, opinions, procedures, 'what is', 'why does', 'how do I'.\n"
        "web — information that post-dates or falls outside the indexed corpus.\n"
        "none — greetings, meta-questions, or anything answerable without retrieval.\n\n"
        "Choose MULTIPLE stores when the question genuinely needs them (e.g. 'summarize "
        "the top 3 customers by revenue' needs relational for the ranking and vector "
        "for the summaries). Do not select every store as a hedge."
    ),
    template=("Available data:\n{schema_hint}\n\nQuestion: {question}\n\nRoute it."),
)


# ===========================================================================
# QUERY CONSTRUCTION
# ===========================================================================
_p(
    name="text_to_sql",
    tags=("construct",),
    description="Natural language to read-only PostgreSQL.",
    system=(
        "You write PostgreSQL SELECT queries against the schema you are given.\n\n"
        "Hard rules:\n"
        "- Read-only. SELECT or WITH only. Never INSERT, UPDATE, DELETE, DROP, ALTER, "
        "CREATE, GRANT, COPY, or call filesystem functions.\n"
        "- One statement. No semicolon-separated batches, no comments.\n"
        "- Use only tables and columns present in the schema. If the schema cannot "
        "answer the question, return the closest valid query and say so in explanation.\n"
        "- Always include LIMIT {max_rows}.\n"
        "- Quote identifiers that need it; never interpolate user text into a string "
        "literal without escaping it.\n"
        "- Prefer explicit JOIN ... ON over implicit comma joins.\n"
        "- For text matching use ILIKE with wildcards, or full-text search if a tsvector "
        "column exists.\n"
        "- Cast dates explicitly (DATE '2024-01-01'), never rely on locale parsing."
    ),
    template="Schema:\n{schema}\n\nQuestion: {question}\n\nWrite the query.",
)

_p(
    name="text_to_cypher",
    tags=("construct",),
    description="Natural language to read-only Cypher.",
    system=(
        "You write read-only Neo4j Cypher queries against the schema you are given.\n\n"
        "Hard rules:\n"
        "- MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT only.\n"
        "- Never CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, LOAD CSV, or CALL "
        "procedures that write.\n"
        "- Use only the labels, relationship types and properties in the schema, with "
        "exactly the casing shown.\n"
        "- Always include LIMIT {max_rows}.\n"
        "- Bound variable-length patterns: write *1..3, never a bare *, which can "
        "traverse the entire graph.\n"
        "- Return specific properties, not whole nodes, so the result is compact.\n"
        "- Use toLower() for case-insensitive comparison on names."
    ),
    template="Graph schema:\n{schema}\n\nQuestion: {question}\n\nWrite the query.",
)

_p(
    name="self_query",
    tags=("construct",),
    description="Split a question into semantic text plus metadata filters.",
    system=(
        "You separate a search request into two parts: the semantic content to embed, "
        "and structured filters over metadata.\n\n"
        "Rules:\n"
        "- The query string must contain ONLY the conceptual part. Strip every "
        "constraint you turned into a filter, or it will be double-counted.\n"
        "- Emit a filter only for fields in the provided metadata schema. Never invent "
        "a field name.\n"
        "- Map comparatives correctly: 'after 2022' is gt, 'since 2022' is gte, "
        "'in 2022' is a range over that year.\n"
        "- If nothing in the question maps to a field, return no conditions.\n\n"
        "Example: 'papers on diffusion models published after 2022 by Ho'\n"
        "  query: 'diffusion models'\n"
        "  conditions: [year gt 2022, author eq 'Ho']"
    ),
    template="Metadata schema:\n{schema}\n\nRequest: {question}\n\nSplit it.",
)


# ===========================================================================
# GRADING — CRAG / Self-RAG / noise handling
# ===========================================================================
_p(
    name="grade_relevance",
    tags=("retrieve", "grade"),
    description="CRAG document relevance gate.",
    system=(
        "You judge whether a retrieved document helps answer a question.\n\n"
        "Be generous about partial usefulness and strict about topical drift. A "
        "document is relevant if it contains any fact, definition, or context that "
        "contributes to an answer — it does not have to contain the whole answer.\n"
        "A document that merely shares keywords with the question, or discusses a "
        "different entity with a similar name, is NOT relevant.\n"
        "Judge the document as written. Do not use your own knowledge to fill gaps."
    ),
    template="Question: {question}\n\nDocument:\n{document}\n\nIs it relevant?",
)

_p(
    name="grade_groundedness",
    tags=("generate", "grade"),
    description="Self-RAG ISSUP: is the answer supported by the context?",
    system=(
        "You detect unsupported statements. You are given a context and an answer that "
        "was supposed to be derived from it.\n\n"
        "An answer is grounded ONLY IF every factual claim in it can be traced to the "
        "context. Apply these tests:\n"
        "- A claim that is true in the world but absent from the context is NOT grounded.\n"
        "- A number, date, name or quantity that differs from the context is NOT grounded.\n"
        "- A causal or comparative claim the context does not make is NOT grounded, "
        "even if both facts appear separately.\n"
        "- Hedges ('may', 'typically') and explicit statements that information is "
        "missing do not require support.\n\n"
        "List every unsupported claim verbatim."
    ),
    template="Context:\n{context}\n\nAnswer:\n{answer}\n\nIs the answer grounded?",
)

_p(
    name="grade_utility",
    tags=("generate", "grade"),
    description="Self-RAG ISUSE: does the answer address the question?",
    system=(
        "You judge whether an answer actually answers the question asked — not whether "
        "it is true, and not whether it is well written.\n"
        "An answer that is accurate but responds to a different question is not useful. "
        "An answer that only partially covers a multi-part question is partially useful; "
        "say what is missing."
    ),
    template="Question: {question}\n\nAnswer:\n{answer}\n\nIs it useful?",
)

_p(
    name="rank_gpt",
    tags=("retrieve", "rerank"),
    description="RankGPT listwise permutation reranking.",
    system=(
        "You rank passages by how well they answer a search query.\n"
        "Return a permutation of the passage numbers, most relevant first. Every "
        "passage number appears exactly once. Judge only the passages given; do not "
        "add outside knowledge."
    ),
    template=(
        "Query: {question}\n\n{passages}\n\nRank all {n} passages from most to least relevant."
    ),
)

_p(
    name="compress_extract",
    tags=("retrieve", "compress"),
    description="Contextual compression: extract relevant spans verbatim.",
    system=(
        "You extract the parts of a document that are relevant to a question.\n\n"
        "Rules:\n"
        "- Copy sentences VERBATIM. Never paraphrase, summarize or merge sentences: "
        "downstream citation verification checks these spans against the source.\n"
        "- Include enough surrounding text that each extract stands alone.\n"
        "- If nothing in the document is relevant, return an empty excerpt and "
        "relevant=false. Returning something irrelevant is worse than returning nothing."
    ),
    template="Question: {question}\n\nDocument:\n{document}\n\nExtract the relevant parts.",
)


# ===========================================================================
# INDEXING
# ===========================================================================
_p(
    name="summarize_chunk",
    tags=("index",),
    description="Multi-representation indexing / RAPTOR node summary.",
    system=(
        "You write dense summaries for a retrieval index. The summary replaces the "
        "original text as the search target, so it must retain everything searchable:\n"
        "- every named entity, identifier, version number, date and quantity;\n"
        "- the specific terminology used, not generic paraphrases;\n"
        "- the main claims, in the order they appear.\n"
        "Omit filler, examples and repetition. No preamble such as 'This document "
        "discusses' — start with the content itself."
    ),
    template="Summarize the following in at most {max_tokens} tokens.\n\n{text}",
)

_p(
    name="raptor_summary",
    tags=("index", "raptor"),
    description="RAPTOR cluster summary across sibling chunks.",
    system=(
        "You summarize a cluster of related passages into one higher-level node of a "
        "hierarchical index. Your summary will be retrieved instead of the passages "
        "when a question is broad, so it must:\n"
        "- state what the whole cluster is about and how the passages relate;\n"
        "- keep every distinct entity and figure that appears;\n"
        "- surface themes that no single passage states alone.\n"
        "Do not enumerate the passages. Write one coherent synthesis."
    ),
    template="Passages:\n\n{texts}\n\nWrite the cluster summary ({max_tokens} tokens max).",
)

_p(
    name="propositions",
    tags=("index", "dense_x"),
    description="Dense-X: decompose a passage into atomic propositions.",
    system=(
        "You decompose text into atomic propositions for a precision-oriented index.\n\n"
        "Each proposition must:\n"
        "- express exactly ONE fact;\n"
        "- be a complete, grammatical sentence;\n"
        "- resolve every pronoun and every 'the company'/'this method' style reference "
        "into the explicit entity name — a proposition is retrieved without its "
        "neighbours, so an unresolved reference makes it useless;\n"
        "- stay faithful. Add nothing that is not stated.\n\n"
        "Split compound sentences. Keep lists as separate propositions."
    ),
    template="Text:\n{text}\n\nDecompose into propositions.",
)

_p(
    name="contextual_prefix",
    tags=("index", "contextual"),
    description="Anthropic-style contextual retrieval prefix.",
    system=(
        "You situate a chunk inside its document so it can be found by search.\n\n"
        "Write 1-2 sentences that say what part of the document this chunk belongs to "
        "and what entity or topic it concerns — information the chunk itself omits "
        "because the surrounding document supplied it.\n"
        "Do NOT summarize the chunk's own content: that is already indexed. Supply the "
        "missing context only. Answer with the context sentences and nothing else."
    ),
    template=(
        "<document>\n{document}\n</document>\n\n"
        "<chunk>\n{chunk}\n</chunk>\n\n"
        "Give the situating context for this chunk."
    ),
)


# ===========================================================================
# GRAPHRAG
# ===========================================================================
_p(
    name="extract_graph",
    tags=("graph", "index"),
    description="Entity and relationship extraction for the knowledge graph.",
    system=(
        "You build a knowledge graph from text.\n\n"
        "Entities:\n"
        "- Extract only entities of these types: {entity_types}.\n"
        "- Use the entity's most complete name as it appears in the text ('Acme "
        "Corporation', not 'Acme' or 'the company'). Consistent naming is what lets "
        "the graph connect mentions across documents — inconsistent naming fragments it.\n"
        "- Describe each entity using only what this text says about it.\n\n"
        "Relationships:\n"
        "- Both endpoints must be entities you extracted, named identically.\n"
        "- Type is SCREAMING_SNAKE_CASE and specific: WORKS_FOR, ACQUIRED, LOCATED_IN, "
        "AUTHORED, COMPETES_WITH — not the generic RELATED_TO unless nothing better fits.\n"
        "- Weight 0-10 by how central the relationship is to this text.\n"
        "- Extract only relationships the text asserts. Do not infer from world knowledge."
    ),
    template="Text:\n{text}\n\nExtract the graph.",
)

_p(
    name="extract_gleaning",
    tags=("graph", "index"),
    description="Follow-up extraction pass for missed entities.",
    system=(
        "You review an extraction for omissions. Many entities were found already; "
        "your job is only to add what was MISSED. Do not repeat anything already "
        "listed, and do not lower the bar — return an empty list if nothing was missed."
    ),
    template=(
        "Text:\n{text}\n\nAlready extracted:\n{existing}\n\n"
        "Extract only entities and relationships that were missed."
    ),
)

_p(
    name="community_report",
    tags=("graph",),
    description="GraphRAG community summary for global search.",
    system=(
        "You write a report on a community of related entities from a knowledge graph.\n\n"
        "The report is the only thing a global search sees, so it must stand alone:\n"
        "- title: what this community IS, in a few words;\n"
        "- summary: the entities, how they relate, and why the community matters;\n"
        "- rating 0-10 for how significant this community is;\n"
        "- findings: the specific, non-obvious insights that follow from these "
        "relationships taken together.\n"
        "Ground every statement in the supplied entities and relationships."
    ),
    template=(
        "Entities:\n{entities}\n\nRelationships:\n{relations}\n\nWrite the community report."
    ),
)

_p(
    name="global_map",
    tags=("graph",),
    description="Map step of GraphRAG global search.",
    system=(
        "You answer a question using ONE community report from a knowledge graph.\n"
        "Extract only what this report contributes. If it contributes nothing, return "
        "an empty answer with score 0 — a confident irrelevant answer poisons the "
        "reduce step. Score 0-10 by how much this report helps."
    ),
    template="Question: {question}\n\nCommunity report:\n{report}\n\nWhat does it contribute?",
)

_p(
    name="global_reduce",
    tags=("graph",),
    description="Reduce step of GraphRAG global search.",
    system=(
        "You synthesize partial answers from different parts of a knowledge graph into "
        "one response.\n"
        "Weight the partials by their scores, merge overlapping points, and keep "
        "disagreements visible rather than averaging them away. Cite which community "
        "each substantive point came from. Ignore empty or zero-scored partials."
    ),
    template="Question: {question}\n\nPartial answers:\n{partials}\n\nSynthesize the answer.",
)

_p(
    name="multihop_reason",
    tags=("graph", "multihop"),
    description="IRCoT-style step: reason, then decide what to retrieve next.",
    system=(
        "You are answering a question that needs several retrieval steps.\n\n"
        "Given what has been retrieved so far:\n"
        "1. State what is now known that bears on the question.\n"
        "2. Decide whether that is sufficient to answer completely.\n"
        "3. If not, say precisely what is still missing, phrased as a search query — "
        "not as a description of the gap.\n"
        "Name the bridge entities worth expanding next.\n"
        "Prefer stopping: an extra hop costs a retrieval and a model call, and most "
        "questions are already answerable after one."
    ),
    template=(
        "Question: {question}\n\nEvidence so far:\n{evidence}\n\n"
        "Previous searches: {history}\n\nAssess sufficiency."
    ),
)


# ===========================================================================
# GENERATION
# ===========================================================================
_p(
    name="answer_default",
    tags=("generate",),
    description="Default grounded RAG answer.",
    system=(
        "You answer questions using only the provided context.\n\n"
        "Rules:\n"
        "- Ground every statement in the context. If the context does not contain the "
        "answer, say so plainly — do not fill the gap from your own knowledge.\n"
        "- Cite sources inline as [1], [2] matching the numbered context passages. "
        "Cite the specific passage that supports each claim, not a range.\n"
        "- Quote exact figures, names and dates from the context; never round or "
        "reformat them.\n"
        "- When passages disagree, report the disagreement and cite both.\n"
        "- Answer at the length the question deserves. No preamble, no restating the "
        "question, no 'based on the provided context'."
    ),
    template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)

_p(
    name="answer_concise",
    tags=("generate",),
    description="Short factual answer with citations.",
    system=(
        "Answer in one to three sentences using only the context. Cite as [n]. "
        "If the context is insufficient, say exactly that and stop."
    ),
    template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)

_p(
    name="answer_technical",
    tags=("generate",),
    description="Technical answer preserving code, config and exact syntax.",
    system=(
        "You answer technical questions from documentation.\n"
        "Reproduce code, commands, configuration keys, flags and version numbers "
        "EXACTLY as they appear in the context — approximate syntax is a wrong answer. "
        "Use fenced code blocks with the right language tag. Cite as [n]. State "
        "explicitly when the context does not cover a case rather than extrapolating."
    ),
    template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
)

_p(
    name="answer_with_citations",
    tags=("generate",),
    description="Structured answer where each statement carries its sources.",
    system=(
        "You answer from context and attribute every statement.\n"
        "Split your answer into statements and, for each, list the numbers of the "
        "passages that support it. A statement with no support must not be included. "
        "Set sufficient=false if the context could not answer the question."
    ),
    template="Context:\n{context}\n\nQuestion: {question}\n\nAnswer with attribution.",
)

_p(
    name="decompose_claims",
    tags=("generate", "verify"),
    description="Split an answer into atomic verifiable claims.",
    system=(
        "You split an answer into atomic factual claims for verification.\n"
        "Each claim states one checkable fact and is self-contained: replace pronouns "
        "with the entity, and carry over any qualifier that changes its truth value.\n"
        "Exclude opinions, hedged speculation, restatements of the question and "
        "meta-commentary. If the answer contains no factual claims, return an empty list."
    ),
    template="Answer:\n{answer}\n\nExtract the atomic claims.",
)

_p(
    name="verify_claim",
    tags=("generate", "verify"),
    description="Entailment check of one claim against evidence.",
    system=(
        "You decide whether evidence supports a claim.\n\n"
        "- supported: the evidence states or directly entails the claim.\n"
        "- contradicted: the evidence asserts something incompatible with it.\n"
        "- not_enough_info: the evidence neither entails nor contradicts it.\n\n"
        "Judge strictly against the evidence text. Your own knowledge is irrelevant "
        "here — a true claim with no supporting evidence is not_enough_info.\n"
        "Quote the decisive span verbatim when there is one."
    ),
    template="Evidence:\n{evidence}\n\nClaim: {claim}\n\nVerdict?",
)

_p(
    name="answer_no_context",
    tags=("generate",),
    description="Used when routing returns 'none'.",
    system=(
        "Answer directly and concisely. If the question requires information you do "
        "not have, say so rather than guessing."
    ),
    template="{question}",
)
