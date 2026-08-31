# `ragorc.generate` — grounded answering, citations, abstention, the loops

Where every guarantee the library makes is actually enforced, in a fixed order chosen
so each step's cost is only paid when the previous step allowed it:

```
abstain(pre) → budget → pack (+compress) → generate
             → citations → validate → ground → abstain(post)
```

Two orderings are load-bearing. **Abstain before generating**: with no usable
evidence the synthesis call is waste, and a model given no evidence answers from its
parameters, which produces the most confident hallucination in the system.
**Validate citations before grading groundedness**: citation validation is string
matching (free) and decisive; groundedness costs model calls.

## Key classes

```python
AnswerGenerator(llm, settings=None, *, router=None, packer=None, budgeter=None,
                summarizer=None, grounding=None, validator=None, abstention=None)
    name = "answer"
    async generate(query, retrieval: RetrievalResult, *, route=None, prompt_name=None, **kw) -> Answer
    async stream(query, retrieval, *, route=None, prompt_name=None, **kw) -> AsyncIterator[str]

GroundednessChecker(llm, settings=None, *, router=None)
    async check(question, answer, chunks, *, method=None) -> GroundednessResult
GroundednessResult(grounded, score, usage, method, unsupported, contradicted, claims)
ClaimCheck(claim, verdict, score, evidence_quote, chunk_id)   # .supported .contradicted

AbstentionPolicy(settings=None)
    before_generation(chunks) -> AbstentionDecision
    after_generation(*, answer_text, grounded, groundedness_score, contradicted,
                     model_says_insufficient, invalid_citations) -> AbstentionDecision
AbstentionDecision(abstain, reason, gate, message, confidence)

SelfConsistencyChecker(llm, settings=None, *, temperature=0.7)
    async generate(prompt, *, system=None, model=None) -> ConsistencyResult
extract_citations(answer_text, chunks, *, attribute=True) -> list[Citation]
attribute_spans(claim, source)  ·  renumber_citations(answer_text, keep)

SelfRAG(llm, settings=None, *, router=None, grounding=None)
    async run(query, retrieve: RetrieveFn, generate: GenerateFn) -> SelfRAGResult
RRR(llm, settings=None, *, router=None, min_chunks=1, min_top_score=0.0)
    async run(query, retrieve: RetrieveFn) -> RRRResult
```

## Hallucination control, by cost

| Mechanism | Catches | Cost |
|---|---|---|
citation existence | `[7]` when six passages were supplied | free |
quote verification | a plausible quote the cited document does not contain | free |
holistic groundedness grade | gross unsupported answers | 1 cheap call |
**claim decomposition + per-claim entailment** | composition errors and detail drift | N cheap calls |
NLI cross-encoder (`ragorc[nli]`) | the same, locally | free after model load |
self-consistency | idiosyncratic fabrication | N × synthesis |
abstention policy | all of the above, as a decision | free |

The claim-level check exists because a whole-answer grade anchors on plausibility and
misses the two failures that matter most: a causal claim the context never makes, and
a number that is *close* to the source but not the source's. Self-consistency compares
**claims, not strings** — paraphrases share almost no word forms while two answers with
different numbers share nearly all of them, so text similarity scores both backwards.

## Abstention is a success state

A system that always answers cannot signal inadequate evidence, so its worst outputs are
indistinguishable from its best. When either gate fires, `Answer.abstained` is `True`,
`text` becomes `generation.abstain_message`, citations are cleared, and the rejected
draft is kept under `metadata["rejected_answer"]` — the most useful artifact for
diagnosing why the pipeline declined, and a caller must opt in to see it.

## The two loops

`RRR` is rewrite → retrieve → read: one cheap rewrite *before* retrieval, retried on
weak *retrieval* signal, so it never pays for a synthesis call to discover the query
was bad. `SelfRAG` is the safety net on the other side: generate, grade ISSUP and
ISUSE **concurrently** (independent judgements about the same text), and on failure
rewrite for the specific failure that occurred — an ungrounded answer needs different
retrieval than a grounded-but-useless one. It terminates in an abstention, never in
the least-bad failed attempt.

Both take `retrieve_fn` / `answer_fn` callables rather than objects, so they compose
with anything and are testable with stubs.

## Usage

```python
from ragorc.generate import RRR, AnswerGenerator, SelfRAG

generator = AnswerGenerator(llm)
answer = await generator.generate(query, retrieval, route=decision)
print(answer.text, answer.grounded, answer.groundedness, answer.usage.cost_usd)
for c in answer.citations:
    print(c.chunk_id, c.support, c.quote[:70])


async def retrieve(q):
    return await hybrid.retrieve_detailed(q)


async def answer_fn(q, r):
    return await generator.generate(q, r)


pre = await RRR(llm).run(query, retrieve)  # rewrite first
result = await SelfRAG(llm).run(pre.query, retrieve, answer_fn)
print(result.report())  # iterations, accepted_at, abstained, verdict per attempt
```

## Settings

| Setting | Effect |
|---|---|
`generation.prompt_name` · `cite_sources` · `citation_style` | `prompt_name` accepts the shorthand (`concise` resolves to `answer_concise`). `json` returns attribution in `AnswerWithCitations.statements`, converted to `Answer.citations`; its system prompt tells the model *not* to write inline `[n]`, so the two styles differ at the parsing layer, and the routed prompt's system block is composed with the attribution contract rather than replaced by it |
`generation.check_groundedness` · `groundedness_method` · `groundedness_threshold` | `llm` \| `nli` \| `both` |
`generation.verify_citations` · `decompose_claims` | the free check and the highest-fidelity one |
`generation.self_consistency_samples` · `self_consistency_threshold` | >1 multiplies synthesis cost by N |
`generation.allow_abstention` · `abstain_message` · `min_context_chunks` | the two gates |
`generation.self_rag_enabled` · `self_rag_max_retries` | the answer-quality loop |
`generation.rrr_enabled` · `rrr_max_rewrites` | the pre-retrieval loop |
`generation.max_answer_tokens` · `reserved_output_tokens` · `stream` | |
`llm.model` | synthesis (`Task.ANSWER`, balanced tier); every grader uses `fast_model` |
`cost.max_cost_per_query_usd` · `max_llm_calls_per_query` | the loops have no natural bound without these |
