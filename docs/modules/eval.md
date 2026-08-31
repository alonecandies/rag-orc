# `ragorc.eval` — datasets, metrics, and the A/B runner

Offline measurement, so a configuration change can be defended with a number rather than
a demo. Four parts, in the order the numbers depend on each other: a labelled dataset,
rank metrics over retrieval, judged metrics over the answer, and the runner whose paired
bootstrap decides whether a difference between two runs is real.

## Key classes

```python
EvalCase(question, expected_answer="", expected_chunk_ids=(), metadata={}, id="")
EvalDataset(cases=[], name="eval", source=None)
    async load(path) / save(path)   # JSONL; blank and '#' lines are skipped
    extend(cases)                   # de-duplicates on the question-derived id
    slice(*, paraphrases=None, with_reference=None, with_labels=None, tag=None)
    sample(n, *, seed=0)  ·  stats()  ·  EvalCase.relevant_ids / .is_paraphrase / .unanswerable

SyntheticQuestionGenerator(llm, settings=None, *, router=None, questions_per_chunk=2,
                           paraphrase=True, min_chunk_chars=240, model=None)
    async generate(chunks, *, limit=None, seed=0, name="synthetic")
        -> tuple[EvalDataset, SyntheticReport]

evaluate_retrieval(retrieved, relevant, *, ks=DEFAULT_KS, gains=None) -> RetrievalReport
    RetrievalReport  # .mean() .to_dict() .to_markdown(); recall_at_k, ndcg_at_k, mrr ...

AnswerMetrics(llm, settings=None, *, embedder=None, router=None, grounding=None,
              model=None, reverse_questions=3)
    async evaluate(question, answer, chunks, *, reference="", metrics=ALL_METRICS) -> Scorecard
    async faithfulness / answer_relevance / context_precision / context_recall /
          answer_correctness
Scorecard(scores)  # .values() .reasons() .usage
MetricScore(name, score, reasoning, usage, detail)  # .computed; nan when inapplicable
```

`EvalCase.id` defaults to a hash of the question, which is what lets an A/B comparison
pair cases across two runs and makes regeneration converge instead of doubling.

## Only two questions about a ranked list

**"Did we retrieve it at all?"** — `recall@k`, `hit_rate@k`. A **ceiling**: a reranker
can only reorder what the first stage returned, so if `recall@50` is 0.6 no downstream
stage gets past 0.6, and every fix is upstream (embedding model, hybrid instead of
dense-only, chunking, a larger `fetch_k`).

**"Did we rank it well?"** — `mrr`, `ndcg@k`, `map`. Position-sensitive, so they move
when the same set is reordered — this is what a reranker improves, measured *within* the
ceiling recall already set. Reading order on a new corpus: `recall@fetch_k` → `recall@top_k` → `ndcg@top_k` →
`precision@k`. Undefined values are `nan`, never `0.0`: scoring an unlabelled query zero
makes an unlabelled dataset look like a broken retriever.

## Five answer metrics, and which need ground truth

| Metric | Asks | Reference? |
|---|---|---|
`faithfulness` | is every claim supported by the retrieved context? | no |
`answer_relevance` | does the answer address the question asked? | no |
`context_precision` | how much of the retrieved context was used? | no |
`context_recall` | how much of the true answer does the context contain? | yes |
`answer_correctness` | is the answer right? | yes |
`lexical_overlap` | token-F1 / ROUGE-L floor — free, no model | yes |

The first three are computable on live traffic, which is what makes them alertable.
`faithfulness` delegates to `GroundednessChecker` rather than reimplementing claim
verification, so the offline metric measures the same thing the production gate does.
`context_recall` is that computation with its arguments swapped — high context recall
with low faithfulness means the evidence was there and the model failed to use it;
both low means retrieval failed. Every score carries its **reasoning**: "faithfulness
0.6" is not actionable, "these two claims are unsupported" tells you which fix applies.

**Pin the judge model.** Evaluation defaults to the balanced tier, against the usual
cost-cascade advice: a judge with 15% error injects more variance than the gap between
two configurations, and changing it re-scales every metric so the diff looks like a
regression.

A **paired** bootstrap, not two means: the pairing removes the between-question variance
that dominates the estimate, so a 2-point difference on 40 cases can be told apart from
noise instead of reported as an improvement.

## The dataset that ships

`examples/eval/questions.jsonl` — 20 hand-written cases over `examples/corpus/`,
tagged `single_hop`, `multi_hop`, `aggregation`, `unanswerable`, `pricing`, `policy`,
`late_chunking`. Two are unanswerable on purpose: abstention is a success state, so a
harness that cannot score it measures the wrong thing.

**An unanswerable case is graded on whether the pipeline abstained**, not on text
overlap. A case is unanswerable when its `metadata.answerable` is `false`
(`EvalCase.unanswerable`), and the runner short-circuits to a single
`abstention` score of 1.0 or 0.0. Grading these the ordinary way compares the answer
to a reference that *is itself a refusal*, so a confident fabrication scores on
wording and a correct abstention — phrased from `generation.abstain_message` rather
than the reference — can score lower than the fabrication.

It carries reference answers and **no** `expected_chunk_ids`: chunk ids are
content-derived, so a hard-coded one becomes unmatchable the moment the splitter or the
corpus changes, and reports a config change as a recall regression. Retrieval labels
come from the synthetic generator; `metadata.source_document` is the stable label here.

## Usage — this is what `examples/06_evaluation.py` runs

```python
from ragorc.eval import AnswerMetrics, EvalDataset, EvalRunner, compare_runs

dataset = await EvalDataset.load("examples/eval/questions.jsonl")
judge = AnswerMetrics(rag.llm, embedder=rag.dense_embedder, model=PINNED_JUDGE)

runner = EvalRunner(lambda q: rag.query(q, pipeline="naive"), metrics=judge)
baseline = await runner.run(dataset, name="naive")
candidate = await EvalRunner(lambda q: rag.query(q, pipeline="crag"), metrics=judge).run(
    dataset, name="crag"
)
print(baseline.to_markdown(), compare_runs(baseline, candidate).to_markdown(), sep="\n\n")

# Slice before reading a mean: originals share wording with their source chunks and
# paraphrases do not, so a mean over both averages two difficulties.
print(dataset.slice(tag="multi_hop").stats())
```

## Settings

| Setting | Effect |
|---|---|
`llm.model` | the default judge (balanced tier) — pin it with `AnswerMetrics(model=...)` |
`llm.max_concurrency` | a run's fan-out ceiling; the metrics nest their own inside it |
`generation.groundedness_method` | `both` forces the claim decomposition `faithfulness` needs |
`cost.*` | **not** enforced per case by `EvalRunner` — a case is a query *plus* its judges, so the request-path ceiling would abort legitimate cases; the bill is aggregated instead |
