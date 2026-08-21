"""Evaluation datasets, and how to get one for a corpus that has none.

Every configuration decision in this library — `fetch_k`, which splitter, whether
reranking earns its latency, whether HyDE helps or hurts — is a guess until it is
measured on the corpus it will actually serve. Published benchmark numbers do not
transfer: BEIR, MS MARCO and NQ have their own vocabulary, document length,
question style and answer distribution, and a reranker that gains six points of
nDCG there can lose recall on a corpus of internal runbooks. So the harness needs
labelled questions over *your* documents, and almost nobody has any.

Why questions are generated from chunks, not collected from users
----------------------------------------------------------------
Labelled retrieval data means, for each question, the set of passages that answer
it. Producing that by hand runs in the expensive direction: an annotator with a
question has to search 200k chunks to find the ones that answer it, which is the
very task being evaluated, done manually, once per label.

Generating in the opposite direction makes the label free *and* exact. Show the
model one chunk, ask for questions that this chunk answers, and record that chunk
as the ground truth. The label is not an annotation to be trusted — it is a
by-product of construction: the passage the question was written from is, by
definition, a passage that answers it. That is the only practical way to obtain
labelled retrieval data for a private corpus, and it is why this module exists.

Two biases it introduces, and what to do about them
--------------------------------------------------
**Vocabulary overlap.** A question written while looking at a passage reuses that
passage's words. Both dense and lexical retrieval are then being asked an easier
question than a user would ask, so absolute recall@k from a purely synthetic set
is an *upper bound*, not a measurement. BM25 flatters worst here (literal term
overlap); dense retrieval flatters second-worst.

The countermeasure is not to discard the number but to generate a **paraphrased
variant** of every question: same information need, same ground-truth chunk,
deliberately different words. The two slices are then comparable, and the *gap*
between them is the interesting quantity — it measures how much of your retrieval
quality is term overlap rather than meaning. A pipeline that scores 0.92
recall@10 on the originals and 0.61 on the paraphrases will disappoint the moment
real users type real questions, and nothing in the original slice would have told
you.

**Incomplete labels.** Only the source chunk is marked relevant, but other chunks
may answer the question too — a near-duplicate passage, a summary, the parent
document. A retriever that finds those is *penalized* by precision@k and MAP,
which count them as misses. Recall-style metrics ("did the labelled chunk come
back at all?") stay meaningful under incomplete labels; precision-style metrics on
a synthetic set are a lower bound and should be read as such. This is worth
remembering before optimizing precision against these labels.

Storage is JSONL: one case per line, appendable, diffable in review, and readable
without loading the whole file — which matters once a dataset is a few thousand
cases with reference answers attached.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import structlog
from pydantic import BaseModel, Field

from ragorc.core.concurrency import bounded_gather
from ragorc.core.errors import ValidationFailed
from ragorc.core.ids import content_hash
from ragorc.core.models import Chunk, Usage
from ragorc.core.protocols import LLM
from ragorc.core.settings import Settings, get_settings
from ragorc.core.telemetry import new_request_context
from ragorc.llm.prompts import Prompt, get_prompt, register_prompt
from ragorc.llm.router import ModelRouter, ModelTier

log = structlog.get_logger(__name__)

__all__ = [
    "DEFAULT_QUESTIONS_PER_CHUNK",
    "MIN_CHUNK_CHARS",
    "EvalCase",
    "EvalDataset",
    "SyntheticQuestionGenerator",
    "SyntheticReport",
]

MIN_CHUNK_CHARS = 240
"""Chunks shorter than this are skipped as question sources.

Not an arbitrary floor: a two-line chunk is usually a heading, a licence stub or a
navigation fragment, and the only questions it supports are referential ones
("what does this section list?") that no retrieval system can answer from the
question text alone. Generating from them spends model calls to add noise."""

DEFAULT_QUESTIONS_PER_CHUNK = 2
"""More than two questions per chunk starts producing near-duplicates that share a
ground-truth label, which inflates the apparent size of the dataset without adding
information — and makes every aggregate metric weight that chunk more heavily."""

_MAX_SOURCE_CHARS = 6_000
"""Hard clip on the passage sent to the generator. A pathologically long chunk
would otherwise dominate the bill and produce questions about its tail only."""


# ---------------------------------------------------------------------------
# Structured output for the generator
#
# These schemas live here rather than in ``core/schemas.py`` on purpose: that
# module is the contract for the *request path*, and nothing in the request path
# generates evaluation data. Keeping them local means the offline harness can
# evolve its prompts without touching a file every pipeline stage imports.
# ---------------------------------------------------------------------------
class _GeneratedQuestion(BaseModel):
    question: str = Field(
        description=(
            "A self-contained question answered by this passage. Never refer to "
            "'the passage', 'this document', 'the author' or 'it' — the retrieval "
            "system sees the question with no context, so a referential question "
            "is unanswerable by construction."
        )
    )
    answer: str = Field(
        default="",
        description="The short answer, taken from the passage. Quote figures and names exactly.",
    )
    answerable: bool = Field(
        default=True,
        description=(
            "False if this passage cannot support a real question (boilerplate, a "
            "table of contents, a licence header, navigation chrome)."
        ),
    )
    self_contained: bool = Field(
        default=True,
        description="False if the question only makes sense to a reader who already has the passage.",
    )


class _GeneratedQuestions(BaseModel):
    questions: list[_GeneratedQuestion] = Field(default_factory=list, max_length=8)


class _ParaphrasedQuestion(BaseModel):
    question: str = Field(
        description=(
            "The same information need expressed in different words. The known "
            "answer must still be the correct answer."
        )
    )
    changed_terms: list[str] = Field(
        default_factory=list,
        description="Domain terms from the original that were replaced, for auditing the rewrite.",
    )


register_prompt(
    Prompt(
        name="eval_synth_question",
        tags=("eval",),
        description="Write evaluation questions answerable from one specific chunk.",
        system=(
            "You write evaluation questions for a document retrieval system. You are "
            "given ONE passage from a private corpus, and your questions become the "
            "test set that decides how that system is configured.\n\n"
            "Rules:\n"
            "- Every question must be answerable from this passage alone.\n"
            "- Every question must be self-contained. The retrieval system is shown "
            "the question and nothing else, so 'what does this section say about X' "
            "cannot be answered by any system and only adds noise. Name the entity.\n"
            "- Ask about specific content: names, quantities, dates, identifiers, "
            "error codes, procedures, causes, constraints. Those are what users ask.\n"
            "- No yes/no questions: they are answerable by guessing, so they measure "
            "nothing about retrieval.\n"
            "- Ask each question about a different part of the passage. Two questions "
            "about the same sentence are one question.\n"
            "- Give the short answer as the passage states it. Copy figures, names and "
            "identifiers exactly; do not round, reformat or paraphrase them.\n"
            "- If the passage is boilerplate and supports no genuine question, return "
            "answerable=false instead of inventing one."
        ),
        template="Passage:\n{passage}\n\nWrite {n} evaluation questions about it.",
    )
)

register_prompt(
    Prompt(
        name="eval_paraphrase_question",
        tags=("eval",),
        description="Rewrite an eval question in different vocabulary to control for term overlap.",
        system=(
            "You rewrite an evaluation question so it asks for the same thing in "
            "different words.\n\n"
            "Why: the original question was written by someone looking at the source "
            "passage, so it reuses that passage's vocabulary. Retrieval then scores "
            "higher than it will on real user questions. Your rewrite is the control "
            "that exposes the difference, so lexical overlap with the original is the "
            "thing you are removing.\n\n"
            "Rules:\n"
            "- Preserve the information need exactly. The known answer must remain the "
            "correct answer; if your rewrite changes what is being asked, it is wrong.\n"
            "- Replace the domain terms with what a user who has NOT read the document "
            "would say: an everyday synonym, a description instead of the term of art, "
            "the expansion of an acronym or the acronym instead of the expansion.\n"
            "- Keep proper nouns, product names, version numbers and identifiers that a "
            "real user would genuinely type. Renaming those changes the question rather "
            "than rephrasing it.\n"
            "- Change the syntax as well as the words: a conversational form, an "
            "indirect question, or a longer question with the framing a person adds."
        ),
        template="Question: {question}\nKnown answer: {answer}\n\nRewrite the question.",
    )
)


# ---------------------------------------------------------------------------
# The dataset
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class EvalCase:
    """One labelled question.

    ``expected_chunk_ids`` is the retrieval ground truth and ``expected_answer``
    the generation ground truth. Both are optional and they are independently
    useful: a case with chunk ids but no reference answer still measures recall and
    nDCG, and a case with a reference answer but no chunk ids still measures
    answer correctness. The harness reports which metrics each slice supports
    rather than silently scoring an absent label as zero.
    """

    question: str
    expected_answer: str = ""
    expected_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self) -> None:
        # Content-derived id, for the same reason chunk ids are: A/B comparison
        # pairs results by case id across two runs, and a random id would break
        # the pairing the moment the dataset is regenerated or re-ordered.
        if not self.id:
            self.id = content_hash("evalcase", self.question, size=8)
        self.expected_chunk_ids = tuple(self.expected_chunk_ids)

    @property
    def has_reference(self) -> bool:
        return bool(self.expected_answer.strip())

    @property
    def has_labels(self) -> bool:
        return bool(self.expected_chunk_ids)

    @property
    def relevant_ids(self) -> set[str]:
        return set(self.expected_chunk_ids)

    @property
    def is_paraphrase(self) -> bool:
        """True for the vocabulary-control variant of another case.

        Slice on this before reading any absolute number: the originals share
        wording with their source chunks and the paraphrases do not, so a mean over
        both slices is a mean over two different difficulties.
        """
        return bool(self.metadata.get("paraphrase_of"))

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "expected_chunk_ids": list(self.expected_chunk_ids),
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> EvalCase:
        question = str(raw.get("question") or "").strip()
        if not question:
            raise ValidationFailed("eval case has no question", case=str(raw)[:200])
        return cls(
            question=question,
            expected_answer=str(raw.get("expected_answer") or ""),
            expected_chunk_ids=tuple(raw.get("expected_chunk_ids") or ()),
            metadata=dict(raw.get("metadata") or {}),
            id=str(raw.get("id") or ""),
        )


@dataclass(slots=True)
class EvalDataset:
    """A named collection of cases, persisted as JSONL."""

    cases: list[EvalCase] = field(default_factory=list)
    name: str = "eval"
    source: str | None = None

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __getitem__(self, index: int) -> EvalCase:
        return self.cases[index]

    # -- persistence -------------------------------------------------------
    @classmethod
    async def load(cls, path: str | Path, *, name: str | None = None) -> EvalDataset:
        """Read a JSONL dataset.

        The read happens in a thread: a 10k-case file with reference answers is a
        few megabytes, and blocking the event loop on it inside a service that is
        also serving queries is exactly the pattern this codebase forbids.
        """
        target = Path(path)
        raw = await asyncio.to_thread(target.read_bytes)
        cases: list[EvalCase] = []
        for lineno, line in enumerate(raw.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            try:
                payload = orjson.loads(stripped)
            except orjson.JSONDecodeError as exc:
                raise ValidationFailed(
                    "malformed JSONL in eval dataset", path=str(target), line=lineno
                ) from exc
            cases.append(EvalCase.from_json(payload))
        dataset = cls(cases=cases, name=name or target.stem, source=str(target))
        # `cases` is also a key in stats(); splatting it alongside the explicit
        # keyword would raise TypeError and fail every load.
        log.info("eval_dataset_loaded", path=str(target), **dataset.stats())
        return dataset

    async def save(self, path: str | Path) -> int:
        """Write the dataset as JSONL. Returns the number of cases written."""
        target = Path(path)
        blob = b"\n".join(orjson.dumps(case.to_json()) for case in self.cases) + b"\n"

        def _write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)

        await asyncio.to_thread(_write)
        log.info("eval_dataset_saved", path=str(target), cases=len(self.cases))
        return len(self.cases)

    # -- slicing -----------------------------------------------------------
    def extend(self, cases: Sequence[EvalCase]) -> list[EvalCase]:
        """Append cases, dropping ones whose id is already present. Returns the kept ones.

        De-duplication is on id, which is derived from the question text, so
        regenerating over a corpus that has barely changed converges on the same
        dataset instead of doubling it.

        ``seen`` grows as the batch is walked, so a duplicate *within* ``cases`` is
        dropped too. Two chunks that yield the same question is not a hypothetical:
        near-duplicate passages are common in real corpora, and admitting both
        copies would weight that question twice in every mean while giving it two
        different ground-truth labels for the same text.

        The kept cases are returned rather than just counted because the caller
        needs to know *which* survived — attributing the batch's counts by slicing
        it at the number added silently mislabels every case after the first
        duplicate.
        """
        seen = {case.id for case in self.cases}
        added: list[EvalCase] = []
        for case in cases:
            if case.id in seen:
                continue
            seen.add(case.id)
            added.append(case)
        self.cases.extend(added)
        return added

    def slice(
        self,
        *,
        paraphrases: bool | None = None,
        with_reference: bool | None = None,
        with_labels: bool | None = None,
        tag: str | None = None,
    ) -> EvalDataset:
        """Filter into a sub-dataset. ``None`` means "don't care" on that axis."""
        cases = self.cases
        if paraphrases is not None:
            cases = [c for c in cases if c.is_paraphrase is paraphrases]
        if with_reference is not None:
            cases = [c for c in cases if c.has_reference is with_reference]
        if with_labels is not None:
            cases = [c for c in cases if c.has_labels is with_labels]
        if tag is not None:
            cases = [c for c in cases if tag in (c.metadata.get("tags") or ())]
        suffix = "+".join(
            part
            for part in (
                "paraphrase" if paraphrases else ("original" if paraphrases is False else ""),
                tag or "",
            )
            if part
        )
        return EvalDataset(
            cases=list(cases),
            name=f"{self.name}[{suffix}]" if suffix else self.name,
            source=self.source,
        )

    def sample(self, n: int, *, seed: int = 0) -> EvalDataset:
        """A seeded random subset — reproducible, which an unseeded one is not."""
        if n >= len(self.cases):
            return EvalDataset(cases=list(self.cases), name=self.name, source=self.source)
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(self.cases), size=n, replace=False)
        return EvalDataset(
            cases=[self.cases[int(i)] for i in sorted(picks)],
            name=f"{self.name}[sample{n}]",
            source=self.source,
        )

    def stats(self) -> dict[str, Any]:
        labelled = sum(1 for c in self.cases if c.has_labels)
        referenced = sum(1 for c in self.cases if c.has_reference)
        paraphrases = sum(1 for c in self.cases if c.is_paraphrase)
        return {
            "cases": len(self.cases),
            "with_chunk_labels": labelled,
            "with_reference_answer": referenced,
            "paraphrases": paraphrases,
            "documents": len({c.metadata.get("document_id") for c in self.cases} - {None}),
        }


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SyntheticReport:
    """What one generation run did, and what it refused to do."""

    chunks_considered: int = 0
    chunks_too_short: int = 0
    chunks_used: int = 0
    chunks_failed: int = 0
    questions: int = 0
    rejected_unanswerable: int = 0
    rejected_referential: int = 0
    rejected_duplicate: int = 0
    paraphrases: int = 0
    paraphrases_failed: int = 0
    usage: Usage = field(default_factory=Usage)

    def summary(self) -> dict[str, Any]:
        return {
            "chunks_considered": self.chunks_considered,
            "chunks_used": self.chunks_used,
            "chunks_too_short": self.chunks_too_short,
            "chunks_failed": self.chunks_failed,
            "questions": self.questions,
            "paraphrases": self.paraphrases,
            "rejected": {
                "unanswerable": self.rejected_unanswerable,
                "referential": self.rejected_referential,
                "duplicate": self.rejected_duplicate,
            },
            "llm_calls": self.usage.calls,
            "cost_usd": round(self.usage.cost_usd, 6),
        }


class SyntheticQuestionGenerator:
    """Turns a corpus into a labelled retrieval dataset, one chunk at a time.

    The generator writes questions **from** a chunk and records that chunk's id as
    ground truth, then optionally writes a vocabulary-controlled paraphrase of each
    question that inherits the same label. See the module docstring for why both
    halves are needed.
    """

    name = "synthetic_questions"

    def __init__(
        self,
        llm: LLM,
        settings: Settings | None = None,
        *,
        router: ModelRouter | None = None,
        questions_per_chunk: int = DEFAULT_QUESTIONS_PER_CHUNK,
        paraphrase: bool = True,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.settings = settings or get_settings()
        self.router = router or ModelRouter(self.settings.llm)
        self.questions_per_chunk = max(int(questions_per_chunk), 1)
        self.paraphrase = paraphrase
        self.min_chunk_chars = int(min_chunk_chars)
        # The balanced tier, not the fast one, and deliberately against the cost
        # cascade's usual advice. A bad grader costs one noisy measurement; a bad
        # *label* is permanently wrong and silently biases every experiment run
        # against this dataset from now on. Generation is also offline and bounded
        # by corpus size, so this is the one place where paying up is correct.
        self.model = model or self.router.model_for_tier(ModelTier.BALANCED)

    async def generate(
        self,
        chunks: Sequence[Chunk],
        *,
        limit: int | None = None,
        seed: int = 0,
        name: str = "synthetic",
    ) -> tuple[EvalDataset, SyntheticReport]:
        """Generate a dataset from a corpus sample.

        ``limit`` caps how many chunks are used and the sample is seeded, because
        generating over an entire corpus is both unnecessary and expensive: a few
        hundred chunks spread across documents already separates configurations
        that differ, and a reproducible sample is what makes two experiments
        comparable at all.
        """
        report = SyntheticReport(chunks_considered=len(chunks))
        eligible = [c for c in chunks if len(c.content.strip()) >= self.min_chunk_chars]
        report.chunks_too_short = len(chunks) - len(eligible)
        if not eligible:
            log.warning("eval_generate_nothing_eligible", **report.summary())
            return EvalDataset(name=name), report

        if limit is not None and limit < len(eligible):
            rng = np.random.default_rng(seed)
            picks = sorted(int(i) for i in rng.choice(len(eligible), size=limit, replace=False))
            eligible = [eligible[i] for i in picks]

        results = await bounded_gather(
            (self._from_chunk(chunk) for chunk in eligible),
            limit=self.settings.llm.max_concurrency,
            return_exceptions=True,
        )

        dataset = EvalDataset(name=name)
        usages: list[Usage] = []
        for chunk, result in zip(eligible, results, strict=True):
            if isinstance(result, BaseException):
                # One chunk's generation failing must not lose the other 400.
                report.chunks_failed += 1
                log.warning(
                    "eval_generate_chunk_failed", chunk_id=chunk.id, error=str(result)[:160]
                )
                continue
            cases, usage, counts = result
            usages.append(usage)
            report.rejected_unanswerable += counts["unanswerable"]
            report.rejected_referential += counts["referential"]
            if not cases:
                continue
            report.chunks_used += 1
            added = dataset.extend(cases)
            report.rejected_duplicate += len(cases) - len(added)
            report.questions += sum(1 for c in added if not c.is_paraphrase)
            report.paraphrases += sum(1 for c in added if c.is_paraphrase)
            report.paraphrases_failed += counts["paraphrase_failed"]

        report.usage = Usage.sum(usages)
        # Nested, not merged: both mappings carry a `paraphrases` key, and splatting
        # them into one call raises TypeError on the duplicate keyword — which would
        # throw away a generation run that had already been paid for in full.
        log.info("eval_dataset_generated", report=report.summary(), dataset=dataset.stats())
        return dataset, report

    # -- one chunk ---------------------------------------------------------
    async def _from_chunk(self, chunk: Chunk) -> tuple[list[EvalCase], Usage, dict[str, int]]:
        """Questions (and paraphrases) for one chunk, plus the bill.

        Each chunk runs inside its own request context. The ledger's ceilings are
        per-*query* limits, so installing one per chunk keeps them meaningful — a
        pathological chunk aborts itself instead of the run — where a single
        corpus-wide ledger would abort a legitimate job partway through, which is
        the same argument the ingest pipeline makes for not installing one at all.
        """
        cost = self.settings.cost
        counts = {"unanswerable": 0, "referential": 0, "paraphrase_failed": 0}
        prompt = get_prompt("eval_synth_question")
        passage = chunk.content.strip()[:_MAX_SOURCE_CHARS]

        with new_request_context(
            request_id=f"evalgen:{chunk.id[:12]}",
            max_cost_usd=cost.max_cost_per_query_usd,
            max_calls=cost.max_llm_calls_per_query,
            max_tokens=cost.max_tokens_per_query,
            trace=self.settings.observability.trace_enabled,
        ):
            generated, usage = await self.llm.structured(
                prompt.render(passage=passage, n=self.questions_per_chunk),
                _GeneratedQuestions,
                system=prompt.system,
                model=self.model,
                stage="eval_generate_questions",
            )
            usages = [usage]

            cases: list[EvalCase] = []
            for item in generated.questions[: self.questions_per_chunk]:
                question = item.question.strip()
                if not question:
                    continue
                if not item.answerable:
                    counts["unanswerable"] += 1
                    continue
                if not item.self_contained or _is_referential(question):
                    # A referential question is unanswerable by any retriever, so
                    # keeping it would depress every metric by a constant that
                    # looks like a retrieval problem.
                    counts["referential"] += 1
                    continue
                cases.append(
                    EvalCase(
                        question=question,
                        expected_answer=item.answer.strip(),
                        expected_chunk_ids=(chunk.id,),
                        metadata={
                            "document_id": chunk.document_id,
                            "chunk_index": chunk.index,
                            "chunk_level": chunk.level,
                            "generator": self.name,
                            "generator_model": self.model,
                            "tags": ["original"],
                        },
                    )
                )

            if self.paraphrase and cases:
                variants = await bounded_gather(
                    (self._paraphrase_of(case) for case in cases),
                    limit=self.settings.llm.max_concurrency,
                    return_exceptions=True,
                )
                extra: list[EvalCase] = []
                for variant in variants:
                    if isinstance(variant, BaseException):
                        counts["paraphrase_failed"] += 1
                        continue
                    case, variant_usage = variant
                    usages.append(variant_usage)
                    if case is None:
                        counts["paraphrase_failed"] += 1
                    else:
                        extra.append(case)
                cases.extend(extra)

        return cases, Usage.sum(usages), counts

    async def _paraphrase_of(self, case: EvalCase) -> tuple[EvalCase | None, Usage]:
        """The vocabulary control: same label, different words.

        The label transfers because the information need is unchanged — the chunk
        that answered the original still answers the rewrite. That is what makes
        the two slices directly comparable rather than two separate datasets.
        """
        prompt = get_prompt("eval_paraphrase_question")
        result, usage = await self.llm.structured(
            prompt.render(question=case.question, answer=case.expected_answer or "(unknown)"),
            _ParaphrasedQuestion,
            system=prompt.system,
            model=self.model,
            stage="eval_paraphrase_question",
        )
        rewritten = result.question.strip()
        if not rewritten or rewritten.casefold() == case.question.casefold():
            # A rewrite that did not rewrite anything is not a control; keeping it
            # would just duplicate the original under a second id and weight that
            # chunk twice in every mean.
            return None, usage
        return (
            EvalCase(
                question=rewritten,
                expected_answer=case.expected_answer,
                expected_chunk_ids=case.expected_chunk_ids,
                metadata={
                    **case.metadata,
                    "paraphrase_of": case.id,
                    "original_question": case.question,
                    "changed_terms": list(result.changed_terms),
                    "tags": ["paraphrase"],
                },
            ),
            usage,
        )


_REFERENTIAL_MARKERS: tuple[str, ...] = (
    "this passage",
    "this document",
    "this text",
    "this chunk",
    "this section",
    "this article",
    "the passage",
    "the document",
    "the text above",
    "the author",
    "according to the above",
    "in the excerpt",
)
"""Phrases that make a question depend on context the retriever never sees.

Checked in addition to the model's own ``self_contained`` flag, because the model
is judging its own output and grades itself generously — this is the cheap, exact
half of the check."""


def _is_referential(question: str) -> bool:
    lowered = question.casefold()
    return any(marker in lowered for marker in _REFERENTIAL_MARKERS)
