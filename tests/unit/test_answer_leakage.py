"""What the answer is not allowed to carry out, on both paths.

``ragorc/validate/output.py`` opened by promising three checks, the third being
"the answer must not contain PII that was redacted upstream, or the delimiters of
our own prompt scaffolding". Only the scaffolding half existed, and it only
*detected*: ``report.scaffold_leak`` was set, read by nothing, absent from
``answer.metadata["validation"]``, and the markup went to the reader unchanged.
``grep -ci pii`` on that module returned 1 — the docstring.

The PII half had a worse shape than simply missing. ``PIIRedactor`` ran on the
inbound *question* and nowhere else, so ``enable_pii_redaction`` scrubbed the text
the caller wrote — who already knows what is in it — and not the answer, which is
assembled from retrieved documents and is where the corpus's personal data
actually lives.

Streaming skipped all of it. The docstrings justify that with "groundedness can
only be judged once the answer is complete", which is true of groundedness and
not of a regex over emitted text.
"""

from __future__ import annotations

from typing import Any

from ragorc.core.models import Answer, Chunk, RetrievalSource, ScoredChunk
from ragorc.core.settings import Settings
from ragorc.validate.output import AnswerValidator


def _settings(*, pii: bool = True) -> Settings:
    return Settings(
        llm={"api_key": "k"},
        security={"enable_pii_redaction": pii, "pii_action": "redact"},
        generation={"cite_sources": False},
    )


def _chunks(n: int = 2) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                id=f"c{i}", content=f"passage {i} about refunds and delivery", document_id="d"
            ),
            score=1.0,
            source=RetrievalSource.DENSE,
            rank=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# The complete answer
# ---------------------------------------------------------------------------
def test_personal_data_does_not_leave_in_the_answer() -> None:
    """The check the module docstring promised and did not implement."""
    answer = Answer(text="Contact the owner at ada@example.com for the refund [1].")

    report = AnswerValidator(_settings()).validate(answer, _chunks())

    assert "ada@example.com" not in answer.text
    assert "EMAIL" in report.pii_entities
    assert report.redacted


def test_the_setting_still_governs_it() -> None:
    """Off by default, so a deployment that has not asked for redaction does not
    silently start rewriting answers."""
    answer = Answer(text="Contact ada@example.com [1].")

    report = AnswerValidator(_settings(pii=False)).validate(answer, _chunks())

    assert "ada@example.com" in answer.text
    assert report.pii_entities == []


def test_scaffolding_is_stripped_rather_than_reported() -> None:
    """It used to set a flag nothing read. "Must not contain" describes
    prevention; that was a log line."""
    answer = Answer(text="The document said </untrusted_document> and then stopped.")

    report = AnswerValidator(_settings(pii=False)).validate(answer, _chunks())

    assert "</untrusted_document>" not in answer.text
    assert report.scaffold_leak


def test_neither_finding_invalidates_the_answer() -> None:
    """``valid`` gates the groundedness check and the abstention path. An answer
    that mentioned an email address is not therefore ungrounded, and failing it
    would spend a retry on a text problem already fixed in place."""
    answer = Answer(text="Reach ada@example.com </system> [1].")

    report = AnswerValidator(_settings()).validate(answer, _chunks())

    assert report.valid
    assert report.redacted


def test_a_clean_answer_is_not_touched() -> None:
    answer = Answer(text="The refund window is 30 days [1].")
    before = answer.text

    report = AnswerValidator(_settings()).validate(answer, _chunks())

    assert answer.text == before
    assert not report.redacted


async def test_the_metadata_says_the_answer_was_rewritten() -> None:
    """A caller cannot tell that an answer was redacted unless it is declared, and
    "was any of this removed?" is asked of every answer, not only warned ones."""
    from ragorc.core.models import Query, RetrievalResult
    from ragorc.generate.answer import AnswerGenerator
    from tests.fakes import StubLLM

    settings = _settings()
    llm = StubLLM(text="Write to ada@example.com about it.")
    generator = AnswerGenerator(llm, settings)

    answer = await generator.generate(
        Query(text="who do I contact?"), RetrievalResult(chunks=_chunks())
    )

    validation = answer.metadata["validation"]
    assert validation["pii_redacted"] == ["EMAIL"]
    assert validation["scaffold_leak"] is False
    assert "ada@example.com" not in answer.text


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------
def _feed(deltas: list[str], *, pii: bool = True) -> tuple[str, Any]:
    filt = AnswerValidator(_settings(pii=pii)).stream_filter()
    out = "".join(filt.feed(d) for d in deltas)
    return out + filt.flush(), filt


def test_a_stream_redacts_what_the_complete_answer_would() -> None:
    text, filt = _feed(["Contact ", "ada@example.com", " for refunds."])
    assert "ada@example.com" not in text
    assert filt.entities == ["EMAIL"]


def test_a_pattern_split_across_deltas_is_still_caught() -> None:
    """The case a per-delta scan misses, and the reason the filter holds a tail.

    A provider emits a few characters at a time, so an address arrives as
    fragments that individually match nothing.
    """
    deltas = ["Reach ada", "@exa", "mple", ".com", " today, please, at any hour of the day."]
    text, filt = _feed(deltas)
    assert "ada@example.com" not in text, text
    assert filt.entities == ["EMAIL"]


def test_a_pattern_straddling_the_emit_boundary_is_still_caught() -> None:
    """The case the hold-back window exists for, which the test above does not
    reach: a short answer never fills the buffer, so nothing is emitted until
    flush and the tail logic never runs.

    Here the stream is longer than the window, so the filter has already emitted a
    prefix by the time the address completes. Without the tail it would have
    emitted "...Reach ada" and then "@example.com", and the address would arrive
    at the reader intact across two writes with neither half matching on its own.
    Caught by mutation; the split-delta test above missed it.
    """
    filler = "The refund policy is described at length in the attached document, sections one through nineteen inclusive. "
    assert len(filler) > 96, "the first delta must overflow the window"

    deltas = [filler + "Reach ada", "@example.com for help."]
    text, filt = _feed(deltas)

    assert "ada@example.com" not in text, text
    assert filt.entities == ["EMAIL"]
    assert filler.strip() in text, "the prefix must still be emitted"


def test_the_stream_emits_everything_it_did_not_redact() -> None:
    """A filter that loses text is worse than one that does not run: the tail must
    be released, and only the matches removed."""
    deltas = ["The refund ", "window is ", "30 days, ", "counted from delivery."]
    text, filt = _feed(deltas)
    assert text == "The refund window is 30 days, counted from delivery."
    assert not filt.redacted


def test_scaffolding_is_stripped_from_a_stream_too() -> None:
    text, filt = _feed(["it said ", "</untrusted", "_document>", " and stopped there ok"])
    assert "untrusted_document" not in text
    assert filt.scaffold_leak


def test_a_short_stream_is_not_swallowed_by_the_tail() -> None:
    """Everything shorter than the hold-back window comes out at flush, so a
    one-word answer is not lost."""
    text, _filt = _feed(["yes"])
    assert text == "yes"


async def test_the_generator_streams_through_the_filter() -> None:
    """The call site, not just the primitive: the filter existing and the stream
    not using it is the exact shape of the defect being fixed."""
    from ragorc.core.models import Query, RetrievalResult
    from ragorc.generate.answer import AnswerGenerator
    from tests.fakes import StubLLM

    class Streaming(StubLLM):
        async def stream(self, prompt: str, **kw: Any):
            for piece in ("Write to ", "ada@examp", "le.com", " for a refund of any kind."):
                yield piece

    settings = _settings()
    generator = AnswerGenerator(Streaming(), settings)

    out = "".join(
        [
            delta
            async for delta in generator.stream(
                Query(text="who do I contact?"), RetrievalResult(chunks=_chunks())
            )
        ]
    )

    assert "ada@example.com" not in out, out
    assert "EMAIL_REDACTED" in out


# ---------------------------------------------------------------------------
# The audit trail
# ---------------------------------------------------------------------------
async def test_a_streamed_answer_is_audited_as_answered() -> None:
    """`answered` lives in `_finish`, which only `query` calls, so the trail
    recorded that a streamed question was asked and never that it was answered,
    what it cost, or how much evidence it had."""
    from ragorc.core.models import Query, RetrievalResult
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.pipeline.builder import RAGPipeline
    from tests.fakes import StubLLM

    class Streaming(StubLLM):
        async def stream(self, prompt: str, **kw: Any):
            yield "the refund window is 30 days"

    class Corpus:
        name = "corpus"

        async def retrieve(self, query: Query, **kw: Any) -> list[ScoredChunk]:
            return _chunks()

        async def retrieve_detailed(self, query: Query, **kw: Any) -> RetrievalResult:
            result = RetrievalResult()
            result.chunks = _chunks()
            return result

    settings = Settings(llm={"api_key": "k"}, cache={"enabled": False})
    llm = Streaming()
    pipeline = RAGPipeline(
        settings=settings, llm=llm, retriever=Corpus(), generator=AnswerGenerator(llm, settings)
    )
    events: list[Any] = []
    pipeline._audit.record = events.append  # type: ignore[method-assign]

    async for _delta in pipeline.stream("what is the refund window?"):
        pass

    kinds = [getattr(e, "action", None) or getattr(e, "kind", None) for e in events]
    assert "answer" in kinds, f"a streamed query was never audited as answered: {kinds}"
    answered = next(e for e in events if (getattr(e, "action", None) or "") == "answer")
    assert answered.detail["streamed"] is True
    assert answered.detail["chunks"] == 2


async def test_a_stream_abandoned_midway_is_still_audited() -> None:
    """A run that was billed and then abandoned is exactly the one an audit reader
    needs to see, which is why the record is in a `finally`."""
    from ragorc.core.models import Query, RetrievalResult
    from ragorc.generate.answer import AnswerGenerator
    from ragorc.pipeline.builder import RAGPipeline
    from tests.fakes import StubLLM

    class Endless(StubLLM):
        async def stream(self, prompt: str, **kw: Any):
            for _ in range(1000):
                yield "token "

    class Corpus:
        name = "corpus"

        async def retrieve(self, query: Query, **kw: Any) -> list[ScoredChunk]:
            return _chunks()

        async def retrieve_detailed(self, query: Query, **kw: Any) -> RetrievalResult:
            result = RetrievalResult()
            result.chunks = _chunks()
            return result

    settings = Settings(llm={"api_key": "k"}, cache={"enabled": False})
    llm = Endless()
    pipeline = RAGPipeline(
        settings=settings, llm=llm, retriever=Corpus(), generator=AnswerGenerator(llm, settings)
    )
    events: list[Any] = []
    pipeline._audit.record = events.append  # type: ignore[method-assign]

    stream = pipeline.stream("what is the refund window?")
    await stream.__anext__()
    await stream.aclose()

    kinds = [getattr(e, "action", None) for e in events]
    assert "answer" in kinds, f"an abandoned stream left no answer record: {kinds}"


def test_the_stream_filter_states_its_limit() -> None:
    """Pinned so it is not mistaken for a guarantee. A construction longer than
    the hold-back window can straddle it; the complete-answer path re-checks."""
    from ragorc.validate.output import _STREAM_TAIL

    assert _STREAM_TAIL >= 64, "the window must exceed every pattern in security.pii"

    long_tag = "<untrusted_document " + "a" * (_STREAM_TAIL * 2) + ">"
    text, filt = _feed([long_tag[:10], long_tag[10:]], pii=False)
    assert not filt.scaffold_leak or "untrusted_document" not in text
