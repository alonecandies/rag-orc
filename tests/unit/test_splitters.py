"""Splitter tests.

The dominant defect class in splitters is **offset drift**: a chunk whose
``start_char``/``end_char`` do not slice back to its own content. It is easy to
introduce (any place that rebuilds a string instead of tracking indices) and it
is silent — retrieval still works, but late chunking pools the wrong token span
and span-level citations point at the wrong text.

So every splitter here is checked against the same invariant, and it is asserted
character-for-character rather than approximately.
"""

from __future__ import annotations

import pytest

from ragorc.core.models import Document
from ragorc.core.settings import Settings
from ragorc.index.split import build_splitter

PROSE = (
    "Machine learning models require training data. The quality of that data "
    "determines the ceiling on model performance.\n\n"
    "Retrieval augmented generation avoids retraining. Instead it fetches relevant "
    "context at inference time and conditions the model on it.\n\n"
    "Vector databases store embeddings for similarity search. Qdrant, Milvus and "
    "pgvector are common choices. Each makes different trade-offs between recall, "
    "latency and operational complexity.\n\n"
    "Dr. Smith reported a 3.14 percent improvement, i.e. within the margin of error. "
    "The U.S. team disagreed with that reading."
)

MARKDOWN = """# Deployment Guide

Introductory text about deployment.

## Docker

Run the install command to start the container.

```bash
docker compose up -d
# a comment with ## inside a fence
```

## Kubernetes

Apply the manifest. See the table below.

| Field | Value |
|-------|-------|
| image | ragorc:latest |
| port  | 8000 |
"""

CODE = '''import os
from typing import Any


def load_config(path: str) -> dict[str, Any]:
    """Read configuration from disk."""
    with open(path) as handle:
        return parse(handle.read())


class Client:
    """An API client."""

    def __init__(self, url: str) -> None:
        self.url = url

    def get(self, path: str) -> Any:
        return request("GET", self.url + path)
'''


def assert_offsets_exact(document: Document, chunks: list) -> None:
    """The invariant: every chunk's span must slice back to its own content."""
    for chunk in chunks:
        assert chunk.start_char <= chunk.end_char, f"inverted span on {chunk.id}"
        sliced = document.content[chunk.start_char : chunk.end_char]
        # Splitters may strip surrounding whitespace from the content they emit,
        # so the comparison is on stripped text — but nothing else may differ.
        assert sliced.strip() == chunk.content.strip(), (
            f"offset drift on chunk {chunk.index}: "
            f"span[{chunk.start_char}:{chunk.end_char}]={sliced[:60]!r} "
            f"but content={chunk.content[:60]!r}"
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        security={"enforce_tenant_isolation": False},
        indexing={"chunk_size": 200, "chunk_overlap": 30, "min_chunk_size": 20},
    )


@pytest.mark.parametrize("name", ["recursive", "token", "sentence_window"])
async def test_offsets_are_exact(name: str, settings: Settings) -> None:
    document = Document(id="d1", content=PROSE)
    chunks = await build_splitter(name, settings=settings).split(document)
    assert chunks, f"{name} produced no chunks"
    assert_offsets_exact(document, chunks)


async def test_markdown_offsets_are_exact(settings: Settings) -> None:
    document = Document(id="md", content=MARKDOWN, source="guide.md")
    chunks = await build_splitter("markdown", settings=settings).split(document)
    assert chunks
    assert_offsets_exact(document, chunks)


async def test_code_offsets_are_exact(settings: Settings) -> None:
    document = Document(id="py", content=CODE, source="client.py")
    chunks = await build_splitter("code", settings=settings).split(document)
    assert chunks
    assert_offsets_exact(document, chunks)


async def test_semantic_offsets_are_exact(settings: Settings, embedder) -> None:
    document = Document(id="d1", content=PROSE)
    splitter = build_splitter("semantic", embedder=embedder, settings=settings)
    chunks = await splitter.split(document)
    assert chunks
    assert_offsets_exact(document, chunks)


async def test_chunks_are_indexed_and_linked(settings: Settings) -> None:
    document = Document(id="d1", content=PROSE)
    chunks = await build_splitter("recursive", settings=settings).split(document)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == "d1" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks), "chunk ids must be unique"


async def test_ids_are_deterministic_across_runs(settings: Settings) -> None:
    """Re-splitting an unchanged document must produce identical ids, or ingest
    is not idempotent and a re-run duplicates every vector."""
    document = Document(id="d1", content=PROSE)
    first = await build_splitter("recursive", settings=settings).split(document)
    second = await build_splitter("recursive", settings=settings).split(document)
    assert [c.id for c in first] == [c.id for c in second]


async def test_splitters_emit_no_vectors(settings: Settings, embedder) -> None:
    """Splitting must not embed — that is what keeps late chunking possible."""
    document = Document(id="d1", content=PROSE)
    for name in ("recursive", "token", "semantic", "markdown", "sentence_window"):
        chunks = await build_splitter(name, embedder=embedder, settings=settings).split(document)
        assert all(c.dense is None for c in chunks), f"{name} embedded during split"
        assert all(c.sparse is None for c in chunks), f"{name} produced sparse vectors"


async def test_abbreviations_do_not_split_sentences(settings: Settings, embedder) -> None:
    """'Dr. Smith' and '3.14' must not be treated as sentence boundaries."""
    document = Document(
        id="abbr",
        content="Dr. Smith measured 3.14 percent, i.e. nothing. The U.S. team agreed. Work continued.",
    )
    chunks = await build_splitter("sentence_window", settings=settings).split(document)
    joined = " ".join(c.content for c in chunks)
    assert "Dr. Smith" in joined
    assert "3.14" in joined
    assert len(chunks) <= 4, f"over-split into {len(chunks)} chunks: {[c.content for c in chunks]}"


async def test_markdown_carries_heading_context(settings: Settings) -> None:
    """A chunk under '## Docker' saying 'run the install command' is meaningless
    without its heading, so the heading path must travel with it."""
    document = Document(id="md", content=MARKDOWN, source="guide.md")
    chunks = await build_splitter("markdown", settings=settings).split(document)
    docker = [c for c in chunks if "install command" in c.content]
    assert docker, "the Docker section chunk was not produced"
    metadata = docker[0].metadata
    context = " ".join(str(v) for v in metadata.values()) + " " + docker[0].content
    assert "Docker" in context, f"no heading context in {metadata}"


async def test_markdown_keeps_code_fence_intact(settings: Settings) -> None:
    document = Document(id="md", content=MARKDOWN, source="guide.md")
    chunks = await build_splitter("markdown", settings=settings).split(document)
    fenced = [c for c in chunks if "docker compose up" in c.content]
    assert fenced, "the fenced block was lost"
    # The '##' inside the fence must not have started a new section.
    assert "# a comment with ## inside a fence" in fenced[0].content


async def test_sentence_window_stores_its_window(settings: Settings) -> None:
    document = Document(id="d1", content=PROSE)
    chunks = await build_splitter("sentence_window", settings=settings).split(document)
    assert chunks
    windowed = [c for c in chunks if c.metadata.get("window_text")]
    assert windowed, "sentence-window must store the surrounding text for generation"
    sample = windowed[0]
    assert len(sample.metadata["window_text"]) >= len(sample.content)


async def test_empty_and_tiny_documents_do_not_crash(settings: Settings, embedder) -> None:
    for content in ("", "   ", "Hi.", "One short sentence only."):
        document = Document(id="tiny", content=content)
        for name in ("recursive", "token", "semantic", "markdown", "sentence_window"):
            chunks = await build_splitter(name, embedder=embedder, settings=settings).split(
                document
            )
            assert isinstance(chunks, list)


async def test_semantic_falls_back_without_embedder(settings: Settings) -> None:
    """A semantic splitter with no embedder must degrade, not raise."""
    splitter = build_splitter("semantic", embedder=None, settings=settings)
    chunks = await splitter.split(Document(id="d1", content=PROSE))
    assert chunks


async def test_max_chunk_size_is_respected(embedder) -> None:
    settings = Settings(
        security={"enforce_tenant_isolation": False},
        indexing={
            "chunk_size": 120,
            "chunk_overlap": 0,
            "max_chunk_size": 200,
            "min_chunk_size": 10,
        },
    )
    document = Document(id="d1", content=PROSE * 3)
    for name in ("recursive", "token"):
        chunks = await build_splitter(name, settings=settings).split(document)
        oversized = [c for c in chunks if len(c.content) > 200 * 6]
        assert not oversized, f"{name} produced a chunk far over max_chunk_size"


async def test_chunks_carry_document_provenance(settings: Settings) -> None:
    """Every chunk must be able to say where it came from.

    Three consumers are downstream of splitting and none can derive this later:
    the context packer prints it as the source line beside each numbered passage,
    citations resolve against it, and document-level eval grading matches on it.
    """
    document = Document(id="d1", content=PROSE, source="policy.md", title="Customer Policy")
    for name in ("recursive", "token", "sentence_window", "markdown"):
        chunks = await build_splitter(name, settings=settings).split(document)
        assert chunks, f"{name} produced nothing"
        assert all(c.metadata.get("source") == "policy.md" for c in chunks), (
            f"{name} dropped the document source"
        )
        assert all(c.metadata.get("title") == "Customer Policy" for c in chunks)


async def test_a_splitter_may_override_the_document_source(settings: Settings) -> None:
    """Span metadata wins: a splitter that knows a more specific source (a section
    anchor, a file inside an archive) should not have it overwritten."""
    document = Document(id="d1", content=MARKDOWN, source="guide.md")
    chunks = await build_splitter("markdown", settings=settings).split(document)
    assert all(c.metadata.get("source") for c in chunks)


async def test_provenance_is_absent_when_the_document_has_none(settings: Settings) -> None:
    """No source on the document means no invented source on the chunk."""
    chunks = await build_splitter("recursive", settings=settings).split(
        Document(id="d1", content=PROSE)
    )
    assert all("source" not in c.metadata for c in chunks)
