"""Test doubles for every external dependency.

Design goal: **the entire unit suite runs with no network, no containers, no API
keys and no model downloads.** A test suite that needs infrastructure is a test
suite that does not run, and one that calls a real model is not a test at all —
it is a sample.

Two properties every fake here maintains:

* **Determinism.** ``StubEmbedder`` seeds an RNG from a hash of the input text, so
  the same string always yields the same vector *and* similar strings do not
  accidentally yield similar vectors. Tests that assert on ranking need vectors
  they can predict.
* **Recording.** Every fake records what it was asked, so a test can assert on the
  *calls* — that grading happened once per document, that embedding was batched
  rather than looped, that a store was queried with the tenant filter applied.
  Behaviour that is invisible to assertions is behaviour that will regress.
"""

from tests.fakes.embedder import (
    StubEmbedder,
    StubLateInteractionEmbedder,
    StubReranker,
    StubSparseEmbedder,
)
from tests.fakes.llm import ScriptedLLM, StubLLM
from tests.fakes.stores import (
    FakeCache,
    FakeDocumentStore,
    FakeGraphStore,
    FakeRelationalStore,
    FakeVectorStore,
)

__all__ = [
    "FakeCache",
    "FakeDocumentStore",
    "FakeGraphStore",
    "FakeRelationalStore",
    "FakeVectorStore",
    "ScriptedLLM",
    "StubEmbedder",
    "StubLLM",
    "StubLateInteractionEmbedder",
    "StubReranker",
    "StubSparseEmbedder",
]
