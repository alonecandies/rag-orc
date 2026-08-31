"""Assertions about source text that only a comment or a docstring can satisfy.

Four times in three rounds a test asserted ``<literal> in inspect.getsource(X)``
where the literal also appears in *X's own prose* — so the behavioural mutation
the test was written to catch walked straight past it:

* ``late_embedder=self.late_embedder`` already appeared three times in
  ``RAGPipeline`` (it is how the Qdrant store is built), so the grep survived the
  argument being deleted from the reranker call.
* a comment containing ``crag_enabled`` satisfied the test asserting the server no
  longer resolves ``auto`` from settings.
* the paragraph *explaining the removal* of ``raptor_collapse_tree`` named it, so
  the "no inert settings" invariant passed with the knob restored.
* the comment explaining the store-errors fix contains ``metadata["errors"]``, so
  deleting the assignment kept the test green.

Every one was found by mutation, rounds after it was written. This asks the
question at test time instead: for each searched literal, does it appear in the
package's *code* — or only in its prose?

**It catches two of those four, and cannot catch the other two.** The
``raptor_collapse_tree`` and ``metadata["errors"]`` cases are *prose-only*
literals: they appear nowhere in the package's code, so asking "is this in code?"
answers them. The ``late_embedder=self.late_embedder`` and ``crag_enabled`` cases
are not — both literals genuinely appear in code, just on a different line of the
same scope than the one the test meant. No source-text rule distinguishes those;
only a mutation does.

So this is a floor, deliberately. It converts the cheap half of the class into a
test-time failure and leaves the expensive half to the mutation harness, which
remains the only thing that establishes a test bites.
"""

from __future__ import annotations

import ast
import io
import pathlib
import tokenize

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PACKAGE = _ROOT / "ragorc"
_TESTS = _ROOT / "tests"

#: Assertions that deliberately check *prose*, each with the reason. A docstring
#: is a legitimate thing to assert on when the docstring is the deliverable — a
#: measured number that must not be quietly dropped, or a caveat a future reader
#: needs. Listing them here keeps "I meant that" distinguishable from "I did not
#: notice", which is the whole distinction this file exists to make.
_DELIBERATE_PROSE: dict[str, str] = {
    "0.9596": "test_core: the docstring must record the measured paraphrase score",
    "0.9924": "test_core: and the wrong-answer pair that clears the threshold",
    "not what bounds an ingest": (
        "test_ingest_budget: the early return in _check_budget must stay documented, "
        "because an undocumented one was cited as a compensating control that does not run"
    ),
}


def _blank_prose(source: str) -> str:
    """The source with comment and docstring spans blanked *in place*.

    Blanked rather than removed, so every surviving character keeps its column and
    a searched literal still matches the layout it was written against. A first
    version joined tokens with spaces — turning ``self._x = None`` into
    ``self . _x = None`` — and reported 31 false positives. A detector needs its
    own verification exactly as much as the code it inspects.
    """
    lines = source.splitlines(keepends=True)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):  # pragma: no cover
        return source

    spans: list[tuple[int, int, int, int]] = []
    previous = tokenize.NEWLINE
    for token in tokens:
        if token.type == tokenize.COMMENT:
            spans.append((*token.start, *token.end))
            continue
        starts_statement = previous in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.DEDENT,
        )
        if token.type == tokenize.STRING and starts_statement:
            spans.append((*token.start, *token.end))
        if token.type != tokenize.NL:
            previous = token.type

    for start_row, start_col, end_row, end_col in spans:
        for row in range(start_row, end_row + 1):
            index = row - 1
            if index >= len(lines):  # pragma: no cover - ragged final line
                continue
            line = lines[index]
            newline = "\n" if line.endswith("\n") else ""
            body = line.rstrip("\n")
            first = start_col if row == start_row else 0
            last = end_col if row == end_row else len(body)
            lines[index] = body[:first] + " " * max(0, last - first) + body[last:] + newline
    return "".join(lines)


def _searched_literals(tree: ast.AST) -> list[tuple[int, str, bool]]:
    """``"literal" in <source>`` where ``<source>`` came from ``getsource``.

    Returns ``(lineno, literal, negated)``. A negated assertion — "this string is
    *gone* from the source" — is satisfied by matching nothing, which is its
    success condition rather than a defect.
    """
    from_getsource: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and "getsource" in ast.unparse(node.value):
            from_getsource.update(t.id for t in node.targets if isinstance(t, ast.Name))

    out: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            if not isinstance(op, ast.In | ast.NotIn):
                continue
            target = ast.unparse(comparator)
            if target not in from_getsource and "getsource" not in target:
                continue
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                out.append((node.lineno, node.left.value, isinstance(op, ast.NotIn)))
    return out


def _test_files() -> list[pathlib.Path]:
    return sorted(_TESTS.rglob("test_*.py"))


def test_the_scan_finds_source_assertions_to_check() -> None:
    """A detector that matches nothing makes every assertion below vacuous —
    which is the exact failure it exists to find, one level up."""
    total = sum(len(_searched_literals(ast.parse(p.read_text()))) for p in _test_files())
    assert total >= 20, f"only {total} source-grep assertions found; the scan is broken"


@pytest.mark.parametrize("path", _test_files(), ids=lambda p: p.name)
def test_no_source_assertion_is_satisfiable_by_prose_alone(path: pathlib.Path) -> None:
    code = "\n".join(_blank_prose(p.read_text()) for p in sorted(_PACKAGE.rglob("*.py")))
    everything = "\n".join(p.read_text() for p in sorted(_PACKAGE.rglob("*.py")))

    offenders: list[str] = []
    for lineno, literal, negated in _searched_literals(ast.parse(path.read_text())):
        if len(literal) < 4 or literal in _DELIBERATE_PROSE:
            continue
        if negated:
            # "this is gone" is satisfied by absence; there is nothing to check.
            continue
        if literal in everything and literal not in code:
            offenders.append(f"{path.name}:{lineno} {literal!r}")

    assert not offenders, (
        "these assertions match only a comment or a docstring, so a behavioural "
        f"mutation walks past them: {offenders}"
    )


def test_the_deliberate_list_has_not_gone_stale() -> None:
    """An allowlist that outlives its entries is how the next one hides in it."""
    everything = "\n".join(p.read_text() for p in sorted(_PACKAGE.rglob("*.py")))
    searched = {
        literal
        for path in _test_files()
        for _lineno, literal, _negated in _searched_literals(ast.parse(path.read_text()))
    }
    for literal, why in _DELIBERATE_PROSE.items():
        assert literal in everything, f"no longer in the package: {literal!r} ({why})"
        assert literal in searched, f"no longer asserted anywhere: {literal!r} ({why})"
