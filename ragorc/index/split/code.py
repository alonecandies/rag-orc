"""Language-aware code splitting.

Why code needs its own strategy
-------------------------------
A character splitter cuts source code in the middle of a function body, and the
result is a chunk that is syntactically meaningless *and* semantically
misleading: an orphaned ``if`` branch retrieves for the query it half-answers and
then shows the reader nothing they can act on. Code has explicit, cheap-to-find
boundaries — definitions — so the split should follow them.

Boundaries are found with per-language regexes anchored at column zero, i.e. at
*top-level* definitions. A class or a Go type therefore stays whole while it fits,
which is what a reader needs: methods only make sense next to the state they
mutate.

Why regex and not tree-sitter
-----------------------------
A parser is strictly more correct — regex boundaries can land on the word ``def``
inside a docstring or a string literal, and this module will occasionally cut
there. The price of correctness is a compiled per-grammar dependency in the base
install for a component whose worst failure mode is one awkwardly-placed
boundary, bounded by the same size enforcement every other strategy goes through.
That trade is not worth it here; if you need exact boundaries, split upstream and
feed this package one document per definition.

What travels with a chunk so it stays interpretable
---------------------------------------------------
A retrieved method is useless without two pieces of context that live elsewhere in
the file, so both are carried in metadata:

* ``class`` — the enclosing type. ``def save(self)`` retrieved on its own does not
  say *what* it saves; ``UserRepository.save`` does.
* ``imports`` — the file's import block, truncated. It is the fastest way to know
  which ``Session``, ``Path`` or ``Model`` a snippet refers to, and it is the
  first thing a reader scrolls up for.

``symbol`` names the definition and ``language`` records what the detector
decided, which is also what makes a wrong detection visible in the payload rather
than mysterious in the results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import structlog

from ragorc.core.models import Document, Modality
from ragorc.core.registry import register
from ragorc.index.split.base import BaseSplitter, Span
from ragorc.index.split.recursive import recursive_spans

log = structlog.get_logger(__name__)

__all__ = ["CodeSplitter"]

_MAX_IMPORT_CHARS = 400
"""The import block goes into every chunk's payload, so it is capped. Four hundred
characters is ~10 import lines: enough to disambiguate the names in a snippet,
small enough that it does not dominate the stored chunk."""

_IMPORT_SCAN_CHARS = 4000
"""How far into a file to look for the import block. Generous enough for a long
license header plus a hundred imports, bounded so the scan stays O(1) per
document rather than O(file)."""

#: Keywords whose definitions contain other definitions. A match on one of these
#: opens a container, so members found inside it inherit its name.
_CONTAINER_KEYWORDS = frozenset(
    {"class", "impl", "trait", "interface", "enum", "struct", "record", "mod", "object"}
)


@dataclass(slots=True, frozen=True)
class _Rules:
    """Per-language patterns. All are ``MULTILINE`` and anchored at a line start.

    ``definition`` finds top-level units, ``member`` finds the indented units
    inside a container (used only when a container is too large to keep whole),
    and ``imports`` recognizes an import line.
    """

    definition: re.Pattern[str]
    imports: re.Pattern[str]
    member: re.Pattern[str] | None = None


_PY_DECORATORS = r"(?:^[ \t]*@[\w.]+[^\n]*\n)*"

_RULES: dict[str, _Rules] = {
    "python": _Rules(
        definition=re.compile(
            _PY_DECORATORS + r"^(?:async[ \t]+)?(?P<kw>def|class)[ \t]+(?P<name>\w+)",
            re.MULTILINE,
        ),
        member=re.compile(
            r"(?:^[ \t]+@[\w.]+[^\n]*\n)*^[ \t]+(?:async[ \t]+)?(?P<kw>def|class)[ \t]+(?P<name>\w+)",
            re.MULTILINE,
        ),
        imports=re.compile(r"^(?:from|import)[ \t]+\S", re.MULTILINE),
    ),
    "javascript": _Rules(
        definition=re.compile(
            r"^(?:export[ \t]+)?(?:default[ \t]+)?(?:declare[ \t]+)?(?:abstract[ \t]+)?"
            r"(?:(?P<kw>class|function\*?|interface|type|enum|namespace)[ \t]+(?P<name>[\w$]+)"
            r"|(?:const|let|var)[ \t]+(?P<name2>[\w$]+)[^\n=]*=[ \t]*"
            r"(?:async[ \t]+)?(?:function\*?|\([^\n)]*\)[ \t]*=>|[\w$]+[ \t]*=>))",
            re.MULTILINE,
        ),
        member=re.compile(
            r"^[ \t]+(?:public[ \t]+|private[ \t]+|protected[ \t]+|readonly[ \t]+|static[ \t]+)*"
            r"(?:async[ \t]+)?(?:get[ \t]+|set[ \t]+)?(?P<name>[\w$]+)[ \t]*\([^\n)]*\)[^\n{]*\{",
            re.MULTILINE,
        ),
        imports=re.compile(r"^(?:import[ \t]|export[ \t]+\*|const[ \t]+\{?[^\n=]*=[ \t]*require)"),
    ),
    "go": _Rules(
        definition=re.compile(
            r"^(?:(?P<kw>func)[ \t]+(?:\([^)]*\)[ \t]*)?(?P<name>\w+)"
            r"|(?P<kw2>type|var|const)[ \t]+(?P<name2>\w+|\()"
            r"|(?P<kw3>package)[ \t]+(?P<name3>\w+))",
            re.MULTILINE,
        ),
        imports=re.compile(r"^import[ \t]*[(\"]", re.MULTILINE),
    ),
    "rust": _Rules(
        definition=re.compile(
            r"^(?:#\[[^\n]*\]\n)*"
            r"^(?:pub(?:\([^)]*\))?[ \t]+)?(?:default[ \t]+)?(?:const[ \t]+)?(?:async[ \t]+)?"
            r"(?:unsafe[ \t]+)?(?:extern[ \t]+\"[^\"]*\"[ \t]+)?"
            r"(?P<kw>fn|struct|enum|trait|impl|mod|union|type|static|macro_rules!)"
            r"(?:[ \t]+(?P<name>[\w<>:]+))?",
            re.MULTILINE,
        ),
        member=re.compile(
            r"^[ \t]+(?:pub(?:\([^)]*\))?[ \t]+)?(?:async[ \t]+)?(?:unsafe[ \t]+)?"
            r"(?P<kw>fn)[ \t]+(?P<name>\w+)",
            re.MULTILINE,
        ),
        imports=re.compile(r"^(?:use|extern[ \t]+crate)[ \t]+\S", re.MULTILINE),
    ),
    "java": _Rules(
        definition=re.compile(
            r"(?:^[ \t]*@[\w.]+[^\n]*\n)*"
            r"^[ \t]*(?:public[ \t]+|protected[ \t]+|private[ \t]+|abstract[ \t]+|final[ \t]+"
            r"|static[ \t]+|sealed[ \t]+|strictfp[ \t]+)*"
            r"(?P<kw>class|interface|enum|record)[ \t]+(?P<name>\w+)",
            re.MULTILINE,
        ),
        member=re.compile(
            r"(?:^[ \t]+@[\w.]+[^\n]*\n)*"
            r"^[ \t]+(?:public[ \t]+|protected[ \t]+|private[ \t]+|static[ \t]+|final[ \t]+"
            r"|synchronized[ \t]+|native[ \t]+|abstract[ \t]+|default[ \t]+)+"
            r"[\w<>\[\],. ]+[ \t]+(?P<name>\w+)[ \t]*\([^\n;]*\)[^\n;{]*\{",
            re.MULTILINE,
        ),
        imports=re.compile(r"^(?:package|import)[ \t]+\S", re.MULTILINE),
    ),
    "sql": _Rules(
        # SQL has no nesting to preserve, so the unit is the statement. Boundaries
        # are statement-initial keywords rather than the trailing semicolon, which
        # also appears inside string literals and PL/pgSQL bodies.
        definition=re.compile(
            r"^[ \t]*(?P<kw>CREATE|ALTER|DROP|TRUNCATE|COMMENT|GRANT|REVOKE|INSERT|UPDATE"
            r"|DELETE|MERGE|SELECT|WITH|BEGIN|DECLARE|SET|COPY|ANALYZE|EXPLAIN|VACUUM)\b"
            r"(?:[ \t]+(?:OR[ \t]+REPLACE[ \t]+)?(?:TABLE|VIEW|MATERIALIZED[ \t]+VIEW|INDEX"
            r"|FUNCTION|PROCEDURE|TRIGGER|SCHEMA|TYPE|SEQUENCE|DATABASE|ROLE|EXTENSION)"
            r"(?:[ \t]+IF[ \t]+NOT[ \t]+EXISTS)?[ \t]+(?P<name>[\w.\"]+))?",
            re.MULTILINE | re.IGNORECASE,
        ),
        imports=re.compile(r"^(?:\\i|SET[ \t]+search_path)", re.MULTILINE | re.IGNORECASE),
    ),
}
_RULES["typescript"] = _RULES["javascript"]

_EXTENSION_LANGUAGES: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".sql": "sql",
    ".ddl": "sql",
    ".psql": "sql",
}

#: Last-resort content sniffing, in priority order. Only used when neither the
#: metadata nor the file extension says anything, which is the case for code
#: pasted into an issue or extracted from a notebook.
_CONTENT_HINTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^package[ \t]+\w+|^func[ \t]+main\(", re.MULTILINE), "go"),
    (re.compile(r"^fn[ \t]+main\(|^use[ \t]+\w+::", re.MULTILINE), "rust"),
    (re.compile(r"^(?:from[ \t]+\S+[ \t]+import|def[ \t]+\w+\()", re.MULTILINE), "python"),
    (
        re.compile(r"^(?:public|private)[ \t]+(?:static[ \t]+)?(?:class|void)\b", re.MULTILINE),
        "java",
    ),
    (
        re.compile(r"^(?:export[ \t]+)?(?:function|const|class)[ \t]+[\w$]+", re.MULTILINE),
        "javascript",
    ),
    (re.compile(r"^\s*(?:CREATE|SELECT|INSERT)\b", re.MULTILINE | re.IGNORECASE), "sql"),
)

_LANGUAGE_KEYS = ("language", "lang", "code_language")
_PATH_KEYS = ("path", "file_path", "filename", "file", "source")

#: Names a loader or a markdown fence is likely to hand us for a language we do
#: support. Anything not resolvable stays as declared, so the log line and the
#: chunk payload show what the corpus actually claimed.
_LANGUAGE_ALIASES: dict[str, str] = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "node": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "golang": "go",
    "rs": "rust",
    "postgres": "sql",
    "postgresql": "sql",
    "psql": "sql",
    "plpgsql": "sql",
    "mysql": "sql",
    "sqlite": "sql",
}


@dataclass(slots=True)
class _Unit:
    """One top-level definition (or the file prologue), as offsets plus context."""

    start: int
    end: int
    symbol: str = ""
    kind: str = ""
    container: str = ""


@register("splitter", "code")
class CodeSplitter(BaseSplitter):
    """Splits source files on definition boundaries, keeping units intact."""

    name = "code"

    @property
    def overlap_chars(self) -> int:
        """No character overlap. A definition boundary is already the right
        boundary, and prepending 64 characters of the previous function's body
        starts every chunk with a syntax fragment — noise in the vector and a lie
        to the reader. Chunks that *are* fragments of one definition carry
        ``split_definition`` in their metadata instead."""
        return 0

    def _spans_sync(self, document: Document) -> list[Span]:
        text = document.content
        language = _detect_language(document)
        rules = _RULES.get(language or "")
        if rules is None:
            # Unknown language: character splitting is still better than one
            # chunk per file, but say so — a wrong or missing language tag is the
            # most likely reason a code corpus retrieves badly.
            log.info(
                "code_language_unresolved",
                document_id=document.id,
                declared=language,
                fallback="recursive",
            )
            return recursive_spans(
                text,
                0,
                len(text),
                chunk_size=self.target_chars,
                template=Span(
                    0,
                    len(text),
                    modality=Modality.CODE,
                    metadata={"language": language or "unknown"},
                ),
            )

        units = _units(text, rules)
        imports = _import_block(text, rules)
        spans: list[Span] = []
        for unit in units:
            self._emit(text, unit, rules, language or "unknown", imports, spans)
        return spans

    def _emit(
        self,
        text: str,
        unit: _Unit,
        rules: _Rules,
        language: str,
        imports: str,
        out: list[Span],
    ) -> None:
        metadata: dict[str, Any] = {"language": language}
        if unit.symbol:
            metadata["symbol"] = unit.symbol
        if unit.kind:
            metadata["kind"] = unit.kind
        if unit.container:
            metadata["class"] = unit.container
        if imports:
            metadata["imports"] = imports
        # Members of one type share a group, so a class split across chunks packs
        # its own small methods together; top-level definitions share a separate
        # group, so two one-line helpers merge but a whole class never absorbs
        # the unrelated function next to it.
        group = unit.container or f"top:{language}"
        template = Span(
            unit.start,
            unit.end,
            group=group,
            modality=Modality.CODE,
            metadata=metadata,
        )

        if unit.end - unit.start <= self.max_chars or self.max_chars <= 0:
            out.append(template)
            return

        # Too big to keep whole. Prefer member boundaries inside the definition:
        # a class cut between two methods is still readable, a class cut mid-body
        # is not.
        if rules.member is not None and unit.kind in _CONTAINER_KEYWORDS:
            members = _members(text, unit, rules)
            if len(members) > 1:
                for member in members:
                    self._emit(text, member, rules, language, imports, out)
                return

        # No usable interior boundary: cut on text structure, and record that the
        # chunk is a fragment of a definition rather than a definition.
        fragment = Span(
            unit.start,
            unit.end,
            group=group,
            modality=Modality.CODE,
            metadata={**metadata, "split_definition": True},
        )
        out.extend(
            recursive_spans(
                text,
                unit.start,
                unit.end,
                chunk_size=self.max_chars,
                template=fragment,
            )
        )


# ---------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------
def _units(text: str, rules: _Rules) -> list[_Unit]:
    """Tile the file into the prologue plus one unit per top-level definition."""
    matches = list(rules.definition.finditer(text))
    if not matches:
        return [_Unit(start=0, end=len(text))] if text else []

    units: list[_Unit] = []
    if matches[0].start() > 0:
        units.append(_Unit(start=0, end=matches[0].start(), kind="prologue"))
    for position, match in enumerate(matches):
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        symbol, kind = _symbol_and_kind(match)
        units.append(
            _Unit(
                start=match.start(),
                end=end,
                symbol=symbol,
                kind=kind,
                container=symbol if kind in _CONTAINER_KEYWORDS else "",
            )
        )
    return units


def _members(text: str, unit: _Unit, rules: _Rules) -> list[_Unit]:
    """Split a container at its member boundaries, header included in the first.

    The header (``class Foo:`` plus everything before the first member) stays with
    the first member rather than becoming a chunk of its own: on its own it is a
    single line, and it is the line that makes the first member readable.
    """
    assert rules.member is not None  # noqa: S101 - narrowing; _emit guards on this
    matches = [
        match
        for match in rules.member.finditer(text, unit.start, unit.end)
        if match.start() > unit.start
    ]
    if not matches:
        return [unit]
    members: list[_Unit] = []
    for position, match in enumerate(matches):
        start = unit.start if position == 0 else match.start()
        end = matches[position + 1].start() if position + 1 < len(matches) else unit.end
        symbol, kind = _symbol_and_kind(match)
        members.append(
            _Unit(
                start=start,
                end=end,
                symbol=symbol,
                kind=kind or "member",
                container=unit.container or unit.symbol,
            )
        )
    return members


def _symbol_and_kind(match: re.Match[str]) -> tuple[str, str]:
    """Pull the declared name and keyword out of whichever branch matched."""
    groups = match.groupdict()
    name = next(
        (
            value
            for key, value in groups.items()
            if key.startswith("name") and isinstance(value, str) and value
        ),
        "",
    )
    keyword = next(
        (
            value
            for key, value in groups.items()
            if key.startswith("kw") and isinstance(value, str) and value
        ),
        "",
    )
    return name.strip("(\"'"), keyword.casefold()


def _import_block(text: str, rules: _Rules) -> str:
    """The file's import lines, joined and truncated.

    Only the head of the file is scanned. The import block is a header phenomenon
    in all six languages — Go and Java put it after ``package``, so keying off the
    first definition boundary would miss it entirely — and an ``import`` further
    down is a local import inside a function body, which is indented and therefore
    does not match a line-start-anchored pattern anyway.
    """
    window = text[:_IMPORT_SCAN_CHARS]
    lines = [line for line in window.splitlines() if rules.imports.match(line)]
    if not lines:
        return ""
    joined = "\n".join(lines)
    if len(joined) <= _MAX_IMPORT_CHARS:
        return joined
    return joined[:_MAX_IMPORT_CHARS].rstrip() + "\n..."


def _detect_language(document: Document) -> str | None:
    """Metadata first, then the file extension, then a content sniff."""
    for key in _LANGUAGE_KEYS:
        raw = document.metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            declared = raw.strip().casefold()
            return _LANGUAGE_ALIASES.get(declared, declared)

    candidates = [document.metadata.get(key) for key in _PATH_KEYS]
    candidates.append(document.source)
    for candidate in candidates:
        if not isinstance(candidate, str) or "." not in candidate:
            continue
        suffix = candidate[candidate.rfind(".") :].casefold()
        language = _EXTENSION_LANGUAGES.get(suffix)
        if language:
            return language

    head = document.content[:4000]
    return next((language for pattern, language in _CONTENT_HINTS if pattern.search(head)), None)
