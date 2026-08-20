"""Structure-aware markdown splitting.

Why the heading path travels with the chunk
-------------------------------------------
Markdown already contains a human-authored outline, and throwing it away is the
most expensive mistake a splitter can make on documentation. Consider a chunk
under ``# Guide / ## Install / ### Docker`` whose body reads:

    Run the install command and restart the service.

On its own that text is unretrievable — it does not contain "Docker", it does not
contain "install command" in any distinctive form, and it answers a question
nobody can phrase to match it. With ``Guide > Install > Docker`` attached, the
same chunk answers "how do I install this with Docker". This is the same failure
late chunking and contextual retrieval attack (ADR-0002), reached for free: the
outline is already in the document, no model call required.

The path is written to ``metadata["heading_path"]`` for display and filtering, and
**prepended for embedding purposes** through ``Chunk.contextual_prefix``, which
:attr:`Chunk.embed_text` puts in front of the content. Splicing it into
``content`` instead would break the invariant that ``content`` equals
``document.content[start:end]`` — and that invariant is what lets the
late-chunking pooler find the chunk's token span. So the prefix goes where the
data model already has a slot for exactly this, and offsets stay exact.

What is never cut
-----------------
Fenced code blocks and tables are carried whole, marked ``atomic``, even when they
exceed ``max_chunk_size`` (the base class notes that in
``metadata["oversized"]``). Half a code block does not compile and half a table
has no header row: both halves are worse than useless, because they still get
indexed and can still be retrieved. A block small enough to sit inside a chunk is
packed together with the prose around it, which is better — the sentence
introducing a snippet is usually the part the query matches.

Scanning is a single line-by-line pass that classifies each line into one of four
block kinds. Blocks tile the document (no gaps, no overlaps), so any run of
consecutive blocks is one exact span, which is what makes the packing step below
offset-safe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from ragorc.core.models import Document, Modality
from ragorc.core.registry import register
from ragorc.index.split.base import BaseSplitter, Span
from ragorc.index.split.recursive import recursive_spans

log = structlog.get_logger(__name__)

__all__ = ["MarkdownSplitter"]

#: ATX heading. Setext headings (an ``===`` underline) are deliberately not
#: recognized: the ``---`` form is ambiguous with a thematic break, YAML front
#: matter and a table delimiter row, and guessing wrong reorders the whole
#: outline.
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

#: Opening or closing fence. The captured run length lets a ```` ```` ```` fence
#: nest inside a ``````` ```` ``````` one, which is how markdown embeds markdown.
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(\S*)")

_TABLE_ROW = re.compile(r"^[ \t]{0,3}\|")
_TABLE_DELIMITER = re.compile(r"^[ \t]{0,3}\|?[ \t]*:?-{2,}[-:| \t]*\|?[ \t]*$")

_ATOMIC_KINDS = frozenset({"code", "table"})

#: A chunk gets a non-text modality only when its content is *entirely* one kind.
#: Prose plus a snippet is prose with a snippet in it, and tagging it ``CODE``
#: would hide it from any retriever filtering on text.
_MODALITIES: dict[frozenset[str], Modality] = {
    frozenset({"code"}): Modality.CODE,
    frozenset({"table"}): Modality.TABLE,
}


@dataclass(slots=True)
class _Block:
    """A contiguous run of lines with one structural role."""

    kind: str  # "heading" | "code" | "table" | "text"
    start: int
    end: int
    level: int = 0
    title: str = ""
    language: str = ""


@dataclass(slots=True)
class _Section:
    """A heading and everything under it, with the full ancestor path."""

    path: tuple[str, ...]
    level: int
    blocks: list[_Block] = field(default_factory=list)


@register("splitter", "markdown")
class MarkdownSplitter(BaseSplitter):
    """Heading-aware splitting that keeps code fences and tables intact."""

    name = "markdown"

    def _spans_sync(self, document: Document) -> list[Span]:
        text = document.content
        sections = _sections(_blocks(text))
        spans: list[Span] = []
        for section in sections:
            self._pack(text, section, spans)
        return spans

    def _pack(self, text: str, section: _Section, out: list[Span]) -> None:
        """Greedily pack a section's blocks into spans of at most ``target_chars``.

        Packing works on block boundaries, so a chunk never begins or ends inside
        a table row or a fence. Only an oversized *prose* block is broken up, and
        that is delegated to the recursive splitter with this section's context
        carried along in the template.
        """
        path = " > ".join(section.path)
        base_meta: dict[str, Any] = {
            "heading_path": path,
            "heading_level": section.level,
            "heading": section.path[-1] if section.path else "",
        }
        pending: list[_Block] = []

        def flush() -> None:
            if not pending:
                return
            # A heading line and blank lines carry no content of their own, so the
            # modality of a group is decided by what is left after removing them.
            substantive = [
                block
                for block in pending
                if block.kind != "heading" and text[block.start : block.end].strip()
            ]
            kinds = {block.kind for block in substantive}
            metadata = dict(base_meta)
            if "code" in kinds:
                metadata["has_code"] = True
            if "table" in kinds:
                metadata["has_table"] = True
            languages = {block.language for block in substantive if block.language}
            if len(languages) == 1:
                metadata["code_language"] = languages.pop()
            out.append(
                Span(
                    pending[0].start,
                    pending[-1].end,
                    # A group holding a fence or a table is atomic even when it
                    # fits: the guarantee "never cut inside one" should hold
                    # structurally, not because the arithmetic happened to keep
                    # this chunk under the ceiling.
                    atomic=bool(kinds & _ATOMIC_KINDS),
                    group=path,
                    prefix=path or None,
                    modality=_MODALITIES.get(frozenset(kinds)),
                    metadata=metadata,
                )
            )
            pending.clear()

        for block in section.blocks:
            length = block.end - block.start
            if block.kind in _ATOMIC_KINDS and length > self.target_chars:
                lead = self._lead_in(text, pending, block)
                if lead is None:
                    flush()
                else:
                    pending.clear()
                out.append(self._atomic_span(block, path, base_meta, start=lead))
                continue
            if block.kind == "text" and length > self.target_chars:
                flush()
                template = Span(
                    block.start,
                    block.end,
                    group=path,
                    prefix=path or None,
                    metadata={**base_meta, "split_block": True},
                )
                out.extend(
                    recursive_spans(
                        text,
                        block.start,
                        block.end,
                        chunk_size=self.target_chars,
                        template=template,
                    )
                )
                continue
            if pending and block.end - pending[0].start > self.target_chars:
                flush()
            pending.append(block)
        flush()

    def _lead_in(self, text: str, pending: list[_Block], block: _Block) -> int | None:
        """Offset the following atomic block should absorb from, or ``None``.

        A heading, or one short sentence introducing a snippet, is not a chunk. On
        its own it is a runt that cannot be merged away — the block after it is
        atomic and refuses to merge — so it is folded into the block it
        introduces, which is also where a reader would expect to find it.
        """
        if not pending:
            return None
        content = sum(
            b.end - b.start
            for b in pending
            if b.kind != "heading" and text[b.start : b.end].strip()
        )
        return pending[0].start if content < self.min_chars else None

    def _atomic_span(
        self, block: _Block, path: str, base_meta: dict[str, Any], *, start: int | None = None
    ) -> Span:
        metadata = {**base_meta, "block": block.kind}
        if block.language:
            metadata["code_language"] = block.language
        return Span(
            block.start if start is None else start,
            block.end,
            atomic=True,
            group=path,
            prefix=path or None,
            modality=Modality.CODE if block.kind == "code" else Modality.TABLE,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Block scanning
# ---------------------------------------------------------------------------
def _line_spans(text: str) -> list[tuple[int, int]]:
    """Offsets of every line, line endings included, so the spans tile the text."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        spans.append((cursor, cursor + len(line)))
        cursor += len(line)
    return spans


def _blocks(text: str) -> list[_Block]:
    """Classify every line into a tiling sequence of blocks.

    Fence state is tracked first and wins over everything else: a ``#`` or a
    ``|`` inside a code block is code, and treating it as a heading would split
    the fence in half and corrupt the outline at the same time.
    """
    lines = _line_spans(text)
    blocks: list[_Block] = []
    index = 0
    total = len(lines)

    while index < total:
        start, end = lines[index]
        raw = text[start:end]
        stripped = raw.rstrip("\r\n")

        fence = _FENCE.match(stripped)
        if fence is not None:
            marker, language = fence.group(1), fence.group(2)
            close = index + 1
            while close < total:
                candidate = text[lines[close][0] : lines[close][1]].rstrip("\r\n")
                closing = _FENCE.match(candidate)
                if (
                    closing is not None
                    and closing.group(1)[0] == marker[0]
                    and len(closing.group(1)) >= len(marker)
                    and not closing.group(2)
                ):
                    break
                close += 1
            block_end = lines[min(close, total - 1)][1]
            blocks.append(
                _Block(kind="code", start=start, end=block_end, language=language.strip())
            )
            index = close + 1
            continue

        heading = _HEADING.match(stripped)
        if heading is not None:
            blocks.append(
                _Block(
                    kind="heading",
                    start=start,
                    end=end,
                    level=len(heading.group(1)),
                    title=heading.group(2).strip(),
                )
            )
            index += 1
            continue

        # A table needs a delimiter row directly under its header; without that
        # rule any line starting with "|" would open a table.
        if _TABLE_ROW.match(stripped) and index + 1 < total:
            following = text[lines[index + 1][0] : lines[index + 1][1]].rstrip("\r\n")
            if _TABLE_DELIMITER.match(following):
                close = index + 2
                while close < total:
                    candidate = text[lines[close][0] : lines[close][1]].rstrip("\r\n")
                    if not _TABLE_ROW.match(candidate):
                        break
                    close += 1
                blocks.append(_Block(kind="table", start=start, end=lines[close - 1][1]))
                index = close
                continue

        run_end = end
        index += 1
        while index < total:
            candidate_start, candidate_end = lines[index]
            candidate = text[candidate_start:candidate_end].rstrip("\r\n")
            if (
                _HEADING.match(candidate)
                or _FENCE.match(candidate)
                or (_TABLE_ROW.match(candidate) and _is_table_head(text, lines, index))
            ):
                break
            run_end = candidate_end
            index += 1
        blocks.append(_Block(kind="text", start=start, end=run_end))

    return blocks


def _is_table_head(text: str, lines: list[tuple[int, int]], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    following = text[lines[index + 1][0] : lines[index + 1][1]].rstrip("\r\n")
    return bool(_TABLE_DELIMITER.match(following))


def _sections(blocks: list[_Block]) -> list[_Section]:
    """Group blocks under their heading, maintaining the ancestor path.

    The path is a stack indexed by heading level, so a jump from ``##`` straight
    to ``####`` (common in generated docs) does not corrupt the ancestry: the
    deeper heading is appended rather than replacing a level that was never
    opened.
    """
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(path=(), level=0)

    for block in blocks:
        if block.kind == "heading":
            if current.blocks:
                sections.append(current)
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, block.title))
            current = _Section(path=tuple(title for _, title in stack), level=block.level)
        current.blocks.append(block)

    if current.blocks:
        sections.append(current)
    return sections
