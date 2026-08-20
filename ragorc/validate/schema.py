"""Ingest-side validation.

Bad documents are cheaper to reject than to index: an empty chunk consumes an
embedding call and a vector slot while being unretrievable, and a 40 MB
"document" that is actually a minified bundle will poison a semantic splitter's
percentile threshold for the whole batch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from ragorc.core.errors import ValidationFailed
from ragorc.core.models import Chunk, Document
from ragorc.core.settings import Settings, get_settings

log = structlog.get_logger(__name__)

__all__ = ["DocumentValidator", "IngestReport"]

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
#: Ratio of non-word characters above which text is probably not prose.
_BINARY_RATIO = 0.35


@dataclass(slots=True)
class IngestReport:
    accepted: list[Document] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def accept_rate(self) -> float:
        total = len(self.accepted) + len(self.rejected)
        return len(self.accepted) / total if total else 1.0


class DocumentValidator:
    def __init__(self, settings: Settings | None = None, *, max_bytes: int = 20_000_000) -> None:
        self.settings = settings or get_settings()
        self.max_bytes = max_bytes

    def validate_document(self, doc: Document) -> Document:
        if isinstance(doc, (list, tuple, set)):
            # A natural confusion between the single- and batch-document entry
            # points; an AttributeError deep inside would not say so.
            raise ValidationFailed(
                "expected a single Document, got a sequence",
                got=type(doc).__name__,
                hint="use validate_batch() / build_many() for several documents",
            )
        if not isinstance(doc, Document):
            raise ValidationFailed("expected a Document", got=type(doc).__name__)
        if not doc.id:
            raise ValidationFailed("document has no id")
        if not doc.content or not doc.content.strip():
            raise ValidationFailed("document has no content", doc_id=doc.id)

        size = len(doc.content.encode("utf-8", errors="ignore"))
        if size > self.max_bytes:
            raise ValidationFailed(
                "document exceeds the size limit", doc_id=doc.id, bytes=size, limit=self.max_bytes
            )

        doc.content = _CONTROL.sub(" ", doc.content)

        if self._looks_binary(doc.content):
            raise ValidationFailed(
                "content does not look like text (high non-word ratio)", doc_id=doc.id
            )

        if self.settings.security.enforce_tenant_isolation and not doc.tenant_id:
            raise ValidationFailed(
                "tenant_id is required on ingest when tenant isolation is enabled",
                doc_id=doc.id,
            )
        return doc

    def validate_batch(self, docs: list[Document]) -> IngestReport:
        report = IngestReport()
        seen_checksums: dict[str, str] = {}
        for doc in docs:
            try:
                validated = self.validate_document(doc)
            except ValidationFailed as exc:
                report.rejected.append((doc.id or "<no id>", exc.message))
                continue
            if validated.checksum:
                prior = seen_checksums.get(validated.checksum)
                if prior:
                    # Same bytes under a different id: indexing both doubles the
                    # storage and lets one document occupy two result slots.
                    report.warnings.append(
                        f"{validated.id} duplicates {prior} (identical checksum); skipped"
                    )
                    continue
                seen_checksums[validated.checksum] = validated.id
            report.accepted.append(validated)
        if report.rejected:
            log.warning("ingest_rejections", count=len(report.rejected), sample=report.rejected[:3])
        return report

    def validate_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Drop chunks that cannot contribute: too short to carry meaning, or
        pure whitespace/punctuation after normalization.

        Logged at **warning**, with the documents it happened to. It used to be
        ``debug``, i.e. invisible at the default level, and that hid a real
        outcome: a document shorter than ``min_chunk_size`` has exactly one chunk,
        so dropping it drops the whole document. Ingesting a corpus of FAQ
        one-liners or glossary entries then reported ``indexed: 0, failed: 0,
        rejected: 0`` and exit 0 over an empty index. Such a document now reaches
        the ingest pipeline as an empty chunk list and is counted there as
        ``documents_empty`` — that is what makes the report reconcile; this log is
        the only thing that says *why* it was empty.
        """
        minimum = self.settings.indexing.min_chunk_size
        out: list[Chunk] = []
        dropped: list[Chunk] = []
        for chunk in chunks:
            content = chunk.content.strip()
            if len(content) < minimum or not re.search(r"\w", content):
                dropped.append(chunk)
                continue
            chunk.content = content
            out.append(chunk)
        if dropped:
            # Document ids rather than chunk ids: a chunk id is content-derived, so
            # it identifies nothing an operator can go and look at, while the
            # document id is what they used to ingest it.
            documents = sorted({chunk.document_id for chunk in dropped if chunk.document_id})
            log.warning(
                "chunks_dropped",
                count=len(dropped),
                kept=len(out),
                minimum=minimum,
                documents=documents[:5],
                reason="below_min_size_or_no_words",
            )
        return out

    @staticmethod
    def _looks_binary(text: str) -> bool:
        sample = text[:4000]
        if not sample:
            return False
        non_word = sum(
            1
            for ch in sample
            if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'\"-()[]{}/@#%&*+=_|<>~`$^\\")
        )
        return (non_word / len(sample)) > _BINARY_RATIO
