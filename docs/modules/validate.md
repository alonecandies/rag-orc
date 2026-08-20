# `ragorc.validate` — inbound queries, outbound answers, ingest documents

Three checkpoints, one per boundary the data crosses. Validation here is about
*shape and honesty*, not security — the guards in
[`ragorc.security`](security.md) handle adversarial input, and this package handles
the far more common case of input and output that is merely wrong.

## Key classes

```python
QueryValidator(settings=None)
    validate(text, *, tenant_id=None, top_k=None) -> ValidatedQuery
ValidatedQuery(query: Query, warnings, injection_risk, pii_entities)

AnswerValidator(settings=None)
    validate(answer: Answer, chunks: list[ScoredChunk]) -> OutputReport
    strip_invalid_citations(answer, chunks) -> Answer
OutputReport(valid, warnings, invalid_citations, unverified_quotes,
             scaffold_leak, citation_coverage)
build_citations(answer_text, chunks) -> list[Citation]

DocumentValidator(settings=None, *, max_bytes=20_000_000)
    validate_document(doc: Document) -> Document
    validate_batch(docs: list[Document]) -> IngestReport
    validate_chunks(chunks: list[Chunk]) -> list[Chunk]
IngestReport(accepted, rejected, warnings)   # .accept_rate
```

## Why answer validation runs before groundedness

Citation validation is string matching: free, and decisive. It catches the most
common fabrication in the system — a real-looking quote attributed to a real
document that the document does not contain — and it catches `[7]` when six
passages were supplied. Groundedness costs model calls. Running the free, decisive
check first means the expensive one is skipped on answers already disqualified,
which is exactly the order `AnswerGenerator` uses.

`strip_invalid_citations` exists for the middle case: an answer that is otherwise
good but carries a phantom marker. Showing a reader a citation they cannot follow is
worse than showing none, so the marker is removed — and the surviving ones keep their
numbers, because `[n]` is an index into the passage list that was packed. Renumbering
here would silently re-point every remaining citation at a different passage. The
renumbering case is the *other* one: when a passage is dropped after generation,
`ragorc.generate.citations.renumber_citations` remaps the markers onto the new list.

## Why chunk validation runs before the LLM does

`validate_chunks` is called by `GraphBuilder` and by the ingest pipeline before any
model call. Extraction and contextual enrichment cost one call per chunk, and
spending one to discover that a chunk is three punctuation marks is pure waste.

## Usage

```python
from ragorc.validate import AnswerValidator, DocumentValidator, QueryValidator

validated = QueryValidator().validate(user_text, tenant_id="acme")
for warning in validated.warnings:
    log.info("query_warning", detail=warning)

report = DocumentValidator().validate_batch(documents)
log.info("ingest_validated", accepted=len(report.accepted), accept_rate=report.accept_rate)

output = AnswerValidator().validate(answer, packed_chunks)
if output.invalid_citations:
    answer = AnswerValidator().strip_invalid_citations(answer, packed_chunks)
```

## Settings

| Setting | Effect |
|---|---|
`security.max_query_length` · `min_query_length` | bounds enforced by `QueryValidator` |
`security.enable_injection_detection` · `injection_action` | the scanner `QueryValidator` and the packer share |
`security.enforce_tenant_isolation` | a query with no tenant is rejected rather than searching every tenant |
`generation.verify_citations` | confirms each cited span exists in the cited chunk |
`generation.cite_sources` · `citation_style` | `inline` \| `footnote` \| `json` |
`indexing.min_chunk_size` · `max_chunk_size` | the floor and ceiling `validate_chunks` applies |
`indexing.dedupe_chunks` · `dedupe_threshold` | exact and near-duplicate rejection at ingest |
