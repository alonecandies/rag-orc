# Implementation contract (internal)

Every module in `ragorc/` follows these rules. Read this plus the four contract
files before writing code:

- `ragorc/core/models.py` — the data model (dataclasses, `slots=True`)
- `ragorc/core/protocols.py` — the interfaces you must satisfy
- `ragorc/core/settings.py` — every tunable, already defined; do NOT invent new settings
- `ragorc/core/schemas.py` — pydantic models for structured LLM output
- `ragorc/llm/prompts.py` — every prompt, by name; do NOT inline prompt strings
- `ragorc/llm/router.py` — `Task` enum; pick the model via `ModelRouter.model_for(Task.X)`

## Non-negotiables

1. **Async only.** No sync public API. Never call blocking I/O in a coroutine;
   wrap CPU-bound work (ONNX inference, UMAP, numpy over big arrays) in
   `asyncio.to_thread` so the event loop keeps serving.
2. **Fan out with `bounded_gather` / `safe_gather` / `map_concurrent`** from
   `ragorc.core.concurrency`. Never bare `asyncio.gather` over an unbounded list.
3. **Vectorize.** Any scoring, normalization, fusion or similarity over N items
   is numpy, not a Python loop. `einsum`/`matmul` for MaxSim, `argpartition`
   (not `sort`) for top-k.
4. **Settings, not magic numbers.** Read from `Settings`; every knob already
   exists. Constructor signature: `def __init__(self, ..., settings: Settings | None = None)`
   then `self.settings = settings or get_settings()`.
5. **Register components**: `@register("<kind>", "<name>")` from
   `ragorc.core.registry` on every user-selectable class (splitters, retrievers,
   rerankers, translators, routers, compressors, embedders).
6. **Errors**: raise from `ragorc.core.errors`. Store failures raise
   `StoreUnavailable`; retryable transport raises `TransientError`; guard
   rejections raise `GuardrailViolation` (never retried).
7. **Every LLM-using component takes `llm: LLM` by constructor injection** and
   returns `(result, Usage)` so cost aggregates. Use
   `prompts.get_prompt("name").render(...)` and pass `system=prompt.system`.
8. **Optional imports are lazy and guarded**: import inside the function/method,
   raise `ImportError` with the exact `pip install 'ragorc[extra]'` hint.
9. **Type hints everywhere**, `from __future__ import annotations` at the top of
   every module.
10. **Structured logging**: `log = structlog.get_logger(__name__)`; log events as
    `log.info("event_name", key=value)`, never f-strings.

## Style

- Module docstring explains *why the design is this way*, not what the code does.
  Include the performance or correctness reasoning behind non-obvious choices.
- Comments explain trade-offs and failure modes. Do not narrate syntax.
- Line length 100. Ruff rules: `E,F,W,I,B,C4,UP,ASYNC,S,RUF,PERF,SIM`.
- No `print`. No TODO/FIXME/placeholder — ship working code or don't ship the file.
- Public API of each package is re-exported in its `__init__.py` with `__all__`.

## Verification before you finish

Run, from the repo root, and fix everything it reports:

```bash
.venv/bin/python -c "import <your.module>; print('ok')"
.venv/bin/ruff check <your files> --fix
.venv/bin/ruff format <your files>
```

Report the exact commands you ran and their output.

## Conventions that recur

- Chunk ids: `ragorc.core.ids.chunk_id(doc_id, index, content, level)` — never random.
- Cache keys: `ragorc.core.ids.cache_key(namespace, *parts)`.
- Timing: `with timed("stage_name"): ...` from `ragorc.core.telemetry`.
- Scores are always "higher is better". If a backend returns a distance,
  convert it and say so in a comment.
- Anything returned to the pipeline is `list[ScoredChunk]` with `rank` filled in
  from 0 and `component_scores` populated with the per-retriever contribution.
