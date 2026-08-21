"""OpenRouter chat client.

Why a hand-written client instead of a framework wrapper
-------------------------------------------------------
This is the hottest path in the library — a single query makes 10-40 calls —
and four things we need are either absent or awkward in generic wrappers:

1. **Native structured output with a repair loop.** Every router and grader is
   a schema-constrained call. When a provider ignores ``response_format`` we
   need to detect it and degrade, not crash.
2. **OpenRouter's provider-routing controls.** ``provider.sort=price`` and
   ``data_collection=deny`` are the two settings that decide what a query costs
   and whether your documents get used as training data.
3. **Real cost accounting.** Passing ``usage: {include: true}`` makes OpenRouter
   return the *actual* charged cost per call, which beats any local price table.
4. **A single shared semaphore, token bucket and cache** across every call site.

Transport: one ``httpx.AsyncClient`` with HTTP/2, so 16 concurrent requests
multiplex over one TCP connection instead of opening 16. Bodies are encoded and
decoded with orjson.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import orjson
import structlog
from pydantic import BaseModel, ValidationError

from ragorc.core.concurrency import RateLimiter, retry_async
from ragorc.core.errors import (
    LLMError,
    RateLimited,
    StructuredOutputError,
    TransientError,
)
from ragorc.core.models import Usage
from ragorc.core.settings import LLMSettings, get_settings
from ragorc.core.telemetry import current_ledger, redact_identifiers

log = structlog.get_logger(__name__)

__all__ = ["OpenRouterLLM", "to_strict_json_schema"]

_RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})


def to_strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Turn a pydantic model into an OpenAI/OpenRouter *strict* JSON schema.

    Strict mode has three requirements pydantic does not emit by default:
    every object must set ``additionalProperties: false``, every property must
    appear in ``required``, and defaults/formats that providers reject must be
    stripped. Optional fields become nullable unions instead of omitted keys —
    strict mode has no notion of an optional property.
    """
    schema = model.model_json_schema(mode="serialization")

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node
        node = {k: v for k, v in node.items() if k not in ("default", "$comment", "examples")}
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            props = node.get("properties") or {}
            if props:
                node["required"] = list(props)
        for key, value in list(node.items()):
            node[key] = walk(value)
        return node

    walked = walk(schema)
    walked.setdefault("type", "object")
    return walked


class OpenRouterLLM:
    """Async chat client satisfying :class:`ragorc.core.protocols.LLM`."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        cache: Any | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings().llm
        self.cache = cache
        self._semaphore = asyncio.Semaphore(self.settings.max_concurrency)
        # Streams get their own pool. See LLMSettings.max_concurrent_streams: a
        # generator holds its permit until the consumer stops reading, so sharing
        # one pool lets slow SSE clients block the entire pipeline.
        self._stream_semaphore = asyncio.Semaphore(max(1, self.settings.max_concurrent_streams))
        self._limiter = RateLimiter(
            requests_per_minute=self.settings.requests_per_minute,
            tokens_per_minute=self.settings.tokens_per_minute,
        )
        self._client = client
        self._owns_client = client is None
        # Models observed to reject json_schema. Cached so we stop paying for
        # a failed strict attempt on every subsequent call to that model.
        self._no_strict: set[str] = set()

    # -- lifecycle ---------------------------------------------------------
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            key = self.settings.api_key.get_secret_value()
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "X-Title": self.settings.app_name,
            }
            if self.settings.site_url:
                headers["HTTP-Referer"] = self.settings.site_url
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url.rstrip("/"),
                headers=headers,
                http2=self.settings.http2,
                timeout=httpx.Timeout(
                    self.settings.timeout_s, connect=self.settings.connect_timeout_s
                ),
                limits=httpx.Limits(
                    max_connections=self.settings.max_concurrency * 2,
                    max_keepalive_connections=self.settings.max_concurrency,
                    keepalive_expiry=60.0,
                ),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> OpenRouterLLM:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # -- request building --------------------------------------------------
    def _messages(
        self, prompt: str, system: str | None, cache_system: bool
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system:
            if cache_system and self.settings.enable_prompt_cache and len(system) > 2000:
                # Anthropic-style prompt caching. A long, static system prompt
                # (schema descriptions, few-shot examples, DB DDL) then costs
                # ~10% on repeat calls instead of full price every time.
                messages.append(
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )
            else:
                messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _body(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        temperature: float | None,
        max_tokens: int | None,
        stop: Sequence[str] | None,
        stream: bool,
        response_format: dict[str, Any] | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "stream": stream,
            # Ask OpenRouter to report the real charged cost with the response.
            "usage": {"include": True},
        }
        if stop:
            body["stop"] = list(stop)
        if response_format:
            body["response_format"] = response_format

        provider: dict[str, Any] = {
            "allow_fallbacks": self.settings.allow_fallbacks,
            "require_parameters": self.settings.require_parameters,
            "data_collection": self.settings.data_collection,
        }
        if self.settings.provider_order:
            provider["order"] = list(self.settings.provider_order)
        if self.settings.provider_sort:
            provider["sort"] = self.settings.provider_sort
        body["provider"] = provider
        body.update(extra)
        return body

    @staticmethod
    def _usage_from(payload: dict[str, Any], model: str, latency_ms: float) -> Usage:
        raw = payload.get("usage") or {}
        # OpenRouter reports `cost` in credits (== USD). Fall back to 0 rather
        # than guessing from a stale price table.
        cost = raw.get("cost")
        if cost is None:
            cost = (raw.get("cost_details") or {}).get("upstream_inference_cost", 0.0)
        return Usage(
            model=payload.get("model") or model,
            prompt_tokens=int(raw.get("prompt_tokens") or 0),
            completion_tokens=int(raw.get("completion_tokens") or 0),
            cost_usd=float(cost or 0.0),
            latency_ms=latency_ms,
            calls=1,
        )

    @staticmethod
    def _provider_detail(text: str, limit: int = 400) -> str:
        """Clip a provider error body and strip the operator's identity from it.

        The body is genuinely useful — it carries the sentence that says what is
        actually wrong — but OpenRouter's 4xx bodies also carry a key-management
        URL containing the key id and an account ``user_id``. That body is
        attached to the raised error, and from there it reaches the abstention
        reason, the HTTP error detail and the CLI, so it ends up shown to whoever
        called the API. Redacting once here covers all of them; the unredacted
        body still goes to the local log.
        """
        return redact_identifiers(text[:limit])

    # -- core call ---------------------------------------------------------
    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """Post one completion, retrying transient failures.

        The retry policy comes from settings. It used to be a decorator evaluated
        at import — `@retry_async(max_attempts=4, ...)` — which froze the policy
        before any configuration existed, so `llm.max_retries`,
        `retry_base_delay_s` and `retry_max_delay_s` were three documented knobs
        that could not move anything.
        """
        return await retry_async(
            max_attempts=max(1, self.settings.max_retries),
            base_delay=self.settings.retry_base_delay_s,
            max_delay=self.settings.retry_max_delay_s,
            retry_on=(TransientError,),
        )(self._post_once)(body)

    async def _post_once(self, body: dict[str, Any]) -> dict[str, Any]:
        await self._limiter.acquire()
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                response = await self.client.post("/chat/completions", content=orjson.dumps(body))
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                raise TransientError(f"transport failure: {exc}") from exc
            latency_ms = (loop.time() - started) * 1000.0

        if response.status_code in _RETRY_STATUS:
            retry_after = response.headers.get("retry-after")
            message = self._provider_detail(response.text)
            if response.status_code == 429:
                raise RateLimited(
                    "rate limited by provider",
                    retry_after=float(retry_after) if retry_after else None,
                    body=message,
                )
            raise TransientError(
                f"provider returned {response.status_code}",
                status=response.status_code,
                body=message,
            )
        if response.status_code == 402:
            # Out of credits. Worth its own branch because the provider's body is
            # ~500 characters of JSON containing a key-management URL with the key
            # id in it, and the CLI prints whatever it is given — so the useful
            # sentence ("you have N tokens of credit left") arrives buried in noise
            # that also puts an identifier on someone's terminal. The full body
            # still goes to the log.
            log.warning("provider_payment_required", body=response.text[:600])
            raise LLMError(
                "OpenRouter rejected the request for insufficient credit",
                status=402,
                hint=(
                    "add credits at https://openrouter.ai/settings/credits, or "
                    "lower llm.max_tokens / generation.max_answer_tokens to fit "
                    "the remaining balance"
                ),
            )
        if response.status_code in (401, 403):
            raise LLMError(
                "OpenRouter rejected the credentials",
                status=response.status_code,
                hint="check RAGORC_LLM__API_KEY; keys are at https://openrouter.ai/keys",
            )
        if response.status_code == 404:
            raise LLMError(
                "OpenRouter does not recognize that model",
                status=404,
                model=body.get("model"),
                hint="check the id against https://openrouter.ai/models",
            )
        if response.status_code >= 400:
            raise LLMError(
                f"provider returned {response.status_code}",
                status=response.status_code,
                body=response.text[:600],
            )

        payload = orjson.loads(response.content)
        # OpenRouter can return HTTP 200 with an error object in the body.
        if "error" in payload and not payload.get("choices"):
            err = payload["error"]
            code = err.get("code")
            message = err.get("message", "unknown provider error")
            if code in (429, 502, 503):
                raise TransientError(message, code=code)
            raise LLMError(message, code=code)
        payload["_latency_ms"] = latency_ms
        return payload

    def _record(self, usage: Usage, stage: str) -> None:
        ledger = current_ledger()
        if ledger is not None:
            ledger.record(usage, stage=stage)

    def _precheck(self) -> None:
        ledger = current_ledger()
        if ledger is not None:
            ledger.check()

    # -- public API --------------------------------------------------------
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: Sequence[str] | None = None,
        stage: str = "complete",
        cache_key_extra: Any = None,
        **kwargs: Any,
    ) -> tuple[str, Usage]:
        model = model or self.settings.model
        self._precheck()

        cached = None
        if self.cache is not None:
            cached = await self.cache.get_completion(
                prompt=prompt,
                system=system,
                model=model,
                temperature=self.settings.temperature if temperature is None else temperature,
                extra=cache_key_extra,
            )
        if cached is not None:
            usage = Usage(model=model, calls=1, cached=1)
            self._record(usage, stage)
            return cached, usage

        body = self._body(
            self._messages(prompt, system, cache_system=True),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            stream=False,
            response_format=None,
            extra=kwargs,
        )
        payload = await self._post(body)
        text = self._first_text(payload)
        usage = self._usage_from(payload, model, payload.get("_latency_ms", 0.0))
        self._record(usage, stage)

        if self.cache is not None:
            await self.cache.set_completion(
                prompt=prompt,
                system=system,
                model=model,
                temperature=self.settings.temperature if temperature is None else temperature,
                extra=cache_key_extra,
                value=text,
            )
        return text, usage

    @staticmethod
    def _first_text(payload: dict[str, Any]) -> str:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("provider returned no choices", payload_keys=list(payload))
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some providers return content parts rather than a string.
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not content:
            # A reasoning model may put everything in `reasoning` if max_tokens
            # was exhausted before it started the visible answer.
            content = message.get("reasoning") or ""
        return content or ""

    async def structured(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        stage: str = "structured",
        max_repairs: int = 1,
        **kwargs: Any,
    ) -> tuple[Any, Usage]:
        """Return a validated ``schema`` instance.

        Three-tier degradation, because provider support is uneven:

        1. native ``json_schema`` with ``strict: true`` — best, no parsing risk;
        2. ``json_object`` mode with the schema inlined in the prompt;
        3. a repair round-trip that shows the model its own validation errors.
        """
        model = model or self.settings.fast_model
        self._precheck()
        json_schema = to_strict_json_schema(schema)

        cached = None
        if self.cache is not None:
            cached = await self.cache.get_completion(
                prompt=prompt,
                system=system,
                model=model,
                temperature=temperature or 0.0,
                extra=("schema", schema.__name__),
            )
        if cached is not None:
            try:
                obj = schema.model_validate_json(cached)
                usage = Usage(model=model, calls=1, cached=1)
                self._record(usage, stage)
                return obj, usage
            except ValidationError:
                pass  # stale cache entry; fall through and re-ask

        use_strict = model not in self._no_strict
        response_format: dict[str, Any] | None = (
            {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "strict": True, "schema": json_schema},
            }
            if use_strict
            else {"type": "json_object"}
        )
        effective_system = system
        if not use_strict:
            effective_system = (f"{system}\n\n" if system else "") + (
                "Respond with a single JSON object and nothing else. It must validate "
                f"against this JSON Schema:\n{orjson.dumps(json_schema).decode()}"
            )

        total = Usage(model=model)
        raw = ""
        last_error = ""
        # Popped once, outside the loop. Inside it, `kwargs.pop` consumed the
        # caller's cap on the first attempt and every repair after it silently
        # fell back to the global `llm.max_tokens` — the repair, which is the
        # attempt most likely to run long, was the one attempt running uncapped.
        structured_max_tokens = kwargs.pop("max_tokens", None)
        for attempt in range(max_repairs + 2):
            body = self._body(
                self._messages(
                    prompt if attempt == 0 else self._repair_prompt(prompt, raw, last_error),
                    effective_system,
                    cache_system=True,
                ),
                model=model,
                temperature=temperature if temperature is not None else 0.0,
                max_tokens=structured_max_tokens,
                stop=None,
                stream=False,
                response_format=response_format,
                extra=kwargs,
            )
            try:
                payload = await self._post(body)
            except LLMError as exc:
                # Provider rejected response_format -> remember and retry once
                # in json_object mode instead of failing the request.
                blob = str(exc).lower()
                if use_strict and (
                    "response_format" in blob or "json_schema" in blob or "schema" in blob
                ):
                    log.info("strict_output_unsupported", model=model)
                    self._no_strict.add(model)
                    use_strict = False
                    response_format = {"type": "json_object"}
                    effective_system = (f"{system}\n\n" if system else "") + (
                        "Respond with a single JSON object and nothing else. It must "
                        f"validate against this JSON Schema:\n{orjson.dumps(json_schema).decode()}"
                    )
                    continue
                raise

            total = total + self._usage_from(payload, model, payload.get("_latency_ms", 0.0))
            raw = self._first_text(payload)
            candidate = _extract_json(raw)
            try:
                obj = schema.model_validate_json(candidate)
            except ValidationError as exc:
                last_error = _compact_validation_error(exc)
                log.info(
                    "structured_output_invalid",
                    model=model,
                    schema=schema.__name__,
                    attempt=attempt,
                    error=last_error[:300],
                )
                continue

            self._record(total, stage)
            if self.cache is not None:
                await self.cache.set_completion(
                    prompt=prompt,
                    system=system,
                    model=model,
                    temperature=temperature or 0.0,
                    extra=("schema", schema.__name__),
                    value=obj.model_dump_json(),
                )
            return obj, total

        self._record(total, stage)
        raise StructuredOutputError(
            f"could not obtain valid {schema.__name__}",
            model=model,
            last_error=last_error,
            raw=raw[:500],
        )

    @staticmethod
    def _repair_prompt(original: str, raw: str, error: str) -> str:
        return (
            f"{original}\n\n"
            f"Your previous response was not valid:\n---\n{raw[:2000]}\n---\n"
            f"Validation errors: {error}\n"
            "Return corrected JSON only."
        )

    async def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str = "stream",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield content deltas. The final ``usage`` chunk is recorded in the
        ledger so streamed answers are costed like any other."""
        model = model or self.settings.model
        self._precheck()
        body = self._body(
            self._messages(prompt, system, cache_system=True),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=None,
            stream=True,
            response_format=None,
            extra=kwargs,
        )
        await self._limiter.acquire()
        # `_stream_semaphore`, not `_semaphore`: this block stays entered across
        # every `yield` below, for as long as the caller takes to consume the
        # generator. Held on the shared pool, `max_concurrency` slow readers were
        # enough to stop all other LLM work in the process.
        async with self._stream_semaphore:
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                async with self.client.stream(
                    "POST", "/chat/completions", content=orjson.dumps(body)
                ) as response:
                    if response.status_code >= 400:
                        detail = self._provider_detail(
                            (await response.aread()).decode(errors="replace"), 600
                        )
                        if response.status_code in _RETRY_STATUS:
                            raise TransientError(
                                f"stream failed {response.status_code}", body=detail
                            )
                        raise LLMError(f"stream failed {response.status_code}", body=detail)
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = orjson.loads(data)
                        except orjson.JSONDecodeError:
                            continue
                        if usage := event.get("usage"):
                            self._record(
                                self._usage_from(
                                    {"usage": usage, "model": event.get("model") or model},
                                    model,
                                    (loop.time() - started) * 1000.0,
                                ),
                                stage,
                            )
                        for choice in event.get("choices") or []:
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                yield delta
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                raise TransientError(f"stream transport failure: {exc}") from exc

    async def batch(
        self,
        prompts: Sequence[str],
        *,
        system: str | None = None,
        model: str | None = None,
        stage: str = "batch",
        **kwargs: Any,
    ) -> list[tuple[str, Usage]]:
        """Fan out under the shared semaphore.

        Every map-style stage (multi-query, per-chunk grading, RAPTOR
        summarization) goes through here, so the concurrency ceiling is
        enforced in exactly one place.
        """
        if not prompts:
            return []
        results = await asyncio.gather(
            *(self.complete(p, system=system, model=model, stage=stage, **kwargs) for p in prompts),
            return_exceptions=True,
        )
        out: list[tuple[str, Usage]] = []
        for item in results:
            if isinstance(item, BaseException):
                log.warning("batch_item_failed", error=str(item)[:200])
                out.append(("", Usage(model=model or self.settings.model)))
            else:
                out.append(item)
        return out

    async def batch_structured(
        self,
        prompts: Sequence[str],
        schema: type[BaseModel],
        *,
        system: str | None = None,
        model: str | None = None,
        stage: str = "batch_structured",
        **kwargs: Any,
    ) -> list[tuple[Any | None, Usage]]:
        """Structured fan-out. A failed item yields ``None`` rather than
        aborting the batch — one unparseable grade should not lose the other 49."""
        if not prompts:
            return []
        results = await asyncio.gather(
            *(
                self.structured(p, schema, system=system, model=model, stage=stage, **kwargs)
                for p in prompts
            ),
            return_exceptions=True,
        )
        out: list[tuple[Any | None, Usage]] = []
        for item in results:
            if isinstance(item, BaseException):
                log.warning("batch_structured_failed", error=str(item)[:200])
                out.append((None, Usage(model=model or self.settings.fast_model)))
            else:
                out.append(item)
        return out

    async def fetch_model_prices(self) -> dict[str, dict[str, float]]:
        """Live prices from ``/models``, for pre-flight cost estimation."""
        response = await self.client.get("/models")
        response.raise_for_status()
        data = orjson.loads(response.content).get("data") or []
        prices: dict[str, dict[str, float]] = {}
        for entry in data:
            pricing = entry.get("pricing") or {}
            try:
                prices[entry["id"]] = {
                    "prompt": float(pricing.get("prompt") or 0.0),
                    "completion": float(pricing.get("completion") or 0.0),
                    "context_length": float(entry.get("context_length") or 0),
                }
            except (TypeError, ValueError):
                continue
        return prices


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a reply that may be wrapped in prose or fences.

    Providers without strict mode routinely add "Here is the JSON:" and a
    ```json fence. Balanced-brace scanning is more reliable than a regex
    because it handles nested objects and braces inside strings.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    if "```" in text:
        start = text.find("```")
        fence_end = text.find("\n", start)
        if fence_end != -1:
            end = text.find("```", fence_end)
            inner = text[fence_end + 1 : end if end != -1 else len(text)].strip()
            if inner.startswith("{"):
                text = inner
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _compact_validation_error(exc: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:6]]
    return "; ".join(parts)
