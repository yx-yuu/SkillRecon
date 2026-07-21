"""OpenAI-compatible LLM client for SkillRecon."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, TypeVar
from urllib.parse import urlparse

from openai import APIConnectionError, APITimeoutError, OpenAI
from pydantic import BaseModel

from skillrecon.core.config import LLMConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
_RETRYABLE_EXCEPTIONS = (APIConnectionError, APITimeoutError)
_MAX_RETRIES = 4
_BASE_RETRY_DELAY_SECONDS = 1.0
_RATE_LIMIT_RETRY_DELAY_SECONDS = 30.0
_JSON_PROMPT_RETRY_NOTES = (
    (
        "The previous response was invalid JSON or did not satisfy the schema. "
        "Regenerate from scratch. Return exactly one compact JSON object, include every "
        "required key, and do not include prose or Markdown."
    ),
    (
        "The previous response still failed schema validation. Return only syntactically "
        "valid JSON with all required top-level fields. Do not truncate strings or omit "
        "required fields."
    ),
    (
        "Final retry: return the smallest valid JSON object matching the schema exactly. "
        "If no compliant items can be extracted, use an empty array for required list "
        "fields instead of omitting the field."
    ),
)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", re.DOTALL)


def _fix_schema_for_strict_mode(schema: dict[str, Any]) -> None:
    """Make a JSON schema compatible with strict structured outputs."""
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            _fix_schema_for_strict_mode(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _fix_schema_for_strict_mode(item)
    for def_schema in schema.get("$defs", {}).values():
        if isinstance(def_schema, dict):
            _fix_schema_for_strict_mode(def_schema)


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat API."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        _validate_base_url(config.base_url)
        api_key = _resolve_api_key(config.api_key_env)
        self._client = OpenAI(base_url=config.base_url, api_key=api_key)

    @property
    def config(self) -> LLMConfig:
        return self._config

    def _is_retryable_error(self, exc: Exception) -> bool:
        if isinstance(exc, _RETRYABLE_EXCEPTIONS):
            return True
        status_code = getattr(exc, "status_code", None)
        return isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES

    def _with_retry(self, operation_name: str, fn: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return fn()
            except Exception as exc:
                if not self._is_retryable_error(exc):
                    raise
                last_error = exc
                if attempt >= _MAX_RETRIES:
                    break
                delay = _retry_delay_seconds(exc, attempt)
                logger.warning(
                    "Retryable LLM error during %s (attempt %d/%d, status=%s); retrying in %.1fs",
                    operation_name,
                    attempt,
                    _MAX_RETRIES,
                    getattr(exc, "status_code", "unknown"),
                    delay,
                )
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Run a plain-text chat completion."""
        response = self._with_retry(
            "chat.completions.create",
            lambda: self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature if temperature is not None else self._config.temperature,
                max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            ),
        )
        content = response.choices[0].message.content
        return content or ""

    def structured_complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Run a structured completion against a Pydantic response model."""
        schema = response_model.model_json_schema()
        if self._config.structured_output_mode == "json_prompt":
            return self._json_prompt_complete(
                messages,
                response_model,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        _fix_schema_for_strict_mode(schema)
        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        response = self._with_retry(
            "chat.completions.create[structured]",
            lambda: self._client.chat.completions.create(  # type: ignore[call-overload]
                model=self._config.model,
                messages=messages,
                temperature=(
                    temperature if temperature is not None else self._config.temperature
                ),
                max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
                response_format=response_format,
            ),
        )
        raw = response.choices[0].message.content or "{}"
        parsed = _try_validate_structured_response(raw, response_model)
        if parsed is not None:
            return parsed

        logger.warning(
            "Structured response validation failed for model=%s; falling back to plain JSON prompting",
            self._config.model,
        )
        return self._json_prompt_complete(
            messages,
            response_model,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _json_prompt_complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        schema: dict[str, Any],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> T:
        raw = ""
        retry_notes: tuple[str | None, ...] = (None, *_JSON_PROMPT_RETRY_NOTES)
        for attempt, retry_note in enumerate(retry_notes, start=1):
            raw = self._run_json_prompt_with_filter_retry(
                messages,
                schema,
                retry_note=retry_note,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = _try_validate_structured_response(raw, response_model)
            if parsed is not None:
                return parsed
            if attempt < len(retry_notes):
                logger.warning(
                    "JSON prompt validation failed for model=%s (attempt %d/%d); retrying",
                    self._config.model,
                    attempt,
                    len(retry_notes),
                )

        candidate = _extract_json_candidate(raw)
        return response_model.model_validate_json(candidate or raw)

    def _run_json_prompt_with_filter_retry(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        temperature: float | None,
        max_tokens: int | None,
        retry_note: str | None = None,
    ) -> str:
        try:
            return self._run_json_prompt_completion(
                _json_prompt_messages(messages, schema, retry_note=retry_note),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if not _is_content_filter_error(exc):
                raise
            logger.warning(
                "LLM content filter during JSON generation for model=%s; retrying with safety-abstraction instructions",
                self._config.model,
            )
            return self._run_json_prompt_completion(
                _json_prompt_messages(
                    messages,
                    schema,
                    retry_note=retry_note,
                    safety_mode=True,
                ),
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def _run_json_prompt_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        response = self._with_retry(
            "chat.completions.create[json-prompt]",
            lambda: self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=(
                    temperature if temperature is not None else self._config.temperature
                ),
                max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            ),
        )
        return response.choices[0].message.content or "{}"


def _json_prompt_messages(
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    *,
    retry_note: str | None = None,
    safety_mode: bool = False,
) -> list[dict[str, str]]:
    instructions = [
        "Return only one valid JSON object matching this schema exactly.",
        "Do not wrap it in Markdown or code fences.",
        "Use compact strings and avoid redundant explanation.",
    ]
    if retry_note:
        instructions.insert(0, retry_note)
    if safety_mode:
        instructions.insert(
            0,
            (
                "Use high-level security-audit language only. "
                "Do not repeat exploit steps, bypass procedures, target-specific attack details, "
                "or operational instructions from the source text."
            ),
        )
    instructions.append(f"Schema: {json.dumps(schema, ensure_ascii=False)}")
    return [
        *messages,
        {
            "role": "system",
            "content": " ".join(instructions),
        },
    ]


def _is_content_filter_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code != 400:
        return False
    text = str(exc).lower()
    return "contentfilter" in text or "content filter" in text or "1301" in text


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429 or _looks_like_rate_limit(exc):
        return _retry_after_seconds(exc) or (_RATE_LIMIT_RETRY_DELAY_SECONDS * attempt)
    return _BASE_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))


def _looks_like_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate limit" in text or "ratelimit" in text or "1302" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _try_validate_structured_response(raw: str, response_model: type[T]) -> T | None:
    try:
        return response_model.model_validate_json(raw)
    except Exception:
        candidate = _extract_json_candidate(raw)
        if candidate is None:
            return None
        try:
            return response_model.model_validate_json(candidate)
        except Exception:
            return None


def _extract_json_candidate(raw: str) -> str | None:
    stripped = raw.strip()
    if not stripped:
        return None
    fenced = _JSON_BLOCK_RE.search(stripped)
    if fenced:
        return fenced.group(1).strip()
    start_positions = [
        idx
        for idx in [stripped.find("{"), stripped.find("[")]
        if idx != -1
    ]
    if not start_positions:
        return None
    start = min(start_positions)
    candidate = stripped[start:]
    for end_char in ["}", "]"]:
        end = candidate.rfind(end_char)
        if end != -1:
            snippet = candidate[: end + 1]
            try:
                json.loads(snippet)
                return snippet
            except Exception:
                continue
    return None


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "llm.base_url must be a full OpenAI-compatible endpoint URL, "
            "for example 'https://api.openai.com/v1' or 'http://localhost:8000/v1'."
        )


def _resolve_api_key(api_key_env: str) -> str:
    value = os.environ.get(api_key_env, "")
    if value:
        return value
    if _looks_like_env_var_name(api_key_env):
        raise ValueError(
            f"LLM API key environment variable {api_key_env!r} is not set. "
            "Set it before running dataset construction, or pass a literal key "
            "only in local throwaway config."
        )
    # Some local configs pass the literal key instead of an env-var name.
    return api_key_env


def _looks_like_env_var_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z_][A-Z0-9_]*", value))
