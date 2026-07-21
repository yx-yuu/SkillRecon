from __future__ import annotations

import httpx
import pytest
from openai import APIConnectionError
from pydantic import BaseModel

from skillrecon.core.config import LLMConfig
from skillrecon.llm.cache import CachedLLMClient, LLMCache
from skillrecon.llm.client import LLMClient


class _NumberList(BaseModel):
    items: list[int]


class _FakeStructuredClient:
    def __init__(self) -> None:
        self.config = LLMConfig(
            base_url="https://example.test/v1",
            model="unit-test-model",
            api_key_env="literal-key",
        )
        self.calls = 0

    def structured_complete(
        self,
        messages: list[dict[str, str]],  # noqa: ARG002
        response_model: type[_NumberList],  # noqa: ARG002
        *,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> _NumberList:
        self.calls += 1
        return _NumberList(items=[7])


def test_llm_retry_handles_transient_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("skillrecon.llm.client.time.sleep", lambda delay: None)
    client = object.__new__(LLMClient)
    attempts = {"count": 0}
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")

    def flaky_call() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise APIConnectionError(request=request)
        return "ok"

    assert client._with_retry("test", flaky_call) == "ok"
    assert attempts["count"] == 3


def test_llm_retry_does_not_hide_non_retryable_errors() -> None:
    client = object.__new__(LLMClient)
    attempts = {"count": 0}

    def broken_call() -> str:
        attempts["count"] += 1
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        client._with_retry("test", broken_call)

    assert attempts["count"] == 1


def test_json_prompt_retries_until_schema_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = object.__new__(LLMClient)
    client._config = LLMConfig(
        base_url="https://example.test/v1",
        model="unit-test-model",
        api_key_env="literal-key",
    )
    responses = iter(["{}", '{"items":', '{"items":[1,2]}'])
    prompts: list[str] = []

    def fake_completion(self, messages, *, temperature, max_tokens):  # noqa: ANN001
        prompts.append(messages[-1]["content"])
        return next(responses)

    monkeypatch.setattr(
        LLMClient,
        "_run_json_prompt_completion",
        fake_completion,
    )

    result = client._json_prompt_complete(
        [{"role": "user", "content": "return numbers"}],
        _NumberList,
        _NumberList.model_json_schema(),
        temperature=None,
        max_tokens=None,
    )

    assert result.items == [1, 2]
    assert len(prompts) == 3
    assert "previous response was invalid JSON" in prompts[1]


def test_json_prompt_still_fails_after_bounded_schema_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = object.__new__(LLMClient)
    client._config = LLMConfig(
        base_url="https://example.test/v1",
        model="unit-test-model",
        api_key_env="literal-key",
    )
    attempts = {"count": 0}

    def fake_completion(self, messages, *, temperature, max_tokens):  # noqa: ANN001, ARG001
        attempts["count"] += 1
        return "{}"

    monkeypatch.setattr(
        LLMClient,
        "_run_json_prompt_completion",
        fake_completion,
    )

    with pytest.raises(Exception, match="Field required"):
        client._json_prompt_complete(
            [{"role": "user", "content": "return numbers"}],
            _NumberList,
            _NumberList.model_json_schema(),
            temperature=None,
            max_tokens=None,
        )

    assert attempts["count"] == 4


def test_cached_structured_completion_refreshes_invalid_cache(tmp_path) -> None:  # noqa: ANN001
    cache = LLMCache(tmp_path, "config-hash")
    cache.put(
        "owner/skill",
        "call-key",
        request={"messages": []},
        response="{}",
    )
    fake_client = _FakeStructuredClient()
    client = CachedLLMClient(fake_client, cache)  # type: ignore[arg-type]

    result = client.structured_complete(
        [{"role": "user", "content": "return numbers"}],
        _NumberList,
        skill_id="owner/skill",
        call_key="call-key",
    )

    assert result.items == [7]
    assert fake_client.calls == 1
    refreshed = cache.get("owner/skill", "call-key")
    assert refreshed is not None
    assert _NumberList.model_validate_json(str(refreshed["response"])).items == [7]
