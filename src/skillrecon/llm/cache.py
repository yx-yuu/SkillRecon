"""LLM I/O cache with config-hash based invalidation."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from skillrecon.core.config import LLMConfig

if TYPE_CHECKING:
    from skillrecon.llm.client import LLMClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_CACHE_ROOT = Path("derived/llm_cache")


def compute_config_hash(config: LLMConfig, prompt_version: str) -> str:
    """Return a stable cache key for one config and prompt version."""
    payload = (
        f"{config.base_url}|{config.model}|{config.temperature}"
        f"|{config.max_tokens}|{config.structured_output_mode}|{prompt_version}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class LLMCache:
    """Store LLM request/response pairs on disk."""

    def __init__(self, cache_root: Path, config_hash: str) -> None:
        self._root = cache_root / config_hash
        self._config_hash = config_hash

    def _entry_path(self, skill_id: str, call_key: str) -> Path:
        return self._root / skill_id / f"{call_key}.json"

    def exists(self, skill_id: str, call_key: str) -> bool:
        """Return whether the cache already contains this call."""
        return self._entry_path(skill_id, call_key).exists()

    def get(self, skill_id: str, call_key: str) -> dict[str, object] | None:
        """Load one cached entry if it exists and is readable."""
        path = self._entry_path(skill_id, call_key)
        if not path.exists():
            return None
        try:
            data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
            return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt cache entry: %s", path)
            return None

    def put(
        self,
        skill_id: str,
        call_key: str,
        request: dict[str, object],
        response: str,
    ) -> None:
        """Write one cache entry."""
        path = self._entry_path(skill_id, call_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "request": request,
            "response": response,
            "timestamp": time.time(),
            "config_hash": self._config_hash,
        }
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")

    def delete(self, skill_id: str, call_key: str) -> None:
        """Remove one cache entry if it exists."""
        path = self._entry_path(skill_id, call_key)
        try:
            path.unlink()
        except FileNotFoundError:
            return


class CachedLLMClient:
    """Wrap an LLM client with a disk-backed cache."""

    def __init__(self, client: LLMClient, cache: LLMCache) -> None:
        self._client = client
        self._cache = cache

    @property
    def config(self) -> LLMConfig:
        """Return the wrapped LLM configuration."""
        return self._client.config

    @classmethod
    def from_config(
        cls,
        config: LLMConfig,
        prompt_version: str,
        cache_root: Path = DEFAULT_CACHE_ROOT,
    ) -> CachedLLMClient:
        """Build a cached client from runtime configuration."""
        from skillrecon.llm.client import LLMClient

        config_hash = compute_config_hash(config, prompt_version)
        client = LLMClient(config)
        cache = LLMCache(cache_root, config_hash)
        return cls(client, cache)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        skill_id: str,
        call_key: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return a cached or freshly generated text completion."""
        cached = self._cache.get(skill_id, call_key)
        if cached is not None:
            logger.debug("Cache hit: %s/%s", skill_id, call_key)
            return str(cached["response"])

        result = self._client.complete(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        self._cache.put(
            skill_id, call_key, request={"messages": messages}, response=result
        )
        return result

    def structured_complete(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        *,
        skill_id: str,
        call_key: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Return a cached or freshly generated structured completion."""
        cached = self._cache.get(skill_id, call_key)
        if cached is not None:
            logger.debug("Cache hit: %s/%s", skill_id, call_key)
            try:
                return response_model.model_validate_json(str(cached["response"]))
            except ValueError as exc:
                logger.warning(
                    "Invalid structured LLM cache entry for %s/%s; refreshing: %s",
                    skill_id,
                    call_key,
                    exc,
                )
                self._cache.delete(skill_id, call_key)

        result = self._client.structured_complete(
            messages, response_model, temperature=temperature, max_tokens=max_tokens
        )
        raw = result.model_dump_json()
        self._cache.put(
            skill_id, call_key, request={"messages": messages}, response=raw
        )
        return result
