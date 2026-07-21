"""LLM abstraction layer: OpenAI-compatible client with caching."""

from __future__ import annotations

__all__ = [
    "CachedLLMClient",
    "LLMCache",
    "LLMClient",
    "compute_config_hash",
]


def __getattr__(name: str) -> object:
    if name in {"CachedLLMClient", "LLMCache", "compute_config_hash"}:
        from skillrecon.llm.cache import (
            CachedLLMClient,
            LLMCache,
            compute_config_hash,
        )

        exports = {
            "CachedLLMClient": CachedLLMClient,
            "LLMCache": LLMCache,
            "compute_config_hash": compute_config_hash,
        }
        return exports[name]
    if name == "LLMClient":
        from skillrecon.llm.client import LLMClient

        return LLMClient
    raise AttributeError(name)
