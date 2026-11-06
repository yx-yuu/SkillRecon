"""Frozen configuration models for SkillRecon experiments."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "env_config.json"
DEFAULT_LLM_CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "llm_config.json"


class LLMConfig(BaseModel):
    """Configuration for an OpenAI-compatible LLM endpoint."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str
    api_key_env: str = "SKILLRECON_API_KEY"
    api_key_file: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    structured_output_mode: Literal["json_schema", "json_prompt"] = "json_schema"
    cache_only: bool = False
    cache_root: str | None = None


class VotePolicy(BaseModel):
    """Policy for self-consistency voting in ICCM."""

    model_config = ConfigDict(frozen=True)

    n_samples: int = 5
    agreement_threshold: float = 0.6


class ReconciliationPolicy(BaseModel):
    """Tunable knobs for candidate generation and reconciliation."""

    model_config = ConfigDict(frozen=True)

    max_candidates_per_behavior: int | None = 8
    max_alignment_fallbacks_per_step: int = 3
    max_semantic_event_fallbacks: int = 3
    max_semantic_path_fallbacks: int = 2
    overlap_policy_path: str = "experiments/configs/overlap_policy_v2.json"
    witness_max_nodes: int = 18
    witness_max_edges: int = 36


class AnalyzerConfig(BaseModel):
    """Global analyzer configuration (one per experiment)."""

    model_config = ConfigDict(frozen=True)

    llm: LLMConfig
    vote_policy: VotePolicy = VotePolicy()
    reconciliation_policy: ReconciliationPolicy = ReconciliationPolicy()
    taxonomy_version: str = "v3"
    prompt_version: str = "v2"

    @property
    def taxonomy_path(self) -> Path:
        """Return the configured taxonomy file path."""
        return PROJECT_ROOT / "experiments" / "configs" / f"taxonomy_{self.taxonomy_version}.json"


class RunConfig(BaseModel):
    """Configuration for a single experiment run."""

    model_config = ConfigDict(frozen=True)

    skill_ids: list[str]
    output_dir: str
    analyzer_config_path: str
    codeql_query_suite: str = "v1"


class CodeQLConfig(BaseModel):
    """Local CodeQL toolchain settings from ``env_config.json``."""

    model_config = ConfigDict(frozen=True)

    bin: str
    home: str | None = None
    note: str | None = None


class EnvConfig(BaseModel):
    """Local environment settings for debugging and experiment runners."""

    model_config = ConfigDict(frozen=True)

    codeql: CodeQLConfig | None = None
    llm: LLMConfig | None = None
    vote_policy: VotePolicy = VotePolicy()
    reconciliation: ReconciliationPolicy = ReconciliationPolicy()


def load_env_config(path: Path = DEFAULT_ENV_CONFIG_PATH) -> EnvConfig:
    """Load the local environment configuration JSON."""
    resolved_path = path.resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    return EnvConfig.model_validate(payload)


def load_llm_config(path: Path = DEFAULT_LLM_CONFIG_PATH) -> LLMConfig:
    """Load a standalone LLM config JSON.

    Accept standalone LLM fields or a top-level ``llm`` object.
    """
    resolved_path = path.resolve()
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("llm"), dict):
        payload = payload["llm"]
    if "url" in payload:
        payload["base_url"] = payload.pop("url")
        payload.pop("api_key")
        payload["api_key_file"] = str(resolved_path)
    return LLMConfig.model_validate(payload)


def resolve_llm_config(
    *,
    llm_config_path: Path | None = DEFAULT_LLM_CONFIG_PATH,
    env_config: EnvConfig | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    structured_output_mode: Literal["json_schema", "json_prompt"] | None = None,
) -> LLMConfig:
    """Resolve the LLM config used by scripts.

    Load the selected configuration and apply explicit command-line overrides.
    """
    base_config: LLMConfig | None = None
    if llm_config_path is not None:
        resolved_path = llm_config_path.resolve()
        if resolved_path.is_file():
            base_config = load_llm_config(resolved_path)
        elif resolved_path != DEFAULT_LLM_CONFIG_PATH.resolve():
            raise FileNotFoundError(f"LLM config file not found: {llm_config_path}")

    if base_config is None and env_config is not None:
        base_config = env_config.llm

    if base_config is None:
        if base_url is None or model is None:
            raise ValueError(
                "missing LLM config: provide --llm-config or --base-url/--model"
            )
        base_config = LLMConfig(base_url=base_url, model=model)

    overrides = {
        key: value
        for key, value in {
            "base_url": base_url,
            "model": model,
            "api_key_env": api_key_env,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "structured_output_mode": structured_output_mode,
        }.items()
        if value is not None
    }
    if api_key_env is not None:
        overrides["api_key_file"] = None
    return base_config.model_copy(update=overrides)
