#!/usr/bin/env python3
"""Generate ablation artifact roots from existing SkillRecon artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    AnalyzerConfig,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.evaluation.ablation import materialize_ablation_artifacts
from skillrecon.evaluation.artifacts import load_skill_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ablation artifact roots from existing SkillRecon outputs"
    )
    parser.add_argument(
        "--skills-file",
        required=True,
        help="Path to newline-delimited skill ids",
    )
    parser.add_argument(
        "--data-root",
        default="data/skill_dataset",
        help="Root directory containing local skill packages",
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Root directory containing full-system artifacts",
    )
    parser.add_argument(
        "--output-root",
        default="derived/ablations",
        help="Root directory for ablation artifact outputs",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json used for LLM defaults",
    )
    parser.add_argument("--base-url", help="LLM API base URL override")
    parser.add_argument("--model", help="LLM model name override")
    parser.add_argument("--api-key-env", help="API key env var or literal key override")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
    parser.add_argument(
        "--taxonomy-version",
        default="v2",
        help="Taxonomy version suffix, e.g. v2 -> taxonomy_v2.json",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt version for analyzer config",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip ablation variants whose required artifacts already exist",
    )
    args = parser.parse_args()

    env_config_path = Path(args.env_config)
    env_config = load_env_config(env_config_path) if env_config_path.is_file() else None
    try:
        llm_config = resolve_llm_config(
            llm_config_path=Path(args.llm_config),
            env_config=env_config,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    analyzer_config = AnalyzerConfig(
        llm=llm_config,
        taxonomy_version=args.taxonomy_version,
        prompt_version=args.prompt_version,
    )
    skill_ids = load_skill_ids(Path(args.skills_file))
    roots = materialize_ablation_artifacts(
        skill_ids=skill_ids,
        data_root=Path(args.data_root),
        base_artifact_root=Path(args.artifact_root),
        output_root=Path(args.output_root),
        analyzer_config=analyzer_config,
        skip_existing=args.skip_existing,
    )
    print(
        json.dumps(
            {
                "skill_count": len(skill_ids),
                "output_roots": {key: str(value) for key, value in roots.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
