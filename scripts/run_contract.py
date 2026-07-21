#!/usr/bin/env python3
"""CLI entry point for running Contract Observation on a single skill."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from skillrecon.contract.pipeline import ContractObservationPipeline
from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    AnalyzerConfig,
    VotePolicy,
    load_env_config,
    resolve_llm_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Contract Observation on a skill")
    parser.add_argument("--skill", required=True, help="Skill directory name")
    parser.add_argument(
        "--data-root",
        default="data/skill_dataset",
        help="Root directory of skill packages",
    )
    parser.add_argument(
        "--output-dir",
        default="derived",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json used for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json used for LLM defaults",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="ICCM sample count; omit to use env_config vote_policy.n_samples",
    )
    parser.add_argument("--base-url", help="LLM API base URL")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--api-key-env", help="API key env var or literal key")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

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

    base_vote_policy = env_config.vote_policy if env_config is not None else VotePolicy()
    vote_policy = (
        base_vote_policy
        if args.n_samples is None
        else base_vote_policy.model_copy(update={"n_samples": args.n_samples})
    )
    analyzer_config = AnalyzerConfig(
        llm=llm_config,
        vote_policy=vote_policy,
    )

    pipeline = ContractObservationPipeline(
        analyzer_config=analyzer_config,
        skill_data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )

    try:
        result = pipeline.run(args.skill)
        print(f"\nContract table saved. {len(result.clauses)} clauses extracted.")
        for clause in result.clauses:
            print(f"  [{clause.operator.value:10s}] {clause.capability}"
                  f"{' -> ' + clause.target if clause.target else ''}"
                  f" (agreement={clause.vote_agreement:.0%})")
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
