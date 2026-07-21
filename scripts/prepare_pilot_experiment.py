#!/usr/bin/env python3
"""Prepare a small AI-generated gold-label pilot package."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.evaluation.datasets import (
    build_openclaw_predictions,
    load_gold_label_records,
    load_flagged_skill_records,
    select_pilot_sample,
    write_jsonl_models,
)
from skillrecon.evaluation.gold_builder import (
    GoldBuildConfig,
    SourceCollectionError,
    build_ai_gold_label_records,
)
from skillrecon.loader.path_resolver import parse_windows_drive_map
from skillrecon.llm.cache import CachedLLMClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a pilot evaluation package with AI-generated gold labels"
    )
    parser.add_argument(
        "--index-path",
        default="data/dataset_index/flagged_security_scan_skills.jsonl",
        help="Path to flagged corpus index (.jsonl or .json)",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/skill_dataset",
        help="Root directory containing local skill packages",
    )
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/pilot20",
        help="Directory for the generated pilot package",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of sampled skills to include",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260326,
        help="Deterministic random seed used for stratified sampling",
    )
    parser.add_argument(
        "--include-unresolved-openclaw",
        action="store_true",
        help="Allow records without a resolved OpenClaw status into the pilot sample",
    )
    parser.add_argument(
        "--openclaw-unresolved-label",
        choices=["violation", "exposure-only", "benign"],
        help="Optional fallback label for unresolved OpenClaw statuses",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for the AI gold-label generator",
    )
    parser.add_argument(
        "--drive-map",
        action="append",
        default=[],
        metavar="DRIVE=ROOT",
        help=(
            "Override Windows drive mapping for source paths, e.g. "
            "E=/mnt/e or E=/data/clawhub-drive"
        ),
    )
    parser.add_argument("--base-url", help="LLM API base URL override")
    parser.add_argument("--model", help="LLM model name override")
    parser.add_argument("--api-key-env", help="API key env var or literal key override")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing records in <output-dir>/gold_labels.jsonl",
    )
    args = parser.parse_args()

    index_path = Path(args.index_path)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    try:
        drive_map = parse_windows_drive_map(args.drive_map)
    except ValueError as exc:
        parser.error(str(exc))

    records = load_flagged_skill_records(index_path)
    sample = select_pilot_sample(
        records,
        sample_size=args.sample_size,
        seed=args.seed,
        resolved_openclaw_only=not args.include_unresolved_openclaw,
    )
    gold_label_path = output_dir / "gold_labels.jsonl"
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
    client = CachedLLMClient.from_config(llm_config, "pilot_ai_gold_v1")
    existing_gold_labels = (
        load_gold_label_records(gold_label_path)
        if args.resume and gold_label_path.is_file()
        else []
    )
    try:
        gold_labels = build_ai_gold_label_records(
            sample,
            dataset_root=dataset_root,
            client=client,
            build_config=GoldBuildConfig(
                prompt_version="pilot_ai_gold_v1",
                max_tokens=args.max_tokens,
            ),
            existing_records=existing_gold_labels,
            windows_drive_map=drive_map,
        )
    except SourceCollectionError as exc:
        parser.error(str(exc))
    openclaw_predictions = build_openclaw_predictions(
        sample,
        unresolved_label=args.openclaw_unresolved_label,
    )

    write_jsonl_models(output_dir / "sample_index.jsonl", sample)
    write_jsonl_models(gold_label_path, gold_labels)
    write_jsonl_models(
        output_dir / "baseline_predictions" / "openclaw.jsonl",
        openclaw_predictions,
    )
    _write_skill_list(output_dir / "skills.txt", sample)
    _write_manifest(
        output_dir / "manifest.json",
        index_path=index_path,
        dataset_root=dataset_root,
        sample=sample,
        openclaw_predictions=openclaw_predictions,
        gold_labels=gold_labels,
        sample_size=args.sample_size,
        seed=args.seed,
        resolved_openclaw_only=not args.include_unresolved_openclaw,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "sample_size": len(sample),
                "risk_tiers": Counter(record.risk_tier for record in sample),
                "dataset_buckets": Counter(record.dataset_bucket for record in sample),
                "openclaw_statuses": Counter(record.openclaw_status for record in sample),
                "gold_labels": str(gold_label_path),
                "gold_label_count": len(gold_labels),
                "predicted_by_openclaw": len(openclaw_predictions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_skill_list(path: Path, sample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{record.skill_id}\n" for record in sample),
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    *,
    index_path: Path,
    dataset_root: Path,
    sample,
    openclaw_predictions,
    gold_labels,
    sample_size: int,
    seed: int,
    resolved_openclaw_only: bool,
) -> None:
    manifest = {
        "source_index": index_path.name,
        "dataset_root": str(dataset_root),
        "sample_size_requested": sample_size,
        "sample_size_realized": len(sample),
        "seed": seed,
        "resolved_openclaw_only": resolved_openclaw_only,
        "risk_tier_counts": dict(Counter(record.risk_tier for record in sample)),
        "dataset_bucket_counts": dict(Counter(record.dataset_bucket for record in sample)),
        "openclaw_status_counts": dict(Counter(record.openclaw_status for record in sample)),
        "virustotal_status_counts": dict(Counter(record.virus_total_status for record in sample)),
        "skills": [record.skill_id for record in sample],
        "artifacts": {
            "sample_index": "sample_index.jsonl",
            "gold_labels": "gold_labels.jsonl",
            "openclaw_predictions": "baseline_predictions/openclaw.jsonl",
            "skill_list": "skills.txt",
        },
        "next_steps": [
            "Run SkillRecon over the listed skills and place artifacts under a shared artifact root.",
            "Invoke scripts/run_experiments.py with the generated gold labels and OpenClaw predictions.",
        ],
        "gold_label_count": len(gold_labels),
        "openclaw_prediction_coverage": len(openclaw_predictions),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
