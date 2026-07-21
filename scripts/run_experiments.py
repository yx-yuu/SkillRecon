#!/usr/bin/env python3
"""Run corpus-level evaluation experiments for the paper RQs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillrecon.core.config import DEFAULT_LLM_CONFIG_PATH, resolve_llm_config
from skillrecon.evaluation.datasets import (
    compare_gold_labels_to_paper_sample,
    load_gold_label_records,
)
from skillrecon.evaluation.runner import ExperimentInputs, run_all_experiments


def _parse_system_artifact_roots(items: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in items:
        system_id, sep, path = item.partition("=")
        if not sep or not system_id or not path:
            raise ValueError(
                f"Invalid --system-artifact-root value: {item!r}; expected system_id=path"
            )
        parsed[system_id] = Path(path)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run corpus-level SkillRecon evaluation experiments"
    )
    parser.add_argument(
        "--artifact-root",
        required=True,
        help="Root directory containing per-skill artifacts (<root>/<owner>/<slug>)",
    )
    parser.add_argument(
        "--paper-dataset",
        help=(
            "Paper High/Medium/Low sample dataset directory "
            "(for example data/evaluation/skill_paper500_dataset)"
        ),
    )
    parser.add_argument(
        "--gold-labels",
        help="Path to gold-label dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--rq1-gold-labels",
        help="Optional path to a separate RQ1 gold-label dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--rq1-artifact-root",
        help="Optional artifact root for the separate RQ1 benchmark",
    )
    parser.add_argument(
        "--seeded-benchmark",
        help="Path to seeded benchmark dataset (.json or .jsonl)",
    )
    parser.add_argument(
        "--external-predictions",
        help="Path to external baseline predictions (.json or .jsonl)",
    )
    parser.add_argument(
        "--enable-llm-judge",
        action="store_true",
        help="Enable the B2 LLM-as-judge baseline",
    )
    parser.add_argument("--llm-base-url", help="LLM judge base URL override")
    parser.add_argument("--llm-model", help="LLM judge model override")
    parser.add_argument("--llm-api-key-env", help="LLM judge API key env or literal")
    parser.add_argument("--llm-temperature", type=float, help="LLM judge temperature")
    parser.add_argument("--llm-max-tokens", type=int, help="LLM judge max_tokens")
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for the LLM-as-judge baseline",
    )
    parser.add_argument(
        "--system-artifact-root",
        action="append",
        default=[],
        help="Additional artifact roots in the form system_id=path (used for ablations)",
    )
    parser.add_argument(
        "--output-dir",
        default="derived/experiments",
        help="Directory for experiment outputs",
    )
    args = parser.parse_args()

    gold_label_path, external_prediction_path = _resolve_dataset_inputs(args, parser)
    llm_judge_config = None
    if args.enable_llm_judge:
        try:
            llm_judge_config = resolve_llm_config(
                llm_config_path=Path(args.llm_config),
                base_url=args.llm_base_url,
                model=args.llm_model,
                api_key_env=args.llm_api_key_env,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    bundle = run_all_experiments(
        ExperimentInputs(
            gold_label_path=gold_label_path,
            seeded_path=Path(args.seeded_benchmark) if args.seeded_benchmark else None,
            artifact_root=Path(args.artifact_root),
            rq1_gold_label_path=(
                Path(args.rq1_gold_labels) if args.rq1_gold_labels else None
            ),
            rq1_artifact_root=(
                Path(args.rq1_artifact_root) if args.rq1_artifact_root else None
            ),
            external_prediction_path=external_prediction_path,
            llm_judge_enabled=args.enable_llm_judge,
            llm_judge_config=llm_judge_config,
            system_artifact_roots=_parse_system_artifact_roots(args.system_artifact_root),
        ),
        output_dir=Path(args.output_dir),
    )

    print(
        json.dumps(
            {
                "output_dir": str(Path(args.output_dir)),
                "paper_dataset": args.paper_dataset,
                "gold_labels": str(gold_label_path) if gold_label_path else None,
                "external_predictions": (
                    str(external_prediction_path) if external_prediction_path else None
                ),
                "rq1_keys": sorted(bundle["rq1"].keys()),
                "rq2_systems": sorted(bundle["rq2"].keys()),
                "rq3_keys": sorted(bundle["rq3"].keys()),
                "rq4_keys": sorted(bundle["rq4"].keys()),
                "appendix_systems": sorted(
                    bundle.get("appendix", {}).get("systems", {}).keys()
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _resolve_dataset_inputs(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[Path | None, Path | None]:
    gold_label_path = Path(args.gold_labels) if args.gold_labels else None
    external_prediction_path = (
        Path(args.external_predictions) if args.external_predictions else None
    )
    if not args.paper_dataset:
        return gold_label_path, external_prediction_path

    paper_dataset = Path(args.paper_dataset)
    if not paper_dataset.is_dir():
        parser.error(f"--paper-dataset does not exist or is not a directory: {paper_dataset}")

    if gold_label_path is None:
        gold_label_path = paper_dataset / "gold_labels.jsonl"
    if not gold_label_path.is_file():
        parser.error(
            f"Missing AI-generated gold labels at {gold_label_path}. "
            "Run scripts/prepare_paper_dataset.py to create gold_labels.jsonl."
        )

    comparison = compare_gold_labels_to_paper_sample(
        load_gold_label_records(gold_label_path),
        paper_dataset,
    )
    if comparison["missing"] or comparison["extra"]:
        parser.error(
            "Gold labels do not match the paper sample index: "
            f"missing={len(comparison['missing'])} {comparison['missing'][:5]}, "
            f"extra={len(comparison['extra'])} {comparison['extra'][:5]}"
        )

    if external_prediction_path is None:
        candidate = paper_dataset / "baseline_predictions" / "openclaw.jsonl"
        if candidate.is_file():
            external_prediction_path = candidate

    return gold_label_path, external_prediction_path


if __name__ == "__main__":
    main()
