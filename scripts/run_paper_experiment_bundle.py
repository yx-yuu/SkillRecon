#!/usr/bin/env python3
"""Orchestrate the paper500 experiment bundle from completed artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillrecon.core.config import DEFAULT_ENV_CONFIG_PATH, DEFAULT_LLM_CONFIG_PATH
from skillrecon.evaluation.ablation import ABLATION_SYSTEM_IDS
from skillrecon.evaluation.artifacts import (
    ABLATION_ARTIFACT_KIND,
    FULL_ARTIFACT_KIND,
    artifact_coverage,
    load_skill_ids,
)
from skillrecon.evaluation.datasets import (
    compare_gold_labels_to_paper_sample,
    load_gold_label_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DATASET = PROJECT_ROOT / "data" / "evaluation" / "skill_paper500_dataset"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "derived" / "paper500"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "derived" / "paper500_source_links"
DEFAULT_ABLATION_ROOT = PROJECT_ROOT / "derived" / "ablations" / "paper500"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "derived" / "experiments" / "paper500"


@dataclass(frozen=True)
class BundlePaths:
    paper_dataset: Path
    skills_file: Path
    gold_labels: Path
    external_predictions: Path
    artifact_root: Path
    source_root: Path
    ablation_root: Path
    output_dir: Path
    rq2_stats_dir: Path
    appendix_dir: Path
    summary_out: Path


class CommandFailed(RuntimeError):
    """Raised when a bundle subprocess exits non-zero."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__(f"Command failed with returncode={result['returncode']}")
        self.result = result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paper500 experiment aggregation, statistics, and appendix exports"
    )
    parser.add_argument(
        "--paper-dataset",
        default=str(DEFAULT_PAPER_DATASET),
        help="Paper dataset directory containing gold_labels.jsonl and all_skills.txt",
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Completed SkillRecon artifact root",
    )
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="owner/slug source-link root used by ablation generation",
    )
    parser.add_argument(
        "--ablation-root",
        default=str(DEFAULT_ABLATION_ROOT),
        help="Root for RQ2 ablation artifact variants",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Experiment output directory",
    )
    parser.add_argument("--skills-file", help="Override newline-delimited skill id list")
    parser.add_argument(
        "--external-predictions",
        help=(
            "Normalized baseline prediction JSON/JSONL. Defaults to "
            "<paper-dataset>/baseline_predictions/paper_method_baselines.jsonl "
            "when present, otherwise OpenClaw only."
        ),
    )
    parser.add_argument("--summary-out", help="Output JSON summary path")
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Do not build or include ablation artifact roots",
    )
    parser.add_argument(
        "--rebuild-ablations",
        action="store_true",
        help="Rebuild ablation variants instead of passing --skip-existing",
    )
    parser.add_argument("--skip-stats", action="store_true", help="Skip RQ2 statistics")
    parser.add_argument("--skip-appendix", action="store_true", help="Skip appendix exports")
    parser.add_argument(
        "--include-human-audit",
        action="store_true",
        help="Generate human-audit task pack and optionally summarize responses",
    )
    parser.add_argument("--human-audit-responses", help="Reviewer response JSONL")
    parser.add_argument("--human-audit-max-per-subtype", type=int, default=12)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Run downstream steps even when artifact coverage is incomplete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the plan and coverage summary without running subprocesses",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for ablation/statistics commands",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for ablation/statistics commands",
    )
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-max-tokens", type=int)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    paths = resolve_paths(args)

    try:
        dataset_summary = validate_dataset(paths)
        skill_ids = load_skill_ids(paths.skills_file)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    main_coverage = artifact_coverage(
        skill_ids=skill_ids,
        artifact_root=paths.artifact_root,
        expected_status_kind=FULL_ARTIFACT_KIND,
    )
    ablation_coverages = {
        system_id: artifact_coverage(
            skill_ids=skill_ids,
            artifact_root=paths.ablation_root / system_id,
            expected_status_kind=ABLATION_ARTIFACT_KIND,
        )
        for system_id in ABLATION_SYSTEM_IDS
    }
    commands = build_commands(args, paths)
    summary: dict[str, Any] = {
        "paper_dataset": str(paths.paper_dataset),
        "dataset": dataset_summary,
        "main_artifact_coverage": main_coverage,
        "ablation_coverage": ablation_coverages,
        "commands": {key: _command_to_string(command) for key, command in commands.items()},
        "dry_run": args.dry_run,
    }

    main_ready = coverage_complete(main_coverage)
    if not main_ready and not args.allow_incomplete and not args.dry_run:
        summary["status"] = "blocked_incomplete_main_artifacts"
        write_summary(paths.summary_out, summary)
        _print_summary(paths.summary_out, summary)
        raise SystemExit(2)

    if args.dry_run:
        summary["status"] = "dry_run"
        write_summary(paths.summary_out, summary)
        _print_summary(paths.summary_out, summary)
        return

    executed: dict[str, dict[str, Any]] = {}
    for step_name, command in commands.items():
        try:
            executed[step_name] = run_command(command)
        except CommandFailed as exc:
            executed[step_name] = exc.result
            summary["status"] = "failed"
            summary["failed_step"] = step_name
            summary["executed"] = executed
            write_summary(paths.summary_out, summary)
            _print_summary(paths.summary_out, summary)
            raise SystemExit(exc.result["returncode"]) from exc
        else:
            summary["executed"] = executed
            write_summary(paths.summary_out, summary)

    summary["status"] = "ok"
    summary["executed"] = executed
    write_summary(paths.summary_out, summary)
    _print_summary(paths.summary_out, summary)


def resolve_paths(args: argparse.Namespace) -> BundlePaths:
    paper_dataset = Path(args.paper_dataset)
    output_dir = Path(args.output_dir)
    rq2_stats_dir = output_dir / "rq2_stats"
    appendix_dir = output_dir / "appendix"
    return BundlePaths(
        paper_dataset=paper_dataset,
        skills_file=Path(args.skills_file) if args.skills_file else paper_dataset / "all_skills.txt",
        gold_labels=paper_dataset / "gold_labels.jsonl",
        external_predictions=_resolve_external_predictions(paper_dataset, args),
        artifact_root=Path(args.artifact_root),
        source_root=Path(args.source_root),
        ablation_root=Path(args.ablation_root),
        output_dir=output_dir,
        rq2_stats_dir=rq2_stats_dir,
        appendix_dir=appendix_dir,
        summary_out=Path(args.summary_out) if args.summary_out else output_dir / "paper_experiment_bundle_summary.json",
    )


def validate_dataset(paths: BundlePaths) -> dict[str, Any]:
    if not paths.paper_dataset.is_dir():
        raise FileNotFoundError(f"Missing paper dataset directory: {paths.paper_dataset}")
    if not paths.gold_labels.is_file():
        raise FileNotFoundError(f"Missing gold labels: {paths.gold_labels}")
    if not paths.skills_file.is_file():
        raise FileNotFoundError(f"Missing skills file: {paths.skills_file}")
    if not paths.external_predictions.is_file():
        raise FileNotFoundError(f"Missing external baseline predictions: {paths.external_predictions}")

    records = load_gold_label_records(paths.gold_labels)
    comparison = compare_gold_labels_to_paper_sample(records, paths.paper_dataset)
    if comparison["missing"] or comparison["extra"]:
        raise ValueError(
            "Gold labels do not match paper sample indexes: "
            f"missing={len(comparison['missing'])} extra={len(comparison['extra'])}"
        )
    return {
        "gold_labels": str(paths.gold_labels),
        "external_predictions": str(paths.external_predictions),
        "records": len(records),
    }


def build_commands(
    args: argparse.Namespace,
    paths: BundlePaths,
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    system_root_args = [] if args.skip_ablation else _system_artifact_root_args(paths.ablation_root)

    if not args.skip_ablation:
        ablation_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_ablation_artifacts.py"),
            "--skills-file",
            str(paths.skills_file),
            "--data-root",
            str(paths.source_root),
            "--artifact-root",
            str(paths.artifact_root),
            "--output-root",
            str(paths.ablation_root),
            "--env-config",
            str(Path(args.env_config)),
            "--llm-config",
            str(Path(args.llm_config)),
        ]
        _append_llm_overrides_for_ablation(ablation_command, args)
        if not args.rebuild_ablations:
            ablation_command.append("--skip-existing")
        commands["ablation"] = ablation_command

    experiment_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_experiments.py"),
        "--paper-dataset",
        str(paths.paper_dataset),
        "--artifact-root",
        str(paths.artifact_root),
        "--external-predictions",
        str(paths.external_predictions),
        "--output-dir",
        str(paths.output_dir),
        "--llm-config",
        str(Path(args.llm_config)),
        *system_root_args,
    ]
    _append_llm_judge_args(experiment_command, args)
    commands["experiments"] = experiment_command

    if not args.skip_stats:
        stats_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "compute_rq2_statistics.py"),
            "--gold-labels",
            str(paths.gold_labels),
            "--external-predictions",
            str(paths.external_predictions),
            "--artifact-root",
            str(paths.artifact_root),
            "--output-dir",
            str(paths.rq2_stats_dir),
            "--env-config",
            str(Path(args.env_config)),
            "--llm-config",
            str(Path(args.llm_config)),
            *system_root_args,
        ]
        _append_llm_judge_args(stats_command, args)
        commands["rq2_stats"] = stats_command

    if not args.skip_appendix and not args.skip_stats:
        commands["appendix"] = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "export_evaluation_appendix.py"),
            "--experiment-json",
            str(paths.output_dir / "experiment_results.json"),
            "--rq2-stats-json",
            str(paths.rq2_stats_dir / "rq2_stats.json"),
            "--output-dir",
            str(paths.appendix_dir),
        ]

    if args.include_human_audit:
        human_audit_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_human_audit_study.py"),
            "--gold-labels",
            str(paths.gold_labels),
            "--artifact-root",
            str(paths.artifact_root),
            "--output-dir",
            str(paths.output_dir / "human_audit"),
            "--max-per-subtype",
            str(args.human_audit_max_per_subtype),
        ]
        _append_optional(human_audit_command, "--responses", args.human_audit_responses)
        commands["human_audit"] = human_audit_command

    return commands


def _resolve_external_predictions(
    paper_dataset: Path,
    args: argparse.Namespace,
) -> Path:
    if args.external_predictions:
        return Path(args.external_predictions)
    merged = paper_dataset / "baseline_predictions" / "paper_method_baselines.jsonl"
    if merged.is_file():
        return merged
    return paper_dataset / "baseline_predictions" / "openclaw.jsonl"


def coverage_complete(coverage: dict[str, object]) -> bool:
    return (
        int(coverage.get("complete", 0)) == int(coverage.get("expected", 0))
        and not coverage.get("missing_dirs")
        and not coverage.get("incomplete")
    )


def run_command(command: list[str]) -> dict[str, Any]:
    started_at = time.time()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        env=_subprocess_env(),
    )
    ended_at = time.time()
    result = {
        "command": _command_to_string(command),
        "returncode": completed.returncode,
        "duration_seconds": round(ended_at - started_at, 3),
    }
    if completed.returncode != 0:
        result["status"] = "failed"
        raise CommandFailed(result)
    result["status"] = "ok"
    return result


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _system_artifact_root_args(ablation_root: Path) -> list[str]:
    args: list[str] = []
    for system_id in ABLATION_SYSTEM_IDS:
        args.extend(["--system-artifact-root", f"{system_id}={ablation_root / system_id}"])
    return args


def _append_llm_judge_args(command: list[str], args: argparse.Namespace) -> None:
    if args.enable_llm_judge:
        command.append("--enable-llm-judge")
    _append_optional(command, "--llm-base-url", args.llm_base_url)
    _append_optional(command, "--llm-model", args.llm_model)
    _append_optional(command, "--llm-api-key-env", args.llm_api_key_env)
    _append_optional(command, "--llm-temperature", args.llm_temperature)
    _append_optional(command, "--llm-max-tokens", args.llm_max_tokens)


def _append_llm_overrides_for_ablation(command: list[str], args: argparse.Namespace) -> None:
    _append_optional(command, "--base-url", args.llm_base_url)
    _append_optional(command, "--model", args.llm_model)
    _append_optional(command, "--api-key-env", args.llm_api_key_env)
    _append_optional(command, "--temperature", args.llm_temperature)
    _append_optional(command, "--max-tokens", args.llm_max_tokens)


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(PROJECT_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    return env


def _command_to_string(command: list[str]) -> str:
    return " ".join(command)


def _print_summary(path: Path, payload: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "summary_out": str(path),
                "status": payload.get("status"),
                "main_complete": payload["main_artifact_coverage"]["complete"],
                "main_expected": payload["main_artifact_coverage"]["expected"],
                "commands": sorted(payload["commands"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
