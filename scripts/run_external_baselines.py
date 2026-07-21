#!/usr/bin/env python3
"""Run or import external paper-method baseline scanner outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from skillrecon.evaluation.datasets import (
    BaselinePredictionRecord,
    flatten_paper_sample_records,
    write_jsonl_models,
)
from skillrecon.evaluation.external_scanners import (
    build_external_scanner_command,
    external_scanner_payload_to_prediction,
    external_scanner_payload_to_report,
    get_external_scanner_spec,
    list_external_scanner_specs,
    load_external_scanner_payload,
)


@dataclass(frozen=True)
class ScanTarget:
    skill_id: str
    skill_path: Path | None = None
    raw_json: Path | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run external paper-method baselines and normalize their outputs "
            "into SkillRecon baseline predictions."
        )
    )
    parser.add_argument(
        "--scanner",
        required=True,
        choices=sorted(spec.system_id for spec in list_external_scanner_specs()),
        help="External baseline system id.",
    )
    parser.add_argument("--skill-id", help="Single skill id for --skill-path.")
    parser.add_argument("--skill-path", help="Single skill source directory or SKILL.md.")
    parser.add_argument(
        "--raw-json",
        action="append",
        default=[],
        help=(
            "Import an existing scanner JSON/SARIF payload. Format: "
            "skill_id=/path/to/raw.json. Can be repeated."
        ),
    )
    parser.add_argument(
        "--reviewer-cases",
        help="Reviewer case config, for example experiments/configs/reviewer_cases_v1.json.",
    )
    parser.add_argument(
        "--reviewer-source-root",
        default="data/skill_dataset",
        help="Root containing reviewer case source_relpath directories.",
    )
    parser.add_argument(
        "--paper-dataset",
        help="Paper sample dataset root containing high/medium/low sample_index.jsonl.",
    )
    parser.add_argument(
        "--artifact-root",
        help=(
            "Optional SkillRecon artifact root. When the original skill source "
            "path is unavailable, use <artifact-root>/<owner>/<slug>/staged_source."
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data/skill_dataset",
        help="Fallback local skill dataset root for resolving paper sample records.",
    )
    parser.add_argument(
        "--drive-map",
        action="append",
        default=[],
        help="Windows drive mapping for sample indexes, for example E=/mnt/e.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of scan targets, useful for smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="derived/external_baselines",
        help="Output directory for raw scanner files, normalized reports, and predictions.",
    )
    parser.add_argument(
        "--predictions-out",
        help="Optional path for the normalized baseline prediction JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned scanner commands without executing them.",
    )
    parser.add_argument(
        "--allow-missing-scanner",
        action="store_true",
        help="Skip executable-backed targets when the scanner CLI is not installed.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep scanning after a target-level scanner failure.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = get_external_scanner_spec(args.scanner)
    targets = _collect_targets(args)
    if args.limit is not None:
        targets = targets[: args.limit]
    if not targets:
        raise SystemExit("No external baseline targets were selected.")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw" / spec.system_id
    report_dir = output_dir / "reports" / spec.system_id
    prediction_path = (
        Path(args.predictions_out)
        if args.predictions_out
        else output_dir / "baseline_predictions" / f"{spec.system_id}.jsonl"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    executable = shutil.which(spec.executable)
    planned: list[dict[str, object]] = []
    predictions: list[BaselinePredictionRecord] = []
    skipped: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []

    for target in targets:
        raw_output = target.raw_json or raw_dir / f"{_safe_name(target.skill_id)}.json"
        scan_skill_path = (
            None
            if target.raw_json is not None or target.skill_path is None
            else _prepare_scanner_skill_path(
                spec,
                target=target,
                adapted_root=output_dir / "adapted" / spec.system_id,
            )
        )
        command = (
            None
            if target.raw_json is not None or scan_skill_path is None
            else build_external_scanner_command(
                spec,
                skill_path=scan_skill_path,
                output_path=raw_output,
            )
        )
        planned.append(
            {
                "skill_id": target.skill_id,
                "skill_path": str(target.skill_path) if target.skill_path else None,
                "scanner_skill_path": str(scan_skill_path) if scan_skill_path else None,
                "raw_json": str(raw_output),
                "command": command,
            }
        )
        if args.dry_run:
            continue

        if target.raw_json is None:
            if command is None:
                failures.append(
                    {
                        "skill_id": target.skill_id,
                        "error": "missing skill_path for executable-backed scan",
                    }
                )
                if not args.continue_on_error:
                    break
                continue
            if executable is None:
                message = f"{spec.executable!r} is not installed; {spec.install_hint}"
                if args.allow_missing_scanner:
                    skipped.append({"skill_id": target.skill_id, "reason": message})
                    continue
                raise SystemExit(message)
            try:
                _run_scanner_command(
                    command,
                    output_path=raw_output,
                    output_to_file=spec.output_to_file,
                    success_exit_codes=spec.success_exit_codes,
                )
            except RuntimeError as exc:
                failures.append({"skill_id": target.skill_id, "error": str(exc)})
                if not args.continue_on_error:
                    break
                continue

        try:
            payload = load_external_scanner_payload(raw_output)
            raw_ref = os.path.relpath(raw_output, Path.cwd())
            prediction = external_scanner_payload_to_prediction(
                skill_id=target.skill_id,
                system_id=spec.system_id,
                payload=payload,
                raw_output_ref=raw_ref,
            )
            report = external_scanner_payload_to_report(
                skill_id=target.skill_id,
                system_id=spec.system_id,
                payload=payload,
                raw_output_ref=raw_ref,
            )
            predictions.append(prediction)
            report_path = report_dir / f"{_safe_name(target.skill_id)}.json"
            report_path.write_text(
                report.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            failures.append({"skill_id": target.skill_id, "error": str(exc)})
            if not args.continue_on_error:
                break

    if not args.dry_run and predictions:
        write_jsonl_models(prediction_path, predictions)

    summary = {
        "scanner": spec.system_id,
        "display_name": spec.display_name,
        "repo_url": spec.repo_url,
        "method_reference": spec.method_reference,
        "planned": len(planned),
        "predictions": len(predictions),
        "skipped": skipped,
        "failures": failures,
        "predictions_out": str(prediction_path),
        "output_dir": str(output_dir),
        "commands": planned if args.dry_run else planned[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures and not args.continue_on_error:
        raise SystemExit(1)


def _collect_targets(args: argparse.Namespace) -> list[ScanTarget]:
    targets: list[ScanTarget] = []
    targets.extend(_parse_raw_json_targets(args.raw_json))
    if args.skill_path:
        if not args.skill_id:
            raise SystemExit("--skill-id is required with --skill-path")
        targets.append(ScanTarget(skill_id=args.skill_id, skill_path=Path(args.skill_path)))
    if args.reviewer_cases:
        targets.extend(
            _reviewer_case_targets(
                Path(args.reviewer_cases),
                source_root=Path(args.reviewer_source_root),
            )
        )
    if args.paper_dataset:
        targets.extend(
            _paper_dataset_targets(
                Path(args.paper_dataset),
                data_root=Path(args.data_root),
                drive_map=_parse_drive_map(args.drive_map),
                artifact_root=Path(args.artifact_root) if args.artifact_root else None,
            )
        )
    return _dedupe_targets(targets)


def _parse_raw_json_targets(items: list[str]) -> list[ScanTarget]:
    targets: list[ScanTarget] = []
    for item in items:
        skill_id, sep, path = item.partition("=")
        if not sep or not skill_id or not path:
            raise SystemExit(
                f"Invalid --raw-json value {item!r}; expected skill_id=/path/to/raw.json"
            )
        targets.append(ScanTarget(skill_id=skill_id, raw_json=Path(path)))
    return targets


def _reviewer_case_targets(path: Path, *, source_root: Path) -> list[ScanTarget]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets: list[ScanTarget] = []
    for item in payload:
        skill_id = item.get("skill_id")
        source_relpath = item.get("source_relpath")
        if not isinstance(skill_id, str) or not isinstance(source_relpath, str):
            continue
        targets.append(
            ScanTarget(
                skill_id=skill_id,
                skill_path=source_root / source_relpath,
            )
        )
    return targets


def _paper_dataset_targets(
    dataset_root: Path,
    *,
    data_root: Path,
    drive_map: dict[str, str],
    artifact_root: Path | None = None,
) -> list[ScanTarget]:
    targets: list[ScanTarget] = []
    for record in flatten_paper_sample_records(dataset_root):
        skill_path = record.resolve_skill_path(
            data_root,
            windows_drive_map=drive_map,
        )
        staged_source = (
            artifact_root / record.owner / record.slug / "staged_source"
            if artifact_root is not None
            else None
        )
        if staged_source is not None and staged_source.exists() and not skill_path.exists():
            skill_path = staged_source
        targets.append(ScanTarget(skill_id=record.skill_id, skill_path=skill_path))
    return targets


def _parse_drive_map(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        drive, sep, root = item.partition("=")
        if not sep or not drive or not root:
            raise SystemExit(f"Invalid --drive-map value {item!r}; expected E=/mnt/e")
        parsed[drive.upper().rstrip(":")] = root
    return parsed


def _dedupe_targets(targets: list[ScanTarget]) -> list[ScanTarget]:
    seen: set[str] = set()
    deduped: list[ScanTarget] = []
    for target in targets:
        if target.skill_id in seen:
            continue
        seen.add(target.skill_id)
        deduped.append(target)
    return deduped


def _run_scanner_command(
    command: list[str],
    *,
    output_path: Path,
    output_to_file: bool,
    success_exit_codes: tuple[int, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in success_exit_codes:
        raise RuntimeError(
            f"scanner exited with {result.returncode}: {result.stderr.strip()}"
        )
    if not output_to_file:
        stdout = result.stdout.strip()
        if not stdout:
            raise RuntimeError("scanner produced no JSON on stdout")
        output_path.write_text(stdout + "\n", encoding="utf-8")
    elif not output_path.is_file():
        raise RuntimeError(f"scanner did not write expected output: {output_path}")


def _prepare_scanner_skill_path(
    spec,
    *,
    target: ScanTarget,
    adapted_root: Path,
) -> Path | None:
    skill_path = target.skill_path
    if skill_path is None or spec.command_kind != "skillfortify":
        return skill_path

    if _skillfortify_native_layout_exists(skill_path):
        return skill_path

    skill_markdown = skill_path / "SKILL.md"
    if not skill_markdown.is_file():
        return skill_path

    adapter_dir = adapted_root / _safe_name(target.skill_id)
    skills_dir = adapter_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for old_skill in skills_dir.glob("*.md"):
        old_skill.unlink()
    skill_slug = target.skill_id.replace("\\", "/").rstrip("/").split("/")[-1]
    adapted_skill = skills_dir / f"{skill_slug or skill_path.name}.md"
    adapted_skill.write_text(skill_markdown.read_text(encoding="utf-8"), encoding="utf-8")
    return adapter_dir


def _skillfortify_native_layout_exists(skill_path: Path) -> bool:
    if any((skill_path / ".claude" / "skills").glob("*.md")):
        return True
    for claw_dir in (skill_path / ".claw", skill_path / ".openclaw"):
        if any(claw_dir.glob("*.yaml")) or any(claw_dir.glob("*.yml")):
            return True
    return False


def _safe_name(skill_id: str) -> str:
    return skill_id.replace("/", "__").replace("\\", "__")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:  # pragma: no cover - shell pipeline boundary
        sys.exit(1)
