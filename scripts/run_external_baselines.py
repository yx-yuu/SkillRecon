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
    write_jsonl_models,
)
from skillrecon.evaluation.external_scanners import (
    build_external_scanner_command,
    external_scanner_payload_to_prediction,
    external_scanner_payload_to_report,
    get_external_scanner_spec,
    list_external_scanner_specs,
    load_external_scanner_payload,
    load_malskills_payload,
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
    parser.add_argument("--scanner-config", type=Path, help="Native MalSkills full-profile TOML, including the chosen GLM/DeepSeek/Qwen provider")
    parser.add_argument("--scanner-python", type=Path, help="Python with native MalSkills installed; defaults to the current interpreter for its full profile")
    parser.add_argument("--scanner-arg", action="append", default=[], help="Append an argument to the recorded native command; repeat as needed")
    parser.add_argument(
        "--raw-json",
        action="append",
        default=[],
        help=(
            "Import an existing scanner JSON/SARIF payload. Format: "
            "skill_id=/path/to/raw.json. Can be repeated."
        ),
    )
    parser.add_argument("--skills-file", type=Path, help="One skill identity per line")
    parser.add_argument("--data-root", type=Path, default=Path("data/packages"))
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of scan targets, useful for a selected subset.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/external-baselines",
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
        raw_output = target.raw_json or (
            raw_dir / _safe_name(target.skill_id) / "verdict.json"
            if spec.command_kind in {"malskills_static", "malskills_full"}
            else raw_dir / f"{_safe_name(target.skill_id)}.json"
        )
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
                scanner_config=args.scanner_config,
            )
        )
        if command is not None:
            command.extend(args.scanner_arg)
            if spec.command_kind == "malskills_full":
                command = [str(args.scanner_python or sys.executable),
                           str(Path(__file__).with_name("run_malskills_native.py")),
                           "--trace",str(raw_output.parent / "native_execution.jsonl"),*command[1:]]
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
            if executable is None and spec.command_kind != "malskills_full":
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
            payload = (
                load_malskills_payload(raw_output, require_full=spec.command_kind == "malskills_full") if spec.command_kind in {"malskills_static", "malskills_full"}
                else load_external_scanner_payload(raw_output)
            )
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
        except Exception as exc: 
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
        "commands": planned,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
    if args.skills_file:
        targets.extend(ScanTarget(skill_id=line.strip(), skill_path=args.data_root/line.strip())
                       for line in args.skills_file.read_text().splitlines() if line.strip())
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
    output_path.with_suffix(".stdout.log").write_text(result.stdout,encoding="utf-8")
    output_path.with_suffix(".stderr.log").write_text(result.stderr,encoding="utf-8")
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
