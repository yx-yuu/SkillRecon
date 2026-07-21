#!/usr/bin/env python3
"""Run the full pipeline for a sampled benign-probe batch with failure tolerance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _skill_id(record: dict[str, object]) -> str:
    owner = record.get("owner")
    slug = record.get("slug")
    if isinstance(owner, str) and isinstance(slug, str):
        return f"{owner}/{slug}"
    skill = record.get("skill")
    if isinstance(skill, dict):
        skill_owner = skill.get("owner")
        skill_slug = skill.get("slug")
        if isinstance(skill_owner, str) and isinstance(skill_slug, str):
            return f"{skill_owner}/{skill_slug}"
    raise ValueError(f"Cannot derive skill id from record: {record}")


def _load_existing_summary(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benign-probe batch")
    parser.add_argument(
        "--sample-index",
        required=True,
        help="JSONL file containing sampled skill records",
    )
    parser.add_argument(
        "--output-dir",
        default="derived/benign10_probe",
        help="Artifact output directory",
    )
    parser.add_argument(
        "--env-config",
        default="experiments/configs/env_config.json",
        help="Path to env config for run_full.py",
    )
    parser.add_argument(
        "--summary-out",
        default="derived/benign10_probe/run_summary.json",
        help="Where to write the batch summary JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only run the first N skills from sample-index",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip skills whose output already contains findings.json or that already appear in summary-out",
    )
    args = parser.parse_args()

    records = _load_jsonl(Path(args.sample_index))
    if args.limit is not None:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    summary_out = Path(args.summary_out)
    summary = _load_existing_summary(summary_out)
    summary_by_skill = {
        str(item["skill_id"]): item
        for item in summary
        if isinstance(item, dict) and isinstance(item.get("skill_id"), str)
    }

    for record in records:
        skill_id = _skill_id(record)
        artifact_dir = output_dir / skill_id
        findings_path = artifact_dir / "findings.json"
        if args.skip_existing and (findings_path.exists() or skill_id in summary_by_skill):
            print(f"SKIP {skill_id}")
            continue
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_full.py"),
            "--skill",
            skill_id,
            "--data-root",
            str(PROJECT_ROOT / "data" / "skill_dataset"),
            "--output-dir",
            str(output_dir),
            "--env-config",
            str(PROJECT_ROOT / args.env_config),
        ]
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        summary.append(
            {
                "skill_id": skill_id,
                "returncode": completed.returncode,
                "succeeded": completed.returncode == 0,
                "artifact_dir": str(artifact_dir),
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        status = "OK" if completed.returncode == 0 else f"FAIL({completed.returncode})"
        print(f"{status} {skill_id}")

    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    main()
