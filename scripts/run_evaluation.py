#!/usr/bin/env python3
"""Build a unified evaluation report from existing SkillRecon artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillrecon.evaluation import build_skillrecon_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a unified evaluation report from existing SkillRecon artifacts"
    )
    parser.add_argument("--skill", required=True, help="Skill identifier")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing findings.json and related artifacts",
    )
    parser.add_argument(
        "--json-out",
        help="Output path for report.json (defaults to <artifact-dir>/report.json)",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_path = Path(args.json_out) if args.json_out else artifact_dir / "report.json"

    report = build_skillrecon_report(args.skill, artifact_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Wrote evaluation report for {args.skill} to {output_path} "
        f"({report.summary.violation_count} violations, "
        f"{report.summary.exposure_count} exposures, "
        f"{report.summary.contract_quality_count} contract-quality alerts)"
    )


if __name__ == "__main__":
    main()
