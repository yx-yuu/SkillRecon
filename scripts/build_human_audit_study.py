#!/usr/bin/env python3
"""Prepare or analyze the witness human-audit study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillrecon.evaluation.datasets import load_gold_label_records
from skillrecon.evaluation.human_audit import (
    analyze_human_audit_responses,
    build_human_audit_task_pack,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build human-audit task packs and summarize reviewer responses"
    )
    parser.add_argument("--gold-labels", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--responses",
        help="Optional reviewer response JSONL. If omitted, only task packs are generated.",
    )
    parser.add_argument("--max-per-subtype", type=int, default=12)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    task_summary = build_human_audit_task_pack(
        gold_labels=load_gold_label_records(Path(args.gold_labels)),
        artifact_root=Path(args.artifact_root),
        output_dir=output_dir,
        max_per_subtype=args.max_per_subtype,
    )
    result: dict[str, object] = {"task_summary": task_summary}
    if args.responses:
        result["response_summary"] = analyze_human_audit_responses(
            response_path=Path(args.responses),
            output_dir=output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
