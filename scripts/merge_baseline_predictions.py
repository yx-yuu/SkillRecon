#!/usr/bin/env python3
"""Merge multiple external baseline prediction files for one experiment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillrecon.evaluation.datasets import (
    BaselinePredictionRecord,
    load_baseline_prediction_records,
    write_jsonl_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge BaselinePredictionRecord JSON/JSONL files."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input prediction file. Can be repeated.",
    )
    parser.add_argument("--output", required=True, help="Merged prediction JSONL.")
    parser.add_argument(
        "--replace-duplicates",
        action="store_true",
        help="Keep the last record when duplicate (system_id, skill_id) pairs appear.",
    )
    args = parser.parse_args()

    merged: dict[tuple[str, str], BaselinePredictionRecord] = {}
    duplicates: list[dict[str, str]] = []
    for input_path in [Path(item) for item in args.input]:
        for record in load_baseline_prediction_records(input_path):
            key = (record.system_id, record.skill_id)
            if key in merged:
                duplicates.append({"system_id": key[0], "skill_id": key[1]})
                if not args.replace_duplicates:
                    raise SystemExit(
                        "Duplicate baseline prediction for "
                        f"system_id={key[0]!r}, skill_id={key[1]!r}. "
                        "Use --replace-duplicates to keep the last one."
                    )
            merged[key] = record

    output = Path(args.output)
    records = [merged[key] for key in sorted(merged)]
    write_jsonl_models(output, records)
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "duplicates": len(duplicates),
                "systems": sorted({record.system_id for record in records}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
