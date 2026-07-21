#!/usr/bin/env python3
"""Compatibility CLI for inspecting generated gold-label datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    compare_gold_labels_to_paper_sample,
    load_gold_label_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect AI-generated gold-label datasets. Dataset construction writes "
            "gold_labels.jsonl directly; this CLI does not create labels from a "
            "separate sheet."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a gold-label JSONL")
    validate_parser.add_argument("--gold-labels", required=True)
    validate_parser.add_argument(
        "--paper-dataset",
        help="Optional paper sample directory to check skill_id coverage",
    )

    summarize_parser = subparsers.add_parser("summarize", help="Summarize gold labels")
    summarize_parser.add_argument("--gold-labels", required=True)

    export_parser = subparsers.add_parser(
        "export-provenance",
        help="Export read-only provenance metadata for inspection",
    )
    export_parser.add_argument("--gold-labels", required=True)
    export_parser.add_argument(
        "--csv-out",
        help="Output CSV path (defaults to <gold-labels stem>.provenance.csv)",
    )

    args = parser.parse_args()
    records = load_gold_label_records(Path(args.gold_labels))
    if args.command == "validate":
        result = _validate(records, Path(args.paper_dataset) if args.paper_dataset else None)
    elif args.command == "summarize":
        result = _summarize(records)
    else:
        csv_path = _export_provenance(
            records,
            Path(args.csv_out) if args.csv_out else Path(args.gold_labels).with_suffix(".provenance.csv"),
        )
        result = {"gold_labels": args.gold_labels, "csv_out": str(csv_path), "records": len(records)}

    print(json.dumps(result, ensure_ascii=False, indent=2))


def _validate(
    records: list[GoldLabelRecord],
    paper_dataset: Path | None,
) -> dict[str, object]:
    duplicate_skills = sorted(
        skill_id
        for skill_id, count in Counter(record.skill_id for record in records).items()
        if count > 1
    )
    invalid_subtypes = [
        record.skill_id
        for record in records
        if record.gold.label == "violation" and not record.gold.violation_subtype
    ]
    result: dict[str, object] = {
        "records": len(records),
        "duplicate_skills": duplicate_skills,
        "invalid_violation_subtypes": invalid_subtypes,
        "ok": not duplicate_skills and not invalid_subtypes,
    }
    if paper_dataset is not None:
        comparison = compare_gold_labels_to_paper_sample(records, paper_dataset)
        result["paper_sample_comparison"] = comparison
        result["ok"] = (
            bool(result["ok"])
            and not comparison["missing"]
            and not comparison["extra"]
        )
    return result


def _summarize(records: list[GoldLabelRecord]) -> dict[str, object]:
    return {
        "records": len(records),
        "gold_label_counts": dict(Counter(record.gold.label for record in records)),
        "violation_subtype_counts": dict(
            Counter(
                record.gold.violation_subtype
                for record in records
                if record.gold.violation_subtype is not None
            )
        ),
        "risk_stratum_counts": dict(Counter(record.risk_stratum for record in records)),
        "bucket_counts": dict(Counter(record.bucket for record in records)),
    }


def _export_provenance(records: list[GoldLabelRecord], csv_out: Path) -> Path:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "skill_id",
                "gold_label",
                "violation_subtype",
                "risk_stratum",
                "bucket",
                "generator",
                "model",
                "prompt_version",
                "source_hash",
                "source_files",
                "rationale",
            ],
        )
        writer.writeheader()
        for record in records:
            generation = record.metadata.get("gold_label_generation", {})
            if not isinstance(generation, dict):
                generation = {}
            source_files = generation.get("source_files", [])
            writer.writerow(
                {
                    "skill_id": record.skill_id,
                    "gold_label": record.gold.label,
                    "violation_subtype": record.gold.violation_subtype or "",
                    "risk_stratum": record.risk_stratum or "",
                    "bucket": record.bucket or "",
                    "generator": generation.get("generator", ""),
                    "model": generation.get("model", ""),
                    "prompt_version": generation.get("prompt_version", ""),
                    "source_hash": generation.get("source_hash", ""),
                    "source_files": ";".join(source_files if isinstance(source_files, list) else []),
                    "rationale": record.gold.rationale,
                }
            )
    return csv_out


if __name__ == "__main__":
    main()
