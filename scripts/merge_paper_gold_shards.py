#!/usr/bin/env python3
"""Merge sharded paper gold-label outputs into the formal dataset file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    build_openclaw_predictions,
    load_gold_label_records,
    load_paper_sample_records,
    write_jsonl_models,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge sharded paper gold labels"
    )
    parser.add_argument(
        "--paper-dataset",
        default="data/evaluation/skill_paper500_dataset",
        help="Paper High/Medium/Low sample dataset directory",
    )
    parser.add_argument(
        "--shard-root",
        default="temp/paper500_gold_shards_v2/full",
        help="Directory containing shard_*/gold_labels.jsonl outputs",
    )
    parser.add_argument(
        "--gold-labels-out",
        default=None,
        help="Output JSONL path, defaults to <paper-dataset>/gold_labels.jsonl",
    )
    args = parser.parse_args()

    paper_dataset = Path(args.paper_dataset)
    shard_root = Path(args.shard_root)
    gold_labels_out = (
        Path(args.gold_labels_out)
        if args.gold_labels_out
        else paper_dataset / "gold_labels.jsonl"
    )

    records_by_slice = load_paper_sample_records(paper_dataset)
    sample = [
        record
        for slice_name in ("high", "medium", "low")
        for record in records_by_slice[slice_name]
    ]
    expected_ids = [record.skill_id for record in sample]
    by_skill: dict[str, GoldLabelRecord] = {}
    duplicates: list[str] = []
    source_missing: list[str] = []

    for shard_path in sorted(shard_root.glob("shard_*/gold_labels.jsonl")):
        for record in load_gold_label_records(shard_path):
            if record.skill_id in by_skill:
                duplicates.append(record.skill_id)
                continue
            by_skill[record.skill_id] = record
            source_files = (
                record.metadata.get("gold_label_generation", {})
                if isinstance(record.metadata, dict)
                else {}
            )
            if isinstance(source_files, dict):
                files = source_files.get("source_files")
            else:
                files = None
            if not files:
                source_missing.append(record.skill_id)

    missing = [skill_id for skill_id in expected_ids if skill_id not in by_skill]
    extra = sorted(set(by_skill) - set(expected_ids))
    if duplicates or missing or extra or source_missing:
        report = {
            "duplicates": sorted(set(duplicates)),
            "missing": missing,
            "extra": extra,
            "source_missing": source_missing,
            "records_found": len(by_skill),
            "records_expected": len(expected_ids),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    ordered = [by_skill[skill_id] for skill_id in expected_ids]
    write_jsonl_models(gold_labels_out, ordered)
    openclaw_predictions = build_openclaw_predictions(sample)
    openclaw_path = gold_labels_out.parent / "baseline_predictions" / "openclaw.jsonl"
    write_jsonl_models(openclaw_path, openclaw_predictions)
    _write_skill_list(gold_labels_out.parent / "all_skills.txt", expected_ids)
    summary_path = gold_labels_out.parent / "paper_dataset_summary.json"
    _write_summary(
        summary_path,
        paper_dataset=paper_dataset,
        records_by_slice=records_by_slice,
        gold_labels_out=gold_labels_out,
        openclaw_path=openclaw_path,
        records=ordered,
    )

    print(
        json.dumps(
            {
                "paper_dataset": str(paper_dataset),
                "shard_root": str(shard_root),
                "gold_labels": str(gold_labels_out),
                "records": len(ordered),
                "label_counts": dict(Counter(record.gold.label for record in ordered)),
                "openclaw_predictions": str(openclaw_path),
                "summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_skill_list(path: Path, skill_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{skill_id}\n" for skill_id in skill_ids), encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    paper_dataset: Path,
    records_by_slice,
    gold_labels_out: Path,
    openclaw_path: Path,
    records: list[GoldLabelRecord],
) -> None:
    summary = {
        "paper_dataset": str(paper_dataset),
        "sample_size": len(records),
        "slice_counts": {
            slice_name: len(records_by_slice[slice_name])
            for slice_name in ("high", "medium", "low")
        },
        "gold_label_count": len(records),
        "gold_label_counts": dict(Counter(record.gold.label for record in records)),
        "risk_tier_counts": dict(Counter(record.risk_stratum for record in records)),
        "bucket_counts": dict(Counter(record.bucket for record in records)),
        "artifacts": {
            "gold_labels": gold_labels_out.name,
            "openclaw_predictions": openclaw_path.relative_to(gold_labels_out.parent).as_posix(),
            "skill_list": "all_skills.txt",
        },
        "notes": [
            "Dataset construction is automatically generated.",
            "The action supervision label is stored only in each record's gold field.",
            "Offline human calibration edits the gold field directly after dataset freeze.",
        ],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
