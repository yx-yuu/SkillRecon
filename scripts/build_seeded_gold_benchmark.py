#!/usr/bin/env python3
"""Build a seeded benchmark from explicit gold-label item specs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from skillrecon.evaluation.datasets import (
    BaselinePredictionRecord,
    GoldLabel,
    SeededBenchmarkRecord,
    load_flagged_skill_records,
    sanitize_gold_metadata,
    write_jsonl_models,
)


class _SeededItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    artifact_dir: str
    gold: GoldLabel
    injection_family: str
    base_skill_id: str


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a seeded benchmark from explicit gold-label item specs"
    )
    parser.add_argument(
        "--items-json",
        required=True,
        help="JSON file listing seeded benchmark items",
    )
    parser.add_argument(
        "--flagged-index",
        default="data/dataset_index/flagged_security_scan_skills.jsonl",
    )
    parser.add_argument(
        "--output-data-dir",
        default="data/evaluation/seeded_gold",
    )
    parser.add_argument(
        "--output-artifact-root",
        default="derived/seeded_gold",
    )
    args = parser.parse_args()

    flagged = {record.skill_id: record for record in load_flagged_skill_records(Path(args.flagged_index))}
    seeded_items = _load_items(Path(args.items_json))
    output_data_dir = Path(args.output_data_dir)
    output_artifact_root = Path(args.output_artifact_root)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_root.mkdir(parents=True, exist_ok=True)

    benchmark_records: list[SeededBenchmarkRecord] = []
    openclaw_predictions: list[BaselinePredictionRecord] = []
    manifest_items: list[dict[str, object]] = []

    for item in seeded_items:
        skill_id = item.skill_id
        source = Path(item.artifact_dir)
        target = output_artifact_root / skill_id
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)

        benchmark_records.append(
            SeededBenchmarkRecord(
                skill_id=skill_id,
                base_skill_id=item.base_skill_id,
                injection_family=item.injection_family,
                gold=item.gold,
                expected_sites=[],
                metadata=sanitize_gold_metadata(
                    {
                        "assessment": "seeded_gold_benchmark",
                    }
                ),
            )
        )
        flagged_record = flagged.get(skill_id)
        if flagged_record is not None and flagged_record.openclaw_status is not None:
            openclaw_predictions.append(
                BaselinePredictionRecord(
                    skill_id=skill_id,
                    system_id="baseline_openclaw",
                    main_label=_map_openclaw_status(flagged_record.openclaw_status),
                    rationale="Mapped from flagged corpus security scan status.",
                    metadata={
                        "openclaw_status": flagged_record.openclaw_status,
                        "virus_total_status": flagged_record.virus_total_status,
                        "risk_tier": flagged_record.risk_tier,
                    },
                )
            )
        manifest_items.append(
            {
                "skill_id": skill_id,
                "gold": item.gold.model_dump(exclude_none=False),
            }
        )

    write_jsonl_models(output_data_dir / "seeded_benchmark.jsonl", benchmark_records)
    write_jsonl_models(
        output_data_dir / "baseline_predictions" / "openclaw.jsonl",
        openclaw_predictions,
    )
    (output_data_dir / "manifest.json").write_text(
        json.dumps({"items": manifest_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seeded_benchmark": str(output_data_dir / "seeded_benchmark.jsonl"),
                "openclaw_predictions": str(
                    output_data_dir / "baseline_predictions" / "openclaw.jsonl"
                ),
                "artifact_root": str(output_artifact_root),
                "records": len(benchmark_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_items(path: Path) -> list[_SeededItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [_SeededItem.model_validate(item) for item in payload]


def _map_openclaw_status(status: str) -> str:
    if status in {"Suspicious", "Malicious"}:
        return "violation"
    return "benign"


if __name__ == "__main__":
    main()
