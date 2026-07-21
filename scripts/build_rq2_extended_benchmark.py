#!/usr/bin/env python3
"""Build an extended RQ2 gold-label benchmark from explicit item specs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    BaselinePredictionRecord,
    GoldLabel,
    load_gold_label_records,
    load_baseline_prediction_records,
    load_flagged_skill_records,
    sanitize_gold_metadata,
    write_jsonl_models,
)


class _ExtendedBenchmarkItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    artifact_dir: str
    gold: GoldLabel
    notes: str = ""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an extended RQ2 gold-label benchmark"
    )
    parser.add_argument(
        "--items-json",
        required=True,
        help="JSON file listing benchmark items with skill_id, artifact_dir, and gold",
    )
    parser.add_argument(
        "--base-gold-labels",
        default="data/evaluation/pilot20/gold_labels.jsonl",
        help="Base gold-label file",
    )
    parser.add_argument(
        "--pilot-openclaw",
        default="data/evaluation/pilot20/baseline_predictions/openclaw.jsonl",
        help="Base OpenClaw prediction file",
    )
    parser.add_argument(
        "--flagged-index",
        default="data/dataset_index/flagged_security_scan_skills.jsonl",
        help="Flagged corpus index used for security-scan metadata",
    )
    parser.add_argument(
        "--output-data-dir",
        default="data/evaluation/rq2_extended",
        help="Directory for the extended benchmark data files",
    )
    parser.add_argument(
        "--output-artifact-root",
        default="derived/rq2_extended",
        help="Unified artifact root for the extended benchmark",
    )
    args = parser.parse_args()

    base_gold_labels = load_gold_label_records(Path(args.base_gold_labels))
    pilot_openclaw = load_baseline_prediction_records(Path(args.pilot_openclaw))
    flagged_records = load_flagged_skill_records(Path(args.flagged_index))
    flagged_by_skill = {record.skill_id: record for record in flagged_records}
    extended_items = _load_items(Path(args.items_json))

    output_data_dir = Path(args.output_data_dir)
    output_artifact_root = Path(args.output_artifact_root)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_root.mkdir(parents=True, exist_ok=True)

    combined_gold_labels = [
        record.model_copy(update={"metadata": sanitize_gold_metadata(record.metadata)})
        for record in base_gold_labels
    ]
    combined_predictions = list(pilot_openclaw)

    for record in base_gold_labels:
        _copy_artifact_tree(
            source=Path("derived/pilot20") / record.skill_id,
            target=output_artifact_root / record.skill_id,
        )

    manifest_items: list[dict[str, object]] = []
    for item in extended_items:
        skill_id = item.skill_id
        flagged = flagged_by_skill.get(skill_id)
        combined_gold_labels.append(
            GoldLabelRecord(
                skill_id=skill_id,
                gold=item.gold,
                risk_stratum=flagged.risk_tier if flagged is not None else "extended_rq2",
                bucket=flagged.dataset_bucket if flagged is not None else "extended_rq2",
                metadata=sanitize_gold_metadata(
                    {
                        "assessment": "ai_generated_extended_rq2_gold",
                        "scan_summary": item.notes,
                    }
                ),
            )
        )
        if flagged is not None and flagged.openclaw_status is not None:
            combined_predictions.append(
                BaselinePredictionRecord(
                    skill_id=skill_id,
                    system_id="baseline_openclaw",
                    main_label=_map_openclaw_status(flagged.openclaw_status),
                    rationale="Mapped from flagged corpus security scan status.",
                    metadata={
                        "openclaw_status": flagged.openclaw_status,
                        "virus_total_status": flagged.virus_total_status,
                        "risk_tier": flagged.risk_tier,
                    },
                )
            )
        _copy_artifact_tree(
            source=Path(item.artifact_dir),
            target=output_artifact_root / skill_id,
        )
        manifest_items.append(
            {
                "skill_id": skill_id,
                "gold": item.gold.model_dump(exclude_none=False),
            }
        )

    write_jsonl_models(output_data_dir / "gold_labels.jsonl", combined_gold_labels)
    write_jsonl_models(
        output_data_dir / "baseline_predictions" / "openclaw.jsonl",
        combined_predictions,
    )
    (output_data_dir / "skills.txt").write_text(
        "".join(f"{record.skill_id}\n" for record in combined_gold_labels),
        encoding="utf-8",
    )
    (output_data_dir / "manifest.json").write_text(
        json.dumps(
            {
                "base_gold_labels": str(Path(args.base_gold_labels)),
                "base_openclaw": str(Path(args.pilot_openclaw)),
                "items": manifest_items,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "gold_labels": str(output_data_dir / "gold_labels.jsonl"),
                "openclaw_predictions": str(
                    output_data_dir / "baseline_predictions" / "openclaw.jsonl"
                ),
                "artifact_root": str(output_artifact_root),
                "records": len(combined_gold_labels),
                "extended_items": len(extended_items),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _copy_artifact_tree(*, source: Path, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _load_items(path: Path) -> list[_ExtendedBenchmarkItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [_ExtendedBenchmarkItem.model_validate(item) for item in payload]


def _map_openclaw_status(status: str) -> str:
    if status in {"Suspicious", "Malicious"}:
        return "violation"
    return "benign"


if __name__ == "__main__":
    main()
