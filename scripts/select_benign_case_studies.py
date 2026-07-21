#!/usr/bin/env python3
"""Select benign case-study candidates by intersecting dataset indexes and findings."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


_VIOLATION_MAP = {
    "unsupported_behavior": "V1",
    "scope_violation": "V2",
    "unjustified_composition": "V3",
}

_BUCKET_FILES = {
    "mixed": "mixed.jsonl",
    "pure_bash": "pure_bash.jsonl",
    "pure_py": "pure_py.jsonl",
    "pure_js": "pure_js.jsonl",
    "pure_ts": "pure_ts.jsonl",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _skill_id(record: dict[str, Any]) -> str:
    owner = record.get("owner")
    slug = record.get("slug")
    if owner and slug:
        return f"{owner}/{slug}"
    skill = record.get("skill", {})
    skill_owner = skill.get("owner")
    skill_slug = skill.get("slug")
    if skill_owner and skill_slug:
        return f"{skill_owner}/{skill_slug}"
    raise ValueError(f"Cannot derive skill id from record: {record}")


def _collect_artifact_dirs(derived_root: Path) -> dict[str, list[Path]]:
    artifact_dirs: dict[str, list[Path]] = defaultdict(list)
    for findings_path in derived_root.glob("**/findings.json"):
        parent = findings_path.parent
        parts = parent.relative_to(derived_root).parts
        if len(parts) < 2:
            continue
        skill_id = "/".join(parts[-2:])
        artifact_dirs[skill_id].append(parent)
    return artifact_dirs


def _collect_finding_types(artifact_dir: Path) -> set[str]:
    findings_path = artifact_dir / "findings.json"
    findings = json.loads(findings_path.read_text(encoding="utf-8"))
    return {str(item.get("finding_type", "")) for item in findings}


def _record_summary(record: dict[str, Any]) -> dict[str, Any]:
    skill = record.get("skill", {})
    security = skill.get("security_scan", {})
    return {
        "skill_id": _skill_id(record),
        "display_name": skill.get("display_name"),
        "bucket": record.get("dataset_bucket"),
        "script_types": record.get("script_types", []),
        "virus_total_status": security.get("virus_total_status"),
        "openclaw_status": security.get("openclaw_status"),
        "scan_confidence": security.get("confidence"),
        "scan_summary": security.get("summary"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select VT/OC benign case-study candidates from dataset indexes"
    )
    parser.add_argument(
        "--index-root",
        default="data/dataset_index",
        help="Directory containing dataset index jsonl files",
    )
    parser.add_argument(
        "--derived-root",
        default="derived",
        help="Directory containing analysis artifacts with findings.json",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["mixed", "pure_bash", "pure_py", "pure_js", "pure_ts"],
        choices=sorted(_BUCKET_FILES),
        help="Dataset buckets to include",
    )
    parser.add_argument(
        "--output",
        default="derived/case_study_candidates/benign_case_candidates.json",
        help="Where to write the candidate summary JSON",
    )
    args = parser.parse_args()

    index_root = Path(args.index_root)
    derived_root = Path(args.derived_root)

    no_risk_records = _load_jsonl(index_root / "flagged_security_scan_skills_no_risk.jsonl")
    no_risk_by_skill = {_skill_id(record): record for record in no_risk_records}

    bucket_membership: dict[str, set[str]] = {}
    for bucket in args.buckets:
        bucket_records = _load_jsonl(index_root / _BUCKET_FILES[bucket])
        bucket_membership[bucket] = {_skill_id(record) for record in bucket_records}

    allowed_skills = set().union(*bucket_membership.values())
    candidate_skills = sorted(set(no_risk_by_skill) & allowed_skills)
    artifact_dirs = _collect_artifact_dirs(derived_root)

    grouped: dict[str, list[dict[str, Any]]] = {"V1": [], "V2": [], "V3": []}
    all_candidates: list[dict[str, Any]] = []
    for skill_id in candidate_skills:
        record = no_risk_by_skill[skill_id]
        artifacts = artifact_dirs.get(skill_id, [])
        if not artifacts:
            continue
        finding_types: set[str] = set()
        artifact_paths: list[str] = []
        for artifact_dir in artifacts:
            artifact_paths.append(str(artifact_dir))
            finding_types.update(_collect_finding_types(artifact_dir))
        matched_labels = sorted({_VIOLATION_MAP[f] for f in finding_types if f in _VIOLATION_MAP})
        if not matched_labels:
            continue
        summary = _record_summary(record)
        summary["finding_types"] = sorted(finding_types)
        summary["violation_labels"] = matched_labels
        summary["artifact_dirs"] = artifact_paths
        all_candidates.append(summary)
        for label in matched_labels:
            grouped[label].append(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "buckets": args.buckets,
        "candidate_count": len(all_candidates),
        "grouped": grouped,
        "all_candidates": all_candidates,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
