#!/usr/bin/env python3
"""Sample VT/OC-benign skills for benign-probe evaluation."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


_BUCKET_FILES = {
    "mixed": "mixed.jsonl",
    "pure_bash": "pure_bash.jsonl",
    "pure_py": "pure_py.jsonl",
    "pure_js": "pure_js.jsonl",
    "pure_ts": "pure_ts.jsonl",
}

_DEFAULT_QUOTAS = {
    "mixed": 4,
    "pure_bash": 2,
    "pure_py": 2,
    "pure_js": 1,
    "pure_ts": 1,
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


def _collect_processed_skill_ids(derived_root: Path) -> set[str]:
    processed: set[str] = set()
    for findings_path in derived_root.glob("**/findings.json"):
        parts = findings_path.parent.relative_to(derived_root).parts
        if len(parts) < 2:
            continue
        processed.add("/".join(parts[-2:]))
    return processed


def _scaled_quotas(size: int) -> dict[str, int]:
    total_weight = sum(_DEFAULT_QUOTAS.values())
    quotas = {
        bucket: (size * weight) // total_weight
        for bucket, weight in _DEFAULT_QUOTAS.items()
    }
    remainder = size - sum(quotas.values())
    if remainder <= 0:
        return quotas

    ranked_buckets = sorted(
        _DEFAULT_QUOTAS,
        key=lambda bucket: ((size * _DEFAULT_QUOTAS[bucket]) % total_weight, _DEFAULT_QUOTAS[bucket]),
        reverse=True,
    )
    for bucket in ranked_buckets[:remainder]:
        quotas[bucket] += 1
    return quotas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample VT/OC-benign skills for a benign-probe batch"
    )
    parser.add_argument(
        "--index-root",
        default="data/dataset_index",
        help="Directory containing dataset index jsonl files",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/benign10_probe/sample_index.jsonl",
        help="Where to write the sampled skill list",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260326,
        help="Deterministic random seed",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10,
        help="Target sample size",
    )
    parser.add_argument(
        "--exclude-processed",
        action="store_true",
        help="Exclude skills that already have findings.json somewhere under derived/",
    )
    parser.add_argument(
        "--derived-root",
        default="derived",
        help="Root directory scanned when --exclude-processed is enabled",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    index_root = Path(args.index_root)
    no_risk_records = _load_jsonl(index_root / "flagged_security_scan_skills_no_risk.jsonl")
    no_risk_by_skill = {_skill_id(record): record for record in no_risk_records}
    processed_ids = (
        _collect_processed_skill_ids(Path(args.derived_root))
        if args.exclude_processed
        else set()
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bucket, filename in _BUCKET_FILES.items():
        for record in _load_jsonl(index_root / filename):
            skill_id = _skill_id(record)
            if skill_id in no_risk_by_skill and skill_id not in processed_ids:
                grouped[bucket].append(no_risk_by_skill[skill_id])

    for records in grouped.values():
        rng.shuffle(records)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    quotas = _scaled_quotas(args.size)
    for bucket in _BUCKET_FILES:
        quota = quotas.get(bucket, 0)
        for record in grouped.get(bucket, []):
            if quota <= 0:
                break
            skill_id = _skill_id(record)
            if skill_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(skill_id)
            quota -= 1

    if len(selected) < args.size:
        pool = []
        for bucket in _BUCKET_FILES:
            pool.extend(grouped.get(bucket, []))
        rng.shuffle(pool)
        for record in pool:
            if len(selected) >= args.size:
                break
            skill_id = _skill_id(record)
            if skill_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(skill_id)

    selected = selected[: args.size]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "output": str(output_path),
        "size": len(selected),
        "exclude_processed": args.exclude_processed,
        "derived_root": str(Path(args.derived_root)) if args.exclude_processed else None,
        "skills": [_skill_id(record) for record in selected],
        "buckets": {
            bucket: sum(1 for record in selected if record.get("dataset_bucket") == bucket)
            for bucket in _BUCKET_FILES
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
