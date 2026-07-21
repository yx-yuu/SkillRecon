"""Extended journal-facing evaluation analyses.

The routines in this module are intentionally derived from existing
SkillRecon artifacts. They do not rerun the analyzer; they summarize the
evidence needed for the expanded journal evaluation: closest-method
baselines, reconciliation granularity, structure/language generalization,
and robustness/cost diagnostics.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from skillrecon.core.config import AnalyzerConfig
from skillrecon.core.types import PackageManifest
from skillrecon.evaluation.baselines import (
    build_capability_only_report,
    build_resource_aware_report,
    build_skillrecon_baseline_report,
)
from skillrecon.evaluation.datasets import GoldLabelRecord
from skillrecon.evaluation.metrics import (
    compute_violation_metrics,
    compute_violation_metrics_by_subtype,
    is_rq2_detection_stratum,
)
from skillrecon.evaluation.types import EvaluationReport


def build_granularity_results(
    *,
    artifact_dirs: dict[str, Path],
    analyzer_config: AnalyzerConfig,
    gold_labels: list[GoldLabelRecord],
) -> dict[str, object]:
    """Compare capability-only, resource-aware, and full clause-scope variants."""
    rq2_records = [
        record for record in gold_labels if is_rq2_detection_stratum(record.risk_stratum)
    ]
    systems = {
        "capability_only": {
            skill_id: build_capability_only_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in artifact_dirs.items()
        },
        "resource_aware": {
            skill_id: build_resource_aware_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in artifact_dirs.items()
        },
        "full_clause_scope": {
            skill_id: build_skillrecon_baseline_report(skill_id, artifact_dir)
            for skill_id, artifact_dir in artifact_dirs.items()
        },
    }
    return {
        system_id: {
            "overall": _counts_to_dict(compute_violation_metrics(rq2_records, reports)),
            "by_subtype": {
                subtype: _counts_to_dict(counts)
                for subtype, counts in compute_violation_metrics_by_subtype(
                    rq2_records,
                    reports,
                ).items()
            },
        }
        for system_id, reports in systems.items()
    }


def build_generalization_results(
    *,
    gold_labels: list[GoldLabelRecord],
    reports_by_skill: dict[str, EvaluationReport],
    artifact_dirs: dict[str, Path],
) -> dict[str, object]:
    """Compute SkillRecon detection metrics by language and package structure."""
    sections: dict[str, dict[str, object]] = {
        "language_profile": {},
        "package_structure": {},
        "documentation_closure": {},
        "dataset_bucket": {},
    }
    records_by_skill = {record.skill_id: record for record in gold_labels}
    grouped: dict[str, dict[str, list[GoldLabelRecord]]] = {
        section: {} for section in sections
    }
    for skill_id, artifact_dir in artifact_dirs.items():
        record = records_by_skill.get(skill_id)
        if record is None:
            continue
        manifest = _load_manifest(artifact_dir)
        labels = {
            "language_profile": _language_profile(manifest),
            "package_structure": _package_structure(manifest),
            "documentation_closure": _documentation_closure(manifest),
            "dataset_bucket": record.bucket or _metadata_string(record, "dataset_bucket") or "unknown",
        }
        for section, label in labels.items():
            grouped[section].setdefault(label, []).append(record)

    for section, groups in grouped.items():
        for label, records in sorted(groups.items()):
            if len(records) < 3:
                continue
            metrics = _counts_to_dict(compute_violation_metrics(records, reports_by_skill))
            sections[section][label] = {
                **metrics,
                "n": len(records),
                "positives": sum(1 for record in records if record.gold.label == "violation"),
            }
    return sections


def build_robustness_and_cost_results(
    *,
    rq2: dict[str, object],
    artifact_dirs: dict[str, Path],
) -> dict[str, object]:
    """Summarize ablation impact and static artifact cost proxies."""
    skillrecon = rq2.get("skillrecon", {})
    baseline_by_slice = (
        skillrecon.get("by_slice", {}) if isinstance(skillrecon, dict) else {}
    )
    ablation_drops: dict[str, object] = {}
    for system_id, payload in sorted(rq2.items()):
        if not system_id.startswith("ablation_") or not isinstance(payload, dict):
            continue
        by_slice = payload.get("by_slice", {})
        if not isinstance(by_slice, dict):
            continue
        ablation_drops[system_id] = {
            "high_f1_drop": _slice_drop(baseline_by_slice, by_slice, "high_risk"),
            "medium_f1_drop": _slice_drop(baseline_by_slice, by_slice, "medium_risk"),
            "medium_recovery_drop": _recovery_drop(skillrecon, payload),
        }

    graph_rows: list[dict[str, int]] = []
    runtime_rows: list[dict[str, object]] = []
    witness_modes = Counter()
    for artifact_dir in artifact_dirs.values():
        graph_rows.append(
            {
                "g_d_nodes": _graph_count(artifact_dir / "g_d.json", "nodes"),
                "g_d_edges": _graph_count(artifact_dir / "g_d.json", "edges"),
                "g_c_nodes": _graph_count(artifact_dir / "g_c.json", "nodes"),
                "g_c_edges": _graph_count(artifact_dir / "g_c.json", "edges"),
                "g_x_nodes": _graph_count(artifact_dir / "g_x.json", "nodes"),
                "g_x_edges": _graph_count(artifact_dir / "g_x.json", "edges"),
            }
        )
        runtime_metrics = _load_json_object(artifact_dir / "runtime_metrics.json")
        if runtime_metrics:
            runtime_rows.append(runtime_metrics)
        for witness in _load_json_list(artifact_dir / "witnesses.json"):
            if isinstance(witness, dict) and witness.get("is_exact"):
                witness_modes["exact"] += 1
            elif isinstance(witness, dict):
                witness_modes["greedy"] += 1

    return {
        "ablation_drops": ablation_drops,
        "graph_size": {
            key: _distribution([row[key] for row in graph_rows])
            for key in (
                "g_d_nodes",
                "g_d_edges",
                "g_c_nodes",
                "g_c_edges",
                "g_x_nodes",
                "g_x_edges",
            )
        },
        "runtime": _runtime_summary(runtime_rows),
        "witness_modes": dict(witness_modes),
    }


def _counts_to_dict(counts) -> dict[str, float | int]:
    return {
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
    }


def _load_manifest(artifact_dir: Path) -> PackageManifest | None:
    path = artifact_dir / "package_manifest.json"
    if not path.is_file():
        return None
    return PackageManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _language_profile(manifest: PackageManifest | None) -> str:
    if manifest is None:
        return "unknown"
    code_kinds = {
        entry.kind.value
        for entry in manifest.files
        if entry.kind.value in {"python", "javascript", "typescript", "bash"}
    }
    if not code_kinds:
        return "docs_or_config_only"
    if len(code_kinds) > 1:
        return "mixed_language"
    kind = next(iter(code_kinds))
    if kind == "python":
        return "python_only"
    if kind in {"javascript", "typescript"}:
        return "js_ts"
    if kind == "bash":
        return "bash_heavy"
    return "other"


def _package_structure(manifest: PackageManifest | None) -> str:
    if manifest is None:
        return "unknown"
    code_files = [
        entry for entry in manifest.files
        if entry.kind.value in {"python", "javascript", "typescript", "bash"}
    ]
    return "single_file" if len(code_files) <= 1 else "multi_file"


def _documentation_closure(manifest: PackageManifest | None) -> str:
    if manifest is None:
        return "unknown"
    max_depth = max((doc.depth for doc in manifest.documents), default=0)
    return "closure_depth_1" if max_depth <= 0 else "closure_depth_2_plus"


def _metadata_string(record: GoldLabelRecord, key: str) -> str | None:
    value = record.metadata.get(key)
    return value if isinstance(value, str) and value else None


def _slice_drop(
    baseline_by_slice: object,
    variant_by_slice: object,
    slice_name: str,
) -> float | None:
    if not isinstance(baseline_by_slice, dict) or not isinstance(variant_by_slice, dict):
        return None
    baseline = baseline_by_slice.get(slice_name, {})
    variant = variant_by_slice.get(slice_name, {})
    if not isinstance(baseline, dict) or not isinstance(variant, dict):
        return None
    if baseline.get("f1") is None or variant.get("f1") is None:
        return None
    return float(baseline["f1"]) - float(variant["f1"])


def _recovery_drop(baseline: object, variant: object) -> float | None:
    if not isinstance(baseline, dict) or not isinstance(variant, dict):
        return None
    base_recovery = baseline.get("medium_disputed", {})
    variant_recovery = variant.get("medium_disputed", {})
    if not isinstance(base_recovery, dict) or not isinstance(variant_recovery, dict):
        return None
    if base_recovery.get("recovery") is None or variant_recovery.get("recovery") is None:
        return None
    return float(base_recovery["recovery"]) - float(variant_recovery["recovery"])


def _graph_count(path: Path, key: str) -> int:
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return 0
    value = payload.get(key, [])
    return len(value) if isinstance(value, list) else 0


def _load_json_list(path: Path) -> list[object]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _runtime_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    totals = [
        float(row["total_seconds"])
        for row in rows
        if isinstance(row.get("total_seconds"), int | float)
    ]
    stages = sorted(
        {
            stage
            for row in rows
            if isinstance(row.get("stage_seconds"), dict)
            for stage in row["stage_seconds"]
        }
    )
    return {
        "packages": len(rows),
        "total_seconds": _float_distribution(totals),
        "stage_seconds": {
            stage: _float_distribution(
                [
                    float(row["stage_seconds"][stage])
                    for row in rows
                    if isinstance(row.get("stage_seconds"), dict)
                    and isinstance(row["stage_seconds"].get(stage), int | float)
                ]
            )
            for stage in stages
        },
        "stage_share": _stage_share(rows, stages),
    }


def _stage_share(rows: list[dict[str, object]], stages: list[str]) -> dict[str, float | None]:
    totals = {
        stage: sum(
            float(row["stage_seconds"][stage])
            for row in rows
            if isinstance(row.get("stage_seconds"), dict)
            and isinstance(row["stage_seconds"].get(stage), int | float)
        )
        for stage in stages
    }
    denominator = sum(totals.values())
    return {
        stage: (None if denominator == 0 else value / denominator)
        for stage, value in totals.items()
    }


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": _percentile(ordered, 0.5),
        "p90": _percentile(ordered, 0.9),
        "max": ordered[-1],
    }


def _float_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": _float_percentile(ordered, 0.5),
        "p90": _float_percentile(ordered, 0.9),
        "max": ordered[-1],
    }


def _percentile(ordered: list[int], q: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def _float_percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)
