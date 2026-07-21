#!/usr/bin/env python3
"""Compute RQ2 bootstrap CIs and pairwise significance tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    AnalyzerConfig,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.evaluation.baselines import (
    build_capability_lattice_report,
    build_doc_code_consistency_report,
    build_external_prediction_report,
    build_instruction_constraint_report,
    build_llm_judge_report,
    build_rule_based_scanner_report,
    build_spec_containment_report,
    build_skillrecon_baseline_report,
    index_baseline_predictions,
)
from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    load_gold_label_records,
    load_baseline_prediction_records,
)
from skillrecon.evaluation.metrics import (
    _normalize_violation_subtype,
    is_rq2_detection_stratum,
)
from skillrecon.evaluation.stats import bootstrap_resample_ci, holm_correction, mcnemar_exact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CIs and McNemar/Holm statistics for RQ2"
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--gold-labels", required=True)
    parser.add_argument("--external-predictions")
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api-key-env")
    parser.add_argument("--llm-temperature", type=float)
    parser.add_argument("--llm-max-tokens", type=int)
    parser.add_argument(
        "--system-artifact-root",
        action="append",
        default=[],
        help="Additional roots in system_id=path form",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for LLM-backed baselines",
    )
    parser.add_argument(
        "--output-dir",
        default="derived/experiments/rq2_stats",
    )
    args = parser.parse_args()

    env_config_path = Path(args.env_config)
    env_config = load_env_config(env_config_path) if env_config_path.is_file() else None
    try:
        analyzer_llm = resolve_llm_config(
            llm_config_path=Path(args.llm_config),
            env_config=env_config,
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key_env=args.llm_api_key_env,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    analyzer_config = AnalyzerConfig(llm=analyzer_llm)

    gold_labels = [
        record
        for record in load_gold_label_records(Path(args.gold_labels))
        if is_rq2_detection_stratum(record.risk_stratum)
    ]
    skill_ids = sorted(record.skill_id for record in gold_labels)
    artifact_root = Path(args.artifact_root)
    external_predictions = index_baseline_predictions(
        load_baseline_prediction_records(Path(args.external_predictions))
        if args.external_predictions
        else []
    )

    systems = {
        "skillrecon": {
            skill_id: build_skillrecon_baseline_report(skill_id, artifact_root / skill_id)
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        },
        "baseline_rule_scanner": {
            skill_id: build_rule_based_scanner_report(
                skill_id=skill_id,
                artifact_dir=artifact_root / skill_id,
                analyzer_config=analyzer_config,
            )
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        },
        "baseline_capability_lattice": {
            skill_id: build_capability_lattice_report(
                skill_id=skill_id,
                artifact_dir=artifact_root / skill_id,
                analyzer_config=analyzer_config,
            )
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        },
        "baseline_doc_code_consistency": {
            skill_id: build_doc_code_consistency_report(
                skill_id=skill_id,
                artifact_dir=artifact_root / skill_id,
                analyzer_config=analyzer_config,
            )
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        },
        "baseline_spec_containment": {
            skill_id: build_spec_containment_report(
                skill_id=skill_id,
                artifact_dir=artifact_root / skill_id,
                analyzer_config=analyzer_config,
            )
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        },
        "baseline_instruction_constraints": {
            skill_id: build_instruction_constraint_report(
                skill_id=skill_id,
                artifact_dir=artifact_root / skill_id,
                analyzer_config=analyzer_config,
            )
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        },
    }
    if args.enable_llm_judge:
        systems["baseline_llm_judge"] = {
            skill_id: build_llm_judge_report(
                skill_id=skill_id,
                artifact_dir=artifact_root / skill_id,
                analyzer_config=analyzer_config,
            )
            for skill_id in skill_ids
            if _artifact_evaluation_ready(artifact_root / skill_id)
        }

    for system_id, root in _parse_system_artifact_roots(args.system_artifact_root).items():
        root_path = Path(root)
        systems[system_id] = {
            skill_id: build_skillrecon_baseline_report(skill_id, root_path / skill_id)
            for skill_id in skill_ids
            if _artifact_evaluation_ready(root_path / skill_id)
        }

    external_system_ids = sorted({system_id for system_id, _ in external_predictions})
    for system_id in external_system_ids:
        systems[system_id] = {
            skill_id: build_external_prediction_report(
                skill_id=skill_id,
                system_id=system_id,
                prediction=prediction,
            )
            for (pred_system_id, skill_id), prediction in external_predictions.items()
            if pred_system_id == system_id
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_payload = _compute_stats(gold_labels, systems)
    (output_dir / "rq2_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_overall_csv(output_dir / "rq2_overall_ci.csv", stats_payload)
    _write_significance_csv(output_dir / "rq2_significance.csv", stats_payload)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "systems": sorted(stats_payload["systems"].keys()),
                "comparisons": sorted(stats_payload["pairwise_mcnemar"].keys()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _compute_stats(
    gold_labels: list[GoldLabelRecord],
    systems: dict[str, dict[str, object]],
) -> dict[str, object]:
    normalized_subtypes = sorted(
        {
            _normalize_violation_subtype(record.gold.violation_subtype)
            for record in gold_labels
            if record.gold.violation_subtype is not None
        }
    )

    per_system: dict[str, dict[str, object]] = {}
    correctness_vectors: dict[str, list[bool]] = {}
    for system_id, reports_by_skill in systems.items():
        rows = [
            _row_for_record(record, reports_by_skill.get(record.skill_id))
            for record in gold_labels
        ]
        correctness_vectors[system_id] = [bool(row["correct_binary"]) for row in rows]
        per_system[system_id] = {
            "prediction_coverage": _prediction_coverage(
                gold_labels,
                reports_by_skill,
            ),
            "overall": _bootstrap_prf(rows, "gold_violation", "pred_violation"),
            "by_slice": {
                stratum: _bootstrap_prf(
                    [
                        row
                        for row in rows
                        if row.get("risk_stratum") == stratum
                    ],
                    "gold_violation",
                    "pred_violation",
                )
                for stratum in ("high_risk", "medium_risk")
            },
            "by_subtype": {
                subtype: _bootstrap_prf(
                    rows,
                    f"gold::{subtype}",
                    f"pred::{subtype}",
                )
                for subtype in normalized_subtypes
            },
        }

    comparisons = {
        f"skillrecon_vs_{system_id}": mcnemar_exact(
            correctness_vectors["skillrecon"],
            correctness_vectors[system_id],
        )
        for system_id in sorted(correctness_vectors)
        if system_id != "skillrecon"
    }
    holm = holm_correction(comparisons)
    pairwise = {
        name: {
            "p_value": p_value,
            "significant_after_holm": holm[name],
        }
        for name, p_value in comparisons.items()
    }

    return {
        "scope": {
            "records": len(gold_labels),
            "risk_strata": _risk_stratum_counts(gold_labels),
            "gold_labels": _gold_label_counts(gold_labels),
            "positive_label": "violation",
            "negative_labels": ["exposure-only", "benign"],
        },
        "systems": per_system,
        "pairwise_mcnemar": pairwise,
    }


def _prediction_coverage(
    records: list[GoldLabelRecord],
    reports_by_skill: dict[str, object],
) -> dict[str, object]:
    expected = {record.skill_id for record in records}
    actual = {skill_id for skill_id in reports_by_skill if skill_id in expected}
    return {
        "expected": len(expected),
        "reported": len(actual),
        "missing": len(expected - actual),
    }


def _row_for_record(record: GoldLabelRecord, report) -> dict[str, object]:
    pred_label = "benign" if report is None else report.overall_label
    row = {
        "gold_violation": record.gold.label == "violation",
        "pred_violation": pred_label == "violation",
        "risk_stratum": _normalize_rq2_stratum(record.risk_stratum),
    }
    row["correct_binary"] = row["gold_violation"] == row["pred_violation"]
    normalized_gold = _normalize_violation_subtype(record.gold.violation_subtype)
    pred_subtypes = set()
    if report is not None:
        pred_subtypes = {
            _normalize_violation_subtype(finding.subtype)
            for finding in report.violation_findings
        }
    if normalized_gold is not None:
        row[f"gold::{normalized_gold}"] = True
    for subtype in pred_subtypes:
        if subtype is not None:
            row[f"pred::{subtype}"] = True
    return row


def _normalize_rq2_stratum(value: str | None) -> str:
    normalized = "" if value is None else value.strip().lower().replace("-", "_")
    if normalized in {"high", "high_risk"}:
        return "high_risk"
    if normalized in {"medium", "medium_risk"}:
        return "medium_risk"
    return normalized


def _risk_stratum_counts(records: list[GoldLabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        stratum = _normalize_rq2_stratum(record.risk_stratum)
        counts[stratum] = counts.get(stratum, 0) + 1
    return counts


def _gold_label_counts(records: list[GoldLabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.gold.label] = counts.get(record.gold.label, 0) + 1
    return counts


def _bootstrap_prf(rows: list[dict[str, object]], gold_key: str, pred_key: str) -> dict[str, object]:
    pairs = [
        (bool(row.get(gold_key, False)), bool(row.get(pred_key, False)))
        for row in rows
    ]
    precision = _prf_stat(pairs, "precision")
    recall = _prf_stat(pairs, "recall")
    f1 = _prf_stat(pairs, "f1")
    return {
        "point": {
            "precision": precision["point"],
            "recall": recall["point"],
            "f1": f1["point"],
        },
        "ci": {
            "precision": precision["ci"],
            "recall": recall["ci"],
            "f1": f1["ci"],
        },
    }


def _prf_stat(pairs: list[tuple[bool, bool]], metric: str) -> dict[str, object]:
    def statistic(sample: list[tuple[bool, bool]]) -> float:
        tp = sum(1 for gold, pred in sample if gold and pred)
        fp = sum(1 for gold, pred in sample if (not gold) and pred)
        fn = sum(1 for gold, pred in sample if gold and (not pred))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        if metric == "precision":
            return precision
        if metric == "recall":
            return recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    return {
        "point": statistic(pairs),
        "ci": bootstrap_resample_ci(pairs, statistic, seed=42),
    }


def _write_overall_csv(path: Path, payload: dict[str, object]) -> None:
    systems = payload["systems"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system_id",
                "precision",
                "precision_ci",
                "recall",
                "recall_ci",
                "f1",
                "f1_ci",
            ],
        )
        writer.writeheader()
        for system_id, system_payload in systems.items():
            overall = system_payload["overall"]
            writer.writerow(
                {
                    "system_id": system_id,
                    "precision": f"{overall['point']['precision'] * 100:.1f}",
                    "precision_ci": _fmt_ci(overall["ci"]["precision"]),
                    "recall": f"{overall['point']['recall'] * 100:.1f}",
                    "recall_ci": _fmt_ci(overall["ci"]["recall"]),
                    "f1": f"{overall['point']['f1'] * 100:.1f}",
                    "f1_ci": _fmt_ci(overall["ci"]["f1"]),
                }
            )


def _write_significance_csv(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["comparison", "p_value", "significant_after_holm"],
        )
        writer.writeheader()
        for name, item in payload["pairwise_mcnemar"].items():
            writer.writerow(
                {
                    "comparison": name,
                    "p_value": item["p_value"],
                    "significant_after_holm": item["significant_after_holm"],
                }
            )


def _artifact_evaluation_ready(artifact_dir: Path) -> bool:
    return all(
        (artifact_dir / filename).is_file()
        for filename in (
            "contract_table.json",
            "event_table.json",
            "resource_table.json",
            "path_table.json",
            "findings.json",
            "exposures.json",
            "diagnostics.json",
            "reconciliation_edges.json",
        )
    )


def _fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"


def _parse_system_artifact_roots(items: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items:
        system_id, sep, path = item.partition("=")
        if not sep or not system_id or not path:
            raise ValueError(f"Invalid --system-artifact-root value: {item!r}")
        parsed[system_id] = path
    return parsed


if __name__ == "__main__":
    main()
