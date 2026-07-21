"""Experiment runner for RQ1-RQ4 corpus-level evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from skillrecon.core.config import AnalyzerConfig, LLMConfig
from skillrecon.core.types import Clause, ReconciliationEdge
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
    GoldLabel,
    load_gold_label_records,
    load_baseline_prediction_records,
    load_seeded_benchmark_records,
)
from skillrecon.evaluation.metrics import (
    compute_boundary_false_violation_rate,
    compute_clause_operator_metrics,
    compute_confusion_matrix,
    compute_discovery_yield,
    compute_edge_accuracy_by_type,
    compute_edge_validity_by_type,
    compute_false_authorization_rate,
    compute_violation_metrics,
    compute_violation_metrics_by_subtype,
    is_rq2_detection_stratum,
)
from skillrecon.evaluation.extended import (
    build_generalization_results,
    build_granularity_results,
    build_robustness_and_cost_results,
)
from skillrecon.evaluation.figures import render_all_figures
from skillrecon.evaluation.tables import render_all_tables, write_table
from skillrecon.evaluation.types import EvaluationReport


@dataclass(frozen=True)
class ExperimentInputs:
    gold_label_path: Path | None
    seeded_path: Path | None
    artifact_root: Path
    rq1_gold_label_path: Path | None = None
    rq1_artifact_root: Path | None = None
    external_prediction_path: Path | None = None
    llm_judge_enabled: bool = False
    llm_judge_config: LLMConfig | None = None
    system_artifact_roots: dict[str, Path] | None = None


def run_all_experiments(inputs: ExperimentInputs, output_dir: Path) -> dict[str, object]:
    """Run all configured experiment routines and return a JSON-serializable bundle."""
    analyzer_config = AnalyzerConfig(llm=LLMConfig(base_url="http://localhost", model="dummy"))
    llm_judge_config = inputs.llm_judge_config or analyzer_config.llm
    gold_labels = (
        load_gold_label_records(inputs.gold_label_path)
        if inputs.gold_label_path is not None
        else []
    )
    rq1_gold_labels = (
        load_gold_label_records(inputs.rq1_gold_label_path)
        if inputs.rq1_gold_label_path is not None
        else gold_labels
    )
    seeded = (
        load_seeded_benchmark_records(inputs.seeded_path)
        if inputs.seeded_path is not None
        else []
    )
    seeded_as_annotations = [
        GoldLabelRecord(
            skill_id=record.skill_id,
            gold=record.gold,
            risk_stratum="seeded",
            bucket="seeded",
            expected_sites=record.expected_sites,
            metadata=record.metadata,
        )
        for record in seeded
    ]
    external_predictions = index_baseline_predictions(
        load_baseline_prediction_records(inputs.external_prediction_path)
        if inputs.external_prediction_path is not None
        else []
    )

    rq2_gold_label_records = [*gold_labels, *seeded_as_annotations]
    rq2_slice_records = [
        record
        for record in gold_labels
        if is_rq2_detection_stratum(record.risk_stratum)
    ]
    rq2_skill_ids = {
        *(record.skill_id for record in gold_labels),
        *(record.skill_id for record in seeded_as_annotations),
        *(record.skill_id for record in seeded),
        *(skill_id for (_system_id, skill_id) in external_predictions),
    }
    skill_ids = sorted(
        {
            *rq2_skill_ids,
            *(record.skill_id for record in rq1_gold_labels),
        }
    )
    rq1_artifact_root = inputs.rq1_artifact_root or inputs.artifact_root

    available_artifact_dirs = {
        skill_id: artifact_dir
        for skill_id in skill_ids
        if (
            artifact_dir := _resolve_artifact_dir(
                skill_id,
                primary_root=inputs.artifact_root,
                fallback_root=rq1_artifact_root,
            )
        )
        is not None
        and _artifact_evaluation_ready(artifact_dir)
    }
    systems = _build_system_reports(
        available_artifact_dirs=available_artifact_dirs,
        system_artifact_roots=inputs.system_artifact_roots or {},
        analyzer_config=analyzer_config,
        llm_judge_enabled=inputs.llm_judge_enabled,
        llm_judge_config=llm_judge_config,
        external_predictions=external_predictions,
    )

    predicted_clauses_by_skill = _load_rq1_clauses(
        rq1_gold_labels,
        primary_artifact_root=inputs.artifact_root,
        fallback_artifact_root=rq1_artifact_root,
    )
    predicted_edges_by_skill = _load_rq1_edges(
        rq1_gold_labels,
        primary_artifact_root=inputs.artifact_root,
        fallback_artifact_root=rq1_artifact_root,
    )
    rq1 = _build_rq1_results(
        rq1_gold_labels=rq1_gold_labels,
        predicted_clauses_by_skill=predicted_clauses_by_skill,
        predicted_edges_by_skill=predicted_edges_by_skill,
    )
    rq2 = _build_rq2_results(
        systems=systems,
        rq2_gold_label_records=rq2_gold_label_records,
        rq2_slice_records=rq2_slice_records,
        rq2_skill_ids=rq2_skill_ids,
    )
    rq2_meta = {
        "subtype_support": _subtype_support_counts(rq2_slice_records),
    }
    rq3 = _build_rq3_results(
        gold_labels=gold_labels,
        systems=systems,
    )
    rq4 = _compute_witness_fidelity(
        skill_ids={record.skill_id for record in gold_labels},
        artifact_root=inputs.artifact_root,
    )
    rq3_granularity = build_granularity_results(
        artifact_dirs=available_artifact_dirs,
        analyzer_config=analyzer_config,
        gold_labels=gold_labels,
    )
    rq5_generalization = build_generalization_results(
        gold_labels=gold_labels,
        reports_by_skill=systems.get("skillrecon", {}),
        artifact_dirs=available_artifact_dirs,
    )
    rq6_robustness_cost = build_robustness_and_cost_results(
        rq2=rq2,
        artifact_dirs=available_artifact_dirs,
    )
    appendix = _build_appendix_results(
        gold_labels=gold_labels,
        systems=systems,
        skill_ids=rq2_skill_ids,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "rq1": rq1,
        "rq2": rq2,
        "rq2_meta": rq2_meta,
        "rq3": rq3,
        "rq3_granularity": rq3_granularity,
        "rq4": rq4,
        "rq5_generalization": rq5_generalization,
        "rq6_robustness_cost": rq6_robustness_cost,
        "appendix": {
            "systems": appendix,
        },
    }
    (output_dir / "experiment_results.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for filename, content in render_all_tables(bundle).items():
        write_table(output_dir / filename, content)
    render_all_figures(bundle, output_dir / "figures")
    return bundle


def _build_system_reports(
    *,
    available_artifact_dirs: dict[str, Path],
    system_artifact_roots: dict[str, Path],
    analyzer_config: AnalyzerConfig,
    llm_judge_enabled: bool,
    llm_judge_config: LLMConfig,
    external_predictions,
) -> dict[str, dict[str, EvaluationReport]]:
    systems = {
        "skillrecon": {
            skill_id: build_skillrecon_baseline_report(skill_id, artifact_dir)
            for skill_id, artifact_dir in available_artifact_dirs.items()
        },
        "baseline_rule_scanner": {
            skill_id: build_rule_based_scanner_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in available_artifact_dirs.items()
        },
        "baseline_capability_lattice": {
            skill_id: build_capability_lattice_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in available_artifact_dirs.items()
        },
        "baseline_doc_code_consistency": {
            skill_id: build_doc_code_consistency_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in available_artifact_dirs.items()
        },
        "baseline_spec_containment": {
            skill_id: build_spec_containment_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in available_artifact_dirs.items()
        },
        "baseline_instruction_constraints": {
            skill_id: build_instruction_constraint_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=analyzer_config,
            )
            for skill_id, artifact_dir in available_artifact_dirs.items()
        },
    }
    if llm_judge_enabled:
        systems["baseline_llm_judge"] = {
            skill_id: build_llm_judge_report(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
                analyzer_config=AnalyzerConfig(llm=llm_judge_config),
            )
            for skill_id, artifact_dir in available_artifact_dirs.items()
        }
    systems.update(
        {
            system_id: {
                skill_id: build_skillrecon_baseline_report(skill_id, root / skill_id)
                for skill_id in available_artifact_dirs
                if (root / skill_id).exists()
            }
            for system_id, root in sorted(system_artifact_roots.items())
        }
    )
    systems.update(
        {
            system_id: {
                skill_id: build_external_prediction_report(
                    skill_id=skill_id,
                    system_id=system_id,
                    prediction=prediction,
                )
                for (pred_system_id, skill_id), prediction in external_predictions.items()
                if pred_system_id == system_id
            }
            for system_id in sorted(
                {system_id for system_id, _skill_id in external_predictions}
            )
        }
    )
    return systems


def _build_rq1_results(
    *,
    rq1_gold_labels: list[GoldLabelRecord],
    predicted_clauses_by_skill: dict[str, list[Clause]],
    predicted_edges_by_skill: dict[str, list[ReconciliationEdge]],
) -> dict[str, object]:
    clause_metrics = compute_clause_operator_metrics(
        rq1_gold_labels,
        predicted_clauses_by_skill,
    )
    edge_accuracy_by_type = compute_edge_accuracy_by_type(
        rq1_gold_labels,
        predicted_edges_by_skill,
    )
    edge_validity_by_type = compute_edge_validity_by_type(
        rq1_gold_labels,
        predicted_edges_by_skill,
    )
    positive_correct = sum(
        int(metric["positive_correct"])
        for metric in edge_validity_by_type.values()
    )
    judged_positive = sum(
        int(metric["judged_positive"])
        for metric in edge_validity_by_type.values()
    )
    negative_correct = sum(
        int(metric["negative_correct"])
        for metric in edge_validity_by_type.values()
    )
    judged_negative = sum(
        int(metric["judged_negative"])
        for metric in edge_validity_by_type.values()
    )
    overall_pos_valid = (
        None
        if not edge_validity_by_type
        else _safe_ratio(positive_correct, judged_positive)
    )
    overall_neg_rej = (
        None
        if not edge_validity_by_type
        else _safe_ratio(negative_correct, judged_negative)
    )

    return {
        "clause_metrics": {
            operator: {
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f1,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
            }
            for operator, counts in clause_metrics.items()
        },
        "overall_clause_metrics": _counts_to_dict(
            _combine_counts(clause_metrics.values())
        ),
        "false_authorization_rate": compute_false_authorization_rate(
            rq1_gold_labels,
            predicted_clauses_by_skill,
        ),
        "edge_accuracy_by_type": edge_accuracy_by_type,
        "overall_edge_accuracy": {
            "accuracy": (
                0.0
                if not edge_accuracy_by_type
                else sum(metric["correct"] for metric in edge_accuracy_by_type.values())
                / sum(metric["sampled"] for metric in edge_accuracy_by_type.values())
            ),
            "sampled": sum(metric["sampled"] for metric in edge_accuracy_by_type.values()),
        },
        "edge_validity_by_type": edge_validity_by_type,
        "overall_edge_validity": {
            "pos_valid": overall_pos_valid,
            "neg_rej": overall_neg_rej,
            "bal_valid": _balanced_average(overall_pos_valid, overall_neg_rej),
            "judged_positive": judged_positive,
            "judged_negative": judged_negative,
        },
    }


def _build_rq2_results(
    *,
    systems: dict[str, dict[str, EvaluationReport]],
    rq2_gold_label_records: list[GoldLabelRecord],
    rq2_slice_records: list[GoldLabelRecord],
    rq2_skill_ids: set[str],
) -> dict[str, object]:
    scoped_systems = {
        system_id: {
            skill_id: report
            for skill_id, report in reports.items()
            if skill_id in rq2_skill_ids
        }
        for system_id, reports in systems.items()
    }
    results: dict[str, object] = {}
    for system_id, reports in scoped_systems.items():
        slice_metrics = {}
        for stratum in ("high_risk", "medium_risk"):
            slice_records = [
                record
                for record in rq2_slice_records
                if _is_slice_stratum(record.risk_stratum, stratum)
            ]
            slice_metrics[stratum] = _counts_to_dict(
                compute_violation_metrics(
                    slice_records,
                    _reports_for_records(reports, slice_records),
                )
            )

        results[system_id] = {
            "prediction_coverage": _prediction_coverage(
                rq2_gold_label_records,
                reports,
            ),
            "overall": _counts_to_dict(
                compute_violation_metrics(rq2_gold_label_records, reports)
            ),
            "by_slice": slice_metrics,
            "by_subtype": {
                subtype: _counts_to_dict(counts)
                for subtype, counts in compute_violation_metrics_by_subtype(
                    rq2_gold_label_records,
                    _reports_for_records(reports, rq2_gold_label_records),
                ).items()
            },
            "paper_by_subtype": {
                subtype: _counts_to_dict(counts)
                for subtype, counts in compute_violation_metrics_by_subtype(
                    rq2_slice_records,
                    _reports_for_records(reports, rq2_slice_records),
                ).items()
            },
            "medium_disputed": _compute_medium_disputed_recovery(
                rq2_slice_records,
                reports,
            ),
        }
    return results


def _reports_for_records(
    reports: dict[str, EvaluationReport],
    records: list[GoldLabelRecord],
) -> dict[str, EvaluationReport]:
    record_ids = {record.skill_id for record in records}
    return {
        skill_id: report
        for skill_id, report in reports.items()
        if skill_id in record_ids
    }


def _prediction_coverage(
    records: list[GoldLabelRecord],
    reports: dict[str, EvaluationReport],
) -> dict[str, object]:
    expected = {record.skill_id for record in records}
    actual = {skill_id for skill_id in reports if skill_id in expected}
    return {
        "expected": len(expected),
        "reported": len(actual),
        "missing": len(expected - actual),
    }


def _build_rq3_results(
    *,
    gold_labels: list[GoldLabelRecord],
    systems: dict[str, dict[str, EvaluationReport]],
) -> dict[str, object]:
    skill_ids = {record.skill_id for record in gold_labels}
    return {
        system_id: compute_discovery_yield(gold_labels, reports)
        for system_id, reports in {
            system_id: {
                skill_id: report
                for skill_id, report in reports.items()
                if skill_id in skill_ids
            }
            for system_id, reports in systems.items()
        }.items()
    }


def _compute_medium_disputed_recovery(
    records: list[GoldLabelRecord],
    reports: dict[str, EvaluationReport],
) -> dict[str, object]:
    disputed = [
        record
        for record in records
        if _is_slice_stratum(record.risk_stratum, "medium_risk")
        and record.gold.label != "violation"
    ]
    corrected = 0
    for record in disputed:
        report = reports.get(record.skill_id)
        predicted_violation = report is not None and report.overall_label == "violation"
        if not predicted_violation:
            corrected += 1
    total = len(disputed)
    return {
        "corrected_fp": corrected,
        "disputed_total": total,
        "recovery": None if total == 0 else corrected / total,
    }


def _build_appendix_results(
    *,
    gold_labels: list[GoldLabelRecord],
    systems: dict[str, dict[str, EvaluationReport]],
    skill_ids: set[str],
) -> dict[str, object]:
    return {
        system_id: {
            "false_violation_rate": compute_boundary_false_violation_rate(
                gold_labels,
                reports,
            ),
            "confusion_matrix": compute_confusion_matrix(gold_labels, reports),
        }
        for system_id, reports in {
            system_id: {
                skill_id: report
                for skill_id, report in reports.items()
                if skill_id in skill_ids
            }
            for system_id, reports in systems.items()
        }.items()
    }


def _load_rq1_clauses(
    rq1_gold_labels: list[GoldLabelRecord],
    *,
    primary_artifact_root: Path,
    fallback_artifact_root: Path,
) -> dict[str, list[Clause]]:
    tables: dict[str, list[Clause]] = {}
    for record in rq1_gold_labels:
        artifact_dir = _resolve_artifact_dir(
            record.skill_id,
            primary_root=primary_artifact_root,
            fallback_root=fallback_artifact_root,
        )
        if artifact_dir is None:
            continue
        table_path = artifact_dir / "contract_table.json"
        if table_path.exists():
            tables[record.skill_id] = _load_clauses(table_path)
    return tables


def _load_rq1_edges(
    rq1_gold_labels: list[GoldLabelRecord],
    *,
    primary_artifact_root: Path,
    fallback_artifact_root: Path,
) -> dict[str, list[ReconciliationEdge]]:
    tables: dict[str, list[ReconciliationEdge]] = {}
    for record in rq1_gold_labels:
        artifact_dir = _resolve_artifact_dir(
            record.skill_id,
            primary_root=primary_artifact_root,
            fallback_root=fallback_artifact_root,
        )
        if artifact_dir is None:
            continue
        edge_path = artifact_dir / "reconciliation_edges.json"
        if edge_path.exists():
            tables[record.skill_id] = _load_edges(edge_path)
    return tables


def _resolve_artifact_dir(
    skill_id: str,
    *,
    primary_root: Path,
    fallback_root: Path,
) -> Path | None:
    for candidate in (primary_root / skill_id, fallback_root / skill_id):
        if candidate.exists():
            return candidate
    return None


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


def _load_clauses(path: Path) -> list[Clause]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Clause.model_validate(item) for item in payload.get("clauses", [])]


def _load_edges(path: Path) -> list[ReconciliationEdge]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ReconciliationEdge.model_validate(item) for item in payload]


def _counts_to_dict(counts) -> dict[str, float]:
    return {
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f1,
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
    }


def _combine_counts(counts_iterable) -> object:
    counts_list = list(counts_iterable)
    from skillrecon.evaluation.metrics import PRFCounts

    return PRFCounts(
        tp=sum(item.tp for item in counts_list),
        fp=sum(item.fp for item in counts_list),
        fn=sum(item.fn for item in counts_list),
    )


def _subtype_support_counts(records: list[GoldLabelRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        if record.gold.label != "violation" or record.gold.violation_subtype is None:
            continue
        from skillrecon.evaluation.metrics import _normalize_violation_subtype

        normalized = _normalize_violation_subtype(record.gold.violation_subtype)
        if normalized is not None:
            counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _compute_witness_fidelity(
    *,
    skill_ids: set[str],
    artifact_root: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for skill_id in skill_ids:
        artifact_dir = artifact_root / skill_id
        if artifact_dir.exists():
            rows.extend(_load_witness_rows(artifact_dir))
    by_subtype = {
        subtype: _aggregate_witness_rows(
            [row for row in rows if row["subtype"] == subtype]
        )
        for subtype in (
            "unsupported_behavior",
            "scope_violation",
            "unjustified_composition",
        )
    }
    return {
        "skillrecon": {
            "by_subtype": by_subtype,
            "overall": _aggregate_witness_rows(rows),
        }
    }


def _load_witness_rows(artifact_dir: Path) -> list[dict[str, object]]:
    findings_path = artifact_dir / "findings.json"
    if not findings_path.exists():
        return []

    findings = _load_json_list(findings_path)
    valid_witnesses = _load_json_list(artifact_dir / "witnesses.json")
    rejected_witnesses = _load_json_list(artifact_dir / "rejected_witnesses.json")
    validations = _load_json_list(artifact_dir / "witness_validation.json")

    witness_by_finding = {
        str(item.get("finding_id", "")): item
        for item in [*valid_witnesses, *rejected_witnesses]
        if isinstance(item, dict) and item.get("finding_id")
    }
    validation_by_witness = {
        str(item.get("witness_id", "")): item
        for item in validations
        if isinstance(item, dict) and item.get("witness_id")
    }

    rows: list[dict[str, object]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        subtype = _normalize_finding_subtype(str(finding.get("finding_type", "")))
        if subtype is None:
            subtype = _normalize_finding_subtype(str(finding.get("subtype", "")))
        if subtype is None:
            continue
        finding_id = str(finding.get("finding_id", ""))
        witness = witness_by_finding.get(finding_id)
        validation = (
            validation_by_witness.get(str(witness.get("witness_id", "")))
            if witness is not None
            else None
        )
        rows.append(
            {
                "subtype": subtype,
                "covered": witness is not None,
                "revalidated": bool(
                    (witness and witness.get("revalidation_passed"))
                    or (
                        validation
                        and validation.get("core_passed", validation.get("passed"))
                    )
                ),
                "irreducible": bool(
                    validation
                    and validation.get("irreducible_passed", validation.get("passed"))
                ),
                "exact": bool(witness and witness.get("is_exact")),
                "cross_modal": _is_cross_modal_witness(witness, validation),
            }
        )
    return rows


def _aggregate_witness_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    if total == 0:
        return {
            "n": 0,
            "coverage": None,
            "revalidation": None,
            "irreducibility": None,
            "exact": None,
            "cross_modal": None,
        }
    return {
        "n": total,
        "coverage": sum(bool(row["covered"]) for row in rows) / total,
        "revalidation": sum(bool(row["revalidated"]) for row in rows) / total,
        "irreducibility": sum(bool(row["irreducible"]) for row in rows) / total,
        "exact": sum(bool(row["exact"]) for row in rows) / total,
        "cross_modal": sum(bool(row["cross_modal"]) for row in rows) / total,
    }


def _load_json_list(path: Path) -> list[object]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _normalize_finding_subtype(value: str) -> str | None:
    from skillrecon.evaluation.metrics import _normalize_violation_subtype

    return _normalize_violation_subtype(value)


def _is_cross_modal_witness(
    witness: dict[str, object] | None,
    validation: dict[str, object] | None,
) -> bool:
    if validation and "cross_modal_grounded" in validation:
        return bool(validation.get("cross_modal_grounded"))
    if witness is None:
        return False
    return bool(witness.get("projection_edge_ids"))


def _is_slice_stratum(value: str | None, canonical: str) -> bool:
    if value is None:
        return False
    normalized = value.lower().replace("-", "_")
    if normalized == canonical:
        return True
    return normalized == canonical.removesuffix("_risk")


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _balanced_average(
    positive_value: float | None,
    negative_value: float | None,
) -> float | None:
    if positive_value is None or negative_value is None:
        return None
    return (positive_value + negative_value) / 2
