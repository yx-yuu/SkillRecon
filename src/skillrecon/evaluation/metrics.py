"""Corpus-level metrics for SkillRecon experiments."""

from __future__ import annotations

from dataclasses import dataclass

from skillrecon.core.enums import ClauseOperator
from skillrecon.core.types import Clause, ReconciliationEdge
from skillrecon.evaluation.datasets import ClauseAnnotation, EdgeAnnotation, GoldLabelRecord
from skillrecon.evaluation.types import EvaluationReport


_RQ2_SUBTYPE_ALIASES = {
    "undeclared_high_impact_capability": "unsupported_behavior",
    "unsupported_behavior": "unsupported_behavior",
    "contradicted_behavior": "unsupported_behavior",
    "scope_violation": "scope_violation",
    "unjustified_composition": "unjustified_composition",
    "unjustified_dangerous_composition": "unjustified_composition",
}
_RQ2_SLICE_STRATA = {"high", "high_risk", "medium", "medium_risk"}
_LOW_SLICE_STRATA = {"low", "low_risk", "low_slice", "no_risk"}


@dataclass(frozen=True)
class PRFCounts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return 0.0 if self.tp + self.fp == 0 else self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        return 0.0 if self.tp + self.fn == 0 else self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)


def compute_binary_prf(
    *,
    gold_positive: set[str],
    predicted_positive: set[str],
) -> PRFCounts:
    tp = len(gold_positive & predicted_positive)
    fp = len(predicted_positive - gold_positive)
    fn = len(gold_positive - predicted_positive)
    return PRFCounts(tp=tp, fp=fp, fn=fn)


def compute_clause_operator_metrics(
    annotations: list[GoldLabelRecord],
    predicted_clauses_by_skill: dict[str, list[Clause]],
) -> dict[str, PRFCounts]:
    """Compute clause extraction metrics grouped by operator."""
    operators = {
        ClauseOperator.ALLOWED.value,
        ClauseOperator.PROHIBITED.value,
        ClauseOperator.UNKNOWN.value,
    }
    metrics: dict[str, PRFCounts] = {}
    for operator in operators:
        gold = {
            (record.skill_id, _clause_signature_from_gold(clause))
            for record in annotations
            for clause in record.clause_labels
            if clause.operator == operator
        }
        predicted = {
            (skill_id, _clause_signature_from_predicted(clause))
            for skill_id, clauses in predicted_clauses_by_skill.items()
            for clause in clauses
            if clause.operator.value == operator
        }
        metrics[operator] = compute_binary_prf(
            gold_positive=gold,
            predicted_positive=predicted,
        )
    return metrics


def compute_false_authorization_rate(
    annotations: list[GoldLabelRecord],
    predicted_clauses_by_skill: dict[str, list[Clause]],
) -> float:
    """Measure how often predicted allowed clauses are not gold-allowed."""
    gold_operator_by_signature = {
        (record.skill_id, _clause_signature_from_gold(clause)): clause.operator
        for record in annotations
        for clause in record.clause_labels
    }
    predicted_allowed = [
        (skill_id, _clause_signature_from_predicted(clause))
        for skill_id, clauses in predicted_clauses_by_skill.items()
        for clause in clauses
        if clause.operator == ClauseOperator.ALLOWED
    ]
    if not predicted_allowed:
        return 0.0
    false_allowed = sum(
        1
        for skill_id, signature in predicted_allowed
        if gold_operator_by_signature.get((skill_id, signature)) not in {ClauseOperator.ALLOWED.value}
    )
    return false_allowed / len(predicted_allowed)


def compute_edge_accuracy(
    edge_labels: list[EdgeAnnotation],
    predicted_edges: list[ReconciliationEdge],
) -> float:
    """Measure sampled edge accuracy using provided signatures."""
    if not edge_labels:
        return 0.0
    predicted_signatures = {_edge_signature(edge) for edge in predicted_edges}
    correct = sum(
        1
        for label in edge_labels
        if (label.signature in predicted_signatures) == label.expected_correct
    )
    return correct / len(edge_labels)


def compute_edge_accuracy_by_type(
    annotations: list[GoldLabelRecord],
    predicted_edges_by_skill: dict[str, list[ReconciliationEdge]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate sampled edge accuracy grouped by edge type."""
    labels_by_type: dict[str, list[tuple[str, EdgeAnnotation]]] = {}
    for record in annotations:
        for label in record.edge_labels:
            labels_by_type.setdefault(label.edge_type, []).append((record.skill_id, label))

    results: dict[str, dict[str, float | int]] = {}
    for edge_type, labeled_items in labels_by_type.items():
        total = len(labeled_items)
        correct = 0
        for skill_id, label in labeled_items:
            predicted_signatures = {
                _edge_signature(edge)
                for edge in predicted_edges_by_skill.get(skill_id, [])
            }
            if (label.signature in predicted_signatures) == label.expected_correct:
                correct += 1
        accuracy = 0.0 if total == 0 else correct / total
        results[edge_type] = {
            "accuracy": accuracy,
            "sampled": total,
            "correct": correct,
        }
    return results


def compute_edge_validity_by_type(
    annotations: list[GoldLabelRecord],
    predicted_edges_by_skill: dict[str, list[ReconciliationEdge]],
) -> dict[str, dict[str, float | int | None]]:
    """Aggregate positive validity / negative rejection by edge type.

    Positive samples are labeled items whose edge signature appears in the
    predicted graph. Negative samples are labeled items whose signature does
    not appear in the predicted graph. ``expected_correct`` expresses whether
    the edge should exist under the annotation rubric.
    """
    labels_by_type: dict[str, list[tuple[str, EdgeAnnotation]]] = {}
    for record in annotations:
        for label in record.edge_labels:
            labels_by_type.setdefault(label.edge_type, []).append((record.skill_id, label))

    results: dict[str, dict[str, float | int | None]] = {}
    for edge_type, labeled_items in labels_by_type.items():
        positive_total = 0
        positive_correct = 0
        negative_total = 0
        negative_correct = 0
        for skill_id, label in labeled_items:
            predicted_signatures = {
                _edge_signature(edge)
                for edge in predicted_edges_by_skill.get(skill_id, [])
            }
            if label.signature in predicted_signatures:
                positive_total += 1
                if label.expected_correct:
                    positive_correct += 1
            else:
                negative_total += 1
                if not label.expected_correct:
                    negative_correct += 1
        pos_valid = (
            None if positive_total == 0 else positive_correct / positive_total
        )
        neg_rej = (
            None if negative_total == 0 else negative_correct / negative_total
        )
        bal_valid = (
            None
            if pos_valid is None or neg_rej is None
            else (pos_valid + neg_rej) / 2
        )
        results[edge_type] = {
            "pos_valid": pos_valid,
            "neg_rej": neg_rej,
            "bal_valid": bal_valid,
            "judged_positive": positive_total,
            "positive_correct": positive_correct,
            "judged_negative": negative_total,
            "negative_correct": negative_correct,
        }
    return results


def compute_violation_metrics(
    annotations: list[GoldLabelRecord],
    reports_by_skill: dict[str, EvaluationReport],
) -> PRFCounts:
    gold = {
        record.skill_id
        for record in annotations
        if record.gold.label == "violation"
    }
    predicted = {
        skill_id
        for skill_id, report in reports_by_skill.items()
        if report.overall_label == "violation"
    }
    return compute_binary_prf(gold_positive=gold, predicted_positive=predicted)


def compute_violation_metrics_by_subtype(
    annotations: list[GoldLabelRecord],
    reports_by_skill: dict[str, EvaluationReport],
) -> dict[str, PRFCounts]:
    subtypes = {
        _normalize_violation_subtype(record.gold.violation_subtype)
        for record in annotations
        if record.gold.violation_subtype is not None
    }
    metrics: dict[str, PRFCounts] = {}
    for subtype in sorted(subtypes):
        gold = {
            record.skill_id
            for record in annotations
            if record.gold.label == "violation"
            and _normalize_violation_subtype(record.gold.violation_subtype) == subtype
        }
        predicted = {
            skill_id
            for skill_id, report in reports_by_skill.items()
            if any(
                _normalize_violation_subtype(finding.subtype) == subtype
                for finding in report.violation_findings
            )
        }
        metrics[subtype] = compute_binary_prf(
            gold_positive=gold,
            predicted_positive=predicted,
        )
    return metrics


def compute_boundary_false_violation_rate(
    annotations: list[GoldLabelRecord],
    reports_by_skill: dict[str, EvaluationReport],
) -> dict[str, float]:
    by_stratum: dict[str, list[bool]] = {}
    for record in annotations:
        stratum = record.risk_stratum or "unstratified"
        report = reports_by_skill.get(record.skill_id)
        if report is None:
            continue
        by_stratum.setdefault(stratum, []).append(
            record.gold.label != "violation" and report.overall_label == "violation"
        )
    return {
        stratum: (sum(values) / len(values) if values else 0.0)
        for stratum, values in by_stratum.items()
    }


def compute_confusion_matrix(
    annotations: list[GoldLabelRecord],
    reports_by_skill: dict[str, EvaluationReport],
) -> dict[str, dict[str, int]]:
    """Compute a simple label-level confusion matrix."""
    labels = ["violation", "exposure-only", "benign"]
    matrix = {
        gold_label: {pred_label: 0 for pred_label in labels}
        for gold_label in labels
    }
    for record in annotations:
        report = reports_by_skill.get(record.skill_id)
        if report is None:
            continue
        matrix[record.gold.label][report.overall_label] += 1
    return matrix


def compute_discovery_yield(
    annotations: list[GoldLabelRecord],
    reports_by_skill: dict[str, EvaluationReport],
) -> dict[str, object]:
    """Compute RQ3 discovery yield on scanner-benign/Low-slice records.

    RQ3 treats ``violation`` and ``exposure-only`` as distinct gold outcomes.
    A transparent high-risk exposure is still a useful scanner-benign
    discovery, but it is not counted as a confirmed violation.
    """
    low_records = [record for record in annotations if is_low_slice_record(record)]
    flagged = 0
    confirmed = 0
    confirmed_violations = 0
    confirmed_exposures = 0
    confirmed_by_type = {
        "unsupported_behavior": 0,
        "scope_violation": 0,
        "unjustified_composition": 0,
    }
    confirmed_exposure_by_type = {
        "declared_sensitive_behavior": 0,
        "declared_sensitive_composition": 0,
    }
    for record in low_records:
        report = reports_by_skill.get(record.skill_id)
        if report is None or report.overall_label == "benign":
            continue
        flagged += 1
        if report.overall_label != record.gold.label:
            continue
        confirmed += 1
        if record.gold.label == "violation":
            confirmed_violations += 1
            subtype = _normalize_violation_subtype(record.gold.violation_subtype)
            if subtype in confirmed_by_type:
                confirmed_by_type[subtype] += 1
        elif record.gold.label == "exposure-only":
            confirmed_exposures += 1
            for exposure in report.exposure_findings:
                subtype = exposure.subtype
                if subtype in confirmed_exposure_by_type:
                    confirmed_exposure_by_type[subtype] += 1
    return {
        "low_total": len(low_records),
        "flagged": flagged,
        "checked": flagged,
        "audited": flagged,
        "confirmed": confirmed,
        "confirmed_violations": confirmed_violations,
        "confirmed_exposures": confirmed_exposures,
        "confirmation_rate": 0.0 if flagged == 0 else confirmed / flagged,
        "confirmed_by_type": confirmed_by_type,
        "confirmed_exposure_by_type": confirmed_exposure_by_type,
    }


def is_rq2_detection_stratum(value: str | None) -> bool:
    return _normalize_stratum(value) in _RQ2_SLICE_STRATA


def is_low_slice_record(record: GoldLabelRecord) -> bool:
    if _is_low_slice_value(record.risk_stratum):
        return True
    metadata_risk = record.metadata.get("risk_tier")
    if isinstance(metadata_risk, str) and _is_low_slice_value(metadata_risk):
        return True
    return (
        record.metadata.get("virus_total_status") == "benign"
        and record.metadata.get("openclaw_status") == "benign"
    )


def _is_low_slice_value(value: str | None) -> bool:
    return _normalize_stratum(value) in _LOW_SLICE_STRATA


def _normalize_stratum(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower().replace("-", "_")


def _clause_signature_from_gold(clause: ClauseAnnotation) -> str:
    if clause.signature:
        return clause.signature
    return "|".join(
        [
            clause.operator,
            clause.capability,
            _normalize_target_surface(clause.target or ""),
            ";".join(sorted(value.strip().lower() for value in clause.constraints)),
        ]
    )


def _clause_signature_from_predicted(clause: Clause) -> str:
    return "|".join(
        [
            clause.operator.value,
            clause.capability,
            _normalize_target_surface(clause.target or ""),
            ";".join(sorted(constraint.value.strip().lower() for constraint in clause.constraints)),
        ]
    )


def _edge_signature(edge: ReconciliationEdge) -> str:
    return "|".join(
        [
            edge.relation.value,
            edge.clause_id or "",
            edge.constraint_id or "",
            edge.event_id or "",
            edge.resource_id or "",
            edge.path_id or "",
            edge.step_id or "",
            edge.code_unit_id or "",
        ]
    )


def _normalize_target_surface(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(("`", '"', "'")) and cleaned.endswith(("`", '"', "'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned.lower()


def _normalize_violation_subtype(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    return _RQ2_SUBTYPE_ALIASES.get(normalized, normalized)
