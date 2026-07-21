"""Human-audit task-pack and result analysis helpers."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from skillrecon.core.types import CapabilityEvent, Clause, Finding, ResourceUse, Witness
from skillrecon.evaluation.datasets import GoldLabelRecord
from skillrecon.evaluation.metrics import _normalize_violation_subtype


def build_human_audit_task_pack(
    *,
    gold_labels: list[GoldLabelRecord],
    artifact_root: Path,
    output_dir: Path,
    max_per_subtype: int = 12,
) -> dict[str, object]:
    """Build a balanced V1/V2/V3 task pack for a witness audit study."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_by_subtype: dict[str, int] = defaultdict(int)
    tasks: list[dict[str, object]] = []
    for record in gold_labels:
        if record.gold.label != "violation":
            continue
        subtype = _normalize_violation_subtype(record.gold.violation_subtype)
        if subtype is None or selected_by_subtype[subtype] >= max_per_subtype:
            continue
        artifact_dir = artifact_root / record.skill_id
        if not _artifact_evaluation_ready(artifact_dir):
            continue
        task = _task_for_record(
            record=record,
            artifact_dir=artifact_dir,
            task_index=len(tasks) + 1,
        )
        if task is None:
            continue
        tasks.append(task)
        selected_by_subtype[subtype] += 1

    task_path = output_dir / "human_audit_tasks.jsonl"
    task_path.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    summary = {
        "tasks": len(tasks),
        "by_subtype": dict(selected_by_subtype),
        "task_path": str(task_path),
        "response_schema": {
            "task_id": "string",
            "reviewer_id": "string",
            "condition": "alert_summary | witness_bundle | full_package",
            "decision_correct": "boolean",
            "clause_localization_correct": "boolean",
            "code_localization_correct": "boolean",
            "repair_correct": "boolean",
            "time_seconds": "number",
            "confidence": "number in [1,5]",
        },
    }
    (output_dir / "human_audit_task_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def analyze_human_audit_responses(
    *,
    response_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Aggregate human-audit responses and export a paper-facing table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = _load_jsonl(response_path)
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    decisions_by_task_condition: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for response in responses:
        condition = str(response.get("condition", "unknown"))
        by_condition[condition].append(response)
        decisions_by_task_condition[
            (str(response.get("task_id", "")), condition)
        ].append(bool(response.get("decision_correct")))

    summary = {
        condition: _summarize_condition(rows, decisions_by_task_condition)
        for condition, rows in sorted(by_condition.items())
    }
    payload = {
        "responses": len(responses),
        "conditions": summary,
    }
    (output_dir / "human_audit_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "t9_human_audit.tex").write_text(
        render_human_audit_table(payload) + "\n",
        encoding="utf-8",
    )
    return payload


def render_human_audit_table(payload: dict[str, object]) -> str:
    """Render the compact human-audit table planned for the journal paper."""
    conditions = payload.get("conditions", {})
    rows: list[str] = []
    if isinstance(conditions, dict):
        for condition, metrics in conditions.items():
            if not isinstance(metrics, dict):
                continue
            rows.append(
                " & ".join(
                    [
                        _condition_label(str(condition)),
                        _fmt_int(metrics.get("responses")),
                        _fmt_pct(metrics.get("decision_accuracy")),
                        _fmt_seconds(metrics.get("median_time_seconds")),
                        _fmt_pct(metrics.get("localization_accuracy")),
                        _fmt_pct(metrics.get("repair_accuracy")),
                        _fmt_pct(metrics.get("decision_agreement")),
                        _fmt_float(metrics.get("mean_confidence")),
                    ]
                )
                + r" \\"
            )
    if not rows:
        rows.append(r"\multicolumn{8}{c}{No human-audit responses available.} \\")
    return "\n".join(
        [
            r"\begin{tabular}{@{}lccccccc@{}}",
            r"\toprule",
            (
                r"\textbf{Evidence} & \textbf{n} & \textbf{Decision} "
                r"& \textbf{Time} & \textbf{Localization} & \textbf{Repair} "
                r"& \textbf{Agreement} & \textbf{Conf.} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def _task_for_record(
    *,
    record: GoldLabelRecord,
    artifact_dir: Path,
    task_index: int,
) -> dict[str, object] | None:
    subtype = _normalize_violation_subtype(record.gold.violation_subtype)
    findings = _load_models(artifact_dir / "findings.json", Finding)
    finding = next(
        (
            item for item in findings
            if _normalize_violation_subtype(item.finding_type.value) == subtype
        ),
        None,
    )
    if finding is None:
        return None
    clauses = _load_clause_map(artifact_dir / "contract_table.json")
    events = {
        event.event_id: event
        for event in _load_models(artifact_dir / "event_table.json", CapabilityEvent)
    }
    resources = {
        resource.resource_id: resource
        for resource in _load_models(artifact_dir / "resource_table.json", ResourceUse)
    }
    witnesses = {
        witness.finding_id: witness
        for witness in _load_models(artifact_dir / "witnesses.json", Witness, allow_missing=True)
    }
    witness = witnesses.get(finding.finding_id)
    related_clauses = [clauses[item] for item in finding.related_clause_ids if item in clauses]
    related_events = [events[item] for item in finding.related_event_ids if item in events]
    related_resources = [resources[item] for item in finding.related_resource_ids if item in resources]
    return {
        "task_id": f"audit-{task_index:04d}",
        "skill_id": record.skill_id,
        "gold_subtype": subtype,
        "gold_rationale": record.gold.rationale,
        "alert_summary": {
            "subtype": subtype,
            "rationale": finding.rationale,
            "capabilities": [event.capability for event in related_events],
            "code_locations": [event.location for event in related_events],
        },
        "witness_bundle": {
            "witness_id": witness.witness_id if witness else None,
            "exact": bool(witness and witness.is_exact),
            "revalidation_passed": bool(witness and witness.revalidation_passed),
            "clauses": [
                {
                    "clause_id": clause.clause_id,
                    "operator": clause.operator.value,
                    "capability": clause.capability,
                    "target": clause.target,
                    "constraints": [constraint.value for constraint in clause.constraints],
                    "evidence": [span.text for span in clause.evidence_spans[:2]],
                }
                for clause in related_clauses
            ],
            "events": [
                {
                    "event_id": event.event_id,
                    "capability": event.capability,
                    "location": event.location,
                    "detail": event.detail or event.api_call,
                }
                for event in related_events
            ],
            "resources": [
                {
                    "resource_id": resource.resource_id,
                    "type": resource.resource_type,
                    "value": resource.value,
                    "location": resource.location,
                }
                for resource in related_resources
            ],
        },
    }


def _artifact_evaluation_ready(artifact_dir: Path) -> bool:
    return all(
        (artifact_dir / filename).is_file()
        for filename in (
            "contract_table.json",
            "event_table.json",
            "resource_table.json",
            "findings.json",
            "witnesses.json",
        )
    )


def _summarize_condition(
    rows: list[dict[str, object]],
    decisions_by_task_condition: dict[tuple[str, str], list[bool]],
) -> dict[str, object]:
    times = [float(row["time_seconds"]) for row in rows if _is_number(row.get("time_seconds"))]
    confidences = [float(row["confidence"]) for row in rows if _is_number(row.get("confidence"))]
    localization_values = [
        (bool(row.get("clause_localization_correct")) + bool(row.get("code_localization_correct"))) / 2
        for row in rows
    ]
    condition = str(rows[0].get("condition", "unknown")) if rows else "unknown"
    agreements = []
    for (task_id, task_condition), decisions in decisions_by_task_condition.items():
        if task_condition != condition or len(decisions) < 2:
            continue
        counts = Counter(decisions)
        agreements.append(max(counts.values()) / len(decisions))
    return {
        "responses": len(rows),
        "decision_accuracy": _mean_bool(rows, "decision_correct"),
        "median_time_seconds": statistics.median(times) if times else None,
        "localization_accuracy": (
            sum(localization_values) / len(localization_values)
            if localization_values else None
        ),
        "repair_accuracy": _mean_bool(rows, "repair_correct"),
        "decision_agreement": sum(agreements) / len(agreements) if agreements else None,
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
    }


def _load_models(path: Path, model_cls: type, *, allow_missing: bool = False) -> list:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [model_cls.model_validate(item) for item in payload]


def _load_clause_map(path: Path) -> dict[str, Clause]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clauses = payload.get("clauses", []) if isinstance(payload, dict) else []
    return {
        clause.clause_id: clause
        for clause in (Clause.model_validate(item) for item in clauses)
    }


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _mean_bool(rows: list[dict[str, object]], key: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get(key)) for row in rows) / len(rows)


def _is_number(value: object) -> bool:
    return isinstance(value, int | float)


def _condition_label(condition: str) -> str:
    return {
        "alert_summary": "Alert/prose",
        "witness_bundle": "Witness",
        "full_package": "Full package",
    }.get(condition, condition.replace("_", " "))


def _fmt_pct(value: object | None) -> str:
    if value is None:
        return r"\tbd"
    return f"{float(value) * 100:.1f}"


def _fmt_int(value: object | None) -> str:
    if value is None:
        return r"\tbd"
    return str(int(value))


def _fmt_float(value: object | None) -> str:
    if value is None:
        return r"\tbd"
    return f"{float(value):.1f}"


def _fmt_seconds(value: object | None) -> str:
    if value is None:
        return r"\tbd"
    return f"{float(value):.0f}s"
