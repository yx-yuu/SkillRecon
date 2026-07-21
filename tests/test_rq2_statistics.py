from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from skillrecon.evaluation.datasets import GoldLabel, GoldLabelRecord
from skillrecon.evaluation.types import EvaluationFinding, EvaluationReport


SCRIPT_PATH = Path("scripts/compute_rq2_statistics.py")


def _load_stats_module():
    spec = importlib.util.spec_from_file_location("compute_rq2_statistics", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rq2_stats_scope_tracks_detection_slices_and_three_label_protocol() -> None:
    stats = _load_stats_module()
    records = [
        GoldLabelRecord(
            skill_id="high-violation",
            gold=GoldLabel(
                label="violation",
                violation_subtype="unsupported_behavior",
            ),
            risk_stratum="High-Risk",
        ),
        GoldLabelRecord(
            skill_id="medium-exposure",
            gold=GoldLabel(label="exposure-only"),
            risk_stratum="medium_risk",
        ),
        GoldLabelRecord(
            skill_id="low-violation",
            gold=GoldLabel(
                label="violation",
                violation_subtype="scope_violation",
            ),
            risk_stratum="low_risk",
        ),
    ]
    rq2_records = [
        record
        for record in records
        if stats.is_rq2_detection_stratum(record.risk_stratum)
    ]
    systems = {
        "skillrecon": {
            "high-violation": _report("high-violation", "violation", "unsupported_behavior"),
            "medium-exposure": _report("medium-exposure", "benign"),
            # This Low-slice hit must not improve RQ2 statistics.
            "low-violation": _report("low-violation", "violation", "scope_violation"),
        },
        "baseline_openclaw": {
            "high-violation": _report("high-violation", "violation", "unsupported_behavior"),
            "medium-exposure": _report("medium-exposure", "violation", "unsupported_behavior"),
            "low-violation": _report("low-violation", "violation", "scope_violation"),
        },
    }

    payload = stats._compute_stats(rq2_records, systems)

    assert payload["scope"]["records"] == 2
    assert payload["scope"]["risk_strata"] == {"high_risk": 1, "medium_risk": 1}
    assert payload["scope"]["gold_labels"] == {"violation": 1, "exposure-only": 1}
    assert payload["scope"]["positive_label"] == "violation"
    assert payload["scope"]["negative_labels"] == ["exposure-only", "benign"]
    assert payload["systems"]["skillrecon"]["overall"]["point"]["precision"] == 1.0
    assert payload["systems"]["skillrecon"]["prediction_coverage"] == {
        "expected": 2,
        "reported": 2,
        "missing": 0,
    }
    assert payload["systems"]["skillrecon"]["overall"]["point"]["recall"] == 1.0
    assert payload["systems"]["baseline_openclaw"]["overall"]["point"]["precision"] == 0.5
    assert payload["systems"]["baseline_openclaw"]["by_slice"]["medium_risk"]["point"]["precision"] == 0.0
    assert "scope_violation" not in payload["systems"]["skillrecon"]["by_subtype"]


def _report(
    skill_id: str,
    label: str,
    subtype: str | None = None,
) -> EvaluationReport:
    findings = []
    if label == "violation":
        findings.append(
            EvaluationFinding(
                finding_id=f"f::{skill_id}",
                main_label="violation",
                subtype=subtype,
            )
        )
    return EvaluationReport(
        skill_id=skill_id,
        system_id="test-system",
        overall_label=label,
        violation_findings=findings,
    )
