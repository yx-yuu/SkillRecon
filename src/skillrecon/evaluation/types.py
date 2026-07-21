"""Unified evaluation result schema shared by SkillRecon and baselines."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CodeLocation(BaseModel):
    """A concrete code location surfaced for evaluation inspection."""

    model_config = ConfigDict(frozen=True)

    path: str
    line: int | None = None


class EvaluationFinding(BaseModel):
    """A normalized finding record used across evaluation systems."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    main_label: str
    subtype: str | None = None
    certificate_ids: list[str] = []
    capability_atoms: list[str] = []
    matched_clauses: list[str] = []
    code_locations: list[CodeLocation] = []
    rationale: str = ""


class ContractQualityAlertRecord(BaseModel):
    """A normalized contract-quality alert record."""

    model_config = ConfigDict(frozen=True)

    alert_id: str
    alert_type: str = "contract_quality_alert"
    affected_clauses: list[str] = []
    related_event_ids: list[str] = []
    code_locations: list[CodeLocation] = []
    rationale: str = ""


class GraphStats(BaseModel):
    """Compact graph metadata written into evaluation reports."""

    model_config = ConfigDict(frozen=True)

    node_count: int = 0
    edge_count: int = 0


class ReportGraphs(BaseModel):
    """Graph counters for the three core artifact graphs."""

    model_config = ConfigDict(frozen=True)

    g_d: GraphStats = GraphStats()
    g_c: GraphStats = GraphStats()
    g_x: GraphStats = GraphStats()


class ReportSummary(BaseModel):
    """Top-level count summary for report consumers."""

    model_config = ConfigDict(frozen=True)

    violation_count: int = 0
    exposure_count: int = 0
    contract_quality_count: int = 0


class EvaluationReport(BaseModel):
    """Machine-readable top-level analysis report."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    system_id: str = "skillrecon"
    analyzer_version: str = "skillrecon-v0"
    config: dict[str, object] = {}
    graphs: ReportGraphs = ReportGraphs()
    overall_label: str = "benign"
    violation_findings: list[EvaluationFinding] = []
    exposure_findings: list[EvaluationFinding] = []
    contract_quality_alerts: list[ContractQualityAlertRecord] = []
    permission_manifest_ref: str = "permission_manifest.json"
    summary: ReportSummary = ReportSummary()
