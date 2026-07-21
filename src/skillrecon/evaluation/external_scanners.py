"""Adapters for paper-method external skill security scanners.

The external tools use different report schemas.  This module keeps the
evaluation runner stable by converting their JSON/SARIF outputs into the
same report and prediction records used by the rest of SkillRecon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillrecon.evaluation.datasets import BaselinePredictionRecord
from skillrecon.evaluation.types import (
    CodeLocation,
    EvaluationFinding,
    EvaluationReport,
    ReportSummary,
)


@dataclass(frozen=True)
class ExternalScannerSpec:
    """Metadata and CLI shape for one external baseline scanner."""

    system_id: str
    display_name: str
    executable: str
    repo_url: str
    method_reference: str
    install_hint: str
    command_kind: str
    output_to_file: bool = True
    success_exit_codes: tuple[int, ...] = (0, 1)
    optional: bool = False
    notes: str = ""


EXTERNAL_SCANNER_SPECS: tuple[ExternalScannerSpec, ...] = (
    ExternalScannerSpec(
        system_id="baseline_skillfortify",
        display_name="SkillFortify",
        executable="skillfortify",
        repo_url="https://github.com/qualixar/skillfortify",
        method_reference=(
            "Formal Analysis and Supply Chain Security for Agentic AI Skills "
            "(arXiv:2603.00195)"
        ),
        install_hint="pip install skillfortify",
        command_kind="skillfortify",
        output_to_file=False,
        notes=(
            "Closest formal capability/supply-chain baseline; not clause-scope "
            "documentation reconciliation."
        ),
    ),
    ExternalScannerSpec(
        system_id="baseline_cisco_skill_scanner",
        display_name="Cisco Skill Scanner",
        executable="skill-scanner",
        repo_url="https://github.com/cisco-ai-defense/skill-scanner",
        method_reference=(
            "AI Agent Skill security scanner with static, behavioral, and "
            "optional LLM analyzers"
        ),
        install_hint="pip install cisco-ai-skill-scanner",
        command_kind="cisco_skill_scanner",
        notes=(
            "Strong security-scanner baseline; JSON/SARIF output can be "
            "normalized into SkillRecon predictions."
        ),
    ),
    ExternalScannerSpec(
        system_id="baseline_skillspector",
        display_name="SkillSpector",
        executable="skillspector",
        repo_url="https://github.com/NVIDIA/SkillSpector",
        method_reference=(
            "Agent Skills in the Wild: empirical skill vulnerability scanner "
            "implementation"
        ),
        install_hint=(
            "Clone https://github.com/NVIDIA/SkillSpector and install with "
            "Python 3.12+, or run its Docker image"
        ),
        command_kind="skillspector",
        notes=(
            "Static scanner baseline with documented JSON/SARIF outputs; "
            "Python 3.12+ requirement may make it optional in older reviewer "
            "environments."
        ),
    ),
    ExternalScannerSpec(
        system_id="baseline_snyk_agent_scan",
        display_name="Snyk Agent Scan",
        executable="uvx",
        repo_url="https://github.com/snyk/agent-scan",
        method_reference="Snyk AI agent, MCP, and skill security scanner",
        install_hint=(
            "uvx snyk-agent-scan@latest scan --json --skills <skill-path> "
            "(requires SNYK_TOKEN for normal operation)"
        ),
        command_kind="snyk_agent_scan",
        output_to_file=False,
        optional=True,
        notes=(
            "Product baseline that requires Snyk credentials; kept optional "
            "instead of a mandatory reviewer smoke dependency."
        ),
    ),
)

_SCANNER_BY_ID = {spec.system_id: spec for spec in EXTERNAL_SCANNER_SPECS}

_SEVERITY_ORDER = {
    "none": 0,
    "safe": 0,
    "ok": 0,
    "info": 1,
    "informational": 1,
    "note": 1,
    "warning": 2,
    "warn": 2,
    "low": 2,
    "medium": 3,
    "moderate": 3,
    "high": 4,
    "error": 4,
    "critical": 5,
    "unsafe": 5,
}

_TEXT_FIELDS = (
    "rule_id",
    "ruleId",
    "id",
    "check_id",
    "category",
    "type",
    "title",
    "name",
    "message",
    "ruleName",
    "description",
    "explanation",
    "finding",
    "details",
    "rationale",
    "intent",
    "recommendation",
    "remediation",
    "issue_code",
)


def list_external_scanner_specs() -> tuple[ExternalScannerSpec, ...]:
    """Return the external baseline scanner registry."""

    return EXTERNAL_SCANNER_SPECS


def get_external_scanner_spec(system_id: str) -> ExternalScannerSpec:
    """Resolve an external scanner spec by system id."""

    try:
        return _SCANNER_BY_ID[system_id]
    except KeyError as exc:
        known = ", ".join(sorted(_SCANNER_BY_ID))
        raise ValueError(f"Unknown external scanner {system_id!r}; known: {known}") from exc


def load_external_scanner_payload(path: Path) -> dict[str, Any] | list[Any]:
    """Load a scanner JSON/SARIF payload from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def external_scanner_payload_to_prediction(
    *,
    skill_id: str,
    system_id: str,
    payload: dict[str, Any] | list[Any],
    raw_output_ref: str | None = None,
) -> BaselinePredictionRecord:
    """Convert one raw scanner payload to a skill-level prediction record."""

    normalized = _normalize_payload(system_id=system_id, payload=payload)
    metadata: dict[str, object] = {
        "external_scanner": system_id,
        "finding_count": normalized["finding_count"],
        "exposure_count": normalized["exposure_count"],
        "max_severity": normalized["max_severity"],
        "source_schema": normalized["source_schema"],
    }
    if raw_output_ref is not None:
        metadata["raw_output"] = raw_output_ref

    first_violation = next(
        (finding for finding in normalized["findings"] if finding.main_label == "violation"),
        None,
    )
    subtype = first_violation.subtype if first_violation is not None else None
    rationale = _prediction_rationale(
        system_id=system_id,
        overall_label=str(normalized["overall_label"]),
        findings=normalized["findings"],
    )
    return BaselinePredictionRecord(
        skill_id=skill_id,
        system_id=system_id,
        main_label=normalized["overall_label"],
        subtype=subtype,
        rationale=rationale,
        score=normalized["score"],
        metadata=metadata,
    )


def external_scanner_payload_to_report(
    *,
    skill_id: str,
    system_id: str,
    payload: dict[str, Any] | list[Any],
    raw_output_ref: str | None = None,
) -> EvaluationReport:
    """Convert one raw scanner payload to a full normalized report."""

    normalized = _normalize_payload(system_id=system_id, payload=payload)
    violation_findings = [
        finding for finding in normalized["findings"] if finding.main_label == "violation"
    ]
    exposure_findings = [
        finding
        for finding in normalized["findings"]
        if finding.main_label == "exposure-only"
    ]
    config: dict[str, object] = {
        "external_scanner": system_id,
        "source_schema": normalized["source_schema"],
        "max_severity": normalized["max_severity"],
    }
    if raw_output_ref is not None:
        config["raw_output"] = raw_output_ref
    return EvaluationReport(
        skill_id=skill_id,
        system_id=system_id,
        analyzer_version=f"{system_id}-adapter-v1",
        config=config,
        overall_label=normalized["overall_label"],
        violation_findings=violation_findings,
        exposure_findings=exposure_findings,
        summary=ReportSummary(
            violation_count=len(violation_findings),
            exposure_count=len(exposure_findings),
        ),
    )


def build_external_scanner_command(
    spec: ExternalScannerSpec,
    *,
    skill_path: Path,
    output_path: Path,
) -> list[str]:
    """Build the offline CLI command for a scanner spec."""

    if spec.command_kind == "skillfortify":
        return [spec.executable, "scan", str(skill_path), "--format", "json"]
    if spec.command_kind == "cisco_skill_scanner":
        return [
            spec.executable,
            "scan",
            str(skill_path),
            "--lenient",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    if spec.command_kind == "skillspector":
        return [
            spec.executable,
            "scan",
            str(skill_path),
            "--no-llm",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    if spec.command_kind == "snyk_agent_scan":
        return [
            spec.executable,
            "snyk-agent-scan@latest",
            "scan",
            "--json",
            "--skills",
            str(skill_path),
        ]
    raise ValueError(f"Unsupported scanner command kind: {spec.command_kind}")


def _normalize_payload(
    *,
    system_id: str,
    payload: dict[str, Any] | list[Any],
) -> dict[str, Any]:
    findings = list(_payload_findings(system_id, payload))
    if not findings:
        synthetic = _synthetic_top_level_finding(system_id, payload)
        if synthetic is not None:
            findings.append(synthetic)

    violation_count = sum(1 for item in findings if item.main_label == "violation")
    exposure_count = sum(1 for item in findings if item.main_label == "exposure-only")
    overall_label = (
        "violation"
        if violation_count
        else "exposure-only"
        if exposure_count
        else "benign"
    )
    return {
        "overall_label": overall_label,
        "findings": findings,
        "finding_count": len(findings),
        "exposure_count": exposure_count,
        "max_severity": _payload_max_severity(payload, findings),
        "score": _payload_score(payload),
        "source_schema": _detect_schema(payload),
    }


def _payload_findings(
    system_id: str,
    payload: dict[str, Any] | list[Any],
) -> list[EvaluationFinding]:
    raw_findings = list(_iter_raw_findings(payload))
    normalized: list[EvaluationFinding] = []
    for index, raw in enumerate(raw_findings, start=1):
        finding = _normalize_raw_finding(system_id, raw, index)
        if finding is not None:
            normalized.append(finding)
    return normalized


def _iter_raw_findings(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    if _is_sarif(payload):
        return _iter_sarif_findings(payload)

    findings: list[dict[str, Any]] = []
    for key in (
        "filtered_findings",
        "findings",
        "issues",
        "results",
        "alerts",
        "detections",
        "threats",
        "vulnerabilities",
        "violations",
        "security_issues",
        "securityIssues",
        "security_results",
        "securityResults",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            findings.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            nested = value.get("findings") or value.get("issues") or value.get("results")
            if isinstance(nested, list):
                findings.extend(item for item in nested if isinstance(item, dict))

    for key in (
        "scan_results",
        "scanResults",
        "report",
        "analysis",
        "risk_assessment",
        "riskAssessment",
        "result",
    ):
        nested_payload = payload.get(key)
        if isinstance(nested_payload, (dict, list)):
            findings.extend(_iter_raw_findings(nested_payload))
    return _dedupe_raw_findings(findings)


def _iter_sarif_findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for run in payload.get("runs", []):
        if not isinstance(run, dict):
            continue
        for result in run.get("results", []):
            if not isinstance(result, dict):
                continue
            flattened = dict(result)
            message = result.get("message")
            if isinstance(message, dict):
                flattened["message"] = message.get("text") or message.get("markdown")
            locations = []
            for location in result.get("locations", []):
                if not isinstance(location, dict):
                    continue
                physical = location.get("physicalLocation", {})
                if not isinstance(physical, dict):
                    continue
                artifact = physical.get("artifactLocation", {})
                region = physical.get("region", {})
                locations.append(
                    {
                        "path": artifact.get("uri") if isinstance(artifact, dict) else None,
                        "line": region.get("startLine") if isinstance(region, dict) else None,
                    }
                )
            if locations:
                flattened["locations"] = locations
            findings.append(flattened)
    return findings


def _normalize_raw_finding(
    system_id: str,
    raw: dict[str, Any],
    index: int,
) -> EvaluationFinding | None:
    text = _finding_text(raw)
    severity = _severity_value(_raw_severity(raw))
    if severity == 0 and not text:
        return None

    main_label = _finding_label(raw, text, severity)
    subtype = _finding_subtype(text) if main_label == "violation" else None
    finding_id = _raw_id(raw) or f"{system_id}::finding-{index:04d}"
    return EvaluationFinding(
        finding_id=f"{system_id}::{finding_id}",
        main_label=main_label,
        subtype=subtype,
        capability_atoms=_capability_atoms(text),
        code_locations=_raw_locations(raw),
        rationale=_raw_rationale(raw, text, severity),
    )


def _finding_label(raw: dict[str, Any], text: str, severity: int) -> str:
    lowered = text.lower()
    if severity >= _SEVERITY_ORDER["high"]:
        return "violation"
    if any(
        token in lowered
        for token in (
            "underdeclared",
            "undeclared",
            "description-behavior mismatch",
            "behavior mismatch",
            "least privilege",
            "permission",
            "scope creep",
            "scope violation",
            "excessive agency",
        )
    ):
        return "violation"
    if raw.get("is_malicious") is True or raw.get("unsafe") is True:
        return "violation"
    return "exposure-only"


def _finding_subtype(text: str) -> str:
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            "chain",
            "taint",
            "pipeline",
            "flow",
            "source-to-sink",
            "tool misuse",
            "file read to network",
            "credential exfiltration chain",
        )
    ):
        return "unjustified_composition"
    if any(
        token in lowered
        for token in (
            "scope",
            "permission",
            "least privilege",
            "underdeclared",
            "undeclared",
            "wildcard",
            "overdeclared",
            "unbounded resource",
            "excessive agency",
        )
    ):
        return "scope_violation"
    return "unsupported_behavior"


def _capability_atoms(text: str) -> list[str]:
    lowered = text.lower()
    atoms: list[str] = []
    for token, atom in (
        ("network", "network"),
        ("external transmission", "network"),
        ("exfiltration", "network"),
        ("credential", "credential_access"),
        ("secret", "credential_access"),
        ("env", "credential_access"),
        ("file", "filesystem"),
        ("subprocess", "process_execution"),
        ("shell", "process_execution"),
        ("exec", "process_execution"),
        ("eval", "process_execution"),
        ("permission", "permission"),
        ("scope", "scope_boundary"),
        ("prompt injection", "prompt_injection"),
    ):
        if token in lowered and atom not in atoms:
            atoms.append(atom)
    return atoms


def _synthetic_top_level_finding(
    system_id: str,
    payload: dict[str, Any] | list[Any],
) -> EvaluationFinding | None:
    if not isinstance(payload, dict):
        return None
    severity = _severity_value(
        _first_string(
            payload,
            (
                "max_severity",
                "risk_severity",
                "severity",
                "status",
                "recommendation",
            ),
        )
    )
    score = _payload_score(payload)
    status_text = _top_level_text(payload).lower()
    if severity >= _SEVERITY_ORDER["high"] or (score is not None and score >= 51):
        label = "violation"
        subtype = _finding_subtype(status_text)
    elif (
        severity >= _SEVERITY_ORDER["medium"]
        or (score is not None and score >= 21)
        or any(token in status_text for token in ("warning", "caution", "review"))
    ):
        label = "exposure-only"
        subtype = None
    else:
        return None
    return EvaluationFinding(
        finding_id=f"{system_id}::top-level-risk",
        main_label=label,
        subtype=subtype,
        rationale=(
            "External scanner reported top-level risk without itemized findings: "
            f"{_top_level_text(payload)}"
        ),
    )


def _payload_max_severity(
    payload: dict[str, Any] | list[Any],
    findings: list[EvaluationFinding],
) -> str | None:
    if isinstance(payload, dict):
        top = _first_string(
            payload,
            ("max_severity", "risk_severity", "severity", "status"),
        )
        if top:
            return top
        for key in ("risk_assessment", "riskAssessment", "summary"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                nested_severity = _first_string(
                    nested,
                    ("max_severity", "risk_severity", "severity", "status"),
                )
                if nested_severity:
                    return nested_severity
    if not findings:
        return None
    label_order = {"violation": "high", "exposure-only": "low", "benign": "none"}
    return max((label_order.get(item.main_label, "none") for item in findings), key=_severity_value)


def _payload_score(payload: dict[str, Any] | list[Any]) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("risk_score", "score", "trust_score"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip().rstrip("%"))
            except ValueError:
                continue
    for key in ("risk_assessment", "riskAssessment", "summary"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            nested_score = _payload_score(nested)
            if nested_score is not None:
                return nested_score
    return None


def _detect_schema(payload: dict[str, Any] | list[Any]) -> str:
    if _is_sarif(payload):
        return "sarif"
    if isinstance(payload, dict):
        if "filtered_findings" in payload or "risk_score" in payload:
            return "skillspector-like"
        if "findings" in payload and ("status" in payload or "max_severity" in payload):
            return "scanner-findings"
        if "issues" in payload:
            return "scanner-issues"
    if isinstance(payload, list):
        return "finding-list"
    return "unknown-json"


def _is_sarif(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("runs"), list)


def _raw_id(raw: dict[str, Any]) -> str | None:
    value = (
        raw.get("id")
        or raw.get("rule_id")
        or raw.get("ruleId")
        or raw.get("check_id")
        or raw.get("issue_code")
    )
    return str(value) if value is not None else None


def _raw_severity(raw: dict[str, Any]) -> str | None:
    value = _first_string(
        raw,
        ("severity", "level", "risk_severity", "priority", "confidence", "impact"),
    )
    if value:
        return value
    properties = raw.get("properties")
    if isinstance(properties, dict):
        return _first_string(properties, ("severity", "level", "precision"))
    return None


def _raw_locations(raw: dict[str, Any]) -> list[CodeLocation]:
    locations: list[CodeLocation] = []
    raw_locations = raw.get("locations")
    if isinstance(raw_locations, list):
        for item in raw_locations:
            if not isinstance(item, dict):
                continue
            path = item.get("path") or item.get("file") or item.get("uri")
            if path:
                locations.append(CodeLocation(path=str(path), line=_int_or_none(item.get("line"))))
    for key in ("location", "sourceLocation", "source_location", "physicalLocation"):
        nested = raw.get(key)
        if not isinstance(nested, dict):
            continue
        artifact = nested.get("artifactLocation")
        region = nested.get("region")
        if isinstance(artifact, dict):
            nested_path = artifact.get("uri")
        else:
            nested_path = (
                nested.get("path")
                or nested.get("file")
                or nested.get("filename")
                or nested.get("file_path")
                or nested.get("uri")
            )
        if isinstance(region, dict):
            nested_line = region.get("startLine") or region.get("start_line")
        else:
            nested_line = (
                nested.get("line")
                or nested.get("line_number")
                or nested.get("startLine")
                or nested.get("start_line")
            )
        if nested_path:
            locations.append(
                CodeLocation(
                    path=str(nested_path),
                    line=_int_or_none(nested_line),
                )
            )
    path = (
        raw.get("path")
        or raw.get("file")
        or raw.get("filename")
        or raw.get("file_path")
        or raw.get("uri")
    )
    if path:
        locations.append(
            CodeLocation(
                path=str(path),
                line=_int_or_none(
                    raw.get("line")
                    or raw.get("line_number")
                    or raw.get("startLine")
                    or raw.get("start_line")
                ),
            )
        )
    return _dedupe_locations(locations)


def _raw_rationale(raw: dict[str, Any], text: str, severity: int) -> str:
    severity_name = _raw_severity(raw) or "unknown"
    if text:
        return f"External scanner finding ({severity_name}, rank={severity}): {text}"
    return f"External scanner finding with severity {severity_name}."


def _finding_text(raw: dict[str, Any]) -> str:
    fragments: list[str] = []
    for key in _TEXT_FIELDS:
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            value = value.get("text") or value.get("markdown") or value.get("message")
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text and text not in fragments:
                fragments.append(text)
    properties = raw.get("properties")
    if isinstance(properties, dict):
        fragments.append(_finding_text(properties))
    return " | ".join(fragment for fragment in fragments if fragment)


def _top_level_text(payload: dict[str, Any]) -> str:
    fragments: list[str] = []
    for key in (
        "status",
        "risk_severity",
        "risk_recommendation",
        "recommendation",
        "summary",
        "message",
    ):
        value = payload.get(key)
        if isinstance(value, (str, int, float)):
            fragments.append(str(value))
    return " | ".join(fragments) if fragments else "no top-level text"


def _prediction_rationale(
    *,
    system_id: str,
    overall_label: str,
    findings: list[EvaluationFinding],
) -> str:
    if not findings:
        return f"{system_id} produced no security findings after normalization."
    first = findings[0].rationale
    return (
        f"{system_id} normalized {len(findings)} finding(s); "
        f"skill-level label={overall_label}. First finding: {first}"
    )


def _severity_value(value: str | None) -> int:
    if value is None:
        return 0
    lowered = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if lowered in _SEVERITY_ORDER:
        return _SEVERITY_ORDER[lowered]
    tokens = set(lowered.split())
    for label, rank in sorted(
        _SEVERITY_ORDER.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        if label in tokens:
            return rank
    return 0


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (str, int, float)):
            return str(value)
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _dedupe_locations(locations: list[CodeLocation]) -> list[CodeLocation]:
    seen: set[tuple[str, int | None]] = set()
    deduped: list[CodeLocation] = []
    for location in locations:
        key = (location.path, location.line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(location)
    return deduped


def _dedupe_raw_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in findings:
        key = json.dumps(item, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped
