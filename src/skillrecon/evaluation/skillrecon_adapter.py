"""Adapter from current SkillRecon artifacts to the unified evaluation schema."""

from __future__ import annotations

import json
from pathlib import Path

from skillrecon.core.enums import FindingSupportLevel, FindingType, SupportStrength
from skillrecon.core.types import (
    CapabilityEvent,
    Diagnostic,
    Exposure,
    Finding,
    ResourceUse,
    RiskPath,
)
from skillrecon.evaluation.types import (
    CodeLocation,
    ContractQualityAlertRecord,
    EvaluationFinding,
    EvaluationReport,
    GraphStats,
    ReportGraphs,
    ReportSummary,
)

_VIOLATION_SUBTYPES = {
    FindingType.UNSUPPORTED_BEHAVIOR,
    FindingType.CONTRADICTED_BEHAVIOR,
    FindingType.SCOPE_VIOLATION,
    FindingType.UNJUSTIFIED_COMPOSITION,
}
_LOW_SIGNAL_CAPABILITIES = {
    "data_encode_send",
    "http_request",
    "dynamic_import",
    "shell_exec",
    "subprocess_spawn",
}
_CREDENTIAL_MARKERS = (
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "bearer_token",
    "client_secret",
    "secret_key",
    "private_key",
    "password",
    "credential",
)
_OPAQUE_INSTALL_MARKERS = (
    "base64",
    "archive_start",
    "self-extract",
    "self_extract",
    "opaque",
)
_NETWORK_CONTROL_MARKERS = (
    "remote-debugging",
    "remote_debugging",
    "0.0.0.0",
    "cdp",
    "tunnel",
    "relay",
    "daemon",
)
_SENSITIVE_ACTION_PATH_MARKERS = (
    "calendar_delete",
    "delete_tracked",
    "email_modify",
    "email_send",
    "transfer",
    "withdraw",
)
_LOCAL_STATE_SUFFIXES = (
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".env",
    ".db",
    ".sqlite",
    ".pkl",
    ".pickle",
    ".csv",
)


def build_skillrecon_report(skill_id: str, artifact_dir: Path) -> EvaluationReport:
    """Build a unified evaluation report from existing SkillRecon artifacts."""

    findings = _load_models(artifact_dir / "findings.json", Finding)
    exposures = _load_models(artifact_dir / "exposures.json", Exposure, allow_missing=True)
    diagnostics = _load_models(artifact_dir / "diagnostics.json", Diagnostic, allow_missing=True)
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)
    resources = _load_models(artifact_dir / "resource_table.json", ResourceUse, allow_missing=True)
    paths = _load_models(artifact_dir / "path_table.json", RiskPath, allow_missing=True)

    events_by_id = {event.event_id: event for event in events}
    resources_by_id = {resource.resource_id: resource for resource in resources}
    paths_by_id = {path.path_id: path for path in paths}

    violation_findings: list[EvaluationFinding] = []
    exposure_findings: list[EvaluationFinding] = []
    contract_quality_alerts: list[ContractQualityAlertRecord] = []

    for finding in findings:
        if finding.finding_type in _VIOLATION_SUBTYPES:
            evaluation_finding = _evaluation_finding_from_finding(
                finding=finding,
                main_label="violation",
                events_by_id=events_by_id,
                resources_by_id=resources_by_id,
                paths_by_id=paths_by_id,
            )
            if _finding_has_violation_evidence(
                finding,
                events_by_id=events_by_id,
                resources_by_id=resources_by_id,
                paths_by_id=paths_by_id,
            ):
                violation_findings.append(evaluation_finding)
            else:
                exposure_findings.append(
                    evaluation_finding.model_copy(update={"main_label": "exposure-only"})
                )
            continue

    for exposure in exposures:
        exposure_findings.append(
            EvaluationFinding(
                finding_id=exposure.exposure_id,
                main_label="exposure-only",
                subtype=exposure.exposure_type.value,
                certificate_ids=[],
                capability_atoms=[
                    events_by_id[event_id].capability
                    for event_id in exposure.related_event_ids
                    if event_id in events_by_id
                ],
                matched_clauses=exposure.related_clause_ids,
                code_locations=_locations_for_finding(
                    finding=exposure,
                    events_by_id=events_by_id,
                    resources_by_id=resources_by_id,
                    paths_by_id=paths_by_id,
                ),
                rationale=exposure.rationale,
            )
        )

    for diagnostic in diagnostics:
        if diagnostic.diagnostic_type.value == "policy_gap":
            contract_quality_alerts.append(
                ContractQualityAlertRecord(
                    alert_id=diagnostic.diagnostic_id,
                    alert_type=diagnostic.diagnostic_type.value,
                    affected_clauses=diagnostic.related_clause_ids,
                    related_event_ids=diagnostic.related_event_ids,
                    code_locations=_locations_for_finding(
                        finding=diagnostic,
                        events_by_id=events_by_id,
                        resources_by_id=resources_by_id,
                        paths_by_id=paths_by_id,
                    ),
                    rationale=diagnostic.rationale,
                )
            )

    overall_label = "benign"
    if violation_findings:
        overall_label = "violation"
    elif exposure_findings:
        overall_label = "exposure-only"

    return EvaluationReport(
        skill_id=skill_id,
        graphs=ReportGraphs(
            g_d=_graph_stats(artifact_dir / "g_d.json"),
            g_c=_graph_stats(artifact_dir / "g_c.json"),
            g_x=_graph_stats(artifact_dir / "g_x.json"),
        ),
        overall_label=overall_label,
        violation_findings=violation_findings,
        exposure_findings=exposure_findings,
        contract_quality_alerts=contract_quality_alerts,
        summary=ReportSummary(
            violation_count=len(violation_findings),
            exposure_count=len(exposure_findings),
            contract_quality_count=len(contract_quality_alerts),
        ),
        permission_manifest_ref=(
            "permission_manifest.json"
            if (artifact_dir / "permission_manifest.json").exists()
            else ""
        ),
    )


def _evaluation_finding_from_finding(
    *,
    finding: Finding,
    main_label: str,
    events_by_id: dict[str, CapabilityEvent],
    resources_by_id: dict[str, ResourceUse],
    paths_by_id: dict[str, RiskPath],
) -> EvaluationFinding:
    return EvaluationFinding(
        finding_id=finding.finding_id,
        main_label=main_label,
        subtype=finding.finding_type.value,
        certificate_ids=finding.certificate_ids,
        capability_atoms=_capabilities_for_finding(finding, events_by_id),
        matched_clauses=finding.related_clause_ids,
        code_locations=_locations_for_finding(
            finding=finding,
            events_by_id=events_by_id,
            resources_by_id=resources_by_id,
            paths_by_id=paths_by_id,
        ),
        rationale=finding.rationale,
    )


def _finding_has_violation_evidence(
    finding: Finding,
    *,
    events_by_id: dict[str, CapabilityEvent],
    resources_by_id: dict[str, ResourceUse],
    paths_by_id: dict[str, RiskPath],
) -> bool:
    events = [
        events_by_id[event_id]
        for event_id in finding.related_event_ids
        if event_id in events_by_id
    ]

    if _has_high_risk_code_evidence(finding, events, resources_by_id, paths_by_id):
        return True

    if finding.support_level == FindingSupportLevel.GRAPH_BACKED and finding.support_strength in {
        SupportStrength.STRONG,
        SupportStrength.MIXED,
        SupportStrength.WEAK_ORCHESTRATED,
        SupportStrength.WEAK_STRUCTURAL,
    }:
        return any(
            _is_code_event(event)
            and event.capability not in _LOW_SIGNAL_CAPABILITIES
            for event in events
        )

    return False


def _has_high_risk_code_evidence(
    finding: Finding,
    events: list[CapabilityEvent],
    resources_by_id: dict[str, ResourceUse],
    paths_by_id: dict[str, RiskPath],
) -> bool:
    code_events = [event for event in events if _is_code_event(event)]
    if not code_events:
        return False

    for event in code_events:
        event_text = _event_text(event)
        if event.capability == "eval_exec":
            return True
        if event.capability == "env_var_read" and _contains_credential_marker(event_text):
            return True
        if event.capability == "deserialization" and "pickle" in event_text:
            return True
        if _is_opaque_installer_event(event_text):
            return True
        if _is_local_service_bridge(event):
            return True
        if _has_sensitive_action_path(event):
            return True
        if any(marker in event_text for marker in _NETWORK_CONTROL_MARKERS):
            return True

    if _has_local_state_to_remote_flow(code_events):
        return True

    linked_text = _linked_structural_text(finding, resources_by_id, paths_by_id)
    if linked_text and any(marker in linked_text for marker in _NETWORK_CONTROL_MARKERS):
        return True

    return False


def _is_code_event(event: CapabilityEvent) -> bool:
    path = event.file_path or event.location.rsplit(":", 1)[0]
    path_lower = path.lower()
    if not path_lower or path_lower.startswith(".skillrecon_synthetic/"):
        return False
    if event.tier == "instruction":
        return False
    if path_lower.endswith((".md", ".json", ".yaml", ".yml", ".txt", ".rst")):
        return False
    parts = path_lower.replace("\\", "/").split("/")
    if "tests" in parts or any(part.startswith("test_") for part in parts):
        return False
    if path_lower.rsplit("/", 1)[-1].startswith("test_"):
        return False
    return True


def _event_text(event: CapabilityEvent) -> str:
    return " ".join(
        [
            event.capability,
            event.api_call,
            event.detail,
            event.file_path,
            event.location,
            *event.arguments,
        ]
    ).lower()


def _is_opaque_installer_event(event_text: str) -> bool:
    if any(marker in event_text for marker in _OPAQUE_INSTALL_MARKERS if marker != "base64"):
        return True
    return "base64" in event_text and any(
        marker in event_text
        for marker in ("archive_start", "skill_dir", "installer", "self_extract")
    )


def _is_local_service_bridge(event: CapabilityEvent) -> bool:
    if event.capability not in {"http_request", "data_encode_send", "dynamic_import"}:
        return False
    event_text = _event_text(event)
    return "localhost" in event_text or "127.0.0.1" in event_text


def _has_sensitive_action_path(event: CapabilityEvent) -> bool:
    path = (event.file_path or event.location.rsplit(":", 1)[0]).lower()
    return any(marker in path for marker in _SENSITIVE_ACTION_PATH_MARKERS)


def _contains_credential_marker(text: str) -> bool:
    normalized = text.lower().replace("-", "_")
    return any(marker in normalized for marker in _CREDENTIAL_MARKERS)


def _has_local_state_to_remote_flow(events: list[CapabilityEvent]) -> bool:
    dynamic_remote = False
    local_state = False
    for event in events:
        if event.capability != "data_encode_send":
            continue
        args = [arg.lower() for arg in event.arguments]
        dynamic_remote = dynamic_remote or any(
            "<dynamic-url>" in arg or arg.startswith(("http://", "https://"))
            for arg in args
        )
        local_state = local_state or any(
            arg.endswith(_LOCAL_STATE_SUFFIXES) and "<dynamic-url>" not in arg
            for arg in args
        )
    return dynamic_remote and local_state


def _linked_structural_text(
    finding: Finding,
    resources_by_id: dict[str, ResourceUse],
    paths_by_id: dict[str, RiskPath],
) -> str:
    parts: list[str] = []
    for resource_id in finding.related_resource_ids:
        resource = resources_by_id.get(resource_id)
        if resource is None:
            continue
        parts.extend(
            [
                resource.resource_type,
                resource.value,
                resource.location,
                resource.origin_kind,
                resource.origin_hint,
            ]
        )
    for path_id in finding.related_path_ids:
        path = paths_by_id.get(path_id)
        if path is None:
            continue
        for segment in [path.source, path.sink, *path.segments]:
            parts.extend(
                [
                    segment.location,
                    segment.label,
                    segment.slot_kind,
                    segment.symbol_hint,
                    segment.expr_hint,
                ]
            )
    return " ".join(parts).lower()


def _load_models(path: Path, model_cls: type, *, allow_missing: bool = False) -> list:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list artifact in {path}")
    return [model_cls.model_validate(item) for item in payload]


def _capabilities_for_finding(
    finding: Finding,
    events_by_id: dict[str, CapabilityEvent],
) -> list[str]:
    capabilities = [
        events_by_id[event_id].capability
        for event_id in finding.related_event_ids
        if event_id in events_by_id
    ]
    return list(dict.fromkeys(capabilities))


def _locations_for_finding(
    *,
    finding: Finding,
    events_by_id: dict[str, CapabilityEvent],
    resources_by_id: dict[str, ResourceUse],
    paths_by_id: dict[str, RiskPath],
) -> list[CodeLocation]:
    locations: list[CodeLocation] = []

    for event_id in finding.related_event_ids:
        event = events_by_id.get(event_id)
        if event is None:
            continue
        locations.extend(_event_locations(event))

    for resource_id in finding.related_resource_ids:
        resource = resources_by_id.get(resource_id)
        if resource is None:
            continue
        location = _parse_location(resource.location)
        if location is not None:
            locations.append(location)

    for path_id in finding.related_path_ids:
        path = paths_by_id.get(path_id)
        if path is None:
            continue
        locations.extend(_path_locations(path))

    return _dedupe_locations(locations)


def _event_locations(event: CapabilityEvent) -> list[CodeLocation]:
    if event.file_path:
        return [CodeLocation(path=event.file_path, line=event.line or None)]
    location = _parse_location(event.location)
    return [location] if location is not None else []


def _path_locations(path: RiskPath) -> list[CodeLocation]:
    locations: list[CodeLocation] = []
    for segment in [path.source, path.sink, *path.segments]:
        location = _parse_location(segment.location)
        if location is not None:
            locations.append(location)
    return _dedupe_locations(locations)


def _parse_location(location: str) -> CodeLocation | None:
    if not location:
        return None
    path_part, _, line_part = location.rpartition(":")
    if path_part and line_part.isdigit():
        return CodeLocation(path=path_part, line=int(line_part))
    return CodeLocation(path=location, line=None)


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


def _graph_stats(path: Path) -> GraphStats:
    if not path.exists():
        return GraphStats()
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", []) if isinstance(payload, dict) else []
    edges = payload.get("edges", []) if isinstance(payload, dict) else []
    return GraphStats(node_count=len(nodes), edge_count=len(edges))
