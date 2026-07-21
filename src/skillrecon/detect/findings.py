"""Finding materialization for Module 04 witness engine."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from skillrecon.core.enums import (
    CertificateKind,
    DiagnosticType,
    ExposureType,
    FindingSupportLevel,
    FindingType,
    JudgmentKind,
    PredicateResult,
    RelationKind,
    SupportStrength,
)
from skillrecon.core.sensitivity import event_capability_family, event_requires_authorization
from skillrecon.core.types import (
    CapabilityEvent,
    Certificate,
    Clause,
    Diagnostic,
    Exposure,
    Finding,
    ResourceUse,
    ReconciliationEdge,
    ReconciliationJudgment,
    RiskPath,
)

_PLANNING_SENSITIVE_CAPABILITIES = {
    "api_key_use",
    "credential_store_access",
    "cron_schedule",
    "data_encode_send",
    "data_upload",
    "deserialization",
    "dynamic_import",
    "env_var_read",
    "file_delete",
    "file_permission_change",
    "keychain_access",
    "shell_exec",
    "sql_exec",
    "ssh_connect",
    "subprocess_spawn",
    "token_file_read",
}


def load_a_req(taxonomy_path: Path) -> set[str]:
    """Load the authorization-sensitive capability subset from taxonomy."""
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return set(data.get("a_req", []))


def materialize_findings(
    judgments: list[ReconciliationJudgment],
    certificates: list[Certificate],
    events: list[CapabilityEvent],
    paths: list[RiskPath],
    taxonomy_path: Path,
    projection_edges: list[ReconciliationEdge] | None = None,
) -> list[Finding]:
    """Materialize confirmed declaration-implementation inconsistency findings."""
    a_req = load_a_req(taxonomy_path)
    events_by_id = {event.event_id: event for event in events}
    paths_by_id = {path.path_id: path for path in paths}
    projection_edges = projection_edges or []
    findings: list[Finding] = []
    counter = 0
    directly_flagged_event_ids = _directly_flagged_event_ids(
        certificates=certificates,
        events_by_id=events_by_id,
        a_req=a_req,
    )

    for certificate in certificates:
        subject_set = set(certificate.subject_refs)
        event_ids = [event_id for event_id in subject_set if event_id in events_by_id]
        path_ids = [path_id for path_id in subject_set if path_id in paths_by_id]
        resource_ids = [
            ref
            for ref in certificate.subject_refs
            if ref.startswith("r") and ref not in event_ids and ref not in path_ids
        ]
        constraint_ids = [
            ref
            for ref in certificate.subject_refs
            if ref.startswith("cst") and ref not in event_ids
        ]

        if certificate.kind == CertificateKind.NO_SUPPORT:
            sensitive_event_ids = [
                event_id
                for event_id in event_ids
                if event_id in events_by_id
                and event_requires_authorization(events_by_id[event_id], a_req)
            ]
            if sensitive_event_ids:
                counter += 1
                findings.append(
                    Finding(
                        finding_id=f"finding-{counter:04d}",
                        finding_type=FindingType.UNSUPPORTED_BEHAVIOR,
                        support_level=FindingSupportLevel.DIAGNOSTIC,
                        support_strength=SupportStrength.NONE,
                        certificate_ids=[certificate.certificate_id],
                        supporting_judgment_ids=list(certificate.supporting_judgment_ids),
                        related_event_ids=sensitive_event_ids,
                        rationale=(
                            "Observed behavior has no supporting declaration path in the "
                            "admitted policy graph."
                        ),
                    )
                )
        elif certificate.kind == CertificateKind.BLOCKING_PROHIBITION:
            counter += 1
            findings.append(
                Finding(
                    finding_id=f"finding-{counter:04d}",
                    finding_type=FindingType.CONTRADICTED_BEHAVIOR,
                    support_level=_support_level_for_blocking_prohibition(
                        certificate.subject_refs,
                        projection_edges,
                    ),
                    support_strength=_support_strength_for_blocking_prohibition(
                        certificate.subject_refs,
                        projection_edges,
                    ),
                    certificate_ids=[certificate.certificate_id],
                    supporting_judgment_ids=list(certificate.supporting_judgment_ids),
                    supporting_edge_ids=_supporting_edge_ids_for_blocking_prohibition(
                        certificate.subject_refs,
                        projection_edges,
                    ),
                    related_clause_ids=sorted(
                        ref for ref in certificate.subject_refs if ref.startswith("c")
                    ),
                    related_event_ids=event_ids,
                    rationale="Observed behavior contradicts an explicit prohibited rule.",
                )
            )
        elif certificate.kind == CertificateKind.SCOPE_FAILURE:
            counter += 1
            findings.append(
                Finding(
                    finding_id=f"finding-{counter:04d}",
                    finding_type=FindingType.SCOPE_VIOLATION,
                    support_level=_support_level_for_scope_failure(
                        constraint_ids,
                        resource_ids,
                        projection_edges,
                    ),
                    support_strength=_support_strength_for_scope_failure(
                        constraint_ids,
                        resource_ids,
                        projection_edges,
                    ),
                    certificate_ids=[certificate.certificate_id],
                    supporting_judgment_ids=list(certificate.supporting_judgment_ids),
                    supporting_edge_ids=_supporting_edge_ids_for_scope_failure(
                        constraint_ids,
                        resource_ids,
                        projection_edges,
                    ),
                    related_constraint_ids=constraint_ids,
                    related_resource_ids=resource_ids,
                    rationale="Observed resource usage violates an explicit documented scope.",
                )
            )
        elif certificate.kind == CertificateKind.NO_JUSTIFIED_PATH:
            path_ids = [
                path_id for path_id in certificate.subject_refs if path_id in paths_by_id
            ]
            sensitive_path_ids = [
                path_id
                for path_id in path_ids
                if _path_requires_composition_review(
                    path=paths_by_id[path_id],
                    events_by_id=events_by_id,
                    a_req=a_req,
                    directly_flagged_event_ids=directly_flagged_event_ids,
                )
            ]
            if not sensitive_path_ids:
                continue
            counter += 1
            findings.append(
                Finding(
                    finding_id=f"finding-{counter:04d}",
                    finding_type=FindingType.UNJUSTIFIED_COMPOSITION,
                    support_level=FindingSupportLevel.DIAGNOSTIC,
                    support_strength=SupportStrength.NONE,
                    certificate_ids=[certificate.certificate_id],
                    supporting_judgment_ids=list(certificate.supporting_judgment_ids),
                    related_path_ids=sensitive_path_ids,
                    rationale=(
                        "Recovered implementation composition lacks a justifying "
                        "declaration chain."
                    ),
                )
            )

    return _aggregate_findings(findings, events_by_id)


def materialize_exposures(
    *,
    judgments: list[ReconciliationJudgment],
    certificates: list[Certificate],
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    paths: list[RiskPath],
    taxonomy_path: Path,
    projection_edges: list[ReconciliationEdge] | None = None,
) -> list[Exposure]:
    """Materialize declared high-risk exposures separate from violations."""
    a_req = load_a_req(taxonomy_path)
    projection_edges = projection_edges or []
    events_by_id = {event.event_id: event for event in events}
    resources_by_id = {resource.resource_id: resource for resource in resources}
    paths_by_id = {path.path_id: path for path in paths}
    judgments_by_kind: dict[JudgmentKind, list[ReconciliationJudgment]] = {}
    for judgment in judgments:
        judgments_by_kind.setdefault(judgment.kind, []).append(judgment)

    directly_flagged_event_ids = _directly_flagged_event_ids(
        certificates=certificates,
        events_by_id=events_by_id,
        a_req=a_req,
    )
    scope_violated_event_ids = _scope_violated_event_ids(
        resources_by_id=resources_by_id,
        projection_edges=projection_edges,
    )
    blocked_event_ids = directly_flagged_event_ids | scope_violated_event_ids

    exposures: list[Exposure] = []
    counter = 0

    event_support_edges: dict[str, list[ReconciliationEdge]] = {}
    for edge in projection_edges:
        if edge.relation != RelationKind.SUPPORTS or edge.event_id is None:
            continue
        event_support_edges.setdefault(edge.event_id, []).append(edge)

    for event_id, support_edges in event_support_edges.items():
        event = events_by_id.get(event_id)
        if event is None:
            continue
        if event.tier == "instruction":
            continue
        if event_id in blocked_event_ids:
            continue
        if not event_requires_authorization(event, a_req):
            continue
        counter += 1
        exposures.append(
            Exposure(
                exposure_id=f"exposure-{counter:04d}",
                exposure_type=ExposureType.DECLARED_SENSITIVE_BEHAVIOR,
                support_level=FindingSupportLevel.GRAPH_BACKED,
                support_strength=_collapse_support_strength(
                    [edge.support_strength for edge in support_edges]
                ),
                supporting_judgment_ids=_merge_str_lists(
                    edge.source_judgment_ids for edge in support_edges
                ),
                supporting_edge_ids=_merge_str_lists(
                    [[edge.edge_id] for edge in support_edges if edge.edge_id]
                ),
                related_clause_ids=_merge_str_lists(
                    [[edge.clause_id] for edge in support_edges if edge.clause_id]
                ),
                related_event_ids=[event_id],
                rationale=(
                    "Documented high-risk behavior is explicitly supported by the "
                    "policy graph and should be reviewed as declared exposure."
                ),
            )
        )

    for judgment in judgments_by_kind.get(JudgmentKind.PATH_JUSTIFICATION, []):
        if judgment.result != PredicateResult.TRUE:
            continue
        path_ids = [ref for ref in judgment.subject_refs if ref in paths_by_id]
        if not path_ids:
            continue
        path_id = path_ids[0]
        path = paths_by_id[path_id]
        sensitive_event_ids = _path_sensitive_event_ids(path, events_by_id, a_req)
        if not sensitive_event_ids:
            continue
        if any(event_id in blocked_event_ids for event_id in sensitive_event_ids):
            continue
        counter += 1
        exposures.append(
            Exposure(
                exposure_id=f"exposure-{counter:04d}",
                exposure_type=ExposureType.DECLARED_SENSITIVE_COMPOSITION,
                support_level=FindingSupportLevel.DIAGNOSTIC,
                support_strength=SupportStrength.NONE,
                supporting_judgment_ids=[judgment.judgment_id],
                related_event_ids=sensitive_event_ids,
                related_path_ids=[path_id],
                rationale=(
                    "Documented high-risk composition remains justified end-to-end "
                    "and should be reviewed as declared exposure."
                ),
            )
        )

    return _aggregate_exposures(exposures, events_by_id, paths_by_id)


def materialize_diagnostics(
    judgments: list[ReconciliationJudgment],
    certificates: list[Certificate],
    events: list[CapabilityEvent],
    paths: list[RiskPath],
    taxonomy_path: Path,
    clauses: list[Clause] | None = None,
    projection_edges: list[ReconciliationEdge] | None = None,
) -> list[Diagnostic]:
    """Materialize auxiliary diagnostic signals that are not confirmed findings."""
    a_req = load_a_req(taxonomy_path)
    clauses_by_id = {clause.clause_id: clause for clause in clauses or []}
    events_by_id = {event.event_id: event for event in events}
    paths_by_id = {path.path_id: path for path in paths}
    projection_edges = projection_edges or []
    diagnostics: list[Diagnostic] = []
    counter = 0
    directly_flagged_event_ids = _directly_flagged_event_ids(
        certificates=certificates,
        events_by_id=events_by_id,
        a_req=a_req,
    )

    for certificate in certificates:
        if certificate.kind != CertificateKind.UNRESOLVED_ROUTE:
            continue
        path_ids = [
            path_id for path_id in certificate.subject_refs if path_id in paths_by_id
        ]
        path_ids = [
            path_id
            for path_id in path_ids
                if _path_requires_route_attention(
                    paths_by_id[path_id],
                    events_by_id,
                    a_req,
                    directly_flagged_event_ids,
                )
        ]
        if not path_ids:
            continue
        counter += 1
        diagnostics.append(
            Diagnostic(
                diagnostic_id=f"diagnostic-{counter:04d}",
                diagnostic_type=DiagnosticType.UNRESOLVED_ROUTE,
                certificate_ids=[certificate.certificate_id],
                supporting_judgment_ids=list(certificate.supporting_judgment_ids),
                related_path_ids=path_ids,
                rationale=(
                    "Recovered path exists, but its declaration-side route remains "
                    "unresolved."
                ),
            )
        )

    for judgment in judgments:
        if (
            judgment.kind != JudgmentKind.AUTHORIZATION
            or judgment.result != PredicateResult.ABSTAIN
        ):
            continue
        if not _judgment_requires_policy_gap(judgment, events_by_id, a_req):
            continue
        diagnostic_type = (
            DiagnosticType.ALIGNMENT_MISMATCH
            if judgment.abstain_reason == "alignment_mismatch"
            else DiagnosticType.POLICY_GAP
        )
        rationale = (
            "The observed behavior matches a declaration semantically, but its recovered "
            "implementation unit diverges from the step-aligned code evidence."
            if diagnostic_type == DiagnosticType.ALIGNMENT_MISMATCH
            else (
                "A declaration-to-implementation alignment candidate remains policy-"
                "underspecified or unresolved."
            )
        )
        counter += 1
        if diagnostic_type == DiagnosticType.ALIGNMENT_MISMATCH:
            supporting_edge_ids = _supporting_edge_ids_for_alignment_mismatch(
                related_clause_ids=sorted(
                    ref for ref in judgment.subject_refs if ref.startswith("c")
                ),
                clauses_by_id=clauses_by_id,
                projection_edges=projection_edges,
            )
            support_level = (
                FindingSupportLevel.GRAPH_BACKED
                if supporting_edge_ids
                else FindingSupportLevel.DIAGNOSTIC
            )
            support_strength = _collapse_support_strength(
                [
                    edge.support_strength
                    for edge in projection_edges
                    if edge.edge_id in supporting_edge_ids
                ]
            )
        else:
            supporting_edge_ids = _supporting_edge_ids_for_contract_quality_alert(
                judgment,
                projection_edges,
            )
            support_level = _support_level_for_contract_quality_alert(
                judgment=judgment,
                projection_edges=projection_edges,
            )
            support_strength = _support_strength_for_contract_quality_alert(
                judgment=judgment,
                projection_edges=projection_edges,
            )
        diagnostics.append(
            Diagnostic(
                diagnostic_id=f"diagnostic-{counter:04d}",
                diagnostic_type=diagnostic_type,
                support_level=support_level,
                support_strength=support_strength,
                supporting_judgment_ids=[judgment.judgment_id],
                supporting_edge_ids=supporting_edge_ids,
                related_clause_ids=sorted(
                    ref for ref in judgment.subject_refs if ref.startswith("c")
                ),
                related_event_ids=sorted(
                    ref for ref in judgment.subject_refs if ref in events_by_id
                ),
                rationale=rationale,
            )
        )

    for judgment in judgments:
        if (
            judgment.kind != JudgmentKind.PATH_JUSTIFICATION
            or judgment.result != PredicateResult.ABSTAIN
            or judgment.abstain_reason != "partial_path_coverage"
        ):
            continue
        related_path_ids = [
            ref for ref in judgment.subject_refs if ref in paths_by_id
        ]
        if not related_path_ids:
            continue
        related_event_ids = _event_ids_from_path_partial_note(judgment.notes)
        counter += 1
        diagnostics.append(
            Diagnostic(
                diagnostic_id=f"diagnostic-{counter:04d}",
                diagnostic_type=DiagnosticType.PARTIAL_PATH_COVERAGE,
                support_level=_support_level_for_partial_path_coverage(
                    related_event_ids=related_event_ids,
                    projection_edges=projection_edges,
                ),
                support_strength=_support_strength_for_partial_path_coverage(
                    related_event_ids=related_event_ids,
                    projection_edges=projection_edges,
                ),
                supporting_judgment_ids=[judgment.judgment_id],
                supporting_edge_ids=_supporting_edge_ids_for_partial_path_coverage(
                    related_event_ids=related_event_ids,
                    projection_edges=projection_edges,
                ),
                related_path_ids=related_path_ids,
                related_event_ids=related_event_ids,
                rationale=(
                    "The recovered path is partially justified, but one or more "
                    "authorization-sensitive segments remain only weakly covered."
                ),
            )
        )

    diagnostics.extend(
        _materialize_planning_gap_diagnostics(
            events=events,
            a_req=a_req,
            projection_edges=projection_edges,
            existing_diagnostics=diagnostics,
            starting_index=counter,
        )
    )
    return _aggregate_diagnostics(diagnostics, events_by_id, paths_by_id)


def _supporting_edge_ids_for_scope_failure(
    constraint_ids: list[str],
    resource_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> list[str]:
    constraint_set = set(constraint_ids)
    resource_set = set(resource_ids)
    return [
        edge.edge_id
        for edge in projection_edges
        if edge.relation == RelationKind.SCOPE_VIOLATES
        and edge.constraint_id in constraint_set
        and edge.resource_id in resource_set
    ]


def _supporting_edge_ids_for_blocking_prohibition(
    subject_refs: list[str],
    projection_edges: list[ReconciliationEdge],
) -> list[str]:
    clause_ids = {ref for ref in subject_refs if ref.startswith("c")}
    event_ids = {ref for ref in subject_refs if ref.startswith("e")}
    return [
        edge.edge_id
        for edge in projection_edges
        if edge.relation == RelationKind.CONTRADICTS
        and edge.clause_id in clause_ids
        and edge.event_id in event_ids
    ]


def _support_level_for_blocking_prohibition(
    subject_refs: list[str],
    projection_edges: list[ReconciliationEdge],
) -> FindingSupportLevel:
    return (
        FindingSupportLevel.GRAPH_BACKED
        if _supporting_edge_ids_for_blocking_prohibition(subject_refs, projection_edges)
        else FindingSupportLevel.DIAGNOSTIC
    )


def _support_strength_for_blocking_prohibition(
    subject_refs: list[str],
    projection_edges: list[ReconciliationEdge],
) -> SupportStrength:
    return _collapse_support_strength(
        [
            edge.support_strength
            for edge in projection_edges
            if edge.edge_id
            in _supporting_edge_ids_for_blocking_prohibition(
                subject_refs,
                projection_edges,
            )
        ]
    )


def _support_level_for_scope_failure(
    constraint_ids: list[str],
    resource_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> FindingSupportLevel:
    return (
        FindingSupportLevel.GRAPH_BACKED
        if _supporting_edge_ids_for_scope_failure(
            constraint_ids,
            resource_ids,
            projection_edges,
        )
        else FindingSupportLevel.DIAGNOSTIC
    )


def _support_strength_for_scope_failure(
    constraint_ids: list[str],
    resource_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> SupportStrength:
    return _collapse_support_strength(
        [
            edge.support_strength
            for edge in projection_edges
            if edge.edge_id
            in _supporting_edge_ids_for_scope_failure(
                constraint_ids,
                resource_ids,
                projection_edges,
            )
        ]
    )


def _supporting_edge_ids_for_contract_quality_alert(
    judgment: ReconciliationJudgment,
    projection_edges: list[ReconciliationEdge],
) -> list[str]:
    return [
        edge.edge_id
        for edge in projection_edges
        if edge.relation == RelationKind.POTENTIALLY_SUPPORTS
        and judgment.judgment_id in edge.source_judgment_ids
    ]


def _supporting_edge_ids_for_alignment_mismatch(
    *,
    related_clause_ids: list[str],
    clauses_by_id: dict[str, Clause],
    projection_edges: list[ReconciliationEdge],
) -> list[str]:
    step_ids: set[str] = set()
    for clause_id in related_clause_ids:
        clause = clauses_by_id.get(clause_id)
        if clause is None:
            continue
        step_ids.update(clause.step_ids)
    if not step_ids:
        return []
    return [
        edge.edge_id
        for edge in projection_edges
        if edge.relation == RelationKind.ALIGNS
        and edge.step_id in step_ids
    ]


def _supporting_edge_ids_for_partial_path_coverage(
    *,
    related_event_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> list[str]:
    event_id_set = set(related_event_ids)
    return [
        edge.edge_id
        for edge in projection_edges
        if edge.event_id in event_id_set
        and edge.relation in {RelationKind.POTENTIALLY_SUPPORTS, RelationKind.RELATES_TO}
    ]


def _support_level_for_contract_quality_alert(
    judgment: ReconciliationJudgment,
    projection_edges: list[ReconciliationEdge],
) -> FindingSupportLevel:
    return (
        FindingSupportLevel.GRAPH_BACKED
        if _supporting_edge_ids_for_contract_quality_alert(judgment, projection_edges)
        else FindingSupportLevel.DIAGNOSTIC
    )


def _support_strength_for_contract_quality_alert(
    judgment: ReconciliationJudgment,
    projection_edges: list[ReconciliationEdge],
) -> SupportStrength:
    return _collapse_support_strength(
        [
            edge.support_strength
            for edge in projection_edges
            if edge.edge_id
            in _supporting_edge_ids_for_contract_quality_alert(
                judgment,
                projection_edges,
            )
        ]
    )


def _support_level_for_partial_path_coverage(
    *,
    related_event_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> FindingSupportLevel:
    return (
        FindingSupportLevel.GRAPH_BACKED
        if _supporting_edge_ids_for_partial_path_coverage(
            related_event_ids=related_event_ids,
            projection_edges=projection_edges,
        )
        else FindingSupportLevel.DIAGNOSTIC
    )


def _support_strength_for_partial_path_coverage(
    *,
    related_event_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> SupportStrength:
    return _collapse_support_strength(
        [
            edge.support_strength
            for edge in projection_edges
            if edge.edge_id
            in _supporting_edge_ids_for_partial_path_coverage(
                related_event_ids=related_event_ids,
                projection_edges=projection_edges,
            )
        ]
    )


def _collapse_support_strength(
    strengths: list[SupportStrength | None],
) -> SupportStrength:
    observed = {strength for strength in strengths if strength is not None}
    if not observed:
        return SupportStrength.NONE
    if len(observed) == 1:
        return observed.pop()
    return SupportStrength.MIXED


def _materialize_planning_gap_diagnostics(
    *,
    events: list[CapabilityEvent],
    a_req: set[str],
    projection_edges: list[ReconciliationEdge],
    existing_diagnostics: list[Diagnostic],
    starting_index: int,
) -> list[Diagnostic]:
    covered_event_ids = {
        event_id
        for diagnostic in existing_diagnostics
        for event_id in diagnostic.related_event_ids
    }
    event_edges: dict[str, list[ReconciliationEdge]] = {}
    for edge in projection_edges:
        if edge.event_id is None:
            continue
        event_edges.setdefault(edge.event_id, []).append(edge)

    diagnostics: list[Diagnostic] = []
    counter = starting_index
    for event in events:
        if event.event_id in covered_event_ids:
            continue
        if event.tier != "instruction":
            continue
        if event.capability not in _PLANNING_SENSITIVE_CAPABILITIES:
            continue
        if not event_requires_authorization(event, a_req):
            continue
        edges = event_edges.get(event.event_id, [])
        if any(
            edge.relation in {RelationKind.SUPPORTS, RelationKind.CONTRADICTS}
            for edge in edges
        ):
            continue
        if edges:
            counter += 1
            diagnostics.append(
                Diagnostic(
                    diagnostic_id=f"diagnostic-{counter:04d}",
                    diagnostic_type=DiagnosticType.PLANNED_BEHAVIOR_GAP,
                    support_level=FindingSupportLevel.GRAPH_BACKED,
                    support_strength=_collapse_support_strength(
                        [edge.support_strength for edge in edges]
                    ),
                    supporting_judgment_ids=sorted(
                        {
                            edge.source_judgment_ids[0]
                            for edge in edges
                            if edge.source_judgment_ids
                        }
                    ),
                    supporting_edge_ids=sorted({edge.edge_id for edge in edges}),
                    related_event_ids=[event.event_id],
                    rationale=(
                        "Instruction-level declared behavior references a sensitive action, "
                        "but no definitive policy support or contradiction was recovered."
                    ),
                )
            )
            continue
        counter += 1
        diagnostics.append(
            Diagnostic(
                diagnostic_id=f"diagnostic-{counter:04d}",
                diagnostic_type=DiagnosticType.PLANNED_BEHAVIOR_GAP,
                support_level=FindingSupportLevel.DIAGNOSTIC,
                support_strength=SupportStrength.NONE,
                related_event_ids=[event.event_id],
                rationale=(
                    "Instruction-level declared behavior references a sensitive action, "
                    "but no definitive policy support or contradiction was recovered."
                ),
            )
        )
    return diagnostics


def _judgment_requires_policy_gap(
    judgment: ReconciliationJudgment,
    events_by_id: dict[str, CapabilityEvent],
    a_req: set[str],
) -> bool:
    event_ids = [
        ref for ref in judgment.subject_refs if ref in events_by_id
    ]
    if not event_ids:
        return False
    return any(
        event_requires_authorization(events_by_id[event_id], a_req)
        for event_id in event_ids
    )


def _directly_flagged_event_ids(
    *,
    certificates: list[Certificate],
    events_by_id: dict[str, CapabilityEvent],
    a_req: set[str],
) -> set[str]:
    flagged: set[str] = set()
    for certificate in certificates:
        event_ids = [
            event_id
            for event_id in certificate.subject_refs
            if event_id in events_by_id
        ]
        if certificate.kind == CertificateKind.NO_SUPPORT:
            flagged.update(
                event_id
                for event_id in event_ids
                if event_requires_authorization(events_by_id[event_id], a_req)
            )
        elif certificate.kind == CertificateKind.BLOCKING_PROHIBITION:
            flagged.update(event_ids)
    return flagged


def _scope_violated_event_ids(
    *,
    resources_by_id: dict[str, ResourceUse],
    projection_edges: list[ReconciliationEdge],
) -> set[str]:
    flagged: set[str] = set()
    for edge in projection_edges:
        if edge.relation != RelationKind.SCOPE_VIOLATES or edge.resource_id is None:
            continue
        resource = resources_by_id.get(edge.resource_id)
        if resource is None or resource.event_id is None:
            continue
        flagged.add(resource.event_id)
    return flagged


def _path_requires_route_attention(
    path: RiskPath,
    events_by_id: dict[str, CapabilityEvent],
    a_req: set[str],
    directly_flagged_event_ids: set[str] | None = None,
) -> bool:
    sink_event_id = path.sink.event_id
    if (
        directly_flagged_event_ids is not None
        and sink_event_id
        and sink_event_id in directly_flagged_event_ids
    ):
        return False
    sink_event = events_by_id.get(sink_event_id) if sink_event_id else None
    if sink_event is not None:
        return event_requires_authorization(sink_event, a_req)
    return path.sink.label in a_req


def _path_requires_composition_review(
    *,
    path: RiskPath,
    events_by_id: dict[str, CapabilityEvent],
    a_req: set[str],
    directly_flagged_event_ids: set[str],
) -> bool:
    if not _path_requires_route_attention(path, events_by_id, a_req):
        return False
    sink_event_id = path.sink.event_id
    return not (sink_event_id and sink_event_id in directly_flagged_event_ids)


def _path_sensitive_event_ids(
    path: RiskPath,
    events_by_id: dict[str, CapabilityEvent],
    a_req: set[str],
) -> list[str]:
    event_ids: list[str] = []
    seen: set[str] = set()
    for segment in path.segments:
        event_id = segment.event_id
        if event_id is None or event_id in seen:
            continue
        event = events_by_id.get(event_id)
        if event is None or not event_requires_authorization(event, a_req):
            continue
        seen.add(event_id)
        event_ids.append(event_id)
    return event_ids


def _aggregate_findings(
    findings: list[Finding],
    events_by_id: dict[str, CapabilityEvent],
) -> list[Finding]:
    grouped: dict[tuple[object, ...], list[Finding]] = {}
    order: list[tuple[object, ...]] = []
    for finding in findings:
        key = _finding_group_key(finding, events_by_id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(finding)

    aggregated: list[Finding] = []
    for index, key in enumerate(order, start=1):
        aggregated.append(
            _merge_findings(
                grouped[key],
                finding_id=f"finding-{index:04d}",
            )
        )
    return aggregated


def _aggregate_exposures(
    exposures: list[Exposure],
    events_by_id: dict[str, CapabilityEvent],
    paths_by_id: dict[str, RiskPath],
) -> list[Exposure]:
    grouped: dict[tuple[object, ...], list[Exposure]] = {}
    order: list[tuple[object, ...]] = []
    for exposure in exposures:
        key = _exposure_group_key(exposure, events_by_id, paths_by_id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(exposure)

    aggregated: list[Exposure] = []
    for index, key in enumerate(order, start=1):
        aggregated.append(
            _merge_exposures(
                grouped[key],
                exposure_id=f"exposure-{index:04d}",
            )
        )
    return aggregated


def _aggregate_diagnostics(
    diagnostics: list[Diagnostic],
    events_by_id: dict[str, CapabilityEvent],
    paths_by_id: dict[str, RiskPath],
) -> list[Diagnostic]:
    grouped: dict[tuple[object, ...], list[Diagnostic]] = {}
    order: list[tuple[object, ...]] = []
    for diagnostic in diagnostics:
        key = _diagnostic_group_key(diagnostic, events_by_id, paths_by_id)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(diagnostic)

    aggregated: list[Diagnostic] = []
    for index, key in enumerate(order, start=1):
        aggregated.append(
            _merge_diagnostics(
                grouped[key],
                diagnostic_id=f"diagnostic-{index:04d}",
            )
        )
    return aggregated


def _finding_group_key(
    finding: Finding,
    events_by_id: dict[str, CapabilityEvent],
) -> tuple[object, ...]:
    if finding.finding_type == FindingType.UNSUPPORTED_BEHAVIOR:
        return (
            finding.finding_type,
            _event_family_signature(finding.related_event_ids, events_by_id),
            _event_file_signature(finding.related_event_ids, events_by_id),
        )
    return (finding.finding_type, finding.finding_id)


def _diagnostic_group_key(
    diagnostic: Diagnostic,
    events_by_id: dict[str, CapabilityEvent],
    paths_by_id: dict[str, RiskPath],
) -> tuple[object, ...]:
    if diagnostic.diagnostic_type == DiagnosticType.UNRESOLVED_ROUTE:
        return (
            diagnostic.diagnostic_type,
            _path_family_signature(diagnostic.related_path_ids, paths_by_id),
        )
    if diagnostic.diagnostic_type == DiagnosticType.POLICY_GAP:
        return (
            diagnostic.diagnostic_type,
            tuple(sorted(diagnostic.related_clause_ids)),
            _event_family_signature(diagnostic.related_event_ids, events_by_id),
        )
    if diagnostic.diagnostic_type == DiagnosticType.ALIGNMENT_MISMATCH:
        return (
            diagnostic.diagnostic_type,
            tuple(sorted(diagnostic.related_clause_ids)),
            _event_family_signature(diagnostic.related_event_ids, events_by_id),
        )
    if diagnostic.diagnostic_type == DiagnosticType.PARTIAL_PATH_COVERAGE:
        return (
            diagnostic.diagnostic_type,
            _path_family_signature(diagnostic.related_path_ids, paths_by_id),
            _event_family_signature(diagnostic.related_event_ids, events_by_id),
        )
    if diagnostic.diagnostic_type == DiagnosticType.PLANNED_BEHAVIOR_GAP:
        return (
            diagnostic.diagnostic_type,
            _event_family_signature(diagnostic.related_event_ids, events_by_id),
            _event_file_signature(diagnostic.related_event_ids, events_by_id),
        )
    return (diagnostic.diagnostic_type, diagnostic.diagnostic_id)


def _event_ids_from_path_partial_note(notes: str) -> list[str]:
    prefix = "partial_path_coverage:"
    if not notes.startswith(prefix):
        return []
    suffix = notes.removeprefix(prefix)
    return [part for part in suffix.split(",") if part]


def _exposure_group_key(
    exposure: Exposure,
    events_by_id: dict[str, CapabilityEvent],
    paths_by_id: dict[str, RiskPath],
) -> tuple[object, ...]:
    if exposure.exposure_type == ExposureType.DECLARED_SENSITIVE_BEHAVIOR:
        return (
            exposure.exposure_type,
            tuple(sorted(exposure.related_clause_ids)),
            _event_family_signature(exposure.related_event_ids, events_by_id),
            _event_file_signature(exposure.related_event_ids, events_by_id),
        )
    return (
        exposure.exposure_type,
        _path_family_signature(exposure.related_path_ids, paths_by_id),
    )


def _event_family_signature(
    event_ids: list[str],
    events_by_id: dict[str, CapabilityEvent],
) -> tuple[str, ...]:
    labels = {
        event_capability_family(events_by_id[event_id])
        for event_id in event_ids
        if event_id in events_by_id
    }
    return tuple(sorted(labels))


def _event_file_signature(
    event_ids: list[str],
    events_by_id: dict[str, CapabilityEvent],
) -> tuple[str, ...]:
    labels = {
        (
            events_by_id[event_id].file_path
            or events_by_id[event_id].location.split(":", 1)[0]
            or events_by_id[event_id].unit_id
        )
        for event_id in event_ids
        if event_id in events_by_id
    }
    return tuple(sorted(labels))


def _path_family_signature(
    path_ids: list[str],
    paths_by_id: dict[str, RiskPath],
) -> tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...]:
    labels = {
        (
            path.source.label,
            path.sink.label,
            path.path_kind,
            tuple(sorted(path.bridges_used)),
            tuple(sorted(path.orchestration_hypotheses)),
        )
        for path_id in path_ids
        if (path := paths_by_id.get(path_id)) is not None
    }
    return tuple(sorted(labels))


def _merge_findings(
    findings: list[Finding],
    *,
    finding_id: str,
) -> Finding:
    template = findings[0]
    return Finding(
        finding_id=finding_id,
        finding_type=template.finding_type,
        support_level=_merge_support_level([finding.support_level for finding in findings]),
        support_strength=_collapse_support_strength(
            [finding.support_strength for finding in findings]
        ),
        certificate_ids=_merge_str_lists(finding.certificate_ids for finding in findings),
        supporting_judgment_ids=_merge_str_lists(
            finding.supporting_judgment_ids for finding in findings
        ),
        supporting_edge_ids=_merge_str_lists(finding.supporting_edge_ids for finding in findings),
        related_clause_ids=_merge_str_lists(finding.related_clause_ids for finding in findings),
        related_constraint_ids=_merge_str_lists(
            finding.related_constraint_ids for finding in findings
        ),
        related_event_ids=_merge_str_lists(finding.related_event_ids for finding in findings),
        related_resource_ids=_merge_str_lists(
            finding.related_resource_ids for finding in findings
        ),
        related_path_ids=_merge_str_lists(finding.related_path_ids for finding in findings),
        rationale=template.rationale,
    )


def _merge_diagnostics(
    diagnostics: list[Diagnostic],
    *,
    diagnostic_id: str,
) -> Diagnostic:
    template = diagnostics[0]
    return Diagnostic(
        diagnostic_id=diagnostic_id,
        diagnostic_type=template.diagnostic_type,
        support_level=_merge_support_level(
            [diagnostic.support_level for diagnostic in diagnostics]
        ),
        support_strength=_collapse_support_strength(
            [diagnostic.support_strength for diagnostic in diagnostics]
        ),
        certificate_ids=_merge_str_lists(
            diagnostic.certificate_ids for diagnostic in diagnostics
        ),
        supporting_judgment_ids=_merge_str_lists(
            diagnostic.supporting_judgment_ids for diagnostic in diagnostics
        ),
        supporting_edge_ids=_merge_str_lists(
            diagnostic.supporting_edge_ids for diagnostic in diagnostics
        ),
        related_clause_ids=_merge_str_lists(
            diagnostic.related_clause_ids for diagnostic in diagnostics
        ),
        related_constraint_ids=_merge_str_lists(
            diagnostic.related_constraint_ids for diagnostic in diagnostics
        ),
        related_event_ids=_merge_str_lists(
            diagnostic.related_event_ids for diagnostic in diagnostics
        ),
        related_resource_ids=_merge_str_lists(
            diagnostic.related_resource_ids for diagnostic in diagnostics
        ),
        related_path_ids=_merge_str_lists(
            diagnostic.related_path_ids for diagnostic in diagnostics
        ),
        rationale=template.rationale,
    )


def _merge_exposures(
    exposures: list[Exposure],
    *,
    exposure_id: str,
) -> Exposure:
    template = exposures[0]
    return Exposure(
        exposure_id=exposure_id,
        exposure_type=template.exposure_type,
        support_level=_merge_support_level([exposure.support_level for exposure in exposures]),
        support_strength=_collapse_support_strength(
            [exposure.support_strength for exposure in exposures]
        ),
        supporting_judgment_ids=_merge_str_lists(
            exposure.supporting_judgment_ids for exposure in exposures
        ),
        supporting_edge_ids=_merge_str_lists(
            exposure.supporting_edge_ids for exposure in exposures
        ),
        related_clause_ids=_merge_str_lists(
            exposure.related_clause_ids for exposure in exposures
        ),
        related_event_ids=_merge_str_lists(
            exposure.related_event_ids for exposure in exposures
        ),
        related_resource_ids=_merge_str_lists(
            exposure.related_resource_ids for exposure in exposures
        ),
        related_path_ids=_merge_str_lists(
            exposure.related_path_ids for exposure in exposures
        ),
        rationale=template.rationale,
    )


def _merge_support_level(
    levels: list[FindingSupportLevel],
) -> FindingSupportLevel:
    if FindingSupportLevel.GRAPH_BACKED in levels:
        return FindingSupportLevel.GRAPH_BACKED
    return FindingSupportLevel.DIAGNOSTIC


def _merge_str_lists(groups: Iterable[Iterable[str]]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged
