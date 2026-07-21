"""Witness assembly and validation for Module 04."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from skillrecon.core.enums import (
    CertificateKind,
    FindingSupportLevel,
    FindingType,
    JudgmentKind,
    PredicateResult,
    RelationKind,
    SupportStrength,
)
from skillrecon.core.types import (
    Certificate,
    Diagnostic,
    Finding,
    PermissionManifestEntry,
    ReconciliationEdge,
    ReconciliationJudgment,
    Witness,
    WitnessContext,
    WitnessContextLink,
    WitnessValidation,
)


@dataclass
class WitnessBundle:
    witnesses: list[Witness]
    validations: list[WitnessValidation]
    permission_manifest: list[PermissionManifestEntry]


@dataclass
class ProofNeighborhood:
    anchor_ids: list[str]
    fact_node_ids: list[str]
    judgment_ids: list[str]
    certificate_ids: list[str]
    projection_edge_ids: list[str]


_EXACT_NODE_LIMIT = 18
_EXACT_EDGE_LIMIT = 36


def assemble_witnesses(
    findings: list[Finding],
    judgments: list[ReconciliationJudgment],
    certificates: list[Certificate],
    projection_edges: list[ReconciliationEdge],
) -> WitnessBundle:
    """Assemble certificate-closed witnesses from 03 artifacts."""
    judgments_by_id = {judgment.judgment_id: judgment for judgment in judgments}
    certificates_by_id = {
        certificate.certificate_id: certificate for certificate in certificates
    }
    judgment_ids = set(judgments_by_id)
    certificate_ids = set(certificates_by_id)

    witnesses: list[Witness] = []
    validations: list[WitnessValidation] = []

    for index, finding in enumerate(findings, start=1):
        neighborhood = build_proof_neighborhood(
            finding,
            judgments_by_id,
            certificates_by_id,
            projection_edges,
            judgment_ids,
            certificate_ids,
        )
        witness = search_minimal_witness(
            witness_id=f"witness-{index:04d}",
            finding=finding,
            neighborhood=neighborhood,
            judgments_by_id=judgments_by_id,
            certificates_by_id=certificates_by_id,
            projection_edges_by_id={edge.edge_id: edge for edge in projection_edges},
        )
        validation = validate_witness(
            finding,
            witness,
            judgments_by_id,
            certificates_by_id,
            {edge.edge_id: edge for edge in projection_edges},
        )
        witnesses.append(
            witness.model_copy(
                update={
                    "revalidation_passed": validation.passed,
                    "support_strength": _collapse_witness_support_strength(
                        finding,
                        witness,
                        {edge.edge_id: edge for edge in projection_edges},
                    ),
                }
            )
        )
        validations.append(validation)

    permission_manifest = build_permission_manifest(judgments, certificates)
    return WitnessBundle(
        witnesses=witnesses,
        validations=validations,
        permission_manifest=permission_manifest,
    )


def build_proof_neighborhood(
    finding: Finding,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges: list[ReconciliationEdge],
    all_judgment_ids: set[str],
    all_certificate_ids: set[str],
) -> ProofNeighborhood:
    """Construct the local proof neighborhood for one finding."""
    witness_certificate_ids = [
        certificate_id
        for certificate_id in finding.certificate_ids
        if certificate_id in certificates_by_id
    ]
    witness_judgment_ids = _collect_judgment_closure(
        witness_certificate_ids,
        finding.supporting_judgment_ids,
        judgments_by_id,
        certificates_by_id,
    )
    fact_node_ids = _collect_fact_refs(
        finding,
        witness_judgment_ids,
        witness_certificate_ids,
        judgments_by_id,
        certificates_by_id,
        all_judgment_ids,
        all_certificate_ids,
    )
    anchor_ids = _build_anchor_ids(finding, witness_certificate_ids)
    edge_ids = [
        edge.edge_id
        for edge in projection_edges
        if set(edge.source_judgment_ids).intersection(witness_judgment_ids)
    ]
    return ProofNeighborhood(
        anchor_ids=anchor_ids,
        fact_node_ids=sorted(fact_node_ids),
        judgment_ids=sorted(witness_judgment_ids),
        certificate_ids=sorted(witness_certificate_ids),
        projection_edge_ids=sorted(edge_ids),
    )


def search_minimal_witness(
    *,
    witness_id: str,
    finding: Finding,
    neighborhood: ProofNeighborhood,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge],
) -> Witness:
    """Search the smallest valid witness inside a bounded proof neighborhood."""
    total_nodes = (
        len(neighborhood.anchor_ids)
        + len(neighborhood.fact_node_ids)
        + len(neighborhood.judgment_ids)
        + len(neighborhood.certificate_ids)
    )
    if (
        total_nodes <= _EXACT_NODE_LIMIT
        and len(neighborhood.projection_edge_ids) <= _EXACT_EDGE_LIMIT
    ):
        return _exact_search_witness(
            witness_id=witness_id,
            finding=finding,
            neighborhood=neighborhood,
            judgments_by_id=judgments_by_id,
            certificates_by_id=certificates_by_id,
            projection_edges_by_id=projection_edges_by_id,
        )
    return _greedy_witness(
        witness_id=witness_id,
        finding=finding,
        neighborhood=neighborhood,
        judgments_by_id=judgments_by_id,
        certificates_by_id=certificates_by_id,
        projection_edges_by_id=projection_edges_by_id,
    )


def validate_witness(
    finding: Finding,
    witness: Witness,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge] | None = None,
) -> WitnessValidation:
    """Validate a certificate-closed, irreducible witness closure."""
    core_validation = _validate_witness_core(
        witness,
        judgments_by_id,
        certificates_by_id,
        projection_edges_by_id,
        require_projection_edge=finding.support_level == FindingSupportLevel.GRAPH_BACKED,
        required_supporting_edge_ids=finding.supporting_edge_ids,
    )
    if not core_validation.passed:
        return core_validation

    replay_failure = _replay_finding_failure(
        finding,
        witness,
        judgments_by_id,
        certificates_by_id,
        projection_edges_by_id or {},
    )
    if replay_failure is not None:
        return WitnessValidation(
            witness_id=witness.witness_id,
            passed=False,
            failure_reason=replay_failure,
            checked_anchor_ids=list(witness.anchor_ids),
            checked_certificate_ids=list(witness.certificate_ids),
        )

    reducibility_failure = _find_reducibility_failure(
        finding,
        witness,
        judgments_by_id,
        certificates_by_id,
        projection_edges_by_id,
        require_projection_edge=finding.support_level == FindingSupportLevel.GRAPH_BACKED,
        required_supporting_edge_ids=finding.supporting_edge_ids,
    )
    if reducibility_failure is not None:
        return WitnessValidation(
            witness_id=witness.witness_id,
            passed=False,
            failure_reason=reducibility_failure,
            checked_anchor_ids=list(witness.anchor_ids),
            checked_certificate_ids=list(witness.certificate_ids),
        )

    return WitnessValidation(
        witness_id=witness.witness_id,
        passed=True,
        checked_anchor_ids=list(witness.anchor_ids),
        checked_certificate_ids=list(witness.certificate_ids),
    )


def _validate_witness_core(
    witness: Witness,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge] | None = None,
    *,
    require_projection_edge: bool = False,
    required_supporting_edge_ids: list[str] | None = None,
) -> WitnessValidation:
    """Validate the non-reducibility constraints for a witness."""
    included_ids = (
        set(witness.anchor_ids)
        | set(witness.fact_node_ids)
        | set(witness.judgment_ids)
        | set(witness.certificate_ids)
    )
    if require_projection_edge and not witness.projection_edge_ids:
        return WitnessValidation(
            witness_id=witness.witness_id,
            passed=False,
            failure_reason="missing_projection_edge",
            checked_anchor_ids=list(witness.anchor_ids),
            checked_certificate_ids=list(witness.certificate_ids),
        )
    if required_supporting_edge_ids and not set(required_supporting_edge_ids).intersection(
        witness.projection_edge_ids
    ):
        return WitnessValidation(
            witness_id=witness.witness_id,
            passed=False,
            failure_reason="missing_supporting_projection_edge",
            checked_anchor_ids=list(witness.anchor_ids),
            checked_certificate_ids=list(witness.certificate_ids),
        )
    for anchor_id in witness.anchor_ids:
        if anchor_id not in included_ids:
            return WitnessValidation(
                witness_id=witness.witness_id,
                passed=False,
                failure_reason=f"missing_anchor:{anchor_id}",
                checked_anchor_ids=list(witness.anchor_ids),
                checked_certificate_ids=list(witness.certificate_ids),
            )

    for certificate_id in witness.certificate_ids:
        certificate = certificates_by_id.get(certificate_id)
        if certificate is None:
            return WitnessValidation(
                witness_id=witness.witness_id,
                passed=False,
                failure_reason=f"missing_certificate:{certificate_id}",
                checked_anchor_ids=list(witness.anchor_ids),
                checked_certificate_ids=list(witness.certificate_ids),
            )
        for judgment_id in certificate.supporting_judgment_ids:
            if judgment_id not in witness.judgment_ids:
                return WitnessValidation(
                    witness_id=witness.witness_id,
                    passed=False,
                    failure_reason=f"missing_supporting_judgment:{judgment_id}",
                    checked_anchor_ids=list(witness.anchor_ids),
                    checked_certificate_ids=list(witness.certificate_ids),
                )

    for judgment_id in witness.judgment_ids:
        judgment = judgments_by_id.get(judgment_id)
        if judgment is None:
            return WitnessValidation(
                witness_id=witness.witness_id,
                passed=False,
                failure_reason=f"missing_judgment:{judgment_id}",
                checked_anchor_ids=list(witness.anchor_ids),
                checked_certificate_ids=list(witness.certificate_ids),
            )
        for premise_id in judgment.premise_judgment_ids:
            if premise_id not in witness.judgment_ids:
                return WitnessValidation(
                    witness_id=witness.witness_id,
                    passed=False,
                    failure_reason=f"missing_premise:{premise_id}",
                    checked_anchor_ids=list(witness.anchor_ids),
                    checked_certificate_ids=list(witness.certificate_ids),
                )
        for fact_ref in judgment.subject_refs:
            if fact_ref not in witness.fact_node_ids and fact_ref not in witness.anchor_ids:
                return WitnessValidation(
                    witness_id=witness.witness_id,
                    passed=False,
                    failure_reason=f"missing_fact_ref:{fact_ref}",
                    checked_anchor_ids=list(witness.anchor_ids),
                    checked_certificate_ids=list(witness.certificate_ids),
                )

    for certificate_id in witness.certificate_ids:
        certificate = certificates_by_id[certificate_id]
        for fact_ref in certificate.subject_refs:
            if fact_ref not in witness.fact_node_ids and fact_ref not in witness.anchor_ids:
                return WitnessValidation(
                    witness_id=witness.witness_id,
                    passed=False,
                    failure_reason=f"missing_certificate_fact_ref:{fact_ref}",
                    checked_anchor_ids=list(witness.anchor_ids),
                    checked_certificate_ids=list(witness.certificate_ids),
                )

    if projection_edges_by_id is not None:
        for edge_id in witness.projection_edge_ids:
            edge = projection_edges_by_id.get(edge_id)
            if edge is None:
                return WitnessValidation(
                    witness_id=witness.witness_id,
                    passed=False,
                    failure_reason=f"missing_projection_edge:{edge_id}",
                    checked_anchor_ids=list(witness.anchor_ids),
                    checked_certificate_ids=list(witness.certificate_ids),
                )
            if not set(edge.source_judgment_ids).intersection(witness.judgment_ids):
                return WitnessValidation(
                    witness_id=witness.witness_id,
                    passed=False,
                    failure_reason=f"unbacked_projection_edge:{edge_id}",
                    checked_anchor_ids=list(witness.anchor_ids),
                    checked_certificate_ids=list(witness.certificate_ids),
                )
            endpoints = [
                edge.clause_id,
                edge.constraint_id,
                edge.step_id,
                edge.code_unit_id,
                edge.event_id,
                edge.resource_id,
                edge.path_id,
            ]
            for endpoint in [endpoint for endpoint in endpoints if endpoint]:
                if endpoint not in witness.fact_node_ids and endpoint not in witness.anchor_ids:
                    return WitnessValidation(
                        witness_id=witness.witness_id,
                        passed=False,
                        failure_reason=f"missing_projection_endpoint:{endpoint}",
                        checked_anchor_ids=list(witness.anchor_ids),
                        checked_certificate_ids=list(witness.certificate_ids),
                    )

    if not _is_connected(
        witness,
        judgments_by_id,
        certificates_by_id,
        projection_edges_by_id or {},
    ):
        return WitnessValidation(
            witness_id=witness.witness_id,
            passed=False,
            failure_reason="disconnected_witness",
            checked_anchor_ids=list(witness.anchor_ids),
            checked_certificate_ids=list(witness.certificate_ids),
        )

    return WitnessValidation(
        witness_id=witness.witness_id,
        passed=True,
        checked_anchor_ids=list(witness.anchor_ids),
        checked_certificate_ids=list(witness.certificate_ids),
    )


def _find_reducibility_failure(
    finding: Finding,
    witness: Witness,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge] | None,
    *,
    require_projection_edge: bool = False,
    required_supporting_edge_ids: list[str] | None = None,
) -> str | None:
    anchor_set = set(witness.anchor_ids)

    for fact_id in witness.fact_node_ids:
        if fact_id in anchor_set:
            continue
        candidate = witness.model_copy(
            update={
                "fact_node_ids": [item for item in witness.fact_node_ids if item != fact_id],
            }
        )
        if _candidate_witness_still_valid(
            finding,
            candidate,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id,
            require_projection_edge=require_projection_edge,
            required_supporting_edge_ids=required_supporting_edge_ids,
        ):
            return f"reducible_fact:{fact_id}"

    for judgment_id in witness.judgment_ids:
        if judgment_id in anchor_set:
            continue
        candidate = witness.model_copy(
            update={
                "judgment_ids": [item for item in witness.judgment_ids if item != judgment_id],
            }
        )
        if _candidate_witness_still_valid(
            finding,
            candidate,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id,
            require_projection_edge=require_projection_edge,
            required_supporting_edge_ids=required_supporting_edge_ids,
        ):
            return f"reducible_judgment:{judgment_id}"

    for certificate_id in witness.certificate_ids:
        if certificate_id in anchor_set:
            continue
        candidate = witness.model_copy(
            update={
                "certificate_ids": [
                    item for item in witness.certificate_ids if item != certificate_id
                ],
            }
        )
        if _candidate_witness_still_valid(
            finding,
            candidate,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id,
            require_projection_edge=require_projection_edge,
            required_supporting_edge_ids=required_supporting_edge_ids,
        ):
            return f"reducible_certificate:{certificate_id}"

    for edge_id in witness.projection_edge_ids:
        candidate = witness.model_copy(
            update={
                "projection_edge_ids": [
                    item for item in witness.projection_edge_ids if item != edge_id
                ],
            }
        )
        if _candidate_witness_still_valid(
            finding,
            candidate,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id,
            require_projection_edge=require_projection_edge,
            required_supporting_edge_ids=required_supporting_edge_ids,
        ):
            return f"reducible_projection_edge:{edge_id}"

    return None


def _candidate_witness_still_valid(
    finding: Finding,
    witness: Witness,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge] | None,
    *,
    require_projection_edge: bool,
    required_supporting_edge_ids: list[str] | None,
) -> bool:
    core = _validate_witness_core(
        witness,
        judgments_by_id,
        certificates_by_id,
        projection_edges_by_id,
        require_projection_edge=require_projection_edge,
        required_supporting_edge_ids=required_supporting_edge_ids,
    )
    if not core.passed:
        return False
    return (
        _replay_finding_failure(
            finding,
            witness,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id or {},
        )
        is None
    )


def _replay_finding_failure(
    finding: Finding,
    witness: Witness,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge],
) -> str | None:
    """Replay the finding motif from the witness-local proof objects."""
    certificates = [
        certificates_by_id[certificate_id]
        for certificate_id in witness.certificate_ids
        if certificate_id in certificates_by_id
    ]
    judgments = [
        judgments_by_id[judgment_id]
        for judgment_id in witness.judgment_ids
        if judgment_id in judgments_by_id
    ]
    edges = [
        projection_edges_by_id[edge_id]
        for edge_id in witness.projection_edge_ids
        if edge_id in projection_edges_by_id
    ]

    if finding.finding_type == FindingType.UNSUPPORTED_BEHAVIOR:
        return _replay_unsupported_behavior(finding, certificates, judgments)
    if finding.finding_type == FindingType.CONTRADICTED_BEHAVIOR:
        return _replay_contradicted_behavior(finding, certificates, judgments, edges)
    if finding.finding_type == FindingType.SCOPE_VIOLATION:
        return _replay_scope_violation(finding, certificates, judgments, edges)
    if finding.finding_type == FindingType.UNJUSTIFIED_COMPOSITION:
        return _replay_unjustified_composition(finding, certificates, judgments)
    return f"unsupported_finding_type:{finding.finding_type.value}"


def _replay_unsupported_behavior(
    finding: Finding,
    certificates: list[Certificate],
    judgments: list[ReconciliationJudgment],
) -> str | None:
    event_ids = set(finding.related_event_ids)
    matching = [
        certificate
        for certificate in certificates
        if certificate.kind == CertificateKind.NO_SUPPORT
        and event_ids.intersection(certificate.subject_refs)
    ]
    if not matching:
        return "replay_missing_no_support_certificate"
    if any(
        judgment.kind == JudgmentKind.AUTHORIZATION
        and judgment.result == PredicateResult.TRUE
        and event_ids.intersection(judgment.subject_refs)
        for judgment in judgments
    ):
        return "replay_conflicting_authorization"
    return None


def _replay_contradicted_behavior(
    finding: Finding,
    certificates: list[Certificate],
    judgments: list[ReconciliationJudgment],
    edges: list[ReconciliationEdge],
) -> str | None:
    clause_ids = set(finding.related_clause_ids)
    event_ids = set(finding.related_event_ids)
    matching = [
        certificate
        for certificate in certificates
        if certificate.kind == CertificateKind.BLOCKING_PROHIBITION
        and clause_ids.intersection(certificate.subject_refs)
        and event_ids.intersection(certificate.subject_refs)
    ]
    if not matching:
        return "replay_missing_blocking_prohibition_certificate"
    if not any(
        judgment.kind == JudgmentKind.PROHIBITION_CHECK
        and judgment.result == PredicateResult.TRUE
        and clause_ids.intersection(judgment.subject_refs)
        and event_ids.intersection(judgment.subject_refs)
        for judgment in judgments
    ):
        return "replay_missing_true_prohibition_judgment"
    if edges and not any(
        edge.relation == RelationKind.CONTRADICTS
        and edge.clause_id in clause_ids
        and edge.event_id in event_ids
        for edge in edges
    ):
        return "replay_missing_contradicts_edge"
    return None


def _replay_scope_violation(
    finding: Finding,
    certificates: list[Certificate],
    judgments: list[ReconciliationJudgment],
    edges: list[ReconciliationEdge],
) -> str | None:
    constraint_ids = set(finding.related_constraint_ids)
    resource_ids = set(finding.related_resource_ids)
    matching = [
        certificate
        for certificate in certificates
        if certificate.kind == CertificateKind.SCOPE_FAILURE
        and constraint_ids.intersection(certificate.subject_refs)
        and resource_ids.intersection(certificate.subject_refs)
    ]
    if not matching:
        return "replay_missing_scope_failure_certificate"
    if not any(
        judgment.kind == JudgmentKind.SCOPE_CHECK
        and judgment.result == PredicateResult.FALSE
        and constraint_ids.intersection(judgment.subject_refs)
        and resource_ids.intersection(judgment.subject_refs)
        for judgment in judgments
    ):
        return "replay_missing_failed_scope_judgment"
    if edges and not any(
        edge.relation == RelationKind.SCOPE_VIOLATES
        and edge.constraint_id in constraint_ids
        and edge.resource_id in resource_ids
        for edge in edges
    ):
        return "replay_missing_scope_violates_edge"
    return None


def _replay_unjustified_composition(
    finding: Finding,
    certificates: list[Certificate],
    judgments: list[ReconciliationJudgment],
) -> str | None:
    path_ids = set(finding.related_path_ids)
    matching = [
        certificate
        for certificate in certificates
        if certificate.kind == CertificateKind.NO_JUSTIFIED_PATH
        and path_ids.intersection(certificate.subject_refs)
    ]
    if not matching:
        return "replay_missing_no_justified_path_certificate"
    if not any(
        judgment.kind == JudgmentKind.PATH_JUSTIFICATION
        and judgment.result == PredicateResult.FALSE
        and path_ids.intersection(judgment.subject_refs)
        for judgment in judgments
    ):
        return "replay_missing_false_path_judgment"
    return None


def _collapse_witness_support_strength(
    finding: Finding,
    witness: Witness,
    projection_edges_by_id: dict[str, ReconciliationEdge],
) -> SupportStrength:
    if not witness.projection_edge_ids:
        return finding.support_strength
    observed = {
        projection_edges_by_id[edge_id].support_strength
        for edge_id in witness.projection_edge_ids
        if edge_id in projection_edges_by_id
    }
    if not observed:
        return finding.support_strength
    if len(observed) == 1:
        return observed.pop()
    return SupportStrength.MIXED


def build_permission_manifest(
    judgments: list[ReconciliationJudgment],
    certificates: list[Certificate],
) -> list[PermissionManifestEntry]:
    """Create a human-auditable summary of authorization and path outcomes."""
    certs_by_judgment: dict[str, list[str]] = {}
    for certificate in certificates:
        for judgment_id in certificate.supporting_judgment_ids:
            certs_by_judgment.setdefault(judgment_id, []).append(certificate.certificate_id)

    manifest: list[PermissionManifestEntry] = []
    counter = 0
    for judgment in judgments:
        if judgment.kind.value not in {"authorization", "path_justification"}:
            continue
        counter += 1
        status = (
            "authorized"
            if judgment.kind.value == "authorization"
            and judgment.result.value == "true"
            else "unauthorized"
            if judgment.kind.value == "authorization"
            and judgment.result.value == "false"
            else "authorization_abstained"
            if judgment.kind.value == "authorization"
            else "path_justified"
            if judgment.result.value == "true"
            else "path_unjustified"
            if judgment.result.value == "false"
            else "path_abstained"
        )
        manifest.append(
            PermissionManifestEntry(
                manifest_id=f"perm-{counter:04d}",
                subject_refs=list(judgment.subject_refs),
                status=status,
                supporting_judgment_ids=[judgment.judgment_id],
                certificate_ids=certs_by_judgment.get(judgment.judgment_id, []),
                notes=judgment.abstain_reason or judgment.notes,
            )
        )
    return manifest


def build_witness_contexts(
    witnesses: list[Witness],
    diagnostics: list[Diagnostic],
) -> list[WitnessContext]:
    """Attach nearby diagnostics to valid witnesses as auxiliary context."""
    contexts: list[WitnessContext] = []
    for index, witness in enumerate(witnesses, start=1):
        witness_fact_ids = set(witness.anchor_ids) | set(witness.fact_node_ids)
        witness_judgment_ids = set(witness.judgment_ids)
        witness_certificate_ids = set(witness.certificate_ids)
        witness_projection_edge_ids = set(witness.projection_edge_ids)
        links: list[WitnessContextLink] = []

        for diagnostic in diagnostics:
            diagnostic_fact_ids = set(
                [
                    *diagnostic.related_clause_ids,
                    *diagnostic.related_constraint_ids,
                    *diagnostic.related_event_ids,
                    *diagnostic.related_resource_ids,
                    *diagnostic.related_path_ids,
                ]
            )
            shared_fact_ids = sorted(witness_fact_ids.intersection(diagnostic_fact_ids))
            shared_judgment_ids = sorted(
                witness_judgment_ids.intersection(diagnostic.supporting_judgment_ids)
            )
            shared_certificate_ids = sorted(
                witness_certificate_ids.intersection(diagnostic.certificate_ids)
            )
            shared_projection_edge_ids = sorted(
                witness_projection_edge_ids.intersection(diagnostic.supporting_edge_ids)
            )
            overlap_kinds = [
                label
                for label, values in [
                    ("facts", shared_fact_ids),
                    ("judgments", shared_judgment_ids),
                    ("certificates", shared_certificate_ids),
                    ("projection_edges", shared_projection_edge_ids),
                ]
                if values
            ]
            if not overlap_kinds:
                continue
            links.append(
                WitnessContextLink(
                    diagnostic_id=diagnostic.diagnostic_id,
                    diagnostic_type=diagnostic.diagnostic_type,
                    support_level=diagnostic.support_level,
                    support_strength=diagnostic.support_strength,
                    shared_fact_ids=shared_fact_ids,
                    shared_judgment_ids=shared_judgment_ids,
                    shared_certificate_ids=shared_certificate_ids,
                    shared_projection_edge_ids=shared_projection_edge_ids,
                    rationale=(
                        "Attached as auxiliary context via shared "
                        + ", ".join(overlap_kinds)
                    ),
                )
            )

        if not links:
            continue
        contexts.append(
            WitnessContext(
                context_id=f"context-{index:04d}",
                witness_id=witness.witness_id,
                finding_id=witness.finding_id,
                diagnostic_links=links,
            )
        )
    return contexts


def _collect_judgment_closure(
    certificate_ids: list[str],
    seed_judgment_ids: list[str],
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
) -> set[str]:
    pending = list(seed_judgment_ids)
    seen: set[str] = set()
    for certificate_id in certificate_ids:
        certificate = certificates_by_id.get(certificate_id)
        if certificate is not None:
            pending.extend(certificate.supporting_judgment_ids)

    while pending:
        judgment_id = pending.pop()
        if judgment_id in seen:
            continue
        seen.add(judgment_id)
        judgment = judgments_by_id.get(judgment_id)
        if judgment is not None:
            pending.extend(judgment.premise_judgment_ids)
    return seen


def _collect_fact_refs(
    finding: Finding,
    judgment_ids: set[str],
    certificate_ids: list[str],
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    all_judgment_ids: set[str],
    all_certificate_ids: set[str],
) -> set[str]:
    fact_refs = set(
        [
            *finding.related_clause_ids,
            *finding.related_constraint_ids,
            *finding.related_event_ids,
            *finding.related_resource_ids,
            *finding.related_path_ids,
        ]
    )
    for certificate_id in certificate_ids:
        certificate = certificates_by_id.get(certificate_id)
        if certificate is None:
            continue
        fact_refs.update(
            ref
            for ref in [*certificate.subject_refs, *certificate.evidence_refs]
            if ref not in all_judgment_ids and ref not in all_certificate_ids
        )
    for judgment_id in judgment_ids:
        judgment = judgments_by_id[judgment_id]
        fact_refs.update(
            ref
            for ref in [*judgment.subject_refs, *judgment.evidence_refs]
            if ref not in all_judgment_ids and ref not in all_certificate_ids
        )
    return fact_refs


def _build_anchor_ids(finding: Finding, certificate_ids: list[str]) -> list[str]:
    anchors = [
        *certificate_ids,
        *finding.related_clause_ids,
        *finding.related_constraint_ids,
        *finding.related_event_ids,
        *finding.related_resource_ids,
        *finding.related_path_ids,
    ]
    return list(dict.fromkeys(anchor for anchor in anchors if anchor))


def _rank_tuple(
    *,
    fact_count: int,
    judgment_count: int,
    certificate_count: int,
    uncertainty_mass: int,
    projection_edge_count: int,
) -> tuple[int, int, int, int, int]:
    return (
        fact_count,
        judgment_count,
        certificate_count,
        uncertainty_mass,
        projection_edge_count,
    )


def _exact_search_witness(
    *,
    witness_id: str,
    finding: Finding,
    neighborhood: ProofNeighborhood,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge],
) -> Witness:
    anchor_set = set(neighborhood.anchor_ids)
    optional_fact_ids = [
        fact_id for fact_id in neighborhood.fact_node_ids if fact_id not in anchor_set
    ]
    optional_edge_ids = list(neighborhood.projection_edge_ids)
    best: Witness | None = None
    best_rank: tuple[int | float, ...] | None = None

    for fact_count in range(len(optional_fact_ids) + 1):
        for fact_subset in combinations(optional_fact_ids, fact_count):
            fact_node_ids = sorted(fact_subset)
            for edge_count in range(len(optional_edge_ids) + 1):
                for edge_subset in combinations(optional_edge_ids, edge_count):
                    candidate = Witness(
                        witness_id=witness_id,
                        finding_id=finding.finding_id,
                        anchor_ids=list(neighborhood.anchor_ids),
                        fact_node_ids=fact_node_ids,
                        judgment_ids=list(neighborhood.judgment_ids),
                        certificate_ids=list(neighborhood.certificate_ids),
                        projection_edge_ids=list(edge_subset),
                        rank_tuple=_rank_tuple(
                            fact_count=len(fact_node_ids),
                            judgment_count=len(neighborhood.judgment_ids),
                            certificate_count=len(neighborhood.certificate_ids),
                            uncertainty_mass=sum(
                                1
                                for judgment_id in neighborhood.judgment_ids
                                if judgments_by_id[judgment_id].result.value == "abstain"
                            ),
                            projection_edge_count=len(edge_subset),
                        ),
                        is_exact=True,
                    )
                    validation = validate_witness(
                        finding,
                        candidate,
                        judgments_by_id,
                        certificates_by_id,
                        projection_edges_by_id,
                    )
                    if not validation.passed:
                        continue
                    if best_rank is None or candidate.rank_tuple < best_rank:
                        best = candidate
                        best_rank = candidate.rank_tuple
        if best is not None:
            break

    if best is None:
        return _greedy_witness(
            witness_id=witness_id,
            finding=finding,
            neighborhood=neighborhood,
            judgments_by_id=judgments_by_id,
            certificates_by_id=certificates_by_id,
            projection_edges_by_id=projection_edges_by_id,
        )
    return best


def _greedy_witness(
    *,
    witness_id: str,
    finding: Finding,
    neighborhood: ProofNeighborhood,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge],
) -> Witness:
    anchor_set = set(neighborhood.anchor_ids)
    fact_node_ids = [
        fact_id
        for fact_id in dict.fromkeys(neighborhood.fact_node_ids)
        if fact_id not in anchor_set
    ]
    edge_ids = list(neighborhood.projection_edge_ids)

    for fact_id in list(fact_node_ids):
        if fact_id in anchor_set:
            continue
        candidate_facts = [item for item in fact_node_ids if item != fact_id]
        candidate = Witness(
            witness_id=witness_id,
            finding_id=finding.finding_id,
            anchor_ids=list(neighborhood.anchor_ids),
            fact_node_ids=candidate_facts,
            judgment_ids=list(neighborhood.judgment_ids),
            certificate_ids=list(neighborhood.certificate_ids),
            projection_edge_ids=list(edge_ids),
            rank_tuple=(),
            is_exact=False,
        )
        if validate_witness(
            finding,
            candidate,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id,
        ).passed:
            fact_node_ids = candidate_facts

    for edge_id in list(edge_ids):
        candidate_edges = [item for item in edge_ids if item != edge_id]
        candidate = Witness(
            witness_id=witness_id,
            finding_id=finding.finding_id,
            anchor_ids=list(neighborhood.anchor_ids),
            fact_node_ids=list(fact_node_ids),
            judgment_ids=list(neighborhood.judgment_ids),
            certificate_ids=list(neighborhood.certificate_ids),
            projection_edge_ids=candidate_edges,
            rank_tuple=(),
            is_exact=False,
        )
        if validate_witness(
            finding,
            candidate,
            judgments_by_id,
            certificates_by_id,
            projection_edges_by_id,
        ).passed:
            edge_ids = candidate_edges

    return Witness(
        witness_id=witness_id,
        finding_id=finding.finding_id,
        anchor_ids=list(neighborhood.anchor_ids),
        fact_node_ids=sorted(fact_node_ids),
        judgment_ids=list(neighborhood.judgment_ids),
        certificate_ids=list(neighborhood.certificate_ids),
        projection_edge_ids=sorted(edge_ids),
        rank_tuple=_rank_tuple(
            fact_count=len(fact_node_ids),
            judgment_count=len(neighborhood.judgment_ids),
            certificate_count=len(neighborhood.certificate_ids),
            uncertainty_mass=sum(
                1
                for judgment_id in neighborhood.judgment_ids
                if judgments_by_id[judgment_id].result.value == "abstain"
            ),
            projection_edge_count=len(edge_ids),
        ),
        is_exact=False,
    )


def _is_connected(
    witness: Witness,
    judgments_by_id: dict[str, ReconciliationJudgment],
    certificates_by_id: dict[str, Certificate],
    projection_edges_by_id: dict[str, ReconciliationEdge],
) -> bool:
    nodes = set([
        *witness.anchor_ids,
        *witness.fact_node_ids,
        *witness.judgment_ids,
        *witness.certificate_ids,
    ])
    if not nodes:
        return True

    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}

    def connect(a: str | None, b: str | None) -> None:
        if not a or not b or a not in adjacency or b not in adjacency:
            return
        adjacency[a].add(b)
        adjacency[b].add(a)

    for certificate_id in witness.certificate_ids:
        certificate = certificates_by_id.get(certificate_id)
        if certificate is None:
            continue
        for judgment_id in certificate.supporting_judgment_ids:
            connect(certificate_id, judgment_id)
        for fact_ref in certificate.subject_refs:
            connect(certificate_id, fact_ref)

    for judgment_id in witness.judgment_ids:
        judgment = judgments_by_id.get(judgment_id)
        if judgment is None:
            continue
        for premise_id in judgment.premise_judgment_ids:
            connect(judgment_id, premise_id)
        for fact_ref in judgment.subject_refs:
            connect(judgment_id, fact_ref)

    for edge_id in witness.projection_edge_ids:
        edge = projection_edges_by_id.get(edge_id)
        if edge is None:
            continue
        proj_node = f"projection::{edge_id}"
        adjacency.setdefault(proj_node, set())
        for judgment_id in edge.source_judgment_ids:
            if judgment_id in adjacency:
                adjacency[proj_node].add(judgment_id)
                adjacency[judgment_id].add(proj_node)
        for endpoint in [
            edge.clause_id,
            edge.constraint_id,
            edge.step_id,
            edge.code_unit_id,
            edge.event_id,
            edge.resource_id,
            edge.path_id,
        ]:
            if endpoint and endpoint in adjacency:
                adjacency[proj_node].add(endpoint)
                adjacency[endpoint].add(proj_node)

    start = next(iter(adjacency))
    seen = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(adjacency[node_id] - seen)
    return nodes.issubset(seen)
