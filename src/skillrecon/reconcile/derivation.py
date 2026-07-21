"""Reconciliation materialization for proof-carrying Module 03.

Builds:
- primitive and derived judgments
- certificates
- projection edges derived from judgments
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from skillrecon.contract.normalize import (
    is_resource_constraint,
    is_scope_constraint,
    looks_typed_constraint_applicable,
)
from skillrecon.core.enums import (
    BehaviorKind,
    CandidateSource,
    CertificateKind,
    ClauseOperator,
    JudgmentKind,
    PredicateResult,
    RelationKind,
    RouteSupport,
    SupportStrength,
)
from skillrecon.core.sensitivity import event_requires_authorization
from skillrecon.core.types import (
    Bridge,
    CandidatePair,
    CapabilityEvent,
    Certificate,
    Clause,
    OrchestrationHypothesis,
    ReconciliationEdge,
    ReconciliationJudgment,
    ResourceUse,
    RiskPath,
    Step,
    StepUnitCandidate,
)
from skillrecon.reconcile.predicate import (
    _check_constraint,
    _constraint_applies_to_resource,
    _OverlapMap,
    capability_overlaps,
    execution_route_justified,
    prohibition_conflict,
    resource_compatible,
)

logger = logging.getLogger(__name__)


@dataclass
class _CandidateContext:
    candidate: CandidatePair
    clause: Clause
    resources: list[ResourceUse]
    behavior_capability: str


@dataclass(frozen=True)
class _PathSequenceDecision:
    result: PredicateResult
    supporting_judgment_ids: list[str] = field(default_factory=list)
    abstain_reason: str | None = None
    notes: str = ""


@dataclass
class MaterializationResult:
    """Full materialization output for Module 03."""

    judgments: list[ReconciliationJudgment] = field(default_factory=list)
    certificates: list[Certificate] = field(default_factory=list)
    edges: list[ReconciliationEdge] = field(default_factory=list)


@dataclass
class _Accumulator:
    judgments: list[ReconciliationJudgment] = field(default_factory=list)
    certificates: list[Certificate] = field(default_factory=list)
    edges: list[ReconciliationEdge] = field(default_factory=list)
    judgment_counter: int = 0
    certificate_counter: int = 0
    edge_counter: int = 0
    edge_keys: set[tuple] = field(default_factory=set)
    auth_results_by_event: dict[str, list[ReconciliationJudgment]] = field(
        default_factory=dict
    )
    scope_failure_events: set[str] = field(default_factory=set)
    path_route_judgments: dict[str, ReconciliationJudgment] = field(default_factory=dict)

    def add_judgment(
        self,
        kind: JudgmentKind,
        result: PredicateResult,
        subject_refs: list[str],
        *,
        premise_judgment_ids: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        candidate_id: str | None = None,
        route_support_type: RouteSupport | None = None,
        abstain_reason: str | None = None,
        notes: str = "",
    ) -> ReconciliationJudgment:
        self.judgment_counter += 1
        judgment = ReconciliationJudgment(
            judgment_id=f"judgment-{self.judgment_counter:04d}",
            kind=kind,
            result=result,
            subject_refs=subject_refs,
            premise_judgment_ids=premise_judgment_ids or [],
            evidence_refs=evidence_refs or [],
            candidate_id=candidate_id,
            route_support_type=route_support_type,
            abstain_reason=abstain_reason,
            notes=notes,
        )
        self.judgments.append(judgment)
        return judgment

    def add_certificate(
        self,
        kind: CertificateKind,
        subject_refs: list[str],
        *,
        supporting_judgment_ids: list[str] | None = None,
        evidence_refs: list[str] | None = None,
        notes: str = "",
    ) -> Certificate:
        self.certificate_counter += 1
        certificate = Certificate(
            certificate_id=f"cert-{self.certificate_counter:04d}",
            kind=kind,
            subject_refs=subject_refs,
            supporting_judgment_ids=supporting_judgment_ids or [],
            evidence_refs=evidence_refs or [],
            notes=notes,
        )
        self.certificates.append(certificate)
        return certificate

    def add_edge(self, edge: ReconciliationEdge) -> None:
        key = (
            edge.relation.value,
            edge.clause_id,
            edge.constraint_id,
            edge.step_id,
            edge.code_unit_id,
            edge.event_id,
            edge.path_id,
            edge.resource_id,
        )
        if key in self.edge_keys:
            return
        self.edge_keys.add(key)
        self.edge_counter += 1
        self.edges.append(edge.model_copy(update={"edge_id": f"edge-{self.edge_counter:04d}"}))


def derive_edges(
    candidates: list[CandidatePair],
    clauses: dict[str, Clause],
    events: dict[str, CapabilityEvent],
    resources_by_event: dict[str, list[ResourceUse]],
    resources_by_id: dict[str, ResourceUse],
    paths: dict[str, RiskPath],
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
    overlap_map: _OverlapMap,
    a_req: set[str] | None = None,
    steps_by_id: dict[str, Step] | None = None,
) -> list[ReconciliationEdge]:
    """Compatibility wrapper returning only projection edges."""
    return materialize_reconciliation(
        candidates=candidates,
        clauses=clauses,
        events=events,
        resources_by_event=resources_by_event,
        resources_by_id=resources_by_id,
        paths=paths,
        bridges=bridges,
        orchestrations=orchestrations,
        overlap_map=overlap_map,
        a_req=a_req,
        steps_by_id=steps_by_id,
    ).edges


def derive_alignment_edges(
    *,
    alignment_candidates: list[StepUnitCandidate],
    clauses: dict[str, Clause],
    events: dict[str, CapabilityEvent],
    resources_by_id: dict[str, ResourceUse],
    projection_edges: list[ReconciliationEdge],
) -> list[ReconciliationEdge]:
    """Derive evidentially grounded aligns(step, code_unit) edges."""
    if not alignment_candidates:
        return []

    clause_ids_by_step: dict[str, set[str]] = {}
    clause_id_by_constraint: dict[str, str] = {}
    for clause in clauses.values():
        for step_id in clause.step_ids:
            clause_ids_by_step.setdefault(step_id, set()).add(clause.clause_id)
        for constraint in clause.constraints:
            clause_id_by_constraint[constraint.constraint_id] = clause.clause_id

    derived: list[ReconciliationEdge] = []
    seen: set[tuple[str, str]] = set()
    for candidate in alignment_candidates:
        step_clause_ids = clause_ids_by_step.get(candidate.step_id, set())
        if not step_clause_ids:
            continue
        supporting_judgment_ids: set[str] = set()
        support_strength: SupportStrength | None = None

        for edge in projection_edges:
            if edge.relation in {
                RelationKind.SUPPORTS,
                RelationKind.CONTRADICTS,
                RelationKind.POTENTIALLY_SUPPORTS,
                RelationKind.RELATES_TO,
            }:
                if not edge.clause_id or edge.clause_id not in step_clause_ids or not edge.event_id:
                    continue
                event = events.get(edge.event_id)
                if event is None or event.unit_id != candidate.code_unit_id:
                    continue
                supporting_judgment_ids.update(edge.source_judgment_ids)
                edge_strength = (
                    SupportStrength.STRONG
                    if edge.relation in {RelationKind.SUPPORTS, RelationKind.CONTRADICTS}
                    else SupportStrength.WEAK_STRUCTURAL
                )
                support_strength = _merge_alignment_support_strength(
                    support_strength,
                    edge_strength,
                )
            elif edge.relation in {RelationKind.SCOPE_MATCHES, RelationKind.SCOPE_VIOLATES}:
                if not edge.constraint_id or not edge.resource_id:
                    continue
                owner_clause_id = clause_id_by_constraint.get(edge.constraint_id)
                if owner_clause_id not in step_clause_ids:
                    continue
                resource = resources_by_id.get(edge.resource_id)
                if resource is None or resource.unit_id != candidate.code_unit_id:
                    continue
                supporting_judgment_ids.update(edge.source_judgment_ids)
                support_strength = _merge_alignment_support_strength(
                    support_strength,
                    SupportStrength.STRONG,
                )

        structural_strength = _alignment_structural_support_strength(candidate)
        support_strength = _merge_alignment_support_strength(
            support_strength,
            structural_strength,
        )

        if support_strength is None:
            continue
        key = (candidate.step_id, candidate.code_unit_id)
        if key in seen:
            continue
        seen.add(key)
        derived.append(
            ReconciliationEdge(
                edge_id=f"align-{len(derived):04d}",
                relation=RelationKind.ALIGNS,
                step_id=candidate.step_id,
                code_unit_id=candidate.code_unit_id,
                source_judgment_ids=sorted(supporting_judgment_ids),
                candidate_sources=list(candidate.candidate_sources),
                predicate_satisfied=support_strength == SupportStrength.STRONG,
                support_strength=support_strength,
            )
        )

    return derived


def _alignment_structural_support_strength(
    candidate: StepUnitCandidate,
) -> SupportStrength | None:
    sources = set(candidate.candidate_sources)
    if CandidateSource.PACKAGE_LINK in sources or CandidateSource.FILE_MENTION in sources:
        return SupportStrength.STRONG
    if CandidateSource.SECTION_ALIGNMENT in sources:
        relation = candidate.shared_signals.get("section_alignment")
        distance_text = candidate.shared_signals.get("section_distance")
        distance = int(distance_text) if distance_text and distance_text.isdigit() else None
        if relation == "exact_heading" and (distance is None or distance <= 4):
            return SupportStrength.WEAK_STRUCTURAL
        if relation == "nested_heading" and (distance is None or distance <= 2):
            return SupportStrength.WEAK_STRUCTURAL
    return None


def _merge_alignment_support_strength(
    current: SupportStrength | None,
    incoming: SupportStrength | None,
) -> SupportStrength | None:
    if incoming is None:
        return current
    if current is None:
        return incoming
    if SupportStrength.STRONG in {current, incoming}:
        return SupportStrength.STRONG
    if current == incoming:
        return current
    return SupportStrength.MIXED


def materialize_reconciliation(
    candidates: list[CandidatePair],
    clauses: dict[str, Clause],
    events: dict[str, CapabilityEvent],
    resources_by_event: dict[str, list[ResourceUse]],
    resources_by_id: dict[str, ResourceUse],
    paths: dict[str, RiskPath],
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
    overlap_map: _OverlapMap,
    a_req: set[str] | None = None,
    steps_by_id: dict[str, Step] | None = None,
) -> MaterializationResult:
    """Materialize proof-carrying reconciliation artifacts."""
    acc = _Accumulator()
    effective_a_req = a_req or set()
    path_candidates: dict[str, CandidatePair] = {}

    for cand in candidates:
        clause = clauses.get(cand.clause_id)
        if clause is None:
            continue
        if cand.behavior_kind == BehaviorKind.PATH and cand.path_id:
            path_candidates.setdefault(cand.path_id, cand)
            route_result, route_support = execution_route_justified(
                cand, events, paths, bridges, orchestrations
            )
            acc.path_route_judgments[cand.path_id] = acc.add_judgment(
                JudgmentKind.ROUTE_CHECK,
                route_result,
                [cand.path_id],
                candidate_id=cand.candidate_id,
                route_support_type=route_support,
                evidence_refs=_candidate_evidence_refs(cand, bridges, orchestrations),
                abstain_reason=_route_abstain_reason(route_result, route_support),
            )
            continue

        context = _build_context(
            cand, clause, events, resources_by_event, resources_by_id, paths
        )
        _materialize_candidate(
            acc, context, overlap_map, events, paths, bridges, orchestrations
        )

    _materialize_no_support_certificates(acc, events)
    _materialize_path_justifications(
        acc,
        path_candidates,
        paths,
        bridges,
        orchestrations,
        clauses,
        events,
        effective_a_req,
        steps_by_id or {},
    )

    logger.info(
        "Materialized %d judgments, %d certificates, %d edges from %d candidates",
        len(acc.judgments),
        len(acc.certificates),
        len(acc.edges),
        len(candidates),
    )
    return MaterializationResult(
        judgments=acc.judgments,
        certificates=acc.certificates,
        edges=acc.edges,
    )


def _build_context(
    cand: CandidatePair,
    clause: Clause,
    events: dict[str, CapabilityEvent],
    resources_by_event: dict[str, list[ResourceUse]],
    resources_by_id: dict[str, ResourceUse],
    paths: dict[str, RiskPath],
) -> _CandidateContext:
    return _CandidateContext(
        candidate=cand,
        clause=clause,
        resources=_gather_resources(cand, resources_by_event, resources_by_id),
        behavior_capability=_get_behavior_capability(cand, events, paths),
    )


def _materialize_candidate(
    acc: _Accumulator,
    context: _CandidateContext,
    overlap_map: _OverlapMap,
    events: dict[str, CapabilityEvent],
    paths: dict[str, RiskPath],
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
) -> None:
    cand = context.candidate
    clause = context.clause
    obj_id = cand.event_id or cand.resource_id or ""
    evidence_refs = _candidate_evidence_refs(cand, bridges, orchestrations)

    cap_judgment = acc.add_judgment(
        JudgmentKind.CAPABILITY_OVERLAP,
        capability_overlaps(clause, context.behavior_capability, overlap_map),
        [ref for ref in [clause.clause_id, obj_id] if ref],
        candidate_id=cand.candidate_id,
        evidence_refs=evidence_refs,
    )

    route_result, route_support = execution_route_justified(
        cand, events, paths, bridges, orchestrations
    )
    route_judgment = acc.add_judgment(
        JudgmentKind.ROUTE_CHECK,
        route_result,
        [obj_id] if obj_id else [],
        candidate_id=cand.candidate_id,
        route_support_type=route_support,
        evidence_refs=evidence_refs,
        abstain_reason=_route_abstain_reason(route_result, route_support),
    )

    resource_judgment: ReconciliationJudgment | None = None
    if clause.target or any(is_resource_constraint(constraint) for constraint in clause.constraints):
        res_result = resource_compatible(clause, context.resources)
        resource_judgment = acc.add_judgment(
            JudgmentKind.RESOURCE_COMPATIBILITY,
            res_result,
            [ref for ref in [clause.clause_id, *(r.resource_id for r in context.resources)] if ref],
            candidate_id=cand.candidate_id,
            evidence_refs=[*evidence_refs, *(r.resource_id for r in context.resources)],
            abstain_reason="missing_resource_evidence"
            if res_result == PredicateResult.ABSTAIN
            else None,
        )

    scope_judgments = _materialize_scope_checks(
        acc,
        clause,
        context.resources,
        context.behavior_capability,
        cand.event_id,
        cand.candidate_id,
        evidence_refs,
        route_support,
        cand.candidate_sources,
        bridges,
        orchestrations,
        allow_scope_violation=_eligible_for_scope_violation(
            clause,
            cap_judgment.result,
            route_judgment.result,
            resource_judgment.result if resource_judgment else None,
        ),
    )

    if clause.operator == ClauseOperator.PROHIBITED and cand.event_id:
        prohib_result = prohibition_conflict(
            clause,
            context.behavior_capability,
            context.resources,
            overlap_map,
        )
        prohibition_judgment = acc.add_judgment(
            JudgmentKind.PROHIBITION_CHECK,
            prohib_result,
            [clause.clause_id, cand.event_id],
            premise_judgment_ids=[cap_judgment.judgment_id, route_judgment.judgment_id],
            candidate_id=cand.candidate_id,
            evidence_refs=evidence_refs,
            route_support_type=route_support,
        )
        if prohibition_judgment.result == PredicateResult.TRUE:
            acc.add_certificate(
                CertificateKind.BLOCKING_PROHIBITION,
                [clause.clause_id, cand.event_id],
                supporting_judgment_ids=[prohibition_judgment.judgment_id],
                evidence_refs=evidence_refs,
            )
            acc.add_edge(
                ReconciliationEdge(
                    edge_id="pending",
                    relation=RelationKind.CONTRADICTS,
                    clause_id=clause.clause_id,
                    event_id=cand.event_id,
                    source_judgment_ids=[prohibition_judgment.judgment_id],
                    candidate_sources=list(cand.candidate_sources),
                    route_support_type=route_support,
                    bridge_ids=_bridge_ids_from_candidate(cand, bridges),
                    orchestration_hypothesis_ids=_orchestration_ids_from_candidate(
                        cand, orchestrations
                    ),
                )
            )

    if clause.operator == ClauseOperator.ALLOWED and cand.event_id:
        authorization = _derive_authorization_result(
            clause,
            cap_judgment.result,
            route_judgment.result,
            resource_judgment.result if resource_judgment else None,
            scope_judgments,
        )
        auth_abstain_reason = _authorization_abstain_reason(
            clause,
            cap_judgment.result,
            route_judgment.result,
            resource_judgment.result if resource_judgment else None,
            scope_judgments,
        )
        if authorization == PredicateResult.TRUE:
            alignment_guard_reason = _alignment_guard_abstain_reason(
                clause=clause,
                candidate=cand,
                route_support=route_support,
            )
            if alignment_guard_reason is not None:
                authorization = PredicateResult.ABSTAIN
                auth_abstain_reason = alignment_guard_reason
        auth_judgment = acc.add_judgment(
            JudgmentKind.AUTHORIZATION,
            authorization,
            [clause.clause_id, cand.event_id],
            premise_judgment_ids=[
                cap_judgment.judgment_id,
                route_judgment.judgment_id,
                *([resource_judgment.judgment_id] if resource_judgment else []),
                *(judgment.judgment_id for judgment in scope_judgments),
            ],
            candidate_id=cand.candidate_id,
            evidence_refs=evidence_refs,
            route_support_type=route_support,
            abstain_reason=auth_abstain_reason,
        )
        acc.auth_results_by_event.setdefault(cand.event_id, []).append(auth_judgment)
        if auth_judgment.result == PredicateResult.TRUE:
            acc.add_edge(
                ReconciliationEdge(
                    edge_id="pending",
                    relation=RelationKind.SUPPORTS,
                    clause_id=clause.clause_id,
                    event_id=cand.event_id,
                    source_judgment_ids=[auth_judgment.judgment_id],
                    candidate_sources=list(cand.candidate_sources),
                    route_support_type=route_support,
                    bridge_ids=_bridge_ids_from_candidate(cand, bridges),
                    orchestration_hypothesis_ids=_orchestration_ids_from_candidate(
                        cand, orchestrations
                    ),
                )
            )
        elif auth_judgment.result == PredicateResult.ABSTAIN and _eligible_for_potential_support(
            cap_judgment.result,
            route_judgment.result,
            resource_judgment.result if resource_judgment else None,
            scope_judgments,
            cand.candidate_sources,
            route_support,
        ):
            acc.add_edge(
                ReconciliationEdge(
                    edge_id="pending",
                    relation=RelationKind.POTENTIALLY_SUPPORTS,
                    clause_id=clause.clause_id,
                    event_id=cand.event_id,
                    predicate_satisfied=False,
                    source_judgment_ids=[auth_judgment.judgment_id],
                    candidate_sources=list(cand.candidate_sources),
                    route_support_type=route_support,
                    bridge_ids=_bridge_ids_from_candidate(cand, bridges),
                    orchestration_hypothesis_ids=_orchestration_ids_from_candidate(
                        cand, orchestrations
                    ),
                )
            )

    if clause.operator == ClauseOperator.UNKNOWN and cand.event_id and _eligible_for_related_edge(
        cap_judgment.result,
        route_judgment.result,
        resource_judgment.result if resource_judgment else None,
        scope_judgments,
    ):
        related_judgment_ids = [cap_judgment.judgment_id, route_judgment.judgment_id]
        if resource_judgment is not None:
            related_judgment_ids.append(resource_judgment.judgment_id)
        related_judgment_ids.extend(judgment.judgment_id for judgment in scope_judgments)
        acc.add_edge(
            ReconciliationEdge(
                edge_id="pending",
                relation=RelationKind.RELATES_TO,
                clause_id=clause.clause_id,
                event_id=cand.event_id,
                predicate_satisfied=False,
                source_judgment_ids=related_judgment_ids,
                candidate_sources=list(cand.candidate_sources),
                route_support_type=route_support,
                bridge_ids=_bridge_ids_from_candidate(cand, bridges),
                orchestration_hypothesis_ids=_orchestration_ids_from_candidate(
                    cand, orchestrations
                ),
            )
        )


def _materialize_scope_checks(
    acc: _Accumulator,
    clause: Clause,
    resources: list[ResourceUse],
    behavior_capability: str,
    event_id: str | None,
    candidate_id: str,
    evidence_refs: list[str],
    route_support: RouteSupport | None,
    candidate_sources: list,
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
    allow_scope_violation: bool,
) -> list[ReconciliationJudgment]:
    judgments: list[ReconciliationJudgment] = []
    for constraint in clause.constraints:
        if looks_typed_constraint_applicable(constraint) and is_scope_constraint(constraint):
            relevant_resources = [
                resource
                for resource in resources
                if _constraint_applies_to_resource(constraint, resource)
            ]
        else:
            relevant_resources = []
        for resource in relevant_resources:
            result = (
                PredicateResult.ABSTAIN
                if not resource.resolved
                else _check_constraint(
                    constraint,
                    [resource],
                    behavior_capability=behavior_capability,
                )
            )
            judgment = acc.add_judgment(
                JudgmentKind.SCOPE_CHECK,
                result,
                [constraint.constraint_id, resource.resource_id],
                candidate_id=candidate_id,
                evidence_refs=[*evidence_refs, resource.resource_id],
                route_support_type=route_support,
                abstain_reason="unresolved_resource" if result == PredicateResult.ABSTAIN else None,
            )
            judgments.append(judgment)
            if result == PredicateResult.TRUE:
                acc.add_edge(
                    ReconciliationEdge(
                        edge_id="pending",
                        relation=RelationKind.SCOPE_MATCHES,
                        constraint_id=constraint.constraint_id,
                        resource_id=resource.resource_id,
                        source_judgment_ids=[judgment.judgment_id],
                        candidate_sources=list(candidate_sources),
                        route_support_type=route_support,
                        bridge_ids=[],
                        orchestration_hypothesis_ids=[],
                    )
                )
            elif result == PredicateResult.FALSE and allow_scope_violation:
                if event_id is not None:
                    acc.scope_failure_events.add(event_id)
                acc.add_certificate(
                    CertificateKind.SCOPE_FAILURE,
                    [constraint.constraint_id, resource.resource_id],
                    supporting_judgment_ids=[judgment.judgment_id],
                    evidence_refs=[*evidence_refs, resource.resource_id],
                )
                acc.add_edge(
                    ReconciliationEdge(
                        edge_id="pending",
                        relation=RelationKind.SCOPE_VIOLATES,
                        constraint_id=constraint.constraint_id,
                        resource_id=resource.resource_id,
                        source_judgment_ids=[judgment.judgment_id],
                        candidate_sources=list(candidate_sources),
                        route_support_type=route_support,
                        bridge_ids=[],
                        orchestration_hypothesis_ids=[],
                    )
                )
    return judgments


def _eligible_for_scope_violation(
    clause: Clause,
    cap_result: PredicateResult,
    route_result: PredicateResult,
    resource_result: PredicateResult | None,
) -> bool:
    """Whether a failed scope check may become a scope violation finding.

    This approximates the paper's requirement that a scope violation should
    hang off an otherwise supported allowed clause-event pair, with scope being
    the failing dimension rather than a missing base support relation.
    """
    if clause.operator != ClauseOperator.ALLOWED:
        return False
    if cap_result != PredicateResult.TRUE or route_result != PredicateResult.TRUE:
        return False
    if clause.target:
        return resource_result == PredicateResult.TRUE
    return True


def _eligible_for_potential_support(
    cap_result: PredicateResult,
    route_result: PredicateResult,
    resource_result: PredicateResult | None,
    scope_judgments: list[ReconciliationJudgment],
    candidate_sources: list[CandidateSource],
    route_support: RouteSupport | None,
) -> bool:
    """Whether an abstained allowed clause still warrants a weak support edge."""
    if cap_result != PredicateResult.TRUE:
        return False
    if route_result == PredicateResult.FALSE:
        return False
    if resource_result == PredicateResult.FALSE:
        return False
    if any(judgment.result == PredicateResult.FALSE for judgment in scope_judgments):
        return False
    if _has_strong_candidate_anchor(candidate_sources, route_support):
        return True
    if resource_result == PredicateResult.TRUE:
        return True
    return any(judgment.result == PredicateResult.TRUE for judgment in scope_judgments)


def _eligible_for_related_edge(
    cap_result: PredicateResult,
    route_result: PredicateResult,
    resource_result: PredicateResult | None,
    scope_judgments: list[ReconciliationJudgment],
) -> bool:
    """Whether an unknown policy clause is still stably related to an event."""
    if cap_result != PredicateResult.TRUE:
        return False
    if route_result == PredicateResult.FALSE:
        return False
    if resource_result == PredicateResult.FALSE:
        return False
    return not any(judgment.result == PredicateResult.FALSE for judgment in scope_judgments)


def _has_strong_candidate_anchor(
    candidate_sources: list[CandidateSource],
    route_support: RouteSupport | None,
) -> bool:
    strong_sources = {
        CandidateSource.LITERAL_OVERLAP,
        CandidateSource.PACKAGE_LINK,
        CandidateSource.FILE_MENTION,
        CandidateSource.STEP_UNIT_ALIGNMENT,
    }
    if any(source in strong_sources for source in candidate_sources):
        return True
    return route_support in {
        RouteSupport.CODEQL,
        RouteSupport.BRIDGE,
        RouteSupport.ORCHESTRATION_CONFIRMED,
    }


def _derive_authorization_result(
    clause: Clause,
    cap_result: PredicateResult,
    route_result: PredicateResult,
    resource_result: PredicateResult | None,
    scope_judgments: list[ReconciliationJudgment],
) -> PredicateResult:
    if cap_result != PredicateResult.TRUE:
        return cap_result
    if route_result != PredicateResult.TRUE:
        return route_result

    if clause.target:
        if resource_result is None:
            return PredicateResult.ABSTAIN
        if resource_result != PredicateResult.TRUE:
            return resource_result

    resource_constraints = [
        constraint for constraint in clause.constraints if is_resource_constraint(constraint)
    ]
    if resource_constraints:
        if resource_result is None:
            return PredicateResult.ABSTAIN
        if resource_result != PredicateResult.TRUE:
            return resource_result

    effective_constraints = [
        constraint
        for constraint in clause.constraints
        if is_scope_constraint(constraint)
    ]
    if effective_constraints:
        if not scope_judgments:
            return PredicateResult.ABSTAIN
        if any(judgment.result == PredicateResult.FALSE for judgment in scope_judgments):
            return PredicateResult.FALSE
        if any(judgment.result == PredicateResult.ABSTAIN for judgment in scope_judgments):
            return PredicateResult.ABSTAIN

    return PredicateResult.TRUE


def _authorization_abstain_reason(
    clause: Clause,
    cap_result: PredicateResult,
    route_result: PredicateResult,
    resource_result: PredicateResult | None,
    scope_judgments: list[ReconciliationJudgment],
) -> str | None:
    if cap_result == PredicateResult.ABSTAIN:
        return "capability_overlap_abstained"
    if route_result == PredicateResult.ABSTAIN:
        return "route_check_abstained"
    if clause.target and resource_result == PredicateResult.ABSTAIN:
        return "resource_compatibility_abstained"
    if any(is_resource_constraint(constraint) for constraint in clause.constraints) and (
        not clause.target and resource_result == PredicateResult.ABSTAIN
    ):
        return "resource_compatibility_abstained"
    effective_constraints = [
        constraint
        for constraint in clause.constraints
        if is_scope_constraint(constraint)
    ]
    if effective_constraints and (
        not scope_judgments
        or any(judgment.result == PredicateResult.ABSTAIN for judgment in scope_judgments)
    ):
        return "scope_check_abstained"
    return None


def _alignment_guard_abstain_reason(
    *,
    clause: Clause,
    candidate: CandidatePair,
    route_support: RouteSupport | None,
) -> str | None:
    if not clause.step_ids:
        return None
    if candidate.shared_signals.get("alignment_status") != "mismatch":
        return None
    if _has_strong_candidate_anchor(candidate.candidate_sources, route_support):
        return None
    return "alignment_mismatch"


def _materialize_no_support_certificates(
    acc: _Accumulator,
    events: dict[str, CapabilityEvent],
) -> None:
    grouped_events: dict[
        tuple[str, str, str, str, str],
        list[tuple[str, CapabilityEvent]],
    ] = {}
    for event_id, event in events.items():
        if event.tier == "instruction":
            continue
        if event_id in acc.scope_failure_events:
            continue
        auth_judgments = acc.auth_results_by_event.get(event_id, [])
        results = {judgment.result for judgment in auth_judgments}
        if PredicateResult.TRUE in results:
            continue
        if PredicateResult.ABSTAIN in results:
            continue
        if not auth_judgments or PredicateResult.FALSE in results:
            grouped_events.setdefault(_no_support_signature(event), []).append((event_id, event))

    for grouped in grouped_events.values():
        subject_refs = [event_id for event_id, _ in grouped]
        supporting_judgment_ids = sorted(
            {
                judgment.judgment_id
                for event_id, _ in grouped
                for judgment in acc.auth_results_by_event.get(event_id, [])
            }
        )
        acc.add_certificate(
            CertificateKind.NO_SUPPORT,
            subject_refs,
            supporting_judgment_ids=supporting_judgment_ids,
            evidence_refs=subject_refs,
            notes="no_stable_authorization_support",
        )


def _materialize_path_justifications(
    acc: _Accumulator,
    path_candidates: dict[str, CandidatePair],
    paths: dict[str, RiskPath],
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
    clauses: dict[str, Clause],
    events: dict[str, CapabilityEvent],
    a_req: set[str],
    steps_by_id: dict[str, Step],
) -> None:
    for path_id, cand in path_candidates.items():
        path = paths.get(path_id)
        if path is None:
            continue
        route_judgment = acc.path_route_judgments.get(path_id)
        if route_judgment is None:
            continue
        path_event_ids = [
            segment.event_id for segment in path.segments if segment.event_id is not None
        ]
        path_event_ids = list(dict.fromkeys(path_event_ids))
        required_event_ids = [
            event_id
            for event_id in path_event_ids
            if (event := events.get(event_id)) is not None
            and event_requires_authorization(event, a_req)
        ]

        result = PredicateResult.TRUE
        abstain_reason: str | None = None
        notes = ""
        premise_judgment_ids = [route_judgment.judgment_id]
        if route_judgment.result == PredicateResult.ABSTAIN:
            result = PredicateResult.ABSTAIN
            abstain_reason = route_judgment.abstain_reason or "route_check_abstained"
        elif route_judgment.result == PredicateResult.FALSE:
            result = PredicateResult.FALSE
        elif not path_event_ids:
            result = PredicateResult.ABSTAIN
            abstain_reason = "missing_path_event_ids"
        elif not required_event_ids:
            result = PredicateResult.ABSTAIN
            abstain_reason = "open_world_path_only"
        else:
            blocking_clause_id, blocking_event_id, blocking_judgment_ids = _path_blocking_prohibition(
                path_event_ids=path_event_ids,
                projection_edges=acc.edges,
            )
            if blocking_clause_id is not None and blocking_event_id is not None:
                result = PredicateResult.FALSE
                notes = f"blocked_by_prohibition:{blocking_clause_id}->{blocking_event_id}"
                premise_judgment_ids.extend(blocking_judgment_ids)
            else:
                segment_statuses, segment_judgment_ids = _path_segment_support_profile(
                    required_event_ids=required_event_ids,
                    projection_edges=acc.edges,
                )
                missing_events = [
                    event_id
                    for event_id, status in segment_statuses.items()
                    if status == "none"
                ]
                partial_events = [
                    event_id
                    for event_id, status in segment_statuses.items()
                    if status == "partial"
                ]
                premise_judgment_ids.extend(segment_judgment_ids)
                if missing_events:
                    result = PredicateResult.FALSE
                    if len(missing_events) == 1:
                        notes = f"missing_support:{missing_events[0]}"
                    else:
                        notes = "missing_support:" + ",".join(missing_events)
                elif partial_events:
                    result = PredicateResult.ABSTAIN
                    abstain_reason = "partial_path_coverage"
                    notes = "partial_path_coverage:" + ",".join(partial_events)
                else:
                    sequence_decision = _search_clause_sequence_for_path(
                        required_event_ids=required_event_ids,
                        auth_results_by_event=acc.auth_results_by_event,
                        clauses=clauses,
                        steps_by_id=steps_by_id,
                    )
                    result = sequence_decision.result
                    abstain_reason = sequence_decision.abstain_reason
                    notes = sequence_decision.notes
                    premise_judgment_ids.extend(sequence_decision.supporting_judgment_ids)

        judgment = acc.add_judgment(
            JudgmentKind.PATH_JUSTIFICATION,
            result,
            [path_id, *path_event_ids],
            premise_judgment_ids=list(dict.fromkeys(premise_judgment_ids)),
            candidate_id=cand.candidate_id,
            evidence_refs=[
                path_id,
                *path.bridges_used,
                *path.orchestration_hypotheses,
            ],
            route_support_type=route_judgment.route_support_type,
            abstain_reason=abstain_reason,
            notes=notes,
        )

        if result == PredicateResult.FALSE:
            acc.add_certificate(
                CertificateKind.NO_JUSTIFIED_PATH,
                [path_id],
                supporting_judgment_ids=[judgment.judgment_id],
                evidence_refs=[path_id, *path_event_ids],
            )
        elif result == PredicateResult.ABSTAIN and route_judgment.result == PredicateResult.ABSTAIN:
            acc.add_certificate(
                CertificateKind.UNRESOLVED_ROUTE,
                [path_id],
                supporting_judgment_ids=[judgment.judgment_id, route_judgment.judgment_id],
                evidence_refs=[
                    path_id,
                    *path.bridges_used,
                    *path.orchestration_hypotheses,
                ],
            )


def _path_blocking_prohibition(
    *,
    path_event_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> tuple[str | None, str | None, list[str]]:
    for edge in projection_edges:
        if edge.relation != RelationKind.CONTRADICTS or edge.event_id not in path_event_ids:
            continue
        return edge.clause_id, edge.event_id, list(edge.source_judgment_ids)
    return None, None, []


def _path_segment_support_profile(
    *,
    required_event_ids: list[str],
    projection_edges: list[ReconciliationEdge],
) -> tuple[dict[str, str], list[str]]:
    statuses = {event_id: "none" for event_id in required_event_ids}
    supporting_judgment_ids: set[str] = set()
    for edge in projection_edges:
        if edge.event_id not in statuses:
            continue
        if edge.relation == RelationKind.SUPPORTS:
            statuses[edge.event_id] = "strong"
            supporting_judgment_ids.update(edge.source_judgment_ids)
            continue
        if (
            edge.relation in {RelationKind.POTENTIALLY_SUPPORTS, RelationKind.RELATES_TO}
            and statuses[edge.event_id] != "strong"
        ):
            statuses[edge.event_id] = "partial"
            supporting_judgment_ids.update(edge.source_judgment_ids)
    return statuses, sorted(supporting_judgment_ids)


def _search_clause_sequence_for_path(
    *,
    required_event_ids: list[str],
    auth_results_by_event: dict[str, list[ReconciliationJudgment]],
    clauses: dict[str, Clause],
    steps_by_id: dict[str, Step],
) -> _PathSequenceDecision:
    option_sets: list[list[tuple[str, str, list[tuple[str, int]]]]] = []
    for event_id in required_event_ids:
        options: list[tuple[str, str, list[tuple[str, int]]]] = []
        for judgment in auth_results_by_event.get(event_id, []):
            if judgment.result != PredicateResult.TRUE:
                continue
            clause_id = next((ref for ref in judgment.subject_refs if ref in clauses), None)
            if clause_id is None:
                continue
            options.append(
                (
                    clause_id,
                    judgment.judgment_id,
                    _clause_step_orders(clauses[clause_id], steps_by_id),
                )
            )
        if not options:
            return _PathSequenceDecision(
                result=PredicateResult.FALSE,
                notes=f"missing_support:{event_id}",
            )
        option_sets.append(options)

    frontier_note = "clause_order_incompatible"
    saw_missing_order = False
    saw_incomparable_order = False

    def search(
        index: int,
        previous_pos: tuple[str, int] | None,
        chosen_judgment_ids: list[str],
    ) -> list[str] | None:
        nonlocal frontier_note, saw_missing_order, saw_incomparable_order
        if index >= len(option_sets):
            return chosen_judgment_ids

        current_event_id = required_event_ids[index]
        previous_event_id = required_event_ids[index - 1] if index > 0 else None

        for clause_id, judgment_id, positions in option_sets[index]:
            if not positions:
                saw_missing_order = True
                frontier_note = f"missing_clause_step_order:{current_event_id}"
                continue
            for pos in positions:
                if previous_pos is None:
                    result = search(index + 1, pos, [*chosen_judgment_ids, judgment_id])
                    if result is not None:
                        return result
                    continue
                if pos[0] != previous_pos[0]:
                    saw_incomparable_order = True
                    frontier_note = (
                        f"incomparable_clause_order:{previous_event_id}->{current_event_id}"
                    )
                    continue
                if pos[1] < previous_pos[1]:
                    frontier_note = f"order_conflict:{previous_event_id}->{current_event_id}"
                    continue
                result = search(index + 1, pos, [*chosen_judgment_ids, judgment_id])
                if result is not None:
                    return result
        return None

    supporting_judgment_ids = search(0, None, [])
    if supporting_judgment_ids is not None:
        return _PathSequenceDecision(
            result=PredicateResult.TRUE,
            supporting_judgment_ids=supporting_judgment_ids,
            notes="ordered_clause_sequence_found",
        )
    if saw_missing_order or saw_incomparable_order:
        return _PathSequenceDecision(
            result=PredicateResult.ABSTAIN,
            abstain_reason="missing_clause_step_order",
            notes=frontier_note,
        )
    return _PathSequenceDecision(
        result=PredicateResult.FALSE,
        notes=frontier_note,
    )


def _get_behavior_capability(
    cand: CandidatePair,
    events: dict[str, CapabilityEvent],
    paths: dict[str, RiskPath],
) -> str:
    if cand.behavior_kind == BehaviorKind.EVENT and cand.event_id:
        event = events.get(cand.event_id)
        return event.capability if event else ""
    if cand.behavior_kind == BehaviorKind.PATH and cand.path_id:
        path = paths.get(cand.path_id)
        if path is not None:
            return path.sink.label
    return cand.shared_signals.get("capability", "")


def _no_support_signature(event: CapabilityEvent) -> tuple[str, str, str, str, str]:
    return (
        event.unit_id,
        event.capability,
        event.tier,
        event.api_call or event.detail,
        event.file_path,
    )


def _gather_resources(
    cand: CandidatePair,
    resources_by_event: dict[str, list[ResourceUse]],
    resources_by_id: dict[str, ResourceUse],
) -> list[ResourceUse]:
    result: list[ResourceUse] = []
    if cand.event_id:
        result.extend(resources_by_event.get(cand.event_id, []))
    if cand.resource_id:
        resource = resources_by_id.get(cand.resource_id)
        if resource is not None:
            result.append(resource)
    return result


def _candidate_evidence_refs(
    cand: CandidatePair,
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
) -> list[str]:
    refs = [cand.clause_id]
    refs.extend(_bridge_ids_from_candidate(cand, bridges))
    refs.extend(_orchestration_ids_from_candidate(cand, orchestrations))
    return refs


def _bridge_ids_from_candidate(
    cand: CandidatePair,
    bridges: dict[str, Bridge],
) -> list[str]:
    return [ref for ref in cand.route_refs if ref in bridges]


def _orchestration_ids_from_candidate(
    cand: CandidatePair,
    orchestrations: dict[str, OrchestrationHypothesis],
) -> list[str]:
    return [ref for ref in cand.route_refs if ref in orchestrations]


def _route_abstain_reason(
    result: PredicateResult,
    support: RouteSupport | None,
) -> str | None:
    if result != PredicateResult.ABSTAIN:
        return None
    if support == RouteSupport.ORCHESTRATION_UNRESOLVED:
        return "orchestration_unresolved"
    if support == RouteSupport.WEAK:
        return "weak_route_only"
    return "route_abstained"


def _has_order_compatible_clause_sequence(
    event_ids: list[str],
    auth_results_by_event: dict[str, list[ReconciliationJudgment]],
    clauses: dict[str, Clause],
    steps_by_id: dict[str, Step],
) -> bool | None:
    ordered_options: list[list[tuple[str, int]]] = []
    for event_id in event_ids:
        step_orders: set[tuple[str, int]] = set()
        for judgment in auth_results_by_event.get(event_id, []):
            if judgment.result != PredicateResult.TRUE:
                continue
            clause_id = next(
                (ref for ref in judgment.subject_refs if ref in clauses),
                None,
            )
            if clause_id is None:
                continue
            step_orders.update(_clause_step_orders(clauses[clause_id], steps_by_id))
        if not step_orders:
            return None
        ordered_options.append(sorted(step_orders))

    previous: tuple[str, int] | None = None
    for options in ordered_options:
        next_step = next(
            (
                step
                for step in options
                if previous is None or (step[0] == previous[0] and step[1] >= previous[1])
            ),
            None,
        )
        if next_step is None:
            return False
        previous = next_step
    return True


def _clause_step_orders(
    clause: Clause,
    steps_by_id: dict[str, Step],
) -> list[tuple[str, int]]:
    return sorted(
        {
            _step_order_key(step_id, steps_by_id)
            for step_id in clause.step_ids
            if step_id
        }
    )


def _step_order_key(
    step_id: str,
    steps_by_id: dict[str, Step],
) -> tuple[str, int]:
    step = steps_by_id.get(step_id)
    if step is not None:
        return (step.doc_id, step.order_index)
    digits = [int(part) for part in re.findall(r"\d+", step_id)]
    fallback_index = digits[-1] if digits else 0
    doc_id = step_id.split("_", 1)[0] if "_" in step_id else "unknown"
    return (doc_id, fallback_index)
