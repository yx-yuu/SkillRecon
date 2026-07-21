"""Pydantic v2 models for all SkillRecon scientific objects.

Organized by the five object categories from the research design:
1. Package-level objects
2. Contract-side objects
3. Behavior-side objects
4. Cross-modal objects
5. Explanation objects
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from skillrecon.core.enums import (
    BehaviorKind,
    BridgeKind,
    CandidateSource,
    CertificateKind,
    ClauseOperator,
    ClauseRole,
    DiagnosticType,
    ExposureType,
    FileKind,
    FindingSupportLevel,
    FindingType,
    HypothesisStatus,
    JudgmentKind,
    PredicateResult,
    ReferenceResolutionStatus,
    ReferenceType,
    RelationKind,
    RouteSupport,
    SupportStrength,
)

# ---------------------------------------------------------------------------
# 1. Package-level objects
# ---------------------------------------------------------------------------


class FileEntry(BaseModel):
    """A single file within a skill package."""

    model_config = ConfigDict(frozen=True)

    file_id: str
    relative_path: str
    kind: FileKind
    size_bytes: int = 0
    language: str | None = None


class DocumentNode(BaseModel):
    """A document admitted into the rooted reference closure."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    file_id: str
    depth: int
    parent_doc_id: str | None = None
    reference_type: ReferenceType | None = None


class CodeUnit(BaseModel):
    """An independent code entity within a skill package.

    For standalone file-based units, provenance fields are None. For
    synthetic units extracted from inline implementation in markdown,
    provenance fields record the original document/block context.
    """

    model_config = ConfigDict(frozen=True)

    unit_id: str
    file_id: str
    language: str
    entry_point: bool = False
    source_doc_id: str | None = None
    source_block_id: str | None = None
    source_offset: int | None = None
    heading_context: str | None = None
    binding_instruction: str | None = None


class PackageLink(BaseModel):
    """A link connecting document mention to code unit or file."""

    model_config = ConfigDict(frozen=True)

    link_id: str
    source_doc_id: str
    source_span: str
    target_path: str | None = None
    target_file_id: str | None = None
    target_unit_id: str | None = None


class DocumentReference(BaseModel):
    """An auditable reference edge discovered in the document network."""

    model_config = ConfigDict(frozen=True)

    reference_id: str
    source_doc_id: str
    source_span: str
    target_path: str
    reference_type: ReferenceType
    discovered_by: str = "markdown_reference_extractor"
    depth: int = 0
    target_file_id: str | None = None
    target_doc_id: str | None = None
    resolution_status: ReferenceResolutionStatus = ReferenceResolutionStatus.UNRESOLVED


class PackageManifest(BaseModel):
    """Top-level manifest describing a skill package."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    root_doc: str
    files: list[FileEntry]
    documents: list[DocumentNode]
    document_references: list[DocumentReference] = []
    code_units: list[CodeUnit]
    links: list[PackageLink]
    declared_artifact_targets: list[str] = []


class CodeDatabaseRef(BaseModel):
    """A language-specific CodeQL database and its primary SARIF output."""

    model_config = ConfigDict(frozen=True)

    language: str
    db_path: str
    sarif_path: str
    source_fingerprint: str


class CodePack(BaseModel):
    """Code-side inventory and analysis references for one skill package."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    staged_root: str
    unit_paths: dict[str, str]
    databases: list[CodeDatabaseRef] = []


# ---------------------------------------------------------------------------
# 2. Contract-side objects
# ---------------------------------------------------------------------------


class EvidenceSpan(BaseModel):
    """A locatable span of text in a document, serving as evidence."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    start_offset: int
    end_offset: int
    text: str


class Constraint(BaseModel):
    """A scope or condition attached to a clause."""

    model_config = ConfigDict(frozen=True)

    constraint_id: str
    constraint_type: str
    value: str
    evidence: EvidenceSpan | None = None


class Clause(BaseModel):
    """A canonical contract clause after voting and canonicalization."""

    model_config = ConfigDict(frozen=True)

    clause_id: str
    capability: str
    operator: ClauseOperator
    role: ClauseRole = ClauseRole.POLICY
    target: str | None = None
    constraints: list[Constraint] = []
    evidence_spans: list[EvidenceSpan] = []
    vote_agreement: float = 0.0
    step_ids: list[str] = []
    source_doc_ids: list[str] = []


class Step(BaseModel):
    """A documentation step or policy-bearing semantic unit."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    doc_id: str
    order_index: int
    local_index: int
    step_type: str
    text: str
    block_ids: list[str] = []
    heading_context: str = ""
    evidence: EvidenceSpan | None = None


class StepOrderEdge(BaseModel):
    """A lightweight precedence edge between two extracted steps."""

    model_config = ConfigDict(frozen=True)

    edge_id: str
    source_step_id: str
    target_step_id: str
    relation: str = "precedes"


class ClauseSample(BaseModel):
    """Raw clause output from a single ICCM LLM invocation.

    This model also serves as the JSON Schema for structured LLM output
    via ``response_format``.
    """

    model_config = ConfigDict(frozen=True)

    step_id: str
    capability: str
    operator: ClauseOperator
    target: str | None = None
    constraint: str | None = None
    evidence_span: str
    confidence_note: str


class ContractTable(BaseModel):
    """Aggregated contract table for a single skill package."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    clauses: list[Clause]
    steps: list[Step] = []
    step_order_edges: list[StepOrderEdge] = []
    unresolved_references: list[str] = []
    cross_doc_conflicts: list[str] = []


# ---------------------------------------------------------------------------
# 3. Behavior-side objects
# ---------------------------------------------------------------------------


class CapabilityEvent(BaseModel):
    """A normalized capability event observed in code."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    unit_id: str
    capability: str
    api_call: str = ""
    location: str
    arguments: list[str] = []
    tier: str = ""
    language: str = ""
    file_path: str = ""
    line: int = 0
    detail: str = ""


class ResourceUse(BaseModel):
    """A resource reference observed in code."""

    model_config = ConfigDict(frozen=True)

    resource_id: str
    unit_id: str
    event_id: str | None = None
    resource_type: str
    value: str
    resolved: bool = True
    location: str = ""
    origin_kind: str = "unknown"
    origin_hint: str = ""


class Bridge(BaseModel):
    """A cross-unit connection between code units."""

    model_config = ConfigDict(frozen=True)

    bridge_id: str
    source_unit_id: str
    target_unit_id: str
    kind: BridgeKind
    evidence: str = ""
    source_event_id: str | None = None
    target_event_id: str | None = None


class OrchestrationHypothesis(BaseModel):
    """A hypothesis about instruction-conditioned script orchestration."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    instruction_evidence: EvidenceSpan
    target_unit_ids: list[str]
    status: HypothesisStatus = HypothesisStatus.UNRESOLVED


class PathSegment(BaseModel):
    """A single segment in a source-to-sink path."""

    model_config = ConfigDict(frozen=True)

    unit_id: str
    location: str
    label: str
    event_id: str | None = None
    is_orchestration_conditioned: bool = False
    slot_kind: str = ""
    symbol_hint: str = ""
    expr_hint: str = ""


class RiskPath(BaseModel):
    """A source-to-sink risk path through code."""

    model_config = ConfigDict(frozen=True)

    path_id: str
    source: PathSegment
    sink: PathSegment
    segments: list[PathSegment]
    bridges_used: list[str] = []
    orchestration_hypotheses: list[str] = []
    evidence_level: str = ""
    path_kind: str = ""


class DataObject(BaseModel):
    """An explicit data-carrying object derived from behavior paths."""

    model_config = ConfigDict(frozen=True)

    object_id: str
    unit_id: str
    location_id: str | None = None
    event_id: str | None = None
    path_id: str | None = None
    object_kind: str
    label: str
    abstraction_level: str = "path_step"
    slot_kind: str = ""
    symbol_hint: str = ""
    expr_hint: str = ""
    origin_kind: str = "unknown"
    origin_hint: str = ""


class DataFlowEdge(BaseModel):
    """A directed data-flow edge between explicit data objects."""

    model_config = ConfigDict(frozen=True)

    edge_id: str
    source_object_id: str
    target_object_id: str
    flow_kind: str
    evidence_level: str = ""
    path_id: str | None = None


class Operation(BaseModel):
    """A concrete operation site associated with a capability event."""

    model_config = ConfigDict(frozen=True)

    operation_id: str
    unit_id: str
    event_id: str
    operation_type: str
    summary: str = ""
    location_id: str | None = None


class LocationNodeRecord(BaseModel):
    """A concrete file/line location materialized in the behavior graph."""

    model_config = ConfigDict(frozen=True)

    location_id: str
    file_path: str
    line: int | None = None
    raw_location: str = ""


class SourceEndpoint(BaseModel):
    """A source-side endpoint abstracted from a behavior event."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    event_id: str
    object_id: str | None = None
    capability: str
    label: str


class SinkEndpoint(BaseModel):
    """A sink-side endpoint abstracted from a behavior event."""

    model_config = ConfigDict(frozen=True)

    sink_id: str
    event_id: str
    object_id: str | None = None
    capability: str
    label: str


# ---------------------------------------------------------------------------
# 4. Cross-modal objects
# ---------------------------------------------------------------------------


class ReconciliationEdge(BaseModel):
    """A projection edge derived from proof-carrying reconciliation judgments."""

    model_config = ConfigDict(frozen=True)

    edge_id: str
    relation: RelationKind
    clause_id: str | None = None
    constraint_id: str | None = None
    step_id: str | None = None
    code_unit_id: str | None = None
    event_id: str | None = None
    path_id: str | None = None
    resource_id: str | None = None
    predicate_satisfied: bool = True
    source_judgment_ids: list[str] = []
    candidate_sources: list[CandidateSource] = []
    route_support_type: RouteSupport | None = None
    bridge_ids: list[str] = []
    orchestration_hypothesis_ids: list[str] = []
    orchestration_provenance: str | None = None
    support_strength: SupportStrength | None = None

    @model_validator(mode="after")
    def _check_has_behavior_target(self) -> ReconciliationEdge:
        if self.relation in {
            RelationKind.SUPPORTS,
            RelationKind.POTENTIALLY_SUPPORTS,
            RelationKind.CONTRADICTS,
            RelationKind.RELATES_TO,
        }:
            if not self.clause_id or not self.event_id:
                raise ValueError(
                    "clause-event edges require clause_id and event_id"
                )
        elif self.relation in {RelationKind.SCOPE_MATCHES, RelationKind.SCOPE_VIOLATES}:
            if not self.constraint_id or not self.resource_id:
                raise ValueError(
                    "scope_* edges require constraint_id and resource_id"
                )
        elif self.relation == RelationKind.ALIGNS:
            if not self.step_id or not self.code_unit_id:
                raise ValueError("aligns edges require step_id and code_unit_id")
        if self.support_strength is None:
            object.__setattr__(self, "support_strength", _infer_edge_support_strength(self))
        return self


class CandidatePair(BaseModel):
    """An auditable candidate pair before predicate materialization."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    clause_id: str
    behavior_kind: BehaviorKind
    event_id: str | None = None
    resource_id: str | None = None
    path_id: str | None = None
    candidate_sources: list[CandidateSource] = []
    shared_signals: dict[str, str] = {}
    route_refs: list[str] = []

    @model_validator(mode="after")
    def _check_has_behavior_ref(self) -> CandidatePair:
        if not any([self.event_id, self.resource_id, self.path_id]):
            raise ValueError(
                "CandidatePair must reference at least one behavior object "
                "(event_id, resource_id, or path_id)"
            )
        return self


class StepUnitCandidate(BaseModel):
    """An auditable step-to-code-unit candidate before alignment materialization."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    step_id: str
    code_unit_id: str
    candidate_sources: list[CandidateSource] = []
    shared_signals: dict[str, str] = {}


class PredicateTrace(BaseModel):
    """Audit record for a single predicate evaluation on a candidate pair."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    candidate_id: str
    predicate_name: str
    result: PredicateResult
    evidence_refs: list[str] = []
    notes: str = ""


class ReconciliationJudgment(BaseModel):
    """A proof-carrying reconciliation judgment with explicit premises."""

    model_config = ConfigDict(frozen=True)

    judgment_id: str
    kind: JudgmentKind
    result: PredicateResult
    subject_refs: list[str]
    premise_judgment_ids: list[str] = []
    evidence_refs: list[str] = []
    candidate_id: str | None = None
    route_support_type: RouteSupport | None = None
    abstain_reason: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# 5. Explanation objects
# ---------------------------------------------------------------------------


class Finding(BaseModel):
    """A confirmed declaration-implementation inconsistency."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    finding_type: FindingType
    support_level: FindingSupportLevel = FindingSupportLevel.DIAGNOSTIC
    support_strength: SupportStrength = SupportStrength.NONE
    certificate_ids: list[str] = []
    supporting_judgment_ids: list[str] = []
    supporting_edge_ids: list[str] = []
    related_clause_ids: list[str] = []
    related_constraint_ids: list[str] = []
    related_event_ids: list[str] = []
    related_resource_ids: list[str] = []
    related_path_ids: list[str] = []
    rationale: str = ""


class Diagnostic(BaseModel):
    """An auxiliary diagnostic signal that is not a confirmed inconsistency."""

    model_config = ConfigDict(frozen=True)

    diagnostic_id: str
    diagnostic_type: DiagnosticType
    support_level: FindingSupportLevel = FindingSupportLevel.DIAGNOSTIC
    support_strength: SupportStrength = SupportStrength.NONE
    certificate_ids: list[str] = []
    supporting_judgment_ids: list[str] = []
    supporting_edge_ids: list[str] = []
    related_clause_ids: list[str] = []
    related_constraint_ids: list[str] = []
    related_event_ids: list[str] = []
    related_resource_ids: list[str] = []
    related_path_ids: list[str] = []
    rationale: str = ""


class Exposure(BaseModel):
    """A documented high-risk behavior surfaced separately from violations."""

    model_config = ConfigDict(frozen=True)

    exposure_id: str
    exposure_type: ExposureType
    support_level: FindingSupportLevel = FindingSupportLevel.DIAGNOSTIC
    support_strength: SupportStrength = SupportStrength.NONE
    supporting_judgment_ids: list[str] = []
    supporting_edge_ids: list[str] = []
    related_clause_ids: list[str] = []
    related_event_ids: list[str] = []
    related_resource_ids: list[str] = []
    related_path_ids: list[str] = []
    rationale: str = ""


class Certificate(BaseModel):
    """A structured certificate derived from reconciliation judgments."""

    model_config = ConfigDict(frozen=True)

    certificate_id: str
    kind: CertificateKind
    finding_id: str | None = None
    subject_refs: list[str] = []
    supporting_judgment_ids: list[str] = []
    evidence_refs: list[str] = []
    notes: str = ""


class Witness(BaseModel):
    """A minimal, certificate-closed proof replay artifact."""

    model_config = ConfigDict(frozen=True)

    witness_id: str
    finding_id: str
    support_strength: SupportStrength = SupportStrength.NONE
    anchor_ids: list[str] = []
    fact_node_ids: list[str] = []
    judgment_ids: list[str] = []
    certificate_ids: list[str] = []
    projection_edge_ids: list[str] = []
    rank_tuple: tuple[int | float, ...] = ()
    is_exact: bool = False
    revalidation_passed: bool = False


class WitnessContextLink(BaseModel):
    """A diagnostic attached to a witness as auxiliary context only."""

    model_config = ConfigDict(frozen=True)

    diagnostic_id: str
    diagnostic_type: DiagnosticType
    support_level: FindingSupportLevel = FindingSupportLevel.DIAGNOSTIC
    support_strength: SupportStrength = SupportStrength.NONE
    shared_fact_ids: list[str] = []
    shared_judgment_ids: list[str] = []
    shared_certificate_ids: list[str] = []
    shared_projection_edge_ids: list[str] = []
    rationale: str = ""


class WitnessContext(BaseModel):
    """Auxiliary diagnostics surrounding a confirmed minimal witness."""

    model_config = ConfigDict(frozen=True)

    context_id: str
    witness_id: str
    finding_id: str
    diagnostic_links: list[WitnessContextLink] = []


class WitnessValidation(BaseModel):
    """Validator result for a generated witness."""

    model_config = ConfigDict(frozen=True)

    witness_id: str
    passed: bool
    failure_reason: str | None = None
    checked_anchor_ids: list[str] = []
    checked_certificate_ids: list[str] = []


class PermissionManifestEntry(BaseModel):
    """Human-auditable summary of authorization and abstention outcomes."""

    model_config = ConfigDict(frozen=True)

    manifest_id: str
    subject_refs: list[str]
    status: str
    supporting_judgment_ids: list[str] = []
    certificate_ids: list[str] = []
    notes: str = ""


def _infer_edge_support_strength(edge: ReconciliationEdge) -> SupportStrength:
    if edge.relation in {
        RelationKind.POTENTIALLY_SUPPORTS,
        RelationKind.RELATES_TO,
    } or not edge.predicate_satisfied:
        if edge.route_support_type in {
            RouteSupport.ORCHESTRATION_CONFIRMED,
            RouteSupport.ORCHESTRATION_UNRESOLVED,
        } or edge.orchestration_hypothesis_ids:
            return SupportStrength.WEAK_ORCHESTRATED
        return SupportStrength.WEAK_STRUCTURAL
    return SupportStrength.STRONG


# ---------------------------------------------------------------------------
# 6. Graph artifact models
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A lightweight typed graph node with extensible attributes."""

    model_config = ConfigDict(frozen=True)

    node_id: str
    kind: str
    attrs: dict[str, object] = {}


class GraphEdge(BaseModel):
    """A lightweight typed graph edge with extensible attributes."""

    model_config = ConfigDict(frozen=True)

    edge_id: str | None = None
    kind: str
    source: str
    target: str
    attrs: dict[str, object] = {}


class GraphObject(BaseModel):
    """In-memory graph object used before JSON artifact serialization."""

    model_config = ConfigDict(frozen=True)

    nodes: list[GraphNode]
    edges: list[GraphEdge]

    def node_by_id(self) -> dict[str, GraphNode]:
        return {node.node_id: node for node in self.nodes}

    def out_edges(self, node_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.source == node_id]

    def in_edges(self, node_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.target == node_id]

    def neighbors(self, node_id: str) -> set[str]:
        return {edge.target for edge in self.out_edges(node_id)}

    def to_wire(self, *, node_kind_field: str = "kind") -> dict[str, object]:
        return {
            "nodes": [
                {
                    "id": node.node_id,
                    node_kind_field: node.kind,
                    **node.attrs,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    **({"edge_id": edge.edge_id} if edge.edge_id else {}),
                    "source": edge.source,
                    "target": edge.target,
                    "kind": edge.kind,
                    **edge.attrs,
                }
                for edge in self.edges
            ],
        }


# ---------------------------------------------------------------------------
# 7. Intermediate artifact models (M1+)
# ---------------------------------------------------------------------------


class ReferenceLink(BaseModel):
    """A reference from one document to another file."""

    model_config = ConfigDict(frozen=True)

    source_doc_id: str
    target_path: str
    reference_type: ReferenceType
    source_span: str


class DocBlock(BaseModel):
    """A parsed block within a document, with character offsets."""

    model_config = ConfigDict(frozen=True)

    block_id: str
    doc_id: str
    block_type: str
    content: str
    start_offset: int
    end_offset: int
    heading_context: str = ""
    language_hint: str = ""


class DocumentPack(BaseModel):
    """Collection of admitted documents and their parsed blocks."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    admitted_docs: list[DocumentNode]
    doc_blocks: list[DocBlock]
    unresolved_references: list[str] = []
    declared_artifact_targets: list[str] = []


class ClauseSampleList(BaseModel):
    """Wrapper for LLM structured output: multiple clauses per call."""

    model_config = ConfigDict(frozen=True)

    clauses: list[ClauseSample]
