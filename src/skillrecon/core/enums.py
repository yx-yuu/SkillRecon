"""Shared enumerations for SkillRecon scientific objects."""

from enum import StrEnum


class ClauseOperator(StrEnum):
    """Operator in a contract clause (unknown-by-default)."""

    ALLOWED = "allowed"
    PROHIBITED = "prohibited"
    UNKNOWN = "unknown"


class ClauseRole(StrEnum):
    """Whether a clause is policy-bearing or background knowledge."""

    POLICY = "policy"
    KNOWLEDGE = "knowledge"


class BridgeKind(StrEnum):
    """Kind of cross-unit bridge."""

    STATIC = "static"
    ARTIFACT = "artifact"
    COMMAND = "command"


class RelationKind(StrEnum):
    """Typed cross-modal relation between contract and behavior objects."""

    SUPPORTS = "supports"
    POTENTIALLY_SUPPORTS = "potentially_supports"
    CONTRADICTS = "contradicts"
    RELATES_TO = "relates_to"
    SCOPE_MATCHES = "scope_matches"
    SCOPE_VIOLATES = "scope_violates"
    ALIGNS = "aligns"


class FindingType(StrEnum):
    """Types of findings produced by the witness engine."""

    UNSUPPORTED_BEHAVIOR = "unsupported_behavior"
    CONTRADICTED_BEHAVIOR = "contradicted_behavior"
    SCOPE_VIOLATION = "scope_violation"
    UNJUSTIFIED_COMPOSITION = "unjustified_composition"


class ExposureType(StrEnum):
    """Types of declared-but-high-risk exposures surfaced separately."""

    DECLARED_SENSITIVE_BEHAVIOR = "declared_sensitive_behavior"
    DECLARED_SENSITIVE_COMPOSITION = "declared_sensitive_composition"


class DiagnosticType(StrEnum):
    """Auxiliary diagnostics that are not confirmed inconsistency findings."""

    POLICY_GAP = "policy_gap"
    ALIGNMENT_MISMATCH = "alignment_mismatch"
    PARTIAL_PATH_COVERAGE = "partial_path_coverage"
    PLANNED_BEHAVIOR_GAP = "planned_behavior_gap"
    UNRESOLVED_ROUTE = "unresolved_route"


class FindingSupportLevel(StrEnum):
    """Whether a finding is backed by projection edges or only diagnostics."""

    GRAPH_BACKED = "graph_backed"
    DIAGNOSTIC = "diagnostic"


class SupportStrength(StrEnum):
    """Granularity of graph support beyond graph-backed vs diagnostic."""

    NONE = "none"
    STRONG = "strong"
    WEAK_STRUCTURAL = "weak_structural"
    WEAK_ORCHESTRATED = "weak_orchestrated"
    MIXED = "mixed"


class CertificateKind(StrEnum):
    """Certificate types for witness validation."""

    NO_SUPPORT = "no_support"
    BLOCKING_PROHIBITION = "blocking_prohibition"
    SCOPE_FAILURE = "scope_failure"
    NO_JUSTIFIED_PATH = "no_justified_path"
    UNRESOLVED_ROUTE = "unresolved_route"


class HypothesisStatus(StrEnum):
    """Status of an orchestration hypothesis."""

    CONFIRMED = "confirmed"
    COMPETING = "competing"
    UNRESOLVED = "unresolved"
    ABSTAIN = "abstain"


class FileKind(StrEnum):
    """File type classification within a skill package."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    BASH = "bash"
    MARKDOWN = "markdown"
    JSON = "json"
    OTHER = "other"


class ReferenceType(StrEnum):
    """How a document reference was discovered."""

    EXPLICIT_LINK = "explicit_link"
    INLINE_PATH = "inline_path"
    CODE_BLOCK_REF = "code_block_ref"


class ReferenceResolutionStatus(StrEnum):
    """Resolution status for a discovered document reference."""

    ADMITTED = "admitted"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Reconciliation-specific enums (Module 03)
# ---------------------------------------------------------------------------


class PredicateResult(StrEnum):
    """Three-valued result for reconciliation predicates."""

    TRUE = "true"
    FALSE = "false"
    ABSTAIN = "abstain"


class BehaviorKind(StrEnum):
    """Which behavior object type a candidate pair targets."""

    EVENT = "event"
    RESOURCE = "resource"
    PATH = "path"


class CandidateSource(StrEnum):
    """How a candidate pair was discovered."""

    LITERAL_OVERLAP = "literal_overlap"
    TYPED_RESOURCE_FAMILY = "typed_resource_family"
    STEP_UNIT_ALIGNMENT = "step_unit_alignment"
    SECTION_ALIGNMENT = "section_alignment"
    FILE_MENTION = "file_mention"
    PACKAGE_LINK = "package_link"
    REFERENCE_ANCESTRY = "reference_ancestry"
    BRIDGE_PROVENANCE = "bridge_provenance"
    ORCHESTRATION_PROVENANCE = "orchestration_provenance"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"


class RouteSupport(StrEnum):
    """Evidence strength for execution route justification."""

    LOCAL = "local"
    CODEQL = "codeql"
    BRIDGE = "bridge"
    ORCHESTRATION_CONFIRMED = "orchestration_confirmed"
    ORCHESTRATION_UNRESOLVED = "orchestration_unresolved"
    WEAK = "weak"


class JudgmentKind(StrEnum):
    """Kinds of proof-carrying reconciliation judgments."""

    CAPABILITY_OVERLAP = "capability_overlap"
    RESOURCE_COMPATIBILITY = "resource_compatibility"
    SCOPE_CHECK = "scope_check"
    ROUTE_CHECK = "route_check"
    PROHIBITION_CHECK = "prohibition_check"
    AUTHORIZATION = "authorization"
    PATH_JUSTIFICATION = "path_justification"
