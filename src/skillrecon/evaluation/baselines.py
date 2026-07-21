"""Baseline interfaces and lightweight baseline adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from skillrecon.core.config import AnalyzerConfig
from skillrecon.core.enums import ClauseOperator
from skillrecon.core.sensitivity import event_capability_family, event_requires_authorization
from skillrecon.core.types import CapabilityEvent, Clause, ResourceUse, RiskPath
from skillrecon.llm.cache import CachedLLMClient
from skillrecon.evaluation.datasets import BaselinePredictionRecord
from skillrecon.evaluation.skillrecon_adapter import build_skillrecon_report
from skillrecon.evaluation.types import EvaluationFinding, EvaluationReport, ReportSummary


class _LLMJudgeDecision(BaseModel):
    """Structured output schema for the LLM-as-judge baseline."""

    model_config = ConfigDict(frozen=True)

    main_label: Literal["violation", "exposure-only", "benign"]
    subtype: str | None = None
    rationale: str


_LLM_JUDGE_SYSTEM_PROMPT = """You are a documentation-code consistency judge for AI agent skills.
Classify the skill as exactly one of:
- violation
- exposure-only
- benign

Definitions:
- violation: observed implementation is not justified by documentation
- exposure-only: the behavior is documented but still high risk
- benign: no meaningful documentation-implementation inconsistency is observed

If and only if main_label = violation, subtype must be exactly one of:
- unsupported_behavior
- scope_violation
- unjustified_composition

Subtype rubric:
- unsupported_behavior: the implementation performs a sensitive capability or action not justified by the documented contract.
- scope_violation: the capability is documented, but the observed concrete target, path, host, or resource exceeds the documented scope.
- unjustified_composition: the individual steps may appear partially documented, but the end-to-end multi-step chain or source-to-sink composition is not justified.

If main_label is exposure-only or benign, subtype must be null.
Return only the structured decision."""


_LLM_JUDGE_CANONICAL_SUBTYPES = {
    "unsupported_behavior",
    "scope_violation",
    "unjustified_composition",
}
_LLM_JUDGE_SUBTYPE_ALIASES = {
    "undeclared_high_impact_capability": "unsupported_behavior",
    "unsupported_behavior": "unsupported_behavior",
    "contradicted_behavior": "unsupported_behavior",
    "scope_violation": "scope_violation",
    "unjustified_dangerous_composition": "unjustified_composition",
    "unjustified_composition": "unjustified_composition",
}


def build_skillrecon_baseline_report(skill_id: str, artifact_dir: Path) -> EvaluationReport:
    """Treat the main SkillRecon output as one evaluation system."""
    return build_skillrecon_report(skill_id, artifact_dir)


def build_rule_based_scanner_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """A documentation-agnostic sink/path baseline over behavior artifacts only."""
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)
    paths = _load_models(artifact_dir / "path_table.json", RiskPath, allow_missing=True)
    findings: list[EvaluationFinding] = []

    for event in events:
        if not event_requires_authorization(event, _load_a_req(analyzer_config)):
            continue
        findings.append(
            EvaluationFinding(
                finding_id=f"rule-event::{event.event_id}",
                main_label="violation",
                subtype="unsupported_behavior",
                capability_atoms=[event.capability],
                code_locations=[],
                rationale="Rule baseline flagged an authorization-sensitive behavior.",
            )
        )

    for path in paths:
        sink_event_id = path.sink.event_id
        if sink_event_id is None:
            continue
        sink_event = next((event for event in events if event.event_id == sink_event_id), None)
        if sink_event is None or not event_requires_authorization(sink_event, _load_a_req(analyzer_config)):
            continue
        findings.append(
            EvaluationFinding(
                finding_id=f"rule-path::{path.path_id}",
                main_label="violation",
                subtype="unjustified_composition",
                capability_atoms=[path.sink.label],
                rationale="Rule baseline flagged a sensitive source-to-sink path.",
            )
        )

    overall_label = "violation" if findings else "benign"
    return EvaluationReport(
        skill_id=skill_id,
        system_id="baseline_rule_scanner",
        analyzer_version="baseline-rule-v1",
        overall_label=overall_label,
        violation_findings=findings,
        summary=ReportSummary(violation_count=len(findings)),
    )


def build_capability_lattice_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """A coarse declared-vs-inferred capability-set baseline."""
    contract_payload = json.loads((artifact_dir / "contract_table.json").read_text(encoding="utf-8"))
    clauses = [Clause.model_validate(item) for item in contract_payload.get("clauses", [])]
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)

    declared_caps = {clause.capability for clause in clauses}
    findings: list[EvaluationFinding] = []
    seen_families: set[str] = set()
    for event in events:
        if not event_requires_authorization(event, _load_a_req(analyzer_config)):
            continue
        cap_family = event_capability_family(event)
        if event.capability in declared_caps or cap_family in declared_caps or cap_family in seen_families:
            continue
        seen_families.add(cap_family)
        findings.append(
            EvaluationFinding(
                finding_id=f"cap-lattice::{event.event_id}",
                main_label="violation",
                subtype="unsupported_behavior",
                capability_atoms=[event.capability],
                rationale=(
                    "Capability-lattice baseline observed an inferred capability family "
                    "outside the declared capability set."
                ),
            )
        )

    overall_label = "violation" if findings else "benign"
    return EvaluationReport(
        skill_id=skill_id,
        system_id="baseline_capability_lattice",
        analyzer_version="baseline-cap-v1",
        overall_label=overall_label,
        violation_findings=findings,
        summary=ReportSummary(violation_count=len(findings)),
    )


def build_doc_code_consistency_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """Paper-inspired doc-code consistency baseline.

    This adapter follows the spirit of documentation-code mismatch systems:
    compare recovered documentation clauses to observed code behavior at the
    pair level. It deliberately does not use SkillRecon reconciliation edges,
    path certificates, or witnesses.
    """
    clauses = _load_clauses(artifact_dir)
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)
    resources = _load_models(artifact_dir / "resource_table.json", ResourceUse, allow_missing=True)
    a_req = _load_a_req(analyzer_config)
    cap_to_family = _taxonomy_capability_families(analyzer_config)
    resources_by_event = _resources_by_event(resources)

    findings: list[EvaluationFinding] = []
    for event in events:
        if not event_requires_authorization(event, a_req):
            continue
        if _has_matching_prohibition(event, clauses, cap_to_family):
            findings.append(
                _baseline_finding(
                    system_prefix="doc-code",
                    finding_id=event.event_id,
                    subtype="unsupported_behavior",
                    capability=event.capability,
                    rationale=(
                        "Doc-code consistency baseline found a sensitive code "
                        "behavior that conflicts with a documented prohibition."
                    ),
                )
            )
            continue

        matching_allowed = [
            clause
            for clause in clauses
            if clause.operator == ClauseOperator.ALLOWED
            and _capability_matches_clause(event, clause, cap_to_family)
        ]
        if not matching_allowed:
            findings.append(
                _baseline_finding(
                    system_prefix="doc-code",
                    finding_id=event.event_id,
                    subtype="unsupported_behavior",
                    capability=event.capability,
                    rationale=(
                        "Doc-code consistency baseline observed an "
                        "authorization-sensitive behavior without a matching "
                        "documentation clause."
                    ),
                )
            )
            continue

        event_resources = resources_by_event.get(event.event_id, [])
        if event_resources and _all_matching_clauses_reject_resources(
            matching_allowed,
            event_resources,
        ):
            findings.append(
                _baseline_finding(
                    system_prefix="doc-code",
                    finding_id=event.event_id,
                    subtype="scope_violation",
                    capability=event.capability,
                    rationale=(
                        "Doc-code consistency baseline found a capability "
                        "match but no clause whose resource text covers the "
                        "observed target."
                    ),
                )
            )

    return _report_from_findings(
        skill_id=skill_id,
        system_id="baseline_doc_code_consistency",
        analyzer_version="baseline-doc-code-v1",
        findings=_dedupe_findings(findings),
    )


def build_spec_containment_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """Paper-inspired specification/capability containment baseline."""
    clauses = _load_clauses(artifact_dir)
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)
    a_req = _load_a_req(analyzer_config)
    cap_to_family = _taxonomy_capability_families(analyzer_config)
    declared = _declared_allowed_families(clauses, cap_to_family)
    prohibited = _declared_prohibited_families(clauses, cap_to_family)

    findings: list[EvaluationFinding] = []
    seen: set[str] = set()
    for event in events:
        if not event_requires_authorization(event, a_req):
            continue
        family = _event_family(event, cap_to_family)
        if family in seen:
            continue
        if family in prohibited or family not in declared:
            seen.add(family)
            findings.append(
                _baseline_finding(
                    system_prefix="spec-containment",
                    finding_id=event.event_id,
                    subtype="unsupported_behavior",
                    capability=event.capability,
                    rationale=(
                        "Specification-containment baseline found an inferred "
                        "sensitive capability family outside the documented "
                        "allowed set."
                    ),
                )
            )

    return _report_from_findings(
        skill_id=skill_id,
        system_id="baseline_spec_containment",
        analyzer_version="baseline-spec-containment-v1",
        findings=findings,
    )


def build_instruction_constraint_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """Paper-inspired instruction-constraint baseline.

    This baseline uses only explicit prohibitions and explicit textual scope
    constraints. It is intentionally conservative and does not perform
    cross-file path justification.
    """
    clauses = _load_clauses(artifact_dir)
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)
    resources = _load_models(artifact_dir / "resource_table.json", ResourceUse, allow_missing=True)
    a_req = _load_a_req(analyzer_config)
    cap_to_family = _taxonomy_capability_families(analyzer_config)
    resources_by_event = _resources_by_event(resources)

    findings: list[EvaluationFinding] = []
    for event in events:
        if not event_requires_authorization(event, a_req):
            continue
        if _has_matching_prohibition(event, clauses, cap_to_family):
            findings.append(
                _baseline_finding(
                    system_prefix="instruction-constraint",
                    finding_id=event.event_id,
                    subtype="unsupported_behavior",
                    capability=event.capability,
                    rationale=(
                        "Instruction-constraint baseline matched an explicit "
                        "negative documentation constraint."
                    ),
                )
            )
            continue

        constrained_allowed = [
            clause
            for clause in clauses
            if clause.operator == ClauseOperator.ALLOWED
            and _capability_matches_clause(event, clause, cap_to_family)
            and _clause_has_resource_boundary(clause)
        ]
        event_resources = resources_by_event.get(event.event_id, [])
        if constrained_allowed and event_resources and _all_matching_clauses_reject_resources(
            constrained_allowed,
            event_resources,
        ):
            findings.append(
                _baseline_finding(
                    system_prefix="instruction-constraint",
                    finding_id=event.event_id,
                    subtype="scope_violation",
                    capability=event.capability,
                    rationale=(
                        "Instruction-constraint baseline found an explicit "
                        "scope boundary that does not cover the observed "
                        "resource."
                    ),
                )
            )

    return _report_from_findings(
        skill_id=skill_id,
        system_id="baseline_instruction_constraints",
        analyzer_version="baseline-instruction-constraints-v1",
        findings=_dedupe_findings(findings),
    )


def build_capability_only_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """Internal granularity variant that only checks declared capabilities."""
    return build_spec_containment_report(
        skill_id=skill_id,
        artifact_dir=artifact_dir,
        analyzer_config=analyzer_config,
    ).model_copy(
        update={
            "system_id": "granularity_capability_only",
            "analyzer_version": "granularity-capability-only-v1",
        }
    )


def build_resource_aware_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
) -> EvaluationReport:
    """Internal granularity variant with capability and resource matching."""
    return build_doc_code_consistency_report(
        skill_id=skill_id,
        artifact_dir=artifact_dir,
        analyzer_config=analyzer_config,
    ).model_copy(
        update={
            "system_id": "granularity_resource_aware",
            "analyzer_version": "granularity-resource-aware-v1",
        }
    )


def build_llm_judge_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    analyzer_config: AnalyzerConfig,
    client: CachedLLMClient | None = None,
) -> EvaluationReport:
    """An end-to-end LLM-as-judge baseline over summarized artifacts."""
    client = client or CachedLLMClient.from_config(
        analyzer_config.llm,
        "evaluation_llm_judge_v2",
    )
    messages = [
        {"role": "system", "content": _LLM_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_llm_judge_prompt(
                skill_id=skill_id,
                artifact_dir=artifact_dir,
            ),
        },
    ]
    decision = client.structured_complete(
        messages,
        _LLMJudgeDecision,
        skill_id=skill_id,
        call_key="evaluation_llm_judge_v2",
    )
    normalized_subtype = _normalize_llm_judge_subtype(
        main_label=decision.main_label,
        subtype=decision.subtype,
        rationale=decision.rationale,
    )

    violation_findings: list[EvaluationFinding] = []
    exposure_findings: list[EvaluationFinding] = []
    if decision.main_label == "violation":
        violation_findings.append(
            EvaluationFinding(
                finding_id=f"llm-judge::{skill_id}",
                main_label="violation",
                subtype=normalized_subtype,
                rationale=decision.rationale,
            )
        )
    elif decision.main_label == "exposure-only":
        exposure_findings.append(
            EvaluationFinding(
                finding_id=f"llm-judge::{skill_id}",
                main_label="exposure-only",
                subtype=None,
                rationale=decision.rationale,
            )
        )

    return EvaluationReport(
        skill_id=skill_id,
        system_id="baseline_llm_judge",
        analyzer_version="baseline-llm-judge-v1",
        overall_label=decision.main_label,
        violation_findings=violation_findings,
        exposure_findings=exposure_findings,
        summary=ReportSummary(
            violation_count=len(violation_findings),
            exposure_count=len(exposure_findings),
        ),
    )


def build_external_prediction_report(
    *,
    skill_id: str,
    system_id: str,
    prediction: BaselinePredictionRecord,
) -> EvaluationReport:
    """Wrap an externally supplied baseline prediction into EvaluationReport."""
    violation_findings: list[EvaluationFinding] = []
    exposure_findings: list[EvaluationFinding] = []
    if prediction.main_label == "violation":
        violation_findings.append(
            EvaluationFinding(
                finding_id=f"{system_id}::{skill_id}",
                main_label="violation",
                subtype=prediction.subtype,
                rationale=prediction.rationale,
            )
        )
    elif prediction.main_label == "exposure-only":
        exposure_findings.append(
            EvaluationFinding(
                finding_id=f"{system_id}::{skill_id}",
                main_label="exposure-only",
                subtype=prediction.subtype,
                rationale=prediction.rationale,
            )
        )

    return EvaluationReport(
        skill_id=skill_id,
        system_id=system_id,
        analyzer_version=f"{system_id}-external-v1",
        overall_label=prediction.main_label,
        violation_findings=violation_findings,
        exposure_findings=exposure_findings,
        summary=ReportSummary(
            violation_count=len(violation_findings),
            exposure_count=len(exposure_findings),
        ),
    )


def index_baseline_predictions(
    predictions: list[BaselinePredictionRecord],
) -> dict[tuple[str, str], BaselinePredictionRecord]:
    """Index external baseline predictions by (system_id, skill_id)."""
    return {(item.system_id, item.skill_id): item for item in predictions}


def _load_models(path: Path, model_cls: type, *, allow_missing: bool = False) -> list:
    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list artifact in {path}")
    return [model_cls.model_validate(item) for item in payload]


def _load_clauses(artifact_dir: Path) -> list[Clause]:
    payload = json.loads((artifact_dir / "contract_table.json").read_text(encoding="utf-8"))
    return [Clause.model_validate(item) for item in payload.get("clauses", [])]


def _load_a_req(config: AnalyzerConfig) -> set[str]:
    payload = json.loads(config.taxonomy_path.read_text(encoding="utf-8"))
    return set(payload.get("a_req", []))


def _taxonomy_capability_families(config: AnalyzerConfig) -> dict[str, str]:
    payload = json.loads(config.taxonomy_path.read_text(encoding="utf-8"))
    families: dict[str, str] = {}
    categories = payload.get("categories", {})
    if isinstance(categories, dict):
        for family, spec in categories.items():
            atoms = spec.get("atoms", []) if isinstance(spec, dict) else []
            for atom in atoms:
                if isinstance(atom, str):
                    families[atom] = str(family)
    return families


def _event_family(event: CapabilityEvent, cap_to_family: dict[str, str]) -> str:
    return cap_to_family.get(event.capability, event_capability_family(event))


def _clause_family(clause: Clause, cap_to_family: dict[str, str]) -> str:
    return cap_to_family.get(clause.capability, clause.capability)


def _declared_allowed_families(
    clauses: list[Clause],
    cap_to_family: dict[str, str],
) -> set[str]:
    return {
        _clause_family(clause, cap_to_family)
        for clause in clauses
        if clause.operator == ClauseOperator.ALLOWED
    }


def _declared_prohibited_families(
    clauses: list[Clause],
    cap_to_family: dict[str, str],
) -> set[str]:
    return {
        _clause_family(clause, cap_to_family)
        for clause in clauses
        if clause.operator == ClauseOperator.PROHIBITED
    }


def _capability_matches_clause(
    event: CapabilityEvent,
    clause: Clause,
    cap_to_family: dict[str, str],
) -> bool:
    return (
        event.capability == clause.capability
        or _event_family(event, cap_to_family) == _clause_family(clause, cap_to_family)
    )


def _has_matching_prohibition(
    event: CapabilityEvent,
    clauses: list[Clause],
    cap_to_family: dict[str, str],
) -> bool:
    return any(
        clause.operator == ClauseOperator.PROHIBITED
        and _capability_matches_clause(event, clause, cap_to_family)
        for clause in clauses
    )


def _resources_by_event(resources: list[ResourceUse]) -> dict[str, list[ResourceUse]]:
    grouped: dict[str, list[ResourceUse]] = {}
    for resource in resources:
        if resource.event_id:
            grouped.setdefault(resource.event_id, []).append(resource)
    return grouped


def _all_matching_clauses_reject_resources(
    clauses: list[Clause],
    resources: list[ResourceUse],
) -> bool:
    bounded = [clause for clause in clauses if _clause_has_resource_boundary(clause)]
    if not bounded:
        return False
    return all(
        not _clause_covers_any_resource(clause, resources)
        for clause in bounded
    )


def _clause_has_resource_boundary(clause: Clause) -> bool:
    return bool(clause.target or clause.constraints)


def _clause_covers_any_resource(
    clause: Clause,
    resources: list[ResourceUse],
) -> bool:
    surfaces = _clause_resource_surfaces(clause)
    if not surfaces:
        return True
    for resource in resources:
        resource_value = _normalize_surface(resource.value)
        if not resource_value or resource_value.startswith("<dynamic"):
            return True
        if any(_surface_overlaps(surface, resource_value) for surface in surfaces):
            return True
    return False


def _clause_resource_surfaces(clause: Clause) -> list[str]:
    values = [clause.target or ""]
    values.extend(constraint.value for constraint in clause.constraints)
    surfaces: list[str] = []
    for value in values:
        normalized = _normalize_surface(value)
        if normalized and normalized not in surfaces:
            surfaces.append(normalized)
    return surfaces


def _normalize_surface(value: str) -> str:
    return value.strip().strip("`\"'").lower()


def _surface_overlaps(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    left_parts = {part for part in left.replace("\\", "/").split("/") if part}
    right_parts = {part for part in right.replace("\\", "/").split("/") if part}
    return bool(left_parts and right_parts and left_parts <= right_parts)


def _baseline_finding(
    *,
    system_prefix: str,
    finding_id: str,
    subtype: str,
    capability: str,
    rationale: str,
) -> EvaluationFinding:
    return EvaluationFinding(
        finding_id=f"{system_prefix}::{finding_id}",
        main_label="violation",
        subtype=subtype,
        capability_atoms=[capability],
        rationale=rationale,
    )


def _dedupe_findings(findings: list[EvaluationFinding]) -> list[EvaluationFinding]:
    seen: set[tuple[str, str | None]] = set()
    deduped: list[EvaluationFinding] = []
    for finding in findings:
        key = (finding.finding_id, finding.subtype)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _report_from_findings(
    *,
    skill_id: str,
    system_id: str,
    analyzer_version: str,
    findings: list[EvaluationFinding],
) -> EvaluationReport:
    return EvaluationReport(
        skill_id=skill_id,
        system_id=system_id,
        analyzer_version=analyzer_version,
        overall_label="violation" if findings else "benign",
        violation_findings=findings,
        summary=ReportSummary(violation_count=len(findings)),
    )


def _build_llm_judge_prompt(*, skill_id: str, artifact_dir: Path) -> str:
    contract_payload = json.loads((artifact_dir / "contract_table.json").read_text(encoding="utf-8"))
    clauses = [Clause.model_validate(item) for item in contract_payload.get("clauses", [])]
    events = _load_models(artifact_dir / "event_table.json", CapabilityEvent, allow_missing=True)
    resources = _load_models(artifact_dir / "resource_table.json", ResourceUse, allow_missing=True)
    paths = _load_models(artifact_dir / "path_table.json", RiskPath, allow_missing=True)

    clause_summary = "\n".join(
        (
            f"- {clause.operator.value} {clause.capability} "
            f"target={clause.target or 'none'} "
            f"constraints={'; '.join(constraint.value for constraint in clause.constraints[:2]) or 'none'}"
        )
        for clause in clauses[:12]
    ) or "- no clauses recovered"
    event_summary = "\n".join(
        f"- {event.capability} at {event.location} detail={event.detail or event.api_call or 'none'}"
        for event in events[:16]
    ) or "- no events recovered"
    resource_summary = "\n".join(
        f"- {resource.resource_type} value={resource.value} event={resource.event_id or 'none'} resolved={resource.resolved}"
        for resource in resources[:16]
    ) or "- no resources recovered"
    path_summary = "\n".join(
        f"- {path.source.label} -> {path.sink.label} ({path.path_kind})"
        for path in paths[:8]
    ) or "- no paths recovered"

    return "\n".join(
        [
            f"Skill: {skill_id}",
            "",
            "Recovered Documentation Contract Summary:",
            clause_summary,
            "",
            "Recovered Implementation Behavior Summary:",
            event_summary,
            "",
            "Recovered Resource Summary:",
            resource_summary,
            "",
            "Recovered Path Summary:",
            path_summary,
            "",
            "Subtype decision rubric:",
            "- unsupported_behavior: undocumented sensitive behavior or contradicted behavior.",
            "- scope_violation: documented behavior exceeds a documented scope/target/resource boundary.",
            "- unjustified_composition: the risky end-to-end path/chain is not justified even if parts are documented.",
            "- If main_label is benign or exposure-only, subtype must be null.",
            "",
            "Return the best label and subtype using only the canonical subtype strings.",
        ]
    )


def _normalize_llm_judge_subtype(
    *,
    main_label: str,
    subtype: str | None,
    rationale: str,
) -> str | None:
    if main_label != "violation":
        return None
    normalized = _normalize_subtype_token(subtype)
    if normalized is not None:
        return normalized

    text = f"{subtype or ''}\n{rationale}".lower()
    if any(token in text for token in _SCOPE_KEYWORDS):
        return "scope_violation"
    if _looks_like_composition_signal(text):
        return "unjustified_composition"
    return "unsupported_behavior"


_SCOPE_KEYWORDS = {
    "out-of-scope",
    "outside scope",
    "scope failure",
    "scope mismatch",
    "allowlist",
    "host mismatch",
    "domain mismatch",
    "target mismatch",
    "path mismatch",
    "resource-level",
}
_COMPOSITION_KEYWORDS = {
    "composition",
    "source-to-sink",
    "source to sink",
    "multi-step",
    "end-to-end",
    "workflow",
    "path-level",
    "path ",
    "chain",
    "message-sending",
    "message sending",
    "data flow",
    "flow",
}
_COMPOSITION_LEFT = {
    "env",
    "environment",
    "secret",
    "credential",
    "token",
    "file",
    "read",
    "store_access",
}
_COMPOSITION_RIGHT = {
    "http",
    "network",
    "request",
    "send",
    "upload",
    "post",
    "webhook",
    "external",
}


def _normalize_subtype_token(value: str | None) -> str | None:
    if value is None:
        return None
    token = value.strip().lower().replace("-", "_").replace(" ", "_")
    if token in _LLM_JUDGE_CANONICAL_SUBTYPES:
        return token
    aliased = _LLM_JUDGE_SUBTYPE_ALIASES.get(token)
    if aliased is not None:
        return aliased
    if any(scope_token in token for scope_token in _SCOPE_TOKEN_HINTS):
        return "scope_violation"
    if _looks_like_composition_signal(token):
        return "unjustified_composition"
    if "undocumented" in token or "contradict" in token:
        return "unsupported_behavior"
    return None


def _looks_like_composition_signal(text: str) -> bool:
    if any(token in text for token in _COMPOSITION_KEYWORDS):
        return True
    return (
        any(token in text for token in _COMPOSITION_LEFT)
        and any(token in text for token in _COMPOSITION_RIGHT)
    )


_SCOPE_TOKEN_HINTS = {
    "scope",
    "out_of_scope",
    "target_mismatch",
    "domain_mismatch",
    "host_mismatch",
    "allowlist",
}
