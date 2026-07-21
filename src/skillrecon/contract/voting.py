"""Self-consistency voting, canonicalization, and authorization guard."""

from __future__ import annotations

import logging
from collections import defaultdict

from skillrecon.contract.normalize import normalize_clause_surface
from skillrecon.core.enums import ClauseOperator
from skillrecon.core.types import (
    Clause,
    ClauseSample,
    ContractTable,
    EvidenceSpan,
    Step,
    StepOrderEdge,
)

logger = logging.getLogger(__name__)

type SemanticKey = tuple[str, str, str, str]
_GENERIC_SCOPE_TERMS = {
    "files",
    "data",
    "the internet",
    "internet",
    "the system",
    "system",
    "project files",
    "user files",
}
_CAPABILITY_CUES: dict[str, tuple[str, ...]] = {
    "shell_exec": ("bash", "shell", "command", "run", "execute"),
    "subprocess_spawn": ("subprocess", "spawn", "process", "run"),
    "http_request": ("http", "https", "request", "fetch", "api"),
    "dynamic_import": ("import", "module", "plugin", "package"),
    "env_var_read": ("env", "environment", "os.getenv", "os.environ"),
    "api_key_use": ("api key", "token", "authorization", "bearer", "secret"),
    "sql_exec": ("sql", "query", "select", "insert", "update", "delete"),
}
_AUTHORIZATION_SIGNALS = {
    "allowed-tools",
    "allowed_tools",
    "permitted",
    "authorized",
    "grant",
    "may use",
    "can use",
    "require",
    "required",
    "requires",
    "must",
    "all requests require",
    "authentication",
    "authorization",
    "endpoint",
    "api",
    "header",
    "environment variable",
    "env var",
    "调用",
    "接口",
    "必须",
    "需要",
    "环境变量",
}


def _normalize_target(target: str | None) -> str:
    """Normalize target string for grouping key."""
    if target is None:
        return ""
    return target.strip().lower()


def _normalize_constraint(constraint: str | None) -> str:
    """Normalize constraint string for canonical grouping."""
    if constraint is None:
        return ""
    return constraint.strip().lower()


def _sample_key(sample: ClauseSample) -> SemanticKey:
    """Semantic key for merging semantically equivalent clause samples."""
    return (
        sample.step_id,
        sample.capability,
        _normalize_target(sample.target),
        _normalize_constraint(sample.constraint),
    )


def _semantic_clause_key(clause: Clause) -> tuple[str, str, tuple[str, ...]]:
    """Semantic key for conflict retention, ignoring operator and step."""
    return (
        clause.capability,
        _normalize_target(clause.target),
        tuple(sorted(constraint.value.strip().lower() for constraint in clause.constraints)),
    )


def aggregate_samples(
    all_samples: list[list[ClauseSample]],
    n_samples: int,
    agreement_threshold: float = 0.6,
) -> list[Clause]:
    """Collapse repeated ICCM samples into canonical clauses."""
    groups: dict[SemanticKey, list[ClauseSample]] = defaultdict(list)
    operator_votes: dict[SemanticKey, dict[ClauseOperator, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    operator_groups: dict[tuple[SemanticKey, ClauseOperator], list[ClauseSample]] = (
        defaultdict(list)
    )

    for pass_index, sample_list in enumerate(all_samples):
        seen_in_sample: set[tuple[SemanticKey, ClauseOperator]] = set()
        for sample in sample_list:
            key = _sample_key(sample)
            vote_key = (key, sample.operator)
            groups[key].append(sample)
            operator_groups[vote_key].append(sample)
            if vote_key not in seen_in_sample:
                seen_in_sample.add(vote_key)
                operator_votes[key][sample.operator].add(pass_index)

    clauses: list[Clause] = []
    clause_idx = 0

    for key, samples in sorted(groups.items()):
        _step_id, capability, _norm_target, _norm_constraint = key
        allowed_votes = len(operator_votes[key].get(ClauseOperator.ALLOWED, set()))
        prohibited_votes = len(operator_votes[key].get(ClauseOperator.PROHIBITED, set()))
        unknown_votes = len(operator_votes[key].get(ClauseOperator.UNKNOWN, set()))
        allowed_agreement = allowed_votes / max(n_samples, 1)
        prohibited_agreement = prohibited_votes / max(n_samples, 1)
        explicit_operators: list[tuple[ClauseOperator, float]] = []
        if allowed_agreement >= agreement_threshold:
            explicit_operators.append((ClauseOperator.ALLOWED, allowed_agreement))
        if prohibited_agreement >= agreement_threshold:
            explicit_operators.append((ClauseOperator.PROHIBITED, prohibited_agreement))

        if explicit_operators:
            for operator, agreement in explicit_operators:
                clause = _build_clause(
                    clause_idx=clause_idx,
                    capability=capability,
                    operator=operator,
                    agreement=agreement,
                    samples=operator_groups[(key, operator)],
                )
                clauses.append(clause)
                clause_idx += 1
            continue

        agreement = max(
            allowed_agreement,
            prohibited_agreement,
            unknown_votes / max(n_samples, 1),
        )
        clause = _build_clause(
            clause_idx=clause_idx,
            capability=capability,
            operator=ClauseOperator.UNKNOWN,
            agreement=agreement,
            samples=samples,
        )
        clauses.append(clause)
        clause_idx += 1

    logger.info("Aggregated %d canonical clauses from %d samples", len(clauses), n_samples)
    return clauses


def _build_clause(
    clause_idx: int,
    capability: str,
    operator: ClauseOperator,
    agreement: float,
    samples: list[ClauseSample],
) -> Clause:
    """Materialize a canonical clause from a sample bucket."""
    evidence_texts: list[str] = []
    constraint_texts: set[str] = set()
    source_docs: set[str] = set()
    step_ids: set[str] = set()
    original_target: str | None = None

    for sample in samples:
        if sample.evidence_span and sample.evidence_span not in evidence_texts:
            evidence_texts.append(sample.evidence_span)
        if sample.target and original_target is None:
            original_target = sample.target
        if sample.constraint:
            constraint_texts.add(sample.constraint)
        step_ids.add(sample.step_id)
        source_docs.add(
            sample.step_id.split("_")[0] if "_" in sample.step_id else sample.step_id
        )

    evidence_spans = [
        EvidenceSpan(doc_id="unresolved", start_offset=0, end_offset=0, text=evidence_text)
        for evidence_text in evidence_texts[:5]
    ]
    normalized_target, constraints = normalize_clause_surface(
        capability,
        original_target,
        sorted(constraint_texts),
        clause_id=f"c{clause_idx}",
        evidence=evidence_spans[0] if evidence_spans else None,
    )
    return Clause(
        clause_id=f"c{clause_idx}",
        capability=capability,
        operator=operator,
        target=normalized_target,
        constraints=constraints,
        evidence_spans=evidence_spans,
        vote_agreement=agreement,
        step_ids=sorted(step_ids),
        source_doc_ids=sorted(source_docs),
    )


def apply_authorization_guard(
    clauses: list[Clause],
    a_req: list[str],
) -> list[Clause]:
    """Downgrade vague authorization claims for high-impact capabilities."""
    a_req_set = set(a_req)
    guarded: list[Clause] = []

    for clause in clauses:
        normalized_clause = _normalize_authorization_surface(clause)
        if normalized_clause.capability in a_req_set:
            is_authorization_specific = _is_authorization_specific(normalized_clause)
            if normalized_clause.operator == ClauseOperator.UNKNOWN and is_authorization_specific:
                logger.info(
                    "Authorization guard: promoting '%s' from unknown to allowed "
                    "(specific operational authorization recovered)",
                    normalized_clause.capability,
                )
                guarded.append(
                    normalized_clause.model_copy(
                        update={"operator": ClauseOperator.ALLOWED}
                    )
                )
                continue
            if (
                normalized_clause.operator == ClauseOperator.ALLOWED
                and not is_authorization_specific
            ):
                logger.warning(
                    "Authorization guard: downgrading '%s' from allowed to unknown "
                    "(insufficient authorization specificity)",
                    normalized_clause.capability,
                )
                guarded.append(
                    normalized_clause.model_copy(update={"operator": ClauseOperator.UNKNOWN})
                )
                continue

        guarded.append(normalized_clause)

    return guarded


def _normalize_authorization_surface(clause: Clause) -> Clause:
    if clause.target and not _is_specific_value(clause.target):
        return clause.model_copy(update={"target": None})
    return clause


def _is_authorization_specific(clause: Clause) -> bool:
    evidence_text = " ".join(e.text.lower() for e in clause.evidence_spans)
    combined_text = " ".join(
        part
        for part in [
            evidence_text,
            (clause.target or "").lower(),
            *[constraint.value.lower() for constraint in clause.constraints],
        ]
        if part
    )
    has_authorization_signal = any(sig in combined_text for sig in _AUTHORIZATION_SIGNALS)
    has_structured_authorization = any(
        sig in combined_text for sig in {"allowed-tools", "allowed_tools"}
    )
    has_capability_cue = any(
        cue in combined_text for cue in _CAPABILITY_CUES.get(clause.capability, ())
    )
    has_specific_target = _is_specific_value(clause.target)
    has_specific_constraint = any(
        _is_specific_value(constraint.value) or _looks_authorizing_constraint(constraint.value)
        for constraint in clause.constraints
    )
    capability_specific = _capability_specific_authorization(clause, combined_text)
    return has_structured_authorization or capability_specific or (
        has_authorization_signal
        and has_capability_cue
        and (has_specific_target or has_specific_constraint)
    )


def _capability_specific_authorization(clause: Clause, combined_text: str) -> bool:
    constraint_values = [constraint.value.lower() for constraint in clause.constraints]
    has_env_constraint = any("env" in constraint.constraint_type for constraint in clause.constraints)
    has_network_hint = any(
        token in combined_text
        for token in ("http://", "https://", "get ", "post ", "endpoint", "api", "request")
    )
    has_command_hint = any(
        token in combined_text
        for token in ("run", "execute", "python", "bash", ".py", ".sh", "脚本", "命令", "运行", "执行")
    )
    if clause.capability in {"env_var_read", "api_key_use"}:
        return has_env_constraint or any(_looks_authorizing_constraint(value) for value in constraint_values)
    if clause.capability == "http_request":
        return has_network_hint and (
            any(_is_specific_value(value) for value in constraint_values)
            or has_specific_http_target(clause.target)
            or any(_looks_authorizing_constraint(value) for value in constraint_values)
        )
    if clause.capability in {"shell_exec", "subprocess_spawn"}:
        return has_command_hint and any(
            _is_specific_value(constraint.value) or constraint.constraint_type.startswith("command")
            for constraint in clause.constraints
        )
    if clause.capability == "sql_exec":
        return any(token in combined_text for token in ("select ", "insert ", "update ", "delete ", "query", "sql"))
    return False


def has_specific_http_target(value: str | None) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return lowered.startswith(("http://", "https://", "get ", "post ", "put ", "delete /", "/")) or "." in lowered


def _looks_authorizing_constraint(value: str) -> bool:
    lowered = value.strip().lower()
    return any(
        token in lowered
        for token in (
            "required",
            "must",
            "need",
            "api key",
            "token",
            "header",
            "endpoint",
            "authentication",
            "environment variable",
            "env var",
            "必需",
            "必须",
            "需要",
            "环境变量",
        )
    )


def _is_specific_value(value: str | None) -> bool:
    """Check whether a target/constraint is specific enough to authorize A_req."""
    if value is None:
        return False
    normalized = value.strip().lower()
    if not normalized or normalized in _GENERIC_SCOPE_TERMS:
        return False
    return any(token in normalized for token in ("/", ".", ":", "*", "_", "-", "http"))


def detect_clause_conflicts(clauses: list[Clause]) -> list[str]:
    """Retain contradictory allow/prohibit clauses with the same semantic key."""
    grouped: dict[tuple[str, str, tuple[str, ...]], list[Clause]] = defaultdict(list)
    for clause in clauses:
        if clause.operator not in {ClauseOperator.ALLOWED, ClauseOperator.PROHIBITED}:
            continue
        grouped[_semantic_clause_key(clause)].append(clause)

    conflicts: list[str] = []
    for (capability, target, constraints), group in sorted(grouped.items()):
        operators = {clause.operator.value for clause in group}
        if operators == {"allowed", "prohibited"}:
            docs = sorted({doc_id for clause in group for doc_id in clause.source_doc_ids})
            steps = sorted({step_id for clause in group for step_id in clause.step_ids})
            constraint_repr = ",".join(constraints) if constraints else "-"
            target_repr = target or "-"
            conflicts.append(
                "conflict:"
                f" capability={capability}"
                f" target={target_repr}"
                f" constraints={constraint_repr}"
                f" operators={','.join(sorted(operators))}"
                f" docs={','.join(docs)}"
                f" steps={','.join(steps)}"
            )
    return conflicts


def locate_evidence_span(
    evidence_text: str,
    doc_id: str,
    doc_content: str,
) -> EvidenceSpan:
    """Locate one evidence snippet inside a document."""
    idx = doc_content.find(evidence_text)
    if idx >= 0:
        return EvidenceSpan(
            doc_id=doc_id,
            start_offset=idx,
            end_offset=idx + len(evidence_text),
            text=evidence_text,
        )

    idx = doc_content.lower().find(evidence_text.lower())
    if idx >= 0:
        return EvidenceSpan(
            doc_id=doc_id,
            start_offset=idx,
            end_offset=idx + len(evidence_text),
            text=doc_content[idx : idx + len(evidence_text)],
        )

    logger.debug("Evidence not found in doc %s: %s", doc_id, evidence_text[:50])
    return EvidenceSpan(doc_id=doc_id, start_offset=0, end_offset=0, text=evidence_text)


def build_contract_table(
    skill_id: str,
    clauses: list[Clause],
    *,
    steps: list[Step] | None = None,
    step_order_edges: list[StepOrderEdge] | None = None,
    unresolved_refs: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> ContractTable:
    """Build the final contract table for one skill package."""
    return ContractTable(
        skill_id=skill_id,
        clauses=clauses,
        steps=steps or [],
        step_order_edges=step_order_edges or [],
        unresolved_references=unresolved_refs or [],
        cross_doc_conflicts=conflicts or [],
    )
