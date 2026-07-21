"""Core reconciliation predicates (deterministic, three-valued).

Each predicate returns PredicateResult (true / false / abstain).
No LLM or embedding scores are used in any predicate.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from skillrecon.contract.normalize import is_resource_constraint, is_scope_constraint
from skillrecon.core.enums import HypothesisStatus, PredicateResult, RouteSupport
from skillrecon.core.types import (
    Bridge,
    CandidatePair,
    CapabilityEvent,
    Clause,
    Constraint,
    OrchestrationHypothesis,
    ResourceUse,
    RiskPath,
)

logger = logging.getLogger(__name__)

_SCOPE_ONLY_CONSTRAINT_TYPES = {
    "user_input_only",
    "config_bound",
    "side_effect_bound",
}
_NETWORK_CAPABILITIES = {
    "http_request",
    "websocket",
    "smtp_send",
    "ssh_connect",
    "data_upload",
    "data_encode_send",
    "dns_lookup",
    "socket_connect",
    "ftp_transfer",
}
_READ_LIKE_CAPABILITIES = {
    "file_read",
    "token_file_read",
    "env_var_read",
    "api_key_use",
    "credential_store_access",
    "directory_list",
}
_WRITE_LIKE_CAPABILITIES = {
    "file_write",
    "temp_file_create",
    "file_delete",
    "file_permission_change",
}
_EFFECTFUL_CAPABILITIES = _NETWORK_CAPABILITIES | _WRITE_LIKE_CAPABILITIES | {
    "shell_exec",
    "subprocess_spawn",
    "dynamic_import",
    "process_kill",
    "cron_schedule",
}


# ---------------------------------------------------------------------------
# Overlap policy loader
# ---------------------------------------------------------------------------

_OverlapMap = dict[str, set[str]]


def load_overlap_policy(policy_path: Path) -> _OverlapMap:
    """Load capability overlap policy and build bidirectional lookup.

    Returns a dict mapping each atom to the set of atoms it is comparable with
    (including itself).
    """
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    lookup: _OverlapMap = {}
    for family in data.get("families", []):
        members = set(family["members"])
        for m in members:
            lookup.setdefault(m, set()).update(members)
    return lookup


# ---------------------------------------------------------------------------
# 1. capability_overlaps
# ---------------------------------------------------------------------------


def capability_overlaps(
    clause: Clause,
    behavior_cap: str,
    overlap_map: _OverlapMap,
) -> PredicateResult:
    """Check whether clause capability and behavior capability are comparable.

    Rules:
    - exact same atom -> TRUE
    - configured overlap family hit -> TRUE
    - atom family explicitly incompatible -> FALSE
    - policy missing or family undecidable -> ABSTAIN
    """
    clause_cap = clause.capability

    if clause_cap == behavior_cap:
        return PredicateResult.TRUE

    family = overlap_map.get(clause_cap)
    if family is not None:
        if behavior_cap in family:
            return PredicateResult.TRUE
        return PredicateResult.FALSE

    if not overlap_map:
        return PredicateResult.ABSTAIN

    return PredicateResult.FALSE


# ---------------------------------------------------------------------------
# 2. resource_compatible
# ---------------------------------------------------------------------------


def resource_compatible(
    clause: Clause,
    resources: list[ResourceUse],
) -> PredicateResult:
    """Check whether clause target/constraints are type-compatible with resources.

    Rules:
    - at least one typed resource clearly compatible -> TRUE
    - all resources present and clearly incompatible -> FALSE
    - only unresolved/abstract resources -> ABSTAIN
    """
    resource_constraints = [
        constraint for constraint in clause.constraints if is_resource_constraint(constraint)
    ]

    if not clause.target and not resource_constraints:
        return PredicateResult.ABSTAIN

    if not resources:
        return PredicateResult.ABSTAIN

    all_unresolved = all(not r.resolved for r in resources)
    if all_unresolved:
        return PredicateResult.ABSTAIN

    resolved = [r for r in resources if r.resolved]
    if not resolved:
        return PredicateResult.ABSTAIN

    if not clause.target:
        if not resource_constraints:
            return PredicateResult.TRUE
        for res in resolved:
            if _any_constraint_matches(resource_constraints, res):
                return PredicateResult.TRUE
        return PredicateResult.FALSE

    for res in resolved:
        if _resource_matches_clause(clause, res):
            return PredicateResult.TRUE

    return PredicateResult.FALSE


def _resource_matches_clause(clause: Clause, resource: ResourceUse) -> bool:
    """Check a single resolved resource against clause target and constraints."""
    target = clause.target
    if not target:
        return _any_constraint_matches(clause.constraints, resource)

    rtype = resource.resource_type
    val = resource.value

    if rtype == "path":
        return _path_compatible(target, val)
    if rtype in ("url", "domain"):
        return _domain_compatible(target, val)
    if rtype == "env_var":
        return _env_compatible(target, val)
    if rtype == "command":
        return _command_compatible(target, val)

    return target.lower() == val.lower()


def _path_compatible(clause_target: str, resource_path: str) -> bool:
    target = _normalize_path_pattern(clause_target)
    value = _normalize_path_pattern(resource_path)
    target_path = PurePosixPath(target)
    value_path = PurePosixPath(value)

    if target == value:
        return True

    if _has_glob_syntax(target):
        candidates = {
            value,
            value.lstrip("./"),
            str(value_path),
            str(PurePosixPath(*value_path.parts[-2:])) if len(value_path.parts) >= 2 else value,
            value_path.name,
        }
        normalized_target = target.lstrip("./")
        return any(
            fnmatch.fnmatch(candidate, target)
            or fnmatch.fnmatch(candidate, normalized_target)
            for candidate in candidates
        )

    if "/" not in target and target_path.name and target_path.name == value_path.name:
        return True

    # Treat slash-terminated or suffix-less targets as directory scopes.
    if target.endswith("/") or ("/" in target and not target_path.suffix):
        directory = target.rstrip("/")
        return value == directory or value.startswith(directory + "/")

    if target_path == value_path:
        return True
    return False


def _domain_compatible(clause_target: str, resource_value: str) -> bool:
    target = _normalize_domain_pattern(clause_target)
    host = _resource_host(resource_value)
    if not host:
        return False

    if target.startswith("*."):
        suffix = target[2:]
        return host != suffix and host.endswith("." + suffix)

    if _has_glob_syntax(target):
        return fnmatch.fnmatch(host, target)

    if target == host:
        return True
    return False


def _env_compatible(clause_target: str, resource_value: str) -> bool:
    ct = clause_target.upper()
    rv = resource_value.upper()
    if ct == rv:
        return True
    if _has_glob_syntax(ct):
        return fnmatch.fnmatch(rv, ct)
    return False


def _command_compatible(clause_target: str, resource_value: str) -> bool:
    ct = PurePosixPath(clause_target.lower()).name
    rv = PurePosixPath(resource_value.lower()).name
    if _has_glob_syntax(ct):
        return fnmatch.fnmatch(rv, ct)
    return ct == rv


def _any_constraint_matches(
    constraints: list[Constraint],
    resource: ResourceUse,
) -> bool:
    for cst in constraints:
        if _constraint_matches_resource(cst, resource):
            return True
    return False


# ---------------------------------------------------------------------------
# 3. scope_satisfied
# ---------------------------------------------------------------------------


def scope_satisfied(
    clause: Clause,
    resources: list[ResourceUse],
    *,
    behavior_capability: str | None = None,
) -> PredicateResult:
    """Check whether clause scope/constraints are satisfied by behavior.

    Rules:
    - no target and no constraints -> ABSTAIN (capability-only)
    - at least one constraint clearly violated -> FALSE
    - all relevant constraints satisfied -> TRUE
    - unresolved resources, cannot decide -> ABSTAIN
    """
    scope_constraints = [
        constraint for constraint in clause.constraints if is_scope_constraint(constraint)
    ]

    if not clause.target and not scope_constraints:
        return PredicateResult.ABSTAIN

    if not resources:
        return PredicateResult.ABSTAIN

    all_unresolved = all(not r.resolved for r in resources)
    if all_unresolved:
        return PredicateResult.ABSTAIN

    resolved = [r for r in resources if r.resolved]

    if scope_constraints:
        violated = False
        all_satisfied = True
        for cst in scope_constraints:
            status = _check_constraint(
                cst,
                resolved,
                behavior_capability=behavior_capability or clause.capability,
            )
            if status == PredicateResult.FALSE:
                violated = True
            elif status != PredicateResult.TRUE:
                all_satisfied = False

        if violated:
            return PredicateResult.FALSE
        if all_satisfied:
            return PredicateResult.TRUE
        return PredicateResult.ABSTAIN

    if clause.target:
        for res in resolved:
            if _resource_matches_clause(clause, res):
                return PredicateResult.TRUE
        return PredicateResult.FALSE

    return PredicateResult.ABSTAIN


def _check_constraint(
    constraint: Constraint,
    resources: list[ResourceUse],
    *,
    behavior_capability: str | None = None,
) -> PredicateResult:
    """Check a single constraint against resolved resources."""
    semantic_status = _semantic_constraint_status(
        constraint,
        resources,
        behavior_capability=behavior_capability,
    )
    if semantic_status is not None:
        return semantic_status

    relevant = [r for r in resources if _constraint_applies_to_resource(constraint, r)]
    if not relevant:
        return PredicateResult.ABSTAIN

    for res in relevant:
        if _constraint_matches_resource(constraint, res):
            return PredicateResult.TRUE

    return PredicateResult.FALSE


def _constraint_applies_to_resource(
    constraint: Constraint,
    resource: ResourceUse,
) -> bool:
    ctype = constraint.constraint_type.lower()
    compatible_types = {
        "path": {"path"},
        "path_glob": {"path"},
        "file_glob": {"path"},
        "domain": {"domain", "url"},
        "domain_glob": {"domain", "url"},
        "url": {"url"},
        "url_glob": {"url"},
        "env_var": {"env_var"},
        "env_glob": {"env_var"},
        "command": {"command"},
        "command_glob": {"command"},
        "query": {"query"},
        "user_input_only": {"path", "url", "domain", "command", "env_var", "secret"},
        "config_bound": {"path", "url", "domain"},
        "side_effect_bound": {"path", "url", "domain", "command", "env_var", "query", "secret"},
    }
    return resource.resource_type in compatible_types.get(ctype, {constraint.constraint_type})


def _constraint_matches_resource(
    constraint: Constraint,
    resource: ResourceUse,
) -> bool:
    ctype = constraint.constraint_type.lower()
    value = constraint.value

    if ctype in {"path", "path_glob", "file_glob"}:
        return resource.resource_type == "path" and _path_compatible(value, resource.value)
    if ctype in {"domain", "domain_glob"}:
        return resource.resource_type in {"domain", "url"} and _domain_compatible(value, resource.value)
    if ctype in {"url", "url_glob"}:
        return resource.resource_type == "url" and _url_compatible(value, resource.value)
    if ctype in {"env_var", "env_glob"}:
        return resource.resource_type == "env_var" and _env_compatible(value, resource.value)
    if ctype in {"command", "command_glob"}:
        return resource.resource_type == "command" and _command_compatible(value, resource.value)
    if ctype == "query":
        return resource.resource_type == "query" and value.lower() == resource.value.lower()
    if ctype in _SCOPE_ONLY_CONSTRAINT_TYPES:
        return False

    return (
        constraint.constraint_type == resource.resource_type
        and value.lower() == resource.value.lower()
    )


def _semantic_constraint_status(
    constraint: Constraint,
    resources: list[ResourceUse],
    *,
    behavior_capability: str | None,
) -> PredicateResult | None:
    ctype = constraint.constraint_type.lower()
    if ctype == "user_input_only":
        return _user_input_only_status(resources)
    if ctype == "config_bound":
        return _config_bound_status(resources)
    if ctype == "side_effect_bound":
        return _side_effect_bound_status(constraint.value, behavior_capability, resources)
    return None


def _user_input_only_status(resources: list[ResourceUse]) -> PredicateResult:
    relevant = [
        resource
        for resource in resources
        if resource.resource_type in {"path", "url", "domain", "command", "env_var", "secret"}
    ]
    if not relevant:
        return PredicateResult.ABSTAIN
    if any(resource.origin_kind in {"env", "config_file", "artifact"} for resource in relevant):
        return PredicateResult.FALSE
    if relevant and all(resource.origin_kind == "user_input" for resource in relevant):
        return PredicateResult.TRUE
    return PredicateResult.ABSTAIN


def _config_bound_status(resources: list[ResourceUse]) -> PredicateResult:
    relevant = [
        resource
        for resource in resources
        if resource.resource_type in {"path", "url", "domain"}
    ]
    if not relevant:
        return PredicateResult.ABSTAIN
    if any(resource.origin_kind in {"env", "user_input", "artifact"} for resource in relevant):
        return PredicateResult.FALSE
    if any(resource.origin_kind == "config_file" for resource in relevant):
        return PredicateResult.TRUE
    return PredicateResult.ABSTAIN


def _side_effect_bound_status(
    bound_value: str,
    behavior_capability: str | None,
    resources: list[ResourceUse],
) -> PredicateResult:
    if behavior_capability is None:
        return PredicateResult.ABSTAIN
    normalized = bound_value.strip().lower()
    if not resources:
        return PredicateResult.ABSTAIN
    if normalized == "no_network":
        if behavior_capability in _NETWORK_CAPABILITIES:
            return PredicateResult.FALSE
        return PredicateResult.TRUE
    if normalized == "read_only":
        if behavior_capability in _EFFECTFUL_CAPABILITIES:
            return PredicateResult.FALSE
        if behavior_capability in _READ_LIKE_CAPABILITIES:
            return PredicateResult.TRUE
        return PredicateResult.ABSTAIN
    if normalized == "write_only":
        if behavior_capability in _READ_LIKE_CAPABILITIES:
            return PredicateResult.FALSE
        if behavior_capability in _WRITE_LIKE_CAPABILITIES:
            return PredicateResult.TRUE
        return PredicateResult.ABSTAIN
    return PredicateResult.ABSTAIN


def _url_compatible(clause_target: str, resource_value: str) -> bool:
    target = clause_target.lower()
    value = resource_value.lower()
    if _has_glob_syntax(target):
        return fnmatch.fnmatch(value, target)
    return value == target


def _resource_host(resource_value: str) -> str:
    try:
        parsed = urlparse(resource_value if "://" in resource_value else f"h://{resource_value}")
        return (parsed.hostname or "").lower()
    except ValueError:
        return resource_value.lower()


def _normalize_domain_pattern(value: str) -> str:
    host = _resource_host(value)
    return host.rstrip(".")


def _normalize_path_pattern(value: str) -> str:
    return PurePosixPath(value.replace("\\", "/")).as_posix().lower()


def _has_glob_syntax(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


# ---------------------------------------------------------------------------
# 4. prohibition_conflict
# ---------------------------------------------------------------------------


def prohibition_conflict(
    clause: Clause,
    behavior_cap: str,
    resources: list[ResourceUse],
    overlap_map: _OverlapMap,
) -> PredicateResult:
    """Check whether a prohibited clause conflicts with a behavior object.

    Rules:
    - capability_overlaps = false -> FALSE
    - broad prohibition (no target/constraint) + capability overlap -> TRUE
    - targeted prohibition needs resource_compatible or scope_satisfied
    - only unresolved resources / unresolved route -> ABSTAIN
    """
    cap_result = capability_overlaps(clause, behavior_cap, overlap_map)
    if cap_result == PredicateResult.FALSE:
        return PredicateResult.FALSE
    if cap_result == PredicateResult.ABSTAIN:
        return PredicateResult.ABSTAIN

    if not clause.target and not clause.constraints:
        return PredicateResult.TRUE

    res_result = resource_compatible(clause, resources)
    if res_result == PredicateResult.TRUE:
        return PredicateResult.TRUE

    scope_result = scope_satisfied(
        clause,
        resources,
        behavior_capability=behavior_cap,
    )
    if scope_result == PredicateResult.TRUE:
        return PredicateResult.TRUE

    if res_result == PredicateResult.ABSTAIN or scope_result == PredicateResult.ABSTAIN:
        return PredicateResult.ABSTAIN

    return PredicateResult.FALSE


# ---------------------------------------------------------------------------
# 5. execution_route_justified
# ---------------------------------------------------------------------------


def execution_route_justified(
    candidate: CandidatePair,
    events: dict[str, CapabilityEvent],
    paths: dict[str, RiskPath],
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
) -> tuple[PredicateResult, RouteSupport | None]:
    """Check whether a behavior conclusion has sufficient route evidence.

    Returns (result, route_support_type).

    Rules:
    - local event -> TRUE (local)
    - codeql path -> TRUE (codeql)
    - bridge path with all bridges resolved -> TRUE (bridge)
    - orchestration confirmed -> TRUE (orchestration_confirmed)
    - orchestration unresolved/competing -> ABSTAIN (orchestration_unresolved)
    - weak path only -> ABSTAIN (weak)
    - evidence contradictory or missing -> FALSE (None)
    """
    from skillrecon.core.enums import BehaviorKind

    if candidate.behavior_kind == BehaviorKind.EVENT:
        event = events.get(candidate.event_id or "")
        if event is None:
            return PredicateResult.FALSE, None
        if event.tier == "instruction":
            return PredicateResult.ABSTAIN, RouteSupport.WEAK
        return PredicateResult.TRUE, RouteSupport.LOCAL

    if candidate.behavior_kind == BehaviorKind.RESOURCE:
        return PredicateResult.TRUE, RouteSupport.LOCAL

    if candidate.behavior_kind == BehaviorKind.PATH:
        path = paths.get(candidate.path_id or "")
        if not path:
            return PredicateResult.FALSE, None
        return _evaluate_path_route(path, bridges, orchestrations)

    return PredicateResult.FALSE, None


def _evaluate_path_route(
    path: RiskPath,
    bridges: dict[str, Bridge],
    orchestrations: dict[str, OrchestrationHypothesis],
) -> tuple[PredicateResult, RouteSupport | None]:
    """Evaluate route support for a specific path."""
    ev_level = path.evidence_level

    if ev_level == "codeql":
        return PredicateResult.TRUE, RouteSupport.CODEQL

    if ev_level == "bridge":
        all_resolved = all(
            bid in bridges for bid in path.bridges_used
        )
        if all_resolved and path.bridges_used:
            return PredicateResult.TRUE, RouteSupport.BRIDGE
        return PredicateResult.ABSTAIN, RouteSupport.WEAK

    if path.orchestration_hypotheses:
        statuses = []
        for hid in path.orchestration_hypotheses:
            oh = orchestrations.get(hid)
            if oh:
                statuses.append(oh.status)
        if any(s == HypothesisStatus.CONFIRMED for s in statuses):
            return PredicateResult.TRUE, RouteSupport.ORCHESTRATION_CONFIRMED
        if statuses:
            return PredicateResult.ABSTAIN, RouteSupport.ORCHESTRATION_UNRESOLVED

    if ev_level == "weak":
        return PredicateResult.ABSTAIN, RouteSupport.WEAK

    if not ev_level:
        return PredicateResult.TRUE, RouteSupport.LOCAL

    return PredicateResult.FALSE, None
