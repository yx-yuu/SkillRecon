"""Core reconciliation predicates (deterministic, three-valued).

Each predicate returns PredicateResult (true / false / abstain).
No LLM or embedding scores are used in any predicate.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import posixpath
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from skillrecon.core.config import PROJECT_ROOT
from skillrecon.contract.steps import extract_step_order
from skillrecon.contract.controls import CONTROL_TYPES
from skillrecon.contract.normalize import (
    is_resource_constraint,
    looks_typed_constraint_applicable,
)
from skillrecon.core.enums import BridgeKind, HypothesisStatus, PredicateResult, RouteSupport
from skillrecon.core.types import (
    Bridge,
    CandidatePair,
    CapabilityEvent,
    Clause,
    Constraint,
    OrchestrationHypothesis,
    ResourceUse,
    RiskPath,
    Step,
    StepOrderEdge,
    ReconciliationJudgment,
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
    "upload_exfil",
    "data_encode",
    "dns_query",
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
    "subprocess",
    "dynamic_import",
    "process_kill",
    "cron_write",
}


# ---------------------------------------------------------------------------
# Overlap policy loader
# ---------------------------------------------------------------------------

_OverlapMap = dict[str, set[str]]


def load_overlap_policy(policy_path: Path) -> _OverlapMap:
    """Load the versioned capability relation.

    V1 records legacy comparison families. V2 records directed coverage:
    a declared capability maps to the event capabilities it authorizes.
    """
    data = json.loads((PROJECT_ROOT / policy_path).read_text(encoding="utf-8"))
    if data["version"] == "v2":
        result = {cap: {cap, *covered} for cap, covered in data["coverage"].items()}
        for cap in result:
            pending = list(result[cap])
            while pending:
                for covered in result.get(pending.pop(), set()) - result[cap]:
                    result[cap].add(covered)
                    pending.append(covered)
        return result
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
    """Check declared-capability coverage of the behavior capability.

    Rules:
    - exact same atom -> TRUE
    - configured directed coverage hit -> TRUE
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
    resources = [r for r in resources if r.resource_type not in {
        "control_condition", "registered_operation", "registered_route", "required_environment", "request_credential"
    }]
    # tau(c) and Sigma(c) are different premises. A failing prefix/domain
    # constraint belongs to ScopeViol, not to target incompatibility.
    if not clause.target:
        return PredicateResult.TRUE

    if not resources:
        return PredicateResult.ABSTAIN

    all_unresolved = all(not r.resolved for r in resources)
    if all_unresolved:
        return PredicateResult.ABSTAIN

    resolved = [r for r in resources if r.resolved]
    if not resolved:
        return PredicateResult.ABSTAIN

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
        }
        normalized_target = target.lstrip("./")
        return any(
            fnmatch.fnmatch(candidate, target)
            or fnmatch.fnmatch(candidate, normalized_target)
            for candidate in candidates
        )

    # Treat slash-terminated or suffix-less targets as directory scopes.
    if target.endswith("/") or ("/" in target and not target_path.suffix):
        directory = target.rstrip("/")
        return value == directory or value.startswith(directory + "/")

    if target_path == value_path:
        return True
    return False


def _domain_compatible(clause_target: str, resource_value: str) -> bool:
    if "://" in clause_target:
        return _url_compatible(clause_target, resource_value)
    if clause_target.rstrip(":") in {"http", "https", "ws", "wss", "ssh", "ftp"}:
        return urlparse(resource_value).scheme == clause_target.rstrip(":")
    if re.search(r":\d+$", clause_target):
        observed = urlparse(resource_value if "://" in resource_value else "//" + resource_value)
        host, port = clause_target.rsplit(":", 1)
        return observed.hostname == host.lower() and (observed.port or {"https":443,"http":80}.get(observed.scheme)) == int(port)
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
    ct = clause_target
    rv = resource_value
    if ct == rv:
        return True
    if _has_glob_syntax(ct):
        return fnmatch.fnmatch(rv, ct)
    return False


def _command_compatible(clause_target: str, resource_value: str) -> bool:
    selector = clause_target.replace(":*", " *")
    target = shlex.split(selector)
    observed = shlex.split(resource_value)
    if not target or not observed:
        return False
    if len(target) == 1:
        executable = observed[0] if "/" in target[0] else PurePosixPath(observed[0]).name
        return fnmatch.fnmatchcase(executable, target[0])
    if any(_has_glob_syntax(part) for part in target):
        return fnmatch.fnmatchcase(" ".join(observed), " ".join(target))
    return observed[:len(target)] == target


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
    resources = [r for r in resources if r.resource_type not in {
        "registered_operation", "registered_route", "required_environment", "request_credential"
    }]
    scope_constraints = clause.constraints
    if not any(c.constraint_type in CONTROL_TYPES | {"opt_out_env", "account_active", "no_auto_retry"} for c in scope_constraints):
        resources = [r for r in resources if r.resource_type != "control_condition"]

    if not clause.target and not scope_constraints:
        return PredicateResult.TRUE

    if not resources:
        return PredicateResult.ABSTAIN

    all_unresolved = all(not r.resolved for r in resources)
    if all_unresolved:
        return PredicateResult.ABSTAIN

    resolved = resources

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
    if not looks_typed_constraint_applicable(constraint):
        return PredicateResult.ABSTAIN
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

    results = [PredicateResult.ABSTAIN if not res.resolved else
               PredicateResult.TRUE if _constraint_matches_resource(constraint,res) else PredicateResult.FALSE
               for res in relevant]
    if PredicateResult.FALSE in results:
        return PredicateResult.FALSE
    return PredicateResult.ABSTAIN if PredicateResult.ABSTAIN in results else PredicateResult.TRUE


def _constraint_applies_to_resource(
    constraint: Constraint,
    resource: ResourceUse,
) -> bool:
    ctype = constraint.constraint_type.lower()
    if ctype in {"var", "cmd"}:
        return resource.resource_type == {"var":"env_var", "cmd":"command"}[ctype]
    if ctype in {"opt_out_env", "account_active"}:
        legacy = {"opt_out_env":"empty_disables", "account_active":"config_enabled"}[ctype]
        return resource.resource_type == "control_condition" and resource.value in {
            f"{ctype}:{constraint.value}", f"{legacy}:{constraint.value}"}
    if ctype == "no_auto_retry":
        return resource.resource_type == "control_condition" and resource.value == "retry_on_error:automatic_retry"
    if ctype == "standalone_call":
        return resource.resource_type == "control_condition" and resource.value in {
            f"standalone_call:{name}" for name in constraint.value.split('|')}
    if ctype in CONTROL_TYPES:
        return (resource.resource_type == "control_condition"
                and resource.value == f"{ctype}:{constraint.value}")
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
        "prefix": {"path"}, "ext": {"path"}, "host": {"url", "domain", "host"},
        "scheme": {"url"}, "port": {"url", "domain", "host"},
        "interpreter": {"command"}, "arg_pattern": {"command"},
        "provenance": {"path", "env_var", "credential", "secret"},
        "location": {"path", "location"}, "trigger": {"trigger"}, "lifetime": {"lifetime"},
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
    if ctype == "var":
        return _env_compatible(value,resource.value)
    if ctype == "cmd":
        return _command_compatible(value,resource.value)

    if ctype == "prefix":
        prefix = _normalize_path_pattern(value).rstrip("/")
        observed = _normalize_path_pattern(resource.value)
        return observed == prefix or observed.startswith(prefix + "/")
    if ctype == "ext":
        return PurePosixPath(resource.value).suffix.lstrip(".").lower() == value.lstrip("*.").lower()
    if ctype == "host":
        return _domain_compatible(value, resource.value)
    if ctype == "scheme":
        return urlparse(resource.value).scheme == value.rstrip(":/")
    if ctype == "port":
        parsed = urlparse(resource.value if "://" in resource.value else "//" + resource.value)
        return (parsed.port or {"http":80,"https":443,"ssh":22,"ftp":21}.get(parsed.scheme)) == int(value)
    if ctype == "interpreter":
        command = shlex.split(resource.value)
        return bool(command) and PurePosixPath(command[0]).name == value
    if ctype == "arg_pattern":
        command = shlex.split(resource.value)
        return fnmatch.fnmatchcase(" ".join(command[1:]), value)
    if ctype == "provenance":
        return resource.origin_kind == value
    if ctype == "location":
        return _path_compatible(value,resource.value)

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
    if ctype == "read_only":
        return _side_effect_bound_status("read_only",behavior_capability,resources)
    if ctype in {"opt_out_env", "account_active"}:
        relevant = [r for r in resources if _constraint_applies_to_resource(constraint,r)]
        if not relevant:
            return PredicateResult.ABSTAIN
        values = [PredicateResult(json.loads(r.origin_hint)["result"]) if r.resolved
                  else PredicateResult.ABSTAIN for r in relevant]
        if PredicateResult.FALSE in values:
            return PredicateResult.FALSE
        return PredicateResult.TRUE if all(value==PredicateResult.TRUE for value in values) else PredicateResult.ABSTAIN
    if ctype == "no_auto_retry":
        retry = constraint.model_copy(update={"constraint_type": "retry_on_error", "value": "automatic_retry"})
        result = _semantic_constraint_status(retry,resources,behavior_capability=behavior_capability)
        return PredicateResult.FALSE if result == PredicateResult.TRUE else PredicateResult.TRUE if result == PredicateResult.FALSE else PredicateResult.ABSTAIN
    if ctype == "provenance":
        relevant = [resource for resource in resources if _constraint_applies_to_resource(constraint,resource)]
        if not relevant or any(resource.origin_kind == "unknown" for resource in relevant):
            return PredicateResult.ABSTAIN
    if ctype in CONTROL_TYPES:
        relevant = [r for r in resources if _constraint_applies_to_resource(constraint, r)]
        if not relevant:
            return PredicateResult.ABSTAIN
        results = [PredicateResult(json.loads(r.origin_hint)["result"]) for r in relevant]
        if ctype == "standalone_call":
            if PredicateResult.TRUE in results:return PredicateResult.TRUE
            return PredicateResult.FALSE if all(result==PredicateResult.FALSE for result in results) else PredicateResult.ABSTAIN
        if PredicateResult.FALSE in results:
            return PredicateResult.FALSE
        return PredicateResult.TRUE if all(r == PredicateResult.TRUE for r in results) else PredicateResult.ABSTAIN
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
        if behavior_capability in {'shell_exec','subprocess','dynamic_import'}:
            return PredicateResult.ABSTAIN
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
    normalized = posixpath.normpath(value.replace("\\", "/"))
    return normalized + "/" if value.endswith("/") and normalized != "/" else normalized


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
    - targeted prohibition needs its target and every constraint satisfied
    - only unresolved resources / unresolved route -> ABSTAIN
    """
    cap_result = capability_overlaps(clause, behavior_cap, overlap_map)
    if cap_result == PredicateResult.FALSE:
        return PredicateResult.FALSE
    if cap_result == PredicateResult.ABSTAIN:
        return PredicateResult.ABSTAIN

    if not clause.target and not clause.constraints:
        return PredicateResult.TRUE

    trigger_clause = clause.model_copy(update={"constraints": [
        constraint.model_copy(update={"constraint_type": "retry_on_error", "value": "automatic_retry"})
        if constraint.constraint_type == "no_auto_retry" else constraint for constraint in clause.constraints]})
    scope_result = scope_satisfied(
        trigger_clause,
        resources,
        behavior_capability=behavior_cap,
    )
    if scope_result != PredicateResult.TRUE:
        return scope_result
    if clause.target or any(is_resource_constraint(c) for c in clause.constraints):
        return resource_compatible(clause, resources)
    return PredicateResult.TRUE


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
    - registered source-slot path -> TRUE (local)
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
        if candidate.shared_signals.get("route_disjoint") == "true":
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

    if ev_level == "registered_source":
        return PredicateResult.TRUE, RouteSupport.LOCAL

    if ev_level == "bridge":
        if not path.bridges_used or any(bid not in bridges for bid in path.bridges_used):
            return PredicateResult.ABSTAIN, RouteSupport.WEAK
        sequence=[path.source,*path.segments,path.sink]
        units=[item.unit_id for index,item in enumerate(sequence)
               if index==0 or item.unit_id!=sequence[index-1].unit_id]
        transitions=list(zip(units,units[1:]))
        if len(transitions)!=len(path.bridges_used):
            return PredicateResult.FALSE, None
        event_ids={item.event_id for item in sequence if item.event_id}
        for (source,target),identifier in zip(transitions,path.bridges_used):
            bridge=bridges[identifier]
            direction=(bridge.source_unit_id,bridge.target_unit_id)
            if direction!=(source,target) and not (
                    bridge.kind==BridgeKind.STATIC and direction==(target,source)):
                return PredicateResult.FALSE, None
            # An invocation/artifact path carries the event that transports the
            # value. An unrelated bridge ID cannot confirm endpoint co-occurrence.
            endpoints=[bridge.source_event_id,bridge.target_event_id]
            if bridge.kind in {BridgeKind.COMMAND,BridgeKind.ARTIFACT} and not bridge.source_event_id:
                return PredicateResult.ABSTAIN, RouteSupport.WEAK
            if any(event_id not in event_ids for event_id in endpoints if event_id):
                return PredicateResult.ABSTAIN, RouteSupport.WEAK
        return PredicateResult.TRUE, RouteSupport.BRIDGE

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
        return PredicateResult.ABSTAIN, RouteSupport.WEAK

    return PredicateResult.FALSE, None


@dataclass(frozen=True)
class PathJustification:
    result: PredicateResult
    supporting_judgment_ids: list[str] = field(default_factory=list)
    abstain_reason: str | None = None
    notes: str = ""
    evidence_refs: list[str] = field(default_factory=list)


def path_justified(
    *,
    required_event_ids: list[str],
    auth_results_by_event: dict[str, list[ReconciliationJudgment]],
    clauses: dict[str, Clause],
    steps_by_id: dict[str, Step],
    path: RiskPath | None = None,
    events: dict[str, CapabilityEvent] | None = None,
    resources_by_event: dict[str, list[ResourceUse]] | None = None,
    step_order_edges: list[StepOrderEdge] | None = None,
) -> PathJustification:
    if step_order_edges is None:
        semantic_edges, unresolved_steps = extract_step_order(list(steps_by_id.values()))
    else:
        # G_D is authoritative: reading order is not execution order. Portable
        # witnesses must supply the actual normative edges they rely on.
        semantic_edges = [edge for edge in step_order_edges if edge.relation == "precedes"]
        unresolved_steps = {step.step_id for step in steps_by_id.values() if step.unresolved_order}
    successors: dict[str, set[str]] = {}
    for edge in semantic_edges:
        successors.setdefault(edge.source_step_id, set()).add(edge.target_step_id)
    reachable: dict[str, set[str]] = {}
    for source in successors:
        frontier = list(successors[source])
        visited: set[str] = set()
        while frontier:
            target = frontier.pop()
            if target in visited:
                continue
            visited.add(target)
            frontier.extend(successors.get(target, set()))
        reachable[source] = visited

    option_sets: list[list[tuple[str, str | None]]] = []
    for event_id in required_event_ids:
        options: list[tuple[str, str | None]] = []
        for judgment in auth_results_by_event.get(event_id, []):
            if judgment.result != PredicateResult.TRUE:
                continue
            clause_id = next((ref for ref in judgment.subject_refs if ref in clauses), None)
            if clause_id is None:
                continue
            options.extend(
                (judgment.judgment_id, step_id)
                for step_id in (clauses[clause_id].step_ids or [None])
            )
        if not options:
            return PathJustification(
                result=PredicateResult.FALSE,
                notes=f"missing_support:{event_id}",
            )
        option_sets.append(options)

    frontier_note = "explicit_order_incompatible"
    uncertain_solution: tuple[str, list[str], list[str]] | None = None
    chosen_clause_ids: list[str] = []
    conflict_steps: set[str] = set()

    def search(
        index: int,
        chosen_steps: list[str],
        chosen_judgment_ids: list[str],
        uncertainty: str | None,
    ) -> tuple[list[str], list[str]] | None:
        nonlocal frontier_note, uncertain_solution
        if index >= len(option_sets):
            if path is not None:
                joint = _joint_path_scope(
                    clauses[chosen_clause_ids[0]], clauses[chosen_clause_ids[-1]],
                    path, events or {}, resources_by_event or {},
                )
                if joint == PredicateResult.FALSE:
                    frontier_note = "data_movement_scope_conflict"
                    return None
                if joint == PredicateResult.ABSTAIN:
                    uncertainty = "unresolved_data_movement_scope"
            if uncertainty is not None:
                uncertain_solution = (uncertainty, chosen_judgment_ids, chosen_steps)
                return None
            return chosen_judgment_ids, chosen_steps

        for judgment_id, step_id in option_sets[index]:
            required_before = reachable.get(step_id, set()) if step_id is not None else set()
            conflict = next(
                (
                    step for step in chosen_steps
                    if step in required_before
                ),
                None,
            )
            if conflict is not None:
                frontier_note = f"order_conflict:{step_id}->{conflict}"
                conflict_steps.update((step_id, conflict))
                continue
            next_uncertainty = uncertainty
            if step_id is not None:
                if step_id not in steps_by_id:
                    next_uncertainty = "missing_step_evidence"
                elif step_id in unresolved_steps:
                    next_uncertainty = "unresolved_step_order"
            clause_id = next(ref for ref in
                next(j for group in auth_results_by_event.values() for j in group
                     if j.judgment_id == judgment_id).subject_refs if ref in clauses)
            chosen_clause_ids.append(clause_id)
            result = search(
                index + 1,
                [*chosen_steps, *([step_id] if step_id is not None else [])],
                [*chosen_judgment_ids, judgment_id],
                next_uncertainty,
            )
            chosen_clause_ids.pop()
            if result is not None:
                return result
        return None

    solution = search(0, [], [], None)
    if solution is not None:
        return PathJustification(
            result=PredicateResult.TRUE,
            supporting_judgment_ids=solution[0],
            evidence_refs=solution[1],
            notes="explicit_order_compatible_clause_sequence",
        )
    if uncertain_solution is not None:
        return PathJustification(
            result=PredicateResult.ABSTAIN,
            abstain_reason=uncertain_solution[0],
            supporting_judgment_ids=uncertain_solution[1],
            evidence_refs=uncertain_solution[2],
            notes=uncertain_solution[0],
        )
    return PathJustification(
        result=PredicateResult.FALSE,
        notes=frontier_note,
        evidence_refs=sorted(conflict_steps),
    )



def _joint_path_scope(source_clause, sink_clause, path, events, resources_by_event):
    """Definition B.4 obligations concern the source/sink pairing."""
    source_id, sink_id = path.source.event_id, path.sink.event_id
    source_resources = resources_by_event.get(source_id, [])
    sink_resources = resources_by_event.get(sink_id, [])
    source_event = events.get(source_id)
    constraints = [constraint for clause in (source_clause,sink_clause)
                   for constraint in clause.constraints
                   if constraint.constraint_type in {"source_class","sink_domain","separate_invocation"}]
    results = []
    for constraint in constraints:
        if constraint.constraint_type == "sink_domain":
            destinations = [resource for resource in sink_resources if resource.resource_type == "url"]
            if not destinations:
                destinations = [resource for resource in sink_resources if resource.resource_type in {"domain","host"}]
            statuses = [PredicateResult.ABSTAIN if not r.resolved else
                        PredicateResult.TRUE if _domain_compatible(constraint.value,r.value) else PredicateResult.FALSE
                        for r in destinations]
            if statuses and all(status == PredicateResult.TRUE for status in statuses):
                results.append(PredicateResult.TRUE)
            elif statuses and all(status == PredicateResult.FALSE for status in statuses):
                results.append(PredicateResult.FALSE)
            else:
                results.append(PredicateResult.ABSTAIN)
        elif constraint.constraint_type == "source_class":
            explicit = [resource.value for resource in source_resources
                        if resource.resource_type=="source_class" and resource.resolved]
            cap = source_event.capability if source_event else ""
            classes = set(explicit)
            if cap in {"env_var_read","token_file_read","api_key_use","keychain_access","credential_store_access"}:
                classes.update({"credential","credentials","secret","secrets"})
            elif cap == "file_read":
                classes.add("file")
            classes.update(resource.origin_kind for resource in source_resources
                           if resource.origin_kind not in {"unknown","code_literal","native_ast"})
            results.append(PredicateResult.ABSTAIN if not classes else
                           PredicateResult.TRUE if constraint.value in classes else PredicateResult.FALSE)
        else:
            source_invocations = {r.value for r in source_resources if r.resource_type=="invocation" and r.resolved}
            sink_invocations = {r.value for r in sink_resources if r.resource_type=="invocation" and r.resolved}
            results.append(PredicateResult.ABSTAIN if not source_invocations or not sink_invocations else
                           PredicateResult.TRUE if source_invocations.isdisjoint(sink_invocations) else PredicateResult.FALSE)
    if PredicateResult.FALSE in results:
        return PredicateResult.FALSE
    return PredicateResult.ABSTAIN if PredicateResult.ABSTAIN in results else PredicateResult.TRUE
