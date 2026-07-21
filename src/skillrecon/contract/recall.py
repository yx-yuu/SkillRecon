"""Behavior-aware targeted contract recall helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath

from skillrecon.core.enums import ClauseOperator, ClauseRole
from skillrecon.core.sensitivity import event_requires_authorization
from skillrecon.core.types import (
    CapabilityEvent,
    Clause,
    Constraint,
    DocBlock,
    PackageManifest,
    ResourceUse,
    Step,
)
from skillrecon.reconcile.predicate import _resource_matches_clause

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "only",
    "this",
    "that",
    "your",
    "user",
    "true",
    "false",
    "none",
    "json",
    "yaml",
    "readme",
    "skill",
}
_POLICY_CUES = (
    "allow",
    "allowed",
    "authorize",
    "authorized",
    "permit",
    "permitted",
    "may",
    "can",
    "must",
    "should",
    "only",
    "without",
    "except",
    "forbid",
    "forbidden",
    "prohibit",
    "prohibited",
    "do not",
    "don't",
    "cannot",
    "can't",
    "never",
    "must not",
)
_CAPABILITY_HINTS: dict[str, tuple[str, ...]] = {
    "http_request": ("http", "https", "request", "api", "fetch"),
    "smtp_send": ("smtp", "email", "mail", "send"),
    "shell_exec": ("shell", "command", "bash", "run", "execute"),
    "subprocess_spawn": ("subprocess", "process", "spawn", "run"),
    "dynamic_import": ("import", "module", "plugin", "package"),
    "env_var_read": ("env", "environment", "variable", "secret", "token"),
    "data_encode_send": ("upload", "send", "export", "post", "submit"),
    "ssh_connect": ("ssh", "remote", "connect"),
    "sql_exec": ("sql", "query", "database"),
}


def select_recall_blocks(
    *,
    doc_blocks: list[DocBlock],
    manifest: PackageManifest,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    a_req: set[str],
    max_blocks: int = 12,
    max_blocks_per_doc: int = 4,
) -> list[DocBlock]:
    """Select high-yield doc blocks for a focused recall pass."""
    sensitive_events = [event for event in events if event_requires_authorization(event, a_req)]
    if not sensitive_events:
        return []

    resources_by_event: dict[str, list[ResourceUse]] = defaultdict(list)
    for resource in resources:
        if resource.event_id is not None:
            resources_by_event[resource.event_id].append(resource)

    docs_by_unit: dict[str, set[str]] = defaultdict(set)
    for link in manifest.links:
        if link.target_unit_id:
            docs_by_unit[link.target_unit_id].add(link.source_doc_id)

    event_queries = [
        (
            event,
            _event_query_tokens(event, resources_by_event.get(event.event_id, [])),
            docs_by_unit.get(event.unit_id, set()),
        )
        for event in sensitive_events
    ]

    scored_blocks: list[tuple[int, int, str, int, DocBlock]] = []
    for block in doc_blocks:
        if block.block_type == "frontmatter":
            continue
        block_tokens = _block_tokens(block)
        if not block_tokens:
            continue
        score = 0
        matched_caps: set[str] = set()
        for event, query_tokens, linked_docs in event_queries:
            shared = block_tokens & query_tokens
            if shared:
                score += min(len(shared), 5)
                matched_caps.add(event.capability)
            if block.doc_id in linked_docs:
                score += 6
                matched_caps.add(event.capability)
        policy_score = _policy_cue_score(block)
        score += policy_score
        if score <= 0:
            continue
        scored_blocks.append(
            (score, policy_score, len(matched_caps), block.doc_id, block.start_offset, block)
        )

    scored_blocks.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))

    selected: list[DocBlock] = []
    per_doc_counts: dict[str, int] = defaultdict(int)
    seen_block_ids: set[str] = set()
    for _score, _policy_score, _matched_count, _doc_id, _offset, block in scored_blocks:
        if block.block_id in seen_block_ids:
            continue
        if per_doc_counts[block.doc_id] >= max_blocks_per_doc:
            continue
        selected.append(block)
        seen_block_ids.add(block.block_id)
        per_doc_counts[block.doc_id] += 1
        if len(selected) >= max_blocks:
            break
    return selected


def select_recall_steps(
    *,
    steps: list[Step],
    manifest: PackageManifest,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    a_req: set[str],
    max_steps: int = 12,
    max_steps_per_doc: int = 4,
) -> list[Step]:
    """Select high-yield steps for a focused recall pass."""
    sensitive_events = [event for event in events if event_requires_authorization(event, a_req)]
    if not sensitive_events:
        return []

    resources_by_event: dict[str, list[ResourceUse]] = defaultdict(list)
    for resource in resources:
        if resource.event_id is not None:
            resources_by_event[resource.event_id].append(resource)

    docs_by_unit: dict[str, set[str]] = defaultdict(set)
    for link in manifest.links:
        if link.target_unit_id:
            docs_by_unit[link.target_unit_id].add(link.source_doc_id)

    event_queries = [
        (
            event,
            _event_query_tokens(event, resources_by_event.get(event.event_id, [])),
            docs_by_unit.get(event.unit_id, set()),
        )
        for event in sensitive_events
    ]

    scored_steps: list[tuple[int, int, str, int, Step]] = []
    for step in steps:
        step_tokens = _step_tokens(step)
        if not step_tokens:
            continue
        score = 0
        matched_caps: set[str] = set()
        for event, query_tokens, linked_docs in event_queries:
            shared = step_tokens & query_tokens
            if shared:
                score += min(len(shared), 5)
                matched_caps.add(event.capability)
            if step.doc_id in linked_docs:
                score += 6
                matched_caps.add(event.capability)
        policy_score = _policy_text_score(step.heading_context, step.text)
        score += policy_score
        if score <= 0:
            continue
        scored_steps.append(
            (score, policy_score, len(matched_caps), step.doc_id, step.local_index, step)
        )

    scored_steps.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))

    selected: list[Step] = []
    per_doc_counts: dict[str, int] = defaultdict(int)
    seen_step_ids: set[str] = set()
    for _score, _policy_score, _matched_count, _doc_id, _offset, step in scored_steps:
        if step.step_id in seen_step_ids:
            continue
        if per_doc_counts[step.doc_id] >= max_steps_per_doc:
            continue
        selected.append(step)
        seen_step_ids.add(step.step_id)
        per_doc_counts[step.doc_id] += 1
        if len(selected) >= max_steps:
            break
    return selected


def select_recall_events(
    *,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    clauses: list[Clause],
    a_req: set[str],
) -> list[CapabilityEvent]:
    """Keep only sensitive events that lack explicit policy coverage."""
    sensitive_events = [event for event in events if event_requires_authorization(event, a_req)]
    if not sensitive_events:
        return []

    resources_by_event: dict[str, list[ResourceUse]] = defaultdict(list)
    for resource in resources:
        if resource.event_id is not None:
            resources_by_event[resource.event_id].append(resource)

    explicit_policy_clauses = [
        clause
        for clause in clauses
        if clause.role == ClauseRole.POLICY
        and clause.operator in {ClauseOperator.ALLOWED, ClauseOperator.PROHIBITED}
    ]
    selected: list[CapabilityEvent] = []
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()
    for event in sensitive_events:
        if _event_has_explicit_policy(
            event,
            resources_by_event.get(event.event_id, []),
            explicit_policy_clauses,
        ):
            continue
        signature = _recall_event_signature(
            event,
            resources_by_event.get(event.event_id, []),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(event)
    return selected


def build_recall_focus_context(
    *,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    a_req: set[str],
    max_events: int = 8,
    max_resources_per_event: int = 3,
) -> str:
    """Build a concise summary of observed sensitive behaviors for recall prompting."""
    sensitive_events = [event for event in events if event_requires_authorization(event, a_req)]
    if not sensitive_events:
        return ""

    resources_by_event: dict[str, list[ResourceUse]] = defaultdict(list)
    for resource in resources:
        if resource.event_id is not None:
            resources_by_event[resource.event_id].append(resource)

    lines: list[str] = []
    for index, event in enumerate(sensitive_events[:max_events], start=1):
        event_resources = resources_by_event.get(event.event_id, [])
        rendered_resources = _render_event_resources(
            event_resources,
            max_items=max_resources_per_event,
        )
        location = event.file_path or event.location or event.unit_id
        detail_parts = [
            f"capability={event.capability}",
            f"location={location}",
        ]
        if event.api_call:
            detail_parts.append(f"api={event.api_call}")
        elif event.detail:
            detail_parts.append(f"detail={event.detail}")
        if rendered_resources:
            detail_parts.append(f"resources={rendered_resources}")
        elif event.arguments:
            arguments = [
                argument
                for argument in event.arguments[:max_resources_per_event]
                if argument
            ]
            detail_parts.append(
                "arguments=" + ", ".join(arguments)
            )
        lines.append(f"{index}. " + " | ".join(detail_parts))
    return "\n".join(lines)


def merge_recall_clauses(
    *,
    existing_clauses: list[Clause],
    recall_clauses: list[Clause],
    target_capabilities: set[str],
) -> list[Clause]:
    """Merge targeted recall clauses into an existing contract table."""
    existing_keys = {_recall_clause_key(clause) for clause in existing_clauses}
    seen_new: set[tuple[object, ...]] = set()
    accepted: list[Clause] = []

    for clause in recall_clauses:
        if clause.capability not in target_capabilities:
            continue
        if clause.role != ClauseRole.POLICY:
            continue
        if clause.operator.value == "unknown":
            continue
        key = _recall_clause_key(clause)
        if key in existing_keys or key in seen_new:
            continue
        seen_new.add(key)
        accepted.append(clause)

    if not accepted:
        return list(existing_clauses)
    return [*existing_clauses, *_reindex_clauses(accepted, start_index=len(existing_clauses))]


def _event_query_tokens(event: CapabilityEvent, resources: list[ResourceUse]) -> set[str]:
    parts = [
        event.capability.replace("_", " "),
        event.api_call,
        event.detail,
        *event.arguments,
        *(resource.value for resource in resources),
        *(resource.resource_type for resource in resources),
        PurePosixPath(event.file_path).name if event.file_path else "",
    ]
    text = " ".join(
        part for part in parts if part and not part.startswith("<dynamic-")
    )
    tokens = set(_tokenize(text))
    tokens.update(_CAPABILITY_HINTS.get(event.capability, ()))
    return tokens


def _render_event_resources(resources: list[ResourceUse], *, max_items: int) -> str:
    values: list[str] = []
    seen: set[tuple[str, str]] = set()
    for resource in resources:
        normalized_value = resource.value.strip()
        if not normalized_value or normalized_value.startswith("<dynamic-"):
            continue
        key = (resource.resource_type, normalized_value)
        if key in seen:
            continue
        seen.add(key)
        values.append(f"{resource.resource_type}:{normalized_value}")
        if len(values) >= max_items:
            break
    return ", ".join(values)


def _event_has_explicit_policy(
    event: CapabilityEvent,
    resources: list[ResourceUse],
    clauses: list[Clause],
) -> bool:
    for clause in clauses:
        if clause.capability != event.capability:
            continue
        if not clause.target and not clause.constraints:
            return True
        resolved_resources = [resource for resource in resources if resource.resolved]
        if resolved_resources and any(
            _resource_matches_clause(clause, resource)
            for resource in resolved_resources
        ):
            return True
    return False


def _recall_event_signature(
    event: CapabilityEvent,
    resources: list[ResourceUse],
) -> tuple[str, tuple[str, ...]]:
    hints: list[str] = []
    for resource in resources:
        if not resource.resolved:
            continue
        value = resource.value.strip()
        if not value or value.startswith("<dynamic-"):
            continue
        hints.append(f"{resource.resource_type}:{value.lower()}")
    if not hints:
        for value in [event.api_call, event.detail, *event.arguments]:
            if not value or value.startswith("<dynamic-"):
                continue
            hints.append(value.strip().lower())
            break
    return (event.capability, tuple(sorted(hints[:3])))


def _block_tokens(block: DocBlock) -> set[str]:
    return set(_tokenize(" ".join(part for part in (block.heading_context, block.content) if part)))


def _step_tokens(step: Step) -> set[str]:
    return set(_tokenize(" ".join(part for part in (step.heading_context, step.text) if part)))


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower().replace("_", " "))
        if len(token) >= 3 and token not in _STOPWORDS
    ]


def _policy_cue_score(block: DocBlock) -> int:
    return _policy_text_score(block.heading_context, block.content)


def _policy_text_score(*parts: str) -> int:
    text = " ".join(part for part in parts if part).lower()
    return sum(2 for cue in _POLICY_CUES if cue in text)


def _recall_clause_key(clause: Clause) -> tuple[object, ...]:
    return (
        tuple(sorted(clause.step_ids)),
        clause.capability,
        clause.operator.value,
        (clause.target or "").strip().lower(),
        tuple(sorted(constraint.value.strip().lower() for constraint in clause.constraints)),
    )


def _reindex_clauses(clauses: list[Clause], *, start_index: int) -> list[Clause]:
    reindexed: list[Clause] = []
    for offset, clause in enumerate(clauses, start=start_index):
        clause_id = f"c{offset}"
        constraints = [
            Constraint(
                constraint_id=f"{clause_id}_cst{index}",
                constraint_type=constraint.constraint_type,
                value=constraint.value,
                evidence=constraint.evidence,
            )
            for index, constraint in enumerate(clause.constraints)
        ]
        reindexed.append(
            clause.model_copy(
                update={
                    "clause_id": clause_id,
                    "constraints": constraints,
                }
            )
        )
    return reindexed
