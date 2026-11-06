"""Algorithm A1 structured documentary permission extraction."""

from __future__ import annotations

import re
import json
import yaml

from skillrecon.contract.normalize import normalize_clause_surface
from skillrecon.contract.parser import parse_yaml_frontmatter
from skillrecon.contract.requirements import registration_environment_fields
from skillrecon.core.enums import ClauseOperator, ClauseRole
from skillrecon.core.types import Clause, DocBlock, EvidenceSpan

_TOOL_CAPABILITY_MAP = {
    "bash": "shell_exec", "read": "file_read", "edit": "file_write",
    "write": "file_write", "websearch": "http_request", "webfetch": "http_request",
    "requests": "http_request", "network": "http_request", "shell": "shell_exec",
}
_TOOL_PATTERN = re.compile(r"([A-Za-z_][\w.-]*)(?:\(([^)]*)\))?")
_POLARITIES = {"allowed": "allowed", "allow": "allowed", "permitted": "allowed",
               "required": "allowed", "prohibited": "prohibited", "deny": "prohibited",
               "denied": "prohibited", "forbidden": "prohibited", "ask": "unknown",
               "true": "allowed", "false": "prohibited"}
_NORMATIVE_SECTION = re.compile(r"(?i)permission|prerequisite|requirement|policy|allowed|权限|前提|要求|约束")


def _field_span(doc_id: str, content: str, field: str) -> EvidenceSpan:
    yaml_text, end = parse_yaml_frontmatter(content)
    source = yaml_text if end else content
    base = content.find(source) if end else 4
    match = re.search(rf"(?ms)^{re.escape(field)}\s*:.*?(?=^[A-Za-z_][\w-]*\s*:|\Z)", source)
    local_start, local_stop = (match.start(), match.end()) if match else (0, len(source))
    start = base + local_start
    text = source[local_start:local_stop]
    stop = start + len(text)
    return EvidenceSpan(doc_id=doc_id, start_offset=start, end_offset=stop, text=text,
                        start_byte=base + len(source[:local_start].encode()),
                        end_byte=base + len(source[:local_stop].encode()))


def _make_clause(doc_id, step_id, capability, operator, target, conditions, evidence, index):
    clause_id = f"det_{doc_id}_{index}"
    target, constraints = normalize_clause_surface(
        capability, str(target) if target is not None else None,
        [str(value) for value in conditions], clause_id=clause_id, evidence=evidence,
    )
    return Clause(clause_id=clause_id, capability=capability, operator=operator,
                  role=ClauseRole.POLICY, target=target, constraints=constraints,
                  evidence_spans=[evidence], vote_agreement=1.0,
                  step_ids=[step_id], source_doc_ids=[doc_id])


def _tool_rules(value):
    values = [value] if isinstance(value, str) else value
    for entry in values:
        for match in _TOOL_PATTERN.finditer(str(entry)):
            name, selector = match.group(1), match.group(2)
            capability = _TOOL_CAPABILITY_MAP.get(name.lower())
            if capability is None and "_" in name:
                capability = name
            if capability is not None:
                conditions = [f"cmd:{selector}"] if selector and capability == "shell_exec" else ([selector] if selector else [])
                yield capability, None, conditions


def extract_from_frontmatter(frontmatter: dict[str, object], doc_id: str, frontmatter_text: str) -> list[Clause]:
    """Map actual YAML declarations without inventing permissions from prose."""
    clauses = []
    tools = frontmatter.get("allowed-tools", [])
    if tools:
        evidence = _field_span(doc_id, frontmatter_text, "allowed-tools")
        for capability, target, conditions in _tool_rules(tools):
            clauses.append(_make_clause(doc_id, f"{doc_id}_frontmatter", capability,
                           "allowed", target, conditions, evidence, len(clauses)))

    permissions = frontmatter.get("permissions", [])
    rules = []
    if isinstance(permissions, list):
        rules = permissions
    elif isinstance(permissions, dict):
        for effect, entries in permissions.items():
            if isinstance(entries, list):
                for entry in entries:
                    rules.append({"operation": entry, "effect": effect} if isinstance(entry, str)
                                 else {**entry, "effect": effect})
            elif isinstance(entries, dict):
                rules.append({"operation": effect, **entries})
            else:
                rules.append({"operation": effect, "effect": entries})
    evidence = _field_span(doc_id, frontmatter_text, "permissions")
    for rule in rules:
        if isinstance(rule,str):
            rule = {"operation":rule}
        operation = str(rule.get("operation", rule.get("capability", "")))
        capability = _TOOL_CAPABILITY_MAP.get(operation.lower(), operation)
        tool_rules = list(_tool_rules(operation))
        tool_conditions = []
        if tool_rules:
            capability, _, tool_conditions = tool_rules[0]
        polarity = _POLARITIES[str(rule.get("effect", rule.get("operator", "allowed"))).lower()]
        conditions = rule.get("conditions", rule.get("constraints", []))
        if isinstance(conditions, str):
            conditions = [conditions]
        elif isinstance(conditions, dict):
            conditions = [f"{key}:{value}" for key, value in conditions.items()]
        conditions = [*tool_conditions,*conditions]
        clauses.append(_make_clause(doc_id, f"{doc_id}_frontmatter", capability, polarity,
                       rule.get("resource", rule.get("target")), conditions, evidence, len(clauses)))
    clauses.extend(extract_required_environment(doc_id, frontmatter_text))
    return clauses


def extract_configuration_clauses(doc_id: str, relative_path: str, content: str) -> list[Clause]:
    """Native JSON permissions are governing declarations with exact source provenance."""
    from skillrecon.loader.platforms import NATIVE_CONFIGS
    if relative_path not in NATIVE_CONFIGS:
        return []
    payload = json.loads(content)
    evidence = EvidenceSpan(doc_id=doc_id,start_offset=0,end_offset=len(content),text=content,
                            start_byte=0,end_byte=len(content.encode('utf-8')))
    clauses = []
    permissions = payload.get('permissions',payload.get('permission',{}))
    if relative_path == '.claude/settings.json':
        for effect in ('allow','deny','ask'):
            for tool in permissions.get(effect,[]):
                for capability,target,conditions in _tool_rules(tool):
                    clauses.append(_make_clause(doc_id,f'{doc_id}_frontmatter',capability,
                        _POLARITIES[effect],target,conditions,evidence,f'config_{len(clauses)}'))
    elif isinstance(permissions,list):
        for rule in permissions:
            operation = rule.get('action',rule.get('operation',''))
            capability = _TOOL_CAPABILITY_MAP.get(operation.lower())
            if capability is not None:
                clauses.append(_make_clause(doc_id,f'{doc_id}_frontmatter',capability,
                    _POLARITIES[rule['effect']],rule.get('resource'),[],evidence,f'config_{len(clauses)}'))
    elif isinstance(permissions,dict):
        for operation,rules in permissions.items():
            capability = _TOOL_CAPABILITY_MAP.get(operation.lower())
            if capability is None:
                continue
            for target,effect in (rules.items() if isinstance(rules,dict) else [(None,rules)]):
                clauses.append(_make_clause(doc_id,f'{doc_id}_frontmatter',capability,
                    _POLARITIES[effect],target,[],evidence,f'config_{len(clauses)}'))
    return clauses


def extract_required_environment(doc_id: str, frontmatter_text: str) -> list[Clause]:
    yaml_text, end = parse_yaml_frontmatter(frontmatter_text)
    source = yaml_text if end else frontmatter_text
    if re.search(r"(?m)^metadata\s*:", source) is None:
        return []
    evidence = _field_span(doc_id, frontmatter_text, "metadata")
    metadata = yaml.safe_load(evidence.text)["metadata"]
    names = [name for _, values in registration_environment_fields(metadata) for name in values]
    clauses = []
    for name in dict.fromkeys(names):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        clause = _make_clause(doc_id, f"{doc_id}_frontmatter", "env_var_read", "allowed",
                              name, [f"var:{name}"], evidence, f"env_{name}")
        clauses.append(clause)
    return clauses


def extract_table_clauses(doc_id: str, blocks: list[DocBlock]) -> list[Clause]:
    """Read normative permission/prerequisite rows with their exact source span."""
    clauses = []
    for block in blocks:
        if block.block_type != "table_row":
            continue
        fields = {key.strip().lower(): value.strip().strip(chr(96))
                  for key, value in zip(block.table_headers, block.table_cells, strict=True)}
        operation = next((fields[key] for key in ("operation", "capability", "tool") if key in fields), None)
        effect = next((fields[key].lower() for key in ("effect", "operator", "polarity", "permission") if key in fields), None)
        if operation is None or (effect not in _POLARITIES and not _NORMATIVE_SECTION.search(block.heading_context)):
            continue
        evidence = EvidenceSpan(doc_id=doc_id, start_offset=block.start_offset,
                                end_offset=block.end_offset, text=block.content,
                                start_byte=block.start_byte, end_byte=block.end_byte)
        capability = _TOOL_CAPABILITY_MAP.get(operation.lower(), operation)
        target = next((fields[key] for key in ("resource", "target") if key in fields), None)
        conditions = [value for key, value in fields.items() if key in {"condition", "conditions", "scope", "constraints"} and value]
        clauses.append(_make_clause(doc_id, block.block_id, capability,
                       _POLARITIES[effect] if effect in _POLARITIES else "allowed",
                       target, conditions, evidence, f"row_{block.block_id}"))
    return clauses
