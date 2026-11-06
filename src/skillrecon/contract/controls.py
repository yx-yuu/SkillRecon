"""Recover explicit configuration and workflow controls from admitted prose."""

from __future__ import annotations

import re

from skillrecon.core.types import Clause, Constraint, EvidenceSpan, Step

CONTROL_TYPES = {"config_enabled", "empty_disables", "standalone_call", "retry_on_error", "credential_source", "dependency_declared"}


def forbidden_standalone_calls(text: str) -> list[str]:
    """Named calls governed by an explicit separate-invocation prohibition."""
    names = []
    for match in re.finditer(r"(?:never|do not|must not)\s+call\s+(.+?)\s+separately", text, re.I):
        names.extend(re.findall(r"`([A-Za-z_]\w*)`",match[1]))
    for match in re.finditer(r"(?:禁止|不要|不得|切勿)(?:单独|分别)调用([^。\n—]+)",text):
        names.extend(re.findall(r"`([A-Za-z_]\w*)`",match[1]))
    return sorted(set(names))


def extract_control_clauses(doc_contents: dict[str, str], steps: list[Step]) -> list[Clause]:
    """Registered prose forms only; examples and descriptive code are not policy."""
    clauses = []
    for doc_id, content in doc_contents.items():
        offset, fenced = 0, False
        for raw_line in content.splitlines(keepends=True):
            line = raw_line.strip()
            if line.startswith(("```", "~~~")):
                fenced = not fenced
            controls = []
            if not fenced:
                if (re.search(r"RPC\s*(?:密钥|secret|key)", line, re.I)
                        and re.search(r"<YOUR_[A-Z_]*(?:SECRET|KEY|TOKEN)[A-Z_]*>", line)
                        and re.search(r"环境变量|\benvironment\b|\benv\b", line, re.I)
                        and re.search(r"请|必须|应当|\b(?:use|must|provide)\b", line, re.I)):
                    controls.append(("credential_source", "configured_input", "allowed"))
                if ("permissions.defaultMode" in line
                        and re.search(r"\b(?:set|configure|enable|use)\b|配置|设置|设为|启用", line, re.I)
                        and not re.search(r"\b(?:example|sample|reference)\b|示例|例如", line, re.I)):
                    modes = re.findall(r"\b(?:bypassPermissions|acceptEdits|dontAsk|plan|default)\b", line)
                    if len(set(modes)) == 1:
                        start = offset + raw_line.index(line)
                        evidence = EvidenceSpan(doc_id=doc_id, start_offset=start,
                                                end_offset=start + len(line), text=line)
                        operator = "prohibited" if re.search(
                            r"\b(?:never|do not|must not)\b|禁止|不得|不要", line, re.I) else "allowed"
                        key = f"permissions.defaultMode={modes[0]}"
                        clause_id = f"permission-{doc_id}-{start}"
                        clauses.append(Clause(
                            clause_id=clause_id, capability="file_permission_change",
                            operator=operator, role="policy", target=key,
                            evidence_spans=[evidence], vote_agreement=1.0,
                            source_doc_ids=[doc_id], step_ids=[
                                s.step_id for s in steps if s.doc_id == doc_id and s.evidence
                                and s.evidence.start_offset <= start < s.evidence.end_offset],
                        ))
                flag = re.search(r"`(\w+)`\s*:\s*true\s*/\s*false", line, re.I)
                if flag and re.search(r"启用.*禁用|enable.*disable", line, re.I):
                    controls.append(("config_enabled", flag[1], "allowed"))
                if re.search(r"set (?:it )?to empty to disable", line, re.I):
                    names = re.findall(r"(?:`|\*\*)([A-Z][A-Z0-9_]+)(?:`|\*\*)", line)
                    if len(set(names)) == 1:
                        controls.append(("empty_disables", names[0], "allowed"))
                controls.extend(("standalone_call",name,"prohibited") for name in forbidden_standalone_calls(line))
                if re.search(r"\bno retry on error\b", line, re.I) and re.search(r"\bdo not retry\b", line, re.I):
                    controls.append(("retry_on_error", "automatic_retry", "prohibited"))
            if controls:
                start = offset + raw_line.index(line)
                evidence = EvidenceSpan(doc_id=doc_id, start_offset=start,
                                        end_offset=start + len(line), text=line)
                source_steps = [s.step_id for s in steps if s.doc_id == doc_id and s.evidence
                                and s.evidence.start_offset <= start < s.evidence.end_offset]
                for kind, key, operator in controls:
                    clause_id = f"ctl-{doc_id}-{start}-{kind}-{key}"
                    clauses.append(Clause(
                        clause_id=clause_id,
                        capability="api_key_use" if kind == "credential_source" else "http_request",
                        operator=operator,
                        target="jsonrpc" if kind == "credential_source" else None,
                        role="policy", constraints=[Constraint(constraint_id=f"{clause_id}-condition",
                                                               constraint_type=kind, value=key, evidence=evidence)],
                        evidence_spans=[evidence], vote_agreement=1.0,
                        step_ids=source_steps, source_doc_ids=[doc_id],
                    ))
            offset += len(raw_line)
    return clauses
