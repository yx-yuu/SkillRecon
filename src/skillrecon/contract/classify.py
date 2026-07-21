"""Heuristic role classification for canonical contract clauses."""

from __future__ import annotations

import re

from skillrecon.core.enums import ClauseOperator, ClauseRole
from skillrecon.core.types import Clause, PackageManifest

_ACTION_CUE_RE = re.compile(
    r"(?i)必须|应当|需要|先|后|检查|选择|读取|查阅|执行|运行|生成|保存|升级|安装|"
    r"点击|打开|访问|导航|过滤|搜索|添加|移除|匹配|复用|"
    r"\bmust\b|\bneed\b|\bcheck\b|\bselect\b|\bread\b|\bconsult\b|\brun\b|"
    r"\bexecute\b|\bgenerate\b|\bsave\b|\binstall\b|\buse\b|\bopen\b|\bvisit\b|"
    r"\bnavigate\b|\bfilter\b|\bsearch\b|\badd\b|\bremove\b|\breuse\b"
)
_NORMATIVE_CUE_RE = re.compile(
    r"(?i)禁止|不得|仅|只|始终|总是|永远|重要|默认|优先|前|后|"
    r"\bnever\b|\balways\b|\bonly\b|\bdefault\b|\bimportant\b|\bbefore\b|\bafter\b|"
    r"\bunless\b|\bwhen\b|\bif\b"
)
_KNOWLEDGE_CUE_RE = re.compile(
    r"(?i)示例|example|格式|schema|签名|参数|默认值|配置项|支持|兼容|说明|速查|索引|"
    r"单位|测试结果|依赖|输出|特性|类型|feature|support(?:s|ed)?|"
    r"default value|parameter|output|dependency|signature|index|reference"
)
_WORKFLOW_PATH_HINT_RE = re.compile(r"(?i)(?:^|/)(?:skill|readme)\.md$")
_OPERATIONAL_POLICY_CUE_RE = re.compile(
    r"(?i)\brequired\b|\brequires\b|\ball requests require\b|\bendpoint\b|\bauthorization\b|"
    r"\bbearer\b|\bx-api-key\b|\bapi key\b|\benvironment variable\b|\benv var\b|"
    r"\bget /|\bpost /|\bput /|\bdelete /|"
    r"必需|必须|需要|环境变量|接口|端点|请求头|鉴权"
)
_SENSITIVE_POLICY_CAPABILITIES = {
    "http_request",
    "env_var_read",
    "api_key_use",
    "shell_exec",
    "subprocess_spawn",
    "dynamic_import",
    "sql_exec",
}


def classify_clause_roles(
    clauses: list[Clause],
    manifest: PackageManifest,
) -> list[Clause]:
    """Assign policy/knowledge role to each canonical clause."""
    doc_to_depth = {doc.doc_id: doc.depth for doc in manifest.documents}
    file_by_id = {entry.file_id: entry.relative_path for entry in manifest.files}
    doc_to_path = {
        doc.doc_id: file_by_id.get(doc.file_id, "")
        for doc in manifest.documents
    }

    classified: list[Clause] = []
    for clause in clauses:
        role = infer_clause_role(clause, doc_to_depth, doc_to_path)
        classified.append(clause.model_copy(update={"role": role}))
    return classified


def infer_clause_role(
    clause: Clause,
    doc_to_depth: dict[str, int],
    doc_to_path: dict[str, str],
) -> ClauseRole:
    """Infer whether a clause is policy-bearing or knowledge-bearing."""
    if clause.operator in {ClauseOperator.ALLOWED, ClauseOperator.PROHIBITED}:
        return ClauseRole.POLICY

    combined_text = _combined_clause_text(clause)
    if _ACTION_CUE_RE.search(combined_text) or _NORMATIVE_CUE_RE.search(combined_text):
        return ClauseRole.POLICY

    source_docs = clause.source_doc_ids
    root_sourced = any(doc_to_depth.get(doc_id, 0) == 0 for doc_id in source_docs)
    path_hints = [doc_to_path.get(doc_id, "") for doc_id in source_docs]
    has_specific_surface = bool(clause.target) or any(
        constraint.constraint_type.lower() != "scope" or len(constraint.value.strip()) >= 8
        for constraint in clause.constraints
    )
    has_operational_policy = _OPERATIONAL_POLICY_CUE_RE.search(combined_text) is not None
    is_sensitive_policy_candidate = clause.capability in _SENSITIVE_POLICY_CAPABILITIES

    if not root_sourced and not (
        is_sensitive_policy_candidate and has_specific_surface and has_operational_policy
    ):
        return ClauseRole.KNOWLEDGE

    if _KNOWLEDGE_CUE_RE.search(combined_text) and not (
        is_sensitive_policy_candidate and has_specific_surface and has_operational_policy
    ):
        return ClauseRole.KNOWLEDGE

    if "|" in combined_text and not (
        is_sensitive_policy_candidate and has_specific_surface and has_operational_policy
    ):
        return ClauseRole.KNOWLEDGE

    if any(_WORKFLOW_PATH_HINT_RE.search(path) for path in path_hints if path):
        return ClauseRole.POLICY

    return ClauseRole.POLICY


def _combined_clause_text(clause: Clause) -> str:
    parts: list[str] = [clause.capability]
    if clause.target:
        parts.append(clause.target)
    parts.extend(constraint.value for constraint in clause.constraints)
    parts.extend(span.text for span in clause.evidence_spans)
    return " ".join(part for part in parts if part)
