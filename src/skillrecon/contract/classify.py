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
    "subprocess",
    "dynamic_import",
    "sql_exec",
}
_DOCUMENT_FORM_RE = re.compile(
    r"(?i)\bexamples?\b|\bschemas?\b|\breference\b|\bsample\b|"
    r"\bfeature description\b|\bdefault values?\b|示例|格式说明|参考示例|默认值|特性说明|"
    r'"why"\s*:|(?:仅|只)?支持[^\n]{0,120}不支持|'
    r"\bonly supports?\b[^\n]{0,120}\b(?:does not|doesn't) support\b"
)
_GOVERNING_EVIDENCE_RE = re.compile(
    r"(?i)\bmust\b|\bshall\b|\brequired\b|\brequires\b|"
    r"\bdo not\b|\bnever\b|\ballowed[-_]tools\b|"
    r"\b(?:is|are)\s+(?:allowed|permitted|authorized|prohibited|forbidden)\b|"
    r"\b(?:may|can)\s+(?:use|read|write|send|post|fetch|connect|upload|execute|run|delete|install|import)\b|"
    r"必须|不得|禁止|切勿|不要|仅允许|只能"
)


def classify_clause_roles(
    clauses: list[Clause],
    manifest: PackageManifest,
    steps=None,
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
        if clause.clause_id.startswith('det_'):
            classified.append(clause.model_copy(update={"role": ClauseRole.POLICY}))
            continue
        role = infer_clause_role(clause, doc_to_depth, doc_to_path)
        origins = [step for step in steps or [] if step.step_id in clause.step_ids]
        if any(step.step_type == "example_code_block" for step in origins):
            role = ClauseRole.KNOWLEDGE
        elif any(re.search(r"(?i)permission|prerequisite|requirements?|policy|instructions?|security|privacy|权限|前提|要求|约束|隐私|安全",
                           step.heading_context) for step in origins):
            role = ClauseRole.POLICY
        classified.append(clause.model_copy(update={"role": role}))
    return classified


def infer_clause_role(
    clause: Clause,
    doc_to_depth: dict[str, int],
    doc_to_path: dict[str, str],
) -> ClauseRole:
    """An extracted operator cannot turn descriptive prose into policy."""
    evidence_text = " ".join(span.text for span in clause.evidence_spans)
    prose = re.sub(r"\x60[^\x60]*\x60", " ", evidence_text)
    governing = bool(_GOVERNING_EVIDENCE_RE.search(prose) or _ACTION_CUE_RE.match(prose.strip()))
    return ClauseRole.POLICY if governing else ClauseRole.KNOWLEDGE


def _combined_clause_text(clause: Clause) -> str:
    parts: list[str] = [clause.capability]
    if clause.target:
        parts.append(clause.target)
    parts.extend(constraint.value for constraint in clause.constraints)
    parts.extend(span.text for span in clause.evidence_spans)
    return " ".join(part for part in parts if part)
