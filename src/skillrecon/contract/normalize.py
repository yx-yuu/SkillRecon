"""Helpers for normalizing clause targets and scope constraints."""

from __future__ import annotations

import re

from skillrecon.core.types import Constraint, EvidenceSpan

_FILE_CAPABILITIES = {
    "file_read",
    "file_write",
    "file_delete",
    "file_execute",
    "file_permission_change",
    "temp_file_create",
    "token_file_read",
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
_EXEC_CAPABILITIES = {
    "shell_exec",
    "subprocess_spawn",
    "dynamic_import",
    "process_kill",
}
_ENV_CAPABILITIES = {
    "env_var_read",
    "api_key_use",
}

_URL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_DOMAIN_RE = re.compile(
    r"^(?:\*\.)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[^\s]*)?$"
)
_ENV_RE = re.compile(r"^[A-Z][A-Z0-9_]*(?:[\*\?][A-Z0-9_]*)*$")
_SQL_RE = re.compile(
    r"(?i)\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\bcreate\b|\balter\b|\bdrop\b"
)
_USER_INPUT_ONLY_RE = re.compile(
    r"(?i)\buser[- ]provided\b|\buser input\b|\bprovided by the user\b|用户提供|用户输入|任务参数"
)
_CONFIG_BOUND_RE = re.compile(
    r"(?i)\bconfigured?\b|\bconfig(?:uration)?\b|\blisted in\b|\bdefined in\b|配置|配置文件"
)
_NO_NETWORK_RE = re.compile(
    r"(?i)\bno network\b|\bnetwork[- ]free\b|\bnever accesses? the network\b|不访问网络|禁止联网"
)
_READ_ONLY_RE = re.compile(
    r"(?i)\bread[- ]only\b|\bonly reads?\b|\bdoes not modify\b|只读|仅读取|不修改"
)
_WRITE_ONLY_RE = re.compile(
    r"(?i)\bwrite[- ]only\b|\bonly writes?\b|仅写入|只写"
)
_SHELL_TOKEN_RE = re.compile(
    r"(?i)^(?:python(?:3)?|bash|sh|zsh|uv|pip(?:3)?|npm|pnpm|poetry|node|npx)\b"
)
_COMMON_FILE_SUFFIXES = {
    "bash",
    "cfg",
    "csv",
    "css",
    "gif",
    "html",
    "ini",
    "jpeg",
    "jpg",
    "js",
    "json",
    "log",
    "md",
    "pdf",
    "png",
    "py",
    "sh",
    "sql",
    "svg",
    "ts",
    "txt",
    "xml",
    "yaml",
    "yml",
}


def normalize_clause_surface(
    capability: str,
    target: str | None,
    raw_constraints: list[str],
    *,
    clause_id: str,
    evidence: EvidenceSpan | None = None,
) -> tuple[str | None, list[Constraint]]:
    """Normalize raw clause surface strings into typed constraints."""
    cleaned_target = _clean_surface(target)
    if not cleaned_target:
        cleaned_target = None
    constraints: list[Constraint] = []
    seen: set[tuple[str, str]] = set()

    promote_target = _should_promote_target_to_constraint(capability, cleaned_target)
    next_index = 0
    if promote_target and cleaned_target:
        next_index = _append_constraint(
            constraints,
            seen,
            capability,
            cleaned_target,
            clause_id=clause_id,
            next_index=next_index,
            evidence=evidence,
        )
        cleaned_target = None

    for raw_constraint in raw_constraints:
        next_index = _append_constraint(
            constraints,
            seen,
            capability,
            raw_constraint,
            clause_id=clause_id,
            next_index=next_index,
            evidence=evidence,
        )

    return cleaned_target, constraints


def _append_constraint(
    constraints: list[Constraint],
    seen: set[tuple[str, str]],
    capability: str,
    raw_value: str | None,
    *,
    clause_id: str,
    next_index: int,
    evidence: EvidenceSpan | None,
) -> int:
    normalized = infer_typed_constraint(capability, raw_value)
    if normalized is None:
        cleaned_value = _clean_surface(raw_value)
        if not cleaned_value:
            return next_index
        normalized = ("scope", cleaned_value)

    key = normalized
    if key in seen:
        return next_index
    seen.add(key)
    constraints.append(
        Constraint(
            constraint_id=f"{clause_id}_cst{next_index}",
            constraint_type=normalized[0],
            value=normalized[1],
            evidence=evidence,
        )
    )
    return next_index + 1


def infer_typed_constraint(
    capability: str,
    raw_value: str | None,
) -> tuple[str, str] | None:
    """Infer a typed constraint from a raw surface string."""
    value = _clean_surface(raw_value)
    if not value:
        return None

    semantic = _infer_semantic_constraint(value)
    if semantic is not None:
        return semantic

    if capability in _ENV_CAPABILITIES:
        return _infer_env_constraint(value)
    if capability in _NETWORK_CAPABILITIES:
        return _infer_network_constraint(value)
    if capability in _EXEC_CAPABILITIES:
        return _infer_command_constraint(value)
    if capability in _FILE_CAPABILITIES:
        return _infer_file_constraint(value)
    if capability == "sql_exec" and _SQL_RE.search(value):
        return "query", value

    return (
        _infer_network_constraint(value)
        or _infer_env_constraint(value)
        or _infer_file_constraint(value)
        or _infer_command_constraint(value)
    )


def _infer_semantic_constraint(value: str) -> tuple[str, str] | None:
    if _USER_INPUT_ONLY_RE.search(value):
        return "user_input_only", "user_input_only"

    if _NO_NETWORK_RE.search(value):
        return "side_effect_bound", "no_network"
    if _READ_ONLY_RE.search(value):
        return "side_effect_bound", "read_only"
    if _WRITE_ONLY_RE.search(value):
        return "side_effect_bound", "write_only"

    if _CONFIG_BOUND_RE.search(value):
        return "config_bound", value

    return None


def _infer_file_constraint(value: str) -> tuple[str, str] | None:
    if value == "*":
        return "path_glob", value
    if _looks_like_file_glob(value):
        return "file_glob", value
    if _looks_like_path(value):
        return ("path_glob" if _has_glob(value) else "path", value)
    return None


def _infer_network_constraint(value: str) -> tuple[str, str] | None:
    if _looks_like_url(value):
        return ("url_glob" if _has_glob(value) else "url", value)
    if _looks_like_domain(value):
        return ("domain_glob" if _has_glob(value) else "domain", value)
    return None


def _infer_env_constraint(value: str) -> tuple[str, str] | None:
    if _looks_like_env_name(value):
        return ("env_glob" if _has_glob(value) else "env_var", value)
    return None


def _infer_command_constraint(value: str) -> tuple[str, str] | None:
    if not value:
        return None
    if _looks_like_url(value) or _looks_like_domain(value) or _looks_like_env_name(value):
        return None
    executable = _command_head(value)
    if not executable:
        return None
    return ("command_glob" if _has_glob(executable) else "command", executable)


def _clean_surface(value: str | None) -> str:
    if value is None:
        return ""
    cleaned = value.strip()
    if cleaned.startswith(("`", '"', "'")) and cleaned.endswith(("`", '"', "'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _should_promote_target_to_constraint(
    capability: str,
    target: str | None,
) -> bool:
    if not target:
        return False
    return capability in _EXEC_CAPABILITIES | _ENV_CAPABILITIES


def _has_glob(value: str) -> bool:
    return any(ch in value for ch in "*?[]")


def _looks_like_url(value: str) -> bool:
    return bool(_URL_RE.match(value))


def _looks_like_domain(value: str) -> bool:
    if not _DOMAIN_RE.match(value):
        return False
    if value.startswith(("/", "./", "../", "~/")) or "\\" in value:
        return False
    host = value.split("/", 1)[0]
    labels = host.removeprefix("*.").split(".")
    return not (len(labels) == 2 and labels[-1].lower() in _COMMON_FILE_SUFFIXES)


def _looks_like_env_name(value: str) -> bool:
    return bool(_ENV_RE.match(value))


def _looks_like_file_glob(value: str) -> bool:
    return value.startswith("*.") or value.startswith("**/*.")


def _looks_like_path(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~/")):
        return True
    if "/" in value or "\\" in value:
        return True
    if value.endswith("/"):
        return True
    if value.startswith("*."):
        return False
    return bool(re.search(r"\.[A-Za-z0-9]{1,8}(?:$|[\*\?])", value))


def _command_head(value: str) -> str | None:
    token = value.strip().split(None, 1)[0]
    if not token:
        return None
    token = token.rstrip(",:;")
    if not token:
        return None
    if _SHELL_TOKEN_RE.match(token):
        return token
    if token.startswith(("/", "./", "../", "~/")):
        return token
    if "/" in token or "\\" in token:
        return token
    if re.search(r"\.(?:py|sh|bash|js|ts|mjs|cjs)$", token, re.IGNORECASE):
        return token
    if _has_glob(token):
        return token
    return None


def looks_typed_constraint_applicable(constraint: Constraint) -> bool:
    """Whether a constraint participates in deterministic scope checking."""
    return constraint.constraint_type.lower() != "scope"


def is_resource_constraint(constraint: Constraint) -> bool:
    """Whether a typed constraint primarily narrows the concrete resource itself."""
    return constraint.constraint_type.lower() in {
        "path",
        "path_glob",
        "file_glob",
        "domain",
        "domain_glob",
        "url",
        "url_glob",
        "env_var",
        "env_glob",
        "command",
        "command_glob",
        "query",
    }


def is_scope_constraint(constraint: Constraint) -> bool:
    """Whether a constraint should be enforced as higher-level scope logic."""
    return constraint.constraint_type.lower() in {
        "user_input_only",
        "config_bound",
        "side_effect_bound",
    }
