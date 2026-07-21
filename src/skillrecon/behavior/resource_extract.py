"""Resource extraction and capability refinement helpers."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

_URL_RE = re.compile(r"https?://[^\s'\"`]+")
_ENV_CALL_RE = re.compile(
    r"""(?:getenv\(\s*['"]|environ\[['"])([A-Z_][A-Z0-9_]*)"""
)
_JS_ENV_RE = re.compile(r"""process\.env(?:\.|\[['"])([A-Z_][A-Z0-9_]*)""")
_BASH_ENV_RE = re.compile(r"""\$\{?([A-Z_][A-Z0-9_]*)\}?""")
_QUOTED_PATH_RE = re.compile(r"""['"]((?:~|/|\.\.?/)[^'"]+)['"]""")
_QUOTED_FILE_RE = re.compile(
    r"""['"]([A-Za-z0-9_.-]+\.(?:py|js|ts|sh|json|ya?ml|md|txt|html|csv|xml|sql|log|cfg|ini))['"]"""
)
_QUOTED_DOMAIN_RE = re.compile(r"""['"]((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?::\d+)?['"]""")
_REDIRECT_RE = re.compile(r"""(?:^|\s)>>?\s*([^\s;|&]+)""")
_IMPORT_LITERAL_RE = re.compile(
    r"""\b(?:import|require|__import__|import_module)\s*\(\s*['"]([^'"]+)['"]"""
)
_SPEC_FROM_FILE_RE = re.compile(
    r"""\bspec_from_file_location\(\s*['"][^'"]+['"]\s*,\s*['"]([^'"]+)['"]"""
)
_PACKAGE_INSTALL_CMD_RE = re.compile(
    r"""\b(?:pip(?:3)?\s+install|npm\s+install|pnpm\s+add|yarn\s+add)\s+([A-Za-z0-9_./@-]+)"""
)
_SUBPROCESS_RE = re.compile(
    r"""\b(?:subprocess\.(?:run|Popen|call|check_call|check_output)|
    child_process\.(?:spawn|execFile|fork)|
    execa\(|spawn\(|execFile\()""",
    re.VERBOSE,
)
_FILE_LIKE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".sh",
    ".json",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".html",
    ".csv",
    ".xml",
    ".sql",
    ".log",
    ".cfg",
    ".ini",
    ".css",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
}


def _extract_resources(
    source_line: str,
    detail: str,
    capability: str,
) -> list[tuple[str, str, bool, str, str]]:
    resources: list[tuple[str, str, bool, str, str]] = []
    seen: set[tuple[str, str]] = set()
    line_text = source_line.strip()
    detail_text = detail.strip()
    texts = [text for text in (line_text, detail_text) if text]
    if not texts:
        return _abstract_resources_for_capability(capability)

    for text in texts:
        for value in _URL_RE.findall(text):
            _append_resource(
                resources,
                seen,
                "url",
                value.rstrip(".,);"),
                True,
                "literal",
                "url_literal",
            )

        for regex in (_ENV_CALL_RE, _JS_ENV_RE, _BASH_ENV_RE):
            for value in regex.findall(text):
                _append_resource(
                    resources,
                    seen,
                    "env_var",
                    value,
                    True,
                    "env",
                    "environment_variable",
                )

        for value in _QUOTED_PATH_RE.findall(text):
            _append_resource(
                resources,
                seen,
                "path",
                value,
                True,
                _path_origin_kind(value),
                value,
            )

        for value in _QUOTED_FILE_RE.findall(text):
            _append_resource(
                resources,
                seen,
                "path",
                value,
                True,
                _path_origin_kind(value),
                value,
            )

        for value in _REDIRECT_RE.findall(text):
            _append_resource(
                resources,
                seen,
                "path",
                value,
                True,
                "artifact",
                "shell_redirect",
            )

        for value in _extract_domains(text):
            _append_resource(
                resources,
                seen,
                "domain",
                value,
                True,
                "literal",
                "domain_literal",
            )

        if capability == "dynamic_import":
            for resource_type, value in _extract_import_targets(text):
                _append_resource(
                    resources,
                    seen,
                    resource_type,
                    value,
                    True,
                    _import_origin_kind(resource_type, value),
                    value,
                )

    if resources:
        return resources
    return _abstract_resources_for_capability(capability)


def _append_resource(
    resources: list[tuple[str, str, bool, str, str]],
    seen: set[tuple[str, str]],
    resource_type: str,
    value: str,
    resolved: bool,
    origin_kind: str,
    origin_hint: str,
) -> None:
    item = (resource_type, value)
    if item in seen:
        return
    seen.add(item)
    resources.append((resource_type, value, resolved, origin_kind, origin_hint))


def _extract_domains(text: str) -> list[str]:
    domains: list[str] = []
    seen: set[str] = set()

    for value in _QUOTED_DOMAIN_RE.findall(text):
        if _looks_like_file_like_token(value):
            continue
        if value not in seen:
            seen.add(value)
            domains.append(value)

    return domains


def _extract_import_targets(text: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for value in _IMPORT_LITERAL_RE.findall(text):
        _append_import_target(targets, seen, value)
    for value in _SPEC_FROM_FILE_RE.findall(text):
        _append_import_target(targets, seen, value)
    for value in _PACKAGE_INSTALL_CMD_RE.findall(text):
        _append_import_target(targets, seen, value)

    return targets


def _append_import_target(
    targets: list[tuple[str, str]],
    seen: set[tuple[str, str]],
    value: str,
) -> None:
    normalized = value.strip()
    if not normalized:
        return
    resource_type = "path" if _looks_like_import_path(normalized) else "module"
    item = (resource_type, normalized)
    if item in seen:
        return
    seen.add(item)
    targets.append(item)


def _looks_like_import_path(value: str) -> bool:
    return value.startswith(("./", "../", "/")) or (
        PurePosixPath(value).suffix in _FILE_LIKE_SUFFIXES
    )


def _looks_like_file_like_token(value: str) -> bool:
    return PurePosixPath(value.lower()).suffix in _FILE_LIKE_SUFFIXES


def _abstract_resources_for_capability(capability: str) -> list[tuple[str, str, bool, str, str]]:
    if capability in {
        "http_request",
        "websocket",
        "smtp_send",
        "ssh_connect",
        "data_encode_send",
    }:
        return [("url", "<dynamic-url>", False, "unknown", "dynamic_url")]
    if capability in {"file_read", "file_write", "token_file_read"}:
        return [("path", "<dynamic-path>", False, "unknown", "dynamic_path")]
    if capability == "env_var_read":
        return [("env_var", "<dynamic-env-var>", False, "unknown", "dynamic_env_var")]
    if capability in {"shell_exec", "subprocess_spawn"}:
        return [("command", "<dynamic-command>", False, "unknown", "dynamic_command")]
    if capability == "sql_exec":
        return [("query", "<dynamic-query>", False, "unknown", "dynamic_query")]
    return []


def _refine_capability(capability: str, source_line: str, detail: str) -> str:
    if capability != "shell_exec":
        return capability
    text = f"{source_line}\n{detail}".lower()
    if _SUBPROCESS_RE.search(text):
        return "subprocess_spawn"
    return capability


def _path_origin_kind(value: str) -> str:
    normalized = value.replace("\\", "/").lower()
    if "/config/" in normalized or normalized.startswith("config/") or normalized.startswith("./config/"):
        return "config_file"
    return "literal"


def _import_origin_kind(resource_type: str, value: str) -> str:
    if resource_type == "path":
        return _path_origin_kind(value)
    return "literal"
