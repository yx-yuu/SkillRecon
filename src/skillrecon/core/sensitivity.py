"""Capability sensitivity helpers shared by reconciliation and witnessing."""

from __future__ import annotations

import re

from skillrecon.core.types import CapabilityEvent

_ENV_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
_SCRIPT_PATH_RE = re.compile(r"(?:^|\s)(?:\./|/)?[A-Za-z0-9_./-]+\.(?:py|js|ts|sh|bash)\b")
_INLINE_EXEC_RE = re.compile(r"\b(?:bash|sh)\s+-c\b|\bpython3?\s+-c\b|\bnode\s+-e\b")
_IMPORT_FILE_RE = re.compile(
    r"(?:^|[\s'\"`])(?:\./|\.\./|/)?[A-Za-z0-9_./-]+\.(?:py|js|ts|mjs|cjs|jsx|tsx|json)\b"
)
_PACKAGE_INSTALL_RE = re.compile(
    r"\b(?:pip(?:3)?\s+install|npm\s+install|pnpm\s+add|yarn\s+add)\b"
)
_MODULE_NAME_RE = re.compile(
    r"^@?[A-Za-z_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)*$"
)
_SECRET_ENV_TOKENS = (
    "ACCESS_KEY",
    "ACCESS_TOKEN",
    "APIKEY",
    "API_KEY",
    "AUTH_TOKEN",
    "BEARER",
    "CLIENT_SECRET",
    "COOKIE",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE",
    "SECRET",
    "SESSION",
    "TOKEN",
)


def event_capability_family(event: CapabilityEvent) -> str:
    """Return a family label that is more stable than the raw capability atom."""
    if event.capability == "env_var_read":
        return (
            "secret_env_var_read"
            if _looks_like_secret_env_access(event)
            else "config_env_var_read"
        )
    if event.capability in {"shell_exec", "subprocess_spawn"}:
        return (
            f"wrapper_{event.capability}"
            if _looks_like_wrapper_exec(event)
            else f"side_effect_{event.capability}"
        )
    if event.capability == "dynamic_import":
        if _looks_like_local_module_import(event):
            return "local_dynamic_import"
        if _looks_like_external_module_import(event):
            return "external_dynamic_import"
        return "dynamic_import_unknown"
    return event.capability


def event_requires_authorization(
    event: CapabilityEvent,
    a_req: set[str],
) -> bool:
    """Decide whether a concrete event should count as authorization-sensitive."""
    if event.capability not in a_req:
        return False
    if event.capability == "env_var_read":
        return _looks_like_secret_env_access(event)
    if event.capability in {"shell_exec", "subprocess_spawn"}:
        return not _looks_like_wrapper_exec(event)
    if event.capability == "dynamic_import":
        return not _looks_like_local_module_import(event)
    return True


def _looks_like_secret_env_access(event: CapabilityEvent) -> bool:
    for name in _candidate_env_names(event):
        upper = name.upper()
        if any(token in upper for token in _SECRET_ENV_TOKENS):
            return True
    return False


def _candidate_env_names(event: CapabilityEvent) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in [*event.arguments, event.api_call, event.detail]:
        if not value or value.startswith("<dynamic-"):
            continue
        for match in _ENV_NAME_RE.findall(value):
            if match in seen:
                continue
            seen.add(match)
            candidates.append(match)
    return candidates


def _looks_like_wrapper_exec(event: CapabilityEvent) -> bool:
    text = " ".join(
        value
        for value in [event.detail, event.api_call, *event.arguments]
        if value and not value.startswith("<dynamic-")
    ).lower()
    if not text or _INLINE_EXEC_RE.search(text):
        return False

    if text.startswith(("python ", "python3 ", "node ", "bash ", "sh ")):
        return bool(_SCRIPT_PATH_RE.search(text))
    return text.startswith(("uv run ", "poetry run ", "npm run ", "pnpm run ", "npx "))


def _looks_like_local_module_import(event: CapabilityEvent) -> bool:
    return any(
        _IMPORT_FILE_RE.search(value)
        for value in _dynamic_import_candidates(event)
    )


def _looks_like_external_module_import(event: CapabilityEvent) -> bool:
    for value in _dynamic_import_candidates(event):
        stripped = value.strip()
        if not stripped:
            continue
        if _PACKAGE_INSTALL_RE.search(stripped):
            return True
        if _IMPORT_FILE_RE.search(stripped):
            continue
        if _MODULE_NAME_RE.fullmatch(stripped):
            return True
    return False


def _dynamic_import_candidates(event: CapabilityEvent) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for value in [*event.arguments, event.api_call, event.detail]:
        if not value or value.startswith("<dynamic-"):
            continue
        normalized = value.strip().strip("\"'")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)
    return candidates
