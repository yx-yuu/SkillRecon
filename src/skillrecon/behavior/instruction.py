"""Instruction-level behavior recovery from markdown skill documents."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from skillrecon.behavior.normalize_types import RawObservation
from skillrecon.core.types import EvidenceSpan, PackageLink, PackageManifest
from skillrecon.loader.inline import has_binding_keyword

_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)\n]+)\)")
_URL_RE = re.compile(r"https?://[^\s)`]+")
_ROUTE_RE = re.compile(r"(?<![A-Za-z0-9._-])(/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+)")
_READ_CUE_RE = re.compile(r"(?i)查阅|读取|\bread\b|\breview\b|\bconsult\b")
_RUN_CUE_RE = re.compile(r"(?i)执行|运行|调用|\brun\b|\bexecute\b|\binvoke\b")
_ENV_VAR_ASSIGN_RE = re.compile(r"\b(?:export\s+)?([A-Z][A-Z0-9_]{2,})\s*=")
_API_KEY_CUE_RE = re.compile(r"(?i)\bapi[- ]?key\b|your-api-key")
_CREDENTIAL_CUE_RE = re.compile(r"(?i)\bsecret\b|credential|secret management|environment variables?")
_UPLOAD_CALL_RE = re.compile(r"(?i)\bupload\s*\(")
_INSTALL_CUE_RE = re.compile(
    r"(?i)\bpip\s+install\b|\buv\s+pip\s+install\b|\buv\s+add\b|\buv\s+sync\b|"
    r"\bnpm\s+(?:install|ci)\b|\bpnpm\s+(?:install|add)\b|\bpoetry\s+(?:install|add)\b"
)
_HTTP_UI_CUE_RE = re.compile(
    r"(?i)点击|打开|访问|导航|搜索|筛选|过滤|选择|添加|移除|浏览|重排|"
    r"\bclick\b|\bopen\b|\bnavigate\b|\bsearch\b|\bfilter\b|\bselect\b|\badd\b|"
    r"\bremove\b|\bbrowse\b|\breorder\b"
)
_SCREENSHOT_CUE_RE = re.compile(r"(?i)截图|快照|\bscreenshot\b|\bsnapshot\b")
_PATH_HINT_RE = re.compile(
    r"(?i)(?:^|/)(?:references|templates|scripts|docs?|config)/|"
    r"\.(?:md|py|js|ts|sh|json|ya?ml|txt|csv|xml|html|png|jpg|jpeg|pdf)$"
)
def extract_instruction_observations(
    skill_path: Path,
    manifest: PackageManifest,
) -> list[RawObservation]:
    """Recover instruction-level behavior hints from admitted markdown docs."""
    doc_paths = _doc_path_map(manifest)
    observations: list[RawObservation] = []
    seen: set[tuple[str, int, str, str]] = set()

    for link in manifest.links:
        evidence = resolve_instruction_evidence(skill_path, manifest, link)
        if evidence is None:
            continue
        relative_path = doc_paths.get(link.source_doc_id)
        if relative_path is None:
            continue
        line = _line_number_for_offset(skill_path / relative_path, evidence.start_offset)
        cap_type, detail, resource_hint = _infer_link_instruction(link, evidence.text)
        if cap_type is None:
            continue
        unit_id = link.target_unit_id or _doc_unit_id(link.source_doc_id)
        observation = _make_observation(
            cap_type=cap_type,
            detail=detail,
            relative_path=relative_path,
            line=line,
            unit_id=unit_id,
            source_text=evidence.text,
            resource_hint=resource_hint,
        )
        _append_unique(observations, seen, observation)

    for doc_id, relative_path in sorted(doc_paths.items()):
        content = (skill_path / relative_path).read_text(encoding="utf-8")
        in_frontmatter = False
        for line_no, raw_line in enumerate(content.splitlines(), start=1):
            stripped = raw_line.strip()
            if line_no == 1 and stripped == "---":
                in_frontmatter = True
                continue
            if in_frontmatter:
                if stripped == "---":
                    in_frontmatter = False
                continue
            if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                continue
            for observation in _infer_line_observations(
                doc_id=doc_id,
                relative_path=relative_path,
                line=line_no,
                text=stripped,
            ):
                _append_unique(observations, seen, observation)

    return observations


def resolve_instruction_evidence(
    skill_path: Path,
    manifest: PackageManifest,
    link: PackageLink,
) -> EvidenceSpan | None:
    """Resolve the local line or table context around a package link."""
    doc_to_path = _doc_path_map(manifest)
    relative_path = doc_to_path.get(link.source_doc_id)
    if relative_path is None:
        return None

    doc_path = skill_path / relative_path
    if not doc_path.is_file():
        return None

    content = doc_path.read_text(encoding="utf-8")
    start = content.find(link.source_span)
    if start < 0:
        return None
    line_start = content.rfind("\n", 0, start) + 1
    line_end = content.find("\n", start)
    if line_end < 0:
        line_end = len(content)
    text = content[line_start:line_end].strip()
    evidence = _table_instruction_evidence(content, start)
    if evidence is not None:
        return EvidenceSpan(
            doc_id=link.source_doc_id,
            start_offset=evidence[0],
            end_offset=evidence[1],
            text=evidence[2],
        )
    if not text or not has_binding_keyword(text):
        return None
    return EvidenceSpan(
        doc_id=link.source_doc_id,
        start_offset=line_start,
        end_offset=line_end,
        text=text,
    )


def _infer_link_instruction(
    link: PackageLink,
    evidence_text: str,
) -> tuple[str | None, str, tuple[str, str, bool, str, str] | None]:
    target_path = link.target_path or ""
    if _INSTALL_CUE_RE.search(evidence_text):
        detail = target_path or evidence_text
        return "dynamic_import", detail, (
            "command",
            detail,
            True,
            "user_input",
            "instruction_command",
        )
    if _READ_CUE_RE.search(evidence_text):
        detail = target_path or link.source_span
        resource_hint = (
            ("path", target_path, True, _path_origin_kind(target_path), target_path)
            if target_path
            else None
        )
        return "file_read", detail, resource_hint
    if _RUN_CUE_RE.search(evidence_text):
        detail = target_path or link.source_span
        resource_hint = ("command", detail, True, "user_input", "instruction_command")
        return "shell_exec", detail, resource_hint
    return None, "", None


def _infer_line_observations(
    *,
    doc_id: str,
    relative_path: str,
    line: int,
    text: str,
) -> list[RawObservation]:
    observations: list[RawObservation] = []
    unit_id = _doc_unit_id(doc_id)

    if _INSTALL_CUE_RE.search(text):
        detail = _primary_code_span(text) or text
        observations.append(
            _make_observation(
                cap_type="dynamic_import",
                detail=detail,
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=("command", detail, True, "user_input", "instruction_command"),
            )
        )

    env_var_hint = _env_var_resource_hint(text)
    if env_var_hint is not None:
        observations.append(
            _make_observation(
                cap_type="env_var_read",
                detail=env_var_hint[1],
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=env_var_hint,
            )
        )

    if _API_KEY_CUE_RE.search(text):
        detail = _primary_code_span(text) or _first_quoted_literal(text) or "api key"
        observations.append(
            _make_observation(
                cap_type="api_key_use",
                detail=detail,
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=("secret", detail, False, "user_input", "instruction_secret"),
            )
        )

    if _CREDENTIAL_CUE_RE.search(text) and (
        "secret management" in text.lower() or "securely" in text.lower()
    ):
        detail = _primary_code_span(text) or "credential storage"
        observations.append(
            _make_observation(
                cap_type="credential_store_access",
                detail=detail,
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=("secret", detail, False, "user_input", "instruction_secret"),
            )
        )

    upload_hint = _upload_resource_hint(text)
    if upload_hint is not None:
        observations.append(
            _make_observation(
                cap_type="data_upload",
                detail=upload_hint[1],
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=upload_hint,
            )
        )

    if _looks_like_command(text) or (_RUN_CUE_RE.search(text) and _contains_command_token(text)):
        detail = _primary_code_span(text) or text
        observations.append(
            _make_observation(
                cap_type="shell_exec",
                detail=detail,
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=("command", detail, True, "user_input", "instruction_command"),
            )
        )

    if _SCREENSHOT_CUE_RE.search(text):
        observations.append(
            _make_observation(
                cap_type="screenshot_capture",
                detail="snapshot",
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
            )
        )

    path_hint = _path_resource_hint(text)
    if _READ_CUE_RE.search(text) and path_hint is not None:
        observations.append(
            _make_observation(
                cap_type="file_read",
                detail=path_hint[1],
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=path_hint,
            )
        )

    url_hint = _url_resource_hint(text)
    if url_hint is not None or (_HTTP_UI_CUE_RE.search(text) and _looks_like_web_action(text)):
        observations.append(
            _make_observation(
                cap_type="http_request",
                detail=(url_hint[1] if url_hint is not None else text),
                relative_path=relative_path,
                line=line,
                unit_id=unit_id,
                source_text=text,
                resource_hint=url_hint or _route_resource_hint(text),
            )
        )

    return observations


def _make_observation(
    *,
    cap_type: str,
    detail: str,
    relative_path: str,
    line: int,
    unit_id: str,
    source_text: str,
    resource_hint: tuple[str, str, bool, str, str] | None = None,
) -> RawObservation:
    fields = {
        "capType": cap_type,
        "detail": detail,
        "tier": "instruction",
    }
    if resource_hint is not None:
        fields["resourceType"] = resource_hint[0]
        fields["resourceValue"] = resource_hint[1]
        fields["resourceResolved"] = str(resource_hint[2]).lower()
        fields["resourceOriginKind"] = resource_hint[3]
        fields["resourceOriginHint"] = resource_hint[4]
    return RawObservation(
        language="markdown",
        relative_path=relative_path,
        line=line,
        message=f"capType={cap_type} | detail={detail} | tier=instruction",
        fields=fields,
        unit_id=unit_id,
        source_text=source_text,
    )


def _append_unique(
    observations: list[RawObservation],
    seen: set[tuple[str, int, str, str]],
    observation: RawObservation,
) -> None:
    key = (
        observation.relative_path,
        observation.line,
        observation.fields.get("capType", ""),
        observation.fields.get("detail", ""),
    )
    if key in seen:
        return
    seen.add(key)
    observations.append(observation)


def _doc_path_map(manifest: PackageManifest) -> dict[str, str]:
    file_by_id = {entry.file_id: entry.relative_path for entry in manifest.files}
    return {
        doc.doc_id: file_by_id[doc.file_id]
        for doc in manifest.documents
        if doc.file_id in file_by_id
    }


def _doc_unit_id(doc_id: str) -> str:
    return f"doc::{doc_id}"


def _looks_like_command(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(("1.", "2.", "3.", "4.", "5.", "-", "*")):
        stripped = stripped.lstrip("1234567890.-* ").strip()
    tokens = stripped.split()
    if len(tokens) < 2:
        return False

    cmd = tokens[0].lower()
    next_token = tokens[1].lower()

    if cmd in {"uv"}:
        return next_token in {"run", "venv", "pip", "add", "sync"}
    if cmd in {"npm", "pnpm"}:
        return next_token in {"install", "ci", "add", "run"}
    if cmd == "poetry":
        return next_token in {"run", "install", "add"}
    if cmd in {"pip", "pip3"}:
        return next_token == "install"
    if cmd in {"python", "python3"}:
        return next_token == "-m" or next_token.endswith((".py", ".pyw"))
    if cmd in {"node", "npx"}:
        return next_token.endswith((".js", ".mjs", ".cjs")) or next_token in {
            "run",
            "install",
        }
    if cmd in {"bash", "sh"}:
        return next_token == "-c" or next_token.endswith((".sh", ".bash"))
    return False


def _contains_command_token(text: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:uv|python3?|pip3?|npm|pnpm|poetry|node|npx|bash|sh)\b", text)
    )


def _looks_like_web_action(text: str) -> bool:
    return bool(_ROUTE_RE.search(text) or _URL_RE.search(text) or "shop" in text.lower())


def _primary_code_span(text: str) -> str | None:
    for span in _CODE_SPAN_RE.findall(text):
        if span:
            return span.strip()
    return None


def _path_resource_hint(text: str) -> tuple[str, str, bool, str, str] | None:
    for target in _MARKDOWN_LINK_RE.findall(text):
        if _looks_like_path_like_value(target):
            return "path", target, True, _path_origin_kind(target), target
    for span in _CODE_SPAN_RE.findall(text):
        if _looks_like_path_like_value(span):
            return "path", span, True, _path_origin_kind(span), span
    return _route_resource_hint(text)


def _route_resource_hint(text: str) -> tuple[str, str, bool, str, str] | None:
    match = _ROUTE_RE.search(text)
    if match is None:
        return None
    return "path", match.group(1), True, "literal", match.group(1)


def _url_resource_hint(text: str) -> tuple[str, str, bool, str, str] | None:
    match = _URL_RE.search(text)
    if match is None:
        return None
    return "url", match.group(0), True, "literal", match.group(0)


def _env_var_resource_hint(text: str) -> tuple[str, str, bool, str, str] | None:
    match = _ENV_VAR_ASSIGN_RE.search(text)
    if match is None:
        return None
    return "env_var", match.group(1), True, "user_input", "instruction_env_assignment"


def _upload_resource_hint(text: str) -> tuple[str, str, bool, str, str] | None:
    if not _UPLOAD_CALL_RE.search(text):
        return None
    path_hint = _path_resource_hint(text)
    if path_hint is not None:
        return path_hint
    url_hint = _url_resource_hint(text)
    if url_hint is not None:
        return url_hint
    detail = _first_quoted_literal(text)
    if detail is None:
        return None
    if detail.startswith(("http://", "https://")):
        return "url", detail, True, "literal", detail
    return "path", detail, True, _path_origin_kind(detail), detail


def _path_origin_kind(value: str) -> str:
    normalized = value.replace("\\", "/").lower()
    if "/config/" in normalized or normalized.startswith("config/") or normalized.startswith("./config/"):
        return "config_file"
    if normalized.startswith(("/tmp/", "./tmp/", "tmp/")):
        return "artifact"
    return "user_input" if "user" in normalized or "input" in normalized else "literal"


def _first_quoted_literal(text: str) -> str | None:
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    if match is None:
        return None
    return match.group(1).strip()


def _looks_like_path_like_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith(("/", "./", "../")):
        return True
    if _PATH_HINT_RE.search(stripped):
        return True
    suffix = PurePosixPath(stripped).suffix.lower()
    return bool(suffix)


def _line_number_for_offset(path: Path, offset: int) -> int:
    content = path.read_text(encoding="utf-8")
    return content.count("\n", 0, offset) + 1


def _table_instruction_evidence(
    content: str,
    match_offset: int,
) -> tuple[int, int, str] | None:
    lines = content.splitlines(keepends=True)
    current_line_index = _line_index_for_offset(lines, match_offset)
    if current_line_index is None:
        return None

    current_line = lines[current_line_index].strip()
    if not _looks_like_markdown_table_row(current_line):
        return None

    separator_index = _find_table_separator(lines, current_line_index)
    if separator_index is None or separator_index == 0:
        return None

    header_line = lines[separator_index - 1].strip()
    if not _looks_like_markdown_table_row(header_line):
        return None

    combined = f"{header_line} {current_line}".strip()
    if not has_binding_keyword(combined):
        return None

    start_offset = sum(len(line) for line in lines[: separator_index - 1])
    end_offset = sum(len(line) for line in lines[: current_line_index + 1])
    return start_offset, end_offset, combined


def _line_index_for_offset(lines: list[str], offset: int) -> int | None:
    cursor = 0
    for index, line in enumerate(lines):
        next_cursor = cursor + len(line)
        if cursor <= offset < next_cursor:
            return index
        cursor = next_cursor
    if offset == cursor and lines:
        return len(lines) - 1
    return None


def _find_table_separator(lines: list[str], current_line_index: int) -> int | None:
    for index in range(current_line_index - 1, -1, -1):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if not _looks_like_markdown_table_row(stripped):
            return None
        if _looks_like_markdown_table_separator(stripped):
            return index
    return None


def _looks_like_markdown_table_row(text: str) -> bool:
    return text.startswith("|") and text.endswith("|")


def _looks_like_markdown_table_separator(text: str) -> bool:
    if not _looks_like_markdown_table_row(text):
        return False
    body = text.strip("|").replace(" ", "")
    return bool(body) and all(ch == "-" or ch == ":" or ch == "|" for ch in body)
