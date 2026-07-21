"""Extract document references and build rooted reference closure."""

from __future__ import annotations

import logging
import re
from pathlib import Path, PurePosixPath

from skillrecon.core.enums import FileKind, ReferenceResolutionStatus, ReferenceType
from skillrecon.core.types import DocumentNode, DocumentReference, FileEntry, ReferenceLink

logger = logging.getLogger(__name__)

_BACKTICK_PATH = re.compile(r"`([^`]+\.\w+)`")
_MD_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")

_TEXT_KINDS = {FileKind.MARKDOWN, FileKind.JSON}


def extract_references(
    content: str,
    doc_id: str,
    source_relative_path: str | None = None,
) -> list[ReferenceLink]:
    """Extract local document references from markdown text."""
    refs: list[ReferenceLink] = []
    seen_paths: set[str] = set()

    for match in _BACKTICK_PATH.finditer(content):
        raw_path = match.group(1)
        if _looks_like_endpoint_path(raw_path):
            continue
        normalized = _normalize_path(raw_path, source_relative_path)
        if normalized and normalized not in seen_paths and _looks_like_file_path(normalized):
            seen_paths.add(normalized)
            refs.append(
                ReferenceLink(
                    source_doc_id=doc_id,
                    target_path=normalized,
                    reference_type=ReferenceType.INLINE_PATH,
                    source_span=match.group(0),
                )
            )

    for match in _MD_LINK.finditer(content):
        raw_path = match.group(2)
        if raw_path.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if _looks_like_endpoint_path(raw_path):
            continue
        normalized = _normalize_path(raw_path, source_relative_path)
        if normalized and normalized not in seen_paths and _looks_like_file_path(normalized):
            seen_paths.add(normalized)
            refs.append(
                ReferenceLink(
                    source_doc_id=doc_id,
                    target_path=normalized,
                    reference_type=ReferenceType.EXPLICIT_LINK,
                    source_span=match.group(0),
                )
            )

    return refs


def _normalize_path(
    raw_path: str,
    source_relative_path: str | None = None,
) -> str | None:
    """Normalize a local reference into a repository-relative POSIX path."""
    path_str = _strip_link_suffix(raw_path.strip())
    if not path_str:
        return None

    path_str = path_str.replace("\\", "/")

    if _has_path_traversal(path_str):
        return None

    if len(path_str) > 2 and path_str[1] == ":":
        parts = PurePosixPath(path_str).parts
        for i, part in enumerate(parts):
            if part in ("scripts", "references", "templates", "config"):
                return str(PurePosixPath(*parts[i:]))
        return None

    if path_str.startswith("./"):
        path_str = path_str[2:]

    if source_relative_path is not None and path_str and not path_str.startswith("/"):
        base_dir = PurePosixPath(source_relative_path).parent
        if str(base_dir) != ".":
            path_str = str(base_dir / path_str)

    return str(PurePosixPath(path_str)) if path_str else None


def _strip_link_suffix(path_str: str) -> str:
    """Drop anchor/query suffixes from a local markdown reference."""
    stripped = path_str
    for delimiter in ("#", "?"):
        if delimiter in stripped:
            stripped = stripped.split(delimiter, 1)[0]
    if stripped.startswith("<") and stripped.endswith(">"):
        stripped = stripped[1:-1]
    return stripped.strip()


def _has_path_traversal(path_str: str) -> bool:
    """Reject references that would escape the skill package root."""
    return any(part == ".." for part in PurePosixPath(path_str).parts)


_KNOWN_EXTENSIONS = {
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".py", ".js", ".ts", ".mjs", ".sh", ".bash",
    ".html", ".css", ".xml", ".csv",
}


def _looks_like_file_path(path: str) -> bool:
    """Return whether a candidate looks like a local file reference."""
    if any(ch.isspace() for ch in path):
        return False
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in _KNOWN_EXTENSIONS


def _looks_like_endpoint_path(raw_path: str) -> bool:
    """Reject obvious API endpoint patterns that are not package documents."""
    path = raw_path.strip()
    return path.startswith("/") and ("{" in path or "}" in path)


def build_document_closure(
    skill_path: Path,
    files: list[FileEntry],
) -> tuple[list[DocumentNode], list[DocumentReference]]:
    """Build the rooted SKILL.md document closure and its reference edges."""
    file_map: dict[str, FileEntry] = {f.relative_path: f for f in files}

    root_entry = file_map.get("SKILL.md")
    if root_entry is None:
        root_candidates = [
            entry
            for entry in files
            if PurePosixPath(entry.relative_path).name.lower() == "skill.md"
            and PurePosixPath(entry.relative_path).parent == PurePosixPath(".")
        ]
        if root_candidates:
            root_entry = root_candidates[0]
    if root_entry is None:
        raise FileNotFoundError("SKILL.md not found in skill package")

    documents: list[DocumentNode] = []
    references: list[DocumentReference] = []
    visited: set[str] = set()
    queued: set[str] = {root_entry.relative_path}

    queue: list[tuple[FileEntry, int, str | None, ReferenceType | None]] = [
        (root_entry, 0, None, None),
    ]
    doc_idx = 0

    while queue:
        entry, depth, parent_id, ref_type = queue.pop(0)
        queued.discard(entry.relative_path)
        if entry.relative_path in visited:
            continue
        visited.add(entry.relative_path)

        doc_id = f"d{doc_idx}"
        doc_idx += 1

        documents.append(
            DocumentNode(
                doc_id=doc_id,
                file_id=entry.file_id,
                depth=depth,
                parent_doc_id=parent_id,
                reference_type=ref_type,
            )
        )

        if entry.kind not in _TEXT_KINDS:
            continue

        file_path = skill_path / entry.relative_path
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Cannot read file: %s", file_path)
            continue

        refs = extract_references(content, doc_id, entry.relative_path)
        for ref in refs:
            target = file_map.get(ref.target_path)
            target_file_id = target.file_id if target is not None else None

            if target is None:
                status = ReferenceResolutionStatus.UNRESOLVED
            elif target.kind in _TEXT_KINDS:
                status = ReferenceResolutionStatus.ADMITTED
                if target.relative_path not in visited and target.relative_path not in queued:
                    queue.append((target, depth + 1, doc_id, ref.reference_type))
                    queued.add(target.relative_path)
            else:
                status = ReferenceResolutionStatus.RESOLVED

            references.append(
                DocumentReference(
                    reference_id=f"ref{len(references)}",
                    source_doc_id=doc_id,
                    source_span=ref.source_span,
                    target_path=ref.target_path,
                    reference_type=ref.reference_type,
                    depth=depth + 1,
                    target_file_id=target_file_id,
                    resolution_status=status,
                )
            )

    doc_id_by_file_id = {doc.file_id: doc.doc_id for doc in documents}
    references = [
        ref.model_copy(
            update={
                "target_doc_id": (
                    doc_id_by_file_id.get(ref.target_file_id)
                    if ref.target_file_id is not None
                    else None
                )
            }
        )
        for ref in references
    ]

    unresolved_count = sum(
        1
        for ref in references
        if ref.resolution_status == ReferenceResolutionStatus.UNRESOLVED
    )

    logger.info(
        "Document closure: %d admitted, %d unresolved",
        len(documents),
        unresolved_count,
    )
    return documents, references
