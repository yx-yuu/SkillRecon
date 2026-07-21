"""Scan a skill package directory and classify files."""

from __future__ import annotations

import logging
from pathlib import Path

from skillrecon.core.enums import FileKind
from skillrecon.core.types import FileEntry

logger = logging.getLogger(__name__)

_SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".mypy_cache"}

_EXT_MAP: dict[str, FileKind] = {
    ".py": FileKind.PYTHON,
    ".js": FileKind.JAVASCRIPT,
    ".ts": FileKind.TYPESCRIPT,
    ".mjs": FileKind.JAVASCRIPT,
    ".sh": FileKind.BASH,
    ".bash": FileKind.BASH,
    ".md": FileKind.MARKDOWN,
    ".json": FileKind.JSON,
}

# Extension → canonical language string (for CodeUnit.language).
# Only includes code-bearing extensions; .md / .json intentionally excluded.
_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".sh": "bash",
    ".bash": "bash",
}


def scan_skill_directory(skill_path: Path) -> list[FileEntry]:
    """Scan a skill package and return classified file entries."""
    if not skill_path.is_dir():
        raise FileNotFoundError(f"Skill directory not found: {skill_path}")

    entries: list[FileEntry] = []
    idx = 0
    for item in sorted(skill_path.rglob("*")):
        if not item.is_file():
            continue
        if any(part in _SKIP_DIRS for part in item.parts):
            continue
        if item.suffix == ".pyc":
            continue

        relative = item.relative_to(skill_path)
        kind = _EXT_MAP.get(item.suffix.lower(), FileKind.OTHER)

        entries.append(
            FileEntry(
                file_id=f"f{idx}",
                relative_path=str(relative),
                kind=kind,
                size_bytes=item.stat().st_size,
            )
        )
        idx += 1

    logger.info("Scanned %d files in %s", len(entries), skill_path.name)
    return entries


def infer_language(file_entry: FileEntry) -> str | None:
    """Infer a canonical code language from a file entry."""
    suffix = Path(file_entry.relative_path).suffix.lower()
    return _LANG_MAP.get(suffix)
