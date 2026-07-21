"""Shared helpers for behavior normalization modules."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def _normalize_relpath(relative_path: str) -> str:
    return PurePosixPath(relative_path).as_posix()


def _read_source_line(staged_root: Path, relative_path: str, line: int) -> str:
    if line <= 0:
        return ""
    file_path = staged_root / relative_path
    if not file_path.is_file():
        return ""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if line > len(lines):
        return ""
    return lines[line - 1]
