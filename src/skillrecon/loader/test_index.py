"""Helpers for loading curated test-index samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Mapping
import os
from pathlib import Path

from skillrecon.loader.path_resolver import (
    normalize_dataset_path,
    resolve_skill_path_from_index,
)


@dataclass(frozen=True)
class TestIndexEntry:
    """One sample record from ``data/test_index/*.jsonl``."""

    dataset_bucket: str
    owner: str
    slug: str
    version: str
    script_types: tuple[str, ...]
    extract_root: str
    manifest_path: str | None = None

    @property
    def skill_ref(self) -> str:
        """Return the owner-qualified skill identifier."""
        return f"{self.owner}/{self.slug}"

    def resolve_skill_path(
        self,
        dataset_root: Path,
        *,
        windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> Path:
        """Resolve the local extracted skill directory."""
        return resolve_skill_path_from_index(
            dataset_root,
            self.owner,
            self.slug,
            extract_root=self.extract_root,
            windows_drive_map=windows_drive_map,
        )


def normalize_extract_root(
    extract_root: str,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path | None:
    """Convert stored extract-root strings into local workspace paths."""
    return normalize_dataset_path(extract_root, windows_drive_map=windows_drive_map)


def load_test_index(index_path: Path) -> list[TestIndexEntry]:
    """Load one JSONL sample index file."""
    entries: list[TestIndexEntry] = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            entries.append(
                TestIndexEntry(
                    dataset_bucket=str(payload["dataset_bucket"]),
                    owner=str(payload["owner"]),
                    slug=str(payload["slug"]),
                    version=str(payload["version"]),
                    script_types=tuple(str(item) for item in payload.get("script_types", [])),
                    extract_root=str(payload["extract_root"]),
                    manifest_path=(
                        str(payload["manifest_path"])
                        if payload.get("manifest_path") is not None
                        else None
                    ),
                )
            )
    return entries


def load_test_index_dir(index_dir: Path) -> list[TestIndexEntry]:
    """Load all JSONL sample index files under one directory."""
    entries: list[TestIndexEntry] = []
    for index_path in sorted(index_dir.glob("*.jsonl")):
        entries.extend(load_test_index(index_path))
    return entries
