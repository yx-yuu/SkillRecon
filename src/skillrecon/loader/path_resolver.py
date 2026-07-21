"""Path normalization helpers for dataset index records.

The crawled corpus indexes were produced on Windows, while the reproducible
pipeline is usually run from Linux/WSL.  These helpers keep that conversion in
one place so dataset loaders do not need local path special cases.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):/?(.*)$")
_WSL_UNC_PREFIXES = ("//wsl.localhost/", "//wsl$/")
_DRIVE_ENV_PREFIX = "SKILLRECON_DRIVE_"
_DRIVE_ENV_SUFFIX = "_ROOT"


def normalize_dataset_path(
    value: str | os.PathLike[str] | None,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path | None:
    """Normalize a stored dataset path for the current WSL/Linux runtime.

    Supported inputs include:
    - ``E:\\...`` / ``E:/...`` Windows drive paths, mapped to ``/mnt/e/...`` by default.
    - ``/mnt/e/...`` WSL drive paths, remapped when ``SKILLRECON_DRIVE_E_ROOT`` is set.
    - ``\\\\wsl.localhost\\Ubuntu\\home\\...`` and ``//wsl.localhost/Ubuntu/home/...``.
    - ordinary POSIX paths, returned as ``Path`` values unchanged.

    Drive roots can be overridden by passing ``windows_drive_map`` or by setting
    environment variables such as ``SKILLRECON_DRIVE_E_ROOT=/path/to/e``.
    """
    if value is None:
        return None

    raw_value = os.fspath(value).strip()
    if not raw_value:
        return None

    slash_path = _to_slash_path(raw_value)
    unc_path = _normalize_wsl_unc_path(slash_path)
    if unc_path is not None:
        return unc_path

    windows_path = _normalize_windows_drive_path(
        slash_path,
        windows_drive_map=windows_drive_map,
    )
    if windows_path is not None:
        return windows_path

    wsl_drive_path = _normalize_wsl_drive_mount_path(
        slash_path,
        windows_drive_map=windows_drive_map,
    )
    if wsl_drive_path is not None:
        return wsl_drive_path

    return Path(raw_value)


def rewrite_dataset_path_string(
    value: str,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> str:
    """Rewrite one standalone Windows/WSL dataset path string to POSIX form."""
    if not is_rewritable_dataset_path(value, windows_drive_map=windows_drive_map):
        return value
    normalized = normalize_dataset_path(value, windows_drive_map=windows_drive_map)
    return value if normalized is None else normalized.as_posix()


def is_rewritable_dataset_path(
    value: str,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> bool:
    """Return whether a JSON string looks like a standalone path to rewrite."""
    if not value:
        return False
    slash_path = _to_slash_path(value.strip())
    if _normalize_wsl_unc_path(slash_path) is not None:
        return True
    return (
        _WINDOWS_DRIVE_RE.match(slash_path) is not None
        or _normalize_wsl_drive_mount_path(
            slash_path,
            windows_drive_map=windows_drive_map,
        ) is not None
    )


def parse_windows_drive_map(items: Sequence[str]) -> dict[str, str]:
    """Parse ``DRIVE=ROOT`` CLI values into normalized drive-root mappings."""
    drive_map: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid drive map value {item!r}; expected DRIVE=ROOT")
        drive, root = item.split("=", 1)
        drive = drive.strip().rstrip(":").upper()
        root = root.strip()
        if len(drive) != 1 or not drive.isalpha() or not root:
            raise ValueError(f"Invalid drive map value {item!r}; expected DRIVE=ROOT")
        drive_map[drive] = root
    return drive_map


def iter_skill_path_candidates(
    dataset_root: Path,
    owner: str,
    slug: str,
    *,
    extract_root: str | os.PathLike[str] | None = None,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Iterator[Path]:
    """Yield candidate local directories for one skill index record."""
    seen: set[str] = set()
    for candidate in (
        dataset_root / owner / slug,
        dataset_root / slug,
        normalize_dataset_path(extract_root, windows_drive_map=windows_drive_map),
    ):
        if candidate is None:
            continue
        key = candidate.as_posix()
        if key in seen:
            continue
        seen.add(key)
        yield candidate


def resolve_skill_path_from_index(
    dataset_root: Path,
    owner: str,
    slug: str,
    *,
    extract_root: str | os.PathLike[str] | None = None,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path:
    """Resolve a skill directory using local dataset roots before index paths."""
    candidates = list(
        iter_skill_path_candidates(
            dataset_root,
            owner,
            slug,
            extract_root=extract_root,
            windows_drive_map=windows_drive_map,
        )
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_wsl_unc_path(slash_path: str) -> Path | None:
    lower_path = slash_path.lower()
    for prefix in _WSL_UNC_PREFIXES:
        if not lower_path.startswith(prefix):
            continue
        remainder = slash_path[len(prefix):]
        parts = remainder.split("/", 1)
        if len(parts) == 1 or not parts[1]:
            return Path("/")
        return Path("/" + parts[1])
    return None


def _normalize_windows_drive_path(
    slash_path: str,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None,
) -> Path | None:
    match = _WINDOWS_DRIVE_RE.match(slash_path)
    if match is None:
        return None

    drive = match.group(1).upper()
    relative_path = match.group(2)
    drive_roots = _windows_drive_roots(windows_drive_map)
    root = drive_roots.get(drive, Path(f"/mnt/{drive.lower()}"))
    return root if not relative_path else root / relative_path


def _normalize_wsl_drive_mount_path(
    slash_path: str,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None,
) -> Path | None:
    parts = Path(slash_path).parts
    if len(parts) < 3 or parts[0] != "/" or parts[1] != "mnt":
        return None

    drive = parts[2].upper()
    if len(drive) != 1 or not drive.isalpha():
        return None

    drive_roots = _windows_drive_roots(windows_drive_map)
    root = drive_roots.get(drive)
    if root is None:
        return None
    return root.joinpath(*parts[3:])


def _windows_drive_roots(
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None,
) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for key, value in os.environ.items():
        if not key.startswith(_DRIVE_ENV_PREFIX) or not key.endswith(_DRIVE_ENV_SUFFIX):
            continue
        drive = key[len(_DRIVE_ENV_PREFIX):-len(_DRIVE_ENV_SUFFIX)].strip(":").upper()
        if len(drive) == 1 and drive.isalpha() and value:
            roots[drive] = Path(value)

    if windows_drive_map:
        for key, value in windows_drive_map.items():
            drive = str(key).strip().rstrip(":").upper()
            if len(drive) != 1 or not drive.isalpha():
                raise ValueError(f"Invalid Windows drive key: {key!r}")
            roots[drive] = Path(value)
    return roots


def _to_slash_path(value: str) -> str:
    if value.startswith("\\\\?\\"):
        value = value[4:]
    return value.replace("\\", "/")
