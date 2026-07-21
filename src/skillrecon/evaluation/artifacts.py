"""Shared helpers for paper experiment artifact completeness checks."""

from __future__ import annotations

import json
import time
from pathlib import Path


STATUS_ARTIFACT = "pipeline_status.json"
FULL_ARTIFACT_KIND = "full"
ABLATION_ARTIFACT_KIND = "ablation"

REQUIRED_ARTIFACTS = (
    "package_manifest.json",
    "document_pack.json",
    "step_table.json",
    "step_edges.json",
    "raw_clause_samples.json",
    "contract_table.json",
    "event_table.json",
    "resource_table.json",
    "bridge_table.json",
    "path_table.json",
    "orchestration_table.json",
    "judgment_table.json",
    "certificate_table.json",
    "reconciliation_edges.json",
    "findings.json",
    "diagnostics.json",
    "exposures.json",
    "witnesses.json",
    "rejected_witnesses.json",
    "witness_validation.json",
    "permission_manifest.json",
    "g_d.json",
    "g_c.json",
    "g_x.json",
    STATUS_ARTIFACT,
)

POST_BEHAVIOR_RECOVERY_INPUT_ARTIFACTS = (
    "package_manifest.json",
    "document_pack.json",
    "contract_table.json",
    "event_table.json",
    "resource_table.json",
    "bridge_table.json",
    "path_table.json",
    "orchestration_table.json",
)


def missing_required_artifacts(
    artifact_dir: Path,
    *,
    required_artifacts: tuple[str, ...] = REQUIRED_ARTIFACTS,
    expected_status_kind: str | None = None,
) -> list[str]:
    """Return required artifact filenames absent from one skill artifact dir."""
    missing: list[str] = []
    for relative in required_artifacts:
        path = artifact_dir / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if relative == STATUS_ARTIFACT and not status_artifact_valid(
            artifact_dir,
            expected_artifact_kind=expected_status_kind,
        ):
            missing.append(relative)
    return missing


def post_behavior_recovery_missing_inputs(artifact_dir: Path) -> list[str]:
    """Return missing inputs required to resume after behavior extraction."""
    return [
        name
        for name in POST_BEHAVIOR_RECOVERY_INPUT_ARTIFACTS
        if not (artifact_dir / name).is_file()
    ]


def post_behavior_recovery_ready(artifact_dir: Path) -> bool:
    """Return whether one partial artifact dir can resume post-behavior stages."""
    return not post_behavior_recovery_missing_inputs(artifact_dir)


def artifact_complete(
    artifact_dir: Path,
    *,
    required_artifacts: tuple[str, ...] = REQUIRED_ARTIFACTS,
    expected_status_kind: str | None = None,
) -> bool:
    """Return whether one skill artifact dir contains all required outputs."""
    return not missing_required_artifacts(
        artifact_dir,
        required_artifacts=required_artifacts,
        expected_status_kind=expected_status_kind,
    )


def artifact_coverage(
    *,
    skill_ids: list[str],
    artifact_root: Path,
    required_artifacts: tuple[str, ...] = REQUIRED_ARTIFACTS,
    expected_status_kind: str | None = None,
) -> dict[str, object]:
    """Summarize per-skill artifact completeness for one artifact root."""
    missing_dirs: list[str] = []
    incomplete: dict[str, list[str]] = {}
    for skill_id in skill_ids:
        artifact_dir = artifact_root / skill_id
        if not artifact_dir.exists():
            missing_dirs.append(skill_id)
            continue
        missing = missing_required_artifacts(
            artifact_dir,
            required_artifacts=required_artifacts,
            expected_status_kind=expected_status_kind,
        )
        if missing:
            incomplete[skill_id] = missing
    complete_count = len(skill_ids) - len(missing_dirs) - len(incomplete)
    return {
        "root": str(artifact_root),
        "expected": len(skill_ids),
        "complete": complete_count,
        "missing_dirs": missing_dirs,
        "incomplete": incomplete,
    }


def write_status_artifact(
    artifact_dir: Path,
    *,
    skill_id: str,
    artifact_kind: str,
    system_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Write the success marker used by batch runners and experiment gates."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "status": "ok",
        "skill_id": skill_id,
        "artifact_kind": artifact_kind,
        "completed_at_unix": time.time(),
    }
    if system_id is not None:
        payload["system_id"] = system_id
    if metadata:
        payload["metadata"] = metadata
    (artifact_dir / STATUS_ARTIFACT).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def remove_status_artifact(artifact_dir: Path) -> None:
    """Remove a stale success marker before rebuilding artifacts."""
    path = artifact_dir / STATUS_ARTIFACT
    if path.exists() or path.is_symlink():
        path.unlink()


def status_artifact_valid(
    artifact_dir: Path,
    *,
    expected_artifact_kind: str | None = None,
) -> bool:
    """Return whether the success marker is present and matches the expected kind."""
    path = artifact_dir / STATUS_ARTIFACT
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("status") != "ok":
        return False
    if not isinstance(payload.get("skill_id"), str) or not payload["skill_id"]:
        return False
    artifact_kind = payload.get("artifact_kind")
    if not isinstance(artifact_kind, str) or not artifact_kind:
        return False
    if expected_artifact_kind is not None and artifact_kind != expected_artifact_kind:
        return False
    return True


def load_skill_ids(path: Path) -> list[str]:
    """Load and validate newline-delimited skill ids."""
    skill_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    duplicates = sorted(
        skill_id
        for skill_id in set(skill_ids)
        if skill_ids.count(skill_id) > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate skill ids in {path}: {duplicates[:5]}")
    return skill_ids
