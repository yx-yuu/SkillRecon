from __future__ import annotations

from pathlib import Path

import pytest

from skillrecon.evaluation.datasets import FlaggedSkillRecord
from skillrecon.evaluation.gold_builder import (
    GoldBuildConfig,
    SourceCollectionError,
    collect_source_bundle,
)


def test_collect_source_bundle_uses_wsl_drive_override(tmp_path: Path) -> None:
    source_dir = (
        tmp_path
        / "e-drive"
        / "clawhub_skills"
        / "artifacts"
        / "extracted_skills"
        / "owner"
        / "slug"
    )
    source_dir.mkdir(parents=True)
    (source_dir / "SKILL.md").write_text(
        "# Test Skill\n\nUse the approved API only.\n",
        encoding="utf-8",
    )
    (source_dir / "_manifest.json").write_text(
        '{"name":"slug"}\n',
        encoding="utf-8",
    )

    bundle = collect_source_bundle(
        _record(),
        dataset_root=tmp_path / "missing-dataset",
        build_config=GoldBuildConfig(max_source_files=4, max_file_chars=100),
        windows_drive_map={"E": tmp_path / "e-drive"},
    )

    assert bundle.skill_path == source_dir.as_posix()
    assert [excerpt.path for excerpt in bundle.excerpts] == [
        "SKILL.md",
        "_manifest.json",
    ]


def test_collect_source_bundle_fails_when_source_is_not_readable(tmp_path: Path) -> None:
    with pytest.raises(SourceCollectionError) as exc_info:
        collect_source_bundle(
            _record(),
            dataset_root=tmp_path / "missing-dataset",
        )

    message = str(exc_info.value)
    assert "Missing local source directory for owner/slug" in message
    assert "/mnt/e/clawhub_skills/artifacts/extracted_skills/owner/slug" in message
    assert "SKILLRECON_DRIVE_E_ROOT" in message


def _record() -> FlaggedSkillRecord:
    return FlaggedSkillRecord(
        dataset_bucket="pure_py",
        owner="owner",
        slug="slug",
        version="1.0.0",
        script_types=["py"],
        extract_root="/mnt/e/clawhub_skills/artifacts/extracted_skills/owner/slug",
        risk_tier="high_risk",
    )
