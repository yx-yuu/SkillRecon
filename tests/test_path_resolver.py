from __future__ import annotations

from pathlib import Path

from skillrecon.evaluation.datasets import FlaggedSkillRecord
from skillrecon.loader.path_resolver import (
    normalize_dataset_path,
    parse_windows_drive_map,
    rewrite_dataset_path_string,
)
from skillrecon.loader import test_index as test_index_loader


def test_normalize_windows_drive_path_defaults_to_wsl_mount() -> None:
    path = normalize_dataset_path(
        r"E:\clawhub_skills\artifacts\extracted_skills\owner\slug"
    )

    assert path == Path("/mnt/e/clawhub_skills/artifacts/extracted_skills/owner/slug")


def test_normalize_windows_drive_path_accepts_forward_slashes_and_overrides() -> None:
    path = normalize_dataset_path(
        "E:/clawhub_skills/artifacts/pages/owner/slug/detail.html",
        windows_drive_map={"E": "/datasets/e-drive"},
    )

    assert path == Path("/datasets/e-drive/clawhub_skills/artifacts/pages/owner/slug/detail.html")


def test_normalize_wsl_drive_mount_path_accepts_overrides() -> None:
    path = normalize_dataset_path(
        "/mnt/e/clawhub_skills/artifacts/extracted_skills/owner/slug",
        windows_drive_map={"E": "/datasets/e-drive"},
    )

    assert path == Path("/datasets/e-drive/clawhub_skills/artifacts/extracted_skills/owner/slug")


def test_normalize_wsl_unc_paths() -> None:
    assert normalize_dataset_path(
        r"\\wsl.localhost\Ubuntu\home\user\project\data"
    ) == Path("/home/user/project/data")
    assert normalize_dataset_path(
        "//wsl.localhost/Ubuntu/home/user/project/data"
    ) == Path("/home/user/project/data")


def test_normalize_extract_root_uses_shared_dataset_path_rules() -> None:
    path = test_index_loader.normalize_extract_root(
        r"E:\clawhub_skills\artifacts\extracted_skills\owner\slug",
        windows_drive_map={"E": "/mirror/e"},
    )

    assert path == Path("/mirror/e/clawhub_skills/artifacts/extracted_skills/owner/slug")


def test_rewrite_dataset_path_string_only_rewrites_standalone_paths() -> None:
    assert rewrite_dataset_path_string(
        r"E:\clawhub_skills\artifacts\zips\owner\slug\1.0.0.zip"
    ) == "/mnt/e/clawhub_skills/artifacts/zips/owner/slug/1.0.0.zip"
    assert rewrite_dataset_path_string("https://clawhub.ai/owner/slug") == (
        "https://clawhub.ai/owner/slug"
    )
    assert rewrite_dataset_path_string("Use E:\\clawhub_skills in docs") == (
        "Use E:\\clawhub_skills in docs"
    )


def test_rewrite_dataset_path_string_rewrites_wsl_drive_mount_with_override() -> None:
    assert rewrite_dataset_path_string(
        "/mnt/e/clawhub_skills/artifacts/extracted_skills/owner/slug",
        windows_drive_map={"E": "/datasets/e-drive"},
    ) == "/datasets/e-drive/clawhub_skills/artifacts/extracted_skills/owner/slug"


def test_parse_windows_drive_map_normalizes_cli_values() -> None:
    assert parse_windows_drive_map(["e:=/datasets/e-drive"]) == {
        "E": "/datasets/e-drive"
    }


def test_test_index_resolves_current_dataset_root_before_extract_root(tmp_path: Path) -> None:
    local_skill = tmp_path / "dataset" / "owner" / "slug"
    local_skill.mkdir(parents=True)
    entry = test_index_loader.TestIndexEntry(
        dataset_bucket="pure_py",
        owner="owner",
        slug="slug",
        version="1.0.0",
        script_types=("py",),
        extract_root=r"E:\clawhub_skills\artifacts\extracted_skills\owner\slug",
    )

    assert entry.resolve_skill_path(tmp_path / "dataset") == local_skill


def test_flagged_record_falls_back_to_normalized_extract_root(tmp_path: Path) -> None:
    extracted_root = (
        tmp_path
        / "external_drive"
        / "clawhub_skills"
        / "artifacts"
        / "extracted_skills"
        / "owner"
        / "slug"
    )
    extracted_root.mkdir(parents=True)
    record = FlaggedSkillRecord(
        dataset_bucket="pure_py",
        owner="owner",
        slug="slug",
        version="1.0.0",
        script_types=["py"],
        extract_root=r"E:\clawhub_skills\artifacts\extracted_skills\owner\slug",
    )

    assert record.resolve_skill_path(
        tmp_path / "missing_dataset",
        windows_drive_map={"E": tmp_path / "external_drive"},
    ) == extracted_root
