from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from skillrecon.evaluation.artifacts import POST_BEHAVIOR_RECOVERY_INPUT_ARTIFACTS


SCRIPT_PATH = Path("scripts/manage_paper_artifact_job.py")


def _load_manager_module():
    spec = importlib.util.spec_from_file_location("manage_paper_artifact_job", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_start_command_passes_drive_map_to_artifact_runner(tmp_path: Path) -> None:
    manager = _load_manager_module()
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(tmp_path / "paper500"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--drive-map",
            "E=/mnt/e",
        ]
    )

    command = manager._build_artifact_command(args)

    assert "--drive-map" in command
    drive_map_index = command.index("--drive-map")
    assert command[drive_map_index + 1] == "E=/mnt/e"


def test_start_command_passes_source_restore_options_to_artifact_runner(
    tmp_path: Path,
) -> None:
    manager = _load_manager_module()
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(tmp_path / "paper500"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--restore-missing-sources",
            "--source-cache-root",
            str(tmp_path / "source-cache"),
            "--source-download-template",
            "https://download.example.invalid/{slug_url}.zip",
            "--source-repo-template",
            "https://example.invalid/{owner}/{slug}.git",
            "--source-restore-depth",
            "2",
            "--source-restore-timeout",
            "17",
        ]
    )

    command = manager._build_artifact_command(args)

    assert "--restore-missing-sources" in command
    assert command[command.index("--source-cache-root") + 1] == str(
        tmp_path / "source-cache"
    )
    assert command[command.index("--source-download-template") + 1] == (
        "https://download.example.invalid/{slug_url}.zip"
    )
    assert command[command.index("--source-repo-template") + 1] == (
        "https://example.invalid/{owner}/{slug}.git"
    )
    assert command[command.index("--source-restore-depth") + 1] == "2"
    assert command[command.index("--source-restore-timeout") + 1] == "17"


def test_start_blocks_when_pending_sources_are_not_visible(tmp_path: Path) -> None:
    manager = _load_manager_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    state_file = tmp_path / "state.json"
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "missing-dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--pid-file",
            str(tmp_path / "job.pid"),
            "--state-file",
            str(state_file),
        ]
    )

    payload = manager.start_job(args)
    state = json.loads(state_file.read_text(encoding="utf-8"))

    assert payload["status"] == "blocked_missing_sources"
    assert payload["running"] is False
    assert payload["source_preflight"]["needs_source"] == 3
    assert payload["source_preflight"]["available"] == 0
    assert payload["source_preflight"]["missing"] == 3
    assert state["status"] == "blocked_missing_sources"


def test_status_includes_last_blocked_source_preflight(tmp_path: Path) -> None:
    manager = _load_manager_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    state_file = tmp_path / "state.json"
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "missing-dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--pid-file",
            str(tmp_path / "job.pid"),
            "--state-file",
            str(state_file),
        ]
    )

    manager.start_job(args)
    status = manager.status_job(args)

    assert status["running"] is False
    assert status["last_status"] == "blocked_missing_sources"
    assert status["last_source_preflight"]["needs_source"] == 3
    assert status["last_source_preflight"]["missing"] == 3


def test_status_reports_post_behavior_partial_recovery_candidates(
    tmp_path: Path,
) -> None:
    manager = _load_manager_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    artifact_root = tmp_path / "artifacts"
    recoverable_dir = artifact_root / "owner-high" / "skill-high"
    for relative in POST_BEHAVIOR_RECOVERY_INPUT_ARTIFACTS:
        _write_json(recoverable_dir / relative, [])

    early_dir = artifact_root / "owner-medium" / "skill-medium"
    _write_json(
        early_dir / "document_pack.json",
        {"skill_id": "owner-medium/skill-medium", "admitted_docs": [], "doc_blocks": []},
    )
    args = manager.build_arg_parser().parse_args(
        [
            "status",
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--artifact-root",
            str(artifact_root),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--pid-file",
            str(tmp_path / "job.pid"),
            "--state-file",
            str(tmp_path / "state.json"),
        ]
    )

    status = manager.status_job(args)
    partial_recovery = status["partial_recovery"]

    assert partial_recovery["recoverable"] == 1
    assert partial_recovery["not_recoverable"] == 1
    assert partial_recovery["recoverable_examples"][0]["skill_id"] == (
        "owner-high/skill-high"
    )
    assert partial_recovery["not_recoverable_examples"][0]["skill_id"] == (
        "owner-medium/skill-medium"
    )
    dry_run_command = " ".join(partial_recovery["dry_run_command"])
    assert "recover_partial_paper_artifact.py" in dry_run_command
    assert "--dry-run" in dry_run_command


def test_source_preflight_blocks_when_any_pending_source_is_missing(tmp_path: Path) -> None:
    manager = _load_manager_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    source_root = tmp_path / "external-e"
    (
        source_root
        / "clawhub_skills"
        / "artifacts"
        / "extracted_skills"
        / "owner-high"
        / "skill-high"
    ).mkdir(parents=True)
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "missing-dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--pid-file",
            str(tmp_path / "job.pid"),
            "--state-file",
            str(tmp_path / "state.json"),
            "--drive-map",
            f"E={source_root}",
        ]
    )

    preflight = manager._source_preflight(args)

    assert preflight["needs_source"] == 3
    assert preflight["available"] == 1
    assert preflight["missing"] == 2
    assert manager._source_preflight_blocks_start(preflight)


def test_source_preflight_accepts_all_visible_pending_sources(tmp_path: Path) -> None:
    manager = _load_manager_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    source_root = tmp_path / "external-e"
    for owner, slug in (
        ("owner-high", "skill-high"),
        ("owner-medium", "skill-medium"),
        ("owner-low", "skill-low"),
    ):
        (
            source_root
            / "clawhub_skills"
            / "artifacts"
            / "extracted_skills"
            / owner
            / slug
        ).mkdir(parents=True)
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "missing-dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--pid-file",
            str(tmp_path / "job.pid"),
            "--state-file",
            str(tmp_path / "state.json"),
            "--drive-map",
            f"E={source_root}",
        ]
    )

    preflight = manager._source_preflight(args)

    assert preflight["needs_source"] == 3
    assert preflight["available"] == 3
    assert preflight["missing"] == 0
    assert not manager._source_preflight_blocks_start(preflight)


def test_source_preflight_accepts_restorable_missing_sources(tmp_path: Path) -> None:
    manager = _load_manager_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    args = manager.build_arg_parser().parse_args(
        [
            "start",
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "missing-dataset"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--summary-out",
            str(tmp_path / "summary.json"),
            "--pid-file",
            str(tmp_path / "job.pid"),
            "--state-file",
            str(tmp_path / "state.json"),
            "--restore-missing-sources",
            "--source-cache-root",
            str(tmp_path / "source-cache"),
            "--source-download-template",
            "https://download.example.invalid/{slug_url}.zip",
            "--source-repo-template",
            "https://example.invalid/{owner}/{slug}.git",
        ]
    )

    preflight = manager._source_preflight(args)

    assert preflight["needs_source"] == 3
    assert preflight["available"] == 0
    assert preflight["restorable"] == 3
    assert preflight["missing"] == 0
    assert preflight["restore_missing_sources"] is True
    assert preflight["restorable_examples"][0]["download_urls"] == [
        "https://download.example.invalid/skill-high.zip"
    ]
    assert preflight["restorable_examples"][0]["repo_urls"] == [
        "https://example.invalid/owner-high/skill-high.git"
    ]
    assert not manager._source_preflight_blocks_start(preflight)


def _write_paper_dataset(tmp_path: Path) -> Path:
    paper_dataset = tmp_path / "paper500"
    specs = [
        ("high", "owner-high", "skill-high", "high_risk"),
        ("medium", "owner-medium", "skill-medium", "medium_risk"),
        ("low", "owner-low", "skill-low", "low_risk"),
    ]
    skill_ids: list[str] = []
    for slice_name, owner, slug, risk_tier in specs:
        skill_ids.append(f"{owner}/{slug}")
        slice_dir = paper_dataset / slice_name
        slice_dir.mkdir(parents=True)
        (slice_dir / "sample_index.jsonl").write_text(
            json.dumps(
                _sample_index_record(owner=owner, slug=slug, risk_tier=risk_tier),
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    (paper_dataset / "all_skills.txt").write_text(
        "".join(f"{skill_id}\n" for skill_id in skill_ids),
        encoding="utf-8",
    )
    return paper_dataset


def _sample_index_record(*, owner: str, slug: str, risk_tier: str) -> dict[str, object]:
    return {
        "dataset_bucket": "pure_py",
        "owner": owner,
        "slug": slug,
        "version": "1.0.0",
        "script_types": ["py"],
        "extract_root": f"E:/clawhub_skills/artifacts/extracted_skills/{owner}/{slug}",
        "risk_tier": risk_tier,
        "skill": {
            "owner": owner,
            "slug": slug,
            "display_name": slug,
            "security_scan": {
                "openclaw_status": "Benign",
                "virus_total_status": "Benign",
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
