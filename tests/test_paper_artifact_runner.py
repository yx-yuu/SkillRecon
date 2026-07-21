from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from skillrecon.evaluation.artifacts import (
    ABLATION_ARTIFACT_KIND,
    FULL_ARTIFACT_KIND,
    STATUS_ARTIFACT,
    write_status_artifact,
)


SCRIPT_PATH = Path("scripts/run_paper_artifacts.py")


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("run_paper_artifacts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_records_uses_all_skills_order_for_full_dataset(tmp_path: Path) -> None:
    runner = _load_runner_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    (paper_dataset / "all_skills.txt").write_text(
        "owner-low/skill-low\nowner-high/skill-high\nowner-medium/skill-medium\n",
        encoding="utf-8",
    )

    records = runner.select_records(
        paper_dataset=paper_dataset,
        skills_file=None,
        slice_names=("high", "medium", "low"),
        start=0,
        stop=None,
        limit=None,
    )

    assert [record.skill_id for record in records] == [
        "owner-low/skill-low",
        "owner-high/skill-high",
        "owner-medium/skill-medium",
    ]


def test_select_records_ignores_all_skills_when_slice_subset_is_selected(tmp_path: Path) -> None:
    runner = _load_runner_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    (paper_dataset / "all_skills.txt").write_text(
        "owner-low/skill-low\nowner-high/skill-high\nowner-medium/skill-medium\n",
        encoding="utf-8",
    )

    records = runner.select_records(
        paper_dataset=paper_dataset,
        skills_file=None,
        slice_names=("high",),
        start=0,
        stop=None,
        limit=None,
    )

    assert [record.skill_id for record in records] == ["owner-high/skill-high"]


def test_build_jobs_resolves_external_source_and_constructs_run_full_command(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    source_root = tmp_path / "external-e"
    source_dir = (
        source_root
        / "clawhub_skills"
        / "artifacts"
        / "extracted_skills"
        / "owner-high"
        / "skill-high"
    )
    source_dir.mkdir(parents=True)
    records = runner.select_records(
        paper_dataset=paper_dataset,
        skills_file=None,
        slice_names=("high",),
        start=0,
        stop=None,
        limit=None,
    )

    jobs, preflight_rows = runner.build_jobs(
        records=records,
        dataset_root=tmp_path / "missing-local-dataset",
        output_dir=tmp_path / "artifacts",
        staging_root=tmp_path / "source-links",
        drive_map={"E": str(source_root)},
        run_full_args=_runner_args(),
    )

    assert preflight_rows == []
    assert len(jobs) == 1
    job = jobs[0]
    assert job.skill_id == "owner-high/skill-high"
    assert job.source_path == source_dir
    assert job.staged_path == tmp_path / "source-links" / "owner-high" / "skill-high"
    assert job.artifact_dir == tmp_path / "artifacts" / "owner-high" / "skill-high"
    assert "--skill" in job.command
    assert "owner-high/skill-high" in job.command
    assert str(tmp_path / "source-links") in job.command


def test_build_jobs_skips_complete_artifact_even_when_source_is_missing(
    tmp_path: Path,
) -> None:
    runner = _load_runner_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    artifact_dir = tmp_path / "artifacts" / "owner-high" / "skill-high"
    for relative in runner.REQUIRED_ARTIFACTS:
        if relative == STATUS_ARTIFACT:
            continue
        (artifact_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        (artifact_dir / relative).write_text("[]\n", encoding="utf-8")
    write_status_artifact(
        artifact_dir,
        skill_id="owner-high/skill-high",
        artifact_kind=FULL_ARTIFACT_KIND,
    )
    records = runner.select_records(
        paper_dataset=paper_dataset,
        skills_file=None,
        slice_names=("high",),
        start=0,
        stop=None,
        limit=None,
    )

    jobs, preflight_rows = runner.build_jobs(
        records=records,
        dataset_root=tmp_path / "missing-local-dataset",
        output_dir=tmp_path / "artifacts",
        staging_root=tmp_path / "source-links",
        drive_map={"E": str(tmp_path / "missing-external-e")},
        run_full_args=_runner_args(),
    )

    assert jobs == []
    assert len(preflight_rows) == 1
    assert preflight_rows[0]["skill_id"] == "owner-high/skill-high"
    assert preflight_rows[0]["status"] == "skipped_complete"
    assert preflight_rows[0]["missing_artifacts"] == []


def test_build_jobs_can_plan_missing_source_restoration(tmp_path: Path) -> None:
    runner = _load_runner_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    records = runner.select_records(
        paper_dataset=paper_dataset,
        skills_file=None,
        slice_names=("high",),
        start=0,
        stop=None,
        limit=None,
    )

    jobs, preflight_rows = runner.build_jobs(
        records=records,
        dataset_root=tmp_path / "missing-local-dataset",
        output_dir=tmp_path / "artifacts",
        staging_root=tmp_path / "source-links",
        drive_map={"E": str(tmp_path / "missing-external-e")},
        run_full_args=_runner_args(
            "--restore-missing-sources",
            "--source-cache-root",
            str(tmp_path / "source-cache"),
            "--source-repo-template",
            "https://example.invalid/{owner}/{slug}.git",
        ),
    )

    assert preflight_rows == []
    assert len(jobs) == 1
    job = jobs[0]
    assert job.source_path == tmp_path / "source-cache" / "owner-high" / "skill-high"
    assert job.source_restore is not None
    assert job.source_restore.download_urls == (
        "https://wry-manatee-359.convex.site/api/v1/download?slug=skill-high",
    )
    assert job.source_restore.repo_urls == (
        "https://example.invalid/owner-high/skill-high.git",
    )

    row = runner._summary_row(
        skill_id=job.skill_id,
        status="dry_run",
        source_path=job.source_path,
        staged_path=job.staged_path,
        artifact_dir=job.artifact_dir,
        command=job.command,
        missing_artifacts=[],
        source_restore=job.source_restore,
    )
    assert row["source_restore"]["cache_path"] == str(job.source_path)
    assert row["source_restore"]["download_urls"] == [
        "https://wry-manatee-359.convex.site/api/v1/download?slug=skill-high"
    ]
    assert row["source_restore"]["repo_urls"] == [
        "https://example.invalid/owner-high/skill-high.git"
    ]


def test_restore_source_downloads_and_extracts_archive(tmp_path: Path) -> None:
    runner = _load_runner_module()
    archive_path = tmp_path / "skill.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("skill-high/SKILL.md", "# Skill\n")
        archive.writestr("skill-high/index.py", "print('ok')\n")
    cache_path = tmp_path / "source-cache" / "owner-high" / "skill-high"
    plan = runner.SourceRestorePlan(
        cache_path=cache_path,
        download_urls=(archive_path.as_uri(),),
        repo_urls=(),
        depth=1,
        timeout_seconds=10,
    )

    result = runner.restore_source(plan, skill_id="owner-high/skill-high")

    assert result["status"] == "restored"
    assert result["provider"] == "download"
    assert result["source_path"] == str(cache_path)
    assert result["archive_sha256"]
    assert (cache_path / "SKILL.md").read_text(encoding="utf-8") == "# Skill\n"
    assert (cache_path / "index.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_runnable_jobs_refresh_stale_missing_source_summary(tmp_path: Path) -> None:
    runner = _load_runner_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    source_root = tmp_path / "external-e"
    source_dir = (
        source_root
        / "clawhub_skills"
        / "artifacts"
        / "extracted_skills"
        / "owner-high"
        / "skill-high"
    )
    source_dir.mkdir(parents=True)
    records = runner.select_records(
        paper_dataset=paper_dataset,
        skills_file=None,
        slice_names=("high",),
        start=0,
        stop=None,
        limit=None,
    )
    jobs, preflight_rows = runner.build_jobs(
        records=records,
        dataset_root=tmp_path / "missing-local-dataset",
        output_dir=tmp_path / "artifacts",
        staging_root=tmp_path / "source-links",
        drive_map={"E": str(source_root)},
        run_full_args=_runner_args(),
    )

    summary_by_skill = {
        "owner-high/skill-high": {
            "skill_id": "owner-high/skill-high",
            "status": "missing_source",
            "source_path": str(tmp_path / "stale-source"),
        }
    }
    runner._mark_runnable_jobs_queued(summary_by_skill, jobs)

    assert preflight_rows == []
    assert summary_by_skill["owner-high/skill-high"]["status"] == "queued"
    assert summary_by_skill["owner-high/skill-high"]["source_path"] == str(source_dir)


def test_stage_source_link_and_artifact_completion(tmp_path: Path) -> None:
    runner = _load_runner_module()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    staged_path = tmp_path / "staged" / "owner" / "slug"

    runner.stage_source_link(source_dir, staged_path)
    runner.stage_source_link(source_dir, staged_path)

    assert staged_path.is_symlink()
    assert staged_path.resolve() == source_dir.resolve()

    artifact_dir = tmp_path / "artifacts" / "owner" / "slug"
    artifact_dir.mkdir(parents=True)
    assert not runner._full_artifact_complete(artifact_dir)
    for relative in runner.REQUIRED_ARTIFACTS:
        if relative == STATUS_ARTIFACT:
            continue
        (artifact_dir / relative).parent.mkdir(parents=True, exist_ok=True)
        (artifact_dir / relative).write_text("[]\n", encoding="utf-8")
    assert not runner._full_artifact_complete(artifact_dir)
    assert runner._missing_full_artifacts(artifact_dir) == [STATUS_ARTIFACT]

    write_status_artifact(
        artifact_dir,
        skill_id="owner/slug",
        artifact_kind=ABLATION_ARTIFACT_KIND,
    )
    assert not runner._full_artifact_complete(artifact_dir)
    assert runner._missing_full_artifacts(artifact_dir) == [STATUS_ARTIFACT]

    write_status_artifact(
        artifact_dir,
        skill_id="owner/slug",
        artifact_kind=FULL_ARTIFACT_KIND,
    )
    assert runner._full_artifact_complete(artifact_dir)
    assert runner._missing_full_artifacts(artifact_dir) == []


def test_run_paper_artifacts_dry_run_writes_summary(tmp_path: Path) -> None:
    paper_dataset = _write_paper_dataset(tmp_path)
    source_root = tmp_path / "external-e"
    source_dir = (
        source_root
        / "clawhub_skills"
        / "artifacts"
        / "extracted_skills"
        / "owner-high"
        / "skill-high"
    )
    source_dir.mkdir(parents=True)
    summary_out = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--paper-dataset",
            str(paper_dataset),
            "--dataset-root",
            str(tmp_path / "missing-local-dataset"),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--staging-root",
            str(tmp_path / "source-links"),
            "--slices",
            "high",
            "--dry-run",
            "--summary-out",
            str(summary_out),
            "--drive-map",
            f"E={source_root}",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    rows = json.loads(summary_out.read_text(encoding="utf-8"))
    assert rows[0]["skill_id"] == "owner-high/skill-high"
    assert rows[0]["status"] == "dry_run"
    assert rows[0]["source_path"] == str(source_dir)
    assert rows[0]["staged_path"] == str(tmp_path / "source-links" / "owner-high" / "skill-high")
    assert "scripts/run_full.py" in " ".join(rows[0]["command"])
    assert '"statuses": {' in result.stdout


def _runner_args(*extra_args: str) -> argparse.Namespace:
    parser = _load_runner_module().build_arg_parser()
    return parser.parse_args(["--dry-run", *extra_args])


def _write_paper_dataset(tmp_path: Path) -> Path:
    paper_dataset = tmp_path / "paper500"
    records = {
        "high": _sample_index_record(
            owner="owner-high",
            slug="skill-high",
            risk_tier="high_risk",
        ),
        "medium": _sample_index_record(
            owner="owner-medium",
            slug="skill-medium",
            risk_tier="medium_risk",
        ),
        "low": _sample_index_record(
            owner="owner-low",
            slug="skill-low",
            risk_tier="low_risk",
        ),
    }
    for slice_name, record in records.items():
        slice_dir = paper_dataset / slice_name
        slice_dir.mkdir(parents=True)
        (slice_dir / "sample_index.jsonl").write_text(
            json.dumps(record, ensure_ascii=False) + "\n",
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
