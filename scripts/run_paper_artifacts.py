#!/usr/bin/env python3
"""Run the full SkillRecon pipeline for the paper500 dataset artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from skillrecon.core.config import DEFAULT_ENV_CONFIG_PATH, DEFAULT_LLM_CONFIG_PATH
from skillrecon.evaluation.artifacts import (
    FULL_ARTIFACT_KIND,
    REQUIRED_ARTIFACTS,
    artifact_complete,
    load_skill_ids,
    missing_required_artifacts,
)
from skillrecon.evaluation.datasets import (
    FlaggedSkillRecord,
    load_paper_sample_records,
)
from skillrecon.loader.path_resolver import (
    iter_skill_path_candidates,
    parse_windows_drive_map,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DATASET = PROJECT_ROOT / "data" / "evaluation" / "skill_paper500_dataset"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "skill_dataset"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "derived" / "paper500"
DEFAULT_STAGING_ROOT = PROJECT_ROOT / "derived" / "paper500_source_links"
DEFAULT_SOURCE_CACHE_ROOT = PROJECT_ROOT / "derived" / "paper500_source_cache"
DEFAULT_SOURCE_DOWNLOAD_TEMPLATES = (
    "https://wry-manatee-359.convex.site/api/v1/download?slug={slug_url}",
)
DEFAULT_SOURCE_REPO_TEMPLATES: tuple[str, ...] = ()

PAPER_SLICE_ORDER = ("high", "medium", "low")


@dataclass(frozen=True)
class SourceRestorePlan:
    """Network restore plan for one missing paper-dataset source package."""

    cache_path: Path
    download_urls: tuple[str, ...]
    repo_urls: tuple[str, ...]
    depth: int
    timeout_seconds: int


@dataclass(frozen=True)
class PaperArtifactJob:
    """Resolved work item for one paper-dataset skill."""

    record: FlaggedSkillRecord
    source_path: Path
    staged_path: Path
    artifact_dir: Path
    command: list[str]
    source_restore: SourceRestorePlan | None = None

    @property
    def skill_id(self) -> str:
        return self.record.skill_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate per-skill artifacts for the paper500 experiment dataset"
    )
    parser.add_argument(
        "--paper-dataset",
        default=str(DEFAULT_PAPER_DATASET),
        help="Paper dataset directory containing high/medium/low sample_index.jsonl files",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Local fallback root for skill packages",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Artifact root expected by scripts/run_experiments.py",
    )
    parser.add_argument(
        "--staging-root",
        default=str(DEFAULT_STAGING_ROOT),
        help="Directory where owner/slug source symlinks are created for run_full.py",
    )
    parser.add_argument(
        "--skills-file",
        help="Optional newline-delimited skill id list; defaults to <paper-dataset>/all_skills.txt",
    )
    parser.add_argument(
        "--slices",
        default=",".join(PAPER_SLICE_ORDER),
        help="Comma-separated paper slices to include: high,medium,low",
    )
    parser.add_argument("--start", type=int, default=0, help="Start offset after filtering")
    parser.add_argument("--stop", type=int, help="Exclusive stop offset after filtering")
    parser.add_argument("--limit", type=int, help="Maximum number of selected skills")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of concurrent run_full.py subprocesses",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun skills even when all required artifacts already exist",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Create source symlinks and summary entries without invoking run_full.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve jobs and print commands without creating symlinks or running the pipeline",
    )
    parser.add_argument(
        "--summary-out",
        help="Summary JSON path; defaults to <output-dir>/paper_artifact_run_summary.json",
    )
    parser.add_argument(
        "--log-dir",
        help="Subprocess log directory; defaults to <output-dir>/paper_artifact_logs",
    )
    parser.add_argument(
        "--drive-map",
        action="append",
        default=[],
        help="Map Windows drive paths from sample indexes, e.g. E=/mnt/e or E=/data/e",
    )
    parser.add_argument(
        "--restore-missing-sources",
        action="store_true",
        help=(
            "Restore locally missing source packages into --source-cache-root before "
            "running run_full.py"
        ),
    )
    parser.add_argument(
        "--source-cache-root",
        default=str(DEFAULT_SOURCE_CACHE_ROOT),
        help="Local cache root used when --restore-missing-sources is enabled",
    )
    parser.add_argument(
        "--source-download-template",
        action="append",
        default=[],
        help=(
            "Archive download URL template for source restoration. Supports {owner}, "
            "{slug}, {skill_id}, {version}, and *_url URL-encoded variants. "
            "Defaults to the ClawHub package download endpoint."
        ),
    )
    parser.add_argument(
        "--source-repo-template",
        action="append",
        default=[],
        help=(
            "Repository URL template for source restoration. Supports {owner}, "
            "{slug}, {skill_id}, {version}, and *_url URL-encoded variants. "
            "No repository fallback is attempted unless this option is provided."
        ),
    )
    parser.add_argument(
        "--source-restore-depth",
        type=int,
        default=1,
        help="Git clone depth used for restored source packages",
    )
    parser.add_argument(
        "--source-restore-timeout",
        type=int,
        default=300,
        help="Per git command timeout in seconds for source restoration",
    )
    _add_run_full_passthrough_args(parser)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if args.source_restore_depth < 1:
        parser.error("--source-restore-depth must be >= 1")
    if args.source_restore_timeout < 1:
        parser.error("--source-restore-timeout must be >= 1")
    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.stop is not None and args.stop < args.start:
        parser.error("--stop must be >= --start")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be >= 0")

    try:
        drive_map = parse_windows_drive_map(args.drive_map)
        slice_names = _parse_slice_names(args.slices)
    except ValueError as exc:
        parser.error(str(exc))

    paper_dataset = Path(args.paper_dataset)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    staging_root = Path(args.staging_root)
    summary_out = Path(args.summary_out) if args.summary_out else (
        output_dir / "paper_artifact_run_summary.json"
    )
    log_dir = Path(args.log_dir) if args.log_dir else output_dir / "paper_artifact_logs"

    records = select_records(
        paper_dataset=paper_dataset,
        skills_file=Path(args.skills_file) if args.skills_file else None,
        slice_names=slice_names,
        start=args.start,
        stop=args.stop,
        limit=args.limit,
    )
    jobs, preflight_rows = build_jobs(
        records=records,
        dataset_root=dataset_root,
        output_dir=output_dir,
        staging_root=staging_root,
        drive_map=drive_map,
        run_full_args=args,
    )

    existing_summary = _load_summary(summary_out)
    summary_by_skill = {
        str(row["skill_id"]): row
        for row in existing_summary
        if isinstance(row, dict) and isinstance(row.get("skill_id"), str)
    }
    ordered_skill_ids = [record.skill_id for record in records]

    for row in preflight_rows:
        summary_by_skill[str(row["skill_id"])] = row

    runnable_jobs = _select_runnable_jobs(jobs, force=args.force)
    if not args.force:
        for job in jobs:
            if _full_artifact_complete(job.artifact_dir):
                summary_by_skill[job.skill_id] = _summary_row(
                    skill_id=job.skill_id,
                    status="skipped_complete",
                    source_path=job.source_path,
                    staged_path=job.staged_path,
                    artifact_dir=job.artifact_dir,
                    command=job.command,
                    missing_artifacts=[],
                    source_restore=job.source_restore,
                )

    if args.dry_run:
        for job in runnable_jobs:
            summary_by_skill[job.skill_id] = _summary_row(
                skill_id=job.skill_id,
                status="dry_run",
                source_path=job.source_path,
                staged_path=job.staged_path,
                artifact_dir=job.artifact_dir,
                command=job.command,
                missing_artifacts=_missing_full_artifacts(job.artifact_dir),
                source_restore=job.source_restore,
            )
            print(" ".join(job.command))
        _write_summary(summary_out, summary_by_skill, ordered_skill_ids)
        _print_result(summary_out, summary_by_skill, ordered_skill_ids)
        return

    if args.stage_only:
        for job in jobs:
            restore_result = None
            try:
                restore_result = ensure_source_available(job)
                stage_source_link(job.source_path, job.staged_path)
                status = "staged"
                error = None
            except (OSError, SourceRestoreError) as exc:
                status = "stage_failed"
                error = str(exc)
                if isinstance(exc, SourceRestoreError):
                    restore_result = exc.result
            summary_by_skill[job.skill_id] = _summary_row(
                skill_id=job.skill_id,
                status=status,
                source_path=job.source_path,
                staged_path=job.staged_path,
                artifact_dir=job.artifact_dir,
                command=job.command,
                missing_artifacts=_missing_full_artifacts(job.artifact_dir),
                source_restore=job.source_restore,
                source_restore_result=restore_result,
                error=error,
            )
        _write_summary(summary_out, summary_by_skill, ordered_skill_ids)
        _print_result(summary_out, summary_by_skill, ordered_skill_ids)
        return

    _mark_runnable_jobs_queued(summary_by_skill, runnable_jobs)
    _write_summary(summary_out, summary_by_skill, ordered_skill_ids)

    log_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_job, job, log_dir=log_dir): job
            for job in runnable_jobs
        }
        for future in as_completed(futures):
            row = future.result()
            summary_by_skill[str(row["skill_id"])] = row
            _write_summary(summary_out, summary_by_skill, ordered_skill_ids)
            print(f"{row['status']} {row['skill_id']}")

    _write_summary(summary_out, summary_by_skill, ordered_skill_ids)
    _print_result(summary_out, summary_by_skill, ordered_skill_ids)


def select_records(
    *,
    paper_dataset: Path,
    skills_file: Path | None,
    slice_names: tuple[str, ...],
    start: int,
    stop: int | None,
    limit: int | None,
) -> list[FlaggedSkillRecord]:
    records_by_slice = load_paper_sample_records(paper_dataset, slice_names=slice_names)
    by_skill: dict[str, FlaggedSkillRecord] = {}
    stable_records: list[FlaggedSkillRecord] = []
    for slice_name in slice_names:
        for record in records_by_slice[slice_name]:
            if record.skill_id in by_skill:
                raise ValueError(f"Duplicate skill_id in paper sample indexes: {record.skill_id}")
            by_skill[record.skill_id] = record
            stable_records.append(record)

    skills_path = _resolve_skills_file(
        paper_dataset,
        skills_file,
        slice_names=slice_names,
    )
    if skills_path is not None:
        skill_ids = load_skill_ids(skills_path)
        missing = [skill_id for skill_id in skill_ids if skill_id not in by_skill]
        if missing:
            raise ValueError(
                f"{skills_path} contains skill ids absent from sample indexes: {missing[:5]}"
            )
        selected = [by_skill[skill_id] for skill_id in skill_ids]
    else:
        selected = stable_records

    sliced = selected[start:stop]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def build_jobs(
    *,
    records: list[FlaggedSkillRecord],
    dataset_root: Path,
    output_dir: Path,
    staging_root: Path,
    drive_map: dict[str, str],
    run_full_args: argparse.Namespace,
) -> tuple[list[PaperArtifactJob], list[dict[str, Any]]]:
    jobs: list[PaperArtifactJob] = []
    preflight_rows: list[dict[str, Any]] = []
    for record in records:
        source_candidates = list(
            iter_skill_path_candidates(
                dataset_root,
                record.owner,
                record.slug,
                extract_root=record.extract_root,
                windows_drive_map=drive_map,
            )
        )
        source_path = next((candidate for candidate in source_candidates if candidate.exists()), None)
        source_restore = None
        if source_path is None and run_full_args.restore_missing_sources:
            source_restore = _build_source_restore_plan(record, run_full_args)
            source_candidates.append(source_restore.cache_path)
            source_path = source_restore.cache_path
        if source_path is None:
            source_path = source_candidates[0]
        staged_path = staging_root / record.skill_id
        artifact_dir = output_dir / record.skill_id
        command = build_run_full_command(
            skill_id=record.skill_id,
            data_root=staging_root,
            output_dir=output_dir,
            args=run_full_args,
        )
        if not run_full_args.force and _full_artifact_complete(artifact_dir):
            preflight_rows.append(
                _summary_row(
                    skill_id=record.skill_id,
                    status="skipped_complete",
                    source_path=source_path,
                    staged_path=staged_path,
                    artifact_dir=artifact_dir,
                    command=command,
                    missing_artifacts=[],
                    source_candidates=source_candidates,
                    source_restore=source_restore,
                )
            )
            continue
        if not source_path.exists() and source_restore is None:
            preflight_rows.append(
                _summary_row(
                    skill_id=record.skill_id,
                    status="missing_source",
                    source_path=source_path,
                    staged_path=staged_path,
                    artifact_dir=artifact_dir,
                    command=command,
                    missing_artifacts=_missing_full_artifacts(artifact_dir),
                    source_candidates=source_candidates,
                    error=f"Source directory does not exist: {source_path}",
                )
            )
            continue
        jobs.append(
            PaperArtifactJob(
                record=record,
                source_path=source_path,
                staged_path=staged_path,
                artifact_dir=artifact_dir,
                command=command,
                source_restore=source_restore,
            )
        )
    return jobs, preflight_rows


def _select_runnable_jobs(
    jobs: list[PaperArtifactJob],
    *,
    force: bool,
) -> list[PaperArtifactJob]:
    if force:
        return jobs
    return [
        job
        for job in jobs
        if not _full_artifact_complete(job.artifact_dir)
    ]


def _mark_runnable_jobs_queued(
    summary_by_skill: dict[str, dict[str, Any]],
    runnable_jobs: list[PaperArtifactJob],
) -> None:
    """Refresh stale summary rows for jobs that this invocation will run."""
    for job in runnable_jobs:
        summary_by_skill[job.skill_id] = _summary_row(
            skill_id=job.skill_id,
            status="queued",
            source_path=job.source_path,
            staged_path=job.staged_path,
            artifact_dir=job.artifact_dir,
            command=job.command,
            missing_artifacts=_missing_full_artifacts(job.artifact_dir),
            source_restore=job.source_restore,
        )


def build_run_full_command(
    *,
    skill_id: str,
    data_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_full.py"),
        "--skill",
        skill_id,
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_dir),
        "--env-config",
        str(Path(args.env_config)),
        "--llm-config",
        str(Path(args.llm_config)),
        "--taxonomy-version",
        args.taxonomy_version,
        "--prompt-version",
        args.prompt_version,
    ]
    _append_optional(command, "--n-samples", args.n_samples)
    _append_optional(command, "--codeql-bin", args.codeql_bin)
    _append_optional(command, "--base-url", args.base_url)
    _append_optional(command, "--model", args.model)
    _append_optional(command, "--api-key-env", args.api_key_env)
    _append_optional(command, "--temperature", args.temperature)
    _append_optional(command, "--max-tokens", args.max_tokens)
    _append_optional(
        command,
        "--max-candidates-per-behavior",
        args.max_candidates_per_behavior,
    )
    _append_optional(
        command,
        "--max-alignment-fallbacks-per-step",
        args.max_alignment_fallbacks_per_step,
    )
    _append_optional(
        command,
        "--max-semantic-event-fallbacks",
        args.max_semantic_event_fallbacks,
    )
    _append_optional(
        command,
        "--max-semantic-path-fallbacks",
        args.max_semantic_path_fallbacks,
    )
    _append_optional(command, "--overlap-policy-path", args.overlap_policy_path)
    if args.render_pyvis:
        command.append("--render-pyvis")
    if args.no_pyvis_full_graph:
        command.append("--no-pyvis-full-graph")
    for witness_id in args.pyvis_witness_id:
        command.extend(["--pyvis-witness-id", witness_id])
    if args.pyvis_output_dir:
        command.extend(["--pyvis-output-dir", str(Path(args.pyvis_output_dir))])
    if args.verbose:
        command.append("--verbose")
    return command


def run_job(job: PaperArtifactJob, *, log_dir: Path) -> dict[str, Any]:
    started_at = time.time()
    stdout_path = log_dir / f"{_safe_log_stem(job.skill_id)}.stdout.txt"
    stderr_path = log_dir / f"{_safe_log_stem(job.skill_id)}.stderr.txt"
    restore_result = None
    try:
        restore_result = ensure_source_available(job)
        stage_source_link(job.source_path, job.staged_path)
        env = _subprocess_env()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle:
            with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                completed = subprocess.run(
                    job.command,
                    cwd=PROJECT_ROOT,
                    check=False,
                    text=True,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    env=env,
                )
        status = "ok" if completed.returncode == 0 and _full_artifact_complete(job.artifact_dir) else "failed"
        return _summary_row(
            skill_id=job.skill_id,
            status=status,
            source_path=job.source_path,
            staged_path=job.staged_path,
            artifact_dir=job.artifact_dir,
            command=job.command,
            missing_artifacts=_missing_full_artifacts(job.artifact_dir),
            source_restore=job.source_restore,
            source_restore_result=restore_result,
            returncode=completed.returncode,
            stdout_log=stdout_path,
            stderr_log=stderr_path,
            stdout_tail=_read_tail(stdout_path),
            stderr_tail=_read_tail(stderr_path),
            started_at=started_at,
            ended_at=time.time(),
        )
    except Exception as exc:  # noqa: BLE001 - batch runs must record and continue.
        return _summary_row(
            skill_id=job.skill_id,
            status="failed",
            source_path=job.source_path,
            staged_path=job.staged_path,
            artifact_dir=job.artifact_dir,
            command=job.command,
            missing_artifacts=_missing_full_artifacts(job.artifact_dir),
            source_restore=job.source_restore,
            source_restore_result=(
                exc.result if isinstance(exc, SourceRestoreError) else restore_result
            ),
            stdout_log=stdout_path,
            stderr_log=stderr_path,
            stdout_tail=_read_tail(stdout_path),
            stderr_tail=_read_tail(stderr_path),
            started_at=started_at,
            ended_at=time.time(),
            error=str(exc),
        )


def stage_source_link(source_path: Path, staged_path: Path) -> None:
    """Create or refresh one owner/slug symlink consumed by run_full.py."""
    resolved_source = source_path.resolve()
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    if staged_path.exists() or staged_path.is_symlink():
        if staged_path.resolve() == resolved_source:
            return
        if staged_path.is_symlink():
            staged_path.unlink()
        else:
            raise FileExistsError(
                f"Cannot replace non-symlink staged source path: {staged_path}"
            )
    os.symlink(resolved_source, staged_path, target_is_directory=True)


class SourceRestoreError(RuntimeError):
    """Raised when a missing source package cannot be restored."""

    def __init__(
        self,
        message: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result


def ensure_source_available(job: PaperArtifactJob) -> dict[str, Any] | None:
    """Restore a missing source package when a job carries a restore plan."""
    if job.source_restore is None:
        if not job.source_path.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {job.source_path}")
        return None

    if job.source_path.is_dir():
        return {
            "status": "already_available",
            "source_path": str(job.source_path),
        }
    return restore_source(job.source_restore, skill_id=job.skill_id)


def restore_source(plan: SourceRestorePlan, *, skill_id: str) -> dict[str, Any]:
    """Restore a source package into the configured cache."""
    if plan.cache_path.is_dir():
        return {
            "status": "already_available",
            "source_path": str(plan.cache_path),
        }
    if plan.cache_path.exists():
        raise SourceRestoreError(f"Source cache path exists but is not a directory: {plan.cache_path}")

    plan.cache_path.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []

    for download_url in plan.download_urls:
        attempt: dict[str, Any] = {"provider": "download", "download_url": download_url}
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{plan.cache_path.name}.restore.",
                dir=str(plan.cache_path.parent),
            ) as temp_name:
                temp_dir = Path(temp_name)
                archive_path = temp_dir / "source.archive"
                download = _download_source_archive(
                    download_url,
                    archive_path,
                    timeout=plan.timeout_seconds,
                )
                attempt["download"] = download
                if not download.get("ok"):
                    attempts.append(attempt)
                    continue

                source_root, extract = _extract_source_archive(
                    archive_path,
                    temp_dir / "extracted",
                )
                attempt["extract"] = extract
                installed_path = _install_extracted_source(source_root, plan.cache_path)
                attempts.append(attempt)
                return {
                    "status": "restored",
                    "provider": "download",
                    "skill_id": skill_id,
                    "source_path": str(installed_path),
                    "download_url": download_url,
                    "archive_sha256": download.get("sha256"),
                    "archive_size_bytes": download.get("size_bytes"),
                    "archive_filename": download.get("filename"),
                    "content_type": download.get("content_type"),
                    "attempts": attempts,
                }
        except SourceRestoreError as exc:
            attempt["error"] = str(exc)
            attempts.append(attempt)

    for repo_url in plan.repo_urls:
        clone = _run_git_command(
            [
                "git",
                "clone",
                "--depth",
                str(plan.depth),
                repo_url,
                str(plan.cache_path),
            ],
            timeout=plan.timeout_seconds,
        )
        attempts.append({"repo_url": repo_url, "clone": clone})
        if clone["returncode"] == 0 and plan.cache_path.is_dir():
            return {
                "status": "restored",
                "provider": "git",
                "skill_id": skill_id,
                "source_path": str(plan.cache_path),
                "repo_url": repo_url,
                "attempts": attempts,
            }

    attempted_sources = list(plan.download_urls) + list(plan.repo_urls)
    result = {
        "status": "failed",
        "skill_id": skill_id,
        "source_path": str(plan.cache_path),
        "attempted_sources": attempted_sources,
        "attempts": attempts,
    }
    raise SourceRestoreError(
        "Unable to restore source for "
        f"{skill_id}; attempted sources: "
        + ", ".join(attempted_sources),
        result=result,
    )


def _download_source_archive(
    url: str,
    archive_path: Path,
    *,
    timeout: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SkillRecon-source-restore/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers = dict(response.headers.items())
            digest = hashlib.sha256()
            size = 0
            with archive_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    digest.update(chunk)
                    handle.write(chunk)
            return {
                "ok": True,
                "status": getattr(response, "status", None),
                "content_type": response.headers.get("content-type"),
                "content_length": response.headers.get("content-length"),
                "filename": _filename_from_content_disposition(
                    response.headers.get("content-disposition")
                ),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "content_type": exc.headers.get("content-type") if exc.headers else None,
            "error": str(exc),
        }
    except (OSError, TimeoutError, urllib.error.URLError) as exc:
        return {
            "ok": False,
            "status": None,
            "error": str(exc),
        }


def _filename_from_content_disposition(value: str | None) -> str | None:
    if not value:
        return None
    for part in value.split(";"):
        part = part.strip()
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    return None


def _extract_source_archive(
    archive_path: Path,
    extract_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    extract_dir.mkdir(parents=True, exist_ok=False)
    if zipfile.is_zipfile(archive_path):
        member_count = _extract_zip_safely(archive_path, extract_dir)
        archive_type = "zip"
    elif tarfile.is_tarfile(archive_path):
        member_count = _extract_tar_safely(archive_path, extract_dir)
        archive_type = "tar"
    else:
        raise SourceRestoreError("Downloaded source archive is not a supported zip/tar file")

    source_root = _select_extracted_source_root(extract_dir)
    if not source_root.is_dir() or not any(source_root.iterdir()):
        raise SourceRestoreError("Downloaded source archive did not contain source files")
    return source_root, {
        "archive_type": archive_type,
        "member_count": member_count,
        "source_root_name": source_root.name if source_root != extract_dir else ".",
    }


def _extract_zip_safely(archive_path: Path, extract_dir: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_archive_target(extract_dir, member.filename)
            if target is None:
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source:
                with target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            extracted += 1
    return extracted


def _extract_tar_safely(archive_path: Path, extract_dir: Path) -> int:
    extracted = 0
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = _safe_archive_target(extract_dir, member.name)
            if target is None:
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise SourceRestoreError(
                    f"Unsupported archive member type in source package: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise SourceRestoreError(f"Unable to read archive member: {member.name}")
            with source:
                with target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            extracted += 1
    return extracted


def _safe_archive_target(extract_dir: Path, member_name: str) -> Path | None:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute():
        return None
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if not parts:
        return None
    if ".." in parts or ":" in parts[0]:
        raise SourceRestoreError(f"Unsafe archive member path: {member_name}")
    target = extract_dir.joinpath(*parts).resolve()
    root = extract_dir.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise SourceRestoreError(f"Unsafe archive member path: {member_name}") from exc
    return target


def _select_extracted_source_root(extract_dir: Path) -> Path:
    entries = [
        path
        for path in extract_dir.iterdir()
        if path.name != "__MACOSX"
    ]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def _install_extracted_source(source_root: Path, cache_path: Path) -> Path:
    if cache_path.is_dir():
        return cache_path
    if cache_path.exists():
        raise SourceRestoreError(f"Source cache path exists but is not a directory: {cache_path}")

    install_tmp = cache_path.parent / f".{cache_path.name}.install.{os.getpid()}.{time.time_ns()}"
    try:
        shutil.move(str(source_root), str(install_tmp))
        if cache_path.is_dir():
            shutil.rmtree(install_tmp, ignore_errors=True)
            return cache_path
        if cache_path.exists():
            raise SourceRestoreError(
                f"Source cache path appeared but is not a directory: {cache_path}"
            )
        install_tmp.rename(cache_path)
    except Exception:
        if install_tmp.exists():
            shutil.rmtree(install_tmp, ignore_errors=True)
        raise
    return cache_path


def _run_git_command(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": -1,
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
            "error": f"timed out after {timeout} seconds",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _full_artifact_complete(artifact_dir: Path) -> bool:
    return artifact_complete(
        artifact_dir,
        expected_status_kind=FULL_ARTIFACT_KIND,
    )


def _missing_full_artifacts(artifact_dir: Path) -> list[str]:
    return missing_required_artifacts(
        artifact_dir,
        expected_status_kind=FULL_ARTIFACT_KIND,
    )


def _add_run_full_passthrough_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json passed to run_full.py",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json passed to run_full.py",
    )
    parser.add_argument("--n-samples", type=int, help="ICCM sample count override")
    parser.add_argument("--codeql-bin", help="CodeQL binary override")
    parser.add_argument("--base-url", help="LLM API base URL override")
    parser.add_argument("--model", help="LLM model name override")
    parser.add_argument("--api-key-env", help="API key env var or literal key override")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
    parser.add_argument(
        "--max-candidates-per-behavior",
        type=int,
        help="Maximum contract candidates kept per behavior object",
    )
    parser.add_argument(
        "--max-alignment-fallbacks-per-step",
        type=int,
        help="Maximum step-to-code-unit fallback candidates per documentation step",
    )
    parser.add_argument(
        "--max-semantic-event-fallbacks",
        type=int,
        help="Maximum semantic event fallback candidates per clause",
    )
    parser.add_argument(
        "--max-semantic-path-fallbacks",
        type=int,
        help="Maximum semantic path fallback candidates per clause",
    )
    parser.add_argument("--overlap-policy-path", help="Path to capability overlap policy JSON")
    parser.add_argument(
        "--taxonomy-version",
        default="v2",
        help="Taxonomy version suffix passed to run_full.py",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt version tag passed to run_full.py",
    )
    parser.add_argument(
        "--render-pyvis",
        action="store_true",
        help="Pass --render-pyvis to run_full.py",
    )
    parser.add_argument("--pyvis-output-dir", help="PyVis output directory override")
    parser.add_argument(
        "--pyvis-witness-id",
        action="append",
        default=[],
        help="Render only specific witness ids; may be passed multiple times",
    )
    parser.add_argument(
        "--no-pyvis-full-graph",
        action="store_true",
        help="Skip full G_X PyVis graph when rendering",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose run_full.py logs")


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _build_source_restore_plan(
    record: FlaggedSkillRecord,
    args: argparse.Namespace,
) -> SourceRestorePlan:
    download_templates = tuple(
        args.source_download_template or DEFAULT_SOURCE_DOWNLOAD_TEMPLATES
    )
    repo_templates = tuple(args.source_repo_template or DEFAULT_SOURCE_REPO_TEMPLATES)
    return SourceRestorePlan(
        cache_path=Path(args.source_cache_root) / record.owner / record.slug,
        download_urls=tuple(
            _format_source_template(template, record)
            for template in download_templates
        ),
        repo_urls=tuple(
            _format_source_template(template, record)
            for template in repo_templates
        ),
        depth=args.source_restore_depth,
        timeout_seconds=args.source_restore_timeout,
    )


def _format_source_template(template: str, record: FlaggedSkillRecord) -> str:
    values = {
        "owner": record.owner,
        "slug": record.slug,
        "skill_id": record.skill_id,
        "version": record.version,
    }
    values.update(
        {
            f"{key}_url": urllib.parse.quote(value, safe="")
            for key, value in values.items()
        }
    )
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported source template placeholder {exc.args[0]!r} in {template!r}"
        ) from exc


def _source_restore_payload(plan: SourceRestorePlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "cache_path": str(plan.cache_path),
        "download_urls": list(plan.download_urls),
        "repo_urls": list(plan.repo_urls),
        "depth": plan.depth,
        "timeout_seconds": plan.timeout_seconds,
    }


def _parse_slice_names(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = [name for name in names if name not in PAPER_SLICE_ORDER]
    if invalid:
        raise ValueError(f"Invalid paper slice(s): {invalid}; expected high, medium, low")
    if not names:
        raise ValueError("At least one paper slice must be selected")
    return names


def _resolve_skills_file(
    paper_dataset: Path,
    skills_file: Path | None,
    *,
    slice_names: tuple[str, ...],
) -> Path | None:
    if skills_file is not None:
        return skills_file
    if slice_names != PAPER_SLICE_ORDER:
        return None
    default_path = paper_dataset / "all_skills.txt"
    return default_path if default_path.is_file() else None


def _load_summary(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected summary JSON list at {path}")
    return [row for row in payload if isinstance(row, dict)]


def _write_summary(
    path: Path,
    summary_by_skill: dict[str, dict[str, Any]],
    ordered_skill_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_skill_set = set(ordered_skill_ids)
    rows = [
        summary_by_skill[skill_id]
        for skill_id in ordered_skill_ids
        if skill_id in summary_by_skill
    ]
    extras = [
        row
        for skill_id, row in sorted(summary_by_skill.items())
        if skill_id not in ordered_skill_set
    ]
    path.write_text(
        json.dumps(rows + extras, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _summary_row(
    *,
    skill_id: str,
    status: str,
    source_path: Path,
    staged_path: Path,
    artifact_dir: Path,
    command: list[str],
    missing_artifacts: list[str],
    source_candidates: list[Path] | None = None,
    source_restore: SourceRestorePlan | None = None,
    source_restore_result: dict[str, Any] | None = None,
    returncode: int | None = None,
    stdout_log: Path | None = None,
    stderr_log: Path | None = None,
    stdout_tail: str | None = None,
    stderr_tail: str | None = None,
    started_at: float | None = None,
    ended_at: float | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "skill_id": skill_id,
        "status": status,
        "source_path": str(source_path),
        "staged_path": str(staged_path),
        "artifact_dir": str(artifact_dir),
        "command": command,
        "missing_artifacts": missing_artifacts,
    }
    if source_candidates is not None:
        row["source_candidates"] = [str(path) for path in source_candidates]
    source_restore_payload = _source_restore_payload(source_restore)
    if source_restore_payload is not None:
        row["source_restore"] = source_restore_payload
    if source_restore_result is not None:
        row["source_restore_result"] = source_restore_result
    if returncode is not None:
        row["returncode"] = returncode
    if stdout_log is not None:
        row["stdout_log"] = str(stdout_log)
    if stderr_log is not None:
        row["stderr_log"] = str(stderr_log)
    if stdout_tail:
        row["stdout_tail"] = stdout_tail
    if stderr_tail:
        row["stderr_tail"] = stderr_tail
    if started_at is not None:
        row["started_at"] = started_at
    if ended_at is not None:
        row["ended_at"] = ended_at
    if started_at is not None and ended_at is not None:
        row["duration_seconds"] = round(ended_at - started_at, 3)
    if error:
        row["error"] = error
    return row


def _subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    src_path = str(PROJECT_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else f"{src_path}{os.pathsep}{current}"
    return env


def _safe_log_stem(skill_id: str) -> str:
    return skill_id.replace("/", "__").replace("\\", "__")


def _read_tail(path: Path, *, max_chars: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _print_result(
    summary_out: Path,
    summary_by_skill: dict[str, dict[str, Any]],
    ordered_skill_ids: list[str],
) -> None:
    statuses: dict[str, int] = {}
    for skill_id in ordered_skill_ids:
        row = summary_by_skill.get(skill_id)
        if row is None:
            continue
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    print(
        json.dumps(
            {
                "summary_out": str(summary_out),
                "selected": len(ordered_skill_ids),
                "recorded": sum(statuses.values()),
                "statuses": statuses,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
