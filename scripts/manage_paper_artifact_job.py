#!/usr/bin/env python3
"""Manage the long-running paper500 artifact build job."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any

from skillrecon.evaluation.artifacts import (
    FULL_ARTIFACT_KIND,
    artifact_complete,
    artifact_coverage,
    load_skill_ids,
    post_behavior_recovery_missing_inputs,
)
from skillrecon.evaluation.datasets import load_paper_sample_records
from skillrecon.loader.path_resolver import (
    iter_skill_path_candidates,
    parse_windows_drive_map,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DATASET = PROJECT_ROOT / "data" / "evaluation" / "skill_paper500_dataset"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "skill_dataset"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "derived" / "paper500"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "derived" / "paper500_source_links"
DEFAULT_PID_FILE = DEFAULT_ARTIFACT_ROOT / "paper_artifact_background.pid"
DEFAULT_STATE_FILE = DEFAULT_ARTIFACT_ROOT / "paper_artifact_background_state.json"
DEFAULT_STDOUT_LOG = DEFAULT_ARTIFACT_ROOT / "paper_artifact_background.stdout.log"
DEFAULT_STDERR_LOG = DEFAULT_ARTIFACT_ROOT / "paper_artifact_background.stderr.log"
DEFAULT_SUMMARY_OUT = DEFAULT_ARTIFACT_ROOT / "paper_artifact_run_summary.json"
DEFAULT_SOURCE_CACHE_ROOT = PROJECT_ROOT / "derived" / "paper500_source_cache"
DEFAULT_SOURCE_DOWNLOAD_TEMPLATES = (
    "https://wry-manatee-359.convex.site/api/v1/download?slug={slug_url}",
)
DEFAULT_SOURCE_REPO_TEMPLATES: tuple[str, ...] = ()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start, inspect, or stop the paper500 artifact background job"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="Start the background artifact builder")
    _add_common_paths(start)
    start.add_argument("--max-workers", type=int, default=4)
    start.add_argument("--max-tokens", type=int, default=8192)
    start.add_argument(
        "--drive-map",
        action="append",
        default=[],
        help="Pass Windows drive mappings to run_paper_artifacts.py, e.g. E=/mnt/e",
    )
    start.add_argument(
        "--skip-source-preflight",
        action="store_true",
        help="Start without checking whether missing/incomplete sample sources are visible",
    )
    start.add_argument(
        "--restore-missing-sources",
        action="store_true",
        help=(
            "Let run_paper_artifacts.py restore locally missing source packages into "
            "--source-cache-root"
        ),
    )
    start.add_argument(
        "--source-cache-root",
        default=str(DEFAULT_SOURCE_CACHE_ROOT),
        help="Local cache root used when --restore-missing-sources is enabled",
    )
    start.add_argument(
        "--source-download-template",
        action="append",
        default=[],
        help=(
            "Archive download URL template for source restoration. Supports {owner}, "
            "{slug}, {skill_id}, {version}, and *_url URL-encoded variants."
        ),
    )
    start.add_argument(
        "--source-repo-template",
        action="append",
        default=[],
        help=(
            "Repository URL template for source restoration. Supports {owner}, "
            "{slug}, {skill_id}, {version}, and *_url URL-encoded variants."
        ),
    )
    start.add_argument(
        "--source-restore-depth",
        type=int,
        default=1,
        help="Git clone depth used for restored source packages",
    )
    start.add_argument(
        "--source-restore-timeout",
        type=int,
        default=300,
        help="Per git command timeout in seconds for source restoration",
    )
    start.add_argument("--force", action="store_true")
    start.add_argument(
        "--allow-existing",
        action="store_true",
        help="Return status instead of failing when a managed job is already running",
    )

    status = subparsers.add_parser("status", help="Print job and artifact coverage status")
    _add_common_paths(status)

    stop = subparsers.add_parser("stop", help="Stop the managed background job")
    _add_common_paths(stop)
    stop.add_argument("--timeout", type=float, default=15.0)

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "start":
        payload = start_job(args)
    elif args.command == "status":
        payload = status_job(args)
    elif args.command == "stop":
        payload = stop_job(args)
    else:  # pragma: no cover - argparse enforces choices.
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def start_job(args: argparse.Namespace) -> dict[str, Any]:
    pid_file = Path(args.pid_file)
    existing_pid = _read_pid(pid_file)
    if existing_pid is not None and _process_running(existing_pid):
        payload = status_job(args)
        if args.allow_existing:
            payload["status"] = "already_running"
            return payload
        raise SystemExit(f"Managed artifact job already running: pid={existing_pid}")

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    command = _build_artifact_command(args)

    if not args.skip_source_preflight:
        source_preflight = _source_preflight(args)
        if _source_preflight_blocks_start(source_preflight):
            state = {
                "status": "blocked_missing_sources",
                "checked_at_unix": time.time(),
                "command": command,
                "cwd": str(PROJECT_ROOT),
                "source_preflight": source_preflight,
                "summary_out": str(Path(args.summary_out)),
            }
            _write_json(Path(args.state_file), state)
            payload = status_job(args)
            payload["status"] = "blocked_missing_sources"
            payload["source_preflight"] = source_preflight
            return payload

    env = dict(os.environ)
    src_path = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    )

    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    with stdout_path.open("ab") as stdout_handle:
        with stderr_path.open("ab") as stderr_handle:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                start_new_session=True,
            )

    state = {
        "pid": process.pid,
        "started_at_unix": time.time(),
        "command": command,
        "cwd": str(PROJECT_ROOT),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "summary_out": str(Path(args.summary_out)),
    }
    _write_json(Path(args.state_file), state)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{process.pid}\n", encoding="utf-8")

    payload = status_job(args)
    payload["status"] = "started"
    return payload


def status_job(args: argparse.Namespace) -> dict[str, Any]:
    pid = _read_pid(Path(args.pid_file))
    coverage = _coverage(args)
    summary_statuses = _summary_statuses(Path(args.summary_out))
    running = bool(pid is not None and _process_running(pid))
    payload: dict[str, Any] = {
        "pid": pid,
        "running": running,
        "cmdline": _read_cmdline(pid) if pid is not None else None,
        "coverage": coverage,
        "partial_recovery": _partial_recovery(args),
        "summary_statuses": summary_statuses,
        "pid_file": str(Path(args.pid_file)),
        "state_file": str(Path(args.state_file)),
        "stdout_log": str(Path(args.stdout_log)),
        "stderr_log": str(Path(args.stderr_log)),
        "summary_out": str(Path(args.summary_out)),
    }
    state = _read_json(Path(args.state_file))
    if (
        isinstance(state, dict)
        and state.get("status") == "blocked_missing_sources"
        and isinstance(state.get("source_preflight"), dict)
    ):
        payload["last_status"] = state["status"]
        payload["last_checked_at_unix"] = state.get("checked_at_unix")
        payload["last_source_preflight"] = state["source_preflight"]
    return payload


def stop_job(args: argparse.Namespace) -> dict[str, Any]:
    pid = _read_pid(Path(args.pid_file))
    if pid is None:
        return {**status_job(args), "status": "not_running"}
    if not _process_running(pid):
        return {**status_job(args), "status": "not_running"}

    cmdline = _read_cmdline(pid) or ""
    if "run_paper_artifacts.py" not in cmdline:
        raise SystemExit(f"Refusing to stop unexpected process pid={pid}: {cmdline}")

    os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not _process_running(pid):
            return {**status_job(args), "status": "stopped"}
        time.sleep(0.25)

    os.killpg(pid, signal.SIGKILL)
    return {**status_job(args), "status": "killed"}


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--paper-dataset", default=str(DEFAULT_PAPER_DATASET))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--summary-out", default=str(DEFAULT_SUMMARY_OUT))
    parser.add_argument("--pid-file", default=str(DEFAULT_PID_FILE))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--stdout-log", default=str(DEFAULT_STDOUT_LOG))
    parser.add_argument("--stderr-log", default=str(DEFAULT_STDERR_LOG))


def _build_artifact_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_paper_artifacts.py"),
        "--paper-dataset",
        str(Path(args.paper_dataset)),
        "--dataset-root",
        str(Path(args.dataset_root)),
        "--output-dir",
        str(Path(args.artifact_root)),
        "--staging-root",
        str(Path(args.source_root)),
        "--max-workers",
        str(args.max_workers),
        "--summary-out",
        str(Path(args.summary_out)),
        "--max-tokens",
        str(args.max_tokens),
    ]
    for drive_map in args.drive_map:
        command.extend(["--drive-map", drive_map])
    if args.restore_missing_sources:
        command.append("--restore-missing-sources")
        command.extend(["--source-cache-root", str(Path(args.source_cache_root))])
        for template in args.source_download_template:
            command.extend(["--source-download-template", template])
        for template in args.source_repo_template:
            command.extend(["--source-repo-template", template])
        command.extend(["--source-restore-depth", str(args.source_restore_depth)])
        command.extend(["--source-restore-timeout", str(args.source_restore_timeout)])
    if args.force:
        command.append("--force")
    return command


def _source_preflight(args: argparse.Namespace) -> dict[str, object]:
    paper_dataset = Path(args.paper_dataset)
    dataset_root = Path(args.dataset_root)
    artifact_root = Path(args.artifact_root)
    skills_file = paper_dataset / "all_skills.txt"
    if not skills_file.is_file():
        return {
            "status": "error",
            "error": f"missing skills file: {skills_file}",
            "expected": 0,
            "needs_source": 0,
            "available": 0,
            "missing": 0,
        }

    try:
        drive_map = parse_windows_drive_map(args.drive_map)
        records_by_slice = load_paper_sample_records(paper_dataset)
    except Exception as exc:  # noqa: BLE001 - startup diagnostics should stay structured.
        return {
            "status": "error",
            "error": str(exc),
            "expected": 0,
            "needs_source": 0,
            "available": 0,
            "missing": 0,
        }

    records_by_skill = {
        record.skill_id: record
        for records in records_by_slice.values()
        for record in records
    }
    skill_ids = load_skill_ids(skills_file)
    needs_source = 0
    available = 0
    missing = 0
    restorable = 0
    missing_records: list[str] = []
    available_examples: list[dict[str, object]] = []
    missing_examples: list[dict[str, object]] = []
    restorable_examples: list[dict[str, object]] = []

    for skill_id in skill_ids:
        if not args.force and artifact_complete(
            artifact_root / skill_id,
            expected_status_kind=FULL_ARTIFACT_KIND,
        ):
            continue

        needs_source += 1
        record = records_by_skill.get(skill_id)
        if record is None:
            missing_records.append(skill_id)
            missing += 1
            continue

        candidates = list(
            iter_skill_path_candidates(
                dataset_root,
                record.owner,
                record.slug,
                extract_root=record.extract_root,
                windows_drive_map=drive_map,
            )
        )
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        if existing is not None:
            available += 1
            if len(available_examples) < 5:
                available_examples.append(
                    {"skill_id": skill_id, "source_path": str(existing)}
                )
            continue

        if getattr(args, "restore_missing_sources", False):
            restorable += 1
            if len(restorable_examples) < 5:
                restorable_examples.append(
                    {
                        "skill_id": skill_id,
                        "cache_path": str(_source_cache_path(args, record.owner, record.slug)),
                        "download_urls": _source_download_urls(args, record),
                        "repo_urls": _source_repo_urls(args, record),
                    }
                )
            continue

        missing += 1
        if len(missing_examples) < 5:
            missing_examples.append(
                {
                    "skill_id": skill_id,
                    "source_candidates": [str(candidate) for candidate in candidates],
                }
            )

    return {
        "status": "ok",
        "expected": len(skill_ids),
        "needs_source": needs_source,
        "available": available,
        "restorable": restorable,
        "missing": missing,
        "missing_records": missing_records[:5],
        "available_examples": available_examples,
        "restorable_examples": restorable_examples,
        "missing_examples": missing_examples,
        "drive_map": drive_map,
        "restore_missing_sources": bool(getattr(args, "restore_missing_sources", False)),
    }


def _source_preflight_blocks_start(source_preflight: dict[str, object]) -> bool:
    if source_preflight.get("status") != "ok":
        return True
    return (
        int(source_preflight.get("needs_source", 0)) > 0
        and int(source_preflight.get("missing", 0)) > 0
    )


def _source_cache_path(args: argparse.Namespace, owner: str, slug: str) -> Path:
    return Path(args.source_cache_root) / owner / slug


def _source_download_urls(
    args: argparse.Namespace,
    record: object,
) -> list[str]:
    templates = tuple(args.source_download_template or DEFAULT_SOURCE_DOWNLOAD_TEMPLATES)
    return [_format_source_template(template, record) for template in templates]


def _source_repo_urls(
    args: argparse.Namespace,
    record: object,
) -> list[str]:
    templates = tuple(args.source_repo_template or DEFAULT_SOURCE_REPO_TEMPLATES)
    return [_format_source_template(template, record) for template in templates]


def _format_source_template(template: str, record: object) -> str:
    owner = str(getattr(record, "owner"))
    slug = str(getattr(record, "slug"))
    skill_id = str(getattr(record, "skill_id"))
    version = str(getattr(record, "version", ""))
    values = {
        "owner": owner,
        "slug": slug,
        "skill_id": skill_id,
        "version": version,
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


def _coverage(args: argparse.Namespace) -> dict[str, object]:
    skills_file = Path(args.paper_dataset) / "all_skills.txt"
    if not skills_file.is_file():
        return {"expected": 0, "complete": 0, "error": f"missing {skills_file}"}
    skill_ids = load_skill_ids(skills_file)
    coverage = artifact_coverage(
        skill_ids=skill_ids,
        artifact_root=Path(args.artifact_root),
        expected_status_kind=FULL_ARTIFACT_KIND,
    )
    return {
        "expected": coverage["expected"],
        "complete": coverage["complete"],
        "missing_dirs": len(coverage["missing_dirs"]),
        "incomplete": len(coverage["incomplete"]),
    }


def _partial_recovery(args: argparse.Namespace) -> dict[str, object]:
    skills_file = Path(args.paper_dataset) / "all_skills.txt"
    if not skills_file.is_file():
        return {
            "recoverable": 0,
            "not_recoverable": 0,
            "error": f"missing {skills_file}",
        }

    skill_ids = load_skill_ids(skills_file)
    coverage = artifact_coverage(
        skill_ids=skill_ids,
        artifact_root=Path(args.artifact_root),
        expected_status_kind=FULL_ARTIFACT_KIND,
    )
    recoverable_examples: list[dict[str, object]] = []
    not_recoverable_examples: list[dict[str, object]] = []
    recoverable = 0
    not_recoverable = 0
    for skill_id, missing_required in coverage["incomplete"].items():  # type: ignore[union-attr]
        artifact_dir = Path(args.artifact_root) / skill_id
        missing_inputs = post_behavior_recovery_missing_inputs(artifact_dir)
        if not missing_inputs:
            recoverable += 1
            if len(recoverable_examples) < 5:
                recoverable_examples.append(
                    {
                        "skill_id": skill_id,
                        "missing_required_artifacts": missing_required[:8],
                    }
                )
            continue

        not_recoverable += 1
        if len(not_recoverable_examples) < 5:
            not_recoverable_examples.append(
                {
                    "skill_id": skill_id,
                    "missing_inputs": missing_inputs[:8],
                    "missing_required_artifacts": missing_required[:8],
                }
            )

    return {
        "recoverable": recoverable,
        "not_recoverable": not_recoverable,
        "recoverable_examples": recoverable_examples,
        "not_recoverable_examples": not_recoverable_examples,
        "dry_run_command": [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "recover_partial_paper_artifact.py"),
            "--skills-file",
            str(skills_file),
            "--output-dir",
            str(Path(args.artifact_root)),
            "--data-root",
            str(Path(args.dataset_root)),
            "--dry-run",
        ],
    }


def _summary_statuses(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"unreadable": 1}
    if not isinstance(rows, list):
        return {"invalid": 1}
    counts = Counter(
        str(row.get("status"))
        for row in rows
        if isinstance(row, dict) and row.get("status") is not None
    )
    return dict(sorted(counts.items()))


def _read_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_cmdline(pid: int | None) -> str | None:
    if pid is None:
        return None
    path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


if __name__ == "__main__":
    main()
