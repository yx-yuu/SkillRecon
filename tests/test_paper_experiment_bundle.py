from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/run_paper_experiment_bundle.py")


def _load_bundle_module():
    spec = importlib.util.spec_from_file_location("run_paper_experiment_bundle", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_validate_dataset_accepts_matching_gold_records(tmp_path: Path) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    args = bundle.build_arg_parser().parse_args(["--paper-dataset", str(paper_dataset)])
    paths = bundle.resolve_paths(args)

    summary = bundle.validate_dataset(paths)

    assert summary["records"] == 3
    assert summary["gold_labels"] == str(paper_dataset / "gold_labels.jsonl")
    assert summary["external_predictions"] == str(
        paper_dataset / "baseline_predictions" / "openclaw.jsonl"
    )


def test_validate_dataset_rejects_gold_records_outside_paper_sample(tmp_path: Path) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    _write_jsonl(
        paper_dataset / "gold_labels.jsonl",
        [
            _gold_record("owner-high/skill-high", "violation"),
            _gold_record("owner-medium/skill-medium", "benign"),
            _gold_record("extra/skill", "benign"),
        ],
    )
    args = bundle.build_arg_parser().parse_args(["--paper-dataset", str(paper_dataset)])
    paths = bundle.resolve_paths(args)

    try:
        bundle.validate_dataset(paths)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("validate_dataset should reject mismatched skill ids")

    assert "missing=1" in message
    assert "extra=1" in message


def test_build_commands_plans_full_bundle_with_skip_existing_ablations(tmp_path: Path) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    args = bundle.build_arg_parser().parse_args(
        [
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--ablation-root",
            str(tmp_path / "ablations"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    paths = bundle.resolve_paths(args)

    commands = bundle.build_commands(args, paths)

    assert list(commands) == ["ablation", "experiments", "rq2_stats", "appendix"]
    assert commands["ablation"][-1] == "--skip-existing"
    assert str(paper_dataset / "all_skills.txt") in commands["ablation"]
    experiment_line = " ".join(commands["experiments"])
    stats_line = " ".join(commands["rq2_stats"])
    for system_id in bundle.ABLATION_SYSTEM_IDS:
        assert f"--system-artifact-root {system_id}=" in experiment_line
    assert "--external-predictions" in experiment_line
    assert str(paper_dataset / "baseline_predictions" / "openclaw.jsonl") in experiment_line
    assert "--external-predictions" in stats_line
    assert str(paper_dataset / "baseline_predictions" / "openclaw.jsonl") in stats_line


def test_build_commands_prefers_merged_external_baseline_predictions(
    tmp_path: Path,
) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    merged = paper_dataset / "baseline_predictions" / "paper_method_baselines.jsonl"
    _write_jsonl(
        merged,
        [
            {
                "skill_id": "owner-high/skill-high",
                "system_id": "baseline_skillfortify",
                "main_label": "violation",
                "subtype": "scope_violation",
                "rationale": "merged external prediction",
            }
        ],
    )
    args = bundle.build_arg_parser().parse_args(
        [
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    paths = bundle.resolve_paths(args)

    summary = bundle.validate_dataset(paths)
    commands = bundle.build_commands(args, paths)

    assert summary["external_predictions"] == str(merged)
    assert str(merged) in " ".join(commands["experiments"])
    assert str(merged) in " ".join(commands["rq2_stats"])


def test_build_commands_accepts_explicit_external_baseline_predictions(
    tmp_path: Path,
) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    explicit = tmp_path / "external.jsonl"
    _write_jsonl(
        explicit,
        [
            {
                "skill_id": "owner-medium/skill-medium",
                "system_id": "baseline_cisco_skill_scanner",
                "main_label": "benign",
                "subtype": None,
                "rationale": "explicit external prediction",
            }
        ],
    )
    args = bundle.build_arg_parser().parse_args(
        [
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--external-predictions",
            str(explicit),
        ]
    )
    paths = bundle.resolve_paths(args)
    commands = bundle.build_commands(args, paths)

    assert paths.external_predictions == explicit
    assert str(explicit) in " ".join(commands["experiments"])
    assert str(explicit) in " ".join(commands["rq2_stats"])


def test_build_commands_can_skip_ablations_and_downstream_system_roots(
    tmp_path: Path,
) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    args = bundle.build_arg_parser().parse_args(
        [
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-ablation",
        ]
    )
    paths = bundle.resolve_paths(args)

    commands = bundle.build_commands(args, paths)

    assert "ablation" not in commands
    assert "--system-artifact-root" not in " ".join(commands["experiments"])
    assert "--system-artifact-root" not in " ".join(commands["rq2_stats"])


def test_build_commands_can_include_human_audit(tmp_path: Path) -> None:
    bundle = _load_bundle_module()
    paper_dataset = _write_paper_dataset(tmp_path)
    responses = tmp_path / "responses.jsonl"
    responses.write_text("", encoding="utf-8")
    args = bundle.build_arg_parser().parse_args(
        [
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--output-dir",
            str(tmp_path / "out"),
            "--include-human-audit",
            "--human-audit-responses",
            str(responses),
            "--human-audit-max-per-subtype",
            "2",
        ]
    )
    paths = bundle.resolve_paths(args)

    commands = bundle.build_commands(args, paths)

    assert "human_audit" in commands
    human_audit_line = " ".join(commands["human_audit"])
    assert "build_human_audit_study.py" in human_audit_line
    assert "--responses" in human_audit_line
    assert "--max-per-subtype 2" in human_audit_line


def test_dry_run_writes_plan_and_does_not_require_complete_artifacts(tmp_path: Path) -> None:
    paper_dataset = _write_paper_dataset(tmp_path)
    summary_out = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "missing-artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--output-dir",
            str(tmp_path / "out"),
            "--summary-out",
            str(summary_out),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["main_artifact_coverage"]["expected"] == 3
    assert summary["main_artifact_coverage"]["complete"] == 0
    assert "ablation" in summary["commands"]
    assert '"status": "dry_run"' in result.stdout


def test_incomplete_main_artifacts_block_formal_run(tmp_path: Path) -> None:
    paper_dataset = _write_paper_dataset(tmp_path)
    summary_out = tmp_path / "summary.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--paper-dataset",
            str(paper_dataset),
            "--artifact-root",
            str(tmp_path / "missing-artifacts"),
            "--source-root",
            str(tmp_path / "sources"),
            "--output-dir",
            str(tmp_path / "out"),
            "--summary-out",
            str(summary_out),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    summary = json.loads(summary_out.read_text(encoding="utf-8"))
    assert result.returncode == 2
    assert summary["status"] == "blocked_incomplete_main_artifacts"
    assert summary["main_artifact_coverage"]["missing_dirs"] == [
        "owner-high/skill-high",
        "owner-medium/skill-medium",
        "owner-low/skill-low",
    ]


def _write_paper_dataset(tmp_path: Path) -> Path:
    paper_dataset = tmp_path / "paper500"
    specs = [
        ("high", "owner-high", "skill-high", "high_risk", "violation"),
        ("medium", "owner-medium", "skill-medium", "medium_risk", "exposure-only"),
        ("low", "owner-low", "skill-low", "low_risk", "benign"),
    ]
    gold_rows: list[dict[str, object]] = []
    openclaw_rows: list[dict[str, object]] = []
    skill_ids: list[str] = []
    for slice_name, owner, slug, risk_tier, label in specs:
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
        skill_id = f"{owner}/{slug}"
        skill_ids.append(skill_id)
        gold_rows.append(_gold_record(skill_id, label, risk_tier=risk_tier))
        openclaw_rows.append(
            {
                "skill_id": skill_id,
                "system_id": "baseline_openclaw",
                "main_label": "benign",
                "subtype": None,
                "rationale": "test prediction",
                "score": None,
                "metadata": {},
            }
        )
    (paper_dataset / "all_skills.txt").write_text(
        "".join(f"{skill_id}\n" for skill_id in skill_ids),
        encoding="utf-8",
    )
    _write_jsonl(paper_dataset / "gold_labels.jsonl", gold_rows)
    _write_jsonl(
        paper_dataset / "baseline_predictions" / "openclaw.jsonl",
        openclaw_rows,
    )
    return paper_dataset


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gold_record(
    skill_id: str,
    label: str,
    *,
    risk_tier: str | None = None,
) -> dict[str, object]:
    return {
        "skill_id": skill_id,
        "gold": {
            "label": label,
            "violation_subtype": (
                "unsupported_behavior" if label == "violation" else None
            ),
            "rationale": "test gold label",
        },
        "risk_stratum": risk_tier,
        "bucket": "pure_py",
        "clause_labels": [],
        "edge_labels": [],
        "expected_sites": [],
        "metadata": {},
    }


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
