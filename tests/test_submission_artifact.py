from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/check_submission_artifact.py")


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_submission_artifact", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_submission_artifact_static_checks_pass_for_committed_inputs() -> None:
    checker = _load_checker_module()
    args = checker.argparse.Namespace(
        paper_dir="paper",
        data_root="data/skill_dataset",
        reviewer_cases="experiments/configs/reviewer_cases_v1.json",
        paper_dataset="data/evaluation/skill_paper500_dataset",
        single_artifact_root=None,
        paper_artifact_root=None,
        json_out=None,
    )

    summary = checker.summarize_checks(checker.run_checks(args))

    assert summary["status"] == "ok"
    assert summary["errors"] == 0
    names = {item["name"] for item in summary["checks"]}
    assert "reviewer_cases" in names
    assert "paper_dataset" in names
    assert "experiment_bundle_entry" in names
    assert "table_renderers" in names
    assert "paper_evaluation_labels" in names


def test_submission_artifact_cli_writes_json_summary(tmp_path: Path) -> None:
    output = tmp_path / "artifact_check.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--json-out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    stdout_payload = json.loads(result.stdout)

    assert payload["status"] == "ok"
    assert stdout_payload["status"] == "ok"
