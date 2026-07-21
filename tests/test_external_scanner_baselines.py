from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from skillrecon.evaluation.datasets import (
    BaselinePredictionRecord,
    load_baseline_prediction_records,
    write_jsonl_models,
)
from skillrecon.evaluation.external_scanners import (
    build_external_scanner_command,
    external_scanner_payload_to_prediction,
    external_scanner_payload_to_report,
    get_external_scanner_spec,
)


def test_skillspector_payload_maps_high_static_finding_to_violation() -> None:
    payload = {
        "risk_score": 78,
        "risk_severity": "HIGH",
        "filtered_findings": [
            {
                "rule_id": "E2",
                "severity": "HIGH",
                "message": "Env Variable Harvesting",
                "explanation": "Collects API keys and sends them to a remote endpoint.",
                "file": "scripts/sync.py",
                "line": 23,
            }
        ],
    }

    prediction = external_scanner_payload_to_prediction(
        skill_id="demo-skill",
        system_id="baseline_skillspector",
        payload=payload,
    )
    report = external_scanner_payload_to_report(
        skill_id="demo-skill",
        system_id="baseline_skillspector",
        payload=payload,
    )

    assert prediction.main_label == "violation"
    assert prediction.subtype == "unsupported_behavior"
    assert prediction.score == 78
    assert report.summary.violation_count == 1
    assert report.violation_findings[0].code_locations[0].path == "scripts/sync.py"


def test_cisco_sarif_payload_maps_taint_flow_to_composition_violation() -> None:
    payload = {
        "version": "2.1.0",
        "runs": [
            {
                "results": [
                    {
                        "ruleId": "TT4",
                        "level": "error",
                        "message": {
                            "text": "File read to network exfiltration taint flow"
                        },
                        "locations": [
                            {
                                "physicalLocation": {
                                    "artifactLocation": {"uri": "tool.py"},
                                    "region": {"startLine": 41},
                                }
                            }
                        ],
                    }
                ]
            }
        ],
    }

    prediction = external_scanner_payload_to_prediction(
        skill_id="owner/demo",
        system_id="baseline_cisco_skill_scanner",
        payload=payload,
    )
    report = external_scanner_payload_to_report(
        skill_id="owner/demo",
        system_id="baseline_cisco_skill_scanner",
        payload=payload,
    )

    assert prediction.main_label == "violation"
    assert prediction.subtype == "unjustified_composition"
    assert report.violation_findings[0].code_locations[0].line == 41


def test_skillfortify_medium_permission_mismatch_maps_to_scope_violation() -> None:
    payload = {
        "status": "WARNING",
        "max_severity": "MEDIUM",
        "findings": [
            {
                "id": "LP1",
                "severity": "MEDIUM",
                "category": "least privilege",
                "message": "Underdeclared permission for filesystem access",
            }
        ],
    }

    prediction = external_scanner_payload_to_prediction(
        skill_id="demo-skill",
        system_id="baseline_skillfortify",
        payload=payload,
    )

    assert prediction.main_label == "violation"
    assert prediction.subtype == "scope_violation"
    assert prediction.metadata["finding_count"] == 1


def test_low_severity_scanner_finding_maps_to_exposure_only() -> None:
    payload = {
        "findings": [
            {
                "rule_id": "SC1",
                "severity": "LOW",
                "message": "Unpinned dependency in requirements.txt",
            }
        ]
    }

    prediction = external_scanner_payload_to_prediction(
        skill_id="demo-skill",
        system_id="baseline_skillspector",
        payload=payload,
    )

    assert prediction.main_label == "exposure-only"
    assert prediction.subtype is None


def test_composite_high_risk_severity_maps_to_violation() -> None:
    payload = {
        "findings": [
            {
                "rule_id": "RISK1",
                "severity": "HIGH_RISK",
                "message": "Remote network sink reachable from skill code",
            }
        ]
    }

    prediction = external_scanner_payload_to_prediction(
        skill_id="demo-skill",
        system_id="baseline_cisco_skill_scanner",
        payload=payload,
    )

    assert prediction.main_label == "violation"
    assert prediction.subtype == "unsupported_behavior"


def test_skillspector_nested_risk_and_location_are_preserved() -> None:
    payload = {
        "risk_assessment": {"score": 82, "severity": "critical severity"},
        "issues": [
            {
                "issue_code": "E4",
                "severity": "HIGH",
                "description": "Environment credential exfiltration",
                "location": {"file": "src/tool.py", "start_line": 17},
            }
        ],
    }

    prediction = external_scanner_payload_to_prediction(
        skill_id="demo-skill",
        system_id="baseline_skillspector",
        payload=payload,
    )
    report = external_scanner_payload_to_report(
        skill_id="demo-skill",
        system_id="baseline_skillspector",
        payload=payload,
    )

    assert prediction.main_label == "violation"
    assert prediction.score == 82
    assert prediction.metadata["max_severity"] == "critical severity"
    assert report.violation_findings[0].finding_id.endswith("E4")
    assert report.violation_findings[0].code_locations[0].path == "src/tool.py"
    assert report.violation_findings[0].code_locations[0].line == 17


def test_snyk_agent_scan_command_matches_open_source_cli_shape() -> None:
    command = build_external_scanner_command(
        get_external_scanner_spec("baseline_snyk_agent_scan"),
        skill_path=Path("data/skill_dataset/demo"),
        output_path=Path("unused.json"),
    )

    assert command[:4] == ["uvx", "snyk-agent-scan@latest", "scan", "--json"]
    assert "--skills" in command
    assert "--format" not in command
    assert "--output" not in command


def test_external_baseline_script_imports_raw_json(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(
        json.dumps(
            {
                "risk_score": 0,
                "risk_severity": "LOW",
                "filtered_findings": [],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_baselines.py",
            "--scanner",
            "baseline_skillspector",
            "--raw-json",
            f"demo-skill={raw}",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    summary = json.loads(result.stdout)
    prediction_path = Path(summary["predictions_out"])
    records = load_baseline_prediction_records(prediction_path)

    assert summary["predictions"] == 1
    assert records[0].skill_id == "demo-skill"
    assert records[0].system_id == "baseline_skillspector"
    assert records[0].main_label == "benign"


def test_external_baseline_script_uses_artifact_staged_source_for_skillfortify(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "paper_dataset"
    for slice_name in ("high", "medium", "low"):
        (dataset / slice_name).mkdir(parents=True)
        (dataset / slice_name / "sample_index.jsonl").write_text(
            "",
            encoding="utf-8",
        )
    record = {
        "dataset_bucket": "mixed",
        "owner": "DemoOwner",
        "slug": "demo-skill",
        "version": "1.0.0",
        "extract_root": "/mnt/e/missing/DemoOwner/demo-skill",
    }
    (dataset / "high" / "sample_index.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    staged_source = (
        tmp_path / "artifacts" / "DemoOwner" / "demo-skill" / "staged_source"
    )
    staged_source.mkdir(parents=True)
    (staged_source / "SKILL.md").write_text(
        "---\nname: demo-skill\n---\n\nUse this skill to inspect files.\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_external_baselines.py",
            "--scanner",
            "baseline_skillfortify",
            "--paper-dataset",
            str(dataset),
            "--data-root",
            str(tmp_path / "missing_dataset_root"),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    summary = json.loads(result.stdout)
    command = summary["commands"][0]["command"]
    scanner_skill_path = Path(summary["commands"][0]["scanner_skill_path"])

    assert summary["planned"] == 1
    assert summary["commands"][0]["skill_path"] == str(staged_source)
    assert scanner_skill_path.name == "DemoOwner__demo-skill"
    assert scanner_skill_path.parent.name == "baseline_skillfortify"
    assert scanner_skill_path.parent.parent.name == "adapted"
    assert command[:2] == ["skillfortify", "scan"]
    assert command[2] == str(scanner_skill_path)
    assert (
        scanner_skill_path / ".claude" / "skills" / "demo-skill.md"
    ).is_file()


def test_external_baseline_script_imports_20_raw_outputs_per_scanner(
    tmp_path: Path,
) -> None:
    for scanner in (
        "baseline_skillfortify",
        "baseline_cisco_skill_scanner",
        "baseline_skillspector",
        "baseline_snyk_agent_scan",
    ):
        raw_args: list[str] = []
        for index in range(1, 21):
            skill_id = f"owner/skill-{index:02d}"
            raw = tmp_path / scanner / f"skill-{index:02d}.json"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text(
                json.dumps(_raw_payload_for_scanner(scanner, index)),
                encoding="utf-8",
            )
            raw_args.extend(["--raw-json", f"{skill_id}={raw}"])

        output_dir = tmp_path / "out" / scanner
        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_external_baselines.py",
                "--scanner",
                scanner,
                *raw_args,
                "--output-dir",
                str(output_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )

        summary = json.loads(result.stdout)
        records = load_baseline_prediction_records(Path(summary["predictions_out"]))

        assert summary["planned"] == 20
        assert summary["predictions"] == 20
        assert len(records) == 20
        assert {record.system_id for record in records} == {scanner}


def test_external_scanner_registry_contains_paper_method_candidates() -> None:
    assert get_external_scanner_spec("baseline_skillfortify").repo_url.endswith(
        "/skillfortify"
    )
    assert "cisco-ai-defense" in get_external_scanner_spec(
        "baseline_cisco_skill_scanner"
    ).repo_url
    assert "NVIDIA" in get_external_scanner_spec("baseline_skillspector").repo_url


def test_merge_baseline_predictions_script_combines_external_systems(tmp_path: Path) -> None:
    first = tmp_path / "skillfortify.jsonl"
    second = tmp_path / "cisco.jsonl"
    output = tmp_path / "merged.jsonl"
    write_jsonl_models(
        first,
        [
            BaselinePredictionRecord(
                skill_id="demo-skill",
                system_id="baseline_skillfortify",
                main_label="violation",
                subtype="scope_violation",
            )
        ],
    )
    write_jsonl_models(
        second,
        [
            BaselinePredictionRecord(
                skill_id="demo-skill",
                system_id="baseline_cisco_skill_scanner",
                main_label="violation",
                subtype="unsupported_behavior",
            )
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/merge_baseline_predictions.py",
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    summary = json.loads(result.stdout)
    records = load_baseline_prediction_records(output)

    assert summary["records"] == 2
    assert {record.system_id for record in records} == {
        "baseline_skillfortify",
        "baseline_cisco_skill_scanner",
    }


def _raw_payload_for_scanner(scanner: str, index: int) -> dict[str, object]:
    if scanner == "baseline_skillfortify":
        return {
            "status": "WARNING",
            "max_severity": "MEDIUM",
            "findings": [
                {
                    "id": f"LP{index}",
                    "severity": "MEDIUM",
                    "category": "least privilege",
                    "message": "Underdeclared permission for filesystem access",
                }
            ],
        }
    if scanner == "baseline_cisco_skill_scanner":
        return {
            "version": "2.1.0",
            "runs": [
                {
                    "results": [
                        {
                            "ruleId": f"TT{index}",
                            "level": "error" if index % 2 else "warning",
                            "message": {
                                "text": "File read to network exfiltration taint flow"
                            },
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "tool.py"},
                                        "region": {"startLine": index},
                                    }
                                }
                            ],
                        }
                    ]
                }
            ],
        }
    if scanner == "baseline_skillspector":
        return {
            "risk_assessment": {
                "score": 70 + index,
                "severity": "HIGH_RISK",
            },
            "issues": [
                {
                    "issue_code": f"E{index}",
                    "severity": "HIGH",
                    "description": "Environment credential exfiltration",
                    "location": {"file": "src/tool.py", "start_line": index},
                }
            ],
        }
    return {
        "score": 65 + index,
        "securityIssues": [
            {
                "id": f"SNYK-{index}",
                "severity": "HIGH" if index % 2 else "MEDIUM",
                "title": "Skill can read secrets and send them to the network",
                "remediation": "Restrict file access and network sinks.",
                "location": {"path": "skill/SKILL.md", "line": index},
            }
        ],
    }
