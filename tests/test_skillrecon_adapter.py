from __future__ import annotations

import json
from pathlib import Path

from skillrecon.evaluation.skillrecon_adapter import build_skillrecon_report


def test_diagnostic_unsupported_dynamic_url_maps_to_exposure(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_minimal_artifact(
        artifact_dir,
        findings=[
            {
                "finding_id": "finding-1",
                "finding_type": "unsupported_behavior",
                "support_level": "diagnostic",
                "support_strength": "none",
                "related_event_ids": ["e1"],
                "rationale": "Observed behavior has no supporting declaration path.",
            }
        ],
        events=[
            {
                "event_id": "e1",
                "unit_id": "u1",
                "capability": "data_encode_send",
                "api_call": "data encoding (JSON)",
                "location": "src/client.py:42",
                "arguments": ["<dynamic-url>"],
                "tier": "concepts",
                "language": "python",
                "file_path": "src/client.py",
                "line": 42,
                "detail": "data encoding (JSON)",
            }
        ],
    )

    report = build_skillrecon_report("owner/skill", artifact_dir)

    assert report.overall_label == "exposure-only"
    assert report.summary.violation_count == 0
    assert report.summary.exposure_count == 1
    assert report.exposure_findings[0].subtype == "unsupported_behavior"


def test_code_credential_read_keeps_violation(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_minimal_artifact(
        artifact_dir,
        findings=[
            {
                "finding_id": "finding-1",
                "finding_type": "contradicted_behavior",
                "support_level": "graph_backed",
                "support_strength": "strong",
                "related_event_ids": ["e1"],
                "rationale": "Observed behavior contradicts an explicit prohibited rule.",
            }
        ],
        events=[
            {
                "event_id": "e1",
                "unit_id": "u1",
                "capability": "env_var_read",
                "api_call": "os.getenv",
                "location": "src/agent.py:17",
                "arguments": ["OPENAI_API_KEY"],
                "tier": "api_graph",
                "language": "python",
                "file_path": "src/agent.py",
                "line": 17,
                "detail": "os.getenv",
            }
        ],
    )

    report = build_skillrecon_report("owner/skill", artifact_dir)

    assert report.overall_label == "violation"
    assert report.summary.violation_count == 1
    assert report.violation_findings[0].subtype == "contradicted_behavior"


def test_eval_execution_keeps_violation(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_minimal_artifact(
        artifact_dir,
        findings=[
            {
                "finding_id": "finding-1",
                "finding_type": "unsupported_behavior",
                "support_level": "diagnostic",
                "support_strength": "none",
                "related_event_ids": ["e1"],
                "rationale": "Observed behavior has no supporting declaration path.",
            }
        ],
        events=[
            {
                "event_id": "e1",
                "unit_id": "u1",
                "capability": "eval_exec",
                "api_call": "eval()",
                "location": "src/calculator.py:40",
                "arguments": [],
                "tier": "concepts",
                "language": "python",
                "file_path": "src/calculator.py",
                "line": 40,
                "detail": "code execution",
            }
        ],
    )

    report = build_skillrecon_report("owner/skill", artifact_dir)

    assert report.overall_label == "violation"
    assert report.summary.violation_count == 1


def test_plain_base64_attachment_encoding_maps_to_exposure(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_minimal_artifact(
        artifact_dir,
        findings=[
            {
                "finding_id": "finding-1",
                "finding_type": "unsupported_behavior",
                "support_level": "diagnostic",
                "support_strength": "none",
                "related_event_ids": ["e1"],
                "rationale": "Observed behavior has no supporting declaration path.",
            }
        ],
        events=[
            {
                "event_id": "e1",
                "unit_id": "u1",
                "capability": "data_encode_send",
                "api_call": "data encoding (Base64)",
                "location": "scripts/attachment.py:22",
                "arguments": ["<dynamic-url>"],
                "tier": "concepts",
                "language": "python",
                "file_path": "scripts/attachment.py",
                "line": 22,
                "detail": "data encoding (Base64)",
            }
        ],
    )

    report = build_skillrecon_report("owner/skill", artifact_dir)

    assert report.overall_label == "exposure-only"
    assert report.summary.violation_count == 0
    assert report.summary.exposure_count == 1


def test_graph_backed_document_network_only_maps_to_exposure(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    _write_minimal_artifact(
        artifact_dir,
        findings=[
            {
                "finding_id": "finding-1",
                "finding_type": "contradicted_behavior",
                "support_level": "graph_backed",
                "support_strength": "strong",
                "related_event_ids": ["e1"],
                "rationale": "Observed behavior contradicts an explicit prohibited rule.",
            }
        ],
        events=[
            {
                "event_id": "e1",
                "unit_id": "doc::d1",
                "capability": "http_request",
                "api_call": "https://example.com/settings",
                "location": "SKILL.md:12",
                "arguments": ["https://example.com/settings"],
                "tier": "instruction",
                "language": "markdown",
                "file_path": "SKILL.md",
                "line": 12,
                "detail": "https://example.com/settings",
            }
        ],
    )

    report = build_skillrecon_report("owner/skill", artifact_dir)

    assert report.overall_label == "exposure-only"
    assert report.summary.violation_count == 0
    assert report.summary.exposure_count == 1


def _write_minimal_artifact(
    artifact_dir: Path,
    *,
    findings: list[dict[str, object]],
    events: list[dict[str, object]],
) -> None:
    artifact_dir.mkdir(parents=True)
    _write_json(artifact_dir / "findings.json", findings)
    _write_json(artifact_dir / "event_table.json", events)
    _write_json(artifact_dir / "resource_table.json", [])
    _write_json(artifact_dir / "path_table.json", [])
    _write_json(artifact_dir / "exposures.json", [])
    _write_json(artifact_dir / "diagnostics.json", [])


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
