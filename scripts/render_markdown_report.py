#!/usr/bin/env python3
"""Render a concise reviewer-facing Markdown summary for one artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillrecon.evaluation import build_skillrecon_report


def _load_json(path: Path, default: object) -> object:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _report_payload(skill_id: str, artifact_dir: Path) -> dict[str, object]:
    report_path = artifact_dir / "report.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    report = build_skillrecon_report(skill_id, artifact_dir)
    payload = report.model_dump()
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _witness_index(artifact_dir: Path) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    witnesses = _load_json(artifact_dir / "witnesses.json", [])
    validations = _load_json(artifact_dir / "witness_validation.json", [])
    by_finding: dict[str, dict[str, object]] = {}
    for witness in witnesses:
        if not isinstance(witness, dict):
            continue
        finding_id = str(witness.get("finding_id", ""))
        if finding_id and finding_id not in by_finding:
            by_finding[finding_id] = witness
    by_witness_id: dict[str, dict[str, object]] = {}
    for validation in validations:
        if not isinstance(validation, dict):
            continue
        witness_id = str(validation.get("witness_id", ""))
        if witness_id:
            by_witness_id[witness_id] = validation
    return by_finding, by_witness_id


def _format_code_locations(locations: list[dict[str, object]]) -> str:
    if not locations:
        return "n/a"
    parts: list[str] = []
    for location in locations[:5]:
        path = str(location.get("path", ""))
        line = location.get("line")
        parts.append(f"{path}:{line}" if line else path)
    return ", ".join(parts)


def _append_finding_section(
    lines: list[str],
    *,
    title: str,
    items: list[dict[str, object]],
    witness_by_finding: dict[str, dict[str, object]],
    validation_by_witness_id: dict[str, dict[str, object]],
    artifact_dir: Path,
    max_items: int,
) -> None:
    lines.extend([f"## {title}", ""])
    if not items:
        lines.extend(["无。", ""])
        return

    for index, item in enumerate(items[:max_items], start=1):
        finding_id = str(item.get("finding_id", item.get("alert_id", f"item-{index:02d}")))
        subtype = item.get("subtype") or item.get("alert_type") or item.get("main_label") or "n/a"
        rationale = str(item.get("rationale", "")).strip() or "n/a"
        capability_atoms = item.get("capability_atoms") or []
        matched_clauses = item.get("matched_clauses") or item.get("affected_clauses") or []
        code_locations = item.get("code_locations") or []
        lines.append(f"### {index}. `{finding_id}`")
        lines.append(f"- type: `{subtype}`")
        if capability_atoms:
            lines.append(f"- capabilities: {', '.join(f'`{atom}`' for atom in capability_atoms[:8])}")
        if matched_clauses:
            lines.append(f"- clauses: {', '.join(f'`{clause}`' for clause in matched_clauses[:8])}")
        lines.append(f"- code locations: {_format_code_locations(code_locations)}")
        lines.append(f"- rationale: {rationale}")

        witness = witness_by_finding.get(finding_id)
        if witness is not None:
            witness_id = str(witness.get("witness_id", ""))
            validation = validation_by_witness_id.get(witness_id, {})
            lines.append(f"- witness: `{witness_id}`")
            lines.append(f"- witness exact: `{bool(witness.get('is_exact', False))}`")
            lines.append(f"- witness validation passed: `{bool(validation.get('passed', False))}`")
            explanatory_path = artifact_dir / "viz" / f"{witness_id}-explanatory.html"
            if explanatory_path.is_file():
                lines.append(
                    f"- explanatory subgraph: [{witness_id}-explanatory.html](viz/{witness_id}-explanatory.html)"
                )
        lines.append("")

    if len(items) > max_items:
        lines.append(f"_仅展示前 {max_items} 条；完整结果见 `report.json`、`findings.json` 与 `witnesses.json`。_")
        lines.append("")


def render_markdown_report(
    *,
    skill_id: str,
    artifact_dir: Path,
    output_path: Path,
    max_findings: int,
) -> None:
    report = _report_payload(skill_id, artifact_dir)
    witness_by_finding, validation_by_witness_id = _witness_index(artifact_dir)

    summary = report.get("summary", {})
    graphs = report.get("graphs", {})
    overall_label = report.get("overall_label", "unknown")
    violation_findings = report.get("violation_findings", [])
    exposure_findings = report.get("exposure_findings", [])
    contract_alerts = report.get("contract_quality_alerts", [])

    lines = [
        f"# SkillRecon Review Report: {skill_id}",
        "",
        "## Summary",
        "",
        f"- overall label: `{overall_label}`",
        f"- violations: `{summary.get('violation_count', 0)}`",
        f"- exposures: `{summary.get('exposure_count', 0)}`",
        f"- contract-quality alerts: `{summary.get('contract_quality_count', 0)}`",
        f"- G_D: `{graphs.get('g_d', {}).get('node_count', 0)}` nodes / `{graphs.get('g_d', {}).get('edge_count', 0)}` edges",
        f"- G_C: `{graphs.get('g_c', {}).get('node_count', 0)}` nodes / `{graphs.get('g_c', {}).get('edge_count', 0)}` edges",
        f"- G_X: `{graphs.get('g_x', {}).get('node_count', 0)}` nodes / `{graphs.get('g_x', {}).get('edge_count', 0)}` edges",
        "",
    ]

    _append_finding_section(
        lines,
        title="Violations",
        items=violation_findings if isinstance(violation_findings, list) else [],
        witness_by_finding=witness_by_finding,
        validation_by_witness_id=validation_by_witness_id,
        artifact_dir=artifact_dir,
        max_items=max_findings,
    )
    _append_finding_section(
        lines,
        title="Exposures",
        items=exposure_findings if isinstance(exposure_findings, list) else [],
        witness_by_finding=witness_by_finding,
        validation_by_witness_id=validation_by_witness_id,
        artifact_dir=artifact_dir,
        max_items=max_findings,
    )
    _append_finding_section(
        lines,
        title="Contract Quality Alerts",
        items=contract_alerts if isinstance(contract_alerts, list) else [],
        witness_by_finding=witness_by_finding,
        validation_by_witness_id=validation_by_witness_id,
        artifact_dir=artifact_dir,
        max_items=max_findings,
    )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a reviewer-facing Markdown report from one SkillRecon artifact directory"
    )
    parser.add_argument("--skill", required=True, help="Skill identifier")
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing findings.json, witnesses.json, and related artifacts",
    )
    parser.add_argument(
        "--md-out",
        help="Output Markdown path (defaults to <artifact-dir>/review_report.md)",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=10,
        help="Maximum number of items rendered per section",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_path = Path(args.md_out) if args.md_out else artifact_dir / "review_report.md"
    render_markdown_report(
        skill_id=args.skill,
        artifact_dir=artifact_dir,
        output_path=output_path,
        max_findings=args.max_findings,
    )
    print(f"Wrote reviewer Markdown report to {output_path}")


if __name__ == "__main__":
    main()
