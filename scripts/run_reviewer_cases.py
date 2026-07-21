#!/usr/bin/env python3
"""Run the reviewer case bundle end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the curated 10-case reviewer bundle through SkillRecon"
    )
    parser.add_argument(
        "--case-manifest",
        default="experiments/configs/reviewer_cases_v1.json",
        help="Path to the curated reviewer-case manifest",
    )
    parser.add_argument(
        "--data-root",
        default="data/skill_dataset",
        help="Root directory containing the curated skill directories",
    )
    parser.add_argument(
        "--output-root",
        default="derived/reviewer_cases",
        help="Output directory for reviewer artifacts",
    )
    parser.add_argument(
        "--env-config",
        default="experiments/configs/env_config.json",
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default="experiments/configs/llm_config.json",
        help="Path to llm_config.json",
    )
    parser.add_argument("--codeql-bin", help="Optional CodeQL binary override")
    parser.add_argument("--base-url", help="Optional LLM base URL override")
    parser.add_argument("--model", help="Optional LLM model override")
    parser.add_argument("--api-key-env", help="Optional LLM API key env or literal key")
    parser.add_argument(
        "--n-samples",
        type=int,
        help="ICCM sample count; omit to use env_config vote_policy.n_samples",
    )
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
    parser.add_argument(
        "--overlap-policy-path",
        help="Path to the capability overlap policy JSON",
    )
    parser.add_argument(
        "--render-pyvis",
        action="store_true",
        help="Render the first witness explanatory subgraph for each case",
    )
    parser.add_argument(
        "--max-findings",
        type=int,
        default=10,
        help="Maximum findings rendered into review_report.md",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Restrict execution to specific skill ids; may be passed multiple times",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo_root / args.case_manifest).read_text(encoding="utf-8"))
    selected_cases = {
        case["skill_id"]: case
        for case in manifest
        if not args.case or case["skill_id"] in set(args.case)
    }

    for skill_id in selected_cases:
        run_cmd = [
            sys.executable,
            "scripts/run_full.py",
            "--skill",
            skill_id,
            "--data-root",
            args.data_root,
            "--output-dir",
            args.output_root,
            "--env-config",
            args.env_config,
            "--llm-config",
            args.llm_config,
        ]
        if args.codeql_bin:
            run_cmd.extend(["--codeql-bin", args.codeql_bin])
        if args.base_url:
            run_cmd.extend(["--base-url", args.base_url])
        if args.model:
            run_cmd.extend(["--model", args.model])
        if args.api_key_env:
            run_cmd.extend(["--api-key-env", args.api_key_env])
        if args.n_samples is not None:
            run_cmd.extend(["--n-samples", str(args.n_samples)])
        if args.max_candidates_per_behavior is not None:
            run_cmd.extend(
                [
                    "--max-candidates-per-behavior",
                    str(args.max_candidates_per_behavior),
                ]
            )
        if args.max_alignment_fallbacks_per_step is not None:
            run_cmd.extend(
                [
                    "--max-alignment-fallbacks-per-step",
                    str(args.max_alignment_fallbacks_per_step),
                ]
            )
        if args.max_semantic_event_fallbacks is not None:
            run_cmd.extend(
                [
                    "--max-semantic-event-fallbacks",
                    str(args.max_semantic_event_fallbacks),
                ]
            )
        if args.max_semantic_path_fallbacks is not None:
            run_cmd.extend(
                [
                    "--max-semantic-path-fallbacks",
                    str(args.max_semantic_path_fallbacks),
                ]
            )
        if args.overlap_policy_path:
            run_cmd.extend(["--overlap-policy-path", args.overlap_policy_path])
        _run(run_cmd, repo_root)

        artifact_dir = Path(args.output_root) / skill_id
        _run(
            [
                sys.executable,
                "scripts/run_evaluation.py",
                "--skill",
                skill_id,
                "--artifact-dir",
                str(artifact_dir),
            ],
            repo_root,
        )
        _run(
            [
                sys.executable,
                "scripts/render_markdown_report.py",
                "--skill",
                skill_id,
                "--artifact-dir",
                str(artifact_dir),
                "--max-findings",
                str(args.max_findings),
            ],
            repo_root,
        )

        if args.render_pyvis:
            witnesses_path = artifact_dir / "witnesses.json"
            witness_id = None
            if witnesses_path.is_file():
                witnesses = json.loads(witnesses_path.read_text(encoding="utf-8"))
                if witnesses:
                    witness_id = witnesses[0].get("witness_id")
            pyvis_cmd = [
                sys.executable,
                "scripts/render_pyvis.py",
                "--artifact-dir",
                str(artifact_dir),
                "--skill",
                skill_id,
                "--no-full-graph",
            ]
            if witness_id:
                pyvis_cmd.extend(["--witness-id", str(witness_id)])
            _run(pyvis_cmd, repo_root)


if __name__ == "__main__":
    main()
