#!/usr/bin/env python3
"""CLI entry point for running the full SkillRecon pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from skillrecon.behavior.pipeline import BehaviorObservationPipeline
from skillrecon.contract.pipeline import ContractObservationPipeline
from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    AnalyzerConfig,
    ReconciliationPolicy,
    VotePolicy,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.core.types import (
    Bridge,
    CapabilityEvent,
    Certificate,
    OrchestrationHypothesis,
    ReconciliationJudgment,
    ResourceUse,
    RiskPath,
)
from skillrecon.detect.pipeline import WitnessPipeline
from skillrecon.evaluation.artifacts import (
    FULL_ARTIFACT_KIND,
    remove_status_artifact,
    write_status_artifact,
)
from skillrecon.loader.manifest import build_manifest
from skillrecon.reconcile.pipeline import ReconciliationPipeline
from skillrecon.visualize import render_artifact_dir


def _load_model_list(path: Path, model_cls: type[BaseModel]) -> list[BaseModel]:  # noqa: UP047
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [model_cls.model_validate(item) for item in payload]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the full pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the full SkillRecon pipeline on a single skill"
    )
    parser.add_argument("--skill", required=True, help="Skill directory name")
    parser.add_argument(
        "--data-root",
        default="data/skill_dataset",
        help="Root directory of skill packages",
    )
    parser.add_argument(
        "--output-dir",
        default="derived",
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json used for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json used for LLM defaults",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        help="ICCM sample count; omit to use env_config vote_policy.n_samples",
    )
    parser.add_argument("--codeql-bin", help="CodeQL binary override")
    parser.add_argument("--base-url", help="LLM API base URL")
    parser.add_argument("--model", help="LLM model name")
    parser.add_argument("--api-key-env", help="API key env var or literal key")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
    parser.add_argument(
        "--max-candidates-per-behavior",
        type=int,
        help="Maximum contract candidates kept per behavior object; omit to use config",
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
        "--taxonomy-version",
        default="v2",
        help="Taxonomy version suffix, e.g. v2 -> taxonomy_v2.json",
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt version tag passed to analyzers",
    )
    parser.add_argument(
        "--render-pyvis",
        action="store_true",
        help="Render PyVis HTML views after the witness stage finishes",
    )
    parser.add_argument(
        "--pyvis-output-dir",
        help="Override the visualization output directory (defaults to <artifact-dir>/viz)",
    )
    parser.add_argument(
        "--pyvis-witness-id",
        action="append",
        default=[],
        help="Render only specific witness ids; may be passed multiple times",
    )
    parser.add_argument(
        "--no-pyvis-full-graph",
        action="store_true",
        help="Skip the full G_X page and render witness pages only",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    env_config_path = Path(args.env_config)
    env_config = load_env_config(env_config_path) if env_config_path.is_file() else None
    try:
        llm_config = resolve_llm_config(
            llm_config_path=Path(args.llm_config),
            env_config=env_config,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    base_reconciliation_policy = (
        env_config.reconciliation if env_config is not None else ReconciliationPolicy()
    )
    reconciliation_overrides = {
        key: value
        for key, value in {
            "max_candidates_per_behavior": args.max_candidates_per_behavior,
            "max_alignment_fallbacks_per_step": args.max_alignment_fallbacks_per_step,
            "max_semantic_event_fallbacks": args.max_semantic_event_fallbacks,
            "max_semantic_path_fallbacks": args.max_semantic_path_fallbacks,
            "overlap_policy_path": args.overlap_policy_path,
        }.items()
        if value is not None
    }
    base_vote_policy = env_config.vote_policy if env_config is not None else VotePolicy()
    vote_policy = (
        base_vote_policy
        if args.n_samples is None
        else base_vote_policy.model_copy(update={"n_samples": args.n_samples})
    )
    analyzer_config = AnalyzerConfig(
        llm=llm_config,
        vote_policy=vote_policy,
        reconciliation_policy=base_reconciliation_policy.model_copy(
            update=reconciliation_overrides
        ),
        taxonomy_version=args.taxonomy_version,
        prompt_version=args.prompt_version,
    )

    codeql_bin = args.codeql_bin
    if codeql_bin is None and env_config is not None and env_config.codeql is not None:
        codeql_bin = env_config.codeql.bin

    skill_data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    artifact_dir = output_dir / args.skill
    remove_status_artifact(artifact_dir)

    contract_pipeline = ContractObservationPipeline(
        analyzer_config=analyzer_config,
        skill_data_root=skill_data_root,
        output_dir=output_dir,
    )
    behavior_pipeline = BehaviorObservationPipeline(
        analyzer_config=analyzer_config,
        skill_data_root=skill_data_root,
        output_dir=output_dir,
        codeql_bin=codeql_bin,
    )
    reconcile_pipeline = ReconciliationPipeline(
        analyzer_config=analyzer_config,
        output_dir=output_dir,
    )
    witness_pipeline = WitnessPipeline(
        analyzer_config=analyzer_config,
        output_dir=output_dir,
    )

    try:
        started_at = time.time()
        stage_timings: dict[str, float] = {}

        stage_started = time.time()
        contract_table = contract_pipeline.run(args.skill)
        stage_timings["contract_seconds"] = round(time.time() - stage_started, 3)

        stage_started = time.time()
        behavior_pipeline.run(args.skill)
        stage_timings["behavior_seconds"] = round(time.time() - stage_started, 3)

        stage_started = time.time()
        manifest, _, _ = build_manifest(skill_data_root / args.skill, args.skill)
        events = _load_model_list(artifact_dir / "event_table.json", CapabilityEvent)
        resources = _load_model_list(artifact_dir / "resource_table.json", ResourceUse)
        bridges = _load_model_list(artifact_dir / "bridge_table.json", Bridge)
        orchestrations = _load_model_list(
            artifact_dir / "orchestration_table.json",
            OrchestrationHypothesis,
        )
        paths = _load_model_list(artifact_dir / "path_table.json", RiskPath)
        stage_timings["artifact_load_seconds"] = round(time.time() - stage_started, 3)

        stage_started = time.time()
        contract_table = contract_pipeline.recall_for_behavior(
            args.skill,
            contract_table,
            events,
            resources,
        )
        stage_timings["targeted_recall_seconds"] = round(time.time() - stage_started, 3)

        stage_started = time.time()
        projection_edges = reconcile_pipeline.run(
            skill_id=args.skill,
            contract_table=contract_table,
            events=events,
            resources=resources,
            bridges=bridges,
            orchestrations=orchestrations,
            paths=paths,
            manifest=manifest,
        )
        stage_timings["reconciliation_seconds"] = round(time.time() - stage_started, 3)

        stage_started = time.time()
        judgments = _load_model_list(
            artifact_dir / "judgment_table.json",
            ReconciliationJudgment,
        )
        certificates = _load_model_list(
            artifact_dir / "certificate_table.json",
            Certificate,
        )
        findings, witnesses = witness_pipeline.run(
            skill_id=args.skill,
            clauses=contract_table.clauses,
            judgments=judgments,
            certificates=certificates,
            projection_edges=projection_edges,
            events=events,
            resources=resources,
            paths=paths,
        )
        stage_timings["witness_seconds"] = round(time.time() - stage_started, 3)
        rendered_paths: list[Path] = []
        if args.render_pyvis:
            stage_started = time.time()
            rendered_paths = render_artifact_dir(
                artifact_dir=artifact_dir,
                skill_id=args.skill,
                output_dir=Path(args.pyvis_output_dir) if args.pyvis_output_dir else None,
                witness_ids=set(args.pyvis_witness_id),
                render_full_graph=not args.no_pyvis_full_graph,
            )
            stage_timings["pyvis_seconds"] = round(time.time() - stage_started, 3)
        ended_at = time.time()
        _write_runtime_metrics(
            artifact_dir,
            skill_id=args.skill,
            started_at=started_at,
            ended_at=ended_at,
            stage_timings=stage_timings,
            llm_model=llm_config.model,
            prompt_version=args.prompt_version,
            taxonomy_version=args.taxonomy_version,
            counts={
                "clauses": len(contract_table.clauses),
                "events": len(events),
                "resources": len(resources),
                "reconciliation_edges": len(projection_edges),
                "findings": len(findings),
                "witnesses": len(witnesses),
            },
        )
        write_status_artifact(
            artifact_dir,
            skill_id=args.skill,
            artifact_kind=FULL_ARTIFACT_KIND,
            metadata={
                "model": llm_config.model,
                "prompt_version": args.prompt_version,
                "taxonomy_version": args.taxonomy_version,
            },
        )
    except Exception:
        logging.exception("Pipeline failed")
        sys.exit(1)

    print(
        f"\nRun complete for {args.skill}: "
        f"{len(contract_table.clauses)} clauses, "
        f"{len(events)} events, "
        f"{len(resources)} resources, "
        f"{len(projection_edges)} reconciliation edges, "
        f"{len(findings)} findings, "
        f"{len(witnesses)} witnesses."
    )
    if args.render_pyvis:
        print(f"Rendered {len(rendered_paths)} PyVis HTML files.")


def _write_runtime_metrics(
    artifact_dir: Path,
    *,
    skill_id: str,
    started_at: float,
    ended_at: float,
    stage_timings: dict[str, float],
    llm_model: str,
    prompt_version: str,
    taxonomy_version: str,
    counts: dict[str, int],
) -> None:
    """Persist per-skill runtime data used by scalability/cost diagnostics."""
    payload = {
        "skill_id": skill_id,
        "started_at_unix": started_at,
        "ended_at_unix": ended_at,
        "total_seconds": round(ended_at - started_at, 3),
        "stage_seconds": stage_timings,
        "counts": counts,
        "metadata": {
            "model": llm_model,
            "prompt_version": prompt_version,
            "taxonomy_version": taxonomy_version,
        },
    }
    (artifact_dir / "runtime_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
