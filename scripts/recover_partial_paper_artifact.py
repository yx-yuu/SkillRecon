#!/usr/bin/env python3
"""Recover the post-behavior stages from existing partial paper artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

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
    ContractTable,
    OrchestrationHypothesis,
    PackageManifest,
    ReconciliationJudgment,
    ResourceUse,
    RiskPath,
)
from skillrecon.detect.pipeline import WitnessPipeline
from skillrecon.evaluation.artifacts import (
    FULL_ARTIFACT_KIND,
    REQUIRED_ARTIFACTS,
    artifact_complete,
    load_skill_ids,
    missing_required_artifacts,
    post_behavior_recovery_missing_inputs,
    remove_status_artifact,
    write_status_artifact,
)
from skillrecon.reconcile.pipeline import ReconciliationPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "derived" / "paper500"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover targeted recall, reconciliation, and witness artifacts "
            "from an existing partial paper artifact directory."
        )
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        help="Skill id, e.g. owner/slug; may be passed multiple times",
    )
    parser.add_argument(
        "--skills-file",
        help="Optional newline-delimited skill id list for batch recovery",
    )
    parser.add_argument(
        "--data-root",
        default="data/skill_dataset",
        help=(
            "Optional source root used only if original docs are visible; "
            "document_pack.json is used as the fallback evidence source."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Artifact root containing <owner>/<slug>/ partial artifacts",
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
    parser.add_argument("--n-samples", type=int, help="ICCM sample count override")
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
    parser.add_argument("--taxonomy-version", default="v2", help="Taxonomy version suffix")
    parser.add_argument("--prompt-version", default="v1", help="Prompt version tag")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report recoverable partial artifacts without running LLM or post-behavior stages",
    )
    parser.add_argument("--limit", type=int, help="Maximum number of recoverable skills to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        skill_ids = _resolve_skill_ids(args)
        result = recover_partial_artifacts(
            skill_ids=skill_ids,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            analyzer_config=None if args.dry_run else _resolve_analyzer_config(args),
            prompt_version=args.prompt_version,
            taxonomy_version=args.taxonomy_version,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    except Exception:
        logging.exception("Partial artifact recovery failed")
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("failed", 0):
        sys.exit(1)


def recover_partial_artifacts(
    *,
    skill_ids: list[str],
    data_root: Path,
    output_dir: Path,
    analyzer_config: AnalyzerConfig | None,
    prompt_version: str,
    taxonomy_version: str,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, object]:
    """Recover every selected skill that has enough persisted inputs."""
    rows: list[dict[str, object]] = []
    recoverable_run_count = 0
    for skill_id in skill_ids:
        artifact_dir = output_dir / skill_id
        if artifact_complete(
            artifact_dir,
            expected_status_kind=FULL_ARTIFACT_KIND,
        ):
            rows.append({"skill_id": skill_id, "status": "skipped_complete"})
            continue

        missing_inputs = post_behavior_recovery_missing_inputs(artifact_dir)
        if missing_inputs:
            rows.append(
                {
                    "skill_id": skill_id,
                    "status": "skipped_missing_inputs",
                    "missing_inputs": missing_inputs,
                    "missing_required_artifacts": missing_required_artifacts(
                        artifact_dir,
                        required_artifacts=REQUIRED_ARTIFACTS,
                        expected_status_kind=FULL_ARTIFACT_KIND,
                    ),
                }
            )
            continue

        if limit is not None and recoverable_run_count >= limit:
            rows.append({"skill_id": skill_id, "status": "skipped_limit"})
            continue

        recoverable_run_count += 1
        if dry_run:
            rows.append(
                {
                    "skill_id": skill_id,
                    "status": "dry_run_recoverable",
                    "missing_required_artifacts": missing_required_artifacts(
                        artifact_dir,
                        required_artifacts=REQUIRED_ARTIFACTS,
                        expected_status_kind=FULL_ARTIFACT_KIND,
                    ),
                }
            )
            continue

        if analyzer_config is None:
            raise ValueError("analyzer_config is required unless dry_run=True")
        try:
            rows.append(
                recover_partial_artifact(
                    skill_id=skill_id,
                    data_root=data_root,
                    output_dir=output_dir,
                    analyzer_config=analyzer_config,
                    status_metadata={
                        "model": analyzer_config.llm.model,
                        "prompt_version": prompt_version,
                        "taxonomy_version": taxonomy_version,
                        "recovered_from": "post_behavior_partial_artifacts",
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - batch recovery should report all rows.
            logging.exception("Failed to recover %s", skill_id)
            rows.append(
                {
                    "skill_id": skill_id,
                    "status": "failed",
                    "error": str(exc),
                    "missing_required_artifacts": missing_required_artifacts(
                        artifact_dir,
                        required_artifacts=REQUIRED_ARTIFACTS,
                        expected_status_kind=FULL_ARTIFACT_KIND,
                    ),
                }
            )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "status": "dry_run" if dry_run else "ok",
        "selected": len(skill_ids),
        "recovered": status_counts.get("ok", 0),
        "failed": status_counts.get("failed", 0),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
    }


def recover_partial_artifact(
    *,
    skill_id: str,
    data_root: Path,
    output_dir: Path,
    analyzer_config: AnalyzerConfig,
    status_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run post-behavior stages from persisted intermediate artifacts."""
    artifact_dir = output_dir / skill_id
    remove_status_artifact(artifact_dir)

    manifest = PackageManifest.model_validate_json(
        (artifact_dir / "package_manifest.json").read_text(encoding="utf-8")
    )
    contract_table = ContractTable.model_validate_json(
        (artifact_dir / "contract_table.json").read_text(encoding="utf-8")
    )
    events = _load_model_list(artifact_dir / "event_table.json", CapabilityEvent)
    resources = _load_model_list(artifact_dir / "resource_table.json", ResourceUse)
    bridges = _load_model_list(artifact_dir / "bridge_table.json", Bridge)
    orchestrations = _load_model_list(
        artifact_dir / "orchestration_table.json",
        OrchestrationHypothesis,
    )
    paths = _load_model_list(artifact_dir / "path_table.json", RiskPath)

    contract_pipeline = ContractObservationPipeline(
        analyzer_config=analyzer_config,
        skill_data_root=data_root,
        output_dir=output_dir,
    )
    contract_table = contract_pipeline.recall_for_behavior(
        skill_id,
        contract_table,
        events,
        resources,
    )

    reconcile_pipeline = ReconciliationPipeline(
        analyzer_config=analyzer_config,
        output_dir=output_dir,
    )
    projection_edges = reconcile_pipeline.run(
        skill_id=skill_id,
        contract_table=contract_table,
        events=events,
        resources=resources,
        bridges=bridges,
        orchestrations=orchestrations,
        paths=paths,
        manifest=manifest,
    )
    judgments = _load_model_list(
        artifact_dir / "judgment_table.json",
        ReconciliationJudgment,
    )
    certificates = _load_model_list(
        artifact_dir / "certificate_table.json",
        Certificate,
    )

    witness_pipeline = WitnessPipeline(
        analyzer_config=analyzer_config,
        output_dir=output_dir,
    )
    findings, witnesses = witness_pipeline.run(
        skill_id=skill_id,
        clauses=contract_table.clauses,
        judgments=judgments,
        certificates=certificates,
        projection_edges=projection_edges,
        events=events,
        resources=resources,
        paths=paths,
    )
    write_status_artifact(
        artifact_dir,
        skill_id=skill_id,
        artifact_kind=FULL_ARTIFACT_KIND,
        metadata=status_metadata,
    )
    return {
        "status": "ok",
        "skill_id": skill_id,
        "clauses": len(contract_table.clauses),
        "events": len(events),
        "resources": len(resources),
        "reconciliation_edges": len(projection_edges),
        "findings": len(findings),
        "witnesses": len(witnesses),
        "missing_required_artifacts": missing_required_artifacts(
            artifact_dir,
            required_artifacts=REQUIRED_ARTIFACTS,
            expected_status_kind=FULL_ARTIFACT_KIND,
        ),
    }


def _resolve_analyzer_config(args: argparse.Namespace) -> AnalyzerConfig:
    env_config_path = Path(args.env_config)
    env_config = load_env_config(env_config_path) if env_config_path.is_file() else None
    llm_config = resolve_llm_config(
        llm_config_path=Path(args.llm_config),
        env_config=env_config,
        base_url=args.base_url,
        model=args.model,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
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
    return AnalyzerConfig(
        llm=llm_config,
        vote_policy=vote_policy,
        reconciliation_policy=base_reconciliation_policy.model_copy(
            update=reconciliation_overrides
        ),
        taxonomy_version=args.taxonomy_version,
        prompt_version=args.prompt_version,
    )


def _resolve_skill_ids(args: argparse.Namespace) -> list[str]:
    skill_ids: list[str] = []
    if args.skills_file:
        skill_ids.extend(load_skill_ids(Path(args.skills_file)))
    skill_ids.extend(args.skill)
    seen: set[str] = set()
    deduped: list[str] = []
    for skill_id in skill_ids:
        if skill_id in seen:
            continue
        seen.add(skill_id)
        deduped.append(skill_id)
    if not deduped:
        raise ValueError("provide --skill or --skills-file")
    return deduped


def _load_model_list(path: Path, model_cls: type[BaseModel]) -> list[BaseModel]:  # noqa: UP047
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [model_cls.model_validate(item) for item in payload]


if __name__ == "__main__":
    main()
