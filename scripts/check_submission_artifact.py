#!/usr/bin/env python3
"""Check that the submitted artifact has complete single-case and full-run paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillrecon.evaluation.artifacts import (
    FULL_ARTIFACT_KIND,
    REQUIRED_ARTIFACTS,
    artifact_coverage,
    load_skill_ids,
    missing_required_artifacts,
)
from skillrecon.evaluation.datasets import (
    compare_gold_labels_to_paper_sample,
    load_baseline_prediction_records,
    load_gold_label_records,
    load_paper_sample_records,
)
from skillrecon.evaluation.tables import render_all_tables


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_METHOD_SCRIPTS = (
    "scripts/run_full.py",
    "scripts/run_evaluation.py",
    "scripts/render_markdown_report.py",
    "scripts/render_pyvis.py",
    "scripts/run_reviewer_cases.py",
)
REQUIRED_FULL_EXPERIMENT_SCRIPTS = (
    "scripts/run_paper_artifacts.py",
    "scripts/manage_paper_artifact_job.py",
    "scripts/recover_partial_paper_artifact.py",
    "scripts/build_ablation_artifacts.py",
    "scripts/run_paper_experiment_bundle.py",
    "scripts/run_experiments.py",
    "scripts/compute_rq2_statistics.py",
    "scripts/export_evaluation_appendix.py",
    "scripts/sync_evaluation_outputs_to_paper.py",
    "scripts/run_external_baselines.py",
    "scripts/merge_baseline_predictions.py",
    "scripts/render_paper_chapter4_figures.py",
    "scripts/build_llm_replacement_plan.py",
)
REQUIRED_CONFIGS = (
    "experiments/configs/env_config.json",
    "experiments/configs/llm_config.json",
    "experiments/configs/taxonomy_v2.json",
    "experiments/configs/overlap_policy_v1.json",
    "experiments/configs/reviewer_cases_v1.json",
    "experiments/configs/smoke_eval_v1.json",
    "experiments/configs/external_baselines_v1.json",
)
EXPECTED_TABLES = {
    "t1_baselines.tex",
    "t2_rq1_clause.tex",
    "t3_rq1_edges.tex",
    "t4_rq2_slices.tex",
    "t5_rq2_types.tex",
    "t6_rq2_ablation.tex",
    "t7_rq3_discovery.tex",
    "t8_rq4_witness.tex",
}
EVALUATION_LABELS = {
    "tables": {
        "tab:baselines",
        "tab:rq1-clause",
        "tab:rq1-edges",
        "tab:rq2-slices",
        "tab:rq2-types",
        "tab:rq2-ablation",
        "tab:rq3",
        "tab:rq4",
    },
    "figures": {
        "fig:cost",
        "fig:error",
        "fig:language",
        "fig:sensitivity",
        "fig:case-study",
    },
}
METHOD_COVERAGE = {
    "package_scan_and_document_closure": {
        "paths": (
            "src/skillrecon/loader/scanner.py",
            "src/skillrecon/loader/manifest.py",
            "src/skillrecon/loader/reference.py",
            "src/skillrecon/loader/inline.py",
            "src/skillrecon/loader/path_resolver.py",
        ),
        "tokens": {
            "src/skillrecon/loader/manifest.py": ("build_manifest",),
            "src/skillrecon/loader/reference.py": ("build_document_closure",),
            "src/skillrecon/loader/inline.py": ("extract_synthetic_code_units",),
        },
    },
    "intent_contract_graph": {
        "paths": (
            "src/skillrecon/contract/steps.py",
            "src/skillrecon/contract/deterministic.py",
            "src/skillrecon/contract/iccm.py",
            "src/skillrecon/contract/voting.py",
            "src/skillrecon/contract/normalize.py",
            "src/skillrecon/contract/classify.py",
            "src/skillrecon/contract/recall.py",
            "src/skillrecon/contract/pipeline.py",
        ),
        "tokens": {
            "src/skillrecon/contract/pipeline.py": (
                "ContractObservationPipeline",
                "recall_for_behavior",
            ),
            "src/skillrecon/contract/voting.py": ("aggregate_samples",),
            "src/skillrecon/contract/classify.py": ("classify_clause_role",),
            "src/skillrecon/contract/recall.py": ("build_recall_focus_context",),
        },
    },
    "behavior_dependence_graph": {
        "paths": (
            "src/skillrecon/behavior/codeql.py",
            "src/skillrecon/behavior/normalize.py",
            "src/skillrecon/behavior/bash.py",
            "src/skillrecon/behavior/instruction.py",
            "src/skillrecon/behavior/path_graph.py",
            "src/skillrecon/behavior/pipeline.py",
        ),
        "tokens": {
            "src/skillrecon/behavior/pipeline.py": ("BehaviorObservationPipeline",),
            "src/skillrecon/behavior/codeql.py": ("run_codeql_analysis",),
            "src/skillrecon/behavior/path_graph.py": ("build_risk_paths",),
        },
    },
    "reconciliation_predicates_and_certificates": {
        "paths": (
            "src/skillrecon/reconcile/candidate.py",
            "src/skillrecon/reconcile/predicate.py",
            "src/skillrecon/reconcile/derivation.py",
            "src/skillrecon/reconcile/pipeline.py",
        ),
        "tokens": {
            "src/skillrecon/reconcile/candidate.py": ("generate_candidates",),
            "src/skillrecon/reconcile/predicate.py": (
                "capability_overlaps",
                "resource_compatible",
                "scope_satisfied",
                "prohibition_conflict",
                "execution_route_justified",
            ),
            "src/skillrecon/reconcile/derivation.py": (
                "materialize_reconciliation",
                "_materialize_path_justifications",
            ),
        },
    },
    "findings_witnesses_and_reports": {
        "paths": (
            "src/skillrecon/detect/findings.py",
            "src/skillrecon/detect/witness.py",
            "src/skillrecon/detect/pipeline.py",
            "src/skillrecon/visualize/pyvis_renderer.py",
            "scripts/render_markdown_report.py",
            "scripts/render_pyvis.py",
        ),
        "tokens": {
            "src/skillrecon/detect/witness.py": (
                "assemble_witnesses",
                "validate_witness",
            ),
            "src/skillrecon/detect/findings.py": ("materialize_findings",),
            "src/skillrecon/visualize/pyvis_renderer.py": ("render_artifact_dir",),
        },
    },
}
EVALUATION_CODE_COVERAGE = {
    "rq1_contract_and_edge_validity": {
        "paths": (
            "scripts/build_rq1_gold_benchmark.py",
            "scripts/build_rq1_edge_validity_benchmark.py",
            "src/skillrecon/evaluation/metrics.py",
            "src/skillrecon/evaluation/runner.py",
            "src/skillrecon/evaluation/tables.py",
        ),
        "tokens": {
            "src/skillrecon/evaluation/metrics.py": (
                "compute_clause_operator_metrics",
                "compute_false_authorization_rate",
                "compute_edge_validity_by_type",
            ),
            "src/skillrecon/evaluation/runner.py": ("_build_rq1_results",),
        },
    },
    "rq2_detection_baselines_ablations_and_stats": {
        "paths": (
            "src/skillrecon/evaluation/baselines.py",
            "src/skillrecon/evaluation/external_scanners.py",
            "src/skillrecon/evaluation/ablation.py",
            "src/skillrecon/evaluation/stats.py",
            "scripts/run_external_baselines.py",
            "scripts/merge_baseline_predictions.py",
            "scripts/build_ablation_artifacts.py",
            "scripts/compute_rq2_statistics.py",
        ),
        "tokens": {
            "src/skillrecon/evaluation/baselines.py": (
                "build_rule_based_scanner_report",
                "build_llm_judge_report",
                "build_capability_lattice_report",
                "build_doc_code_consistency_report",
                "build_spec_containment_report",
                "build_instruction_constraint_report",
            ),
            "src/skillrecon/evaluation/external_scanners.py": (
                "external_scanner_payload_to_prediction",
                "external_scanner_payload_to_report",
                "baseline_skillfortify",
                "baseline_cisco_skill_scanner",
                "baseline_skillspector",
            ),
            "src/skillrecon/evaluation/stats.py": (
                "bootstrap_resample_ci",
                "mcnemar_exact",
                "holm_correction",
            ),
            "scripts/compute_rq2_statistics.py": ("positive_label", "by_slice"),
        },
    },
    "rq3_low_slice_discovery": {
        "paths": (
            "src/skillrecon/evaluation/metrics.py",
            "src/skillrecon/evaluation/runner.py",
            "src/skillrecon/evaluation/tables.py",
        ),
        "tokens": {
            "src/skillrecon/evaluation/metrics.py": (
                "compute_discovery_yield",
                "confirmed_exposures",
            ),
            "src/skillrecon/evaluation/tables.py": ("render_rq3_discovery_table",),
        },
    },
    "rq4_witness_fidelity": {
        "paths": (
            "src/skillrecon/evaluation/runner.py",
            "src/skillrecon/detect/witness.py",
            "src/skillrecon/evaluation/tables.py",
        ),
        "tokens": {
            "src/skillrecon/evaluation/runner.py": ("_compute_witness_fidelity",),
            "src/skillrecon/detect/witness.py": ("validate_witness",),
            "src/skillrecon/evaluation/tables.py": ("render_rq4_witness_table",),
        },
    },
    "paper_tables_figures_and_appendix": {
        "paths": (
            "src/skillrecon/evaluation/tables.py",
            "src/skillrecon/evaluation/figures.py",
            "src/skillrecon/evaluation/chapter4_figures.py",
            "scripts/export_evaluation_appendix.py",
            "scripts/sync_evaluation_outputs_to_paper.py",
            "scripts/select_benign_case_studies.py",
            "scripts/render_paper_evaluation_smoke.py",
            "scripts/render_paper_chapter4_figures.py",
        ),
        "tokens": {
            "src/skillrecon/evaluation/tables.py": ("render_all_tables",),
            "src/skillrecon/evaluation/figures.py": ("render_all_figures",),
            "src/skillrecon/evaluation/chapter4_figures.py": (
                "render_chapter4_figures",
            ),
            "scripts/render_paper_evaluation_smoke.py": ("--experiment-json",),
            "scripts/render_paper_chapter4_figures.py": (
                "--figure-spec",
                "fig:sensitivity",
            ),
        },
    },
    "cost_generalization_and_runtime": {
        "paths": (
            "scripts/run_full.py",
            "src/skillrecon/evaluation/extended.py",
            "src/skillrecon/evaluation/figures.py",
        ),
        "tokens": {
            "scripts/run_full.py": ("runtime_metrics.json", "_write_runtime_metrics"),
            "src/skillrecon/evaluation/extended.py": (
                "build_generalization_results",
                "build_robustness_and_cost_results",
                "_runtime_summary",
            ),
            "src/skillrecon/evaluation/figures.py": (
                "_render_generalization",
                "_render_robustness_cost",
            ),
        },
    },
}
PAPER_EMBEDDED_FIGURES = {
    "fig:cost": {
        "support": "runtime_metrics.json + rq6_robustness_cost summarize cost proxies",
        "generator": "scripts/render_paper_chapter4_figures.py --figure fig:cost",
    },
    "fig:error": {
        "support": "RQ1 metrics plus optional false-authorization/backbone figure spec",
        "generator": "scripts/render_paper_chapter4_figures.py --figure fig:error",
    },
    "fig:language": {
        "support": "rq5_generalization plus optional per-baseline language-profile figure spec",
        "generator": "scripts/render_paper_chapter4_figures.py --figure fig:language",
    },
    "fig:sensitivity": {
        "support": "hyperparameter-sensitivity figure spec generated from sweep outputs",
        "generator": "scripts/render_paper_chapter4_figures.py --figure fig:sensitivity",
    },
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    data: dict[str, Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the paper artifact has runnable single-case and "
            "full-experiment code paths. This command does not run LLM or CodeQL."
        )
    )
    parser.add_argument("--paper-dir", default="paper")
    parser.add_argument("--data-root", default="data/skill_dataset")
    parser.add_argument(
        "--reviewer-cases",
        default="experiments/configs/reviewer_cases_v1.json",
    )
    parser.add_argument(
        "--paper-dataset",
        default="data/evaluation/skill_paper500_dataset",
    )
    parser.add_argument(
        "--single-artifact-root",
        help="Optional existing reviewer/single-case artifact root to check.",
    )
    parser.add_argument(
        "--paper-artifact-root",
        help="Optional existing full paper artifact root to summarize.",
    )
    parser.add_argument("--json-out", help="Optional JSON output path")
    args = parser.parse_args()

    checks = run_checks(args)
    payload = summarize_checks(checks)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["errors"]:
        raise SystemExit(2)


def run_checks(args: argparse.Namespace) -> list[Check]:
    paper_dir = Path(args.paper_dir)
    data_root = Path(args.data_root)
    reviewer_cases = Path(args.reviewer_cases)
    paper_dataset = Path(args.paper_dataset)
    checks: list[Check] = []
    checks.extend(_check_required_files())
    checks.append(_check_claim_coverage("method_design_coverage", METHOD_COVERAGE))
    checks.append(_check_claim_coverage("evaluation_code_coverage", EVALUATION_CODE_COVERAGE))
    checks.append(_check_reviewer_cases(reviewer_cases, data_root))
    checks.append(_check_paper_dataset(paper_dataset))
    checks.append(_check_experiment_bundle_entry())
    checks.append(_check_table_renderers())
    checks.append(_check_paper_evaluation_labels(paper_dir))
    checks.append(_check_embedded_figure_generators())
    if args.single_artifact_root:
        checks.append(_check_single_artifacts(Path(args.single_artifact_root), reviewer_cases))
    if args.paper_artifact_root:
        checks.append(_check_paper_artifacts(Path(args.paper_artifact_root), paper_dataset))
    return checks


def summarize_checks(checks: list[Check]) -> dict[str, object]:
    counts = Counter(check.status for check in checks)
    return {
        "status": "error" if counts["error"] else "ok",
        "errors": counts["error"],
        "warnings": counts["warning"],
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "detail": check.detail,
                "data": check.data or {},
            }
            for check in checks
        ],
    }


def _check_required_files() -> list[Check]:
    checks: list[Check] = []
    groups = {
        "method_scripts": REQUIRED_METHOD_SCRIPTS,
        "full_experiment_scripts": REQUIRED_FULL_EXPERIMENT_SCRIPTS,
        "configs": REQUIRED_CONFIGS,
    }
    for name, files in groups.items():
        missing = [path for path in files if not (PROJECT_ROOT / path).is_file()]
        checks.append(
            Check(
                name=name,
                status="error" if missing else "ok",
                detail="missing required files" if missing else "all required files exist",
                data={"missing": missing, "checked": list(files)},
            )
        )
    return checks


def _check_claim_coverage(name: str, coverage: dict[str, dict[str, object]]) -> Check:
    missing_paths: dict[str, list[str]] = {}
    missing_tokens: dict[str, dict[str, list[str]]] = {}
    for claim, spec in coverage.items():
        paths = tuple(str(path) for path in spec.get("paths", ()))
        claim_missing_paths = [path for path in paths if not (PROJECT_ROOT / path).is_file()]
        if claim_missing_paths:
            missing_paths[claim] = claim_missing_paths
        token_spec = spec.get("tokens", {})
        if not isinstance(token_spec, dict):
            continue
        for relative_path, tokens in token_spec.items():
            path = PROJECT_ROOT / str(relative_path)
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            absent = [str(token) for token in tokens if str(token) not in text]
            if absent:
                missing_tokens.setdefault(claim, {})[str(relative_path)] = absent

    status = "error" if missing_paths or missing_tokens else "ok"
    return Check(
        name,
        status,
        "paper claims have concrete code coverage"
        if status == "ok"
        else "paper claims are missing implementation coverage",
        data={
            "checked_claims": sorted(coverage),
            "missing_paths": missing_paths,
            "missing_tokens": missing_tokens,
        },
    )


def _check_reviewer_cases(manifest_path: Path, data_root: Path) -> Check:
    try:
        cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Check("reviewer_cases", "error", str(exc))
    if not isinstance(cases, list) or not cases:
        return Check("reviewer_cases", "error", "reviewer case manifest is empty")

    missing_sources = []
    outcomes = Counter()
    missing_fields = []
    for case in cases:
        if not isinstance(case, dict):
            missing_fields.append("<non-object>")
            continue
        skill_id = str(case.get("skill_id") or "")
        source_relpath = str(case.get("source_relpath") or skill_id)
        outcome = str(case.get("expected_outcome") or "")
        outcomes[outcome] += 1
        if not skill_id or not outcome or not case.get("recommended_artifact"):
            missing_fields.append(skill_id or "<missing skill_id>")
        if not (data_root / source_relpath).is_dir():
            missing_sources.append(source_relpath)

    required_outcomes = {"violation", "exposure_only", "benign"}
    missing_outcomes = sorted(required_outcomes - set(outcomes))
    problems = {
        "missing_sources": missing_sources,
        "missing_fields": missing_fields,
        "missing_outcomes": missing_outcomes,
    }
    has_error = any(problems.values())
    return Check(
        "reviewer_cases",
        "error" if has_error else "ok",
        (
            "reviewer single-case bundle is incomplete"
            if has_error
            else "reviewer single-case bundle covers violation, exposure-only, and benign cases"
        ),
        data={
            "case_count": len(cases),
            "outcomes": dict(outcomes),
            **problems,
        },
    )


def _check_paper_dataset(dataset_root: Path) -> Check:
    required = [
        dataset_root / "all_skills.txt",
        dataset_root / "gold_labels.jsonl",
        dataset_root / "baseline_predictions" / "openclaw.jsonl",
        dataset_root / "high" / "sample_index.jsonl",
        dataset_root / "medium" / "sample_index.jsonl",
        dataset_root / "low" / "sample_index.jsonl",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return Check(
            "paper_dataset",
            "error",
            "paper dataset is missing required files",
            data={"missing": missing},
        )
    try:
        sample_records = load_paper_sample_records(dataset_root)
        gold_records = load_gold_label_records(dataset_root / "gold_labels.jsonl")
        openclaw_records = load_baseline_prediction_records(
            dataset_root / "baseline_predictions" / "openclaw.jsonl"
        )
        comparison = compare_gold_labels_to_paper_sample(gold_records, dataset_root)
        skill_ids = load_skill_ids(dataset_root / "all_skills.txt")
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        return Check("paper_dataset", "error", str(exc))

    expected_ids = {record.skill_id for records in sample_records.values() for record in records}
    openclaw_ids = {record.skill_id for record in openclaw_records}
    skill_file_ids = set(skill_ids)
    problems = {
        "gold_missing": comparison["missing"],
        "gold_extra": comparison["extra"],
        "openclaw_missing": sorted(expected_ids - openclaw_ids),
        "openclaw_extra": sorted(openclaw_ids - expected_ids),
        "skills_file_missing": sorted(expected_ids - skill_file_ids),
        "skills_file_extra": sorted(skill_file_ids - expected_ids),
    }
    has_error = any(problems.values())
    return Check(
        "paper_dataset",
        "error" if has_error else "ok",
        "paper dataset indexes, gold labels, and OpenClaw baseline align"
        if not has_error
        else "paper dataset has inconsistent indexes or labels",
        data={
            "slice_counts": {name: len(records) for name, records in sample_records.items()},
            "gold_count": len(gold_records),
            "openclaw_count": len(openclaw_records),
            "skills_file_count": len(skill_ids),
            **{key: len(value) for key, value in problems.items()},
        },
    )


def _check_experiment_bundle_entry() -> Check:
    module = _load_script_module("scripts/run_paper_experiment_bundle.py")
    try:
        parser = module.build_arg_parser()
        args = parser.parse_args(["--dry-run"])
        paths = module.resolve_paths(args)
        commands = module.build_commands(args, paths)
    except Exception as exc:
        return Check("experiment_bundle_entry", "error", str(exc))
    required_steps = {"ablation", "experiments", "rq2_stats", "appendix"}
    missing = sorted(required_steps - set(commands))
    return Check(
        "experiment_bundle_entry",
        "error" if missing else "ok",
        "full experiment bundle plans ablation, experiments, stats, and appendix"
        if not missing
        else "full experiment bundle is missing planned steps",
        data={"planned_steps": sorted(commands), "missing": missing, "output_dir": str(paths.output_dir)},
    )


def _check_table_renderers() -> Check:
    try:
        tables = render_all_tables(
            {
                "rq1": {},
                "rq2": {
                    "skillrecon": {"by_slice": {}, "medium_disputed": {}},
                    "baseline_openclaw": {"by_slice": {}},
                },
                "rq2_meta": {},
                "rq3": {"skillrecon": {}},
                "rq4": {"skillrecon": {}},
            }
        )
    except Exception as exc:
        return Check("table_renderers", "error", str(exc))
    missing = sorted(EXPECTED_TABLES - set(tables))
    extra = sorted(set(tables) - EXPECTED_TABLES)
    return Check(
        "table_renderers",
        "error" if missing else "ok",
        "paper-facing table renderers cover RQ1-RQ4"
        if not missing
        else "paper-facing table renderers are incomplete",
        data={"rendered": sorted(tables), "missing": missing, "extra": extra},
    )


def _check_paper_evaluation_labels(paper_dir: Path) -> Check:
    main_tex = paper_dir / "main.tex"
    if not main_tex.is_file():
        return Check("paper_evaluation_labels", "error", f"missing {main_tex}")
    text = main_tex.read_text(encoding="utf-8")
    evaluation_text = _evaluation_section(text)
    missing_tables = sorted(label for label in EVALUATION_LABELS["tables"] if label not in evaluation_text)
    missing_figures = sorted(label for label in EVALUATION_LABELS["figures"] if label not in evaluation_text)
    static_missing = []
    case_study_path = paper_dir / "figures" / "fig-case-study.png"
    if not case_study_path.is_file():
        static_missing.append(str(case_study_path))
    status = "error" if missing_tables or missing_figures or static_missing else "ok"
    return Check(
        "paper_evaluation_labels",
        status,
        "paper Evaluation labels match expected table/figure surface"
        if status == "ok"
        else "paper Evaluation label surface is incomplete",
        data={
            "missing_tables": missing_tables,
            "missing_figures": missing_figures,
            "static_missing": static_missing,
        },
    )


def _check_embedded_figure_generators() -> Check:
    missing_generators = {
        label: spec["support"]
        for label, spec in PAPER_EMBEDDED_FIGURES.items()
        if spec.get("generator") is None
    }
    return Check(
        "embedded_figure_generators",
        "warning" if missing_generators else "ok",
        (
            "paper has embedded TikZ Evaluation figures without exact standalone generators"
            if missing_generators
            else "paper embedded Evaluation figures have standalone generators"
        ),
        data={"embedded_without_generator": missing_generators},
    )


def _check_single_artifacts(artifact_root: Path, manifest_path: Path) -> Check:
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))
    skill_ids = [str(case["skill_id"]) for case in cases if isinstance(case, dict) and case.get("skill_id")]
    incomplete: dict[str, list[str]] = {}
    missing_reports = []
    for skill_id in skill_ids:
        artifact_dir = artifact_root / skill_id
        missing = missing_required_artifacts(
            artifact_dir,
            expected_status_kind=FULL_ARTIFACT_KIND,
        )
        if missing:
            incomplete[skill_id] = missing
        if not (artifact_dir / "report.json").is_file() or not (artifact_dir / "review_report.md").is_file():
            missing_reports.append(skill_id)
    status = "error" if incomplete or missing_reports else "ok"
    return Check(
        "single_artifacts",
        status,
        "existing single-case artifacts are complete"
        if status == "ok"
        else "existing single-case artifacts are incomplete",
        data={
            "root": str(artifact_root),
            "expected": len(skill_ids),
            "incomplete": incomplete,
            "missing_reports": missing_reports,
        },
    )


def _check_paper_artifacts(artifact_root: Path, dataset_root: Path) -> Check:
    try:
        skill_ids = load_skill_ids(dataset_root / "all_skills.txt")
        coverage = artifact_coverage(
            skill_ids=skill_ids,
            artifact_root=artifact_root,
            required_artifacts=REQUIRED_ARTIFACTS,
            expected_status_kind=FULL_ARTIFACT_KIND,
        )
    except Exception as exc:
        return Check("paper_artifacts", "error", str(exc))
    complete = int(coverage["complete"])
    expected = int(coverage["expected"])
    status = "ok" if complete == expected else "warning"
    return Check(
        "paper_artifacts",
        status,
        "full paper artifacts are complete"
        if status == "ok"
        else "full paper artifacts are not complete yet; code path exists but full run is pending",
        data={
            "root": str(artifact_root),
            "expected": expected,
            "complete": complete,
            "missing_dirs": len(coverage["missing_dirs"]),
            "incomplete": len(coverage["incomplete"]),
        },
    )


def _load_script_module(relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _evaluation_section(text: str) -> str:
    start = text.find(r"\section{Evaluation}")
    if start < 0:
        start = text.find(r"\subsection{Experimental Setup}")
    end = text.find(r"\section{Discussion}", start)
    if end < 0:
        end = len(text)
    return text[start:end]


if __name__ == "__main__":
    main()
