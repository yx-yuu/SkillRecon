"""Ablation artifact generation from existing SkillRecon intermediate artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from skillrecon.contract.classify import classify_clause_roles
from skillrecon.contract.deterministic import extract_from_frontmatter
from skillrecon.contract.parser import parse_frontmatter_fields, parse_yaml_frontmatter
from skillrecon.contract.pipeline import _build_contract_graph_artifact, _ground_evidence
from skillrecon.contract.recall import merge_recall_clauses
from skillrecon.contract.voting import aggregate_samples, build_contract_table, detect_clause_conflicts
from skillrecon.core.config import AnalyzerConfig
from skillrecon.core.types import (
    Bridge,
    CapabilityEvent,
    Certificate,
    Clause,
    ClauseSample,
    ContractTable,
    DocumentPack,
    OrchestrationHypothesis,
    PackageManifest,
    ReconciliationJudgment,
    ResourceUse,
    RiskPath,
    Step,
    StepOrderEdge,
)
from skillrecon.detect.pipeline import WitnessPipeline
from skillrecon.evaluation.artifacts import (
    ABLATION_ARTIFACT_KIND,
    artifact_complete,
    remove_status_artifact,
    write_status_artifact,
)
from skillrecon.reconcile.pipeline import ReconciliationPipeline

ABLATION_SYSTEM_IDS = (
    "ablation_no_iccm",
    "ablation_no_scope_constraints",
    "ablation_no_composition_analysis",
    "ablation_no_authorization_guard",
)


def materialize_ablation_artifacts(
    *,
    skill_ids: list[str],
    data_root: Path,
    base_artifact_root: Path,
    output_root: Path,
    analyzer_config: AnalyzerConfig,
    skip_existing: bool = False,
) -> dict[str, Path]:
    """Generate all configured ablation artifact roots from existing artifacts."""
    system_roots = {
        system_id: output_root / system_id
        for system_id in ABLATION_SYSTEM_IDS
    }
    for skill_id in skill_ids:
        base_artifact_dir = base_artifact_root / skill_id
        if not base_artifact_dir.exists():
            continue
        for system_id, system_root in system_roots.items():
            variant_dir = system_root / skill_id
            if skip_existing and artifact_complete(
                variant_dir,
                expected_status_kind=ABLATION_ARTIFACT_KIND,
            ):
                continue
            _materialize_single_ablation(
                system_id=system_id,
                skill_id=skill_id,
                skill_path=data_root / skill_id,
                base_artifact_dir=base_artifact_dir,
                output_root=system_root,
                analyzer_config=analyzer_config,
            )
    return system_roots


def _materialize_single_ablation(
    *,
    system_id: str,
    skill_id: str,
    skill_path: Path,
    base_artifact_dir: Path,
    output_root: Path,
    analyzer_config: AnalyzerConfig,
) -> None:
    variant_dir = output_root / skill_id
    if variant_dir.exists():
        shutil.rmtree(variant_dir)
    shutil.copytree(base_artifact_dir, variant_dir)
    remove_status_artifact(variant_dir)

    manifest = PackageManifest.model_validate_json(
        (base_artifact_dir / "package_manifest.json").read_text(encoding="utf-8")
    )
    document_pack = DocumentPack.model_validate_json(
        (base_artifact_dir / "document_pack.json").read_text(encoding="utf-8")
    )
    steps = _load_model_list(base_artifact_dir / "step_table.json", Step)
    step_edges = _load_model_list(base_artifact_dir / "step_edges.json", StepOrderEdge)
    events = _load_model_list(base_artifact_dir / "event_table.json", CapabilityEvent)
    resources = _load_model_list(base_artifact_dir / "resource_table.json", ResourceUse)
    bridges = _load_model_list(base_artifact_dir / "bridge_table.json", Bridge)
    orchestrations = _load_model_list(
        base_artifact_dir / "orchestration_table.json",
        OrchestrationHypothesis,
    )
    paths = _load_model_list(base_artifact_dir / "path_table.json", RiskPath)

    if system_id == "ablation_no_iccm":
        contract_table = _build_deterministic_only_contract_table(
            skill_id=skill_id,
            skill_path=skill_path,
            manifest=manifest,
            steps=steps,
            step_edges=step_edges,
            unresolved_refs=document_pack.unresolved_references,
        )
        ablation_paths = paths
    elif system_id == "ablation_no_scope_constraints":
        contract_table = _strip_scope_constraints(
            ContractTable.model_validate_json(
                (base_artifact_dir / "contract_table.json").read_text(encoding="utf-8")
            )
        )
        ablation_paths = paths
    elif system_id == "ablation_no_composition_analysis":
        contract_table = ContractTable.model_validate_json(
            (base_artifact_dir / "contract_table.json").read_text(encoding="utf-8")
        )
        ablation_paths = []
    elif system_id == "ablation_no_authorization_guard":
        contract_table = _build_unguarded_contract_table(
            skill_id=skill_id,
            skill_path=skill_path,
            manifest=manifest,
            steps=steps,
            step_edges=step_edges,
            unresolved_refs=document_pack.unresolved_references,
            events=events,
            raw_samples_path=base_artifact_dir / "raw_clause_samples.json",
            raw_recall_samples_path=base_artifact_dir / "recall_raw_clause_samples.json",
        )
        ablation_paths = paths
    else:
        raise ValueError(f"Unsupported ablation system_id: {system_id}")

    _write_contract_artifacts(
        artifact_dir=variant_dir,
        contract_table=contract_table,
    )
    if system_id == "ablation_no_composition_analysis":
        _write_json(
            variant_dir / "path_table.json",
            [],
        )

    reconcile_pipeline = ReconciliationPipeline(
        analyzer_config=analyzer_config,
        output_dir=output_root,
    )
    projection_edges = reconcile_pipeline.run(
        skill_id=skill_id,
        contract_table=contract_table,
        events=events,
        resources=resources,
        bridges=bridges,
        orchestrations=orchestrations,
        paths=ablation_paths,
        manifest=manifest,
    )
    judgments = _load_model_list(variant_dir / "judgment_table.json", ReconciliationJudgment)
    certificates = _load_model_list(variant_dir / "certificate_table.json", Certificate)

    witness_pipeline = WitnessPipeline(
        analyzer_config=analyzer_config,
        output_dir=output_root,
    )
    witness_pipeline.run(
        skill_id=skill_id,
        clauses=contract_table.clauses,
        judgments=judgments,
        certificates=certificates,
        projection_edges=projection_edges,
        events=events,
        resources=resources,
        paths=ablation_paths,
    )
    write_status_artifact(
        variant_dir,
        skill_id=skill_id,
        artifact_kind=ABLATION_ARTIFACT_KIND,
        system_id=system_id,
    )


def _build_deterministic_only_contract_table(
    *,
    skill_id: str,
    skill_path: Path,
    manifest: PackageManifest,
    steps: list[Step],
    step_edges: list[StepOrderEdge],
    unresolved_refs: list[str],
) -> ContractTable:
    clauses = _extract_deterministic_clauses(skill_path=skill_path, manifest=manifest)
    clauses = classify_clause_roles(clauses, manifest)
    return build_contract_table(
        skill_id,
        clauses,
        steps=steps,
        step_order_edges=step_edges,
        unresolved_refs=unresolved_refs,
        conflicts=detect_clause_conflicts(clauses),
    )


def _build_unguarded_contract_table(
    *,
    skill_id: str,
    skill_path: Path,
    manifest: PackageManifest,
    steps: list[Step],
    step_edges: list[StepOrderEdge],
    unresolved_refs: list[str],
    events: list[CapabilityEvent],
    raw_samples_path: Path,
    raw_recall_samples_path: Path,
) -> ContractTable:
    deterministic_clauses = _extract_deterministic_clauses(skill_path=skill_path, manifest=manifest)
    doc_contents = _load_doc_contents(skill_path=skill_path, manifest=manifest)
    llm_clauses = _aggregate_clause_samples(raw_samples_path)
    llm_clauses = _ground_evidence(llm_clauses, doc_contents)
    llm_clauses = classify_clause_roles(llm_clauses, manifest)

    all_clauses = deterministic_clauses + llm_clauses
    recall_path = raw_recall_samples_path
    if recall_path.exists():
        recall_clauses = _aggregate_clause_samples(recall_path)
        recall_clauses = _ground_evidence(recall_clauses, doc_contents)
        recall_clauses = classify_clause_roles(recall_clauses, manifest)
        target_capabilities = {
            event.capability
            for event in events
        }
        all_clauses = merge_recall_clauses(
            existing_clauses=all_clauses,
            recall_clauses=recall_clauses,
            target_capabilities=target_capabilities,
        )

    all_clauses = classify_clause_roles(all_clauses, manifest)
    return build_contract_table(
        skill_id,
        all_clauses,
        steps=steps,
        step_order_edges=step_edges,
        unresolved_refs=unresolved_refs,
        conflicts=detect_clause_conflicts(all_clauses),
    )


def _strip_scope_constraints(contract_table: ContractTable) -> ContractTable:
    stripped_clauses = [
        clause.model_copy(update={"target": None, "constraints": []})
        for clause in contract_table.clauses
    ]
    return build_contract_table(
        contract_table.skill_id,
        stripped_clauses,
        steps=contract_table.steps,
        step_order_edges=contract_table.step_order_edges,
        unresolved_refs=contract_table.unresolved_references,
        conflicts=detect_clause_conflicts(stripped_clauses),
    )


def _extract_deterministic_clauses(*, skill_path: Path, manifest: PackageManifest) -> list[Clause]:
    file_map = {entry.file_id: entry for entry in manifest.files}
    deterministic_clauses: list[Clause] = []
    for doc in manifest.documents:
        if doc.depth != 0:
            continue
        entry = file_map.get(doc.file_id)
        if entry is None:
            continue
        content = (skill_path / entry.relative_path).read_text(encoding="utf-8")
        fm_text, _ = parse_yaml_frontmatter(content)
        if not fm_text:
            continue
        fields = parse_frontmatter_fields(fm_text)
        deterministic_clauses.extend(extract_from_frontmatter(fields, doc.doc_id, fm_text))
    return deterministic_clauses


def _aggregate_clause_samples(path: Path) -> list[Clause]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = [
        [ClauseSample.model_validate(item) for item in group]
        for group in payload
    ]
    if not samples:
        return []
    return aggregate_samples(samples, n_samples=len(samples))


def _write_contract_artifacts(*, artifact_dir: Path, contract_table: ContractTable) -> None:
    _write_json(artifact_dir / "contract_table.json", contract_table.model_dump())
    _write_json(
        artifact_dir / "canonical_clauses.json",
        [clause.model_dump() for clause in contract_table.clauses],
    )
    _write_json(
        artifact_dir / "policy_clauses.json",
        [clause.model_dump() for clause in contract_table.clauses if clause.role.value == "policy"],
    )
    _write_json(
        artifact_dir / "knowledge_clauses.json",
        [clause.model_dump() for clause in contract_table.clauses if clause.role.value == "knowledge"],
    )
    _write_json(
        artifact_dir / "g_d.json",
        _build_contract_graph_artifact(
            contract_table.clauses,
            contract_table.steps,
            contract_table.step_order_edges,
        ),
    )


def _write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_model_list(path: Path, model_cls: type) -> list:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return [model_cls.model_validate(item) for item in payload]


def _load_doc_contents(*, skill_path: Path, manifest: PackageManifest) -> dict[str, str]:
    file_map = {entry.file_id: entry for entry in manifest.files}
    doc_contents: dict[str, str] = {}
    for doc in manifest.documents:
        entry = file_map.get(doc.file_id)
        if entry is None:
            continue
        file_path = skill_path / entry.relative_path
        try:
            doc_contents[doc.doc_id] = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
    return doc_contents
