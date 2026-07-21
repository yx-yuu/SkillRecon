"""Contract observation pipeline combining deterministic parsing with ICCM."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skillrecon.contract.classify import classify_clause_roles
from skillrecon.contract.deterministic import extract_from_frontmatter
from skillrecon.contract.iccm import ICCMExtractor, _load_taxonomy_atoms
from skillrecon.contract.parser import (
    parse_document,
    parse_frontmatter_fields,
    parse_yaml_frontmatter,
)
from skillrecon.contract.recall import (
    build_recall_focus_context,
    merge_recall_clauses,
    select_recall_steps,
    select_recall_events,
)
from skillrecon.contract.steps import build_steps
from skillrecon.contract.steps import select_steps_for_llm
from skillrecon.contract.voting import (
    aggregate_samples,
    apply_authorization_guard,
    build_contract_table,
    detect_clause_conflicts,
    locate_evidence_span,
)
from skillrecon.core.config import AnalyzerConfig
from skillrecon.core.enums import ClauseOperator
from skillrecon.core.sensitivity import event_requires_authorization
from skillrecon.core.types import (
    CapabilityEvent,
    Clause,
    ContractTable,
    DocBlock,
    DocumentPack,
    PackageManifest,
    ResourceUse,
    Step,
    StepOrderEdge,
)
from skillrecon.loader.manifest import build_manifest

logger = logging.getLogger(__name__)


def _load_a_req(taxonomy_path: Path) -> list[str]:
    """Load the authorization-sensitive capability list from taxonomy."""
    with taxonomy_path.open(encoding="utf-8") as f:
        data = json.load(f)
    result: list[str] = data.get("a_req", [])
    return result


class ContractObservationPipeline:
    """Run contract observation and persist intermediate artifacts."""

    def __init__(
        self,
        analyzer_config: AnalyzerConfig,
        skill_data_root: Path,
        output_dir: Path,
    ) -> None:
        self._config = analyzer_config
        self._data_root = skill_data_root
        self._output_dir = output_dir
        self._taxonomy_path = analyzer_config.taxonomy_path
        self._a_req = _load_a_req(self._taxonomy_path)

    def run(self, skill_id: str) -> ContractTable:
        """Run contract observation for one skill package."""
        skill_path = self._data_root / skill_id
        artifact_dir = self._output_dir / skill_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Step 1: Building manifest...", skill_id)
        manifest, unresolved_refs, artifact_targets = build_manifest(skill_path, skill_id)
        self._save_artifact(artifact_dir / "package_manifest.json", manifest.model_dump())
        self._save_artifact(
            artifact_dir / "doc_ref_graph.json",
            [ref.model_dump() for ref in manifest.document_references],
        )
        self._save_artifact(
            artifact_dir / "package_links.json",
            [link.model_dump() for link in manifest.links],
        )

        logger.info("[%s] Step 2: Parsing documents...", skill_id)
        file_map = {f.file_id: f for f in manifest.files}
        all_blocks: list[DocBlock] = []
        all_steps: list[Step] = []
        all_step_edges: list[StepOrderEdge] = []
        deterministic_clauses: list[Clause] = []
        doc_contents: dict[str, str] = {}

        for doc in manifest.documents:
            entry = file_map[doc.file_id]
            file_path = skill_path / entry.relative_path
            try:
                content = file_path.read_text(encoding="utf-8")
            except OSError:
                logger.warning("Cannot read: %s", file_path)
                continue

            doc_contents[doc.doc_id] = content
            blocks = parse_document(doc.doc_id, content)
            all_blocks.extend(blocks)
            steps, step_edges = build_steps(blocks)
            all_steps.extend(steps)
            all_step_edges.extend(step_edges)

            if doc.depth == 0:
                fm_text, _ = parse_yaml_frontmatter(content)
                if fm_text:
                    fields = parse_frontmatter_fields(fm_text)
                    det_clauses = extract_from_frontmatter(fields, doc.doc_id, fm_text)
                    deterministic_clauses.extend(det_clauses)

        doc_pack = DocumentPack(
            skill_id=skill_id,
            admitted_docs=list(manifest.documents),
            doc_blocks=all_blocks,
            unresolved_references=unresolved_refs,
            declared_artifact_targets=artifact_targets,
        )
        self._save_artifact(artifact_dir / "document_pack.json", doc_pack.model_dump())
        self._save_artifact(
            artifact_dir / "doc_blocks.json",
            [block.model_dump() for block in all_blocks],
        )
        self._save_artifact(
            artifact_dir / "step_table.json",
            [step.model_dump() for step in all_steps],
        )
        self._save_artifact(
            artifact_dir / "step_edges.json",
            [edge.model_dump() for edge in all_step_edges],
        )
        logger.info(
            "[%s] Parsed %d blocks and %d steps from %d docs, %d deterministic clauses",
            skill_id, len(all_blocks), len(all_steps), len(manifest.documents),
            len(deterministic_clauses),
        )

        if deterministic_clauses:
            self._save_artifact(
                artifact_dir / "deterministic_clauses.json",
                [c.model_dump() for c in deterministic_clauses],
            )

        logger.info("[%s] Step 3: ICCM extraction (LLM layer)...", skill_id)
        llm_steps = select_steps_for_llm(all_steps)
        logger.info(
            "[%s] Sending %d steps to LLM (from %d total steps; skipped %d)",
            skill_id,
            len(llm_steps),
            len(all_steps),
            len(all_steps) - len(llm_steps),
        )

        from skillrecon.llm.cache import CachedLLMClient

        client = CachedLLMClient.from_config(
            self._config.llm,
            self._config.prompt_version,
        )
        extractor = ICCMExtractor(
            client,
            taxonomy_atoms=_load_taxonomy_atoms(str(self._taxonomy_path)),
            prompt_version=self._config.prompt_version,
        )
        n_samples = self._config.vote_policy.n_samples

        all_samples = extractor.extract_all(skill_id, llm_steps, n_samples)
        raw_samples_data = [[s.model_dump() for s in sl] for sl in all_samples]
        self._save_artifact(artifact_dir / "raw_clause_samples.json", raw_samples_data)

        logger.info("[%s] Step 4: Voting and canonicalization...", skill_id)
        llm_clauses = aggregate_samples(
            all_samples,
            n_samples=n_samples,
            agreement_threshold=self._config.vote_policy.agreement_threshold,
        )

        logger.info("[%s] Step 5: Authorization guard...", skill_id)
        guarded_llm_clauses = apply_authorization_guard(llm_clauses, self._a_req)

        logger.info("[%s] Step 6: Evidence grounding...", skill_id)
        grounded_llm_clauses = _ground_evidence(guarded_llm_clauses, doc_contents)

        logger.info("[%s] Step 7: Merging deterministic + LLM clauses...", skill_id)
        all_clauses = deterministic_clauses + grounded_llm_clauses
        all_clauses = classify_clause_roles(all_clauses, manifest)
        conflicts = detect_clause_conflicts(all_clauses)

        self._save_artifact(
            artifact_dir / "canonical_clauses.json",
            [clause.model_dump() for clause in all_clauses],
        )
        self._save_artifact(
            artifact_dir / "policy_clauses.json",
            [clause.model_dump() for clause in all_clauses if clause.role.value == "policy"],
        )
        self._save_artifact(
            artifact_dir / "knowledge_clauses.json",
            [clause.model_dump() for clause in all_clauses if clause.role.value == "knowledge"],
        )

        contract_table = build_contract_table(
            skill_id,
            all_clauses,
            steps=all_steps,
            step_order_edges=all_step_edges,
            unresolved_refs=unresolved_refs,
            conflicts=conflicts,
        )
        self._save_artifact(
            artifact_dir / "contract_table.json",
            contract_table.model_dump(),
        )
        self._save_artifact(
            artifact_dir / "g_d.json",
            _build_contract_graph_artifact(all_clauses, all_steps, all_step_edges),
        )

        logger.info(
            "[%s] Done: %d clauses (%d deterministic + %d LLM)",
            skill_id,
            len(contract_table.clauses),
            len(deterministic_clauses),
            len(grounded_llm_clauses),
        )
        return contract_table

    def _save_artifact(self, path: Path, data: object) -> None:
        """Persist one JSON artifact."""
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Saved artifact: %s", path)

    def recall_for_behavior(
        self,
        skill_id: str,
        contract_table: ContractTable,
        events: list[CapabilityEvent],
        resources: list[ResourceUse],
    ) -> ContractTable:
        """Run a focused recall pass for uncovered authorization-sensitive behavior."""
        artifact_dir = self._output_dir / skill_id
        skill_path = self._data_root / skill_id
        document_pack_path = artifact_dir / "document_pack.json"
        manifest_path = artifact_dir / "package_manifest.json"
        if not document_pack_path.is_file() or not manifest_path.is_file():
            return contract_table

        document_pack = DocumentPack.model_validate_json(
            document_pack_path.read_text(encoding="utf-8")
        )
        manifest = PackageManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        target_events = select_recall_events(
            events=events,
            resources=resources,
            clauses=contract_table.clauses,
            a_req=set(self._a_req),
        )
        existing_steps = contract_table.steps or build_steps(document_pack.doc_blocks)[0]
        selected_steps = select_recall_steps(
            steps=existing_steps,
            manifest=manifest,
            events=target_events,
            resources=resources,
            a_req=set(self._a_req),
        )
        focus_context = build_recall_focus_context(
            events=target_events,
            resources=resources,
            a_req=set(self._a_req),
        )
        self._save_artifact(
            artifact_dir / "recall_steps.json",
            [step.model_dump() for step in selected_steps],
        )
        self._save_artifact(
            artifact_dir / "recall_target_events.json",
            [event.model_dump() for event in target_events],
        )
        self._save_artifact(artifact_dir / "recall_focus_context.json", focus_context)
        if not selected_steps or not focus_context:
            return contract_table

        from skillrecon.llm.cache import CachedLLMClient

        client = CachedLLMClient.from_config(
            self._config.llm,
            self._config.prompt_version,
        )
        extractor = ICCMExtractor(
            client,
            taxonomy_atoms=_load_taxonomy_atoms(str(self._taxonomy_path)),
            prompt_version=self._config.prompt_version,
        )
        recall_samples = extractor.extract_recall_all(
            skill_id,
            selected_steps,
            focus_context,
            self._config.vote_policy.n_samples,
            call_key_prefix="iccm_recall",
        )
        self._save_artifact(
            artifact_dir / "recall_raw_clause_samples.json",
            [[sample.model_dump() for sample in group] for group in recall_samples],
        )

        recall_clauses = aggregate_samples(
            recall_samples,
            n_samples=self._config.vote_policy.n_samples,
            agreement_threshold=self._config.vote_policy.agreement_threshold,
        )
        recall_clauses = apply_authorization_guard(recall_clauses, self._a_req)
        doc_contents = _load_doc_contents(skill_path, manifest)
        if len(doc_contents) < len(manifest.documents):
            doc_contents = {
                **_load_doc_contents_from_document_pack(document_pack),
                **doc_contents,
            }
        recall_clauses = _ground_evidence(recall_clauses, doc_contents)
        recall_clauses = classify_clause_roles(recall_clauses, manifest)
        self._save_artifact(
            artifact_dir / "recall_clauses.json",
            [clause.model_dump() for clause in recall_clauses],
        )

        a_req_set = set(self._a_req)
        target_capabilities = {
            event.capability
            for event in events
            if event_requires_authorization(event, a_req_set)
        }
        merged_clauses = merge_recall_clauses(
            existing_clauses=contract_table.clauses,
            recall_clauses=recall_clauses,
            target_capabilities=target_capabilities,
        )
        if len(merged_clauses) == len(contract_table.clauses):
            return contract_table

        conflicts = detect_clause_conflicts(merged_clauses)
        merged_contract_table = build_contract_table(
            skill_id,
            merged_clauses,
            steps=contract_table.steps,
            step_order_edges=contract_table.step_order_edges,
            unresolved_refs=contract_table.unresolved_references,
            conflicts=conflicts,
        )
        self._save_artifact(
            artifact_dir / "canonical_clauses.json",
            [clause.model_dump() for clause in merged_clauses],
        )
        self._save_artifact(
            artifact_dir / "policy_clauses.json",
            [clause.model_dump() for clause in merged_clauses if clause.role.value == "policy"],
        )
        self._save_artifact(
            artifact_dir / "knowledge_clauses.json",
            [clause.model_dump() for clause in merged_clauses if clause.role.value == "knowledge"],
        )
        self._save_artifact(
            artifact_dir / "contract_table.json",
            merged_contract_table.model_dump(),
        )
        self._save_artifact(
            artifact_dir / "g_d.json",
            _build_contract_graph_artifact(
                merged_clauses,
                merged_contract_table.steps,
                merged_contract_table.step_order_edges,
            ),
        )
        logger.info(
            "[%s] Targeted recall added %d clauses from %d focused steps",
            skill_id,
            len(merged_clauses) - len(contract_table.clauses),
            len(selected_steps),
        )
        return merged_contract_table


def _ground_evidence(
    clauses: list[Clause],
    doc_contents: dict[str, str],
) -> list[Clause]:
    """Resolve placeholder evidence spans against the available documents."""
    grounded: list[Clause] = []
    resolved_count = 0
    total_count = 0

    for clause in clauses:
        new_spans = []
        for span in clause.evidence_spans:
            total_count += 1
            if span.doc_id != "unresolved":
                new_spans.append(span)
                resolved_count += 1
                continue
            found = False
            for doc_id, content in doc_contents.items():
                result = locate_evidence_span(span.text, doc_id, content)
                if result.start_offset != 0 or result.end_offset != 0:
                    new_spans.append(result)
                    resolved_count += 1
                    found = True
                    break
            if not found:
                new_spans.append(span)

        grounded.append(
            Clause(
                clause_id=clause.clause_id,
                capability=clause.capability,
                operator=clause.operator,
                role=clause.role,
                target=clause.target,
                constraints=clause.constraints,
                evidence_spans=new_spans,
                vote_agreement=clause.vote_agreement,
                step_ids=clause.step_ids,
                source_doc_ids=clause.source_doc_ids,
            )
        )

    logger.info("Evidence grounding: %d/%d spans resolved", resolved_count, total_count)
    return grounded


def _load_doc_contents(
    skill_path: Path,
    manifest: PackageManifest,
) -> dict[str, str]:
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


def _load_doc_contents_from_document_pack(
    document_pack: DocumentPack,
) -> dict[str, str]:
    """Rebuild sparse document text from saved DocBlocks for resume paths."""
    blocks_by_doc: dict[str, list[DocBlock]] = {}
    for block in document_pack.doc_blocks:
        blocks_by_doc.setdefault(block.doc_id, []).append(block)

    doc_contents: dict[str, str] = {}
    for doc_id, blocks in blocks_by_doc.items():
        length = max(
            max(block.end_offset for block in blocks),
            max(block.start_offset + len(block.content) for block in blocks),
        )
        chars = [" "] * max(length, 0)
        for block in sorted(blocks, key=lambda item: (item.start_offset, item.end_offset)):
            start = max(block.start_offset, 0)
            text = block.content
            end = start + len(text)
            if end > len(chars):
                chars.extend([" "] * (end - len(chars)))
            chars[start:end] = text
        doc_contents[doc_id] = "".join(chars).rstrip()
    return doc_contents


def _build_contract_graph_artifact(
    clauses: list[Clause],
    steps: list[Step],
    step_order_edges: list[StepOrderEdge],
) -> dict[str, object]:
    """Materialize a lightweight contract graph derived from canonical clauses."""
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    seen_nodes: set[str] = set()

    def add_node(node_id: str, kind: str, **attrs: object) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append({"id": node_id, "kind": kind, **attrs})

    for step in steps:
        add_node(
            step.step_id,
            "step",
            doc_id=step.doc_id,
            order_index=step.order_index,
            local_index=step.local_index,
            step_type=step.step_type,
            heading_context=step.heading_context,
            text=step.text,
        )

    for edge in step_order_edges:
        edges.append(
            {
                "source": edge.source_step_id,
                "target": edge.target_step_id,
                "kind": edge.relation,
            }
        )

    for clause in clauses:
        for step_id in clause.step_ids:
            add_node(step_id, "step")
            edges.append(
                {
                    "source": step_id,
                    "target": clause.clause_id,
                    "kind": (
                        "prohibits"
                        if clause.operator == ClauseOperator.PROHIBITED
                        else "declares"
                    ),
                }
            )

        add_node(
            clause.clause_id,
            "clause",
            capability=clause.capability,
            operator=clause.operator.value,
            role=clause.role.value,
            target=clause.target,
        )

        capability_id = f"cap:{clause.capability}"
        add_node(capability_id, "capability", value=clause.capability)
        edges.append({"source": clause.clause_id, "target": capability_id, "kind": "about"})

        if clause.target:
            target_id = f"{clause.clause_id}:target"
            add_node(target_id, "resource", value=clause.target)
            edges.append({"source": clause.clause_id, "target": target_id, "kind": "targets"})

        for constraint in clause.constraints:
            add_node(
                constraint.constraint_id,
                "constraint",
                constraint_type=constraint.constraint_type,
                value=constraint.value,
            )
            edges.append(
                {
                    "source": clause.clause_id,
                    "target": constraint.constraint_id,
                    "kind": "constrains",
                }
            )

        for index, span in enumerate(clause.evidence_spans):
            evidence_id = f"{clause.clause_id}:ev:{index}"
            add_node(
                evidence_id,
                "evidence_span",
                doc_id=span.doc_id,
                start_offset=span.start_offset,
                end_offset=span.end_offset,
                text=span.text,
            )
            edges.append(
                {
                    "source": clause.clause_id,
                    "target": evidence_id,
                    "kind": "supported_by",
                }
            )

    return {"nodes": nodes, "edges": edges}
