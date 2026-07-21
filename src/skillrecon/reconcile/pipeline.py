"""Reconciliation pipeline from candidate generation to G_X export."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skillrecon.core.config import AnalyzerConfig
from skillrecon.core.enums import ClauseRole
from skillrecon.core.types import (
    Bridge,
    CapabilityEvent,
    Certificate,
    ContractTable,
    GraphEdge,
    GraphNode,
    GraphObject,
    OrchestrationHypothesis,
    PackageManifest,
    ReconciliationEdge,
    ReconciliationJudgment,
    ResourceUse,
    RiskPath,
)
from skillrecon.reconcile.candidate import generate_alignment_candidates, generate_candidates
from skillrecon.reconcile.derivation import derive_alignment_edges, materialize_reconciliation
from skillrecon.reconcile.predicate import load_overlap_policy

logger = logging.getLogger(__name__)


def _load_a_req(taxonomy_path: Path) -> set[str]:
    """Load the authorization-sensitive capability set from taxonomy."""
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return set(data.get("a_req", []))


class ReconciliationPipeline:
    """Build reconciliation judgments, certificates, and graph artifacts."""

    def __init__(
        self,
        analyzer_config: AnalyzerConfig,
        output_dir: Path,
    ) -> None:
        self._config = analyzer_config
        self._output_dir = output_dir
        self._reconciliation_policy = analyzer_config.reconciliation_policy
        self._overlap_policy_path = Path(self._reconciliation_policy.overlap_policy_path)
        self._a_req = _load_a_req(analyzer_config.taxonomy_path)

    def run(
        self,
        skill_id: str,
        contract_table: ContractTable,
        events: list[CapabilityEvent],
        resources: list[ResourceUse],
        bridges: list[Bridge],
        orchestrations: list[OrchestrationHypothesis],
        paths: list[RiskPath],
        manifest: PackageManifest,
    ) -> list[ReconciliationEdge]:
        """Run reconciliation for one skill package."""
        logger.info("Starting reconciliation for skill %s", skill_id)
        skill_dir = self._output_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        overlap_map = load_overlap_policy(self._overlap_policy_path)
        policy_contract_table = _policy_only_contract_table(contract_table)
        self._save_artifact(
            skill_dir / "reconcile_contract_table.json",
            policy_contract_table.model_dump(),
        )

        candidates = generate_candidates(
            contract_table=policy_contract_table,
            events=events,
            resources=resources,
            paths=paths,
            bridges=bridges,
            orchestrations=orchestrations,
            manifest=manifest,
            overlap_map=overlap_map,
            a_req=self._a_req,
            max_candidates_per_behavior=(
                self._reconciliation_policy.max_candidates_per_behavior
            ),
            max_alignment_fallbacks_per_step=(
                self._reconciliation_policy.max_alignment_fallbacks_per_step
            ),
            max_semantic_event_fallbacks=(
                self._reconciliation_policy.max_semantic_event_fallbacks
            ),
            max_semantic_path_fallbacks=(
                self._reconciliation_policy.max_semantic_path_fallbacks
            ),
        )
        self._save_artifact(
            skill_dir / "candidate_pairs.json",
            [c.model_dump() for c in candidates],
        )
        alignment_candidates = generate_alignment_candidates(
            steps=policy_contract_table.steps,
            manifest=manifest,
            max_fallbacks_per_step=(
                self._reconciliation_policy.max_alignment_fallbacks_per_step
            ),
        )
        self._save_artifact(
            skill_dir / "alignment_candidates.json",
            [candidate.model_dump() for candidate in alignment_candidates],
        )

        clauses_map = {c.clause_id: c for c in policy_contract_table.clauses}
        events_map = {e.event_id: e for e in events}
        resources_by_event: dict[str, list[ResourceUse]] = {}
        for r in resources:
            if r.event_id:
                resources_by_event.setdefault(r.event_id, []).append(r)
        resources_by_id = {r.resource_id: r for r in resources}
        paths_map = {p.path_id: p for p in paths}
        bridges_map = {b.bridge_id: b for b in bridges}
        orch_map = {o.hypothesis_id: o for o in orchestrations}

        result = materialize_reconciliation(
            candidates=candidates,
            clauses=clauses_map,
            events=events_map,
            resources_by_event=resources_by_event,
            resources_by_id=resources_by_id,
            paths=paths_map,
            bridges=bridges_map,
            orchestrations=orch_map,
            overlap_map=overlap_map,
            a_req=self._a_req,
            steps_by_id={step.step_id: step for step in policy_contract_table.steps},
        )
        alignment_edges = derive_alignment_edges(
            alignment_candidates=alignment_candidates,
            clauses=clauses_map,
            events=events_map,
            resources_by_id=resources_by_id,
            projection_edges=result.edges,
        )
        result.edges.extend(alignment_edges)
        self._save_artifact(
            skill_dir / "judgment_table.json",
            [judgment.model_dump() for judgment in result.judgments],
        )
        self._save_artifact(
            skill_dir / "certificate_table.json",
            [certificate.model_dump() for certificate in result.certificates],
        )
        self._save_artifact(
            skill_dir / "reconciliation_edges.json",
            [e.model_dump() for e in result.edges],
        )

        g_x = _build_g_x(
            result.judgments,
            result.certificates,
            result.edges,
            policy_contract_table,
            manifest,
            events,
            resources,
            paths,
        )
        self._save_artifact(skill_dir / "g_x.json", g_x)

        logger.info(
            "Reconciliation complete: %d candidates -> %d judgments, %d certificates, %d edges",
            len(candidates),
            len(result.judgments),
            len(result.certificates),
            len(result.edges),
        )
        return result.edges

    @staticmethod
    def _save_artifact(path: Path, data: object) -> None:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        logger.debug("Saved artifact: %s", path)


def _build_g_x(
    judgments: list[ReconciliationJudgment],
    certificates: list[Certificate],
    edges: list[ReconciliationEdge],
    contract_table: ContractTable,
    manifest: PackageManifest,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    paths: list[RiskPath],
) -> dict:
    """Build proof-carrying G_X as a serializable dict."""
    nodes: dict[str, GraphNode] = {}
    edge_list: list[GraphEdge] = []
    seen_structural_edges: set[tuple[str, str, str, tuple[tuple[str, object], ...]]] = set()

    def add_edge(kind: str, source: str, target: str, **attrs: object) -> None:
        key = (kind, source, target, tuple(sorted(attrs.items())))
        if key in seen_structural_edges:
            return
        seen_structural_edges.add(key)
        edge_list.append(
            GraphEdge(
                kind=kind,
                source=source,
                target=target,
                attrs=attrs,
            )
        )

    clause_ids_used = {edge.clause_id for edge in edges if edge.clause_id}
    constraint_ids_used = {edge.constraint_id for edge in edges if edge.constraint_id}
    step_ids_used = {edge.step_id for edge in edges if edge.step_id}
    code_unit_ids_used = {edge.code_unit_id for edge in edges if edge.code_unit_id}
    for step in contract_table.steps:
        if step.step_id in step_ids_used or any(
            clause.clause_id in clause_ids_used and step.step_id in clause.step_ids
            for clause in contract_table.clauses
        ):
            nodes[step.step_id] = GraphNode(
                node_id=step.step_id,
                kind="step",
                attrs={
                    "doc_id": step.doc_id,
                    "order_index": step.order_index,
                    "step_type": step.step_type,
                },
            )
    for step_edge in contract_table.step_order_edges:
        if step_edge.source_step_id in nodes and step_edge.target_step_id in nodes:
            add_edge(step_edge.relation, step_edge.source_step_id, step_edge.target_step_id)
    for c in contract_table.clauses:
        if c.clause_id in clause_ids_used:
            nodes[c.clause_id] = GraphNode(
                node_id=c.clause_id,
                kind="clause",
                attrs={
                    "capability": c.capability,
                    "operator": c.operator.value,
                    "role": c.role.value,
                },
            )
        for constraint in c.constraints:
            if constraint.constraint_id in constraint_ids_used:
                nodes[constraint.constraint_id] = GraphNode(
                    node_id=constraint.constraint_id,
                    kind="constraint",
                    attrs={
                        "constraint_type": constraint.constraint_type,
                        "value": constraint.value,
                    },
                )
            if c.clause_id in clause_ids_used:
                for step_id in c.step_ids:
                    if step_id in nodes:
                        add_edge("declares", step_id, c.clause_id)

    code_units_by_id = {unit.unit_id: unit for unit in manifest.code_units}
    for code_unit_id in code_unit_ids_used:
        unit = code_units_by_id.get(code_unit_id)
        nodes[code_unit_id] = GraphNode(
            node_id=code_unit_id,
            kind="code_unit",
            attrs={
                "language": unit.language if unit is not None else "",
            },
        )

    events_map = {e.event_id: e for e in events}
    resources_map = {r.resource_id: r for r in resources}
    paths_map = {p.path_id: p for p in paths}

    for edge in edges:
        if edge.event_id and edge.event_id not in nodes:
            ev = events_map.get(edge.event_id)
            if ev:
                nodes[edge.event_id] = GraphNode(
                    node_id=edge.event_id,
                    kind="event",
                    attrs={
                        "capability": ev.capability,
                        "unit_id": ev.unit_id,
                    },
                )
        if edge.event_id:
            ev = events_map.get(edge.event_id)
            if ev and ev.unit_id in code_unit_ids_used:
                add_edge("emits", ev.unit_id, edge.event_id)
        if edge.resource_id and edge.resource_id not in nodes:
            res = resources_map.get(edge.resource_id)
            if res:
                nodes[edge.resource_id] = GraphNode(
                    node_id=edge.resource_id,
                    kind="resource",
                    attrs={
                        "resource_type": res.resource_type,
                        "value": res.value,
                    },
                )
        if edge.path_id and edge.path_id not in nodes:
            p = paths_map.get(edge.path_id)
            if p:
                nodes[edge.path_id] = GraphNode(
                    node_id=edge.path_id,
                    kind="path",
                    attrs={
                        "source_label": p.source.label,
                        "sink_label": p.sink.label,
                    },
                )

    for judgment in judgments:
        nodes[judgment.judgment_id] = GraphNode(
            node_id=judgment.judgment_id,
            kind="judgment",
            attrs={
                "kind": judgment.kind.value,
                "result": judgment.result.value,
                "subject_refs": judgment.subject_refs,
            },
        )
        for subject_ref in judgment.subject_refs:
            edge_list.append(
                GraphEdge(
                    kind="subject_of",
                    source=subject_ref,
                    target=judgment.judgment_id,
                )
            )
        for premise_judgment_id in judgment.premise_judgment_ids:
            edge_list.append(
                GraphEdge(
                    kind="premise_for",
                    source=premise_judgment_id,
                    target=judgment.judgment_id,
                )
            )

    for certificate in certificates:
        nodes[certificate.certificate_id] = GraphNode(
            node_id=certificate.certificate_id,
            kind="certificate",
            attrs={
                "kind": certificate.kind.value,
                "subject_refs": certificate.subject_refs,
            },
        )
        for supporting_judgment_id in certificate.supporting_judgment_ids:
            edge_list.append(
                GraphEdge(
                    kind="certified_by",
                    source=supporting_judgment_id,
                    target=certificate.certificate_id,
                )
            )

    for edge in edges:
        edge_list.append(
            GraphEdge(
                edge_id=edge.edge_id,
                kind="projection",
                source=edge.clause_id or edge.constraint_id or edge.step_id or "",
                target=edge.event_id
                or edge.resource_id
                or edge.path_id
                or edge.code_unit_id
                or "",
                attrs={
                    "relation": edge.relation.value,
                    "route_support_type": edge.route_support_type.value
                    if edge.route_support_type
                    else None,
                    "source_judgment_ids": edge.source_judgment_ids,
                },
            )
        )

    return GraphObject(
        nodes=list(nodes.values()),
        edges=edge_list,
    ).to_wire(node_kind_field="type")


def _policy_only_contract_table(contract_table: ContractTable) -> ContractTable:
    """Return the reconciliation-facing subset of contract clauses."""
    return ContractTable(
        skill_id=contract_table.skill_id,
        clauses=[
            clause for clause in contract_table.clauses if clause.role == ClauseRole.POLICY
        ],
        steps=list(contract_table.steps),
        step_order_edges=list(contract_table.step_order_edges),
        unresolved_references=list(contract_table.unresolved_references),
        cross_doc_conflicts=list(contract_table.cross_doc_conflicts),
    )
