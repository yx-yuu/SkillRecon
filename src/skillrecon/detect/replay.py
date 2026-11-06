"""Mechanical certificate revalidation from the selected witness graph alone."""
from __future__ import annotations

import json
from pathlib import Path

from skillrecon.core.types import (Bridge, CandidatePair, CapabilityEvent, Certificate,
    ContractTable, Finding, OrchestrationHypothesis, ReconciliationEdge,
    ReconciliationJudgment, ResourceUse, RiskPath, Witness)
from skillrecon.reconcile.derivation import materialize_reconciliation


def replay_reconciliation(inputs: dict):
    table = ContractTable.model_validate(inputs['contract'])
    resources = [ResourceUse.model_validate(item) for item in inputs['resources']]
    by_event = {}
    for resource in resources:
        if resource.event_id:
            by_event.setdefault(resource.event_id, []).append(resource)
    def index(name, model, key):
        return {getattr(value, key): value for value in map(model.model_validate, inputs[name])}
    return materialize_reconciliation(
        candidates=[CandidatePair.model_validate(item) for item in inputs['candidates']],
        clauses={c.clause_id:c for c in table.clauses},
        events=index('events', CapabilityEvent, 'event_id'),
        resources_by_event=by_event, resources_by_id={r.resource_id:r for r in resources},
        paths=index('paths', RiskPath, 'path_id'), bridges=index('bridges', Bridge, 'bridge_id'),
        orchestrations=index('orchestrations', OrchestrationHypothesis, 'hypothesis_id'),
        overlap_map={key:set(value) for key,value in inputs['overlap_map'].items()},
        a_req=set(inputs['a_req']), steps_by_id={s.step_id:s for s in table.steps},
        step_order_edges=table.step_order_edges)


def validate_bundle(bundle: dict) -> dict:
    """Rebuild typed premises and perform B1's frozen read-only acceptance."""
    witness = Witness.model_validate(bundle['witness'])
    finding = Finding.model_validate(bundle['finding'])
    if bundle.get('schema_version')!=2 or 'evidence_graph' not in bundle:
        return {'witness_id':witness.witness_id,'passed':False,'failure_reason':'requires_self_contained_graph_bundle_v2'}
    from skillrecon.detect.evidence_graph import validate_portable_graph
    return validate_portable_graph(bundle['evidence_graph'],witness,finding).model_dump(mode='json')


def export_bundles(artifact_dir: Path) -> list[Path]:
    """Export all accepted and rejected candidates with replay inputs and proof snapshots."""
    load = lambda name: json.loads((artifact_dir/name).read_text(encoding='utf-8'))
    findings = {item['finding_id']:item for item in load('findings.json')}
    judgments = {item['judgment_id']:item for item in load('judgment_table.json')}
    certificates = {item['certificate_id']:item for item in load('certificate_table.json')}
    edges = {item['edge_id']:item for item in load('reconciliation_edges.json')}
    output = artifact_dir/'witness_bundles'
    output.mkdir(exist_ok=True)
    written = []
    for witness in [*load('witnesses.json'), *load('rejected_witnesses.json')]:
        graph=witness.pop('evidence_graph')
        bundle = {'schema_version':2, 'witness':witness,
                  'finding':findings[witness['finding_id']], 'evidence_graph':graph,
                  'judgments':[judgments[key] for key in witness['judgment_ids']],
                  'certificates':[certificates[key] for key in witness['certificate_ids']],
                  'projection_edges':[edges[key] for key in witness['projection_edge_ids']]}
        path = output/(witness['witness_id']+'.json')
        path.write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        written.append(path)
    return written
