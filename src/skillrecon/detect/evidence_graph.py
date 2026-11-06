"""Algorithm B1 over an explicit, self-contained proof-carrying G_X.

Every package object used by revalidation is a selected graph node. Structural
scope/span/flow/order edges are selected graph edges. Predicates and certificates
reify the proof relations already exported by reconciliation. No package-wide
replay context is consulted by feasibility or the portable validator.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
import networkx as nx

from skillrecon.core.types import Witness, WitnessValidation


def build_evidence_graph(inputs, judgments, certificates, projections):
    nodes, edges, dependencies, required_edges = {}, {}, {}, {}

    def node(identifier, kind, data):
        nodes[identifier]={'id':identifier,'type':kind,'data':data}
        dependencies.setdefault(identifier,set());required_edges.setdefault(identifier,set())

    def edge(kind, source, target, data=None, identifier=None):
        identifier=identifier or f'{kind}|{source}|{target}'
        edges[identifier]={'id':identifier,'kind':kind,'source':source,'target':target,'data':data or {}}
        return identifier

    def require(owner, target, kind):
        dependencies[owner].add(target)
        required_edges[owner].add(edge(kind,target,owner))

    table=inputs['contract']
    for step in table['steps']:node(step['step_id'],'step',step)
    for clause in table['clauses']:
        identifier=clause['clause_id'];node(identifier,'clause',clause)
        for step_id in clause['step_ids']:require(identifier,step_id,'declares')
        for index,span in enumerate(clause['evidence_spans']):
            span_id=f'{identifier}:ev:{index}';node(span_id,'evidence_span',span)
            require(identifier,span_id,'grounds')
        for scope in clause['constraints']:
            scope_id=scope['constraint_id'];node(scope_id,'constraint',scope)
            dependencies[identifier].add(scope_id)
            required_edges[identifier].add(edge('has_scope',identifier,scope_id))
            if scope.get('evidence') is not None:
                span_id=scope_id+':ev';node(span_id,'evidence_span',scope['evidence'])
                require(scope_id,span_id,'grounds')
    for order in table['step_order_edges']:
        edge(order['relation'],order['source_step_id'],order['target_step_id'],order,
             identifier='order:'+order['edge_id'])
    for kind,field,key in [('event','events','event_id'),('resource','resources','resource_id'),
            ('path','paths','path_id'),('bridge','bridges','bridge_id'),('orchestration','orchestrations','hypothesis_id')]:
        for item in inputs[field]:node(item[key],kind,item)
    for resource in inputs['resources']:
        if resource.get('event_id'):require(resource['resource_id'],resource['event_id'],'uses')
    for bridge in inputs['bridges']:
        for field in ['source_event_id','target_event_id']:
            if bridge.get(field):require(bridge['bridge_id'],bridge[field],'bridge_endpoint')
    for path in inputs['paths']:
        identifier=path['path_id']
        sequence=[segment['event_id'] for segment in path['segments'] if segment.get('event_id')]
        if not sequence:sequence=[value for value in [path['source']['event_id'],path['sink']['event_id']] if value]
        for event_id in sequence:require(identifier,event_id,'path_event')
        for segment in [path['source'],path['sink'],*path['segments']]:
            for resource_id in segment.get('resource_ids',[]):require(identifier,resource_id,'path_resource')
        for source,target in zip(sequence,sequence[1:]):
            required_edges[identifier].add(edge('flows_to',source,target,{'path_id':identifier},
                identifier=f'flow:{identifier}:{source}:{target}'))
        for bridge_id in path['bridges_used']:require(identifier,bridge_id,'path_bridge')
        for hypothesis in path.get('orchestration_hypotheses',[]):require(identifier,hypothesis,'path_hypothesis')
    frontiers={item['event_id']:[] for item in inputs['events']}
    frontiers.update({item['path_id']:[] for item in inputs['paths']})
    for candidate in inputs['candidates']:
        subject=candidate.get('event_id') or candidate.get('path_id') or candidate.get('resource_id')
        identifier=candidate['candidate_id'];node(identifier,'candidate',candidate)
        require(identifier,candidate['clause_id'],'candidate_clause')
        for field in ['event_id','resource_id','path_id']:
            if candidate.get(field):require(identifier,candidate[field],'candidate_subject')
        for ref in candidate['route_refs']:
            if ref in nodes:require(identifier,ref,'candidate_route')
        frontiers.setdefault(subject,[]).append(identifier)
    for subject,candidates in frontiers.items():
        identifier='frontier:'+subject
        node(identifier,'frontier',{'subject':subject,'candidate_ids':candidates})
        require(identifier,subject,'inspected_frontier')
        for candidate_id in candidates:require(identifier,candidate_id,'inspected_candidate')
    for judgment in judgments:
        raw=judgment.model_dump(mode='json');node(judgment.judgment_id,'judgment',raw)
    for certificate in certificates:
        raw=certificate.model_dump(mode='json');node(certificate.certificate_id,'certificate',raw)
    for judgment in judgments:
        if judgment.candidate_id:
            if judgment.candidate_id not in nodes:
                # Path routes without retrieved clauses have a deterministic
                # synthetic candidate, reconstructed by materialization itself.
                node(judgment.candidate_id,'derived_reference',{'reference':judgment.candidate_id})
            require(judgment.judgment_id,judgment.candidate_id,'evaluates_candidate')
        for ref in [*judgment.subject_refs,*judgment.evidence_refs,*judgment.premise_judgment_ids]:
            if ref not in nodes:node(ref,'derived_reference',{'reference':ref})
            require(judgment.judgment_id,ref,'premise_for')
    for certificate in certificates:
        for ref in [*certificate.subject_refs,*certificate.evidence_refs,*certificate.supporting_judgment_ids]:
            if ref not in nodes:node(ref,'derived_reference',{'reference':ref})
            require(certificate.certificate_id,ref,'certified_by')
    for projection in projections:
        raw=projection.model_dump(mode='json')
        source=projection.constraint_id if projection.relation.value=='scope_violates' else projection.clause_id or projection.step_id
        target=projection.resource_id if projection.relation.value=='scope_violates' else projection.event_id or projection.path_id or projection.resource_id or projection.code_unit_id
        if source in nodes and target in nodes:
            edge(projection.relation.value,source,target,raw,projection.edge_id)
    return {'nodes':nodes,'edges':edges,'dependencies':dependencies,'required_edges':required_edges,
            'protocol':{'overlap_map':inputs['overlap_map'],'a_req':inputs['a_req']},'skill_id':table['skill_id']}


def _closure(graph,node_ids,edge_ids):
    nodes,edges=set(node_ids),set(edge_ids)
    pending=list(nodes)
    for identifier in list(edges):
        for endpoint in ['source','target']:
            value=graph['edges'][identifier][endpoint]
            if value not in nodes:nodes.add(value);pending.append(value)
    while pending:
        owner=pending.pop()
        for target in graph['dependencies'][owner]:
            if target not in nodes:nodes.add(target);pending.append(target)
        edges.update(graph['required_edges'][owner])
    return nodes,edges


def _neighborhood(graph,finding):
    roots=set(finding.certificate_ids)
    roots.update(finding.related_event_ids+finding.related_resource_ids+finding.related_clause_ids+
                 finding.related_constraint_ids+finding.related_path_ids+finding.supporting_judgment_ids)
    roots&=set(graph['nodes'])
    anchors=set(finding.certificate_ids+finding.related_event_ids+finding.related_resource_ids+
                finding.related_clause_ids+finding.related_constraint_ids+finding.related_path_ids)
    for identifier in finding.certificate_ids:anchors.update(graph['nodes'][identifier]['data']['subject_refs'])
    anchor_edges=set(finding.supporting_edge_ids)&set(graph['edges'])
    nodes,edges=_closure(graph,roots,anchor_edges)
    # Freeze the complete inspected candidate frontier for negative premises.
    negative=any(graph['nodes'][identifier]['data']['kind'] in {'no_support','scope_failure','no_justified_path'}
                 for identifier in finding.certificate_ids)
    if negative:
        subjects={key for key in nodes if graph['nodes'][key]['type'] in {'event','path','resource'}}
        frontiers={'frontier:'+key for key in subjects if 'frontier:'+key in graph['nodes']}
        anchors.update(frontiers);nodes,edges=_closure(graph,nodes|frontiers,edges)
    # Contextual resources and alternative relations enter N_v and remain removable.
    events={key for key in nodes if graph['nodes'][key]['type']=='event'}
    resources={key for key,item in graph['nodes'].items() if item['type']=='resource' and item['data'].get('event_id') in events}
    nodes,edges=_closure(graph,nodes|resources,edges)
    for identifier,item in graph['edges'].items():
        if item['source'] in nodes and item['target'] in nodes:
            if item['kind']=='flows_to' and item['data']['path_id'] not in nodes:
                continue  # A different path over the same endpoints is not a premise of this N_v.
            if 'source_judgment_ids' in item['data'] and not set(item['data']['source_judgment_ids'])<=nodes:
                continue
            edges.add(identifier)
    for certificate_id in finding.certificate_ids:
        certificate=graph['nodes'][certificate_id]['data']
        if certificate['kind'] in {'blocking_prohibition','scope_failure'}:
            for ref in certificate['subject_refs']:
                if ref in graph['nodes'] and graph['nodes'][ref]['type']=='clause':
                    anchors.add(ref)
                    anchors.update(key for key in nodes if key.startswith(ref+':ev:'))
                if ref in graph['nodes'] and graph['nodes'][ref]['type']=='constraint':
                    anchors.add(ref)
                    attachments={key for key in edges if graph['edges'][key]['kind']=='has_scope' and graph['edges'][key]['target']==ref}
                    anchor_edges.update(attachments)
                    for key in attachments:
                        clause_id=graph['edges'][key]['source'];anchors.add(clause_id)
                        anchors.update(key for key in nodes if key.startswith(clause_id+':ev:'))
                if ref in graph['nodes'] and graph['nodes'][ref]['type']=='resource':
                    event_id=graph['nodes'][ref]['data'].get('event_id')
                    if event_id:anchors.add(event_id)
        if certificate['kind']=='no_justified_path':
            path_ids=set(certificate['subject_refs'])
            anchor_edges.update(key for key in edges if
                (graph['edges'][key]['kind']=='flows_to' and graph['edges'][key]['data']['path_id'] in path_ids) or
                (graph['edges'][key]['kind']=='path_bridge' and graph['edges'][key]['target'] in path_ids))
            # The conflicting normative edge is part of the V3 motif, rather
            # than a temporal requirement re-inferred from hidden prose.
            if any('order_conflict' in graph['nodes'][key]['data'].get('notes','')
                   for key in nodes if graph['nodes'][key]['type']=='judgment'):
                anchor_edges.update(key for key in edges if graph['edges'][key]['kind']=='precedes')
    anchors&=nodes
    return nodes,edges,anchors|anchor_edges


def _context(graph,node_ids,edge_ids):
    by_kind={}
    for identifier in sorted(node_ids):
        item=graph['nodes'][identifier];by_kind.setdefault(item['type'],[]).append(item['data'])
    orders=[graph['edges'][key]['data'] for key in edge_ids if graph['edges'][key]['kind'] in {'precedes','document_precedes'}]
    candidates=by_kind.get('candidate',[])
    return {'schema_version':2,'contract':{'skill_id':graph['skill_id'],'clauses':by_kind.get('clause',[]),
        'steps':by_kind.get('step',[]),'step_order_edges':orders,'conflicts':[],'unresolved_references':[]},
        'candidates':sorted(candidates,key=lambda item:item['candidate_id']),
        'events':by_kind.get('event',[]),'resources':by_kind.get('resource',[]),
        'paths':by_kind.get('path',[]),'bridges':by_kind.get('bridge',[]),
        'orchestrations':by_kind.get('orchestration',[]),**graph['protocol']}


def _proof_signatures(judgments,certificates):
    by_id={j.judgment_id:j.model_dump(mode='json') if hasattr(j,'model_dump') else j for j in judgments} if not isinstance(judgments,dict) else judgments
    memo={}
    def signature(identifier):
        if identifier not in by_id:return identifier
        if identifier in memo:return memo[identifier]
        raw=by_id[identifier]
        value={key:item for key,item in raw.items() if key!='judgment_id'}
        # Reference collections are sets of typed premises. Execution order is
        # carried by RiskPath.segments/bridges and normative edges, not this list.
        value['subject_refs']=sorted(raw['subject_refs'])
        value['premise_judgment_ids']=sorted(signature(ref) for ref in raw['premise_judgment_ids'])
        value['evidence_refs']=sorted(signature(ref) for ref in raw['evidence_refs'])
        memo[identifier]=json.dumps(value,sort_keys=True,ensure_ascii=False)
        return memo[identifier]
    result_j={identifier:signature(identifier) for identifier in by_id}
    result_c={}
    for certificate in certificates:
        raw=certificate.model_dump(mode='json') if hasattr(certificate,'model_dump') else certificate
        value={key:item for key,item in raw.items() if key not in {'certificate_id','finding_id','supporting_judgment_ids'}}
        value['subject_refs']=sorted(raw['subject_refs'])
        value['evidence_refs']=sorted(signature(ref) for ref in raw['evidence_refs'])
        value['supporting_judgment_ids']=sorted(signature(ref) for ref in raw['supporting_judgment_ids'])
        result_c[raw['certificate_id']]=json.dumps(value,sort_keys=True,ensure_ascii=False)
    return result_j,result_c


def _feasible(graph,node_ids,edge_ids,anchors):
    graph['_feasibility_calls']=graph.get('_feasibility_calls',0)+1
    if not anchors<=node_ids|edge_ids:return False
    if any(not set(graph['dependencies'][key])<=node_ids or not set(graph['required_edges'][key])<=edge_ids for key in node_ids):return False
    network=nx.Graph();network.add_nodes_from(node_ids)
    for key in edge_ids:
        item=graph['edges'][key]
        if item['source'] not in node_ids or item['target'] not in node_ids:return False
        if item['kind']=='flows_to' and item['data']['path_id'] not in node_ids:return False
        network.add_edge(item['source'],item['target'])
    if not node_ids or not nx.is_connected(network):return False
    context=_context(graph,node_ids,edge_ids)
    from skillrecon.detect.replay import replay_reconciliation
    result=replay_reconciliation(context)
    expected={key:graph['nodes'][key]['data'] for key in node_ids if graph['nodes'][key]['type']=='judgment'}
    certificates=[graph['nodes'][key]['data'] for key in node_ids if graph['nodes'][key]['type']=='certificate']
    expected_j,expected_c=_proof_signatures(expected,certificates)
    actual_j,actual_c=_proof_signatures(result.judgments,result.certificates)
    if not (set(expected_j.values())<=set(actual_j.values()) and set(expected_c.values())<=set(actual_c.values())):return False
    def projection_signature(raw,judgment_signatures):
        value={key:item for key,item in raw.items() if key not in {'edge_id','source_judgment_ids'}}
        value['source_judgment_ids']=sorted(judgment_signatures.get(ref,ref) for ref in raw['source_judgment_ids'])
        return json.dumps(value,sort_keys=True,ensure_ascii=False)
    actual_edges={projection_signature(edge.model_dump(mode='json'),actual_j) for edge in result.edges}
    for key in edge_ids:
        raw=graph['edges'][key]['data']
        if 'source_judgment_ids' in raw and projection_signature(raw,expected_j) not in actual_edges:return False
    return True


def _delete(graph,nodes,edges,identifier):
    if identifier in nodes:
        return nodes-{identifier},{key for key in edges if identifier not in {graph['edges'][key]['source'],graph['edges'][key]['target']}}
    return nodes,edges-{identifier}


def _rank(nodes,edges):
    return len(nodes),len(edges),tuple(sorted(nodes)),tuple(sorted(edges))


def _search(graph,nodes,edges,anchors,exact):
    forced_nodes,forced_edges=_closure(graph,anchors&nodes,anchors&edges)
    if exact:
        optional_nodes=sorted(nodes-forced_nodes)
        for count in range(len(optional_nodes)+1):
            best=None
            for chosen in combinations(optional_nodes,count):
                selected=forced_nodes|set(chosen)
                usable={key for key in edges if {graph['edges'][key]['source'],graph['edges'][key]['target']}<=selected}
                if not forced_edges<=usable:continue
                optional_edges=sorted(usable-forced_edges)
                for edge_count in range(len(optional_edges)+1):
                    feasible=[]
                    for chosen_edges in combinations(optional_edges,edge_count):
                        selected_edges=forced_edges|set(chosen_edges)
                        if _feasible(graph,selected,selected_edges,anchors):feasible.append((selected,selected_edges))
                    if feasible:
                        candidate=min(feasible,key=lambda item:_rank(*item))
                        if best is None or _rank(*candidate)<_rank(*best):best=candidate
                        break
            if best is not None:return best
        return nodes,edges
    proof=nx.DiGraph();proof.add_nodes_from(nodes)
    for key in edges:
        item=graph['edges'][key]
        if item['kind'] in {'premise_for','certified_by'}:proof.add_edge(item['source'],item['target'])
    order=list(nx.lexicographical_topological_sort(proof))+sorted(edges)
    current_nodes,current_edges=set(nodes),set(edges)
    for identifier in order:
        if identifier in anchors or identifier not in current_nodes|current_edges:continue
        candidate=_delete(graph,current_nodes,current_edges,identifier)
        if _feasible(graph,*candidate,anchors):current_nodes,current_edges=candidate
    return current_nodes,current_edges


def validate_graph_witness(graph,nodes,edges,anchors,witness_id):
    feasible=_feasible(graph,nodes,edges,anchors)
    reducible=None
    if feasible:
        for identifier in sorted((nodes|edges)-anchors):
            if _feasible(graph,*_delete(graph,nodes,edges,identifier),anchors):
                reducible=identifier;break
    return WitnessValidation(witness_id=witness_id,passed=feasible and reducible is None,
        failure_reason='mechanical_rederivation_failed' if not feasible else f'removable:{reducible}' if reducible else None,
        checked_anchor_ids=sorted(anchors),
        checked_certificate_ids=sorted(key for key in nodes if graph['nodes'][key]['type']=='certificate'),
        mechanical_revalidation_passed=feasible,operational_irreducibility_passed=feasible and reducible is None)


def validate_portable_graph(graph,witness,finding):
    """Rebuild structural obligations and rerun primitives from the exported W."""
    from skillrecon.core.types import ReconciliationJudgment,Certificate,ReconciliationEdge
    nodes=set(witness.fact_node_ids+witness.judgment_ids+witness.certificate_ids)
    edges=set(witness.graph_edge_ids)
    def reject(reason):
        return WitnessValidation(witness_id=witness.witness_id,passed=False,failure_reason=reason)
    if set(graph['nodes'])!=nodes or set(graph['edges'])!=edges:return reject('graph_payload_differs_from_selected_W')
    context=_context(graph,nodes,edges)
    judgments=[ReconciliationJudgment.model_validate(graph['nodes'][key]['data']) for key in witness.judgment_ids]
    certificates=[Certificate.model_validate(graph['nodes'][key]['data']) for key in witness.certificate_ids]
    projections=[ReconciliationEdge.model_validate(graph['edges'][key]['data']) for key in witness.projection_edge_ids]
    rebuilt=build_evidence_graph(context,judgments,certificates,projections)
    if any(key not in rebuilt['nodes'] or rebuilt['nodes'][key]!=graph['nodes'][key] for key in nodes):return reject('typed_source_node_mismatch')
    if any(key not in rebuilt['edges'] or rebuilt['edges'][key]!=graph['edges'][key] for key in edges):return reject('source_or_proof_edge_mismatch')
    _,_,anchors=_neighborhood(rebuilt,finding)
    if anchors!=set(witness.anchor_ids):return reject('motif_anchor_mismatch')
    return validate_graph_witness(rebuilt,nodes,edges,anchors,witness.witness_id)


def assemble_evidence_witnesses(findings,judgments,certificates,projections,inputs,*,n_max=18,m_max=36):
    from skillrecon.detect.witness import WitnessBundle,build_permission_manifest
    graph=build_evidence_graph(inputs,judgments,certificates,projections)
    witnesses,validations=[],[]
    for index,finding in enumerate(findings,1):
        graph['_feasibility_calls']=0
        nodes,edges,anchors=_neighborhood(graph,finding)
        exact=len(nodes)<=n_max and len(edges)<=m_max
        started=time.perf_counter()
        selected_nodes,selected_edges=_search(graph,nodes,edges,anchors,exact)
        search_seconds=time.perf_counter()-started
        identifier=f'witness-{index:04d}'
        started=time.perf_counter()
        validation=validate_graph_witness(graph,selected_nodes,selected_edges,anchors,identifier)
        acceptance_seconds=time.perf_counter()-started
        local_graph={'nodes':{key:graph['nodes'][key] for key in selected_nodes},
            'edges':{key:graph['edges'][key] for key in selected_edges},
            'dependencies':{key:sorted(graph['dependencies'][key]) for key in selected_nodes},
            'required_edges':{key:sorted(graph['required_edges'][key]) for key in selected_nodes},
            'protocol':graph['protocol'],'skill_id':graph['skill_id']}
        proof_ids={kind:[key for key in selected_nodes if graph['nodes'][key]['type']==kind] for kind in ['judgment','certificate']}
        witness=Witness(witness_id=identifier,finding_id=finding.finding_id,support_strength=finding.support_strength,
            anchor_ids=sorted(anchors),fact_node_ids=sorted(selected_nodes-set(proof_ids['judgment'])-set(proof_ids['certificate'])),
            judgment_ids=sorted(proof_ids['judgment']),certificate_ids=sorted(proof_ids['certificate']),
            projection_edge_ids=sorted(key for key in selected_edges if 'source_judgment_ids' in graph['edges'][key]['data']),
            graph_edge_ids=sorted(selected_edges),evidence_graph=local_graph,rank_tuple=(len(selected_nodes),len(selected_edges)),
            is_exact=exact,revalidation_passed=validation.passed,lex_id=repr(_rank(selected_nodes,selected_edges)[2:]),
            search_seconds=search_seconds,acceptance_seconds=acceptance_seconds,feasibility_checks=graph['_feasibility_calls'],
            neighborhood_nodes=len(nodes),neighborhood_edges=len(edges))
        witnesses.append(witness);validations.append(validation)
    wire={'nodes':list(graph['nodes'].values()),'edges':list(graph['edges'].values())}
    return WitnessBundle(witnesses,validations,build_permission_manifest(judgments,certificates),wire)
