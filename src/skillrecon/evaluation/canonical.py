"""Evaluation of extraction, detection, discovery, witnesses, and generalization."""
from __future__ import annotations

import csv
import json
import re
from statistics import median
from collections import Counter, defaultdict
from pathlib import Path

from skillrecon.evaluation.stats import bootstrap_resample_ci, exact_binomial_ci, mcnemar_exact, holm_adjusted

BASELINES={
 'baseline_rule_scanner':'Rule-based scanner', 'baseline_llm_judge':'LLM-as-judge',
 'baseline_capability_lattice':'Capability-lattice', 'baseline_doc_code_consistency':'Doc-code consistency',
 'baseline_spec_containment':'Spec containment', 'baseline_instruction_constraints':'Instruction constraints',
 'baseline_openclaw':'OpenClaw Scan', 'baseline_skillfortify':'SkillFortify',
 'baseline_cisco_skill_scanner':'Cisco Skill Scanner', 'baseline_skillspector':'SkillSpector',
 'baseline_snyk_agent_scan':'Snyk Agent Scan', 'baseline_malskills_full':'MalSkills'}
TYPES={'V1':{'V1','unsupported_behavior','contradicted_behavior','undeclared_high_impact_capability'},
       'V2':{'V2','scope_violation'}, 'V3':{'V3','unjustified_composition','unjustified_dangerous_composition'}}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def categories(values):
    return {category for category, aliases in TYPES.items() if aliases.intersection(value for value in values if value)}


def prf(tp,fp,fn,tn=0):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'n':tp+fp+fn+tn,'precision':p,'recall':r,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}


def binary_metrics(ids,gold,predictions):
    counts=Counter((gold[key]['positive'],predictions[key]['positive']) for key in ids)
    result=prf(counts[True,True],counts[False,True],counts[True,False],counts[False,False])
    if not result['n']:
        result.update(precision=None,recall=None,f1=None)
    return result


def category_metrics(ids,gold,predictions,*,native_types,benign_ids=None):
    result={}
    benign=({key for key in ids if not gold[key]['positive'] and gold[key]['slice']=='medium'}
            if benign_ids is None else set(ids)&set(benign_ids))
    for category in TYPES:
        positive={key for key in ids if category in gold[key]['types']}
        tp=sum(predictions[key]['positive'] for key in positive)
        fp=sum(predictions[key]['positive'] for key in benign)
        bf=prf(tp,fp,len(positive)-tp)
        strict=None
        if native_types:
            predicted={key for key in ids if category in predictions[key]['types']}
            strict=prf(len(positive&predicted),len(predicted-positive),len(positive-predicted))
        result[category]={'positive_skills':len(positive),'benign_shared':len(benign),
                          'strict':strict,'broadcast_free':bf,'tcr':bf['recall']}
    return result


def _prediction(item):
    metadata=item.get('metadata',item.get('config',{}))
    label=item.get('main_label',item.get('overall_label'))
    if label not in {'violation','exposure-only','benign'}:
        raise ValueError('Prediction requires main_label/overall_label')
    native=metadata.get('native_flag')
    subtype_values=[item.get('subtype'),*metadata.get('violation_subtypes',[])]
    subtype_values += [finding.get('subtype') for finding in item.get('violation_findings',[])]
    return {'positive':bool(native) if isinstance(native,bool) else label=='violation',
            'types':categories(subtype_values),'exposure':label=='exposure-only',
            'success':metadata.get('analysis_status','success') in {'success','completed_degraded'}}


def _gold(item):
    gold=item['gold']
    metadata=item.get('metadata',{})
    slice_name=str(item.get('risk_stratum') or metadata.get('slice') or 'unspecified').lower()
    slice_name='low' if slice_name=='no_risk' else slice_name.removesuffix('_risk')
    return {'positive':gold['label']=='violation','types':categories([gold.get('violation_subtype'),*gold.get('violation_subtypes',[])]),
            'slice':slice_name,
            'platform':str(item.get('platform') or metadata.get('platform') or '').lower().replace('-','_'),
            'words':metadata.get('documentation_words'),
            'language':metadata.get('language_profile'),'metadata':metadata}


def _join_path(base,value):
    path=Path(value)
    return path if path.is_absolute() else base/path


def _system_predictions(spec,base,ids):
    predictions={}
    if 'predictions' in spec:
        for item in load_jsonl(_join_path(base,spec['predictions'])):
            if spec.get('system_id') and item.get('system_id')!=spec['system_id']:continue
            key=item['skill_id']
            if key in predictions:raise ValueError(f'Duplicate prediction: {key}')
            prediction=_prediction(item)
            if prediction['success']:predictions[key]=prediction
    if 'reports' in spec:
        for key in ids:
            path=_join_path(base,spec['reports'])/key/'report.json'
            if path.is_file():predictions[key]=_prediction(load_json(path))
    if 'artifacts' in spec:
        from skillrecon.evaluation.skillrecon_adapter import build_skillrecon_report
        for key in ids:
            path=_join_path(base,spec['artifacts'])/key
            if (path/'findings.json').is_file():
                predictions[key]=_prediction(build_skillrecon_report(key,path).model_dump(mode='json'))
    return {key:value for key,value in predictions.items() if key in ids and value['success']}


def _paired_stats(ids,gold,systems,main):
    rows={}
    for name,predictions in systems.items():
        if name==main:continue
        common=sorted(set(ids)&set(systems[main])&set(predictions))
        a=[systems[main][key]['positive']==gold[key]['positive'] for key in common]
        b=[predictions[key]['positive']==gold[key]['positive'] for key in common]
        n10=sum(x and not y for x,y in zip(a,b))
        n01=sum(y and not x for x,y in zip(a,b))
        rows[name]={'n':len(common),'main_only_correct':n10,'other_only_correct':n01,
                    'accuracy_difference':(n10-n01)/len(common) if common else None,
                    'p_two_sided':mcnemar_exact(a,b) if common else None}
    adjusted=holm_adjusted({key:row['p_two_sided'] for key,row in rows.items() if row['p_two_sided'] is not None})
    for key,value in adjusted.items():rows[key].update(p_holm=value,reject_0_05=value<=0.05)
    return rows


def _witness_rows(roots,base,ids):
    rows={}
    from skillrecon.core.enums import FindingType
    type_map={FindingType.UNSUPPORTED_BEHAVIOR.value:'V1',FindingType.CONTRADICTED_BEHAVIOR.value:'V1',
              FindingType.SCOPE_VIOLATION.value:'V2',FindingType.UNJUSTIFIED_COMPOSITION.value:'V3'}
    for category in [*TYPES,'all']:
        counts=Counter()
        flagged_packages=set();exact_packages=set();greedy_packages=set();certified_packages=set()
        search_seconds=[];acceptance_seconds=[]
        for root in roots:
            for key in ids:
                directory=_join_path(base,root)/key
                if not (directory/'witness_validation.json').is_file():continue
                findings={item['finding_id']:item for item in load_json(directory/'findings.json')}
                validations={item['witness_id']:item for item in load_json(directory/'witness_validation.json')}
                witnesses=load_json(directory/'witnesses.json')+load_json(directory/'rejected_witnesses.json')
                selected=[]
                for witness in witnesses:
                    if category!='all' and type_map[findings[witness['finding_id']]['finding_type']]!=category:continue
                    selected.append(witness)
                    counts['candidates']+=1
                    validation=validations[witness['witness_id']]
                    counts['revalidated']+=validation.get('mechanical_revalidation_passed',validation['passed'])
                    counts['irreducible']+=validation.get('operational_irreducibility_passed',validation['passed'])
                    counts['accepted']+=validation['passed']
                    counts['exact_attempted']+=witness['is_exact']
                    counts['greedy_attempted']+=not witness['is_exact']
                    counts['exact']+=witness['is_exact'] and validation['passed']
                    if witness.get('search_seconds') is not None:search_seconds.append(witness['search_seconds'])
                    if witness.get('acceptance_seconds') is not None:acceptance_seconds.append(witness['acceptance_seconds'])
                finding_count=sum(category=='all' or type_map[item['finding_type']]==category for item in findings.values())
                counts['findings']+=finding_count
                if finding_count:
                    flagged_packages.add(key)
                    if any(w['is_exact'] for w in selected):exact_packages.add(key)
                    if any(not w['is_exact'] for w in selected):greedy_packages.add(key)
                    if len(selected)==finding_count and all(w['is_exact'] and validations[w['witness_id']]['passed'] for w in selected):
                        certified_packages.add(key)
        n=counts['candidates']
        rows[category]={**dict(counts),'rejected':n-counts['accepted'],
            'coverage':n/counts['findings'] if counts['findings'] else None,
            'reval_rate':counts['revalidated']/n if n else None,
            'irred_rate':counts['irreducible']/n if n else None,'exact_rate':counts['exact']/n if n else None,
            'exact_attempt_rate':counts['exact_attempted']/n if n else None,
            'greedy_attempt_rate':counts['greedy_attempted']/n if n else None,
            'packages':{'flagged':len(flagged_packages),'with_exact_attempt':len(exact_packages),
                'with_greedy_attempt':len(greedy_packages),'all_findings_certified_exact':len(certified_packages)},
            'mean_search_seconds':sum(search_seconds)/len(search_seconds) if search_seconds else None,
            'mean_acceptance_seconds':sum(acceptance_seconds)/len(acceptance_seconds) if acceptance_seconds else None,
            'grounding':None}  # Grounding is an independent annotation.
    return rows


_PRIMITIVE_KINDS={'capability_overlap','resource_compatibility','scope_check','prohibition_check','route_check','path_justification'}


def _predicate_tallies(directory):
    judgments=load_json(directory/'judgment_table.json')
    primitives=[j for j in judgments if j['kind'] in _PRIMITIVE_KINDS]
    routes=[j for j in primitives if j['kind']=='route_check']
    return Counter(predicate_checks=len(primitives),abstain=sum(j['result']=='abstain' for j in primitives),
                   route_checks=len(routes),unconfirmed_routes=sum(j['result']=='abstain' for j in routes))


def _package_language(directory):
    units=load_json(directory/'package_manifest.json')['code_units']
    return _language_profile({u['language'] for u in units if u.get('source_doc_id') is None})


def _coverage(roots,base,ids,*,gold=None,predictions=None):
    totals=Counter();languages=defaultdict(Counter);tiers=Counter()
    profiles={name:Counter() for name in LANGUAGE_PROFILES}
    profile_tiers={name:Counter() for name in LANGUAGE_PROFILES}
    missing_predicates=set()
    for root in roots:
        for key in ids:
            directory=_join_path(base,root)/key
            if not (directory/'behavior_coverage.json').is_file():continue
            data=load_json(directory/'behavior_coverage.json')
            profile=(gold[key]['language'] if gold and gold[key]['language'] else _package_language(directory))
            profile=profile if profile in profiles else 'Other'
            counts=Counter(packages=1,codeql_applicable=int(bool(data['codeql'])),
                codeql_success=int(bool(data['codeql']) and all(run['status']=='success' for run in data['codeql'])),
                codeql_degraded=int(bool(data['codeql']) and any(run['status']!='success' for run in data['codeql'])))
            contribution={str(k):v for k,v in data['tier_counts'].items()}
            tiers.update(contribution);profile_tiers[profile].update(contribution)
            for run in data['codeql']:
                languages[run['language']]['applicable']+=1
                languages[run['language']]['success']+=run['status']=='success'
            resources=load_json(directory/'resource_table.json')
            counts['resources']=len(resources)
            counts['unresolved_resources']=sum(not item.get('resolved',False) for item in resources)
            if (directory/'judgment_table.json').is_file():
                counts.update(_predicate_tallies(directory))
                counts['predicate_measured_packages']=1
            else:
                missing_predicates.add(key)
            if gold is not None and predictions is not None:
                counts['false_negatives']=int(key in predictions and gold[key]['positive'] and not predictions[key]['positive'])
            totals.update(counts);profiles[profile].update(counts)
    def summarize(counts,contribution):
        def ratio(a,b):return counts[a]/counts[b] if counts[b] else None
        measured={key:counts[key] for key in ['packages','codeql_applicable','codeql_success','codeql_degraded',
                    'resources','unresolved_resources','predicate_measured_packages','predicate_checks','abstain','route_checks','unconfirmed_routes','false_negatives']}
        return {**measured,'tier_counts':dict(contribution),
                'tier_shares':{key:value/sum(contribution.values()) for key,value in contribution.items()} if sum(contribution.values()) else {},
                'codeql_success_rate':ratio('codeql_success','codeql_applicable'),
                'unresolved_resource_rate':ratio('unresolved_resources','resources'),
                'abstain_rate':ratio('abstain','predicate_checks'),
                'unconfirmed_route_rate':ratio('unconfirmed_routes','route_checks')}
    return {**summarize(totals,tiers),
            'languages':{key:dict(value) for key,value in languages.items()},
            'language_profiles':{name:summarize(profiles[name],profile_tiers[name]) for name in LANGUAGE_PROFILES},
            'missing_predicate_skills':sorted(missing_predicates)}


def _documentation_quality(ids,gold,systems,roots,base):
    from skillrecon.contract.voting import has_concrete_boundary
    from skillrecon.core.types import Clause
    slices={**_documentation_slices(ids,gold),'Overall':set(ids)}
    result={}
    for name,selected in slices.items():
        totals=Counter();available=set()
        for root in roots:
            for key in selected:
                directory=_join_path(base,root)/key
                if not (directory/'canonical_clauses.json').is_file():continue
                clauses=[Clause.model_validate(c) for c in load_json(directory/'canonical_clauses.json') if c['role']=='policy']
                available.add(key);totals['policy_skills']+=bool(clauses)
                totals['scoped_skills']+=any(has_concrete_boundary(c) or any(
                    s.constraint_type in {'ext','scheme','port','provenance','account_active','config_enabled','read_only',
                        'side_effect_bound','no_auto_retry','trigger','lifetime','source_class','sink_domain','separate_invocation','arg_pattern'}
                    and s.value.strip() not in {'','*','**'} for s in c.constraints) for c in clauses)
                totals.update(_predicate_tallies(directory))
        words=[gold[key]['words'] for key in selected if gold[key]['words'] is not None]
        result[name]={'n':len(selected),'share':len(selected)/len(ids) if ids else None,
            'words_median':median(words) if words else None,'quality_measured_skills':len(available),
            'policy_skills':totals['policy_skills'],'policy_rate':totals['policy_skills']/len(available) if available else None,
            'scoped_skills':totals['scoped_skills'],'scope_rate':totals['scoped_skills']/len(available) if available else None,
            'predicate_checks':totals['predicate_checks'],
            'abstain_rate':totals['abstain']/totals['predicate_checks'] if totals['predicate_checks'] else None,
            'systems':{name:binary_metrics(selected,gold,values) for name,values in systems.items()}}
    return result


def run_measurements(manifest_path: Path,output: Path):
    config=load_json(manifest_path);base=manifest_path.resolve().parent
    records=load_jsonl(_join_path(base,config['gold']))
    gold={item['skill_id']:_gold(item) for item in records}
    if len(gold)!=len(records):raise ValueError('Gold skill IDs must be unique')
    systems={name:_system_predictions(spec,base,gold) for name,spec in config['systems'].items()}
    if not systems:raise ValueError('At least one system is required')
    main=config.get('main_system','skillrecon')
    if main not in systems:raise ValueError('main_system must appear in systems')
    roots=[config['systems'][main]['artifacts']] if 'artifacts' in config['systems'][main] else []
    # Fill metadata from the admitted documentation closure, without editing gold.
    for root in roots:
        for key in gold:
            directory=_join_path(base,root)/key
            if gold[key]['language']:
                gold[key]['language']=_canonical_language_profile(gold[key]['language'])
            if (directory/'step_table.json').is_file() and gold[key]['words'] is None:
                steps=load_json(directory/'step_table.json')
                gold[key]['words']=sum(len(re.findall(r'\b\w+\b',step['text'])) for step in steps if step['step_type'] not in {'bound_code_block','example_code_block'})
            if (directory/'package_manifest.json').is_file() and not gold[key]['language']:
                units=load_json(directory/'package_manifest.json')['code_units']
                languages={unit['language'] for unit in units if unit.get('source_doc_id') is None}
                gold[key]['language']=_language_profile(languages)
    requested=set(gold)
    common=requested.intersection(*(set(predictions) for predictions in systems.values()))
    result={'schema_version':1,'protocol':'SkillRecon','measured_skill_ids':sorted(common),
            'requested_skills':len(gold),'common_successful_skills':len(common),
            'coverage':{name:{'success':len(values),'missing':sorted(requested-set(values))} for name,values in systems.items()}}
    highmedium={key for key in common if gold[key]['slice'] in {'high','medium'}}
    result['table7']={name:{slice_name:binary_metrics({key for key in highmedium if gold[key]['slice']==slice_name},gold,values)
                          for slice_name in ['high','medium']} for name,values in systems.items()}
    for name,row in result['table7'].items():
        medium=row['medium'];den=medium['fp']+medium['tn']
        row['medium_fp_recovery']=medium['tn']/den if den else None
        row['medium_false_positives_retracted']={'corrected':medium['tn'],'benign_controls':den,'rate':row['medium_fp_recovery']}
    result['table8']={name:category_metrics(highmedium,gold,values,native_types=config['systems'][name].get('native_types',name==main)) for name,values in systems.items()}
    result['table9']={name:result['table7'][name] for name in systems if name==main or name.startswith('ablation_')}
    result['table10']=_documentation_quality(common,gold,systems,roots,base)
    low={key for key in common if gold[key]['slice'] in {'low','no_risk'}}
    result['table12']={}
    for name,values in systems.items():
        counts=binary_metrics(low,gold,values)
        result['table12'][name]={**counts,'flagged':counts['tp']+counts['fp'],'confirmed':counts['tp'],
            'precision_ci95':exact_binomial_ci(counts['tp'],counts['tp']+counts['fp']),
            'recall_ci95':exact_binomial_ci(counts['tp'],counts['tp']+counts['fn']),
            'exposure_only':sum(values[key]['exposure'] for key in low),
            'confirmed_by_type':{category:sum(values[key]['positive'] and gold[key]['positive'] and category in gold[key]['types'] for key in low) for category in TYPES}}
    rq4_ids={key for key in requested if gold[key]['slice'] in {'high','medium'}}
    result['table13']=_witness_rows(roots,base,rq4_ids)
    result['table14']=_coverage(roots,base,requested,gold=gold,predictions=systems[main])
    missed={key for key in systems[main] if gold[key]['positive'] and not systems[main][key]['positive']}
    result['table14']['false_negatives']=len(missed)
    result['table14']['fn_attribution']=dict(Counter(gold[key]['metadata'].get('false_negative_reason','unannotated') for key in missed))
    if config.get('witness_grounding'):
        annotations=load_jsonl(_join_path(base,config['witness_grounding']))
        for category,row in result['table13'].items():
            records=[item for item in annotations if category=='all' or item['category']==category]
            row['grounding_annotated']=len(records)
            row['grounding']=sum(item['grounded'] for item in records)/len(records) if records else None
    result['table15']={}
    for platform in ['claude_code','opencode','combined']:
        ids={key for key in common if (gold[key]['platform'] in {'claude_code','opencode'} if platform=='combined' else gold[key]['platform']==platform)}
        benign={key for key in ids if not gold[key]['positive']}
        result['table15'][platform]={'n':len(ids),'systems':{name:{**binary_metrics(ids,gold,values),
                'categories':category_metrics(ids,gold,values,native_types=config['systems'][name].get('native_types',name==main),benign_ids=benign)}
                for name,values in systems.items()},'paired_mcnemar':_paired_stats(ids,gold,systems,main)}
        for name,row in result['table15'][platform]['systems'].items():
            row['flagged']=row['tp']+row['fp']
            row['witness_revalidation']=_witness_rows(roots,base,ids)['all'] if name==main else None
        privileged={key for key in ids if gold[key]['metadata'].get('benign_high_privilege') is True and not gold[key]['positive']}
        result['table15'][platform]['benign_high_privilege']={'n':len(privileged),
            'systems':{name:binary_metrics(privileged,gold,values) for name,values in systems.items()}}
    result['paired_mcnemar_high_medium']=_paired_stats(highmedium,gold,systems,main)
    result.update(_rq1(config,base,roots))
    variant_names=set(config.get('backbone_systems',[]))|{row['system'] for row in config.get('sensitivity',[])}
    result['variant_metrics']={}
    for name in sorted(variant_names):
        if name not in systems:raise ValueError(f'Variant system not measured: {name}')
        spec=config['systems'][name]
        variant_roots=[spec['artifacts']] if 'artifacts' in spec else []
        annotation_config=config if name==main else spec
        objects=_rq1(annotation_config,base,variant_roots)
        witnesses=_witness_rows(variant_roots,base,requested)['all']
        exact=witnesses['exact_rate']
        result['variant_metrics'][name]={
            'overall':binary_metrics(common,gold,systems[name]),
            'medium':result['table7'][name]['medium'],
            'false_authorization_rate':objects['table4'].get('false_auth_fractions',{}).get('raw'),
            'event_recall':objects['table6'].get('events',{}).get('recall'),
            'exact_share':exact,'exact_attempt_share':witnesses['exact_attempt_rate'],
            'greedy_share':witnesses['greedy_attempt_rate']}
    result['table11']=_robustness(config,base,gold,systems,main)
    result['language_profiles']={profile:{name:binary_metrics({key for key in highmedium if gold[key]['language']==profile},gold,values) for name,values in systems.items()} for profile in LANGUAGE_PROFILES}
    result['sensitivity']=config.get('sensitivity',[])
    result['error_attribution']=config.get('error_annotations',[])
    output.mkdir(parents=True,exist_ok=True)
    (output/'measurements.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    _write_tables(result,output)
    return result


LANGUAGE_PROFILES=['Python','JS/TS','Bash','Py+Bash','JS/TS+Bash','Multi>=3','Other']
def _canonical_language_profile(value):
    aliases={'Python-only':'Python','JS/TS-only':'JS/TS','Bash-only':'Bash',
             'Python + Bash':'Py+Bash','JS/TS + Bash':'JS/TS+Bash','Multi (>=3 langs)':'Multi>=3','Config / Other':'Other'}
    return value if value in LANGUAGE_PROFILES else aliases.get(value,'Other')


def _language_profile(languages):
    groups=set('Python' if x=='python' else 'JS/TS' if x in {'javascript','typescript'} else 'Bash' if x=='bash' else x for x in languages)
    if len(groups)>=3:return 'Multi>=3'
    return {frozenset({'Python'}):'Python',frozenset({'JS/TS'}):'JS/TS',frozenset({'Bash'}):'Bash',
            frozenset({'Python','Bash'}):'Py+Bash',frozenset({'JS/TS','Bash'}):'JS/TS+Bash'}.get(frozenset(groups),'Other')


def _documentation_slices(ids,gold):
    return {name:{key for key in ids if gold[key]['words'] is not None and low<=gold[key]['words']<high}
            for name,low,high in [('Well-specified',350,float('inf')),('Moderate',100,350),('Sparse',0,100)]}


def _rq1(config,base,roots):
    annotation_path=config.get('rq1_clauses')
    unavailable={'status':'unavailable','reason':'Provide located clause/operator/role annotations in rq1_clauses'}
    object_path=config.get('rq1_objects')
    table6=(_object_metrics(load_jsonl(_join_path(base,object_path)),roots,base) if object_path else
            {'status':'unavailable','reason':'Provide rq1_objects with gold object keys'})
    if annotation_path is None:return {'table4':unavailable,'table5':unavailable,'table6':table6}
    annotations=load_jsonl(_join_path(base,annotation_path))
    gold={(row['skill_id'],row['clause_id']):row['gold'] for row in annotations}
    snapshots={name:{} for name in ['raw','classified','guarded','final']}
    deterministic={}
    edges={}
    for root in roots:
        for skill in {key[0] for key in gold}:
            directory=_join_path(base,root)/skill
            if (directory/'deterministic_clauses.json').is_file():
                deterministic.update({(skill,c['clause_id']):c for c in load_json(directory/'deterministic_clauses.json')})
            files={'raw':'raw_canonical_clauses.json','classified':'classified_clauses.json',
                   'guarded':'guarded_clauses.json','final':'contract_table.json'}
            for stage,filename in files.items():
                if not (directory/filename).is_file():continue
                payload=load_json(directory/filename)
                clauses=payload['clauses'] if isinstance(payload,dict) else payload
                snapshots[stage].update({(skill,clause['clause_id']):clause for clause in clauses})
            edges[skill]=load_json(directory/'reconciliation_edges.json')
    raw=snapshots['raw']
    all_raw_allowed={key for key,clause in raw.items() if clause['operator']=='allowed'}
    raw_allowed=all_raw_allowed&set(gold)
    false={key for key in raw_allowed if gold[key]['operator'] in {'unknown','prohibited'}}
    classified={key for key,clause in snapshots['classified'].items() if clause['role']=='policy'}
    classified|={key for key,clause in deterministic.items() if clause['role']=='policy'}
    guarded={key for key,clause in snapshots['guarded'].items() if clause['role']=='policy' and clause['operator']=='allowed'}
    guarded|={key for key,clause in deterministic.items() if clause['operator']=='allowed'}
    support=set();effective=set()
    for skill,relations in edges.items():
        blocked={(edge.get('event_id'),edge.get('binding_resource_id') or edge.get('resource_id')) for edge in relations if edge['relation']=='contradicts'}
        for edge in relations:
            if edge['relation']!='supports':continue
            key=(skill,edge['clause_id']);support.add(key)
            if (edge.get('event_id'),edge.get('binding_resource_id') or edge.get('resource_id')) not in blocked:effective.add(key)
    denom=len(all_raw_allowed)
    complete_allowed_annotation=raw_allowed==all_raw_allowed
    gate1=false-classified;gate2=(false&classified)-guarded
    gate3=(false&classified&guarded)-support;gate4=(false&classified&guarded&support)-effective
    residual=false&classified&guarded&support&effective
    operator_metrics={}
    for operator in ['allowed','prohibited','unknown']:
        positives={key for key,row in gold.items() if row['operator']==operator and key not in deterministic}
        predicted={key for key,row in raw.items() if key in gold and row['operator']==operator}
        operator_metrics[operator]=prf(len(positives&predicted),len(predicted-positives),len(positives-predicted))
    macro_p=sum(row['precision'] for row in operator_metrics.values())/3
    raw_gold_count=sum(row['tp']+row['fn'] for row in operator_metrics.values())
    micro_r=sum(row['tp'] for row in operator_metrics.values())/raw_gold_count if raw_gold_count else 0
    governing_gold={key for key,row in gold.items() if row['role']=='policy'}
    governing={key for key,row in snapshots['final'].items() if row['role']=='policy' and key in gold}
    table4={'raw_operators':operator_metrics,'raw_overall_macro_precision':macro_p,'raw_overall_micro_recall':micro_r,
            'raw_overall_f1':2*macro_p*micro_r/(macro_p+micro_r) if macro_p+micro_r else 0,
            'governing_policy':prf(len(governing&governing_gold),len(governing-governing_gold),len(governing_gold-governing)),
            'raw_allowed_denominator':denom,'false_authorizations':len(false),
            'annotated_raw_allowed':len(raw_allowed),'complete_raw_allowed_annotation':complete_allowed_annotation,
            'false_auth_fractions':{'raw':len(false)/denom if denom and complete_allowed_annotation else None,
               'governing':len(false&classified&guarded)/denom if denom and complete_allowed_annotation else None,
               'effective':len(residual)/denom if denom and complete_allowed_annotation else None},
            'annotated':len(gold),'unannotated_emitted':len(set(raw)-set(gold))}
    table4['raw_iccm_emitted']=len(raw)
    table4['deterministic_emitted']=len(deterministic)
    role_counts=Counter((gold[key]['role'],clause['role']) for key,clause in snapshots['classified'].items() if key in gold)
    for key,clause in deterministic.items():
        if key in gold:role_counts[gold[key]['role'],clause['role']]+=1
    role_scores={}
    for role in ['policy','knowledge']:
        opposite='knowledge' if role=='policy' else 'policy'
        role_scores[role]=prf(role_counts[role,role],role_counts[opposite,role],role_counts[role,opposite])
    table5={'confusion':{role:{pred:role_counts[role,pred] for pred in ['policy','knowledge']} for role in ['policy','knowledge']},
            'role_metrics':role_scores,'macro_f1':sum(row['f1'] for row in role_scores.values())/2,
            'false_allowed_funnel':{'initial':len(false),'role_grounding_removed':len(gate1),'guard_removed':len(gate2),
             'predicates_removed':len(gate3),'prohibition_removed':len(gate4),'remaining':len(residual),
             'neutralization_rate':1-len(residual)/len(false) if false else None}}
    if 'supports' in table6:table4['effective_support']=table6['supports']
    return {'table4':table4,'table5':table5,'table6':table6}


def _object_metrics(annotations,roots,base):
    """Each annotation lists a skill, artifact filename, key_fields and all gold keys."""
    grouped=defaultdict(lambda:defaultdict(lambda:[0,0,0]))
    for row in annotations:
        predicted=set()
        for root in roots:
            path=_join_path(base,root)/row['skill_id']/row['artifact']
            if not path.is_file():continue
            items=load_json(path)
            blocked={(item.get('event_id'),item.get('binding_resource_id') or item.get('resource_id'))
                     for item in items if item.get('relation')=='contradicts'}
            for item in items:
                if row.get('relation') and item.get('relation')!=row['relation']:continue
                if row['kind']=='supports' and (item.get('event_id'),item.get('binding_resource_id') or item.get('resource_id')) in blocked:continue
                predicted.add(json.dumps([item.get(field) for field in row['key_fields']],sort_keys=True))
        gold={json.dumps(key,sort_keys=True) for key in row['gold_keys']}
        counts=(len(gold&predicted),len(predicted-gold),len(gold-predicted))
        for index,value in enumerate(counts):grouped[row['kind']][row['skill_id']][index]+=value
    result={}
    for kind,by_skill in grouped.items():
        counts=list(by_skill.values())
        score=prf(*(sum(row[i] for row in counts) for i in range(3)))
        score['skills']=len(counts)
        score['ci95']={metric:list(bootstrap_resample_ci(counts,lambda sample:prf(*(sum(row[i] for row in sample) for i in range(3)))[metric],seed=0))
                       for metric in ['precision','recall','f1']}
        result[kind]=score
    return result


def _perturbation_mechanisms(spec,base,ids):
    if not spec.get('mechanism_annotations'):
        return {'status':'unavailable','reason':'Provide reviewed clause/constraint/event targets in mechanism_annotations'}
    if 'artifacts' not in spec:
        raise ValueError('Mechanism evaluation requires variant artifacts')
    rows=load_jsonl(_join_path(base,spec['mechanism_annotations']))
    scores={name:Counter() for name in ['guard_downgrade','scope_abstain','prohibition_block']}
    for row in rows:
        if row['skill_id'] not in ids:continue
        directory=_join_path(base,spec['artifacts'])/row['skill_id']
        before={c['clause_id']:c for c in load_json(directory/'raw_canonical_clauses.json')}
        after={c['clause_id']:c for c in load_json(directory/'guarded_clauses.json')}
        judgments=load_json(directory/'judgment_table.json')
        certificates=load_json(directory/'certificate_table.json')
        observed={
            'guard_downgrade':any(before.get(key,{}).get('operator')=='allowed' and
                after.get(key,{}).get('operator')=='unknown' for key in row.get('clause_ids',[])),
            'scope_abstain':bool(row.get('constraint_ids')) and all(any(
                j['kind']=='scope_check' and key in j['subject_refs'] and j['result']=='abstain'
                for j in judgments) for key in row.get('constraint_ids',[])),
            'prohibition_block':any(c['kind']=='blocking_prohibition' and
                set(row.get('event_ids',[])).intersection(c['subject_refs']) for c in certificates)}
        for name,expected in row['gold'].items():
            if name not in scores:raise ValueError(f'Unknown perturbation mechanism: {name}')
            scores[name][bool(expected),observed[name]]+=1
    return {name:prf(c[True,True],c[False,True],c[True,False],c[False,False]) for name,c in scores.items()}


def _robustness(config,base,gold,systems,main):
    result={};totals=Counter();base_totals=Counter();agreements=0
    for spec in config.get('robustness',[]):
        records=load_jsonl(_join_path(base,spec['gold']))
        variant_gold={item['skill_id']:_gold(item) for item in records}
        if len(variant_gold)!=len(records):raise ValueError('Variant gold skill IDs must be unique')
        predictions=_system_predictions(spec,base,variant_gold)
        requested=set(gold)&set(variant_gold)
        common=requested&set(systems[main])&set(predictions)
        original=binary_metrics(common,gold,systems[main]);variant=binary_metrics(common,variant_gold,predictions)
        agreement=sum(predictions[key]['positive']==systems[main][key]['positive'] for key in common)
        changed=sum(gold[key]['positive']!=variant_gold[key]['positive'] for key in common)
        result[spec['name']]={'skills':len(common),'base':original,'variant':variant,
            'requested_skills':len(requested),
            'missing_original_predictions':sorted(requested-set(systems[main])),
            'missing_variant_predictions':sorted(requested-set(predictions)),
            'f1_change':variant['f1']-original['f1'] if common else None,'gold_changed':changed,
            'decision_agreement':agreement/len(common) if common else None,
            'mechanisms':_perturbation_mechanisms(spec,base,common) if common else
                {'status':'unavailable','reason':'No common successful predictions'}}
        totals.update({key:variant[key] for key in ['tp','fp','fn','tn']})
        base_totals.update({key:original[key] for key in ['tp','fp','fn','tn']})
        totals['gold_changed']+=changed;agreements+=agreement
        totals['requested']+=len(requested)
    if result:
        variant=prf(*(totals[key] for key in ['tp','fp','fn','tn']))
        original=prf(*(base_totals[key] for key in ['tp','fp','fn','tn']))
        if not variant['n']:
            variant.update(precision=None,recall=None,f1=None)
            original.update(precision=None,recall=None,f1=None)
        result['Overall']={'skills':variant['n'],'base':original,'variant':variant,
            'requested_skills':totals['requested'],
            'f1_change':variant['f1']-original['f1'] if variant['n'] else None,'gold_changed':totals['gold_changed'],
            'decision_agreement':agreements/variant['n'] if variant['n'] else None}
    return result


def _flatten(value,prefix=''):
    if isinstance(value,dict):
        for key,item in value.items():yield from _flatten(item,f'{prefix}.{key}' if prefix else key)
    elif isinstance(value,list):
        yield prefix,json.dumps(value,ensure_ascii=False)
    else:yield prefix,value


def _write_tables(result,output):
    tables=output/'metrics';tables.mkdir(exist_ok=True)
    for number in range(4,16):
        payload=result[f'table{number}']
        (tables/f'table{number}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        rows=list(_flatten(payload))
        with (tables/f'table{number}.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.writer(handle);writer.writerow(['measurement','value']);writer.writerows(rows)
