"""Execution helpers for the canonical comparison protocol."""
from __future__ import annotations

import json
from pathlib import Path

from skillrecon.core.config import AnalyzerConfig, LLMConfig, load_llm_config
from skillrecon.evaluation.datasets import load_gold_label_records
from skillrecon.evaluation import baselines


def run_internal_baselines(gold_path:Path,artifacts:Path,output:Path,llm_path:Path|None,judge:bool):
    config=AnalyzerConfig(llm=load_llm_config(llm_path) if llm_path else LLMConfig(base_url='http://unused.invalid',model='no-llm'))
    mechanisms=[baselines.build_rule_based_scanner_report,baselines.build_capability_lattice_report,
                baselines.build_doc_code_consistency_report,baselines.build_spec_containment_report,
                baselines.build_instruction_constraint_report]
    if judge and llm_path is None:raise ValueError('--llm-judge requires --llm-config')
    output.mkdir(parents=True,exist_ok=True)
    if judge:mechanisms.append(baselines.build_llm_judge_report)
    predictions=[];failures=[]
    records=load_gold_label_records(gold_path)
    with (output/'predictions.jsonl').open('w',encoding='utf-8') as stream:
        for record in records:
            directory=artifacts/record.skill_id
            if not (directory/'event_table.json').is_file():
                raise FileNotFoundError(f'Analyze the package first: {directory}')
            coverage=json.loads((directory/'behavior_coverage.json').read_text())
            for mechanism in mechanisms:
                try:
                    report=mechanism(skill_id=record.skill_id,artifact_dir=directory,analyzer_config=config)
                except Exception as error:
                    failures.append({'skill_id':record.skill_id,'mechanism':mechanism.__name__,
                                     'error_type':type(error).__name__,'error':str(error)})
                    continue
                status='success' if mechanism is baselines.build_llm_judge_report else coverage['analysis_status']
                report=report.model_copy(update={'config':{**report.config,'analysis_status':status}})
                path=output/report.system_id/record.skill_id/'report.json'
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(report.model_dump_json(indent=2)+'\n',encoding='utf-8')
                prediction={'skill_id':report.skill_id,'system_id':report.system_id,
                    'main_label':report.overall_label,'subtype':None,
                    'metadata':{**report.config,'violation_subtypes':sorted({finding.subtype for finding in report.violation_findings if finding.subtype})}}
                predictions.append(prediction)
                stream.write(json.dumps(prediction,ensure_ascii=False)+'\n');stream.flush()
    (output/'run_summary.json').write_text(json.dumps({'planned':len(records)*len(mechanisms),
        'success':len(predictions),'failures':failures},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if failures:raise RuntimeError(f'{len(failures)} internal baseline analyses failed; see {output / "run_summary.json"}')
    return output/'predictions.jsonl'
