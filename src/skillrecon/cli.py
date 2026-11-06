"""Command-line interface for SkillRecon analysis and experiments."""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from skillrecon.core.config import PROJECT_ROOT

SCRIPTS={
    'analyze':('run_full.py','Analyze one skill package'),
    'batch':('run_batch.py','Analyze a collection of skill packages'),
    'prepare':('prepare_dataset.py','Prepare skill sources from a dataset catalog'),
    'baselines':('run_external_baselines.py','Run or import external baseline scanners'),
    'ablate':('build_ablation_artifacts.py','Run the four ablation configurations'),
    'sweep':('run_canonical_sweep.py','Run a hyperparameter grid'),
}


def main():
    if len(sys.argv)>1 and sys.argv[1] in SCRIPTS:
        name=SCRIPTS[sys.argv[1]][0]
        sys.argv=[name,*sys.argv[2:]]
        runpy.run_path(str(PROJECT_ROOT/'scripts'/name),run_name='__main__')
        return
    parser=argparse.ArgumentParser(description='SkillRecon: documentation–behavior reconciliation for agent skills')
    subs=parser.add_subparsers(dest='command',required=True)
    for name,(_,description) in SCRIPTS.items():subs.add_parser(name,help=description)
    evaluate=subs.add_parser('evaluate',help='Compute research-question metrics and paired statistical comparisons')
    evaluate.add_argument('--manifest',type=Path,required=True)
    evaluate.add_argument('--output',type=Path,required=True)
    replay=subs.add_parser('replay',help='Validate an inconsistency witness')
    replay.add_argument('bundle',type=Path)
    perturb=subs.add_parser('perturb',help='Generate controlled documentation variants')
    perturb.add_argument('--plan',type=Path,required=True)
    perturb.add_argument('--source-root',type=Path,required=True)
    perturb.add_argument('--output-root',type=Path,required=True)
    internal=subs.add_parser('internal-baselines',help='Run internal comparison methods')
    internal.add_argument('--gold',type=Path,required=True)
    internal.add_argument('--artifacts',type=Path,required=True)
    internal.add_argument('--output',type=Path,required=True)
    internal.add_argument('--llm-config',type=Path)
    internal.add_argument('--llm-judge',action='store_true')
    args=parser.parse_args()
    if args.command=='evaluate':
        from skillrecon.evaluation.canonical import run_measurements
        result=run_measurements(args.manifest,args.output)
        print(json.dumps({'output':str(args.output),'requested_skills':result['requested_skills'],
                          'common_successful_skills':result['common_successful_skills']}))
    elif args.command=='replay':
        from skillrecon.detect.replay import validate_bundle
        result=validate_bundle(json.loads(args.bundle.read_text(encoding='utf-8')))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result['passed']:raise SystemExit(1)
    elif args.command=='perturb':
        from skillrecon.evaluation.perturb import build_variants
        print(json.dumps({'variants':len(build_variants(args.plan,args.source_root,args.output_root))}))
    elif args.command=='internal-baselines':
        from skillrecon.evaluation.reproduction import run_internal_baselines
        run_internal_baselines(args.gold,args.artifacts,args.output,args.llm_config,args.llm_judge)


if __name__=='__main__':main()
