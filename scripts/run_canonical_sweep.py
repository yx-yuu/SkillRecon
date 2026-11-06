#!/usr/bin/env python3
"""Execute or print explicitly selected parameter variants; never select by score."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',required=True)
p.add_argument('--skills',type=Path,required=True,help='One fixed identity per line')
p.add_argument('--llm-config',required=True)
p.add_argument('--grid',type=Path,required=True,help='JSON rows: name, n, theta, k_c, witness_bounds')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--dry-run',action='store_true')
p.add_argument('--native-only',action='store_true')
p.add_argument('--codeql-bin')
args=p.parse_args()
skills=[line.strip() for line in args.skills.read_text().splitlines() if line.strip()]
grid=json.loads(args.grid.read_text())
commands=[]
for row in grid:
    for skill in skills:
        command=[sys.executable,'-m','skillrecon','analyze','--skill',skill,'--data-root',args.data_root,
                 '--llm-config',args.llm_config,'--output-dir',str(args.output/row['name'])]
        for key,flag in [('n','--n-samples'),('theta','--theta'),('k_c','--max-candidates-per-behavior')]:
            if key in row:command.extend([flag,str(row[key])])
        if 'witness_bounds' in row:
            command.extend(['--witness-max-nodes',str(row['witness_bounds'][0]),'--witness-max-edges',str(row['witness_bounds'][1])])
        if args.native_only:command.append('--native-only')
        if args.codeql_bin:command.extend(['--codeql-bin',args.codeql_bin])
        commands.append({'variant':row['name'],'skill_id':skill,'command':command})
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'sweep-plan.json').write_text(json.dumps(commands,indent=2)+'\n')
if args.dry_run:
    print(json.dumps({'variants':len(grid),'skills':len(skills),'commands':len(commands),'executed':False}))
else:
    results=[]
    for entry in commands:
        result=subprocess.run(entry['command'],check=False)
        results.append({**entry,'returncode':result.returncode})
        (args.output/'run-records.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(row['returncode'] for row in results):raise SystemExit(1)
