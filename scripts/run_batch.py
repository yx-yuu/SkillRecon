#!/usr/bin/env python3
"""Run SkillRecon over a selected list of skill identities."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skills-file',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--llm-config',type=Path,required=True)
    parser.add_argument('--limit',type=int)
    args,method_args=parser.parse_known_args()
    skills=[line.strip() for line in args.skills_file.read_text().splitlines() if line.strip()]
    if args.limit is not None:skills=skills[:args.limit]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    completed=[]
    for skill in skills:
        command=[sys.executable,'-m','skillrecon','analyze','--skill',skill,
                 '--data-root',str(args.data_root),'--output-dir',str(args.output_dir),
                 '--llm-config',str(args.llm_config),*method_args]
        subprocess.run(command,check=True)
        completed.append(skill)
        (args.output_dir/'batch.json').write_text(json.dumps({'requested':skills,'completed':completed},indent=2)+'\n')


if __name__=='__main__':main()
