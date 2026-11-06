"""Four controlled documentation perturbations, applied only from located plans."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

KINDS={'missing_authorization','stale_resource','ambiguous_scope','conflicting_allow_prohibit'}


def build_variants(plan_path:Path,source_root:Path,output_root:Path):
    """Apply located edits; emit scoring inputs only for supplied reviewed gold."""
    rows=[json.loads(line) for line in plan_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    manifest=[]
    for row in rows:
        if row['kind'] not in KINDS:raise ValueError(f"Unknown perturbation: {row['kind']}")
        source=source_root/row['skill_id']
        target=output_root/row['kind']/row['skill_id']
        if target.exists():raise FileExistsError(f'Variant output already exists: {target}')
        text=(source/row['file']).read_text(encoding='utf-8')
        before=row['original'];after=row['replacement']
        start=row.get('start_byte')
        raw=text.encode('utf-8');needle=before.encode('utf-8')
        if start is None:
            if raw.count(needle)!=1:raise ValueError('Ambiguous edit: provide an exact start_byte')
            start=raw.index(needle)
        if raw[start:start+len(needle)]!=needle:raise ValueError('Located source text does not match the plan')
        shutil.copytree(source,target)
        (target/row['file']).write_bytes(raw[:start]+after.encode('utf-8')+raw[start+len(needle):])
        manifest.append({'skill_id':row['skill_id'],'kind':row['kind'],'file':row['file'],
                         'start_byte':start,'original':before,'replacement':after,
                         'gold':row.get('gold'),'provenance':'Explicit controlled edit; labels only when supplied in the plan'})
    output_root.mkdir(parents=True,exist_ok=True)
    (output_root/'perturbation-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for kind in KINDS:
        selected=[row for row in manifest if row['kind']==kind and row['gold'] is not None]
        if selected:
            (output_root/kind/'gold.jsonl').write_text(''.join(json.dumps({'skill_id':row['skill_id'],'gold':row['gold'],
                'risk_stratum':'medium','metadata':{'perturbation':kind}},ensure_ascii=False)+'\n' for row in selected),encoding='utf-8')
    return manifest
