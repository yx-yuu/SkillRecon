"""Native documentary roots and explicit configuration commands for Section 4.6."""
from __future__ import annotations

import fnmatch
import json
from pathlib import PurePosixPath

NATIVE_CONFIGS={'.claude/settings.json','opencode.json','.opencode/opencode.json'}


def documentary_roots(files):
    by_path={entry.relative_path:entry for entry in files}
    roots=[entry for entry in files if entry.relative_path.lower()=='skill.md']
    for name in ['CLAUDE.md','AGENTS.md']:
        if name in by_path and not roots:
            roots.append(by_path[name])
    command_patterns=['.claude/commands/*.md','.claude/skills/*/SKILL.md',
                      '.opencode/command/*.md','.opencode/commands/*.md','.opencode/skills/*/SKILL.md']
    if 'opencode.json' in by_path:
        command_patterns += ['command/*.md','commands/*.md','skills/*/SKILL.md']
    roots += [entry for entry in files if any(fnmatch.fnmatchcase(entry.relative_path,pattern) for pattern in command_patterns)
              and entry not in roots]
    roots += [by_path[name] for name in sorted(NATIVE_CONFIGS) if name in by_path]
    return roots


def configuration_hooks(content):
    """Return JSON pointers to actual command hook values, excluding matcher text."""
    payload=json.loads(content)
    hooks=payload.get('hooks',{})
    commands=[]
    def visit(value,pointer):
        if isinstance(value,dict):
            if value.get('type')=='command' and isinstance(value.get('command'),str):
                commands.append((pointer+'/command',value['command']))
            else:
                for key,item in value.items():
                    escaped=key.replace('~','~0').replace('/','~1')
                    if key in {'Bash','Read','Write','Edit'} and isinstance(item,str):
                        commands.append((pointer+'/'+escaped,item))
                    elif isinstance(item,(dict,list)):
                        visit(item,pointer+'/'+escaped)
        elif isinstance(value,list):
            for index,item in enumerate(value):visit(item,pointer+'/'+str(index))
    visit(hooks,'/hooks')
    return commands


def json_pointer(payload,pointer):
    for part in pointer.lstrip('/').split('/'):
        part=part.replace('~1','/').replace('~0','~')
        payload=payload[int(part)] if isinstance(payload,list) else payload[part]
    return payload
