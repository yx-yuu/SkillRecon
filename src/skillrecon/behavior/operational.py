"""Source-bound Shell operations in executable scripts and documentation."""
from __future__ import annotations

import fnmatch
import json
import re
from collections import Counter
from pathlib import PurePosixPath

from tree_sitter import Language, Parser
import tree_sitter_bash

from skillrecon.behavior.bash import _environment_references
from skillrecon.behavior.resource_extract import _refine_capability
from skillrecon.contract.parser import parse_document, parse_yaml_frontmatter, parse_frontmatter_fields
from skillrecon.loader.inline import check_binding_signal, is_prohibitive_binding
from skillrecon.core.types import CapabilityEvent, ResourceUse

_OPERATION_HEADING = re.compile(r"(?i)\b(?:installation|install|setup|usage|quick\s*start|authentication|configure|commands?|steps?|execute|run|workflow|procedure|instructions?)\b|安装|运行|执行|调用|使用|配置|步骤|认证|操作|流程")
_EXAMPLE = re.compile(r"(?i)\b(?:examples?|reference|schema|samples?|background)\b|示例|参考|格式说明|背景")
_INLINE_RUN = re.compile(r"(?i)(?:\b(?:run|execute|invoke)\s+(?:(?:(?:(?:the\s+)?following|the|this|a)\s+)?(?:commands?|scripts?)\s*:?\s*)?(?:(?:and|or|then)\s*)*|(?:运行|执行|调用)(?:以下|下面|此|该)?(?:命令|脚本)?\s*[:：]?\s*)$")
_SHELL_BUILTINS = {'alias','bg','bind','break','builtin','cd','command','compgen','complete',
    'continue','declare','dirs','disown','echo','enable','exit','export','false','fc','fg','getopts',
    'hash','help','history','jobs','let','local','mapfile','popd','printf','pushd','pwd','read',
    'readarray','readonly','return','set','shift','shopt','test','times','trap','true','type',
    'typeset','ulimit','umask','unalias','unset','wait',':','['}
_USE_COMMAND = re.compile(r"(?i)\buse\b.{0,180}\bcommands?\b|使用.{0,80}(?:命令|指令)")
_CAPABILITIES = {'curl':'http_request','wget':'http_request','rm':'file_delete','chmod':'file_permission_change',
    'chown':'file_permission_change','crontab':'cron_write','ssh':'ssh_connect','scp':'upload_exfil',
    'sftp':'ftp_transfer','at':'cron_write','atrm':'cron_write',
    'systemctl':'shell_exec','eval':'shell_exec','bash':'shell_exec','sh':'shell_exec','zsh':'shell_exec',
    'python':'shell_exec','python3':'shell_exec','node':'shell_exec','npx':'dynamic_import','uvx':'dynamic_import'}

def _walk(node):
    yield node
    for child in node.named_children:yield from _walk(child)

def _literal(node):
    if node.type in {'word','raw_string','string','command_name'} and not any(
        n.type in {'simple_expansion','expansion','command_substitution'} for n in _walk(node)
    ):
        return node.text.decode().strip('"\'')
    return None

def recover_operational_commands(events, resources, doc_contents, doc_paths, unit_paths, staged_root):
    """Recover actual command nodes, retaining document binding and input origin."""
    parser=Parser(Language(tree_sitter_bash.language()));changes=[];invocations=[]
    by_key={};by_id={e.event_id:e for e in events}
    for event in events:
        if event.tier!='instruction':by_key.setdefault((event.unit_id,event.line,event.capability),[]).append(event)
    existing={r.resource_id for r in resources}

    def add(unit,relative,line,column,cap,api,args,binding,block_id='',resource=None,direct=False,reachable=True):
        event_id=f'operation-{unit}-{line}-{column}-{cap}'+('-'+args[0] if cap=='env_var_read' else '')
        event=by_id.get(event_id)
        candidates=by_key.get((unit,line,cap),[])
        if cap=='env_var_read':candidates=[e for e in candidates if args[0] in e.arguments]
        if event is None and len(candidates)==1 and f'operation-proof-{candidates[0].event_id}' not in existing:
            event=candidates[0]
        if event is None and len(candidates)==1:
            proof=next((r for r in resources if r.resource_id==f'operation-proof-{candidates[0].event_id}'),None)
            if proof is not None and json.loads(proof.origin_hint)['column']==column:event=candidates[0]
        if event is None:
            event=CapabilityEvent(event_id=event_id,unit_id=unit,
                capability=cap,api_call=api,arguments=args,tier='registered_source',language='bash',
                file_path=relative,line=line,location=f'{relative}:{line}',detail='Parsed executable Shell operation')
            events.append(event);by_id[event_id]=event
        proof_id=f'operation-proof-{event.event_id}'
        if proof_id in existing:return event
        if api in {'source','.'}:
            revised=event.model_copy(update={'api_call':api,'arguments':args,'detail':'Parsed Shell source file invocation'})
            events[events.index(event)]=revised
            event=revised;by_id[event.event_id]=event
        existing.add(proof_id)
        fact={'model':'operational_shell_v1','binding':binding,'source_block_id':block_id,
              'location':event.location,'column':column,'operation':api,'capability':cap,'direct_command':direct,'reachable':reachable}
        resources.append(ResourceUse(resource_id=proof_id,unit_id=unit,event_id=event.event_id,
            resource_type='registered_operation',value=cap,resolved=True,location=event.location,
            origin_kind='source_binding',origin_hint=json.dumps(fact,ensure_ascii=False)))
        if resource:
            kind,value,*status=resource
            if not any(r.event_id==event.event_id and r.resource_type==kind and r.value==value for r in resources):
                resources.append(ResourceUse(resource_id=f'operation-target-{event.event_id}',unit_id=unit,
                    event_id=event.event_id,resource_type=kind,value=value,resolved=status[0] if status else True,
                    location=event.location,origin_kind='code_literal' if not status or status[0] else 'unknown',
                    origin_hint='Parsed command argument'))
        changes.append(fact)
        return event

    def scan(content,unit,relative,offset,binding,block_id='',allowed=None,column_offset=0):
        tree=parser.parse(content.encode());reads,_,_=_environment_references(content)
        composite_parts={}
        assignments=[n for n in _walk(tree.root_node) if n.type=='variable_assignment']
        assigned=Counter(n.child_by_field_name('name').text.decode() for n in assignments)
        constants={n.child_by_field_name('name').text.decode():_literal(n.child_by_field_name('value'))
                   for n in tree.root_node.named_children if n.type=='variable_assignment'
                   and n.child_by_field_name('value') is not None
                   and assigned[n.child_by_field_name('name').text.decode()]==1}
        for node in _walk(tree.root_node):
            if node.type=='declaration_command':
                for child in node.named_children:
                    if child.type=='variable_name':constants.pop(child.text.decode(),None)
            if node.type=='command' and node.child_by_field_name('name').text in {b'read',b'unset'}:
                for child in node.children_by_field_name('argument'):
                    constants.pop(_literal(child),None)

        def source_path(node):
            literal=_literal(node)
            if literal is not None:return literal
            expression=node.text.decode().strip('"')
            match=re.fullmatch(r'\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([A-Za-z_][A-Za-z0-9_]*)\})',expression)
            if match:return constants.get(match[1] or match[2])
            return None

        function_names={n.child_by_field_name('name').text.decode() for n in _walk(tree.root_node)
                        if n.type=='function_definition' and n.child_by_field_name('name') is not None}
        def owner(node):
            node=node.parent
            while node is not None and node.type!='function_definition':node=node.parent
            return node.child_by_field_name('name').text.decode() if node is not None else '__root__'
        calls={}
        for node in _walk(tree.root_node):
            if node.type=='command' and node.child_by_field_name('name') is not None:
                name=_literal(node.child_by_field_name('name'))
                if name in function_names:calls.setdefault(owner(node),set()).add(name)
        reachable_functions={'__root__'};pending=['__root__']
        while pending:
            for name in calls.get(pending.pop(),set())-reachable_functions:
                reachable_functions.add(name);pending.append(name)
        for pipeline in (n for n in _walk(tree.root_node) if n.type=='pipeline'):
            commands=pipeline.named_children
            if len(commands)!=2 or any(n.type!='command' for n in commands):continue
            download,execute=commands
            source_node=download.child_by_field_name('name');shell_node=execute.child_by_field_name('name')
            if source_node is None or shell_node is None:continue
            source_name=_literal(source_node)
            shell_name=_literal(shell_node)
            if source_name is None or shell_name is None:continue
            if PurePosixPath(source_name).name!='curl' or PurePosixPath(shell_name).name not in {'bash','sh','zsh'}:continue
            if source_name in function_names or shell_name in function_names:continue
            arguments=[_literal(n) for n in download.children_by_field_name('argument')]
            urls=[v for v in arguments if v and v.startswith(('https://','http://'))]
            safe_flags={'--fail','--silent','--show-error','--location','--insecure','--compressed','--no-progress-meter'}
            if len(urls)!=1 or any(v is None or v not in urls and v not in safe_flags and not re.fullmatch(r'-[fsSLk]+',v) for v in arguments):continue
            shell_args=[_literal(n) for n in execute.children_by_field_name('argument')]
            if any(v is None for v in shell_args):continue
            split=shell_args.index('--') if '--' in shell_args else len(shell_args)
            shell_flags=shell_args[:split]
            reads_stdin=(all(v=='-' or re.fullmatch(r'-[euxvs]+',v) for v in shell_flags)
                         and (split==len(shell_args) or split==len(shell_args)-1
                              or any('s' in v for v in shell_flags)))
            if not reads_stdin or owner(execute) not in reachable_functions:continue
            if allowed is not None and not any(fnmatch.fnmatch(pipeline.text.decode(),p) for p in allowed):continue
            parent=pipeline.parent;direct=True
            while parent is not None and parent.type!='program':
                if parent.type in {'function_definition','if_statement','while_statement','for_statement','command_substitution','subshell'}:direct=False
                parent=parent.parent
            composite=add(unit,relative,offset+execute.start_point.row,execute.start_point.column+column_offset,
                'eval_exec','curl | '+PurePosixPath(shell_name).name,[urls[0]],binding,block_id,
                ('url',urls[0]),direct,True)
            composite_parts[download.start_byte]=composite.event_id
            composite_parts[execute.start_byte]=composite.event_id
        for node in _walk(tree.root_node):
            if node.type!='command':continue
            name_node=node.child_by_field_name('name')
            if name_node is None:continue
            literal=_literal(name_node)
            if not literal:continue
            name='.' if literal=='.' else PurePosixPath(literal).name
            parent=node.parent;direct=True
            while parent is not None and parent.type!='program':
                if parent.type in {'function_definition','if_statement','while_statement','for_statement','command_substitution','subshell'}:direct=False
                parent=parent.parent
            cap=_CAPABILITIES.get(name)
            arguments=node.children_by_field_name('argument');values=[_literal(n) for n in arguments]
            if name in {'source','.'} and literal not in function_names and arguments:
                path=source_path(arguments[0])
                if (path and owner(node) in reachable_functions
                        and (binding=='standalone_script' or not tree.root_node.has_error)):
                    if allowed is None or any(fnmatch.fnmatch(node.text.decode(),p) for p in allowed):
                        add(unit,relative,offset+node.start_point.row,node.start_point.column+column_offset,
                            'shell_exec',name,[path],binding,block_id,('path',path),direct,True)
                continue
            if name in {'pip','pip3'} and values and values[0]=='install':cap='dynamic_import'
            if name in {'npm','pnpm','yarn'} and values and values[0] in {'install','ci','add','dlx','exec'}:cap='dynamic_import'
            generic_command=(cap is None and literal not in function_names
                and name not in _SHELL_BUILTINS and not tree.root_node.has_error)
            if generic_command:cap='shell_exec'
            if cap is None:continue
            # Frontmatter binds only the command patterns that it actually lists.
            if allowed is not None and not any(fnmatch.fnmatch(node.text.decode(),p) for p in allowed):continue
            line=offset+node.start_point.row;target=None;args=[]
            if cap=='http_request':
                urls=[v for v in values if v and v.startswith(('https://','http://'))]
                if urls:target=('url',urls[0]);args=[urls[0]]
            elif cap=='dynamic_import':
                packages=[v for v in values if v and not v.startswith('-') and v not in {'install','ci','add','dlx','exec'}]
                if packages:target=('command',name+' '+packages[0]);args=[packages[0]]
            elif cap in {'shell_exec','ssh_connect','cron_write','ftp_transfer'}:
                target=(('command',node.text.decode(),all(value is not None for value in values))
                        if generic_command else ('command',name));args=[name]
                if name=='eval':cap=_refine_capability('shell_exec',node.text.decode(),'eval')
                if name=='eval' and cap!='eval_exec':continue
            elif cap in {'file_delete','file_permission_change'}:
                paths=[v for v in values if v and not v.startswith('-') and not v.isdigit()]
                if paths:target=('path',paths[-1]);args=[paths[-1]]
            reachable=owner(node) in reachable_functions
            event=add(unit,relative,line,node.start_point.column+column_offset,cap,name,args,binding,block_id,target,direct,reachable)
            if node.start_byte in composite_parts:
                proof=next(r for r in resources if r.resource_id==f'operation-proof-{event.event_id}')
                fact=json.loads(proof.origin_hint)
                if fact.get('part_of_event')!=composite_parts[node.start_byte]:
                    fact['part_of_event']=composite_parts[node.start_byte]
                    resources[resources.index(proof)]=proof.model_copy(update={'origin_hint':json.dumps(fact,ensure_ascii=False)})
                    changes.append(fact)
            if name in {'python','python3','node','bash','sh','zsh'}:
                script=next((v for v in values if v and not v.startswith('-')),None)
                if script:
                    normalized=PurePosixPath(script).as_posix()
                    targets=[key for key,path in unit_paths.items() if path==normalized]
                    invocations.extend((unit,t,line,binding) for t in targets)
            for env in reads.get(node.start_point.row+1,[]):
                add(unit,relative,line,node.start_point.column+column_offset,'env_var_read','Shell input expansion',[env],binding,block_id,('env_var',env),direct,reachable)

    for unit,relative in unit_paths.items():
        if PurePosixPath(relative).suffix in {'.sh','.bash'} and not relative.startswith('.skillrecon_synthetic/'):
            scan((staged_root/relative).read_text(encoding='utf-8'),unit,relative,1,'standalone_script')
    for doc_id,content in doc_contents.items():
        fm,_=parse_yaml_frontmatter(content);fields=parse_frontmatter_fields(fm)
        allowed_tools = fields.get('allowed-tools', '')
        allowed=re.findall(r'Bash\(([^)]*)\)', allowed_tools if isinstance(allowed_tools, str) else ' '.join(allowed_tools))
        blocks=parse_document(doc_id,content)
        prohibited_list=False
        for index,block in enumerate(blocks):
            if block.block_type in {'paragraph','heading'}:prohibited_list=is_prohibitive_binding(block.content)
            if block.block_type!='code_block' or block.language_hint not in {None,'','bash','sh','shell','zsh'}:continue
            if prohibited_list:continue
            preceding=blocks[index-1] if index else None
            if preceding and is_prohibitive_binding(preceding.content):continue
            binding=check_binding_signal(preceding,block)
            explicit = binding is not None
            if _EXAMPLE.search(block.heading_context) and not explicit:continue
            if not binding:continue
            # Code block offsets point to the opening fence, not the first command.
            offset=content[:block.start_offset].count('\n')+2
            scan(block.content,f'doc::{doc_id}',doc_paths[doc_id],offset,
                 ('document_command_block' if binding=='document_command_block' else 'document_instruction')
                 if binding else 'frontmatter_allowed_tool',block.block_id,
                 None if binding else allowed)
        prohibited_list=False
        for block in blocks:
            if block.block_type in {'paragraph','heading'}:prohibited_list=is_prohibitive_binding(block.content)
            if block.block_type not in {'paragraph','list_item'} or _EXAMPLE.search(block.heading_context):continue
            if prohibited_list:continue
            for match in re.finditer(r'`([^`\n]+)`',block.content):
                prefix=re.sub(r'`[^`]*`','',block.content[:match.start()].rsplit('\n',1)[-1]).replace('**','').replace('__','')
                if not _INLINE_RUN.search(prefix):continue
                if re.search(r"(?i)\b(?:never|do not|must not|don't)\b|禁止|不得|不要|切勿",prefix):continue
                start=content.find(match[1],block.start_offset,block.end_offset)
                if start<0:continue
                scan(match[1],f'doc::{doc_id}',doc_paths[doc_id],content[:start].count('\n')+1,
                     'document_instruction',block.block_id,
                     column_offset=len(content[content.rfind('\n',0,start)+1:start].encode()))
    routes={}
    pending=[x for x in invocations if x[0].startswith('doc::')]
    while pending:
        caller,target,line,binding=pending.pop()
        if target in routes:continue
        routes[target]={'caller_unit':caller,'call_line':line,'binding':binding,'target_path':unit_paths[target]}
        pending.extend(x for x in invocations if x[0]==target)
    for event in events:
        if event.unit_id not in routes:continue
        rid=f'operation-route-{event.event_id}'
        if rid in existing:continue
        existing.add(rid)
        proof={'model':'bound_command_route_v1',**routes[event.unit_id]}
        resources.append(ResourceUse(resource_id=rid,unit_id=event.unit_id,event_id=event.event_id,
            resource_type='registered_route',value=unit_paths[event.unit_id],resolved=True,location=event.location,
            origin_kind='source_binding',origin_hint=json.dumps(proof,ensure_ascii=False)))
        changes.append(proof)
    return changes
