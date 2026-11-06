"""Exact named prerequisites and explicitly declared dependency boundaries."""
from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path

import yaml

from skillrecon.contract.parser import parse_document, parse_yaml_frontmatter
from skillrecon.core.types import Clause, Constraint, EvidenceSpan

logger=logging.getLogger(__name__)
_ENV_CONTEXT=re.compile(r"(?i)environment\s+variables?|\benv\s*(?:vars?|variables?)?\b|环境变量")
_EXAMPLE=re.compile(r"(?i)\b(?:examples?|reference|schema|samples?)\b|示例|参考|格式说明")
_ENV_NAME=re.compile(r"`([A-Z][A-Z0-9_]{2,})`")

def registration_environment_fields(metadata):
    """The three observed platform names use the same requires.env schema."""
    if not isinstance(metadata,dict):return []
    result=[]
    for namespace in ['openclaw','clawdbot','clawdis']:
        platform=metadata.get(namespace)
        requirements=platform.get('requires',{}) if isinstance(platform,dict) else {}
        if isinstance(requirements,dict) and isinstance(requirements.get('env'),list) and all(isinstance(n,str) for n in requirements['env']):
            result.append((namespace+'.requires.env',requirements['env']))
    return result

def extract_named_requirements(doc_contents,steps):
    clauses=[]
    for doc_id,content in doc_contents.items():
        for block in parse_document(doc_id,content):
            if block.block_type not in {'paragraph','list_item','table_row'} or _EXAMPLE.search(block.heading_context):continue
            for line in block.content.splitlines():
                table_context=block.block_type=='table_row' and _ENV_CONTEXT.search(' '.join(block.table_headers))
                if not (_ENV_CONTEXT.search(line) or _ENV_CONTEXT.search(block.heading_context) or table_context):continue
                names=list(dict.fromkeys(_ENV_NAME.findall(line)))
                if not names:continue
                start=content.find(line,block.start_offset,block.end_offset)
                if start<0:continue
                evidence=EvidenceSpan(doc_id=doc_id,start_offset=start,end_offset=start+len(line),text=line)
                negative=re.search(r"(?i)\b(?:never|do not|must not|don't)\b|禁止|不得|不要",line)
                prohibited=bool(re.search(r"(?i)(?:never|do not|must not|don't)[*\s]+(?:read|access)\b|(?:禁止|不得|不要)(?:读取|访问)",line))
                # A ban on exporting/storing a key is not a ban on reading it.
                if negative and not prohibited:continue
                for name in names:
                    cid=f'requirement-{doc_id}-{start}-{name}'
                    clauses.append(Clause(clause_id=cid,capability='env_var_read',operator='prohibited' if prohibited else 'allowed',
                        role='policy',constraints=[Constraint(constraint_id=cid+'-env',constraint_type='env_var',value=name,evidence=evidence)],
                        evidence_spans=[evidence],vote_agreement=1.0,source_doc_ids=[doc_id],
                        step_ids=[s.step_id for s in steps if s.doc_id==doc_id and s.evidence
                                  and s.evidence.start_offset<=start<s.evidence.end_offset]))
    return clauses

def extract_document_command_clauses(doc_contents,doc_paths,steps):
    """Preserve the exact scope of directly instructed primitive commands.

    Commands inside functions, conditions and substitutions are implementation
    details; the enclosing instruction does not grant those inner effects.
    """
    from skillrecon.behavior.operational import recover_operational_commands
    events,resources=[],[]
    recover_operational_commands(events,resources,doc_contents,doc_paths,{},Path('.'))
    clauses=[]
    for event in events:
        proof=next(r for r in resources if r.event_id==event.event_id and r.resource_type=='registered_operation')
        fact=json.loads(proof.origin_hint)
        if not fact['direct_command']:continue
        targets=[r for r in resources if r.event_id==event.event_id and r.resolved and r.resource_type in {'url','command','path','env_var'}]
        doc_id=event.unit_id.removeprefix('doc::');content=doc_contents[doc_id];lines=content.splitlines(keepends=True)
        start=sum(len(line) for line in lines[:event.line-1]);text=lines[event.line-1].rstrip('\r\n')
        evidence=EvidenceSpan(doc_id=doc_id,start_offset=start,end_offset=start+len(text),text=text)
        for index,target in enumerate(targets or [None]):
            cid=f'document-command-{event.event_id}-{index}'
            clauses.append(Clause(clause_id=cid,capability=event.capability,operator='allowed' if target else 'unknown',role='policy',
                constraints=[Constraint(constraint_id=cid+'-target',constraint_type=target.resource_type if target else 'scope',
                                        value=target.value if target else 'Unresolved operand of the bound command',evidence=evidence)],
                evidence_spans=[evidence],source_doc_ids=[doc_id],vote_agreement=1.0,
                bound_event_ids=[event.event_id],
                step_ids=[s.step_id for s in steps if s.doc_id==doc_id and s.evidence and s.evidence.start_offset<=start<s.evidence.end_offset]))
    return clauses


def preserve_command_conditions(clauses):
    """A deterministic command match must retain policy on the same source span."""
    result=[]
    for clause in clauses:
        if not clause.clause_id.startswith('document-command-'):
            result.append(clause);continue
        constraints=list(clause.constraints)
        for policy in clauses:
            if (policy.clause_id.startswith('document-command-') or policy.role.value!='policy'
                    or policy.operator.value!='allowed' or policy.capability!=clause.capability):continue
            if not any(a.doc_id==b.doc_id and max(a.start_offset,b.start_offset)<min(a.end_offset,b.end_offset)
                       for a in clause.evidence_spans for b in policy.evidence_spans):continue
            for condition in policy.constraints:
                spans=[condition.evidence] if condition.evidence else policy.evidence_spans
                if not any(a.doc_id==b.doc_id and max(a.start_offset,b.start_offset)<min(a.end_offset,b.end_offset)
                           for a in clause.evidence_spans for b in spans):continue
                # A second spelling of the command is an operation identity,
                # not an independent requirement on every command in a block.
                if condition.constraint_type in {'command','command_glob'} and not any(
                    re.search(r'(?i)\b(?:must|shall|required to|never|do not)\s+(?:run|execute|invoke|use)\b'
                              r'|\bonly\s+(?:(?:run|execute|invoke|use)\s+)?[`\"\']?'+re.escape(condition.value)+r'(?:\b|$)'
                              r'|(?:仅|必须|不得|禁止)(?:运行|执行|调用|使用)',s.text) for s in spans):continue
                if not any(c.constraint_type==condition.constraint_type and c.value==condition.value for c in constraints):
                    constraints.append(condition)
        result.append(clause.model_copy(update={'constraints':constraints}))
    return result


def ground_policy_conditions(clauses,doc_contents):
    """Keep command slots and locally scoped tool rules tied to their evidence."""
    contexts={}
    for doc_id,content in doc_contents.items():
        contexts[doc_id]=parse_document(doc_id,content)
    result=[]
    for clause in clauses:
        if clause.role.value!='policy':
            result.append(clause);continue
        constraints=[]
        lost_command_identity=False
        source_commands={}
        for condition in clause.constraints:
            text=condition.evidence.text.strip() if condition.evidence else ''
            if condition.constraint_type=='command' and text and not clause.clause_id.startswith('document-command-'):
                value=condition.value
                descriptive=bool(re.fullmatch(r'`?'+re.escape(value)+r'`?\s*\([^\n]*\)',text))
                inferred=value.lower() not in text.lower()
                prerequisite=False
                if clause.operator.value=='allowed' and condition.evidence and not re.search(
                    r'(?i)\b(?:run|execute|invoke|use|only|allowed|permitted|never)\b|运行|执行|调用|仅|允许',text):
                    for block in contexts.get(condition.evidence.doc_id,[]):
                        if not block.start_offset<=condition.evidence.start_offset<block.end_offset:continue
                        heading=block.heading_context.rsplit(' > ',1)[-1]
                        table_header=block.content.splitlines()[0] if block.content else ''
                        prerequisite=(block.block_type in {'paragraph','list_item','table_row'} and bool(
                            re.search(r'(?i)\b(?:prerequisites?|requirements?|dependencies)\b|前置|依赖|要求',heading)
                            or block.block_type=='table_row' and re.search(r'(?i)\binstall\b',' '.join(block.table_headers))
                            or table_header.startswith('|') and re.search(r'(?i)\binstall\b',table_header)
                            and re.search(r'(?i)\b(?:tool|package|dependency)\b',table_header)))
                # A literal interpreter argument is not its executable name.
                argument=False
                if re.match(r'^(?:python3?|node|bash|sh|zsh)\s',text):
                    try:parts=shlex.split(text)
                    except ValueError:inferred=True
                    else:
                        argument=value in parts[1:]
                        if parts:source_commands[parts[0]]=condition.evidence
                if descriptive or inferred or argument or prerequisite:
                    condition=condition.model_copy(update={'constraint_type':'scope'})
                    lost_command_identity=True
            constraints.append(condition)
        if lost_command_identity:
            for command,evidence in source_commands.items():
                if not any(c.constraint_type=='command' and c.value==command for c in constraints):
                    constraints.append(Constraint(constraint_id=clause.clause_id+'-source-command-'+command,
                        constraint_type='command',value=command,evidence=evidence))
        if clause.operator.value=='prohibited' and not clause.target:
            for evidence in clause.evidence_spans:
                for block in contexts.get(evidence.doc_id,[]):
                    if not block.start_offset<=evidence.start_offset<block.end_offset:continue
                    # API/tool member descriptions constrain that member, not
                    # package installation or unrelated command examples.
                    context=block.heading_context
                    if not re.search(r'(?i)\btools?\b|工具',context):continue
                    subject=re.search(r'`([A-Za-z_]\w*)`\s*\(',context.rsplit(' > ',1)[-1])
                    if subject:
                        key=f'tool:{subject[1]}'
                        if not any(c.constraint_type=='scope' and c.value==key for c in constraints):
                            constraints.append(Constraint(constraint_id=clause.clause_id+'-tool-scope',
                                constraint_type='scope',value=key,evidence=evidence))
        updates={'constraints':constraints}
        if clause.operator.value=='prohibited':
            from skillrecon.contract.controls import forbidden_standalone_calls
            names=sorted({name for span in clause.evidence_spans for name in forbidden_standalone_calls(span.text)})
            if names:
                # The named functions specify an invocation trigger, rather
                # than a URL. Normalize an already extracted prohibition; this
                # never adds a new prose-derived authorization clause.
                def same_restriction(condition):
                    return condition.constraint_type=='scope' and (
                        bool(re.search(r'(?i)separate|standalone|单独|分别',condition.value)) or
                        bool(re.fullmatch(r'(?is)(?:must |always )?use .+? instead(?: .*)?',condition.value))) and not re.search(
                            r'(?i)\b(?:if|when|unless|except|before|after)\b|如果|除非|之前|之后',condition.value)
                constraints=[condition for condition in constraints if condition.constraint_type!='standalone_call' and not same_restriction(condition)]
                constraints.append(Constraint(constraint_id=clause.clause_id+'-standalone',
                    constraint_type='standalone_call',value='|'.join(names),evidence=clause.evidence_spans[0]))
                updates['constraints']=constraints
                if clause.target and any(name in clause.target for name in names):updates['target']=None
        if (lost_command_identity and clause.operator.value=='allowed' and not clause.target
                and not any(c.constraint_type=='command' for c in constraints)):
            updates['operator']=type(clause.operator).UNKNOWN
        result.append(clause.model_copy(update=updates))
    return result

def declared_environment_dependencies(doc_contents,source_root=None):
    """An omitted field supplies no assertion; an explicit env list is checked."""
    declarations=[]
    main_requirements={};feature_only=set()
    for doc_id,content in doc_contents.items():
        for block in parse_document(doc_id,content):
            if _EXAMPLE.search(block.heading_context):continue
            for line in block.content.splitlines():
                if re.search(r'(?i)\bis\s+not\s+a\s+runtime\s+requirement\b',line.replace('**','')):
                    feature_only.update(_ENV_NAME.findall(line))
            if not re.search(r'(?i)\b(?:setup|installation|prerequisites?)\b|(?:required|mandatory).*environment|environment.*(?:required|mandatory)|安装|必需环境',block.heading_context):continue
            if re.search(r'(?i)\b(?:optional|fallback|emergency)\b|可选|应急',block.heading_context):continue
            for line in block.content.splitlines():
                if re.search(r'(?i)\b(?:optional|fallback|emergency)\b|not\s+(?:a\s+)?(?:runtime\s+)?requirement|可选|应急',line):continue
                names=_ENV_NAME.findall(line)
                if block.block_type=='code_block':
                    names+=re.findall(r'^\s*(?:export\s+)?([A-Z][A-Z0-9_]{2,})\s*=',line)
                for name in names:
                    main_requirements.setdefault(name,EvidenceSpan(doc_id=doc_id,start_offset=block.start_offset,
                        end_offset=block.end_offset,text=block.content).model_dump(mode='json'))
        fm,_=parse_yaml_frontmatter(content)
        section=re.search(r'(?ms)^metadata:.*?(?=^\w[\w-]*[ \t]*:|\Z)',fm)
        if section:
            try:parsed=yaml.safe_load(section.group())
            except yaml.YAMLError:
                logger.warning('Cannot parse explicit metadata dependency in %s',doc_id)
                continue
            metadata=parsed.get('metadata',{}) if isinstance(parsed,dict) else {}
            for field,names in registration_environment_fields(metadata):
                evidence=EvidenceSpan(doc_id=doc_id,start_offset=4+section.start(),end_offset=4+section.end(),text=section.group())
                declarations.append({'key':f'{doc_id}:metadata.{field}','names':names,'evidence':evidence})
        for block in parse_document(doc_id,content):
            if block.block_type not in {'paragraph','list_item'} or _EXAMPLE.search(block.heading_context):continue
            text=block.content
            if re.fullmatch(r'(?i)\s*(?:no\s+(?:api\s+keys?|credentials?|environment\s+variables?)\s+(?:are\s+)?(?:required|needed)|无需(?:任何)?(?:API\s*密钥|凭据|环境变量))[.!。\s]*',text.replace('**','')):
                evidence=EvidenceSpan(doc_id=doc_id,start_offset=block.start_offset,end_offset=block.end_offset,text=text)
                declarations.append({'key':f'{doc_id}:no-required-credentials:{block.start_offset}','names':[],'evidence':evidence})
    if source_root is not None:
        for name in ['skill.json','package.json']:
            path=source_root/name
            if not path.exists():continue
            content=path.read_text(encoding='utf-8')
            try:data=json.loads(content)
            except json.JSONDecodeError:
                logger.warning('Cannot parse registry dependency file %s',path)
                continue
            for field,names in registration_environment_fields(data):
                evidence=EvidenceSpan(doc_id='registry:'+name,start_offset=0,end_offset=len(content),text=content)
                declarations.append({'key':f'registry:{name}:{field}','names':names,'evidence':evidence})
    for name in feature_only:main_requirements.pop(name,None)
    for declaration in declarations:declaration['main_requirements']=main_requirements
    return declarations

def extract_dependency_clauses(declarations):
    return [Clause(clause_id=f'dependency-{d["key"]}',capability='env_var_read',operator='allowed',role='policy',
        constraints=[Constraint(constraint_id=f'dependency-{d["key"]}-condition',constraint_type='dependency_declared',value=d['key'],evidence=d['evidence'])],
        evidence_spans=[d['evidence']],source_doc_ids=[d['evidence'].doc_id],vote_agreement=1.0) for d in declarations]
