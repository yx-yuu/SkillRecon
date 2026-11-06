"""Source-proved environment prerequisites bound to explicit declarations."""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from skillrecon.behavior.python_calls import _imports,_qualified,_scope_nodes
from skillrecon.core.sensitivity import event_requires_authorization
from skillrecon.core.types import CapabilityEvent,ResourceUse

def _python_context(node, parents):
    """Describe where a prerequisite is enforced, rather than where it is read."""
    while node in parents:
        node=parents[node]
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            return 'function:'+node.name
        if isinstance(node,(ast.If,ast.For,ast.While,ast.Try,ast.With,ast.Match)):
            return 'conditional:'+str(node.lineno)
    return 'package'

def _script_context(node):
    while node.parent is not None:
        node=node.parent
        if node.type in {'function_definition','function_declaration','function_expression','arrow_function','method_definition','generator_function_declaration'}:
            name=node.child_by_field_name('name')
            return 'function:'+(name.text.decode() if name is not None else str(node.start_point.row+1))
        if node.type in {'if_statement','for_statement','while_statement','switch_statement','case_statement','try_statement','catch_clause','command_substitution','subshell'}:
            return 'conditional:'+str(node.start_point.row+1)
    return 'package'

def _python_lookup(node,imports,bindings):
    if isinstance(node,ast.Name):return bindings.get(node.id)
    if isinstance(node,ast.BoolOp) and isinstance(node.op,ast.Or) and all(isinstance(x,ast.Constant) and not x.value for x in node.values[1:]):
        return _python_lookup(node.values[0],imports,bindings)
    if not isinstance(node,ast.Call) or _qualified(node.func,imports) not in {'os.getenv','os.environ.get'} or not node.args:return None
    name=node.args[0];defaults=node.args[1:]+[k.value for k in node.keywords if k.arg=='default']
    if not isinstance(name,ast.Constant) or not isinstance(name.value,str):return None
    if any(not isinstance(d,ast.Constant) or d.value for d in defaults):return None
    return name.value,node.lineno

def _python_inputs(tree):
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    scopes=[tree]+[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    globals={};required=[]
    for scope in scopes:
        imports=_imports(tree).copy();imports.update(_imports(scope))
        assignments={} if scope is tree else globals.copy()
        if isinstance(scope,(ast.FunctionDef,ast.AsyncFunctionDef)):
            for arg in scope.args.posonlyargs+scope.args.args+scope.args.kwonlyargs:
                imports.pop(arg.arg,None);assignments.pop(arg.arg,None)
        def missing(test):
            if isinstance(test,ast.UnaryOp) and isinstance(test.op,ast.Not):
                value=_python_lookup(test.operand,imports,assignments)
                return [value] if value else []
            if isinstance(test,ast.BoolOp) and isinstance(test.op,ast.Or):return [v for item in test.values for v in missing(item)]
            return []
        for node in _scope_nodes(scope):
            if isinstance(node,ast.Assign):
                value=_python_lookup(node.value,imports,assignments)
                for target in node.targets:
                    if isinstance(target,ast.Name):imports.pop(target.id,None);assignments[target.id]=value
            if isinstance(node,ast.Subscript) and _qualified(node.value,imports)=='os.environ' and isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
                ancestor=parents.get(node);caught=False;guarded=False;child=node
                while ancestor is not None and ancestor is not scope:
                    if isinstance(ancestor,ast.Try) and ancestor.handlers:caught=True
                    if isinstance(ancestor,ast.If) and child in ancestor.body:
                        test=ancestor.test
                        guarded=guarded or (isinstance(test,ast.Call) and _qualified(test.func,imports) in {'os.getenv','os.environ.get'}
                            and bool(test.args) and isinstance(test.args[0],ast.Constant) and test.args[0].value==node.slice.value)
                    child=ancestor;ancestor=parents.get(ancestor)
                if not caught and not guarded:required.append((node.slice.value,node.lineno,node.lineno,_python_context(node,parents)))
            if isinstance(node,ast.If):
                failure=any(isinstance(s,ast.Raise) or isinstance(s,ast.Expr) and isinstance(s.value,ast.Call)
                    and _qualified(s.value.func,imports)=='sys.exit' for s in node.body)
                if failure:
                    required.extend((name,line,node.lineno,_python_context(node,parents)) for name,line in missing(node.test))
        if scope is tree:globals=assignments.copy()
    return list(dict.fromkeys(required))

def _javascript_inputs(source,suffix):
    from tree_sitter import Language,Parser
    import tree_sitter_javascript
    import tree_sitter_typescript
    from skillrecon.behavior.resource_fields import _walk,_text,_field
    language=(tree_sitter_typescript.language_typescript() if suffix=='.ts' else tree_sitter_javascript.language())
    tree=Parser(Language(language)).parse(source.encode());required=[]
    def value(node,bindings):
        if node is None:return None
        if node.type=='parenthesized_expression':return value(node.named_children[0],bindings)
        if node.type=='identifier':return bindings.get(_text(node))
        if node.type=='binary_expression' and _text(_field(node,'operator'))=='||' and _text(_field(node,'right')) in {"''",'""','null','undefined'}:
            return value(_field(node,'left'),bindings)
        if node.type not in {'member_expression','subscript_expression'} or 'process' in bindings:return None
        match=re.fullmatch(r'''process\s*\.\s*env(?:\s*\.\s*([A-Za-z_]\w*)|\s*\[\s*['"]([A-Za-z_]\w*)['"]\s*\])''',_text(node))
        return (next(x for x in match.groups() if x),node.start_point.row+1) if match else None
    def missing(node,bindings):
        if node.type=='parenthesized_expression':return missing(node.named_children[0],bindings)
        if node.type=='unary_expression' and node.children[0].text==b'!':
            found=value(_field(node,'argument'),bindings);return [found] if found else []
        if node.type=='binary_expression' and _text(_field(node,'operator'))=='||':
            return missing(_field(node,'left'),bindings)+missing(_field(node,'right'),bindings)
        return []
    def fails(node,bindings):
        statements=node.named_children if node.type=='statement_block' else [node]
        for statement in statements:
            if statement.type=='throw_statement':return True
            if statement.type=='expression_statement' and statement.named_children and 'process' not in bindings:
                call=statement.named_children[0]
                if call.type=='call_expression' and _text(_field(call,'function'))=='process.exit':return True
        return False
    functions={'function_declaration','function_expression','arrow_function','method_definition','generator_function_declaration'}
    def walk(node,bindings):
        if node.type in functions:
            bindings=bindings.copy();params=_field(node,'parameters')
            if params is not None:
                for child in _walk(params):
                    if child.type=='identifier':bindings[_text(child)]=None
        if node.type=='statement_block':
            bindings=bindings.copy()
            for child in node.named_children:
                if child.type in {'lexical_declaration','variable_declaration'}:
                    for declaration in child.named_children:
                        name=_field(declaration,'name')
                        if name is not None and name.type=='identifier':bindings[_text(name)]=None
        if node.type=='variable_declarator':
            name=_field(node,'name')
            if name.type=='identifier':bindings[_text(name)]=value(_field(node,'value'),bindings)
            return
        if node.type=='assignment_expression':
            name=_field(node,'left')
            if name.type=='identifier':bindings[_text(name)]=value(_field(node,'right'),bindings)
            return
        if node.type=='if_statement':
            consequence=_field(node,'consequence')
            if fails(consequence,bindings):
                required.extend((name,line,node.start_point.row+1,_script_context(node))
                                for name,line in missing(_field(node,'condition'),bindings))
            for child in node.named_children:walk(child,bindings.copy())
            for child in _walk(node):
                if child.type=='assignment_expression' and _field(child,'left').type=='identifier':bindings[_text(_field(child,'left'))]=None
            return
        for child in node.named_children:walk(child,bindings)
    walk(tree.root_node,{})
    return list(dict.fromkeys(required))

def _bash_inputs(source):
    from tree_sitter import Language,Parser
    import tree_sitter_bash
    from skillrecon.behavior.bash import _environment_references
    from skillrecon.behavior.resource_fields import _walk,_text,_field
    tree=Parser(Language(tree_sitter_bash.language())).parse(source.encode());reads,_,_=_environment_references(source);required=[]
    for node in _walk(tree.root_node):
        line=node.start_point.row+1
        if node.type=='expansion':
            match=re.match(r'\$\{([A-Z_][A-Z0-9_]*):?\?',_text(node))
            if match and match[1] in reads.get(line,[]):required.append((match[1],line,line,_script_context(node)))
        if node.type!='if_statement':continue
        condition=_field(node,'condition')
        if condition.type!='test_command':continue
        expression=next((n for n in condition.named_children if n.type=='unary_expression'),None)
        if expression is None or _text(_field(expression,'operator'))!='-z':continue
        operand=expression.named_children[-1];text=_text(operand).strip('"')
        match=re.fullmatch(r'\$(?:([A-Z_][A-Z0-9_]*)|\{([A-Z_][A-Z0-9_]*)(?::-)?\})',text)
        if not match:continue
        name=next(x for x in match.groups() if x)
        if name not in reads.get(condition.start_point.row+1,[]):continue
        for statement in node.named_children:
            if statement.type in {'elif_clause','else_clause'}:break
            if statement.type=='command' and _text(_field(statement,'name'))=='exit':
                required.append((name,condition.start_point.row+1,node.start_point.row+1,_script_context(node)))
                break
    return list(dict.fromkeys(required))

def recover_required_dependencies(events,resources,unit_paths,staged_root,declarations):
    facts=[]
    existing={r.resource_id for r in resources}
    for unit,relative in unit_paths.items():
        suffix=Path(relative).suffix
        if suffix not in {'.py','.js','.mjs','.ts','.sh','.bash'} or relative.startswith('.skillrecon_synthetic/'):continue
        source=(staged_root/relative).read_text(encoding='utf-8')
        if suffix=='.py':
            try:tree=ast.parse(source)
            except SyntaxError:continue
            required=_python_inputs(tree);language='python'
        elif suffix in {'.js','.mjs','.ts'}:required=_javascript_inputs(source,suffix);language='javascript' if suffix!='.ts' else 'typescript'
        else:required=_bash_inputs(source);language='bash'
        for name,line,requirement_line,scope in required:
                event=next((e for e in events if e.unit_id==unit and e.line==line and e.capability=='env_var_read' and name in e.arguments),None)
                if event is None:
                    event=CapabilityEvent(event_id=f'dependency-{unit}-{line}-{name}',unit_id=unit,capability='env_var_read',
                        api_call='Required environment input',arguments=[name],file_path=relative,line=line,location=f'{relative}:{line}',
                        tier='registered_source',language=language,detail='Required environment lookup')
                    events.append(event)
                if not event_requires_authorization(event,{'env_var_read'}):continue
                if not any(r.event_id==event.event_id and r.resource_type=='env_var' and r.value==name for r in resources):
                    resources.append(ResourceUse(resource_id=f'dependency-env-{event.event_id}',unit_id=unit,event_id=event.event_id,
                        resource_type='env_var',value=name,resolved=True,location=event.location,origin_kind='code_literal'))
                proof_id=f'dependency-required-{event.event_id}-{requirement_line}'
                if proof_id not in existing:
                    existing.add(proof_id)
                    proof={'model':'required_environment_v2','input':name,'location':event.location,
                           'requirement_location':f'{relative}:{requirement_line}','scope':scope}
                    resources.append(ResourceUse(resource_id=proof_id,unit_id=unit,event_id=event.event_id,
                        resource_type='required_environment',value=name,resolved=True,location=event.location,
                        origin_kind='source_binding',origin_hint=json.dumps(proof)))
                    facts.append(proof)
                # A function/conditional prerequisite is not a package startup
                # prerequisite. Keep its source fact without applying a global
                # requires.env assertion to a feature that was not selected.
                for declaration in declarations:
                    requirement=declaration.get('main_requirements',{}).get(name)
                    if scope!='package' and requirement is None:continue
                    rid=f'dependency-check-{event.event_id}-{declaration["key"]}'
                    if rid in existing:continue
                    existing.add(rid)
                    proof={'model':'required_dependency_v2','kind':'dependency_declared','key':declaration['key'],
                        'result':'true' if name in declaration['names'] else 'false','input':name,
                        'locations':[event.location,f'{relative}:{requirement_line}'],'scope':scope,
                        'applicability':'package_startup' if scope=='package' else 'documented_prerequisite',
                        'prerequisite_evidence':requirement,
                        'declaration_doc_id':declaration['evidence'].doc_id}
                    resources.append(ResourceUse(resource_id=rid,unit_id=unit,event_id=event.event_id,resource_type='control_condition',
                        value='dependency_declared:'+declaration['key'],resolved=True,location=event.location,
                        origin_kind='static_condition',origin_hint=json.dumps(proof)))
                    facts.append(proof)
    return facts
