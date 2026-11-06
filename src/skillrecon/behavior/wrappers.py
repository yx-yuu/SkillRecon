"""Tier-2 API-graph summaries with proven resource argument slots.

A local wrapper is promoted at its invocation only when a formal parameter
reaches a registered API operand, or a registered source reaches its return.
Imports and aliases resolve to package-local functions; code is never executed.
"""
from __future__ import annotations

import ast
import json
import posixpath
from pathlib import Path

from skillrecon.behavior.normalize_types import RawObservation
from skillrecon.behavior.python_calls import _constant, _qualified, _scope_nodes
from skillrecon.behavior.python import python_modules, local_python_function

_SOURCES = {'file_read', 'token_file_read', 'env_var_read', 'keychain_access'}


def recover_python_wrapper_observations(unit_paths, staged_root):
    from skillrecon.behavior.native import _PYTHON_APIS

    modules = python_modules(unit_paths, staged_root)

    def local_function(node, relative):
        return local_python_function(node, relative, modules)

    def parameters(function):
        return [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]

    def operand(call, slot, parameter_names):
        if slot < len(call.args):
            return call.args[slot]
        names = {parameter_names[slot]} if slot < len(parameter_names) else set()
        return next((kw.value for kw in call.keywords if kw.arg in names), None)

    def slot_binding(expression, function, values):
        params = {p.arg: index for index, p in enumerate(parameters(function))}
        aliases = {name: {('parameter', index)} for name, index in params.items()}
        # Multiple definitions are joined; an unrelated constant cannot be
        # treated as the previously forwarded parameter on another CFG branch.
        assignments = [node for node in _scope_nodes(function) if isinstance(node, ast.Assign)]
        changed = True
        while changed:
            changed = False
            for assignment in assignments:
                if isinstance(assignment.value, ast.Name):
                    binding = aliases.get(assignment.value.id, {('unknown', None)})
                else:
                    value = _constant(assignment.value, values)
                    binding = {('literal', value)} if isinstance(value, str) else {('unknown', None)}
                for target in assignment.targets:
                    if isinstance(target, ast.Name):
                        old = aliases.get(target.id, set())
                        merged = old | binding
                        if merged != old:
                            aliases[target.id] = merged
                            changed = True
        if isinstance(expression, ast.Name):
            choices = aliases.get(expression.id, set())
            return next(iter(choices)) if len(choices) == 1 else ('unknown', None)
        value = _constant(expression, values)
        return ('literal', value) if isinstance(value, str) else ('unknown', None)

    def reaches_return(call, function):
        nodes = _scope_nodes(function)
        names = {target.id for node in nodes if isinstance(node, ast.Assign) and node.value is call
                 for target in node.targets if isinstance(target, ast.Name)}
        changed = True
        while changed:
            old = set(names)
            for node in nodes:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in names:
                    names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            changed = names != old
        # A later replacement would invalidate the proposed source-to-return link.
        for node in nodes:
            if isinstance(node, ast.Assign) and node.value is not call:
                if not (isinstance(node.value, ast.Name) and node.value.id in names):
                    names.difference_update(t.id for t in node.targets if isinstance(t, ast.Name))
        return any(isinstance(node, ast.Return) and (
            node.value is call or isinstance(node.value, ast.Name) and node.value.id in names)
            for node in nodes)

    summaries = {key: set() for relative, module in modules.items()
                 for key in ((relative, name) for name in module['functions'])}
    changed = True
    while changed:
        changed = False
        for (relative, name), known in summaries.items():
            module = modules[relative]
            function = module['functions'][name]
            aliases = {key: value for key, value in module['aliases'].items()
                       if key not in {p.arg for p in parameters(function)}}
            for call in [node for node in _scope_nodes(function) if isinstance(node, ast.Call)]:
                api = _qualified(call.func, aliases)
                registered = _PYTHON_APIS.get(api)
                models = []
                if registered:
                    capability, resource_type, api_slot = registered
                    api_names = (['method', 'url'] if api == 'requests.request' else
                                 ['key'] if capability == 'env_var_read' else
                                 ['args'] if capability in {'shell_exec', 'subprocess'} else ['url', 'key'])
                    argument = operand(call, api_slot, api_names)
                    kind, value = slot_binding(argument, function, module['values'])
                    models.append((api, capability, resource_type, kind, value, relative, call.lineno))
                target = local_function(call.func, relative)
                if target in summaries:
                    child = modules[target[0]]['functions'][target[1]]
                    for api, cap, resource, kind, value, sink_file, sink_line in summaries[target]:
                        if kind == 'parameter':
                            argument = operand(call, value, [p.arg for p in parameters(child)])
                            kind, value = slot_binding(argument, function, module['values'])
                        models.append((api, cap, resource, kind, value, sink_file, sink_line))
                for model in models:
                    api, cap, resource, kind, value, sink_file, sink_line = model
                    if kind == 'unknown' or cap in _SOURCES and not reaches_return(call, function):
                        continue
                    if model not in known:
                        known.add(model)
                        changed = True

    observations = []
    for relative, module in modules.items():
        scopes = [module['tree'], *module['functions'].values()]
        for scope in scopes:
            values = dict(module['values'])
            params = {p.arg for p in parameters(scope)} if scope is not module['tree'] else set()
            for node in _scope_nodes(scope):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            values[target.id] = _constant(node.value, values)
                if not isinstance(node, ast.Call) or isinstance(node.func, ast.Name) and node.func.id in params:
                    continue
                target = local_function(node.func, relative)
                if target not in summaries:
                    continue
                function = modules[target[0]]['functions'][target[1]]
                for api, cap, resource, kind, slot, sink_file, sink_line in sorted(summaries[target]):
                    value = slot if kind == 'literal' else _constant(
                        operand(node, slot, [p.arg for p in parameters(function)]), values)
                    resolved = isinstance(value, str)
                    evidence = {'model': 'python_api_graph_wrapper', 'definition': f'{target[0]}:{function.lineno}',
                                'api_operand': f'{sink_file}:{sink_line}', 'api': api,
                                'formal_slot': slot if kind == 'parameter' else None,
                                'call': f'{relative}:{node.lineno}'}
                    fields = {'capType': cap, 'tier': 'api_graph', 'detail': f'{ast.unparse(node.func)} -> {api}',
                              'resourceType': resource, 'resourceValue': value if resolved else f'<dynamic-{resource}>',
                              'resourceResolved': str(resolved).lower(), 'resourceOriginKind': 'api_wrapper',
                              'resourceOriginHint': json.dumps(evidence)}
                    observations.append(RawObservation(language='python',relative_path=relative,line=node.lineno,
                        message='Proven API-graph parameter/source wrapper', fields=fields,unit_id=module['unit']))
    return observations


def recover_javascript_wrapper_observations(unit_paths, staged_root):
    from skillrecon.behavior.native import javascript_api_model
    from skillrecon.behavior.bridge_calls import _qualified as js_qualified
    from skillrecon.behavior.resource_fields import _field, _text, _js_literal
    from skillrecon.behavior.javascript import (
        javascript_modules, parameter_names, scope_nodes, local_function as resolve_function,
    )

    modules = javascript_modules(unit_paths, staged_root)

    def local_function(node, relative):
        return resolve_function(node, relative, modules)

    def slot_binding(node,function,module):
        params=parameter_names(function)
        aliases={name:('parameter',index) for index,name in enumerate(params) if name is not None}
        values=dict(module['constants'])
        definitions={}
        for statement in scope_nodes(_field(function,'body')):
            if statement.type=='variable_declarator':
                name,value=_field(statement,'name'),_field(statement,'value')
                if name is None or value is None:continue
                definitions.setdefault(_text(name),[]).append(value)
            elif statement.type in {'assignment_expression','augmented_assignment_expression','update_expression'}:
                name=_field(statement,'left') or _field(statement,'argument')
                if name is not None:aliases.pop(_text(name),None);definitions[_text(name)]=[]
        changed=True
        while changed:
            changed=False
            for name,options in definitions.items():
                if len(options)!=1:continue
                value=options[0]
                binding=aliases.get(_text(value)) if value.type=='identifier' else None
                literal=_js_literal(value,values)
                if binding is None and literal is not None:binding=('literal',literal)
                if binding is not None and name not in aliases:aliases[name]=binding;changed=True
        if node is None:return 'unknown',None
        if node.type=='identifier':return aliases.get(_text(node),('unknown',None))
        value=_js_literal(node,values)
        return ('literal',value) if value is not None else ('unknown',None)

    def reaches_return(node,function):
        body=_field(function,'body')
        if body==node:return True
        names={_text(_field(n,'name')) for n in scope_nodes(body) if n.type=='variable_declarator'
               and _field(n,'value')==node}
        for statement in scope_nodes(body):
            if statement.type in {'assignment_expression','augmented_assignment_expression','update_expression'}:
                target=_field(statement,'left') or _field(statement,'argument')
                if target is not None:names.discard(_text(target))
        def carries_source(expression):
            if expression==node or expression.type=='identifier' and _text(expression) in names:return True
            if expression.type in {'await_expression','parenthesized_expression'}:
                return any(carries_source(child) for child in expression.named_children)
            if expression.type=='call_expression' and _text(_field(expression,'function')) in {'JSON.parse','JSON.stringify','String','Buffer.from'}:
                return any(carries_source(child) for child in _field(expression,'arguments').named_children)
            return False
        return carries_source(body) or any(n.type=='return_statement' and any(carries_source(child)
                   for child in n.named_children) for n in scope_nodes(body))

    summaries={(relative,name):set() for relative,module in modules.items() for name in module['functions']}
    changed=True
    while changed:
        changed=False
        for (relative,name),known in summaries.items():
            module=modules[relative];function=module['functions'][name]
            bindings={key:value for key,value in module['bindings'].items() if key not in parameter_names(function)}
            for call in scope_nodes(_field(function,'body')):
                if call.type!='call_expression':continue
                callee=_field(call,'function');arguments=_field(call,'arguments').named_children
                api=js_qualified(callee,bindings);registered=javascript_api_model(api);models=[]
                if registered:
                    kind,value=slot_binding(arguments[0] if arguments else None,function,module)
                    models.append((api,*registered,kind,value,relative,call.start_point.row+1))
                target=None if callee.type=='identifier' and _text(callee) in parameter_names(function) else local_function(callee,relative)
                if target in summaries:
                    for api,cap,resource,kind,value,sink_file,sink_line in summaries[target]:
                        if kind=='parameter':kind,value=slot_binding(arguments[value] if value<len(arguments) else None,function,module)
                        models.append((api,cap,resource,kind,value,sink_file,sink_line))
                for model in models:
                    api,cap,resource,kind,value,sink_file,sink_line=model
                    if kind=='unknown' or cap in _SOURCES and not reaches_return(call,function):continue
                    if model not in known:known.add(model);changed=True

    observations=[]
    for relative,module in modules.items():
        scopes=[(module['root'],set()),*((_field(function,'body'),set(parameter_names(function))) for function in module['functions'].values())]
        for scope,shadowed in scopes:
            for call in scope_nodes(scope):
                if call.type!='call_expression':continue
                callee=_field(call,'function')
                target=None if callee.type=='identifier' and _text(callee) in shadowed else local_function(callee,relative)
                if target not in summaries:continue
                arguments=_field(call,'arguments').named_children
                for api,cap,resource,kind,slot,sink_file,sink_line in sorted(summaries[target]):
                    value=slot if kind=='literal' else _js_literal(arguments[slot],module['constants']) if slot<len(arguments) else None
                    evidence={'model':'javascript_api_graph_wrapper','definition':f'{target[0]}:{modules[target[0]]["functions"][target[1]].start_point.row+1}',
                              'api_operand':f'{sink_file}:{sink_line}','api':api,'formal_slot':slot if kind=='parameter' else None}
                    fields={'capType':cap,'tier':'api_graph','detail':f'{_text(callee)} -> {api}',
                            'resourceType':resource,'resourceValue':str(value) if value is not None else f'<dynamic-{resource}>',
                            'resourceResolved':str(value is not None).lower(),'resourceOriginKind':'api_wrapper',
                            'resourceOriginHint':json.dumps(evidence)}
                    observations.append(RawObservation(language='typescript' if Path(relative).suffix in {'.ts','.tsx'} else 'javascript',
                        relative_path=relative,line=call.start_point.row+1,message='Proven API-graph wrapper argument slot',
                        fields=fields,unit_id=module['unit']))
    return observations
