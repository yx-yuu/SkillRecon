"""Tier-2 Python def-use flow through assignments, wrappers and resolved bridges."""
from __future__ import annotations

import ast
from collections import defaultdict

from skillrecon.behavior.lineage import PayloadOrigin, invocation_payload, payload_path, source_origin
from skillrecon.behavior.python import local_python_function, python_modules
from skillrecon.behavior.python_calls import _imports, _qualified
from skillrecon.behavior.native import _PYTHON_APIS
from skillrecon.core.enums import BridgeKind

_SOURCES={'file_read','token_file_read','env_var_read','keychain_access','credential_store_access'}
_SINKS={'http_request','upload_exfil','telemetry_post','file_write','shell_exec','subprocess','eval_exec','remote_sync'}
_VALUE_BUILTINS={'str','bytes','bytearray','int','float','bool','len','dict','list','tuple','set'}
_VALUE_APIS={'json.load','json.loads','json.dumps','base64.b64encode','base64.b64decode',
             'urllib.parse.urlencode','urllib.parse.quote','urllib.parse.unquote',
             'urllib.request.Request','os.path.join','os.path.dirname','os.path.basename',
             'os.path.abspath','os.path.normpath','os.path.expanduser'}
_VALUE_METHODS={'read','readline','readlines','strip','lstrip','rstrip','encode','decode',
                'lower','upper','replace','join','format','split','splitlines','get','copy'}


def recover_ast_flows(events, resources, unit_paths, staged_root, bridges,
                      *, invocations=None, entry=None):
    """Emit paths only for an AST operand's proven lineage, never source proximity."""
    by_location=defaultdict(list)
    by_id={event.event_id:event for event in events}
    by_bridge={bridge.bridge_id:bridge for bridge in bridges}
    resources_by_event=defaultdict(list)
    for resource in resources:
        if resource.event_id:
            resources_by_event[resource.event_id].append(resource)
    for event in events:
        if event.tier!='instruction':
            by_location[(event.unit_id,event.line)].append(event)
    modules=python_modules(unit_paths, staged_root)
    trees={module['unit']:module['tree'] for module in modules.values()}
    module_functions={unit:{node.name:node for node in ast.walk(tree)
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))} for unit,tree in trees.items()}
    import_bridges={(bridge.source_unit_id,bridge.target_unit_id):bridge for bridge in bridges
                    if bridge.kind==BridgeKind.STATIC}
    paths={}
    def entry_bindings(payload):
        return {**{f'sys.argv[{index}]':set(origins) for index,origins in enumerate(payload.arguments)},
                **{'env:'+name:set(origins) for name,origins in payload.environment.items()}}

    pending=([(by_bridge[entry.bridge_id].target_unit_id,entry_bindings(entry))] if entry is not None
             else [(unit,{}) for unit in trees])
    seen=set()

    while pending:
        unit,scope_entry=pending.pop(0)
        key=(unit,frozenset((name,frozenset(value)) for name,value in scope_entry.items()))
        if key in seen:continue
        seen.add(key)
        tree=trees.get(unit)
        if tree is None:continue
        functions=module_functions[unit]
        aliases=modules[unit_paths[unit]]['aliases']
        global_states={}
        active=set()
        call_slots={}

        def function_aliases(target,function):
            bindings={**modules[unit_paths[target]]['aliases'],**_imports(function)}
            parameters=[*function.args.posonlyargs,*function.args.args,*function.args.kwonlyargs]
            if function.args.vararg is not None:parameters.append(function.args.vararg)
            if function.args.kwarg is not None:parameters.append(function.args.kwarg)
            for parameter in parameters:bindings.pop(parameter.arg,None)
            return bindings

        def cross(origins,bridge,target,location,expression):
            return {origin if origin.trail[-1][0]==target else PayloadOrigin(origin.event_id,
                (*origin.trail,(target,location,'AST carrier',None,'function_binding',expression)),
                (*origin.bridges,bridge.bridge_id)) for origin in origins}

        def dictionary(node,state):
            """Field-specific lineages for explicitly passed environment maps."""
            if isinstance(node,ast.Dict):
                result={}
                for name,value in zip(node.keys,node.values):
                    if name is None:result.update(dictionary(value,state))
                    elif isinstance(name,ast.Constant) and isinstance(name.value,str):
                        result[name.value]=expression(value,state)
                return result
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='copy':
                return dictionary(node.func.value,state)
            prefix='env:' if ast.unparse(node)=='os.environ' else 'field:'+ast.unparse(node)+':'
            return {key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}

        def emit_path(source,event,node):
            path_key=(source,event.event_id)
            kind=('verified_cli_env_payload' if any(by_bridge[key].kind==BridgeKind.COMMAND for key in source.bridges)
                  else 'python_call_payload' if source.bridges else 'ast_def_use')
            paths[path_key]=payload_path(source,event,ast.unparse(node),'api_argument',kind,f'ast-{len(paths)}')

        def expression(node,state):
            nonlocal unit,functions,aliases
            if node is None:return set()
            if isinstance(node,ast.Name):return set(state.get(node.id,set()))
            if isinstance(node,ast.Attribute):return set(state.get(ast.unparse(node),expression(node.value,state)))
            if isinstance(node,ast.Subscript):
                exact=ast.unparse(node)
                if isinstance(node.slice,ast.Constant) and isinstance(node.slice.value,str):
                    field=dictionary(node.value,state).get(node.slice.value)
                    if field is not None:return set(field)
                return set(state.get(exact,expression(node.value,state))) | expression(node.slice,state)
            if isinstance(node,ast.Call):
                api=ast.unparse(node.func)
                qualified=_qualified(node.func,aliases)
                model=_PYTHON_APIS.get(qualified)
                source_kind=model[0] if model is not None else None
                if api=='open' or qualified in {'io.open','json.load'} or isinstance(node.func,ast.Attribute) and node.func.attr in {'open','read','readline','readlines','read_text','read_bytes'}:
                    source_kind='file_read'
                returned_sources=set()
                inputs=set().union(*(expression(arg,state) for arg in node.args),
                                  *(expression(keyword.value,state) for keyword in node.keywords))
                if isinstance(node.func,ast.Attribute):
                    inputs |= expression(node.func.value,state)
                local_events=by_location.get((unit,node.lineno),[])
                for event in local_events:
                    if event.capability in _SINKS:
                        url_slot=1 if api in {'requests.request','httpx.request'} else 0
                        payload_inputs=set().union(*(expression(arg,state) for arg in node.args[url_slot:url_slot+1]),
                            *(expression(keyword.value,state) for keyword in node.keywords
                              if keyword.arg in {'url','data','json','headers','files','params','args','env','input'}))
                        for keyword in node.keywords:
                            if keyword.arg is None:
                                fields=dictionary(keyword.value,state)
                                payload_inputs.update(set().union(*(value for name,value in fields.items()
                                    if name in {'url','data','json','headers','files','params','args','env','input'})))
                        for source in payload_inputs:
                            if source.event_id==event.event_id:continue
                            emit_path(source,event,node)
                    if event.capability in _SOURCES and (
                        event.capability==source_kind or source_kind=='file_read' and event.capability=='token_file_read'
                        or event.detail.startswith(api+' -> ')):
                        returned_sources.add(source_origin(event))
                        if event.capability=='env_var_read':
                            for resource in resources_by_event[event.event_id]:
                                if resource.resource_type=='env_var' and resource.resolved:
                                    returned_sources.update(state.get('env:'+resource.value,set()))
                for bridge in bridges:
                    if bridge.kind!=BridgeKind.COMMAND or bridge.source_unit_id!=unit or bridge.source_event_id not in {event.event_id for event in local_events}:continue
                    invocation=node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg=='args'),None)
                    if not isinstance(invocation,(ast.List,ast.Tuple)):continue
                    argv=list(invocation.elts)
                    if any(isinstance(arg,ast.Starred) for arg in argv):continue
                    # The bridge has resolved whether argv includes an
                    # interpreter (including sys.executable) or a direct script.
                    offset=len(argv)-len(bridge.cli_arguments)
                    if offset not in {0,1}:continue
                    argv=argv[offset:]
                    environment={}
                    for keyword in node.keywords:
                        if keyword.arg=='env':
                            environment=dictionary(keyword.value,state)
                    payload=invocation_payload(bridge,by_id[bridge.source_event_id],
                        [expression(arg,state) for arg in argv],environment,unit_paths[bridge.target_unit_id])
                    if invocations is not None:
                        invocations.append(payload)
                    elif bridge.target_unit_id in trees:
                        pending.append((bridge.target_unit_id,entry_bindings(payload)))
                called=functions.get(api) if isinstance(node.func,ast.Name) else (
                    functions.get(node.func.attr) if isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id in {'self','cls'} else None)
                target_unit=unit
                imported=local_python_function(node.func,unit_paths[unit],modules)
                bridge=None
                if imported is not None and imported[0]!=unit_paths[unit]:
                    target_unit=modules[imported[0]]['unit']
                    bridge=import_bridges.get((unit,target_unit))
                    called=modules[imported[0]]['functions'][imported[1]] if bridge is not None else None
                call_key=(target_unit,called.lineno,called.name) if called is not None else None
                if called is not None and call_key not in active:
                    active.add(call_key)
                    caller=unit
                    caller_functions=functions
                    caller_aliases=aliases
                    arguments=[expression(arg,state) for arg in node.args]
                    keywords={keyword.arg:expression(keyword.value,state) for keyword in node.keywords if keyword.arg is not None}
                    fields=[dictionary(arg,state) for arg in node.args]
                    keyword_fields={keyword.arg:dictionary(keyword.value,state) for keyword in node.keywords if keyword.arg is not None}
                    for keyword in node.keywords:
                        if keyword.arg is None:
                            keywords.update(dictionary(keyword.value,state))
                    if target_unit!=caller:
                        unit=target_unit
                        functions=module_functions[unit]
                        aliases=modules[unit_paths[unit]]['aliases']
                        if unit not in global_states:
                            global_states[unit]=dict(scope_entry)
                            statements(trees[unit].body,global_states[unit])
                        local=global_states[unit].copy()
                    else:
                        local=state.copy()
                    aliases=function_aliases(target_unit,called)
                    parameters=list(called.args.posonlyargs) + list(called.args.args)
                    if parameters and parameters[0].arg in {'self','cls'}:parameters=parameters[1:]
                    for parameter in [*parameters, *called.args.kwonlyargs]:local.pop(parameter.arg,None)
                    binding={parameter.arg:value for parameter,value in zip(parameters,arguments)}
                    binding.update(keywords)
                    field_binding={parameter.arg:value for parameter,value in zip(parameters,fields)}
                    field_binding.update(keyword_fields)
                    if called.args.kwarg is not None:
                        named={parameter.arg for parameter in [*parameters,*called.args.kwonlyargs]}
                        extras={name:value for name,value in keywords.items() if name not in named}
                        binding[called.args.kwarg.arg]=set().union(*extras.values())
                        field_binding[called.args.kwarg.arg]=extras
                    if called.args.vararg is not None:
                        extras=arguments[len(parameters):]
                        binding[called.args.vararg.arg]=set().union(*extras)
                        field_binding[called.args.vararg.arg]={str(index):value for index,value in enumerate(extras)}
                    location=f'{unit_paths[target_unit]}:{called.lineno}'
                    for name,value in binding.items():
                        local[name]=cross(value,bridge,target_unit,location,api) if bridge is not None else value
                    for name,items in field_binding.items():
                        for field,value in items.items():
                            local['field:'+name+':'+field]=cross(value,bridge,target_unit,location,api) if bridge is not None else value
                    slots={}
                    returns=statements(called.body,local,slots)
                    unit=caller
                    functions=caller_functions
                    aliases=caller_aliases
                    if bridge is not None:
                        location=f'{unit_paths[caller]}:{node.lineno}'
                        returns=cross(returns,bridge,caller,location,api)
                        slots={index:cross(value,bridge,caller,location,api) for index,value in slots.items()}
                    call_slots[id(node)]=slots
                    active.remove(call_key)
                    return returns|returned_sources
                if qualified in _VALUE_APIS or isinstance(node.func,ast.Name) and qualified is None and api in _VALUE_BUILTINS and api not in state:
                    return inputs|returned_sources
                if isinstance(node.func,ast.Attribute) and node.func.attr in _VALUE_METHODS:
                    return inputs|returned_sources
                # A network/opaque call's result does not prove propagation of
                # its request/arguments. Sources and modeled value carriers do.
                return returned_sources
            return set().union(*(expression(child,state) for child in ast.iter_child_nodes(node)))
        def statements(nodes,state,return_slots=None):
            returns=set()
            for node in nodes:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):continue
                if isinstance(node,(ast.Assign,ast.AnnAssign)):
                    value=expression(node.value,state)
                    fields=dictionary(node.value,state) if node.value is not None else {}
                    targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target,(ast.Name,ast.Attribute)):
                            name=ast.unparse(target)
                            state[name]=value
                            for key in [key for key in state if key.startswith('field:'+name+':')]:state.pop(key)
                            for key,lineage in fields.items():state['field:'+name+':'+key]=lineage
                        elif isinstance(target,ast.Subscript) and isinstance(target.slice,ast.Constant):
                            state['field:'+ast.unparse(target.value)+':'+str(target.slice.value)]=value
                        elif isinstance(target,(ast.Tuple,ast.List)):
                            slots=call_slots.get(id(node.value),{}) if isinstance(node.value,ast.Call) else {}
                            operands=node.value.elts if isinstance(node.value,(ast.Tuple,ast.List)) else []
                            for index,item in enumerate(target.elts):
                                state[ast.unparse(item)]=(expression(operands[index],state) if index<len(operands)
                                                         else set(slots.get(index,set())))
                elif isinstance(node,ast.Return):
                    returns|=expression(node.value,state)
                    if return_slots is not None and isinstance(node.value,(ast.Tuple,ast.List)):
                        for index,item in enumerate(node.value.elts):
                            return_slots.setdefault(index,set()).update(expression(item,state))
                elif isinstance(node,ast.If):
                    expression(node.test,state)
                    left,right=state.copy(),state.copy()
                    returns|=statements(node.body,left,return_slots)|statements(node.orelse,right,return_slots)
                    for name in set(left)|set(right):state[name]=left.get(name,set())|right.get(name,set())
                elif isinstance(node,(ast.With,ast.AsyncWith)):
                    for item in node.items:
                        value=expression(item.context_expr,state)
                        if item.optional_vars:state[ast.unparse(item.optional_vars)]=value
                    returns|=statements(node.body,state,return_slots)
                elif isinstance(node,ast.Try):
                    returns|=statements(node.body,state,return_slots)
                    for handler in node.handlers:returns|=statements(handler.body,state.copy(),return_slots)
                    returns|=statements(node.orelse,state,return_slots)|statements(node.finalbody,state,return_slots)
                elif isinstance(node,(ast.For,ast.AsyncFor,ast.While)):
                    expression(node.iter if hasattr(node,'iter') else node.test,state)
                    local=state.copy()
                    returns|=statements(node.body,local,return_slots)
                    for name,value in local.items():state[name]=state.get(name,set())|value
                else:expression(node,state)
            return returns
        state=dict(scope_entry)
        global_states[unit]=state
        statements(tree.body,state)
        # Entry points not reached by a static call remain analyzable. Class field
        # assignments share their actual read lineage across methods.
        for function in functions.values():
            aliases=function_aliases(unit,function)
            local=state.copy()
            for parameter in [*function.args.posonlyargs,*function.args.args,*function.args.kwonlyargs]:
                local.pop(parameter.arg,None)
            statements(function.body,local)
            state.update({name:value for name,value in local.items() if name.startswith('self.')})
    return list(paths.values())
