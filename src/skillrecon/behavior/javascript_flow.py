"""Tier-2 JS/TS def-use paths over resolved calls and actual payload slots.

Values carry field-specific source lineages. Local calls use parameter bindings
and actual returns; package-local calls additionally retain static-import hops.
Unknown calls do not acquire the taint of every argument by default.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from skillrecon.behavior.javascript import FUNCTION_TYPES, javascript_modules
from skillrecon.behavior.lineage import PayloadOrigin, invocation_payload, source_origin
from skillrecon.behavior.native import javascript_api_model
from skillrecon.behavior.resource_fields import _field, _js_literal, _text
from skillrecon.core.enums import BridgeKind
from skillrecon.core.types import PathSegment, RiskPath

_SOURCES = {'file_read', 'token_file_read', 'env_var_read', 'keychain_access', 'credential_store_access'}
_SINKS = {'http_request', 'upload_exfil', 'telemetry_post', 'remote_sync',
          'file_write', 'shell_exec', 'subprocess', 'eval_exec'}
_TRANSFORMS = {'JSON.parse', 'JSON.stringify', 'String', 'Number', 'Buffer.from',
               'Array.from', 'Object.values', 'Object.entries', 'encodeURIComponent', 'decodeURIComponent'}
_TRANSFORMS.update(f'{prefix}.{name}' for prefix in ['path', 'node:path'] for name in
                   ['join', 'resolve', 'dirname', 'basename', 'extname', 'normalize', 'relative'])
_STRING_METHODS = {'trim', 'trimStart', 'trimEnd', 'toString', 'toLowerCase', 'toUpperCase',
                   'normalize', 'replace', 'replaceAll', 'substring', 'substr', 'split', 'join', 'slice', 'concat'}


@dataclass(frozen=True)
class _Function:
    relative: str
    node: object
    closure: dict | None = field(default=None, compare=False, hash=False)


@dataclass
class _Value:
    origins: frozenset[PayloadOrigin] = frozenset()
    fields: dict[str, '_Value'] = field(default_factory=dict)
    functions: tuple[_Function, ...] = ()
    apis: frozenset[str] = frozenset()
    promise: bool = False
    requests: frozenset[str] = frozenset()

    def sources(self):
        return self.origins | frozenset(origin for value in self.fields.values() for origin in value.sources())


def _merge(*values):
    keys = set().union(*(value.fields for value in values))
    return _Value(origins=frozenset(origin for value in values for origin in value.origins),
                  fields={key: _merge(*(value.fields[key] for value in values if key in value.fields)) for key in keys},
                  functions=tuple(dict.fromkeys(function for value in values for function in value.functions)),
                  apis=frozenset(api for value in values for api in value.apis),
                  promise=any(value.promise for value in values),
                  requests=frozenset(identifier for value in values for identifier in value.requests))


def recover_javascript_flows(events, resources, unit_paths, staged_root, bridges,
                             *, invocations=None, entry=None):
    modules = javascript_modules(unit_paths, staged_root)
    by_event = {event.event_id: event for event in events}
    locations = defaultdict(list)
    for event in events:
        if event.tier != 'instruction':
            locations[event.unit_id, event.line].append(event)
    import_bridges = {}
    for bridge in bridges:
        if bridge.kind == BridgeKind.STATIC:
            import_bridges.setdefault((bridge.source_unit_id, bridge.target_unit_id), bridge)
    command_bridges = defaultdict(list)
    for bridge in bridges:
        if bridge.kind == BridgeKind.COMMAND:
            command_bridges[bridge.source_event_id].append(bridge)
    incoming_bridge = next((bridge for bridge in bridges if entry is not None
                            and bridge.bridge_id == entry.bridge_id), None)
    environments, paths, active = {}, {}, set()

    def origin(event):
        return source_origin(event)

    def cross(value, bridge, unit, location, expression):
        def move(item):
            if item.trail[-1][0] == unit:
                return item
            return PayloadOrigin(item.event_id, (*item.trail, (unit, location, 'AST carrier', None,
                           'function_binding', expression)), (*item.bridges, bridge.bridge_id))
        return _Value(frozenset(move(item) for item in value.origins),
                      {key: cross(item, bridge, unit, location, expression) for key, item in value.fields.items()},
                      value.functions, value.apis, value.promise, value.requests)

    def emit(value, event, expression, slot):
        for source in value.sources():
            if source.event_id == event.event_id:
                continue
            trail = (*source.trail, (event.unit_id, event.location, event.capability,
                                    event.event_id, slot, expression))
            segments = [PathSegment(unit_id=u, location=loc, label=label, event_id=identifier,
                                    slot_kind=kind, expr_hint=expr) for u, loc, label, identifier, kind, expr in trail]
            key = (source.event_id, event.event_id, source.bridges, source.trail)
            paths[key] = RiskPath(path_id=f'js-ast-{len(paths)}', source=segments[0], sink=segments[-1],
                segments=segments, bridges_used=list(source.bridges),
                evidence_level='bridge' if source.bridges else 'registered_source',
                path_kind='bridge:javascript_call_payload' if source.bridges else 'registered_source:javascript_def_use')

    def select(value, name):
        selected = value.fields.get(name, value.fields.get('*', _Value()))
        return _merge(selected, _Value(value.origins,
                      apis=frozenset(f'{api}.{name}' for api in value.apis)))

    def imported(target):
        relative, name = target
        if relative not in modules:
            return _Value()
        module = modules[relative]
        def exported(name):
            symbol = module['exports'].get(name)
            if symbol in module['functions']:
                return _Value(functions=(_Function(relative, module['functions'][symbol]),))
            return environments.get(relative, {}).get(symbol, _Value())
        if name == '*':
            return _Value(fields={key: exported(key) for key in module['exports']},
                          functions=exported('default').functions)
        return exported(name)

    def initial(relative):
        module = modules[relative]
        state = {name: _Value(apis=frozenset({name})) for name in [
            'JSON', 'String', 'Number', 'Buffer', 'Array', 'Object', 'Promise', 'Set', 'Map',
            'process', 'encodeURIComponent', 'decodeURIComponent']}
        state.update({name: _Value(apis=frozenset({api})) for name, api in module['bindings'].items()})
        state.update({name: imported(target) for name, target in module['imports'].items()})
        state.update({name: _Value(functions=(_Function(relative, node),)) for name, node in module['functions'].items()})
        if entry is not None:
            state['process'].fields = {
                'env': _Value(fields={name: _Value(origins) for name, origins in entry.environment.items()},
                              apis=frozenset({'process.env'})),
                'argv': _Value(fields={str(index+1): _Value(origins)
                                      for index, origins in enumerate(entry.arguments)})}
        return state

    def command_payload(event, node, arguments):
        if invocations is None:
            return
        nodes = _field(node, 'arguments').named_children
        if len(nodes) < 2 or nodes[1].type != 'array':
            return
        # An unknown spread has no exact positional binding. In particular,
        # later literal slots must not be shifted onto the wrong child argv.
        if any(child.type == 'spread_element' for child in nodes[1].named_children):
            return
        length = len(nodes[1].named_children)
        if {key for key in arguments[1].fields if key.isdigit()} != {str(index) for index in range(length)}:
            return  # A hole before a supplied argument has no normalized slot yet.
        environment = select(arguments[2], 'env').fields if len(arguments) > 2 else {}
        for bridge in command_bridges[event.event_id]:
            slots = [arguments[1].fields[str(index)].sources() for index in range(length)]
            count = len(bridge.cli_arguments)
            if count == len(slots) + 1:  # execFile(script, userArgs)
                slots.insert(0, frozenset())
            elif count != len(slots):  # spawn(interpreter, [script, ...userArgs])
                continue
            invocations.append(invocation_payload(bridge, event, slots,
                {name: value.sources() for name, value in environment.items()},
                unit_paths[bridge.target_unit_id]))

    def bind(pattern, value, state, relative):
        if pattern.type in {'identifier', 'shorthand_property_identifier_pattern'}:
            state[_text(pattern)] = value
        elif pattern.type in {'required_parameter', 'optional_parameter'}:
            bind(_field(pattern, 'pattern'), value, state, relative)
        elif pattern.type == 'assignment_pattern':
            bind(_field(pattern, 'left'), _merge(value, evaluate(_field(pattern, 'right'), state, relative)), state, relative)
        elif pattern.type == 'object_pattern':
            for child in pattern.named_children:
                if child.type == 'shorthand_property_identifier_pattern':
                    bind(child, select(value, _text(child)), state, relative)
                elif child.type == 'pair_pattern':
                    bind(_field(child, 'value'), select(value, _text(_field(child, 'key')).strip('\"\'')), state, relative)
        elif pattern.type == 'array_pattern':
            for index, child in enumerate(pattern.children):
                if child.is_named:
                    # Commas, rather than named-child indices, preserve holes.
                    position=sum(item.type == ',' for item in pattern.children[:index])
                    bind(child, select(value, str(position)), state, relative)
        elif pattern.type in {'member_expression', 'subscript_expression'}:
            base = _field(pattern, 'object')
            key_node = _field(pattern, 'property') or _field(pattern, 'index')
            key = _text(key_node) if pattern.type == 'member_expression' else _js_literal(key_node, {})
            if base.type == 'identifier' and key is not None:
                previous = state.get(_text(base), _Value())
                state[_text(base)] = _Value(previous.origins, {**previous.fields, key: value},
                                            previous.functions, previous.apis, previous.promise, previous.requests)

    def invoke(function, args, caller, call):
        key = (function.relative, function.node.start_byte)
        if key in active:
            return _Value()
        relative = function.relative
        bridge = None
        if relative != caller:
            bridge = import_bridges.get((modules[caller]['unit'], modules[relative]['unit']))
            if bridge is None:
                return _Value()
            args = [cross(value, bridge, modules[relative]['unit'],
                          f'{relative}:{function.node.start_point.row+1}', _text(call)) for value in args]
        state = dict(function.closure if function.closure is not None else environments.get(relative, initial(relative)))
        parameters = _field(function.node, 'parameters')
        parameters = parameters.named_children if parameters is not None else [_field(function.node, 'parameter')]
        for index, parameter in enumerate(parameters):
            if parameter is not None:
                if parameter.type == 'assignment_pattern' and index < len(args):
                    parameter = _field(parameter, 'left')
                bind(parameter, args[index] if index < len(args) else _Value(), state, relative)
        active.add(key)
        body = _field(function.node, 'body')
        if body.type == 'statement_block':
            result, _ = execute(body.named_children, state, relative)
        else:
            result = evaluate(body, state, relative)
        active.remove(key)
        if bridge is not None:
            result = cross(result, bridge, modules[caller]['unit'],
                           f'{caller}:{call.start_point.row+1}', _text(call))
        return result

    def callbacks(value, args, state, relative, call):
        return _merge(*(invoke(function, args, relative, call) for function in value.functions))

    def evaluate(node, state, relative):
        if node is None:
            return _Value()
        kind = node.type
        unit = modules[relative]['unit']
        if kind == 'identifier':
            return state.get(_text(node), _Value())
        if kind in FUNCTION_TYPES:
            return _Value(functions=(_Function(relative, node, state.copy()),))
        if kind in {'member_expression', 'subscript_expression'}:
            base = evaluate(_field(node, 'object'), state, relative)
            selector = _field(node, 'property') or _field(node, 'index')
            name = _text(selector) if kind == 'member_expression' else _js_literal(selector, {})
            value = select(base, name) if name is not None else _merge(
                _Value(base.origins), *(value for value in base.fields.values()))
            if any(api.removeprefix('node:') == 'process.env' for api in base.apis):
                reads = [event for event in locations[unit, node.start_point.row+1] if event.capability == 'env_var_read']
                value = _merge(value, _Value(frozenset(origin(event) for event in reads)))
            return value
        if kind == 'object':
            fields = {}
            for child in node.named_children:
                if child.type == 'pair':
                    name = _text(_field(child, 'key')).strip('\"\'')
                    fields[name] = evaluate(_field(child, 'value'), state, relative)
                elif child.type == 'shorthand_property_identifier':
                    fields[_text(child)] = state.get(_text(child), _Value())
                elif child.type == 'spread_element':
                    fields.update(evaluate(child.named_children[0], state, relative).fields)
            return _Value(fields=fields)
        if kind == 'array':
            fields = {}
            index = 0
            for child in node.children:
                if child.type == ',':
                    index += 1
                elif child.is_named:
                    value = evaluate(child, state, relative)
                    fields[str(index)] = value
                    fields['*'] = _merge(fields.get('*', _Value()), value)
            return _Value(fields=fields)
        if kind in {'assignment_expression', 'augmented_assignment_expression'}:
            value = evaluate(_field(node, 'right'), state, relative)
            left = _field(node, 'left')
            if kind == 'augmented_assignment_expression':
                value = _merge(evaluate(left, state, relative), value)
            bind(left, value, state, relative)
            return value
        if kind in {'ternary_expression', 'conditional_expression'}:
            evaluate(_field(node, 'condition'), state, relative)
            return _merge(evaluate(_field(node, 'consequence'), state.copy(), relative),
                          evaluate(_field(node, 'alternative'), state.copy(), relative))
        if kind == 'binary_expression':
            operator = _text(_field(node, 'operator'))
            left, right = evaluate(_field(node, 'left'), state, relative), evaluate(_field(node, 'right'), state, relative)
            return _merge(left, right) if operator in {'+', '||', '&&', '??'} else _Value()
        if kind in {'call_expression', 'new_expression'}:
            callee_node = _field(node, 'function') or _field(node, 'constructor')
            callee = evaluate(callee_node, state, relative)
            args_node = _field(node, 'arguments')
            arguments = [evaluate(arg, state, relative) for arg in args_node.named_children] if args_node is not None else []
            result = _merge(*(invoke(function, arguments, relative, node) for function in callee.functions))
            for api in callee.apis:
                model = javascript_api_model(api)
                local = locations[unit, node.start_point.row+1]
                if model is not None:
                    capability, _ = model
                    matching = [event for event in local if event.capability == capability or
                                capability == 'file_read' and event.capability == 'token_file_read']
                    if capability in _SOURCES:
                        result = _merge(result, _Value(frozenset(origin(event) for event in matching)))
                    if api.removeprefix('node:') in {'http.request', 'https.request'}:
                        result = _merge(result, _Value(requests=frozenset(event.event_id for event in matching)))
                    if capability in _SINKS:
                        if capability == 'file_write':
                            payload = arguments[1] if len(arguments) > 1 else _Value()
                            slot = 'api_argument:1'
                        elif capability == 'http_request':
                            target = arguments[0] if arguments else _Value()
                            if target.fields:
                                target = _merge(*(select(target, key) for key in
                                                  ['url', 'host', 'hostname', 'path', 'auth', 'headers', 'body', 'data']))
                            payload = _merge(target,
                                *(select(value, key) for value in arguments[1:] for key in
                                  ['body', 'headers', 'data', 'json', 'params']),
                                *(arguments[1:2] if api in {'axios.post','axios.put','axios.patch'} else []))
                            slot = 'http_payload'
                        else:
                            payload = _merge(*arguments[:2], *(select(value, key) for value in arguments[2:]
                                                               for key in ['env', 'input']))
                            slot = 'execution_argument'
                        for event in matching:
                            emit(payload, event, _text(node), slot)
                            if capability in {'shell_exec', 'subprocess'}:
                                command_payload(event, node, arguments)
                if api in _TRANSFORMS:
                    combined = _merge(*arguments)
                    result = _merge(result, _Value(combined.sources(), combined.fields))
                elif api in {'util.promisify', 'node:util.promisify'} and arguments:
                    wrapped = arguments[0]
                    result = _merge(result, _Value(functions=wrapped.functions, apis=wrapped.apis, promise=True))
                elif api == 'Promise':
                    if arguments:
                        callbacks(arguments[0], [_Value(), _Value()], state, relative, node)
                    result.promise = True
                elif api in {'Promise.resolve', 'Promise.all'}:
                    result = _merge(result, *arguments)
                    result.promise = True
                elif api in {'Set', 'Map'}:
                    result = _merge(result, *arguments)
            if callee_node.type == 'member_expression' and not callee.functions:
                receiver = evaluate(_field(callee_node, 'object'), state, relative)
                method = _text(_field(callee_node, 'property'))
                if method in {'write', 'end'} and arguments:
                    for identifier in receiver.requests:
                        emit(arguments[0], by_event[identifier], _text(node), 'request_body:0')
                element = _merge(_Value(receiver.origins), receiver.fields.get('*', _Value()))
                if method in {'map', 'filter', 'flatMap', 'find', 'forEach', 'some', 'every'} and arguments:
                    returned = callbacks(arguments[0], [element, _Value(), receiver], state, relative, node)
                    if method in {'filter', 'find'}:
                        result = _merge(result, receiver if method == 'filter' else element)
                    elif method in {'map', 'flatMap'}:
                        result = _merge(result, _Value(fields={'*': returned}))
                elif method == 'then' and receiver.promise and arguments:
                    result = _merge(result, callbacks(arguments[0], [receiver], state, relative, node))
                    result.promise = True
                elif method in _STRING_METHODS:
                    result = _merge(result, _Value(_merge(receiver, *arguments).sources()))
                elif method == 'add' and arguments and _field(callee_node, 'object').type == 'identifier':
                    target = _field(callee_node, 'object')
                    bind(target, _merge(receiver, _Value(fields={'*': arguments[0]})), state, relative)
            return result
        if kind in {'await_expression', 'parenthesized_expression', 'spread_element', 'template_substitution'}:
            return _merge(*(evaluate(child, state, relative) for child in node.named_children))
        if kind == 'template_string':
            return _Value(_merge(*(evaluate(child, state, relative) for child in node.named_children
                                   if child.type == 'template_substitution')).sources())
        return _Value()

    def merge_states(state, branches):
        for name in set().union(*(branch for branch in branches)):
            state[name] = _merge(*(branch.get(name, _Value()) for branch in branches))

    def execute(nodes, state, relative):
        returned = _Value()
        for node in nodes:
            kind = node.type
            if kind in {'lexical_declaration', 'variable_declaration'}:
                for declaration in node.named_children:
                    if declaration.type == 'variable_declarator':
                        value = _field(declaration, 'value')
                        pattern = _field(declaration, 'name')
                        # Resolved imports were installed in initial().
                        if pattern.type == 'identifier' and _text(pattern) in modules[relative]['imports']:
                            continue
                        if value is not None and value.type == 'call_expression' and _text(_field(value, 'function')) == 'require':
                            args = _field(value, 'arguments').named_children
                            if len(args) == 1 and _js_literal(args[0], {}) is not None:
                                continue
                        bind(pattern, evaluate(value, state, relative), state, relative)
            elif kind in FUNCTION_TYPES:
                name = _field(node, 'name')
                if name is not None and _text(name) not in modules[relative]['functions']:
                    state[_text(name)] = _Value(functions=(_Function(relative, node, state.copy()),))
            elif kind == 'return_statement':
                return _merge(returned, *(evaluate(child, state, relative) for child in node.named_children)), False
            elif kind in {'throw_statement', 'break_statement', 'continue_statement'}:
                for child in node.named_children:
                    evaluate(child, state, relative)
                return returned, False
            elif kind == 'if_statement':
                evaluate(_field(node, 'condition'), state, relative)
                continuing = []
                for branch in [_field(node, 'consequence'), _field(node, 'alternative')]:
                    local = state.copy()
                    value, falls = execute(branch.named_children if branch is not None and branch.type == 'statement_block'
                                           else [branch] if branch is not None else [], local, relative)
                    returned = _merge(returned, value)
                    if falls:
                        continuing.append(local)
                if not continuing:
                    return returned, False
                merge_states(state, continuing)
            elif kind == 'try_statement':
                branches = []
                for branch in [_field(node, 'body'), _field(node, 'handler')]:
                    if branch is None:
                        continue
                    local = state.copy()
                    body = _field(branch, 'body') if branch.type == 'catch_clause' else branch
                    value, falls = execute(body.named_children, local, relative)
                    returned = _merge(returned, value)
                    if falls:
                        branches.append(local)
                if branches:
                    merge_states(state, branches)
                final = _field(node, 'finalizer')
                if final is not None:
                    body = _field(final, 'body')
                    value, falls = execute(body.named_children, state, relative)
                    returned = _merge(returned, value)
                    if not falls:
                        return returned, False
                if not branches:
                    return returned, False
            elif kind in {'for_statement', 'for_in_statement', 'while_statement', 'do_statement'}:
                local = state.copy()
                initial_node = _field(node, 'initializer')
                if initial_node is not None:
                    execute([initial_node], local, relative)
                right = _field(node, 'right')
                if right is not None:
                    value = evaluate(right, local, relative)
                    target = _field(node, 'left')
                    if target is not None:
                        bind(target, _merge(_Value(value.origins), value.fields.get('*', _Value())), local, relative)
                body = _field(node, 'body')
                value, _ = execute(body.named_children if body.type == 'statement_block' else [body], local, relative)
                returned = _merge(returned, value)
                merge_states(state, [state.copy(), local])
            elif kind == 'statement_block':
                value, falls = execute(node.named_children, state, relative)
                returned = _merge(returned, value)
                if not falls:
                    return returned, False
            elif kind == 'export_statement':
                declaration = _field(node, 'declaration') or _field(node, 'value')
                if declaration is not None:
                    value, falls = execute([declaration], state, relative)
                    returned = _merge(returned, value)
                    if not falls:
                        return returned, False
            elif kind == 'expression_statement':
                for child in node.named_children:
                    evaluate(child, state, relative)
        return returned, True

    # Imports are declarations, so initialize every module before traversing a
    # top-level entry point that may call another package-local module.
    for relative in modules:
        environments[relative] = initial(relative)
    for relative, module in modules.items():
        execute(module['root'].named_children, environments[relative], relative)
    for relative, module in modules.items():
        if incoming_bridge is not None and module['unit'] != incoming_bridge.target_unit_id:
            continue
        for function in module['functions'].values():
            invoke(_Function(relative, function), [], relative, function)
    return list(paths.values())
