"""Shell AST def-use, stdout pipelines, function scope and command entry slots."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import shlex

from tree_sitter import Language, Parser
import tree_sitter_bash

from skillrecon.behavior.lineage import invocation_payload, payload_path, source_origin
from skillrecon.behavior.operational import _SHELL_BUILTINS
from skillrecon.behavior.shell import timeout_command_index
from skillrecon.core.enums import BridgeKind


class _State(dict):
    def __init__(self, values=(), exported=(), aliases=(), local=(), expand_aliases=False):
        super().__init__(values)
        self.exported = set(exported)
        self.aliases = dict(aliases)
        self.local = set(local)
        self.expand_aliases = expand_aliases

    def copy(self):
        return _State(self, self.exported, self.aliases, self.local, self.expand_aliases)


def _union(values):
    return frozenset(origin for value in values for origin in value)


def recover_shell_flows(events, resources, unit_paths, staged_root, bridges,
                        *, invocations=None, entry=None):
    locations, environment_reads, operations = defaultdict(list), defaultdict(set), defaultdict(set)
    for event in events:
        if event.language == 'bash' and event.tier != 'instruction':
            locations[event.unit_id, event.line].append(event)
    for resource in resources:
        if resource.resource_type == 'env_var':
            environment_reads[resource.event_id].add(resource.value)
        elif resource.resource_type == 'registered_operation':
            fact = json.loads(resource.origin_hint)
            operations[resource.event_id].add((fact['column'], fact['operation']))
    command_bridges = defaultdict(list)
    for bridge in bridges:
        if bridge.kind == BridgeKind.COMMAND:
            command_bridges[bridge.source_event_id].append(bridge)
    incoming_bridge = next((bridge for bridge in bridges if entry is not None
                            and bridge.bridge_id == entry.bridge_id), None)
    paths = {}

    for unit, relative in unit_paths.items():
        if Path(relative).suffix not in {'.sh', '.bash'}:
            continue
        if incoming_bridge is not None and unit != incoming_bridge.target_unit_id:
            continue
        root = Parser(Language(tree_sitter_bash.language())).parse((staged_root/relative).read_bytes()).root_node
        functions, active = {}, set()

        def collect(node):
            if node.type == 'function_definition':
                functions[node.child_by_field_name('name').text.decode()] = node
            for child in node.named_children:
                collect(child)
        collect(root)

        def emit(origins, event, node, slot):
            for origin in sorted(origins, key=lambda item: (item.event_id, item.bridges, item.trail)):
                if origin.event_id == event.event_id:
                    continue
                key = (origin, event.event_id, slot)
                paths[key] = payload_path(origin, event, node.text.decode(), slot,
                    'verified_cli_env_payload' if origin.bridges else 'shell_def_use',
                    f'shell-ast-{len(paths)}')

        def value(node, state):
            if node is None:
                return frozenset()
            kind = node.type
            if kind in {'simple_expansion', 'expansion'}:
                variable = next((child for child in node.named_children
                                 if child.type in {'variable_name', 'subscript'}), None)
                if variable is None:
                    return frozenset()
                name_node = variable.child_by_field_name('name') if variable.type == 'subscript' else variable
                name = name_node.text.decode()
                bound = state.get(name, frozenset())
                reads = frozenset(source_origin(event) for event in locations[unit, node.start_point.row+1]
                                  if event.capability == 'env_var_read' and name in environment_reads[event.event_id])
                operands = [child for child in node.named_children if child != variable]
                fallback = _union(value(child, state) for child in operands)
                return fallback if ':+' in node.text.decode() else bound | reads | fallback
            if kind in {'command_substitution', 'process_substitution'}:
                return execute(node.named_children, state.copy())[0]
            if kind in {'string', 'concatenation', 'arithmetic_expansion'}:
                return _union(value(child, state) for child in node.named_children)
            return frozenset()

        def merge(state, branches):
            for name in set().union(*(branch.keys() for branch in branches)):
                state[name] = _union(branch.get(name, frozenset()) for branch in branches)
            state.exported = set().union(*(branch.exported for branch in branches))
            state.local = set().union(*(branch.local for branch in branches))
            state.aliases = {name: replacement for name, replacement in branches[0].aliases.items()
                             if all(branch.aliases.get(name) == replacement for branch in branches)}
            state.expand_aliases = any(branch.expand_aliases for branch in branches)

        def call_function(name, arguments, state):
            if name in active:
                return frozenset()
            local = state.copy()
            local.local = set()
            for key in [key for key in local if key.isdigit() and key != '0']:
                local.pop(key)
            local.update({str(index+1): argument for index, argument in enumerate(arguments)})
            local['@'] = local['*'] = _union(arguments)
            active.add(name)
            output, _ = execute(functions[name].child_by_field_name('body').named_children, local)
            active.remove(name)
            # Bash function parameters and declared locals return to their caller;
            # assignments to non-local variables retain their actual effects.
            for key in set(state) | set(local):
                if key not in local.local and not key.isdigit() and key not in {'@', '*', '#'}:
                    if key in local:
                        state[key] = local[key]
                    else:
                        state.pop(key, None)
            state.exported = (local.exported - local.local) | (state.exported & local.local)
            state.aliases, state.expand_aliases = local.aliases, local.expand_aliases
            return output

        def command(node, state, stdin):
            local = state.copy()
            prefix = set()
            for child in node.named_children:
                if child.type == 'variable_assignment':
                    name = child.child_by_field_name('name').text.decode()
                    local[name] = value(child.child_by_field_name('value'), local)
                    prefix.add(name)
            name_node = node.child_by_field_name('name')
            name = name_node.text.decode().strip('\"\'')
            argument_nodes = node.children_by_field_name('argument')
            arguments = [value(arg, local) for arg in argument_nodes]
            literals = [arg.text.decode().strip('\"\'') for arg in argument_nodes]
            if local.expand_aliases and name in local.aliases:
                expanded = shlex.split(local.aliases[name])
                name = expanded[0]
                arguments = [frozenset()] * (len(expanded)-1) + arguments
                literals = expanded[1:] + literals
            name = Path(name).name if name != '.' else name
            if name in functions:
                return call_function(name, arguments, state), True
            if name == 'alias':
                for literal in literals:
                    if '=' in literal:
                        key, replacement = literal.split('=', 1)
                        state.aliases[key] = replacement.strip('\"\'')
            elif name == 'unalias':
                for literal in literals:
                    state.aliases.pop(literal, None)
            elif name == 'shopt' and 'expand_aliases' in literals:
                state.expand_aliases = '-s' in literals
            elif name in {'return', 'exit', 'break', 'continue'}:
                return frozenset(), False
            elif name == 'read':
                names = [literal for literal in literals if re.fullmatch(r'[A-Za-z_]\w*', literal)]
                for key in names or ['REPLY']:
                    state[key] = stdin
            payload = _union(arguments) | stdin | value(name_node, local)
            network_name, network_arguments, network_literals = name, arguments, literals
            if name == 'timeout':
                index = timeout_command_index(literals)
                if index is not None:
                    network_name = literals[index]
                    network_arguments, network_literals = arguments[index+1:], literals[index+1:]
            local_events = locations[unit, node.start_point.row+1]
            for event in local_events:
                if event.capability in {'env_var_read', 'file_write'}:
                    continue
                if operations[event.event_id] and (node.start_point.column, name) not in operations[event.event_id]:
                    continue
                if event.capability in {'shell_exec', 'subprocess', 'eval_exec'}:
                    if name in _SHELL_BUILTINS:
                        continue
                    emit(payload, event, node, 'command_arguments_stdin')
                    for key in sorted(local.exported | prefix):
                        emit(local.get(key, frozenset()) - payload, event, node, 'command_environment:'+key)
                elif event.capability in {'http_request', 'upload_exfil', 'remote_sync', 'telemetry_post', 'ssh_connect'}:
                    network = _union(network_arguments) | stdin
                    if network_name in {'curl', 'wget'}:
                        network, option = stdin, None
                        data_options = {'--url', '--data', '--data-raw', '--data-binary', '--data-urlencode',
                            '-d', '--header', '-H', '--form', '-F', '--upload-file', '-T', '--user', '-u',
                            '--post-data', '--post-file'}
                        value_options = data_options | {'--max-time', '-m', '--connect-timeout', '--retry',
                            '--retry-delay', '--output', '-o', '--output-document', '-O', '--request', '-X',
                            '--limit-rate', '--user-agent', '-A', '--proxy', '-x', '--config', '-K'}
                        for literal, argument in zip(network_literals, network_arguments):
                            if option is not None:
                                if option in data_options:
                                    network |= argument
                                option = None
                            elif literal in value_options:
                                option = literal
                            elif any(literal.startswith(flag+'=') for flag in data_options if flag.startswith('--')):
                                network |= argument
                            elif any(literal.startswith(flag) and len(literal) > len(flag)
                                     for flag in data_options if flag.startswith('-') and not flag.startswith('--')):
                                network |= argument
                            elif not literal.startswith('-'):
                                network |= argument
                    emit(network, event, node, 'network_arguments_stdin')
                for bridge in command_bridges[event.event_id] if invocations is not None else []:
                    slots = [value(name_node, local), *arguments]
                    offset = len(slots) - len(bridge.cli_arguments)
                    if offset in {0, 1}:
                        invocations.append(invocation_payload(bridge, event, slots[offset:],
                            {key: local.get(key, frozenset()) for key in local.exported | prefix},
                            unit_paths[bridge.target_unit_id]))
            sources = frozenset(source_origin(event) for event in local_events
                                if event.capability in {'file_read', 'token_file_read', 'keychain_access'})
            if name == 'printenv':
                sources |= frozenset(source_origin(event) for event in local_events if event.capability == 'env_var_read')
                sources |= _union(local.get(key, frozenset()) for key in literals)
            if name in {'echo', 'printf', 'basename', 'dirname'}:
                return _union(arguments), True
            if name in {'cat', 'head', 'tail', 'grep', 'sed', 'awk', 'cut', 'tr', 'sort', 'uniq', 'base64', 'jq', 'tee'}:
                return stdin | sources, True
            return sources, True

        def execute(nodes, state, stdin=frozenset()):
            output = frozenset()
            for node in nodes:
                kind = node.type
                if kind in {'comment', 'function_definition'}:
                    continue
                if kind == 'variable_assignment':
                    state[node.child_by_field_name('name').text.decode()] = value(node.child_by_field_name('value'), state)
                elif kind == 'declaration_command':
                    operator = node.children[0].type
                    names = []
                    for child in node.named_children:
                        if child.type == 'variable_assignment':
                            name = child.child_by_field_name('name').text.decode()
                            state[name] = value(child.child_by_field_name('value'), state)
                            names.append(name)
                        elif child.type == 'variable_name':
                            name = child.text.decode()
                            names.append(name)
                            if operator == 'local':
                                state[name] = frozenset()
                    if operator == 'local':
                        state.local.update(names)
                        for name in names:
                            state.setdefault(name, frozenset())
                    elif operator == 'export' and not re.search(rb'\bexport\s+-f\b', node.text):
                        if re.search(rb'\bexport\s+-n\b', node.text):
                            state.exported.difference_update(names)
                        else:
                            state.exported.update(names)
                elif kind == 'unset_command':
                    for child in node.named_children:
                        name = child.text.decode()
                        state.pop(name, None)
                        state.exported.discard(name)
                elif kind == 'command':
                    result, falls = command(node, state, stdin)
                    output |= result
                    if not falls:
                        return output, False
                elif kind == 'pipeline':
                    result = stdin
                    for child in node.named_children:
                        result, _ = execute([child], state.copy(), result)
                    output |= result
                elif kind == 'redirected_statement':
                    body = node.child_by_field_name('body')
                    result, falls = execute([body], state, stdin)
                    redirected = False
                    for redirect in node.children_by_field_name('redirect'):
                        destination = redirect.child_by_field_name('destination')
                        if destination is None:
                            continue
                        if destination.type in {'number', 'file_descriptor'}:
                            redirected |= redirect.text.lstrip().startswith(b'>&')
                            continue
                        if b'>' in redirect.text:
                            redirected = True
                            for event in locations[unit, redirect.start_point.row+1]:
                                if event.capability == 'file_write':
                                    emit(result, event, redirect, 'redirected_stdout')
                        value(destination, state)
                    if not redirected:
                        output |= result
                    if not falls:
                        return output, False
                elif kind == 'if_statement':
                    branches, results = [], []
                    branch_nodes = []
                    has_else = False
                    conditions = node.children_by_field_name('condition')
                    execute([child for child in conditions if child.is_named], state)
                    for child in node.named_children:
                        if child in conditions:
                            continue
                        if child.type in {'elif_clause', 'else_clause'}:
                            branches.append(branch_nodes)
                            branch_nodes = [item for item in child.named_children
                                            if item not in child.children_by_field_name('condition')]
                            execute([item for item in child.children_by_field_name('condition') if item.is_named], state.copy())
                            has_else |= child.type == 'else_clause'
                        else:
                            branch_nodes.append(child)
                    branches.append(branch_nodes)
                    if not has_else:
                        branches.append([])
                    continuing = []
                    for branch_nodes in branches:
                        branch = state.copy()
                        result, falls = execute(branch_nodes, branch)
                        results.append(result)
                        if falls:
                            continuing.append(branch)
                    output |= _union(results)
                    if not continuing:
                        return output, False
                    merge(state, continuing)
                elif kind == 'case_statement':
                    branches = [state.copy()]
                    for child in node.named_children:
                        if child.type == 'case_item':
                            branch = state.copy()
                            result, _ = execute(child.named_children, branch)
                            output |= result
                            branches.append(branch)
                    merge(state, branches)
                elif kind in {'for_statement', 'while_statement'}:
                    branch = state.copy()
                    variable = node.child_by_field_name('variable')
                    if variable is not None:
                        branch[variable.text.decode()] = _union(value(child, state) for child in node.children_by_field_name('value'))
                    body = node.child_by_field_name('body')
                    if body is not None:
                        result, _ = execute(body.named_children, branch, stdin)
                        output |= result
                    merge(state, [state.copy(), branch])
                elif kind == 'list':
                    children = node.named_children
                    if children:
                        result, falls = execute(children[:1], state, stdin)
                        output |= result
                        if not falls:
                            return output, False
                    for child in children[1:]:
                        branch = state.copy()
                        result, falls = execute([child], branch, stdin)
                        output |= result
                        if falls:
                            merge(state, [state.copy(), branch])
                elif kind in {'subshell', 'command_substitution', 'process_substitution'}:
                    output |= execute(node.named_children, state.copy(), stdin)[0]
                elif kind in {'compound_statement', 'do_group', 'program'}:
                    result, falls = execute(node.named_children, state, stdin)
                    output |= result
                    if not falls:
                        return output, False
                else:
                    # Tests and conditions evaluate expansions but do not carry
                    # the tested value into a function's stdout.
                    value(node, state)
            return output, True

        state = _State()
        if entry is not None:
            state.update({str(index): origins for index, origins in enumerate(entry.arguments)})
            state.update(entry.environment)
            state.exported.update(entry.environment)
            state['@'] = state['*'] = _union(entry.arguments[1:])
        execute(root.named_children, state)
    return list(paths.values())
