"""Resolved command bridges and propagated CLI/environment argument slots."""
from __future__ import annotations

import ast
import posixpath
import re
import shlex
from itertools import product
from pathlib import Path

from skillrecon.behavior.shell import shell_commands
from skillrecon.core.enums import BridgeKind
from skillrecon.core.types import Bridge, RiskPath

_INTERPRETERS={'python','python3','bash','sh','node','zsh'}


def _python_invocations(source, relative):
    """Resolve operands in lexical scopes; never evaluate package Python code."""
    tree = ast.parse(source)
    apis = {'subprocess.run', 'subprocess.call', 'subprocess.Popen',
            'subprocess.check_call', 'subprocess.check_output', 'os.system', 'os.popen'}
    invocations = []

    def qualified(node, aliases):
        first, *rest = ast.unparse(node).split('.')
        return '.'.join([aliases.get(first, first), *rest])

    def value(node, state, aliases):
        if isinstance(node, ast.Constant):
            return str(node.value), True
        if isinstance(node, ast.Name):
            if node.id == '__file__':
                return relative, True
            bound = state.get(node.id)
            if isinstance(bound, tuple) and bound[1]:
                return bound
        if isinstance(node, ast.Attribute):
            if qualified(node, aliases) == 'sys.executable':
                return 'python', True
            if node.attr == 'parent':
                path, resolved = value(node.value, state, aliases)
                if resolved:
                    return posixpath.dirname(path) or '.', True
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
            left, left_known = value(node.left, state, aliases)
            right, right_known = value(node.right, state, aliases)
            if left_known and right_known:
                return (posixpath.normpath(posixpath.join(left, right))
                        if isinstance(node.op, ast.Div) else left + right), True
        if isinstance(node, ast.Call):
            api = qualified(node.func, aliases)
            if api in {'str', 'pathlib.Path'} and len(node.args) == 1:
                return value(node.args[0], state, aliases)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {'resolve', 'absolute'} and not node.args:
                path, resolved = value(node.func.value, state, aliases)
                if resolved:
                    return posixpath.normpath(path), True
            if api in {'os.path.join', 'posixpath.join'}:
                parts = [value(arg, state, aliases) for arg in node.args]
                if parts and all(known for _, known in parts):
                    return posixpath.normpath(posixpath.join(*(part for part, _ in parts))), True
            if api in {'os.path.dirname', 'posixpath.dirname'} and len(node.args) == 1:
                path, resolved = value(node.args[0], state, aliases)
                if resolved:
                    return posixpath.dirname(path) or '.', True
        return ast.unparse(node), False

    def environment(node, state, aliases):
        if isinstance(node, ast.Name) and isinstance(state.get(node.id), dict):
            return state[node.id].copy()
        result = {}
        if isinstance(node, ast.Dict):
            for key, operand in zip(node.keys, node.values):
                if key is None:
                    result.update(environment(operand, state, aliases))
                elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                    result[key.value] = value(operand, state, aliases)
        return result

    def calls(node, state, aliases):
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or qualified(call.func, aliases) not in apis:
                continue
            operand = call.args[0] if call.args else next(
                (kw.value for kw in call.keywords if kw.arg == 'args'), None)
            if operand is None:
                continue
            if isinstance(operand, (ast.List, ast.Tuple)):
                argv = [value(arg, state, aliases) for arg in operand.elts]
            else:
                command, known = value(operand, state, aliases)
                argv = [(arg, True) for arg in shlex.split(command)] if known else []
            env = next((environment(kw.value, state, aliases) for kw in call.keywords if kw.arg == 'env'), {})
            invocations.append((call.lineno, argv, env))

    def merge(state, branches):
        for name in set().union(*(branch.keys() for branch in branches)):
            alternatives = [branch.get(name) for branch in branches]
            state[name] = alternatives[0] if all(item == alternatives[0] for item in alternatives) else (name, False)

    def statements(nodes, state, aliases):
        functions = []
        for node in nodes:
            if isinstance(node, ast.Import):
                for item in node.names:
                    aliases[item.asname or item.name] = item.name
            elif isinstance(node, ast.ImportFrom):
                for item in node.names:
                    aliases[item.asname or item.name] = f'{node.module}.{item.name}'
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                functions.append(node)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if node.value is None:
                    continue
                calls(node.value, state, aliases)
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                bound = environment(node.value, state, aliases) if isinstance(node.value, ast.Dict) else value(node.value, state, aliases)
                for target in targets:
                    if isinstance(target, ast.Name):
                        state[target.id] = bound
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and isinstance(target.slice, ast.Constant):
                        mapping = state.get(target.value.id)
                        if isinstance(mapping, dict):
                            mapping = mapping.copy()
                            mapping[str(target.slice.value)] = value(node.value, state, aliases)
                            state[target.value.id] = mapping
            elif isinstance(node, ast.If):
                calls(node.test, state, aliases)
                left, right = state.copy(), state.copy()
                statements(node.body, left, aliases.copy())
                statements(node.orelse, right, aliases.copy())
                merge(state, [left, right])
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                # Each control-flow arm has its own assignments and call sites.
                branches = [state.copy()]
                arms = [node.body, getattr(node, 'orelse', []), getattr(node, 'finalbody', [])]
                arms.extend(handler.body for handler in getattr(node, 'handlers', []))
                for arm in arms:
                    branch = state.copy()
                    statements(arm, branch, aliases.copy())
                    branches.append(branch)
                merge(state, branches)
            else:
                calls(node, state, aliases)
        for function in functions:
            local, local_aliases = state.copy(), aliases.copy()
            if not isinstance(function, ast.ClassDef):
                for param in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]:
                    local.pop(param.arg, None)
                    local_aliases.pop(param.arg, None)
            statements(function.body, local, local_aliases)

    statements(tree.body, {}, {})
    return invocations


def _javascript_bindings(call):
    """Literal argv slots and explicit env entries at an actual JS call site."""
    from skillrecon.behavior.resource_fields import _field, _text, _js_literal

    scopes = []
    scope = call.parent
    while scope is not None:
        if scope.type in {'program', 'statement_block', 'function_declaration', 'arrow_function', 'method_definition'}:
            scopes.append(scope)
        scope = scope.parent
    definitions = {}
    assignments = []

    def local_nodes(node):
        yield node
        for child in node.named_children:
            if child.type not in {'statement_block', 'function_declaration', 'function_expression', 'arrow_function', 'method_definition', 'class_declaration'}:
                yield from local_nodes(child)

    for scope in reversed(scopes):
        for node in local_nodes(scope):
            if node.start_byte >= call.start_byte:
                continue
            if node.type == 'formal_parameters':
                for param in node.named_children:
                    if param.type == 'identifier':
                        definitions.pop(_text(param), None)
            elif node.type == 'variable_declarator' and _field(node, 'name').type == 'identifier':
                definitions[_text(_field(node, 'name'))] = _field(node, 'value')
            elif node.type == 'assignment_expression':
                assignments.append(node)

    # Mutations in earlier conditional arms affect the same object even though
    # their lexical declarations do not escape the block.
    from skillrecon.behavior.resource_fields import _walk
    function_scope=next((scope for scope in scopes if scope.type in {
        'function_declaration','arrow_function','method_definition'}),scopes[-1])
    def function_owner(node):
        while node.parent is not None and node.type not in {
                'program','function_declaration','function_expression','arrow_function','method_definition'}:
            node=node.parent
        return node
    assignments=[node for node in _walk(function_scope) if node.type=='assignment_expression'
                 and node.start_byte<call.start_byte and function_owner(node)==function_scope]

    def literal(node, active=()):
        if node.type in {'identifier', 'shorthand_property_identifier'} and _text(node) in definitions and _text(node) not in active:
            defined = definitions[_text(node)]
            if defined is not None:
                return literal(defined, (*active, _text(node)))
        constants = {name: value for name, definition in definitions.items() if definition is not None
                     and (value := _js_literal(definition, {})) is not None}
        return _js_literal(node, constants)

    def object_entries(node, active=()):
        if node.type in {'identifier', 'shorthand_property_identifier'} and _text(node) in definitions and _text(node) not in active:
            definition = definitions[_text(node)]
            return object_entries(definition, (*active, _text(node))) if definition is not None else {}
        result = {}
        if node.type == 'object':
            for item in node.named_children:
                if item.type == 'pair':
                    key, operand = _field(item, 'key'), _field(item, 'value')
                    result[_text(key).strip('\"\'')] = operand
                elif item.type == 'shorthand_property_identifier':
                    result[_text(item)] = item
                elif item.type == 'spread_element':
                    result.update(object_entries(item.named_children[0], active))
        return result

    args = _field(call, 'arguments').named_children
    argv = [(literal(node) if literal(node) is not None else _text(node), literal(node) is not None)
            for node in args[1].named_children] if len(args) > 1 and args[1].type == 'array' else []
    options = object_entries(args[2]) if len(args) > 2 else {}
    env_node = options.get('env')
    entries = object_entries(env_node) if env_node is not None else {}
    environment = {key: (literal(node) if literal(node) is not None else _text(node), literal(node) is not None)
                   for key, node in entries.items()}
    if env_node is not None and env_node.type in {'identifier', 'shorthand_property_identifier'}:
        for assignment in assignments:
            left, right = _field(assignment, 'left'), _field(assignment, 'right')
            if left.type != 'member_expression' or _text(_field(left, 'object')) != _text(env_node):
                continue
            key = _text(_field(left, 'property'))
            ancestor = assignment.parent
            conditional = False
            while ancestor is not None and ancestor.type not in {'program', 'function_declaration', 'function_expression', 'arrow_function', 'method_definition'}:
                conditional |= ancestor.type in {'if_statement', 'for_statement', 'while_statement'}
                ancestor = ancestor.parent
            resolved = literal(right) is not None and not conditional
            environment[key] = (literal(right) if resolved else _text(right), resolved)
    executable = literal(args[0]) if args else None
    return executable, argv, environment


def recover_invocation_bindings(unit_paths, events, resources, staged_root, bridges):
    by_path={posixpath.normpath(value):key for key,value in unit_paths.items()}
    by_event={event.event_id:event for event in events}
    result=[]

    def target_path(value, source):
        choices=[posixpath.normpath(value),posixpath.normpath(posixpath.join(posixpath.dirname(source),value))]
        exact = next((path for path in choices if path in by_path),None)
        if exact is not None:
            return exact
        if value == posixpath.basename(value):
            matches = [path for path in by_path if posixpath.basename(path) == value]
            if len(matches) == 1:
                return matches[0]
        return None

    def add(unit, source, line, arguments, environment, resolved, env_resolved):
        if not arguments:return
        arguments=list(arguments)
        if Path(arguments[0]).name in _INTERPRETERS:
            arguments=arguments[1:]
            resolved=resolved[1:]
        if not arguments:return
        if not resolved or not resolved[0]:return
        target=target_path(arguments[0],source)
        matches=[event for event in events if event.unit_id==unit and event.line==line and event.capability in {'shell_exec','subprocess'}]
        if target is None or by_path[target]==unit or len(matches)!=1:return
        arguments[0] = target
        event=matches[0]
        candidates=[bridge for bridge in bridges if bridge.kind==BridgeKind.COMMAND
                    and bridge.source_event_id==event.event_id and bridge.target_unit_id==by_path[target]]
        same=next((bridge for bridge in candidates if bridge.cli_arguments==arguments
                   and bridge.cli_argument_resolved==resolved and bridge.environment==environment
                   and bridge.environment_resolved==env_resolved),None)
        if same is not None:
            result.append(same)
            return
        existing=next((bridge for bridge in candidates if not bridge.cli_arguments and not bridge.environment),None)
        bridge=Bridge(bridge_id=existing.bridge_id if existing else f'invoke-{len(bridges)}',
            source_unit_id=unit,target_unit_id=by_path[target],kind=BridgeKind.COMMAND,
            evidence=f'AST command operand {target} at {source}:{line}',source_event_id=event.event_id,
            cli_arguments=arguments,environment=environment,
            cli_argument_resolved=resolved,environment_resolved=env_resolved)
        if existing:bridges[bridges.index(existing)]=bridge
        else:bridges.append(bridge)
        result.append(bridge)

    for unit,relative in unit_paths.items():
        if not (staged_root/relative).is_file():continue
        source=(staged_root/relative).read_text(encoding='utf-8')
        if Path(relative).suffix=='.sh':
            for command in shell_commands(source, script_path=relative):
                names=list(command.environment)
                slots=len(command.arguments)
                for binding in product(*command.arguments,*(command.environment[name] for name in names)):
                    argv=binding[:slots]
                    environment=dict(zip(names,binding[slots:]))
                    add(unit,relative,command.line,[command.name,*[value for value,_ in argv]],
                        {name:value for name,(value,_) in environment.items()},
                        [True,*[known for _,known in argv]],
                        {name:known for name,(_,known) in environment.items()})
        elif Path(relative).suffix=='.py':
            try:invocations=_python_invocations(source, relative)
            except SyntaxError:continue
            for line, argv, env in invocations:
                add(unit,relative,line,[value for value,_ in argv],{key:value for key,(value,_) in env.items()},
                    [known for _,known in argv],{key:known for key,(_,known) in env.items()})
    # JS/TS API-graph bridges already identify actual execFile/spawn sites.
    from skillrecon.behavior.resource_fields import _walk, _field, _text, _js_literal
    from tree_sitter import Language, Parser
    import tree_sitter_javascript, tree_sitter_typescript
    for bridge in list(bridges):
        if bridge.kind!=BridgeKind.COMMAND or bridge in result or not bridge.source_event_id:continue
        event=by_event.get(bridge.source_event_id)
        if event is None or Path(event.file_path).suffix not in {'.js','.mjs','.cjs','.jsx','.ts','.tsx'}:continue
        suffix=Path(event.file_path).suffix
        grammar=(tree_sitter_typescript.language_tsx() if suffix=='.tsx' else
                 tree_sitter_typescript.language_typescript() if suffix=='.ts' else tree_sitter_javascript.language())
        tree=Parser(Language(grammar)).parse((staged_root/event.file_path).read_bytes())
        calls=[node for node in _walk(tree.root_node) if node.type=='call_expression' and node.start_point.row+1==event.line]
        for call in calls:
            args=_field(call,'arguments').named_children
            if len(args)<2 or args[1].type!='array':continue
            executable, argv, env = _javascript_bindings(call)
            # An interpreter argv includes its script operand; a direct execFile
            # argv starts with the first user argument and must be prepended.
            if executable is not None and Path(executable).name in _INTERPRETERS:
                argv=argv[1:]
            argv=[(unit_paths[bridge.target_unit_id],True),*argv]
            revised=bridge.model_copy(update={'cli_arguments':[value for value,_ in argv],
                'cli_argument_resolved':[known for _,known in argv],
                'environment':{key:value for key,(value,_) in env.items()},
                'environment_resolved':{key:known for key,(_,known) in env.items()}})
            bridges[bridges.index(bridge)]=revised
            result.append(revised)
            break
    return result


def propagate_invocation_resources(bridges, events, resources, unit_paths, staged_root):
    """Bind target argv/env operands; preserve alternatives from distinct callers."""
    replacements={}
    for bridge in bridges:
        if not bridge.cli_arguments and not bridge.environment:continue
        for resource in resources:
            if resource.unit_id!=bridge.target_unit_id or resource.resolved:continue
            expression=resource.value
            replacement=None
            known=False
            index=re.fullmatch(r'(?:sys\.argv|process\.argv)\[(\d+)\]|\$\{?(\d+)\}?',expression)
            if index:
                position=int(index.group(1) or index.group(2))
                if 'process.argv' in expression:position-=1
                if 0<=position<len(bridge.cli_arguments):
                    replacement=bridge.cli_arguments[position]
                    known=bool(bridge.cli_argument_resolved and bridge.cli_argument_resolved[position])
            env=re.fullmatch(r"""(?:os\.(?:getenv|environ\.get)\(\s*['"]([A-Za-z_]\w*)['"](?:\s*,[^)]*)?\)|os\.environ\[\s*['"]([A-Za-z_]\w*)['"]\s*\]|process\.env\.([A-Za-z_]\w*)|process\.env\[\s*['"]([A-Za-z_]\w*)['"]\s*\]|\$\{?([A-Za-z_]\w*)\}?)""",expression)
            if env:
                key=next(value for value in env.groups() if value is not None)
                replacement=bridge.environment.get(key)
                known=bridge.environment_resolved.get(key,False)
            if replacement is not None:
                replacements.setdefault(resource.resource_id,[]).append((replacement,bridge.bridge_id,known))
    revised=[]
    for resource in resources:
        options=replacements.get(resource.resource_id)
        if not options:
            revised.append(resource);continue
        for index,(value,bridge_id,resolved) in enumerate(dict.fromkeys(options)):
            revised.append(resource.model_copy(update={'resource_id':resource.resource_id if index==0 else f'{resource.resource_id}-caller{index}',
                'value':value,'resolved':resolved,'origin_kind':'bridge_cmd','origin_hint':bridge_id}))
    resources[:]=revised


def compose_artifact_paths(paths, bridges, events):
    """Join verified local payload flows through an identical resolved artifact."""
    verified=[path for path in paths if path.evidence_level in {'codeql','registered_source','bridge'}]
    result=[]
    for bridge in bridges:
        if bridge.kind!=BridgeKind.ARTIFACT:continue
        incoming=[path for path in verified if path.sink.event_id==bridge.source_event_id]
        outgoing=[path for path in verified if path.source.event_id==bridge.target_event_id]
        for left in incoming:
            for right in outgoing:
                result.append(RiskPath(path_id=f'artifact-flow-{len(result)}',source=left.source,sink=right.sink,
                    segments=[*left.segments,*right.segments],
                    bridges_used=[*left.bridges_used,bridge.bridge_id,*right.bridges_used],
                    evidence_level='bridge',path_kind='bridge:verified_artifact_payload'))
    return result
