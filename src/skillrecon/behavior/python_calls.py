"""Registered pathlib operations and source-bound requests forwarding calls."""

from __future__ import annotations

import ast
import posixpath
from collections import Counter
from pathlib import Path, PurePosixPath

import yaml

from skillrecon.core.types import CapabilityEvent, ResourceUse


def _imports(tree):
    names = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names[alias.asname or alias.name.split('.')[0]] = alias.name if alias.asname else alias.name.split('.')[0]
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names[alias.asname or alias.name] = ('.' * node.level + (node.module or ''), alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.pop(node.name, None)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.pop(target.id, None)
    return names


def _qualified(node, imports):
    if isinstance(node, ast.Name):
        name = imports.get(node.id)
        return '.'.join(name) if isinstance(name, tuple) else name
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value, imports)
        return f'{base}.{node.attr}' if base else None
    return None


def _scope_nodes(scope):
    def walk(node):
        if node is not scope and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        yield node
        for child in ast.iter_child_nodes(node):
            yield from walk(child)
    return sorted(walk(scope), key=lambda n: (getattr(n, 'lineno', 0), getattr(n, 'col_offset', 0)))


def _constant(node, values):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.Subscript):
        container, key = _constant(node.value, values), _constant(node.slice, values)
        if isinstance(container, dict) and isinstance(key, (str, int)):
            return container.get(key)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _constant(node.left, values), _constant(node.right, values)
        return left + right if isinstance(left, str) and isinstance(right, str) else None
    if isinstance(node, ast.JoinedStr):
        parts = [_constant(n.value if isinstance(n, ast.FormattedValue) else n, values) for n in node.values]
        return ''.join(parts) if all(isinstance(p, str) for p in parts) else None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _constant(node.operand, values)
        return not value if isinstance(value, bool) else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'startswith' and len(node.args) == 1:
        value, prefix = _constant(node.func.value, values), _constant(node.args[0], values)
        return value.startswith(prefix) if isinstance(value, str) and isinstance(prefix, str) else None
    return None


def _string_return(statements, values):
    for statement in statements:
        if isinstance(statement, ast.Return):
            return True, _constant(statement.value, values)
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            values[statement.targets[0].id] = _constant(statement.value, values)
        elif isinstance(statement, ast.If):
            condition = _constant(statement.test, values)
            if not isinstance(condition, bool):
                return True, None
            returned, value = _string_return(statement.body if condition else statement.orelse, values)
            if returned:
                return True, value
        elif not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Constant):
            return True, None
    return False, None


def recover_python_calls(events, resources, unit_paths, staged_root: Path):
    """Recover located writes and actual local HTTP forwarding invocations."""
    modules = {}
    for unit, relative in unit_paths.items():
        if Path(relative).suffix != '.py':
            continue
        try:
            tree = ast.parse((staged_root / relative).read_text(encoding='utf-8'))
        except SyntaxError:
            continue  # Unsupported source forms remain outside this AST model.
        constants = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = _constant(statement.value, constants)
        modules[relative] = {'unit': unit, 'tree': tree, 'imports': _imports(tree), 'constants': constants,
                             'functions': {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}}

    def local_function(call, relative):
        module = modules[relative]
        if not isinstance(call.func, ast.Name):
            return None
        target = module['imports'].get(call.func.id)
        if isinstance(target, tuple):
            imported, function = target
            level = len(imported) - len(imported.lstrip('.'))
            parent = PurePosixPath(relative).parent
            for _ in range(max(0, level - 1)):
                parent = parent.parent
            suffix = imported.lstrip('.').replace('.', '/') + '.py'
            candidates = {str(parent/suffix)} if level else {str(parent/suffix), suffix}
            candidates &= modules.keys()
            if len(candidates) != 1:
                return None
            relative = next(iter(candidates))
        else:
            function = call.func.id
        definition = modules[relative]['functions'].get(function)
        return (relative, definition) if definition else None

    def url_value(expression, relative, values):
        if not isinstance(expression, ast.Call):
            value = _constant(expression, values)
            return value if isinstance(value, str) else None
        target = local_function(expression, relative)
        if target is None:
            return None
        source, function = target
        bound = modules[source]['constants'].copy()
        for parameter, argument in zip(function.args.args, expression.args):
            bound[parameter.arg] = _constant(argument, values)
        _, value = _string_return(function.body, bound)
        return value if isinstance(value, str) else None

    wrappers = {}
    for relative, module in modules.items():
        for function in module['functions'].values():
            parameters = [a.arg for a in function.args.args]
            models = []
            for call in [n for n in _scope_nodes(function) if isinstance(n, ast.Call)]:
                api = _qualified(call.func, module['imports'])
                if api == 'requests.request' and len(call.args) >= 2:
                    method, url = call.args[:2]
                elif api in {f'requests.{verb}' for verb in ['get', 'post', 'put', 'patch', 'delete', 'head', 'options']} and call.args:
                    method, url = ast.Constant(api.split('.')[-1].upper()), call.args[0]
                else:
                    continue
                receiver = call.func.value if isinstance(call.func, ast.Attribute) else call.func
                if isinstance(receiver, ast.Name) and receiver.id in parameters:
                    continue
                if not isinstance(url, ast.Name) or url.id not in parameters:
                    continue
                method_binding = ('parameter', parameters.index(method.id)) if isinstance(method, ast.Name) and method.id in parameters else ('literal', _constant(method, module['constants']))
                models.append((method_binding, parameters.index(url.id), call.lineno))
            if models and len({(m, u) for m, u, _ in models}) == 1:
                wrappers[relative, function.name] = (models[0][0], models[0][1], [line for _, _, line in models])

    changes = []
    for relative, module in modules.items():
        unit = module['unit']
        scopes = [module['tree']] + [n for n in ast.walk(module['tree']) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for scope in scopes:
            imports = module['imports'].copy()
            paths = module.get('paths', {}).copy() if scope is not module['tree'] else {}
            values = module['constants'].copy()
            config_sources = []
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for argument in scope.args.args:
                    paths.pop(argument.arg, None)
                    if _qualified(argument.annotation, imports) == 'pathlib.Path':
                        paths[argument.arg] = (argument.arg, False)
                for argument in scope.args.args:
                    imports.pop(argument.arg, None)

            def path_value(expression):
                if isinstance(expression, ast.Name):
                    return paths.get(expression.id)
                if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
                    base, part = path_value(expression.left), _constant(expression.right, values)
                    if base:
                        return ((PurePosixPath(base[0])/part).as_posix(), True) if base[1] and isinstance(part, str) else (ast.unparse(expression), False)
                if isinstance(expression, ast.Call):
                    if _qualified(expression.func, imports) == 'pathlib.Path' and expression.args:
                        value = _constant(expression.args[0], values)
                        return (value, True) if isinstance(value, str) else (ast.unparse(expression.args[0]), False)
                    if isinstance(expression.func, ast.Attribute) and expression.func.attr in {'expanduser', 'resolve', 'absolute', 'with_name', 'with_suffix', 'joinpath'} and path_value(expression.func.value):
                        base = path_value(expression.func.value)
                        if expression.func.attr == 'resolve' and base[1] and PurePosixPath(base[0]).is_absolute():
                            return posixpath.normpath(base[0]), True
                        return ast.unparse(expression), False
                    if isinstance(expression.func, ast.Name) and expression.func.id in module['functions'] and _qualified(module['functions'][expression.func.id].returns, imports) == 'pathlib.Path':
                        return ast.unparse(expression), False
                if isinstance(expression, ast.Attribute) and expression.attr == 'parent' and path_value(expression.value):
                    return ast.unparse(expression), False
                return None

            def source_value(expression):
                if isinstance(expression, ast.Call) and _qualified(expression.func, imports) == 'yaml.safe_load' and len(expression.args) == 1:
                    read = expression.args[0]
                    if isinstance(read, ast.Call) and isinstance(read.func, ast.Attribute) and read.func.attr == 'read_text':
                        source = path_value(read.func.value)
                        if source and source[1] and not PurePosixPath(source[0]).is_absolute():
                            filename = (staged_root/source[0]).resolve()
                            if filename.is_relative_to(staged_root.resolve()) and filename.is_file():
                                try:
                                    loaded = yaml.safe_load(filename.read_text(encoding='utf-8'))
                                except yaml.YAMLError:
                                    return None
                                config_sources.append(str(filename.relative_to(staged_root.resolve())))
                                return loaded
                    return None
                return url_value(expression, relative, values)

            nodes = _scope_nodes(scope)
            definitions = Counter(t.id for n in nodes if isinstance(n, ast.Assign)
                                  for t in n.targets if isinstance(t, ast.Name))
            shadowed = {a.arg for a in scope.args.args} if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else set()
            for node in nodes:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            imports.pop(target.id, None)
                            shadowed.add(target.id)
                            value = path_value(node.value)
                            if value:
                                paths[target.id] = value if definitions[target.id] == 1 else (target.id, False)
                            else:
                                paths.pop(target.id, None)
                            values[target.id] = source_value(node.value) if definitions[target.id] == 1 else None
                if not isinstance(node, ast.Call):
                    continue
                location = f'{relative}:{node.lineno}'
                receiver = path_value(node.func.value) if isinstance(node.func, ast.Attribute) else None
                if receiver and node.func.attr in {'write_text', 'write_bytes', 'mkdir'}:
                    existing = [e for e in events if e.unit_id == unit and e.line == node.lineno and e.capability == 'file_write']
                    if existing and any(r.event_id == existing[0].event_id and r.resource_type == 'path'
                                        and (r.value, r.resolved) == receiver for r in resources):
                        continue
                    if not existing:
                        other_reads = any(isinstance(n, ast.Call) and n.lineno == node.lineno and
                                          ((isinstance(n.func, ast.Attribute) and n.func.attr in {'read_text', 'read_bytes', 'open'}) or
                                           isinstance(n.func, ast.Name) and n.func.id == 'open') for n in nodes)
                        if not other_reads:
                            existing = [e for e in events if e.unit_id == unit and e.line == node.lineno and e.capability == 'file_read']
                    event = existing[0] if existing else CapabilityEvent(event_id=f'e{len(events)}', unit_id=unit,
                            capability='file_write', location=location, language='python', file_path=relative, line=node.lineno)
                    previous = event.capability if existing else None
                    event = event.model_copy(update={'capability': 'file_write', 'api_call': ast.unparse(node.func),
                            'tier': 'registered_source', 'detail': f'pathlib.{node.func.attr}; imported/annotated Path receiver at {location}; config sources={config_sources}',
                            'arguments': [receiver[0]]})
                    resources[:] = [r for r in resources if r.event_id != event.event_id]
                    resources.append(ResourceUse(resource_id=f'pathlib-{event.event_id}', unit_id=unit, event_id=event.event_id,
                            resource_type='path', value=receiver[0], resolved=receiver[1], location=location,
                            origin_kind='code_literal' if receiver[1] else 'unknown', origin_hint=event.detail))
                    if not existing:
                        events.append(event)
                    else:
                        events[events.index(existing[0])] = event
                    changes.append({'unit_id': unit, 'location': location, 'event_id': event.event_id,
                                    'model': f'pathlib.{node.func.attr}', 'previous_capability': previous})
                    continue
                if isinstance(node.func, ast.Name) and node.func.id in shadowed:
                    continue
                target = local_function(node, relative)
                if target is None or (target[0], target[1].name) not in wrappers:
                    continue
                source, function = target
                method, url_index, lines = wrappers[source, function.name]
                if len(node.args) <= url_index or any(e.unit_id == unit and e.line == node.lineno and e.capability == 'http_request' for e in events):
                    continue
                url = url_value(node.args[url_index], relative, values)
                verb = _constant(node.args[method[1]], values) if method[0] == 'parameter' and len(node.args) > method[1] else method[1] if method[0] == 'literal' else None
                event = CapabilityEvent(event_id=f'e{len(events)}', unit_id=unit, capability='http_request',
                        api_call=ast.unparse(node.func), location=location, arguments=[url or ast.unparse(node.args[url_index])],
                        tier='registered_source', language='python', file_path=relative, line=node.lineno,
                        detail=f'requests forwarding wrapper; method={verb}; definition={source}:{function.lineno}; transport={source}:{lines}')
                events.append(event)
                resources.append(ResourceUse(resource_id=f'wrapper-{event.event_id}', unit_id=unit, event_id=event.event_id,
                        resource_type='url', value=url or '<dynamic-url>', resolved=url is not None, location=location,
                        origin_kind='code_literal' if url else 'unknown', origin_hint=event.detail))
                changes.append({'unit_id': unit, 'location': location, 'event_id': event.event_id,
                                'model': 'requests.wrapper', 'definition': f'{source}:{function.lineno}', 'transport_lines': lines})
            if scope is module['tree']:
                module['paths'] = paths.copy()
    return changes
