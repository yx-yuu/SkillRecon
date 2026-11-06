"""Source-bound request fields, runtime messages, and static URL expressions."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_javascript
import tree_sitter_typescript

from skillrecon.behavior.python_calls import _constant, _imports, _qualified, _scope_nodes
from skillrecon.behavior.resource_extract import shell_variable_paths
from skillrecon.core.types import ResourceUse

_AUTH_HEADERS = {'authorization', 'proxy-authorization', 'x-api-key', 'api-key', 'apikey', 'x-auth-token', 'cookie'}
_AUTH_FIELDS = {'api_key', 'apikey', 'password', 'passwd', 'private_key', 'secret_key', 'access_token', 'auth_token', 'refresh_token', 'seed_phrase'}


def _walk(node):
    yield node
    for child in node.named_children:
        yield from _walk(child)


def _text(node):
    return node.text.decode('utf-8')


def _field(node, name):
    return node.child_by_field_name(name)


def _owner(node):
    while node.parent is not None and node.type not in {'class_declaration', 'program'}:
        node = node.parent
    return node.start_byte


def _js_literal(node, constants):
    if node.type == 'string' and all(n.type == 'string_fragment' for n in node.named_children):
        return _text(node)[1:-1]
    if node.type == 'number' and _text(node).isdecimal():
        return _text(node)
    if node.type == 'identifier':
        return constants.get(_text(node))
    if node.type == 'parenthesized_expression':
        return _js_literal(node.named_children[0], constants)
    if node.type == 'binary_expression' and _text(_field(node, 'operator')) == '+':
        left = _js_literal(_field(node, 'left'), constants)
        right = _js_literal(_field(node, 'right'), constants)
        return left + right if left is not None and right is not None else None
    if node.type == 'template_string':
        parts = []
        for child in node.named_children:
            if child.type == 'string_fragment':
                parts.append(_text(child))
            elif child.type == 'template_substitution':
                parts.append(_js_literal(child.named_children[0], constants))
            else:
                return None
        return ''.join(parts) if all(p is not None for p in parts) else None
    return None


def _visible_constants(node, constants):
    """Exclude lexical bindings that shadow a module constant at this use."""
    visible = constants.copy()
    scope = node.parent
    while scope is not None and scope.type != 'program':
        if scope.type in {'statement_block', 'function_declaration', 'arrow_function', 'method_definition'}:
            def local_nodes(current):
                yield current
                for child in current.named_children:
                    if child.type not in {'statement_block', 'function_declaration', 'arrow_function', 'method_definition', 'class_declaration'}:
                        yield from local_nodes(child)
            for binding in local_nodes(scope):
                name = None
                if binding.type == 'variable_declarator':
                    name = _field(binding, 'name')
                elif binding.type in {'required_parameter', 'optional_parameter'}:
                    name = _field(binding, 'pattern')
                elif binding.type == 'formal_parameters':
                    for child in binding.named_children:
                        if child.type == 'identifier':
                            visible.pop(_text(child), None)
                if name is not None:
                    visible.pop(_text(name), None)
        scope = scope.parent
    return visible


def _js_url_prefix(node, constants):
    value = _js_literal(node, constants)
    if value is not None:
        return value
    if node.type == 'template_string':
        parts = []
        for child in node.named_children:
            value = (_text(child) if child.type == 'string_fragment' else
                     _js_literal(child.named_children[0], constants)
                     if child.type == 'template_substitution' else None)
            if value is None:
                break
            parts.append(value)
        return ''.join(parts)
    if node.type == 'binary_expression' and _text(_field(node, 'operator')) == '+':
        left = _field(node, 'left')
        value = _js_literal(left, constants)
        return value + _js_url_prefix(_field(node, 'right'), constants) if value is not None else _js_url_prefix(left, constants)
    return ''


def _js_alternatives(node, bindings, visiting=()):
    """Resolve immutable expressions, retaining short-circuit branch premises."""
    if node.type == 'identifier' and _text(node) in bindings:
        name = _text(node)
        if name not in visiting:
            return _js_alternatives(bindings[name], bindings, (*visiting, name))
    if node.type == 'parenthesized_expression':
        return _js_alternatives(node.named_children[0], bindings, visiting)
    literal = _js_literal(node, {})
    if literal is not None:
        return [(literal, True, '')]
    if node.type == 'binary_expression':
        left, right = _field(node, 'left'), _field(node, 'right')
        operator = _text(_field(node, 'operator'))
        lhs = _js_alternatives(left, bindings, visiting)
        rhs = _js_alternatives(right, bindings, visiting)
        if operator == '+':
            return [(a + b, ka and kb, '; '.join(filter(None, (ca, cb))))
                    for a, ka, ca in lhs for b, kb, cb in rhs]
        if operator in {'||', '??'}:
            alternatives = []
            for value, known, context in lhs:
                if known:
                    alternatives.extend([(value, True, context)] if value or operator == '??' else rhs)
                    continue
                test = _text(left)
                alternatives.append((value, False, f'{test} is ' + ('truthy' if operator == '||' else 'non-nullish')))
                alternatives.extend((value, known, '; '.join(filter(None, (
                    f'{test} is ' + ('falsy (including empty string)' if operator == '||' else 'nullish'), context))))
                    for value, known, context in rhs)
            return alternatives
    return [(_text(node), False, '')]


def _python_url_prefix(node, constants):
    value = _constant(node, constants)
    if isinstance(value, str):
        return value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for part in node.values:
            value = _constant(part.value if isinstance(part, ast.FormattedValue) else part, constants)
            if not isinstance(value, str):
                break
            parts.append(value)
        return ''.join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _constant(node.left, constants)
        return left + _python_url_prefix(node.right, constants) if isinstance(left, str) else _python_url_prefix(node.left, constants)
    return ''


def recover_resource_fields(events, resources, unit_paths, staged_root: Path):
    """Enrich resources without creating events or claiming new payload flow."""
    changes = []

    def add(unit, relative, line, kind, value, resolved, origin, hint, event_id=None):
        location = f'{relative}:{line}'
        if any((r.unit_id, r.event_id, r.resource_type, r.value, r.location) ==
               (unit, event_id, kind, value, location) for r in resources):
            return
        resource = ResourceUse(resource_id=f'field-{unit}-{line}-{kind}-{len(changes)}',
            unit_id=unit, event_id=event_id, resource_type=kind, value=value, resolved=resolved,
            location=location, origin_kind=origin, origin_hint=hint)
        resources.append(resource)
        changes.append({'unit_id': unit, 'location': location, 'event_id': event_id,
                        'resource_id': resource.resource_id, 'model': kind, 'value': value,
                        'resolved': resolved, 'origin_hint': hint})

    def url_resource(unit, relative, event, value, prefix, hint):
        if value is not None and value.startswith(('http://', 'https://')):
            old = [r for r in resources if r.event_id == event.event_id and r.resource_type == 'url']
            if len(old) == 1 and (old[0].value, old[0].resolved) == (value, True):
                return
            resources[:] = [r for r in resources if r not in old]
            add(unit, relative, event.line, 'url', value, True, 'code_literal', hint, event.event_id)
        else:
            # A dynamic path/query does not erase a syntactically fixed host.
            # Require the authority terminator: "https://api." + runtime is unknown.
            host = re.match(r'^https?://([^/?#\s]+)[/?#]', prefix)
            if host and '@' not in host[1]:
                add(unit, relative, event.line, 'domain', host[1], True,
                    'source_binding', hint + '; fixed authority, dynamic path/query', event.event_id)

    for unit, relative in unit_paths.items():
        source = (staged_root / relative).read_text(encoding='utf-8')
        native = [e for e in events if e.unit_id == unit and e.tier != 'instruction']
        suffix = Path(relative).suffix
        if suffix == '.sh':
            lines = source.splitlines()
            for event in native:
                if event.capability in {'file_read', 'file_write', 'file_delete', 'file_permission_change', 'token_file_read'}:
                    for value, resolved in shell_variable_paths(lines[event.line-1]):
                        add(unit, relative, event.line, 'path', value, resolved,
                            'code_literal' if resolved else 'unknown', 'shell_variable_path', event.event_id)
        elif suffix == '.py':
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue  # Source forms outside Python's AST model stay unmodified.
            module_imports = _imports(tree)
            module_values = {}
            for statement in tree.body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name):
                            module_values[target.id] = _constant(statement.value, module_values)
            scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for scope in scopes:
                nodes = _scope_nodes(scope)
                imports = module_imports.copy()
                values = module_values.copy()
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for argument in scope.args.args:
                        imports.pop(argument.arg, None)
                        values.pop(argument.arg, None)
                definitions = Counter(t.id for n in nodes if isinstance(n, ast.Assign)
                                      for t in n.targets if isinstance(t, ast.Name))
                dictionaries = {}
                for node in nodes:
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                imports.pop(target.id, None)
                                dictionaries.pop(target.id, None)
                                values[target.id] = _constant(node.value, values) if definitions[target.id] == 1 else None
                                if definitions[target.id] == 1 and isinstance(node.value, ast.Dict):
                                    dictionaries[target.id] = node.value
                            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                                dictionaries.pop(target.value.id, None)
                    if not isinstance(node, ast.Call):
                        continue
                    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.attr == 'update':
                        dictionaries.pop(node.func.value.id, None)
                    api = _qualified(node.func, imports)
                    if api not in {f'requests.{v}' for v in ['request', 'get', 'post', 'put', 'patch', 'delete', 'head', 'options']}:
                        continue
                    matched = [e for e in native if e.line == node.lineno and e.capability == 'http_request']
                    if len(matched) != 1:
                        continue
                    method = (_constant(node.args[0], values) if node.args else None) if api == 'requests.request' else api.rsplit('.', 1)[-1].upper()
                    if isinstance(method, str):
                        add(unit, relative, node.lineno, 'http_method', method.upper(), True,
                            'code_literal', f'Bound requests method at {relative}:{node.lineno}', matched[0].event_id)
                    index = 1 if api == 'requests.request' else 0
                    argument = node.args[index] if len(node.args) > index else next(
                        (k.value for k in node.keywords if k.arg == 'url'), None)
                    if argument is not None:
                        value = _constant(argument, values)
                        url_resource(unit, relative, matched[0], value if isinstance(value, str) else None,
                            _python_url_prefix(argument, values),
                            f'Bound requests URL argument {ast.unparse(argument)} at {relative}:{node.lineno}')
                    for keyword in node.keywords:
                        if keyword.arg not in {'params', 'data', 'json', 'headers'}:
                            continue
                        payload = dictionaries.get(keyword.value.id) if isinstance(keyword.value, ast.Name) else keyword.value
                        if not isinstance(payload, ast.Dict):
                            continue
                        label = ast.unparse(keyword.value) if isinstance(keyword.value, ast.Name) else keyword.arg
                        for key, value in zip(payload.keys, payload.values):
                            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                credential = (key.value.lower() in _AUTH_HEADERS if keyword.arg == 'headers'
                                              else key.value.lower() in _AUTH_FIELDS)
                                if credential and not (isinstance(value, ast.Constant) and not value.value):
                                    add(unit, relative, value.lineno, 'request_credential',
                                        f'{keyword.arg}:{key.value}', True, 'source_binding',
                                        f'Credential slot bound to requests at {relative}:{node.lineno}; value not evaluated', matched[0].event_id)
                                if keyword.arg == 'headers':
                                    continue
                                descriptor = f'{label}[{key.value!r}] = {ast.unparse(value)}'
                                add(unit, relative, value.lineno, 'http_parameter', descriptor, True,
                                    'source_binding', f'Known parameter key/expression binding; payload value not evaluated; {keyword.arg} at {relative}:{node.lineno}', matched[0].event_id)
        elif suffix in {'.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx'}:
            language = (tree_sitter_typescript.language_tsx() if suffix == '.tsx' else
                        tree_sitter_typescript.language_typescript() if suffix == '.ts' else
                        tree_sitter_javascript.language())
            tree = Parser(Language(language)).parse(source.encode())
            nodes = list(_walk(tree.root_node))
            constants, expressions, axios_names = {}, {}, set()
            for statement in tree.root_node.named_children:
                if statement.type == 'import_statement':
                    package = _field(statement, 'source')
                    if package is not None and _text(package)[1:-1] == 'axios':
                        clause = next(n for n in statement.named_children if n.type == 'import_clause')
                        axios_names.update(_text(n) for n in clause.named_children if n.type == 'identifier')
                if statement.type != 'lexical_declaration' or not _text(statement).startswith('const '):
                    continue
                for declaration in statement.named_children:
                    if declaration.type != 'variable_declarator':
                        continue
                    name, value = _field(declaration, 'name'), _field(declaration, 'value')
                    if value is None or name.type != 'identifier':
                        continue
                    expressions[_text(name)] = value
                    if _text(value) in {"require('axios')", 'require("axios")'}:
                        axios_names.add(_text(name))
                    literal = _js_literal(value, constants)
                    if literal is not None:
                        constants[_text(name)] = literal
            for node in nodes:
                if node.type in {'assignment_expression', 'augmented_assignment_expression', 'update_expression'}:
                    left = _field(node, 'left') or _field(node, 'argument')
                    if left is not None:
                        constants.pop(_text(left), None)
                        expressions.pop(_text(left), None)
                        axios_names.discard(_text(left))
            factories = set()
            for node in nodes:
                if node.type != 'assignment_expression':
                    continue
                left, right = _field(node, 'left'), _field(node, 'right')
                if right.type == 'call_expression' and _text(_field(right, 'function')) in {f'{name}.create' for name in axios_names}:
                    factories.add((_owner(node), _text(left)))
            calls = [n for n in nodes if n.type == 'call_expression']
            for call in calls:
                function = _field(call, 'function')
                arguments = _field(call, 'arguments').named_children
                matched = [e for e in native if e.line == call.start_point.row+1 and e.capability == 'http_request']
                if matched and arguments and _text(function) == 'fetch':
                    same_line = [n for n in calls if n.start_point.row == call.start_point.row and _text(_field(n, 'function')) == 'fetch']
                    value = _js_literal(arguments[0], _visible_constants(call, constants))
                    if len(same_line) == 1:
                        for event in matched:
                            alternatives = _js_alternatives(arguments[0], _visible_constants(call, expressions))
                            old = [r for r in resources if r.event_id == event.event_id and r.resource_type == 'url']
                            if len(alternatives) > 1 and not any(r.resolved for r in old):
                                # Refine the unknown operand of this same event;
                                # preserve both the configured and default branch.
                                resources[:] = [r for r in resources if r not in old]
                                for target, resolved, context in alternatives:
                                    resource = ResourceUse(
                                        resource_id=f'field-{unit}-{event.line}-url-{len(changes)}',
                                        unit_id=unit, event_id=event.event_id, resource_type='url',
                                        value=target, resolved=resolved, location=event.location,
                                        origin_kind='source_binding', context=context,
                                        origin_hint=f'Immutable fetch operand {_text(arguments[0])}: {context}')
                                    resources.append(resource)
                                    changes.append({'resource_id':resource.resource_id,'model':'js_short_circuit_operand',
                                                    'value':target,'resolved':resolved,'context':context})
                            else:
                                url_resource(unit, relative, event, value,
                                    _js_url_prefix(arguments[0], _visible_constants(call, constants)),
                                    f'Static fetch argument {_text(arguments[0])}; module const bindings')
                        if len(arguments) > 1 and arguments[1].type == 'object':
                            for option in arguments[1].named_children:
                                if option.type != 'pair' or _text(_field(option, 'key')).strip('"\'') != 'headers':
                                    continue
                                header_object = _field(option, 'value')
                                if header_object.type != 'object':
                                    continue
                                for header in header_object.named_children:
                                    if header.type != 'pair':
                                        continue
                                    key = _text(_field(header, 'key')).strip('"\'')
                                    value = _field(header, 'value')
                                    if key.lower() in _AUTH_HEADERS and _text(value) not in {'null', 'undefined', "''", '""'}:
                                        for event in matched:
                                            add(unit, relative, header.start_point.row+1, 'request_credential',
                                                f'headers:{key}', True, 'source_binding',
                                                f'Credential slot bound to fetch at {relative}:{event.line}; value not evaluated', event.event_id)
                tail = '.interceptors.request.use'
                if not _text(function).endswith(tail) or not arguments:
                    continue
                receiver = _text(function)[:-len(tail)]
                if (_owner(call), receiver) not in factories:
                    continue
                callback = arguments[0]
                if callback.type not in {'arrow_function', 'function_expression'}:
                    continue
                parameters = _field(callback, 'parameters')
                if parameters is None or len(parameters.named_children) != 1:
                    continue
                parameter = _text(parameters.named_children[0])
                for assignment in _walk(callback):
                    if assignment.type != 'assignment_expression':
                        continue
                    left, right = _field(assignment, 'left'), _field(assignment, 'right')
                    if _text(left) != f'{parameter}.headers.Authorization':
                        continue
                    value = _js_literal(right, _visible_constants(assignment, constants))
                    descriptor = value if value is not None else _text(right).strip('`')
                    conditions, parent = [], assignment.parent
                    while parent is not None and parent != callback:
                        if parent.type == 'if_statement':
                            conditions.append(_text(_field(parent, 'condition')))
                        parent = parent.parent
                    add(unit, relative, assignment.start_point.row+1, 'authorization_header',
                        f'Authorization: {descriptor}', value is not None, 'request_interceptor',
                        f'axios.create receiver {receiver}; request interceptor; conditions={conditions}; runtime template preserved')
            for node in nodes:
                if node.type != 'pair':
                    continue
                value = _field(node, 'value')
                if value.type != 'member_expression' or _text(_field(value, 'property')) != 'message':
                    continue
                base = _field(value, 'object')
                if base.type == 'member_expression' and _text(_field(base, 'property')) == 'error':
                    add(unit, relative, value.start_point.row+1, 'error_message', _text(value), False,
                        'runtime_field', f'Object field {_text(_field(node, "key"))}; runtime message expression')
    return changes
