"""Located local import and source-bound script invocation bridges."""

from pathlib import Path, PurePosixPath
import posixpath
import re

from tree_sitter import Language, Parser
import tree_sitter_bash
import tree_sitter_javascript
import tree_sitter_typescript

from skillrecon.behavior.resource_fields import _field, _text, _walk, _visible_constants, _js_literal
from skillrecon.core.enums import BridgeKind
from skillrecon.core.types import Bridge, CapabilityEvent, ResourceUse


def resolve_js_import(source_path, imported, unit_by_path):
    base = posixpath.normpath(posixpath.join(PurePosixPath(source_path).parent.as_posix(), imported))
    extensions=['.js','.ts','.mjs','.cjs','.jsx','.tsx']
    candidates = [base,*[base+suffix for suffix in extensions],*[base+'/index'+suffix for suffix in extensions]]
    if Path(source_path).suffix in {'.ts', '.tsx'} and base.endswith('.js'):
        candidates = [base[:-3]+'.ts', base[:-3]+'.tsx', base]
    return next((p for p in candidates if p in unit_by_path), None)


def _qualified(node, bindings):
    if node.type == 'identifier':
        return bindings.get(_text(node))
    if node.type == 'member_expression':
        base = _qualified(_field(node, 'object'), bindings)
        return f'{base}.{_text(_field(node, "property"))}' if base else None
    return None


def _context(node):
    contexts = []
    child, parent = node, node.parent
    while parent is not None:
        if parent.type == 'if_statement':
            condition = _field(parent, 'condition')
            if condition is not None:
                contexts.append(f'{_text(condition)} [{child.type}]')
        child, parent = parent, parent.parent
    return '; '.join(contexts)


def _bind_import(pattern, package, bindings):
    if pattern.type == 'identifier':
        bindings[_text(pattern)] = package
    elif pattern.type == 'object_pattern':
        for child in pattern.named_children:
            if child.type == 'shorthand_property_identifier_pattern':
                bindings[_text(child)] = f'{package}.{_text(child)}'
            elif child.type == 'pair_pattern':
                bindings[_text(_field(child, 'value'))] = f'{package}.{_text(_field(child, "key"))}'


def _js_path(node, values, bindings, source_path):
    if node.type == 'identifier' and _text(node) == '__dirname':
        return PurePosixPath(source_path).parent.as_posix()
    if node.type == 'call_expression' and _qualified(_field(node, 'function'), bindings) == 'node:path.join':
        parts = [_js_path(n, values, bindings, source_path) for n in _field(node, 'arguments').named_children]
        return posixpath.normpath(posixpath.join(*parts)) if parts and all(p is not None for p in parts) else None
    return _js_literal(node, values)


def _shell_value(node, values, source_path):
    if node.type in {'word', 'string_content'}:
        return _text(node)
    if node.type == 'raw_string':
        return _text(node)[1:-1]
    if node.type in {'simple_expansion', 'expansion'}:
        names = [n for n in node.named_children if n.type == 'variable_name']
        return values.get(_text(names[0])) if len(names) == 1 else None
    if node.type == 'command_substitution' and re.fullmatch(
        r'\$\(cd "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)" && pwd(?: -P)?\)', _text(node)
    ):
        return PurePosixPath(source_path).parent.as_posix()
    if node.type in {'string', 'concatenation', 'command_name'}:
        parts = [_shell_value(n, values, source_path) for n in node.named_children]
        return ''.join(parts) if parts and all(p is not None for p in parts) else None
    return None


def recover_source_bridges(unit_paths, events, resources, staged_root):
    """Recover actual local references; only bound Shell calls add events."""
    unit_by_path = {posixpath.normpath(p): u for u, p in unit_paths.items()}
    bridges = []

    def bridge(unit, relative, target, node, kind, detail, event_id=None):
        target = posixpath.normpath(target)
        target_unit = unit_by_path.get(target)
        if target_unit is None or target_unit == unit:
            return
        location = f'{relative}:{node.start_point.row+1}'
        context = _context(node)
        evidence = f'source-bound {kind.value} {target} at {location}; {detail}'
        if context:
            evidence += f'; branch context: {context}'
        bridges.append(Bridge(bridge_id=f'source-{unit}-{node.start_byte}-{target_unit}',
            source_unit_id=unit, target_unit_id=target_unit, kind=kind,
            evidence=evidence, source_event_id=event_id))

    for unit, relative in unit_paths.items():
        path = staged_root/relative
        if not path.is_file():
            continue  # Artifact-only unit inventories need no source parse.
        suffix = path.suffix
        if suffix in {'.js', '.mjs', '.cjs', '.ts', '.tsx'}:
            language = (tree_sitter_typescript.language_tsx() if suffix == '.tsx' else
                        tree_sitter_typescript.language_typescript() if suffix == '.ts' else
                        tree_sitter_javascript.language())
            tree = Parser(Language(language)).parse(path.read_bytes())
            nodes = list(_walk(tree.root_node))
            bindings, values = {}, {}
            for statement in tree.root_node.named_children:
                if statement.type == 'import_statement':
                    package = _field(statement, 'source')
                    if package is None:
                        continue
                    package = _text(package)[1:-1]
                    package = package if package.startswith('node:') else 'node:'+package
                    for spec in _walk(statement):
                        if spec.type == 'import_specifier':
                            name, alias = _field(spec, 'name'), _field(spec, 'alias')
                            bindings[_text(alias or name)] = f'{package}.{_text(name)}'
                        elif spec.type == 'import_clause':
                            for child in spec.named_children:
                                if child.type == 'identifier':
                                    bindings[_text(child)] = package
                if statement.type != 'lexical_declaration' or not _text(statement).startswith('const '):
                    continue
                for declaration in statement.named_children:
                    if declaration.type != 'variable_declarator':
                        continue
                    name, value = _field(declaration, 'name'), _field(declaration, 'value')
                    if value is None:
                        continue
                    if value.type == 'call_expression':
                        function = _field(value, 'function')
                        arguments = _field(value, 'arguments').named_children
                        if _text(function) == 'require' and len(arguments) == 1:
                            package = _js_literal(arguments[0], {})
                            if package is not None:
                                package = package if package.startswith('node:') else 'node:'+package
                                _bind_import(name, package, bindings)
                        elif _qualified(function, bindings) == 'node:util.promisify' and len(arguments) == 1 and name.type == 'identifier':
                            api = _qualified(arguments[0], bindings)
                            if api in {'node:child_process.execFile', 'node:child_process.exec'}:
                                bindings[_text(name)] = api
                    if name.type == 'identifier':
                        literal = _js_path(value, values, bindings, relative)
                        if literal is not None:
                            values[_text(name)] = literal
            for node in nodes:
                if node.type in {'assignment_expression', 'augmented_assignment_expression', 'update_expression'}:
                    left = _field(node, 'left') or _field(node, 'argument')
                    if left is not None:
                        bindings.pop(_text(left), None)
                        values.pop(_text(left), None)
            for node in nodes:
                if node.type == 'import_statement':
                    literal = _field(node, 'source')
                elif node.type == 'call_expression' and _text(_field(node, 'function')) in {'require', 'import'}:
                    arguments = _field(node, 'arguments').named_children
                    literal = arguments[0] if len(arguments) == 1 else None
                    if _text(_field(node, 'function')) == 'require' and 'require' not in _visible_constants(node, {'require': True}):
                        continue
                else:
                    literal = None
                imported = _js_literal(literal, {}) if literal is not None else None
                if imported is not None and imported.startswith('.'):
                    target = resolve_js_import(relative, imported, unit_by_path)
                    if target is not None:
                        bridge(unit, relative, target, node, BridgeKind.STATIC, f'literal local import {imported}')
                if node.type != 'call_expression':
                    continue
                api = _qualified(_field(node, 'function'), _visible_constants(node, bindings))
                if api not in {'node:child_process.execFile', 'node:child_process.spawn'}:
                    continue
                arguments = _field(node, 'arguments').named_children
                if not arguments:
                    continue
                visible = _visible_constants(node, values)
                target = _js_path(arguments[0], visible, bindings, relative)
                if target in {'bash', 'sh', 'python', 'python3', 'node'} and len(arguments) >= 2 and arguments[1].type == 'array' and arguments[1].named_children:
                    target = _js_path(arguments[1].named_children[0], visible, bindings, relative)
                matched = [e for e in events if e.unit_id == unit and e.line == node.start_point.row+1
                           and e.capability in {'shell_exec', 'subprocess'}]
                if target is not None and len(matched) == 1:
                    bridge(unit, relative, target, node, BridgeKind.COMMAND,
                           f'{_text(_field(node, "function"))} -> {api}; const target {_text(arguments[0])}', matched[0].event_id)
        elif suffix == '.sh':
            tree = Parser(Language(tree_sitter_bash.language())).parse(path.read_bytes())

            def visit(node, values):
                if node.type == 'variable_assignment':
                    name, value = _field(node, 'name'), _field(node, 'value')
                    if value is not None:
                        visit(value, values.copy())
                    values[_text(name)] = _shell_value(value, values, relative) if value is not None else None
                    return
                if node.type == 'if_statement':
                    branches, branch, has_else = [], values.copy(), False
                    for child in node.named_children:
                        if child.type in {'else_clause', 'elif_clause'}:
                            branches.append(branch)
                            branch = values.copy()
                            has_else |= child.type == 'else_clause'
                        visit(child, branch)
                    branches.append(branch)
                    if not has_else:
                        branches.append(values.copy())
                    for name in set().union(*(b.keys() for b in branches)):
                        options = {b.get(name) for b in branches}
                        values[name] = next(iter(options)) if len(options) == 1 else None
                    return
                if node.type in {'function_definition', 'subshell', 'command_substitution', 'for_statement', 'while_statement', 'case_statement'}:
                    for child in node.named_children:
                        visit(child, values.copy())
                    return
                if node.type == 'heredoc_body':
                    for child in node.named_children:
                        if child.type in {'command_substitution', 'expansion', 'simple_expansion'}:
                            visit(child, values.copy())
                    return
                if node.type == 'command':
                    name = _field(node, 'name')
                    command = _shell_value(name, values, relative)
                    arguments = [c for i, c in enumerate(node.named_children) if node.field_name_for_named_child(i) == 'argument']
                    target = _shell_value(arguments[0], values, relative) if command in {'bash', 'sh', 'python', 'python3', 'node'} and arguments else command
                    normalized = posixpath.normpath(target) if target else None
                    if normalized in unit_by_path and unit_by_path[normalized] != unit:
                        matched = [e for e in events if e.unit_id == unit and e.line == node.start_point.row+1
                                   and e.capability in {'shell_exec', 'subprocess'}]
                        if not matched:
                            location = f'{relative}:{node.start_point.row+1}'
                            event = CapabilityEvent(event_id=f'bound-shell-{unit}-{node.start_byte}', unit_id=unit,
                                capability='shell_exec', api_call=_text(name), location=location,
                                arguments=[normalized], tier='registered_source', language='bash', file_path=relative,
                                line=node.start_point.row+1, detail='Source-bound local script invocation')
                            events.append(event)
                            resources.append(ResourceUse(resource_id=f'bound-target-{event.event_id}', unit_id=unit,
                                event_id=event.event_id, resource_type='command', value=normalized, resolved=True,
                                location=location, origin_kind='code_literal', origin_hint='Ordered Shell binding at actual command position'))
                            matched = [event]
                        bridge(unit, relative, normalized, node, BridgeKind.COMMAND,
                               f'ordered Shell target {_text(name)}; source-directory bindings', matched[0].event_id)
                    if command == 'read':
                        for argument in arguments:
                            values.pop(_text(argument), None)
                for child in node.named_children:
                    visit(child, values.copy() if node.type == 'list' and child != node.named_children[0] else values)

            visit(tree.root_node, {})
    return bridges
