"""Package-local JavaScript/TypeScript declarations and resolved API bindings."""
from __future__ import annotations

from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_javascript
import tree_sitter_typescript

from skillrecon.behavior.bridge_calls import _bind_import, resolve_js_import
from skillrecon.behavior.resource_fields import _field, _js_literal, _text, _walk

FUNCTION_TYPES = {'function_declaration', 'function_expression', 'arrow_function',
                  'method_definition', 'generator_function_declaration'}


def parameter_names(function):
    parameters = _field(function, 'parameters')
    if parameters is None:
        parameter = _field(function, 'parameter')
        return [_text(parameter)] if parameter is not None and parameter.type == 'identifier' else []
    names = []
    for parameter in parameters.named_children:
        if parameter.type in {'required_parameter', 'optional_parameter'}:
            parameter = _field(parameter, 'pattern')
        if parameter is not None and parameter.type == 'assignment_pattern':
            parameter = _field(parameter, 'left')
        names.append(_text(parameter) if parameter is not None and parameter.type == 'identifier' else None)
    return names


def scope_nodes(node):
    """Traverse this scope, including immediate IIFEs and Promise executors."""
    yield node
    for child in node.named_children:
        if child.type in FUNCTION_TYPES:
            wrapped = child
            while wrapped.parent is not None and wrapped.parent.type == 'parenthesized_expression':
                wrapped = wrapped.parent
            invoked = (wrapped.parent is not None and wrapped.parent.type == 'call_expression'
                       and _field(wrapped.parent, 'function') == wrapped)
            parent = child.parent
            executor = (parent is not None and parent.type == 'arguments' and parent.parent.type == 'new_expression'
                        and _text(_field(parent.parent, 'constructor')) == 'Promise')
            if not (invoked or executor):
                continue
        yield from scope_nodes(child)


def javascript_modules(unit_paths, staged_root):
    modules = {}
    by_path = {relative: unit for unit, relative in unit_paths.items()}
    for unit, relative in unit_paths.items():
        suffix = Path(relative).suffix
        if suffix not in {'.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx'}:
            continue
        language = (tree_sitter_typescript.language_tsx() if suffix == '.tsx' else
                    tree_sitter_typescript.language_typescript() if suffix == '.ts' else
                    tree_sitter_javascript.language())
        tree = Parser(Language(language)).parse((staged_root / relative).read_bytes())
        functions, imports, constants, exports = {}, {}, {}, {}
        bindings = {'fetch': 'fetch', 'eval': 'eval'}
        declarations = []
        for statement in tree.root_node.named_children:
            if statement.type == 'export_statement':
                declaration = _field(statement, 'declaration') or _field(statement, 'value')
                if declaration is not None:
                    declarations.append(declaration)
                    if b'default' in statement.text.split():
                        name = _field(declaration, 'name')
                        key = _text(name) if name is not None else 'default'
                        if declaration.type in FUNCTION_TYPES:
                            functions[key] = declaration
                        exports['default'] = key
                    elif declaration.type in FUNCTION_TYPES:
                        name = _field(declaration, 'name')
                        if name is not None:
                            exports[_text(name)] = _text(name)
                    else:
                        for child in declaration.named_children:
                            if child.type == 'variable_declarator':
                                name = _field(child, 'name')
                                if name.type == 'identifier':
                                    exports[_text(name)] = _text(name)
                for child in _walk(statement):
                    if child.type == 'export_specifier':
                        name, alias = _field(child, 'name'), _field(child, 'alias')
                        exports[_text(alias or name)] = _text(name)
            else:
                declarations.append(statement)
        for node in declarations:
            if node.type in FUNCTION_TYPES:
                name = _field(node, 'name')
                if name is not None:
                    functions[_text(name)] = node
            if node.type == 'import_statement':
                package = _js_literal(_field(node, 'source'), {})
                target = resolve_js_import(relative, package, by_path) if package and package.startswith('.') else None
                for child in _walk(node):
                    if child.type == 'import_specifier':
                        name, alias = _field(child, 'name'), _field(child, 'alias')
                        if target:
                            imports[_text(alias or name)] = (target, _text(name))
                        else:
                            bindings[_text(alias or name)] = f'{package}.{_text(name)}'
                    elif child.type == 'namespace_import':
                        name = child.named_children[-1]
                        if target:
                            imports[_text(name)] = (target, '*')
                        else:
                            bindings[_text(name)] = package
                    elif child.type == 'import_clause':
                        for name in child.named_children:
                            if name.type == 'identifier':
                                if target:
                                    imports[_text(name)] = (target, 'default')
                                else:
                                    bindings[_text(name)] = package
            if node.type in {'lexical_declaration', 'variable_declaration'}:
                for declaration in node.named_children:
                    if declaration.type != 'variable_declarator':
                        continue
                    name, value = _field(declaration, 'name'), _field(declaration, 'value')
                    if value is None:
                        continue
                    if value.type in FUNCTION_TYPES and name.type == 'identifier':
                        functions[_text(name)] = value
                    if name.type == 'identifier':
                        literal = _js_literal(value, constants)
                        if literal is not None:
                            constants[_text(name)] = literal
                    if value.type == 'call_expression' and _text(_field(value, 'function')) == 'require':
                        args = _field(value, 'arguments').named_children
                        package = _js_literal(args[0], constants) if len(args) == 1 else None
                        if package is None:
                            continue
                        if not package.startswith('.'):
                            _bind_import(name, package, bindings)
                            continue
                        target = resolve_js_import(relative, package, by_path)
                        if target and name.type == 'identifier':
                            imports[_text(name)] = (target, '*')
                        elif target and name.type == 'object_pattern':
                            for member in name.named_children:
                                if member.type == 'shorthand_property_identifier_pattern':
                                    imports[_text(member)] = (target, _text(member))
                                elif member.type == 'pair_pattern':
                                    imports[_text(_field(member, 'value'))] = (target, _text(_field(member, 'key')))
            if node.type == 'expression_statement':
                assignment = next((child for child in node.named_children if child.type == 'assignment_expression'), None)
                if assignment is None:
                    continue
                left, right = _field(assignment, 'left'), _field(assignment, 'right')
                exported = _text(left)
                if exported == 'module.exports' and right.type == 'object':
                    for member in right.named_children:
                        if member.type == 'shorthand_property_identifier':
                            exports[_text(member)] = _text(member)
                        elif member.type == 'pair':
                            key, value = _field(member, 'key'), _field(member, 'value')
                            name = _text(key).strip('\"\'')
                            if value.type in FUNCTION_TYPES:
                                functions[name] = value
                                exports[name] = name
                            elif value.type == 'identifier':
                                exports[name] = _text(value)
                elif exported == 'module.exports' and right.type in FUNCTION_TYPES:
                    functions['default'] = right
                    exports['default'] = 'default'
                elif exported == 'module.exports' and right.type == 'identifier':
                    exports['default'] = _text(right)
                elif exported.startswith(('exports.', 'module.exports.')):
                    name = exported.rsplit('.', 1)[-1]
                    if right.type in FUNCTION_TYPES:
                        functions[name] = right
                        exports[name] = name
                    elif right.type == 'identifier':
                        exports[name] = _text(right)
        modules[relative] = {'unit': unit, 'root': tree.root_node, 'functions': functions,
                             'bindings': bindings, 'imports': imports, 'constants': constants, 'exports': exports}
    return modules


def local_function(node, relative, modules):
    """Resolve actual local declarations or exported package-local bindings."""
    module = modules[relative]
    if node.type == 'identifier':
        name = _text(node)
        if name in module['functions']:
            return relative, name
        imported = module['imports'].get(name)
    elif node.type == 'member_expression':
        imported = module['imports'].get(_text(_field(node, 'object')))
        imported = (imported[0], _text(_field(node, 'property'))) if imported and imported[1] == '*' else None
    else:
        imported = None
    if imported and imported[0] in modules:
        target, name = imported
        name = modules[target]['exports'].get(name)
        if name in modules[target]['functions']:
            return target, name
    return None
