"""Registered source models for file-copy and redirected Shell read value flows."""
from __future__ import annotations

import ast
from pathlib import Path
import re

from tree_sitter import Language, Parser
import tree_sitter_bash

from skillrecon.behavior.python_calls import _imports, _qualified, _scope_nodes
from skillrecon.core.types import CapabilityEvent, PathSegment, ResourceUse, RiskPath


def _walk(node):
    yield node
    for child in node.named_children:
        if child.type not in {'function_definition', 'while_statement', 'for_statement'}:
            yield from _walk(child)


def _text(node):
    return node.text.decode('utf-8')


def recover_payload_flows(events, resources, unit_paths, staged_root: Path):
    """Recover concrete argument-slot flows without inferring them from proximity."""
    paths = []

    def event(unit, relative, line, capability, api, argument, language):
        existing = next((e for e in events if e.unit_id==unit and e.line==line
                         and e.capability==capability and e.tier!='instruction'), None)
        if existing is not None:
            return existing
        result = CapabilityEvent(event_id=f'payload-{unit}-{line}-{capability}', unit_id=unit,
            capability=capability, api_call=api, arguments=[argument], tier='registered_source',
            language=language, file_path=relative, line=line, location=f'{relative}:{line}',
            detail='Registered payload slot model')
        events.append(result)
        return result

    def resource(e, kind, value, resolved, model):
        if any(r.event_id==e.event_id and r.resource_type==kind and r.value==value for r in resources):
            return
        resources.append(ResourceUse(resource_id=f'payload-{e.event_id}-{kind}', unit_id=e.unit_id,
            event_id=e.event_id, resource_type=kind, value=value, resolved=resolved, location=e.location,
            origin_kind='code_literal' if resolved else 'unknown', origin_hint=model))

    def segment(e, slot, value, expression):
        return PathSegment(unit_id=e.unit_id, location=e.location, label=e.capability,
                           event_id=e.event_id, slot_kind=slot, symbol_hint=value, expr_hint=expression)

    def add(source, sink, model, middle):
        paths.append(RiskPath(path_id=f'payload-{source.unit_id}-{len(paths)}',
            source=source, sink=sink, segments=[source, *middle, sink],
            evidence_level='registered_source', path_kind=f'registered_source:{model}'))

    for unit, relative in unit_paths.items():
        source = (staged_root/relative).read_text(encoding='utf-8')
        suffix = Path(relative).suffix
        if suffix == '.py':
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue  # This registered model requires a Python AST.
            module_imports = _imports(tree)
            scopes = [tree]+[n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for scope in scopes:
                imports = module_imports.copy()
                if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for argument in scope.args.args:
                        imports.pop(argument.arg, None)
                for node in _scope_nodes(scope):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                imports.pop(target.id, None)
                    if not isinstance(node, ast.Call) or _qualified(node.func, imports) not in {'shutil.copy', 'shutil.copy2', 'shutil.copyfile'}:
                        continue
                    if len(node.args)!=2:
                        continue
                    values = [ast.unparse(arg) for arg in node.args]
                    incoming = event(unit, relative, node.lineno, 'file_read', _qualified(node.func, imports), values[0], 'python')
                    outgoing = event(unit, relative, node.lineno, 'file_write', _qualified(node.func, imports), values[1], 'python')
                    for e, arg, value in zip([incoming, outgoing], node.args, values):
                        literal = isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                        resource(e, 'path', arg.value if literal else value, literal, 'file_copy_argument')
                    add(segment(incoming, 'file_copy_source_arg', values[0], ast.unparse(node)),
                        segment(outgoing, 'file_copy_target_arg', values[1], ast.unparse(node)), 'file_copy', [])
        elif suffix == '.sh':
            tree = Parser(Language(tree_sitter_bash.language())).parse(source.encode('utf-8'))
            def nodes(n):
                yield n
                for child in n.named_children:
                    yield from nodes(child)
            for loop in nodes(tree.root_node):
                if loop.type!='while_statement' or loop.parent.type!='redirected_statement':
                    continue
                condition = loop.child_by_field_name('condition')
                body = loop.child_by_field_name('body')
                if condition.type!='command' or _text(condition.child_by_field_name('name'))!='read':
                    continue
                arguments = [n for n in condition.named_children if n.type!='command_name']
                if len(arguments)==2 and _text(arguments[0])=='-r':
                    arguments = arguments[1:]
                if len(arguments)!=1 or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', _text(arguments[0])):
                    continue
                carrier = _text(arguments[0])
                inputs = [n for n in loop.parent.named_children if n.type=='file_redirect'
                          and _text(n).lstrip().startswith('<') and not _text(n).lstrip().startswith('<<')]
                if len(inputs)!=1:
                    continue
                destination = inputs[0].child_by_field_name('destination')
                file_expression = _text(destination)
                incoming = None
                for node in _walk(body):
                    if node.type=='variable_assignment' and _text(node.child_by_field_name('name'))==carrier:
                        break  # A reassignment ends this read-value binding.
                    if node.type!='command':
                        continue
                    name = _text(node.child_by_field_name('name'))
                    if name in {'echo', 'printf', 'read', 'test', '[', 'true', 'false', ':', 'exit', 'return', 'break', 'continue'}:
                        continue
                    arguments = [n for n in node.named_children if n.type!='command_name']
                    for index, argument in enumerate(arguments):
                        variables = [n for n in nodes(argument) if n.type=='variable_name']
                        if not any(_text(v)==carrier for v in variables):
                            continue
                        if incoming is None:
                            incoming = event(unit, relative, condition.start_point.row+1, 'file_read', 'read', file_expression, 'bash')
                            literal = not any(n.type in {'simple_expansion','expansion','command_substitution'} for n in nodes(destination))
                            resource(incoming, 'path', file_expression.strip('\"\''), literal, 'redirected_read_input')
                        outgoing = event(unit, relative, node.start_point.row+1, 'shell_exec', name, _text(argument), 'bash')
                        resource(outgoing, 'command', name, bool(re.fullmatch(r'[A-Za-z0-9_./-]+', name)), 'read_value_command_argument')
                        option = _text(arguments[index-1]) if index and _text(arguments[index-1]).startswith('-') else str(index)
                        add(segment(incoming, 'file_read_result', carrier,
                                    f'{_text(condition)}; loop input {_text(inputs[0])}'),
                            segment(outgoing, 'command_arg', carrier, f'{option}: {_text(argument)}'), 'shell_read_argument', [])
    return paths
