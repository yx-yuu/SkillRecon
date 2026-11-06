"""Package-local Python module and function bindings shared by Tier 2."""
from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

from skillrecon.behavior.python_calls import _constant, _imports


def python_modules(unit_paths, staged_root):
    modules = {}
    for unit, relative in unit_paths.items():
        if Path(relative).suffix != '.py':
            continue
        try:
            tree = ast.parse((staged_root / relative).read_text(encoding='utf-8'))
        except SyntaxError:
            continue  # Parse failures are reported by native capability recovery.
        values = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        values[target.id] = _constant(statement.value, values)
        modules[relative] = {'unit': unit, 'tree': tree, 'aliases': _imports(tree),
            'values': values, 'functions': {node.name: node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}}
    return modules


def local_python_function(node, relative, modules):
    aliases = modules[relative]['aliases']
    if isinstance(node, ast.Name):
        binding = aliases.get(node.id)
        if isinstance(binding, tuple):
            imported, name = binding
        else:
            return (relative, node.id) if node.id in modules[relative]['functions'] else None
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        binding = aliases.get(node.value.id)
        if isinstance(binding, str):
            imported, name = binding, node.attr
        elif isinstance(binding, tuple):
            imported, member = binding
            imported = '.'.join(filter(None, (imported, member)))
            name = node.attr
        else:
            return None
    else:
        return None
    level = len(imported) - len(imported.lstrip('.'))
    parent = PurePosixPath(relative).parent
    for _ in range(max(level - 1, 0)):
        parent = parent.parent
    suffix = imported.lstrip('.').replace('.', '/')
    candidates = [str(parent / (suffix + '.py')), str(parent / suffix / '__init__.py')]
    if not level:
        candidates += [suffix + '.py', suffix + '/__init__.py']
    matches = {(path, name) for path in candidates
               if path in modules and name in modules[path]['functions']}
    return next(iter(matches)) if len(matches) == 1 else None
