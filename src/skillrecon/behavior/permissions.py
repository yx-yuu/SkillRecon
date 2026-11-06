"""Recover persisted permission-policy changes from Python JSON writes."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from skillrecon.behavior.python_calls import _imports, _qualified, _scope_nodes
from skillrecon.core.types import CapabilityEvent, ResourceUse


def recover_permission_changes(events, resources, unit_paths, staged_root: Path):
    """Bind a policy assignment to serialization of that same object to settings.

    A dictionary literal or a comment alone does not establish a persisted
    permission change. The registered setting is permissions.defaultMode.
    """
    changes = []
    existing = {(e.unit_id, e.line, e.api_call) for e in events}
    for unit, relative in unit_paths.items():
        if Path(relative).suffix != '.py':
            continue
        text = (staged_root / relative).read_text(encoding='utf-8')
        if 'defaultMode' not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # Outside the registered Python syntax model.
        imports = _imports(tree)
        scopes = [tree, *(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))]
        for scope in scopes:
            bindings, policies = {}, {}
            local_imports = imports.copy()
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in scope.args.args:
                    local_imports.pop(arg.arg, None)
            for node in _scope_nodes(scope):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            bindings[target.id] = node.value
                            policies.pop(target.id, None)
                            local_imports.pop(target.id, None)
                        elif isinstance(target, ast.Subscript):
                            parts = []
                            base = target
                            while isinstance(base, ast.Subscript) and isinstance(base.slice, ast.Constant):
                                parts.insert(0, base.slice.value)
                                base = base.value
                            if isinstance(base, ast.Name) and parts == ['permissions']:
                                policies.pop(base.id, None)
                            if isinstance(base, ast.Name) and parts == ['permissions', 'defaultMode']:
                                policies.pop(base.id, None)
                                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                                    policies[base.id] = (node.value.value, node.lineno)
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if isinstance(node.func.value, ast.Name) and node.func.attr in {'update', 'clear', 'pop', '__setitem__'}:
                    policies.pop(node.func.value.id, None)
                if node.func.attr not in {'write_text', 'write_bytes'} or not node.args:
                    continue
                destination = node.func.value
                if isinstance(destination, ast.Name):
                    destination = bindings.get(destination.id)
                if destination is None or not any(
                    isinstance(n, ast.Constant) and isinstance(n.value, str)
                    and Path(n.value).name == 'settings.json' for n in ast.walk(destination)
                ):
                    continue
                serializers = [n for n in ast.walk(node.args[0]) if isinstance(n, ast.Call)
                               and _qualified(n.func, local_imports) == 'json.dumps' and n.args
                               and isinstance(n.args[0], ast.Name) and n.args[0].id in policies]
                if len(serializers) != 1:
                    continue
                value, assignment_line = policies[serializers[0].args[0].id]
                key = (unit, node.lineno, 'permission policy write')
                if key in existing:
                    continue
                existing.add(key)
                setting = f'permissions.defaultMode={value}'
                event = CapabilityEvent(
                    event_id=f'e{len(events)}', unit_id=unit, capability='file_permission_change',
                    api_call=key[2], location=f'{relative}:{node.lineno}', file_path=relative,
                    line=node.lineno, language='python', tier='registered_source',
                    arguments=[setting], detail='Persisted permission policy; ' + setting,
                )
                events.append(event)
                evidence = {'model': 'permission_policy_write_v1',
                            'assignment_location': f'{relative}:{assignment_line}',
                            'write_location': event.location,
                            'object': serializers[0].args[0].id,
                            'setting': setting}
                resource = ResourceUse(
                    resource_id=f'permission-{event.event_id}', unit_id=unit,
                    event_id=event.event_id, resource_type='permission_setting', value=setting,
                    location=f'{relative}:{assignment_line}', resolved=True,
                    origin_kind='source_binding', origin_hint=json.dumps(evidence),
                )
                resources.append(resource)
                changes.append(evidence)
    return changes
