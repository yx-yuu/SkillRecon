"""Resolve requests.Session receivers returned by local import helpers."""

from __future__ import annotations

import ast
from pathlib import Path

from skillrecon.core.types import CapabilityEvent, ResourceUse


def recover_session_requests(events, resources, unit_paths, staged_root: Path):
    """Add located HTTP calls only for statically resolved requests factories."""
    recovered = []
    for unit, relative in unit_paths.items():
        if Path(relative).suffix != ".py":
            continue
        text = (staged_root / relative).read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue  # This source form is outside the registered AST model.
        helpers = {}
        for function in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
            imports = {a.asname or a.name for n in ast.walk(function) if isinstance(n, ast.Import)
                       for a in n.names if a.name == "requests"}
            returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
            if returns and all(isinstance(n.value, ast.Name) and n.value.id in imports for n in returns):
                helpers[function.name] = function.lineno
        direct = {a.asname or a.name for n in tree.body if isinstance(n, ast.Import)
                  for a in n.names if a.name == "requests"}
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            constructors = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
            if len(constructors) != 1:
                continue
            init = constructors[0]
            modules = {**{name: init.lineno for name in direct}}
            constants, receivers = {}, {}
            for statement in init.body:
                if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                    continue
                target, value = statement.targets[0], statement.value
                if isinstance(target, ast.Name) and isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id in helpers:
                    modules[target.id] = helpers[value.func.id]
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        constants[target.attr] = value.value
                    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == "Session"
                            and isinstance(value.func.value, ast.Name) and value.func.value.id in modules):
                        receivers[target.attr] = (modules[value.func.value.id], statement.lineno)
            for receiver in list(receivers):
                writes = [n for n in ast.walk(cls) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                          and isinstance(n.value, ast.Name) and n.value.id == "self" and n.attr == receiver]
                if len(writes) != 1:
                    del receivers[receiver]

            def literal(expression, values):
                if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
                    return expression.value
                if isinstance(expression, ast.Name):
                    return values.get(expression.id)
                if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name) and expression.value.id == "self":
                    return constants.get(expression.attr)
                if isinstance(expression, ast.JoinedStr):
                    parts = [literal(n.value if isinstance(n, ast.FormattedValue) else n, values) for n in expression.values]
                    return "".join(parts) if all(p is not None for p in parts) else None
                return None

            for function in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                values = {}
                for statement in ast.walk(function):
                    if isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Name):
                                value = literal(statement.value, values)
                                if value is not None:
                                    values[target.id] = value
                    if not isinstance(statement, ast.Call) or not isinstance(statement.func, ast.Attribute):
                        continue
                    call, receiver = statement.func, statement.func.value
                    if not (call.attr in {"get", "post", "put", "patch", "delete", "head", "options"}
                            and isinstance(receiver, ast.Attribute) and isinstance(receiver.value, ast.Name)
                            and receiver.value.id == "self" and receiver.attr in receivers and statement.args):
                        continue
                    if any(e.unit_id == unit and e.line == statement.lineno and e.capability == "http_request" for e in events):
                        continue
                    url = literal(statement.args[0], values)
                    helper_line, factory_line = receivers[receiver.attr]
                    event = CapabilityEvent(event_id=f"e{len(events)}", unit_id=unit, capability="http_request",
                                            api_call=ast.unparse(call), location=f"{relative}:{statement.lineno}",
                                            arguments=[url or ast.unparse(statement.args[0])], tier="registered_source",
                                            language="python", file_path=relative, line=statement.lineno,
                                            detail=f"requests.Session via local import helper; helper={relative}:{helper_line}; factory={relative}:{factory_line}")
                    resource = ResourceUse(resource_id=f"session-{event.event_id}", unit_id=unit, event_id=event.event_id,
                                           resource_type="url", value=url or "<dynamic-url>", resolved=url is not None,
                                           location=event.location, origin_kind="code_literal" if url else "unknown",
                                           origin_hint=event.detail)
                    events.append(event)
                    resources.append(resource)
                    recovered.append(event.event_id)
    return recovered
