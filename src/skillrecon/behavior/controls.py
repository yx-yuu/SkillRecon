"""Source models for explicit configuration gates and workflow restrictions.

These models bind control evidence to existing network events. They neither
execute skill code nor infer a violation from an unrecognized condition.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_javascript

from skillrecon.core.types import CapabilityEvent, Clause, ResourceUse

logger = logging.getLogger(__name__)


def recover_control_resources(
    events: list[CapabilityEvent], unit_paths: dict[str, str], staged_root: Path,
    clauses: list[Clause],
) -> list[ResourceUse]:
    """Return event-bound, source-located control evidence for registered forms."""
    requested = {(c.constraint_type, c.value) for clause in clauses for c in clause.constraints
                 if c.constraint_type in {"config_enabled", "empty_disables", "standalone_call", "retry_on_error"}}
    if not requested:
        return []
    resources = []
    for unit, relative in unit_paths.items():
        local = [e for e in events if e.unit_id == unit and e.capability == "http_request" and e.tier != "instruction"]
        if not local:
            continue
        text = (staged_root / relative).read_text(encoding="utf-8")
        suffix = Path(relative).suffix
        if suffix == ".py":
            facts = _python_facts(text, local, requested)
        elif suffix in {".js", ".mjs", ".cjs"}:
            facts = _javascript_facts(text, local, requested)
        else:
            facts = []
        for event, kind, key, result, lines, explanation in facts:
            evidence = {"kind": kind, "key": key, "result": result,
                        "locations": [f"{relative}:{line}" for line in sorted(set(lines))],
                        "source": [text.splitlines()[line - 1] for line in sorted(set(lines))],
                        "explanation": explanation, "model": "registered_source_control_v1"}
            resources.append(ResourceUse(
                resource_id=f"control-{event.event_id}-{kind}-{key}", unit_id=unit,
                event_id=event.event_id, resource_type="control_condition", value=f"{kind}:{key}",
                location=event.location, origin_kind="static_condition",
                origin_hint=json.dumps(evidence, ensure_ascii=False),
            ))
    return resources


def _method_call(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _field_read(node: ast.AST, key: str) -> bool:
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
        return node.attr == key
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        return isinstance(node.slice, ast.Constant) and node.slice.value == key
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        return bool(node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == key)
    return False


def _request_uses_record(call: ast.Call, names: set[str], function: ast.AST) -> bool:
    assignments = {t.id: n.value for n in ast.walk(function) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)}

    def depends(expression, visited):
        for node in ast.walk(expression):
            if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.value, ast.Name) and node.value.id in names:
                return True
            if isinstance(node, ast.Name) and node.id in assignments and node.id not in visited:
                if depends(assignments[node.id], visited | {node.id}):
                    return True
        return False

    return any(depends(arg, set()) for arg in [*call.args, *(kw.value for kw in call.keywords)])


def _recoverable_retry(statements, count, limit, constants, *, error_name=None, request_returns=None):
    """Check a bounded network-error path using registered pure expressions."""
    values = dict(constants)
    if not isinstance(values.get(count), int) or values[count] + 1 >= limit:
        return None
    request_returns = request_returns or {}
    increments = []

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return values.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = value(node.operand)
            return not operand if isinstance(operand, bool) else None
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            operands = [value(n) for n in node.values]
            return any(operands) if all(isinstance(n, bool) for n in operands) else None
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
            left, right = value(node.left), value(node.comparators[0])
            return left == right if left is not None and right is not None else None
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt):
            left, right = value(node.left), value(node.comparators[0])
            return left < right if isinstance(left, int) and isinstance(right, int) else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "self" and node.func.attr in request_returns:
                return request_returns[node.func.attr]
            if node.func.attr == "get" and node.args:
                mapping = value(node.func.value)
                if isinstance(mapping, dict):
                    return mapping.get(value(node.args[0]), value(node.args[1]) if len(node.args) > 1 else None)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "str" and len(node.args) == 1:
                if isinstance(node.args[0], ast.Name) and node.args[0].id == error_name:
                    return "Connection"
                operand = value(node.args[0])
                return str(operand) if operand is not None else None
            if node.func.id == "isinstance" and len(node.args) == 2 and isinstance(node.args[1], ast.Name) and node.args[1].id == "str":
                operand = value(node.args[0])
                return isinstance(operand, str) if operand is not None else None
            if node.func.id == "any" and len(node.args) == 1 and isinstance(node.args[0], ast.GeneratorExp):
                generator = node.args[0]
                if len(generator.generators) == 1:
                    comprehension = generator.generators[0]
                    check = generator.elt
                    if (isinstance(comprehension.target, ast.Name) and isinstance(comprehension.iter, (ast.List, ast.Tuple))
                            and not comprehension.ifs and isinstance(check, ast.Compare) and len(check.ops) == 1
                            and isinstance(check.ops[0], ast.In) and isinstance(check.left, ast.Name)
                            and check.left.id == comprehension.target.id):
                        haystack = value(check.comparators[0])
                        tokens = [value(n) for n in comprehension.iter.elts]
                        if isinstance(haystack, str) and all(isinstance(t, str) for t in tokens):
                            return any(t in haystack for t in tokens)
        return None

    def follow(statements):
        for statement in statements:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
                result = value(statement.value)
                if result is None:
                    return False
                values[statement.targets[0].id] = result
            elif (isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name)
                  and statement.target.id == count and isinstance(statement.op, ast.Add)
                  and isinstance(statement.value, ast.Constant) and statement.value.value == 1):
                values[count] += 1
                increments.append(statement.lineno)
            elif isinstance(statement, ast.If):
                condition = value(statement.test)
                if not isinstance(condition, bool) or not follow(statement.body if condition else statement.orelse):
                    return False
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if not ((isinstance(call.func, ast.Name) and call.func.id == "print")
                        or (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                            and call.func.value.id == "time" and call.func.attr == "sleep")):
                    return False
            else:
                return False
        return True

    return increments if follow(statements) and increments and values[count] < limit else None


def _network_error_return(function, events):
    """A requests failure caught locally and returned as a concrete error dict."""
    for attempt in [n for n in ast.walk(function) if isinstance(n, ast.Try)]:
        if not any(attempt.lineno <= e.line <= attempt.body[-1].end_lineno for e in events):
            continue
        for handler in attempt.handlers:
            if not (isinstance(handler.type, ast.Name) and handler.type.id == "Exception"):
                continue
            for statement in handler.body:
                if not (isinstance(statement, ast.Return) and isinstance(statement.value, ast.Dict)):
                    continue
                fields = {k.value: v for k, v in zip(statement.value.keys, statement.value.values)
                          if isinstance(k, ast.Constant)}
                code, message = fields.get("code"), fields.get("msg")
                if (isinstance(code, ast.UnaryOp) and isinstance(code.op, ast.USub) and isinstance(code.operand, ast.Constant)
                        and isinstance(message, ast.Call) and isinstance(message.func, ast.Name) and message.func.id == "str"
                        and len(message.args) == 1 and isinstance(message.args[0], ast.Name) and message.args[0].id == handler.name):
                    return {"code": -code.operand.value, "msg": "Connection"}, statement.lineno
    return None


def _python_facts(text, events, requested):
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        logger.warning("Unsupported Python control syntax at %s:%s", events[0].file_path, error.lineno)
        return []
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    owners = {n: parents.get(n) for n in functions}
    facts = []

    def bound(name, caller):
        matches = [f for f in functions if f.name == name and owners[f] is owners[caller]]
        return matches[0] if len(matches) == 1 else None

    def function_events(function):
        return [e for e in events if function.lineno <= e.line <= function.end_lineno]

    for kind, key in requested:
        if kind == "config_enabled":
            for selector in functions:
                # A direct keyed record selection, with only assignments and an
                # existence check before return. More involved selectors stay unknown.
                if any(_field_read(n, key) for n in ast.walk(selector)):
                    continue
                returns = [n for n in selector.body if isinstance(n, ast.Return)]
                if len(returns) != 1:
                    continue
                returned = returns[0].value
                selected = returned.elts[0] if isinstance(returned, ast.Tuple) else returned
                if not (isinstance(selected, ast.Subscript) and isinstance(selected.value, ast.Name)
                        and isinstance(selected.slice, ast.Name)):
                    continue
                record = selected.value.id
                assignments = [n for n in selector.body if isinstance(n, ast.Assign)]
                if not any(any(isinstance(t, ast.Name) and t.id == record for t in n.targets)
                           and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                           and n.value.func.attr == "get" for n in assignments):
                    continue
                admissible = True
                for n in selector.body:
                    if isinstance(n, (ast.Assign, ast.Return)) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)):
                        continue
                    if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name):
                        if (n.test.left.id == selected.slice.id and len(n.test.ops) == 1
                                and isinstance(n.test.ops[0], ast.NotIn)
                                and isinstance(n.test.comparators[0], ast.Name)
                                and n.test.comparators[0].id == record and not n.orelse
                                and all(isinstance(s, ast.Raise) for s in n.body)):
                            continue
                    admissible = False
                if not admissible:
                    continue
                for caller in functions:
                    calls = [n for n in ast.walk(caller) if _method_call(n) == selector.name
                             and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"]
                    if not calls or bound(selector.name, caller) is not selector:
                        continue
                    if any(_field_read(n, key) for n in ast.walk(owners[caller])):
                        continue
                    # Require selected-record fields to supply a request, rather
                    # than treating an unused lookup as a side effect.
                    selected_names = set()
                    for call in calls:
                        expression = parents[call]
                        if isinstance(expression, ast.Subscript):
                            expression = parents[expression]
                        if isinstance(expression, ast.Assign):
                            for target in expression.targets:
                                if isinstance(target, (ast.Tuple, ast.List)):
                                    target = target.elts[0]
                                selected_names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
                    reads = [n for n in ast.walk(caller) if isinstance(n, (ast.Subscript, ast.Attribute))
                             and isinstance(n.value, ast.Name) and n.value.id in selected_names
                             and isinstance(n.ctx, ast.Load)]
                    if not reads:
                        continue
                    for event in function_events(caller):
                        requests = [n for n in ast.walk(caller) if isinstance(n, ast.Call) and n.lineno == event.line]
                        if (event.line > min(n.lineno for n in calls) and any(n.lineno < event.line for n in reads)
                                and any(_request_uses_record(n, selected_names, caller) for n in requests)):
                            facts.append((event, kind, key, "false", [selector.lineno, returns[0].lineno,
                                          calls[-1].lineno, reads[-1].lineno, event.line],
                                          "An existing selected record with the flag false is returned; the record supplies request fields without reading the activation flag."))
        elif kind == "standalone_call":
            # Only a concrete CLI branch is independent invocation. Internal
            # calls made by a merged workflow do not satisfy this condition.
            for branch in [n for n in ast.walk(tree) if isinstance(n, ast.If)]:
                test = branch.test
                if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                        and isinstance(test.left, ast.Attribute) and test.left.attr == "command"
                        and isinstance(test.comparators[0], ast.Constant)):
                    continue
                command = test.comparators[0].value
                if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_parser"
                           and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == command for n in ast.walk(tree)):
                    continue
                for statement in branch.body:
                    for call in ast.walk(statement):
                        if _method_call(call) != key:
                            continue
                        receiver = ast.unparse(call.func.value)
                        classes = [n.value.func.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                                   and any(isinstance(t, ast.Name) and t.id == receiver for t in n.targets)
                                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)]
                        targets = [f for f in functions if f.name == key and isinstance(owners[f], ast.ClassDef)
                                   and owners[f].name in classes]
                        if len(targets) == 1:
                            for event in function_events(targets[0]):
                                facts.append((event, kind, key, "true", [branch.lineno, call.lineno, targets[0].lineno, event.line],
                                              f"Registered CLI command {command!r} invokes {key} independently; calls inside a merged workflow are excluded."))
        elif kind == "retry_on_error":
            for loop in [n for n in ast.walk(tree) if isinstance(n, ast.While)]:
                test = loop.test
                if not (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                        and len(test.ops) == 1 and isinstance(test.ops[0], ast.Lt)):
                    continue
                count = test.left.id
                function = parents[loop]
                while function not in functions and function in parents:
                    function = parents[function]
                if function not in functions:
                    continue
                limit = test.comparators[0]
                constants = {t.id: n.value.value for n in ast.walk(function) if isinstance(n, ast.Assign)
                             and isinstance(n.value, ast.Constant) for t in n.targets if isinstance(t, ast.Name)}
                bound_value = limit.value if isinstance(limit, ast.Constant) else constants.get(limit.id) if isinstance(limit, ast.Name) else None
                if not isinstance(bound_value, int) or bound_value < 2:
                    continue
                siblings = getattr(parents[loop], "body", [])
                if loop not in siblings:
                    continue
                initializers = [n for n in siblings[:siblings.index(loop)] if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == count for t in n.targets)]
                if not initializers or not isinstance(initializers[-1].value, ast.Constant):
                    continue
                constants[count] = initializers[-1].value.value
                for attempt in loop.body:
                    if not isinstance(attempt, ast.Try):
                        continue
                    for statement in attempt.body:
                        for call in ast.walk(statement):
                            name = _method_call(call)
                            if name and isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
                                target = bound(name, function)
                                if target:
                                    returned_error = _network_error_return(target, function_events(target))
                                    repeatable, error_line = [], target.lineno
                                    if returned_error:
                                        increments = _recoverable_retry(attempt.body, count, bound_value, constants,
                                                                       request_returns={name: returned_error[0]})
                                        repeatable = increments or []
                                        error_line = returned_error[1]
                                    else:
                                        for handler in attempt.handlers:
                                            if handler.type is None or (isinstance(handler.type, ast.Name) and handler.type.id == "Exception"):
                                                increments = _recoverable_retry(handler.body, count, bound_value, constants,
                                                                               error_name=handler.name)
                                                repeatable.extend(increments or [])
                                    if not repeatable:
                                        continue
                                    for event in function_events(target):
                                        facts.append((event, kind, key, "true", [loop.lineno, call.lineno,
                                                      repeatable[0], target.lineno, error_line, event.line],
                                                      f"A network failure represented as Connection reaches the error-handling branch, increments {count}, and repeats the same request in a loop bounded by {bound_value}; this is automatic retry, not ordinary polling."))
    return facts


def _javascript_facts(text, events, requested):
    parser = Parser(Language(tree_sitter_javascript.language()))
    source = text.encode("utf-8")
    tree = parser.parse(source)
    if tree.root_node.has_error:
        logger.warning("Unsupported JavaScript control syntax in %s", events[0].file_path)
        return []

    def walk(node):
        yield node
        for child in node.named_children:
            yield from walk(child)

    def surface(node):
        return source[node.start_byte:node.end_byte].decode("utf-8")

    nodes = list(walk(tree.root_node))
    facts = []
    for kind, key in requested:
        if kind != "empty_disables":
            continue
        for declaration in [n for n in nodes if n.type == "variable_declarator"]:
            name = declaration.child_by_field_name("name")
            value = declaration.child_by_field_name("value")
            if name is None or value is None or surface(name) != key or value.type != "binary_expression":
                continue
            left, right = value.child_by_field_name("left"), value.child_by_field_name("right")
            if (surface(left) != f"process.env.{key}" or right.type != "string"
                    or not any(n.type == "||" for n in value.children) or len(surface(right)) <= 2):
                continue
            # A reassignment or another declaration needs a different model.
            if sum(n.type == "variable_declarator" and n.child_by_field_name("name") is not None
                   and surface(n.child_by_field_name("name")) == key for n in nodes) != 1:
                continue
            if any(n.type == "assignment_expression" and surface(n.child_by_field_name("left")) == key for n in nodes):
                continue
            for function in [n for n in nodes if n.type == "function_declaration"]:
                # A guard on the raw configuration could still block the call.
                guards = [n for n in nodes if n.type in {"if_statement", "ternary_expression", "while_statement"}]
                if any(key in surface(n.child_by_field_name("condition")) for n in guards
                       if n.child_by_field_name("condition") is not None):
                    continue
                for call in [n for n in walk(function) if n.type == "call_expression"]:
                    callee, arguments = call.child_by_field_name("function"), call.child_by_field_name("arguments")
                    if surface(callee) != "fetch" or not arguments.named_children or surface(arguments.named_children[0]) != key:
                        continue
                    for event in events:
                        if call.start_point.row + 1 <= event.line <= call.end_point.row + 1:
                            facts.append((event, kind, key, "false", [declaration.start_point.row + 1,
                                          function.start_point.row + 1, call.start_point.row + 1],
                                          f"For process.env.{key} = empty string, || selects the nonempty default {surface(right)} and the request still uses it."))
    return facts
