"""Native Tier-2 observations for imported APIs and concrete AST operands."""

from __future__ import annotations

import ast
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_javascript
import tree_sitter_typescript

from skillrecon.behavior.normalize_types import RawObservation
from skillrecon.behavior.bridge_calls import _bind_import, _qualified
from skillrecon.behavior.resource_fields import _field, _text, _walk, _js_literal

_PYTHON_APIS = {
    **{f"{library}.{verb}": ("http_request", "url", 0)
       for library in ("requests", "httpx", "aiohttp") for verb in
       ("get", "post", "put", "patch", "delete", "head", "options")},
    "requests.request": ("http_request", "url", 1),
    "urllib.request.urlopen": ("http_request", "url", 0),
    "urllib.request.Request": ("http_request", "url", 0),
    "os.getenv": ("env_var_read", "env_var", 0),
    "os.environ.get": ("env_var_read", "env_var", 0),
    "os.system": ("shell_exec", "command", 0),
    "os.popen": ("shell_exec", "command", 0),
    **{f"subprocess.{name}": ("subprocess", "command", 0) for name in
       ("run", "Popen", "call", "check_call", "check_output")},
    "os.remove": ("file_delete", "path", 0),
    "os.unlink": ("file_delete", "path", 0),
    "shutil.rmtree": ("file_delete", "path", 0),
    "os.chmod": ("file_permission_change", "path", 0),
    "os.kill": ("process_kill", "process", 0),
    "socket.getaddrinfo": ("dns_query", "host", 0),
    "socket.gethostbyname": ("dns_query", "host", 0),
    "socket.create_connection": ("socket_connect", "host", 0),
    "keyring.get_password": ("keychain_access", "credential", 0),
    "importlib.import_module": ("dynamic_import", "import_target", 0),
    "__import__": ("dynamic_import", "import_target", 0),
    "eval": ("eval_exec", "command", 0), "exec": ("eval_exec", "command", 0),
}


def javascript_api_model(api):
    """The registered API graph shared by direct and wrapper recovery."""
    if api in {"fetch", "node-fetch", "axios"} or api and api.startswith(("axios.","node-fetch.")):
        return "http_request","url"
    if api and api.removeprefix("node:") in {"http.get","http.request","https.get","https.request"}:
        return "http_request","url"
    if api and api.startswith(("fs.","node:fs.")):
        method=api.rsplit(".",1)[-1]
        if method.startswith(("read","write","append","unlink","rm")):
            return ("file_read" if method.startswith("read") else
                    "file_delete" if method.startswith(("unlink","rm")) else "file_write"),"path"
    if api and api.startswith(("child_process.","node:child_process.")):
        return ("shell_exec" if api.rsplit(".",1)[-1] in {'exec', 'execSync'} else "subprocess"),"command"
    if api=="eval":return "eval_exec","command"
    return None


def recover_native_observations(unit_paths, staged_root: Path):
    observations, parse_failures = [], []

    def emit(relative, unit, language, node, capability, resource_type, alternatives, api, context=""):
        line = node.lineno if language == "python" else node.start_point.row + 1
        for value, resolved in alternatives:
            fields = {"capType": capability, "detail": api, "tier": "api_graph",
                      "resourceType": resource_type, "resourceValue": value,
                      "resourceResolved": str(resolved).lower(),
                      "resourceOriginKind": "native_ast", "resourceOriginHint": context,
                      "branchContext": context}
            observations.append(RawObservation(language=language, relative_path=relative,
                line=line, message="native AST argument-slot observation", fields=fields, unit_id=unit))

    for unit, relative in unit_paths.items():
        path = staged_root / relative
        language = path.suffix
        if language not in {".py", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx"}:
            continue
        text = path.read_text(encoding="utf-8")
        if language == ".py":
            try:
                tree = ast.parse(text)
            except SyntaxError as error:
                parse_failures.append({"path": relative, "line": error.lineno, "reason": str(error)})
                continue

            def qualified(node, aliases):
                if isinstance(node, ast.Name):
                    return aliases.get(node.id, node.id if node.id in {"open", "eval", "exec", "__import__"} else None)
                if isinstance(node, ast.Attribute):
                    base = qualified(node.value, aliases)
                    return f"{base}.{node.attr}" if base else None
                return None

            def value(node, constants):
                if isinstance(node, ast.Constant):
                    return ((str(node.value), True),)
                if isinstance(node, ast.Name):
                    return constants.get(node.id, ((node.id, False),))
                if isinstance(node, (ast.Tuple, ast.List)):
                    parts = [value(child, constants) for child in node.elts]
                    if all(len(part) == 1 and part[0][1] for part in parts):
                        return ((" ".join(part[0][0] for part in parts), True),)
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                    return tuple((a+b, known_a and known_b) for a,known_a in value(node.left,constants)
                                 for b,known_b in value(node.right,constants))
                if isinstance(node, ast.JoinedStr):
                    result = (("", True),)
                    for child in node.values:
                        part = value(child.value if isinstance(child,ast.FormattedValue) else child, constants)
                        result = tuple((a+b, ka and kb) for a,ka in result for b,kb in part)
                    return result
                return ((ast.unparse(node), False),)

            def body(statements, aliases, constants, context=""):
                for statement in statements:
                    if isinstance(statement, ast.Import):
                        for item in statement.names:
                            aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
                    elif isinstance(statement, ast.ImportFrom):
                        for item in statement.names:
                            aliases[item.asname or item.name] = f"{statement.module}.{item.name}"
                    if isinstance(statement, ast.Try):
                        body(statement.body, aliases, constants, context)
                        for handler in statement.handlers:
                            body(handler.body, aliases.copy(), constants.copy(), context)
                        body(statement.orelse, aliases, constants, context)
                        body(statement.finalbody, aliases, constants, context)
                        continue
                    if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        local = aliases.copy()
                        bound = constants.copy()
                        for argument in statement.args.args:
                            local[argument.arg] = None
                            bound.pop(argument.arg, None)
                        body(statement.body, local, bound, context)
                        continue
                    if isinstance(statement, ast.ClassDef):
                        body(statement.body, aliases.copy(), constants.copy(), context)
                        continue
                    if isinstance(statement, ast.If):
                        branch_constants = []
                        for branch, label in ((statement.body,"true"), (statement.orelse,"false")):
                            current = constants.copy()
                            body(branch, aliases.copy(), current, f"{ast.unparse(statement.test)} [{label}]")
                            branch_constants.append(current)
                        for name in set().union(*(branch.keys() for branch in branch_constants)):
                            constants[name] = tuple(dict.fromkeys(option for branch in branch_constants
                                for option in branch.get(name, ((name, False),))))
                        continue
                    if isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Name):
                                constants[target.id] = value(statement.value, constants)
                                reference = qualified(statement.value, aliases)
                                if isinstance(statement.value, ast.Call):
                                    factory = qualified(statement.value.func, aliases)
                                    if factory in {"requests.Session", "httpx.Client", "httpx.AsyncClient"}:
                                        reference = factory.split(".")[0]
                                aliases[target.id] = reference
                    for node in ast.walk(statement):
                        if isinstance(node, ast.Subscript) and qualified(node.value, aliases) == "os.environ":
                            emit(relative,unit,"python",node,"env_var_read","env_var",value(node.slice,constants),"os.environ[]",context)
                        if not isinstance(node,ast.Call):
                            continue
                        api = qualified(node.func, aliases)
                        model = _PYTHON_APIS.get(api)
                        if api == "open":
                            mode = next((keyword.value for keyword in node.keywords if keyword.arg == "mode"),
                                        node.args[1] if len(node.args)>1 else ast.Constant("r"))
                            modes = value(mode,constants)
                            write = any(known and any(flag in mode for flag in "wax+") for mode,known in modes)
                            model = ("file_write" if write else "file_read","path",0)
                        if model is None:
                            continue
                        cap, resource_type, slot = model
                        operand = node.args[slot] if len(node.args)>slot else next(
                            (keyword.value for keyword in node.keywords if keyword.arg in {"url","key","filename","path","args"}),None)
                        alternatives = value(operand,constants) if operand is not None else ((f"<dynamic-{resource_type}>",False),)
                        emit(relative,unit,"python",node,cap,resource_type,alternatives,api,context)
            body(tree.body,{}, {})
        else:
            grammar = (tree_sitter_typescript.language_tsx() if language==".tsx" else
                       tree_sitter_typescript.language_typescript() if language==".ts" else
                       tree_sitter_javascript.language())
            root = Parser(Language(grammar)).parse(text.encode()).root_node
            bindings, constants = {"fetch":"fetch", "process":"process", "eval":"eval"}, {}
            for node in _walk(root):
                if node.type == "import_statement":
                    source = _field(node,"source")
                    package = _text(source).strip(chr(34)+chr(39)) if source else ""
                    for pattern in _walk(node):
                        if pattern.type == "import_clause":
                            for child in pattern.named_children:
                                if child.type=="identifier":
                                    bindings[_text(child)] = package
                                elif child.type=="named_imports":
                                    for specifier in child.named_children:
                                        name = _field(specifier,"name")
                                        alias = _field(specifier,"alias")
                                        if name is not None:
                                            bindings[_text(alias or name)] = f"{package}.{_text(name)}"
                if node.type == "variable_declarator":
                    name, expression = _field(node,"name"),_field(node,"value")
                    if name is not None and expression is not None:
                        constants[_text(name)] = _js_literal(expression,constants)
                        if expression.type=="call_expression" and _text(_field(expression,"function"))=="require":
                            args = _field(expression,"arguments").named_children
                            if args:
                                package = _js_literal(args[0],constants)
                                if package is not None:
                                    _bind_import(name,package,bindings)
                        elif name.type == 'identifier':
                            reference = _qualified(expression, bindings)
                            if expression.type == 'call_expression' and _qualified(
                                    _field(expression, 'function'), bindings) in {'util.promisify', 'node:util.promisify'}:
                                operands = _field(expression, 'arguments').named_children
                                reference = _qualified(operands[0], bindings) if len(operands) == 1 else None
                            if reference is not None:
                                bindings[_text(name)] = reference
                if node.type in {"member_expression","subscript_expression"} and _qualified(
                        _field(node,"object"),bindings)=="process.env":
                    operand = _field(node,"index") if node.type=="subscript_expression" else _field(node,"property")
                    key = _js_literal(operand,constants) if node.type=="subscript_expression" else _text(operand)
                    emit(relative,unit,"typescript" if language in {".ts",".tsx"} else "javascript",node,
                         "env_var_read","env_var",((key or "<dynamic-env-var>",key is not None),),"process.env")
                if node.type!="call_expression":
                    continue
                api = _qualified(_field(node,"function"),bindings)
                args = _field(node,"arguments").named_children
                model = javascript_api_model(api)
                if model:
                    operand=args[0] if args else None
                    value=_js_literal(operand,constants) if operand is not None else None
                    emit(relative,unit,"typescript" if language in {".ts",".tsx"} else "javascript",node,
                         *model,((value or "<dynamic-"+model[1]+">",value is not None),),api)
    return observations, parse_failures
