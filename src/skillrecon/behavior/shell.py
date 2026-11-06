"""Executable Shell AST nodes with branch-sensitive symbolic operands."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import posixpath
import re

from tree_sitter import Language, Parser
import tree_sitter_bash


@dataclass(frozen=True)
class ShellCommand:
    name: str
    line: int
    column: int
    source: str
    arguments: tuple[tuple[tuple[str, bool], ...], ...]
    environment: dict[str, tuple[tuple[str, bool], ...]]
    context: str = ""


_EMPTY = (("", True),)


@dataclass(frozen=True)
class _Flow:
    # Keep stdout correlated with success/failure through && and ||. Returned
    # branches never rejoin later statements in the same function.
    success: tuple = _EMPTY
    failure: tuple = _EMPTY
    terminated: tuple = ()

    @property
    def stdout(self):
        return tuple(dict.fromkeys((*self.success, *self.failure)))

    @property
    def falls(self):
        return bool(self.success or self.failure)

    def __iter__(self):
        yield self.stdout
        yield self.falls


class _ShellBindings(dict):
    """Keep shell-local values separate from the exported child environment."""

    def __init__(self, values=(), exported=(), conditional=(), aliases=(), functions=(),
                 local_names=(), positionals=None, expand_aliases=False):
        super().__init__(values)
        self.exported = set(exported)
        self.conditional = set(conditional)
        self.aliases = dict(aliases)
        self.functions = dict(functions)
        self.local_names = set(local_names)
        self.positionals = positionals
        self.expand_aliases = expand_aliases

    def copy(self):
        return _ShellBindings(self, self.exported, self.conditional, self.aliases,
                              self.functions, self.local_names, self.positionals, self.expand_aliases)


def timeout_command_index(arguments):
    """GNU timeout's executable operand, after options and the duration slot."""
    index = 0
    while index < len(arguments) and arguments[index] is not None and arguments[index].startswith('-'):
        option = arguments[index]
        if option == '--':
            index += 1
            break
        if option in {'-s', '--signal', '-k', '--kill-after'}:
            index += 2
        elif option in {'--preserve-status', '--foreground', '-v', '--verbose'} or option.startswith(('--signal=', '--kill-after=')):
            index += 1
        else:
            return None
    return index + 1 if index + 1 < len(arguments) else None


def network_target_bindings(command):
    """URL operands of curl/wget; headers, bodies and time budgets are not targets."""
    targets, waiting = [], None
    value_options = {'-d', '--data', '--data-raw', '--data-binary', '--data-urlencode',
        '-H', '--header', '-F', '--form', '-T', '--upload-file', '-u', '--user',
        '--post-data', '--post-file', '--max-time', '-m', '--connect-timeout', '--retry',
        '--retry-delay', '--max-redirs', '--output', '-o', '--output-document', '-O',
        '--request', '-X', '--limit-rate', '--user-agent', '-A', '--proxy', '-x',
        '--config', '-K', '--write-out', '-w', '--cert', '--key', '--cacert'}
    for options in command.arguments:
        token = options[0][0] if len(options) == 1 and options[0][1] else None
        if waiting is not None:
            if waiting == '--url':targets.extend(options)
            waiting = None
        elif token == '--url' or token in value_options:
            waiting = token
        elif token is not None and token.startswith('--url='):
            targets.append((token.split('=', 1)[1], True))
        elif token is None or not token.startswith('-'):
            targets.extend(options)
    return tuple(dict.fromkeys(targets))


def shell_commands(content: str, entry_bindings=None, *, script_path=None) -> list[ShellCommand]:
    """Never execute the script. Unknown expansions remain symbolic alternatives."""
    root = Parser(Language(tree_sitter_bash.language())).parse(content.encode()).root_node
    commands = []
    active, visited, definitions = set(), set(), {}

    def combine(parts):
        return tuple(dict.fromkeys((''.join(text for text, _ in items), all(known for _, known in items))
                                  for items in product(*parts))) if parts else (("", True),)

    def choices(parts):
        return tuple(dict.fromkeys(option for part in parts for option in part))

    def positional(state, arguments):
        for name in [name for name in state if name.isdigit() and name != '0']:
            state.pop(name)
        for name in ['@', '*', '#']:
            state.pop(name, None)
        state.positionals = arguments
        if arguments is not None:
            state.update({str(index+1): argument for index, argument in enumerate(arguments)})
            state['#'] = ((str(len(arguments)), True),)
            state['*'] = combine([part for index, argument in enumerate(arguments)
                                  for part in ([((" ", True),), argument] if index else [argument])])

    def values(node, state, context='', read_context=None):
        if node is None:
            return (("", True),)
        text = node.text.decode()
        if node.type in {"simple_expansion", "expansion"}:
            variable = next((child for child in node.named_children
                             if child.type in {"variable_name", "special_variable_name", "subscript"}), None)
            if variable is not None:
                name = variable.text.decode()
                result = state.get(name, ((text, False),))
                operator = next((child.type for child in node.children if child.type in {':-', ':+', '-', '+'}), None)
                if operator is not None:
                    default = combine([values(child, state, context, read_context) for child in node.named_children if child != variable])
                    alternatives = []
                    for value, known in result:
                        if operator in {':-', '-'}:
                            alternatives.extend(default if known and not value else [(value, known), *(() if known else default)])
                        else:
                            alternatives.extend(default if known and value else [("", True)] if known else [(text, False), *default])
                    result = tuple(dict.fromkeys(alternatives))
                return result
            return ((text, False),)
        if node.type == "command_substitution":
            flow = sequence(node.named_children, state.copy(), context, read_context, capture=True)
            output = choices([flow.stdout, flow.terminated])
            if script_path is not None:
                from skillrecon.behavior.bridge_calls import _shell_value
                anchor = _shell_value(node, {name: options[0][0] for name, options in state.items()
                                            if len(options) == 1 and options[0][1]}, script_path)
                if anchor is not None:
                    return ((anchor, True),)
            return tuple(dict.fromkeys((value.rstrip('\n'), known) for value, known in output))
        if node.type == "process_substitution":
            sequence(node.named_children, state.copy(), context, read_context)
            return ((text, False),)
        if node.type == "raw_string":
            return ((text[1:-1], True),)
        if node.type in {"string", "concatenation", "command_name"} and node.named_children:
            return combine([values(child, state, context, read_context) for child in node.named_children])
        return ((text, node.type not in {"variable_name", "subscript"}),)

    def merge(state, branches):
        for name in set().union(*(branch.keys() for branch in branches)):
            state[name] = tuple(dict.fromkeys(option for branch in branches
                               for option in branch.get(name, (("$" + "{" + name + "}", False),))))
        exported = set.intersection(*(branch.exported for branch in branches))
        possible = set.union(*(branch.exported | branch.conditional for branch in branches))
        state.exported = exported
        state.conditional = possible - exported
        state.local_names = set.union(*(branch.local_names for branch in branches))
        state.aliases = {name: value for name, value in branches[0].aliases.items()
                         if all(branch.aliases.get(name) == value for branch in branches)}
        state.functions = {name: value for name, value in branches[0].functions.items()
                           if all(branch.functions.get(name) == value for branch in branches)}
        state.expand_aliases = any(branch.expand_aliases for branch in branches)
        if not all(branch.positionals == branches[0].positionals for branch in branches):
            state.positionals = None

    def child_environment(state, prefix=()):
        environment = {}
        for name in state.exported | state.conditional | set(prefix):
            choices = state.get(name, (("${" + name + "}", False),))
            if name in state.conditional and name not in prefix:
                choices = tuple(dict.fromkeys([*choices, ("${" + name + "}", False)]))
            environment[name] = choices
        return environment

    def sequence(nodes, state, context, read_context=None, capture=False):
        current, terminated, end_line, frozen = _Flow(_EMPTY, ()), (), -1, read_context
        for child in nodes:
            if read_context is None and child.start_point.row > end_line:
                frozen = (state.aliases.copy(), state.expand_aliases)
            end_line = max(end_line, child.end_point.row)
            result = walk(child, state, context, frozen, capture)
            if capture:
                prefix = current.stdout
                if result.terminated:
                    terminated = choices([terminated, combine([prefix, result.terminated])])
                current = _Flow(combine([prefix, result.success]) if result.success else (),
                                combine([prefix, result.failure]) if result.failure else (), terminated)
            else:
                current = result
            if not result.falls:
                return current
        return current

    def invoke(name, arguments, state, context, capture, prefix=()):
        definition, read_context = state.functions[name]
        identifier = definition.start_byte
        if identifier in active:
            return _Flow((("<recursive-output>", False),), (("<recursive-output>", False),))
        active.add(identifier);visited.add(identifier)
        local = state.copy()
        local.local_names = set(prefix)
        positional(local, arguments)
        result = walk(definition.child_by_field_name('body'), local, context, read_context, capture)
        active.remove(identifier)
        restore = local.local_names
        for key in set(state) | set(local):
            if key not in restore and not key.isdigit() and key not in {'@', '*', '#'}:
                if key in local:
                    state[key] = local[key]
                else:
                    state.pop(key, None)
        state.exported = (local.exported - restore) | (state.exported & restore)
        state.conditional = (local.conditional - restore) | (state.conditional & restore)
        state.aliases, state.functions = local.aliases, local.functions
        state.expand_aliases = local.expand_aliases
        output = choices([result.stdout, result.terminated])
        return _Flow(output, output)

    def walk(node, state, context="", read_context=None, capture=False):
        empty = (("", True),)
        if read_context is None:
            read_context = (state.aliases.copy(), state.expand_aliases)
        if node.type == 'function_definition':
            name = node.child_by_field_name('name').text.decode()
            state.functions[name] = (node, read_context)
            definitions[node.start_byte] = (name, node, read_context)
            return _Flow()
        if node.type == "unset_command":
            for child in node.named_children:
                if child.type == "variable_name":
                    name = child.text.decode()
                    state.pop(name, None)
                    state.exported.discard(name)
                    state.conditional.discard(name)
            return _Flow()
        if node.type == "declaration_command":
            operator = node.children[0].type
            names = []
            for child in node.named_children:
                if child.type == "variable_assignment":
                    walk(child, state, context, read_context)
                    names.append(child.child_by_field_name("name").text.decode())
                elif child.type in {"variable_name", "word"} and re.fullmatch(r"[A-Za-z_]\w*", child.text.decode()):
                    names.append(child.text.decode())
                    if operator == 'local':
                        state[child.text.decode()] = (("", True),)
            if operator == 'local':
                state.local_names.update(names)
                return _Flow()
            if operator != 'export':
                return _Flow()
            if re.search(rb"\bexport\s+-f\b", node.text):
                return _Flow()  # Exported functions are not scalar env vars.
            for name in names:
                state.conditional.discard(name)
                if re.search(rb"\bexport\s+-n\b", node.text):
                    state.exported.discard(name)
                else:
                    state.exported.add(name)
            return _Flow()
        if node.type == "variable_assignment":
            name = node.child_by_field_name("name").text.decode()
            state[name] = values(node.child_by_field_name("value"), state, context, read_context)
            return _Flow()
        if node.type == 'test_command':
            # A syntactic test has an unknown exit status unless its predicate
            # was actually evaluated. Visiting its operands does not prove true.
            for child in node.named_children:
                walk(child, state, context, read_context)
            return _Flow()
        if node.type == 'negated_command':
            result = sequence(node.named_children, state, context, read_context, capture)
            return _Flow(result.failure, result.success, result.terminated)
        if node.type == "if_statement":
            context = context + '; ' + node.text.decode().splitlines()[0]
            branches, successes, failures, terminated = [], [], [], []
            branch_nodes = []
            has_else = False
            conditions = node.children_by_field_name('condition')
            sequence([child for child in conditions if child.is_named], state, context, read_context)
            for child in node.named_children:
                if child in conditions:
                    continue
                if child.type in {"elif_clause", "else_clause"}:
                    branch = state.copy()
                    result = sequence(branch_nodes, branch, context, read_context, capture)
                    successes.append(result.success);failures.append(result.failure);terminated.append(result.terminated)
                    if result.falls:branches.append(branch)
                    branch_nodes = list(child.named_children)
                    has_else |= child.type == "else_clause"
                else:
                    branch_nodes.append(child)
            branch = state.copy()
            result = sequence(branch_nodes, branch, context, read_context, capture)
            successes.append(result.success);failures.append(result.failure);terminated.append(result.terminated)
            if result.falls:branches.append(branch)
            if not has_else:
                branches.append(state.copy())
                successes.append(empty)
            if branches:merge(state, branches)
            return _Flow(choices(successes), choices(failures), choices(terminated))
        if node.type == "case_statement":
            branches, successes, failures, terminated = [state.copy()], [empty], [], []
            for child in node.named_children:
                if child.type == "case_item":
                    branch = state.copy()
                    result = sequence(child.named_children, branch, context, read_context, capture)
                    successes.append(result.success);failures.append(result.failure);terminated.append(result.terminated)
                    if result.falls:branches.append(branch)
            merge(state, branches)
            return _Flow(choices(successes), choices(failures), choices(terminated))
        if node.type in {"subshell", "command_substitution", "process_substitution"}:
            result = sequence(node.named_children, state.copy(), context, read_context, capture)
            output = choices([result.stdout, result.terminated])
            return _Flow(output, output)
        if node.type in {"while_statement", "for_statement"}:
            branch = state.copy()
            variable = node.child_by_field_name('variable')
            if variable is not None:
                branch[variable.text.decode()] = choices(values(child, state, context, read_context)
                                                         for child in node.children_by_field_name('value'))
            outputs = []
            for child in node.named_children:
                result, _ = walk(child, branch, context, read_context, capture)
                outputs.append(result)
            merge(state, [state.copy(), branch])
            output = choices([empty, *outputs])
            return _Flow(output, output)
        if node.type == "list":
            children = node.named_children
            current = walk(children[0], state, context, read_context, capture)
            operators = [child.type for child in node.children if child.type in {'&&', '||'}]
            for operator, child in zip(operators, children[1:]):
                running = current.success if operator == '&&' else current.failure
                skipped = current.failure if operator == '&&' else current.success
                if not running:
                    continue
                branch = state.copy()
                result = walk(child, branch, context, read_context, capture)
                success = combine([running, result.success]) if result.success else ()
                failure = combine([running, result.failure]) if result.failure else ()
                terminated = choices([current.terminated,
                    combine([running, result.terminated]) if result.terminated else ()])
                if operator == '&&':failure = choices([failure, skipped])
                else:success = choices([success, skipped])
                continuing = ([state.copy()] if skipped else []) + ([branch] if result.falls else [])
                if continuing:merge(state, continuing)
                current = _Flow(success, failure, terminated)
            return current
        if node.type == 'pipeline':
            output = empty
            for child in node.named_children:
                output, _ = walk(child, state.copy(), context, read_context, capture)
            return _Flow(output, output)
        if node.type == "command":
            local = state.copy()
            prefix = set()
            for child in node.named_children:
                if child.type == "variable_assignment":
                    walk(child, local, context, read_context)
                    prefix.add(child.child_by_field_name("name").text.decode())
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return _Flow()
            arguments = []
            for child in node.children_by_field_name('argument'):
                if child.text in {b'"$@"', b'"${@}"', b'$@'} and local.positionals is not None:
                    arguments.extend(local.positionals)
                else:
                    arguments.append(values(child, local, context, read_context))
            arguments = tuple(arguments)
            outputs = []
            for name, resolved in values(name_node, local, context, read_context):
                aliases, enabled = read_context
                expanded, bound_arguments = name, arguments
                if enabled and name_node.text.decode() == name and name in aliases:
                    replacement = Parser(Language(tree_sitter_bash.language())).parse(aliases[name].encode()).root_node
                    if len(replacement.named_children) == 1 and replacement.named_children[0].type == 'command':
                        alias_command = replacement.named_children[0]
                        alias_name = values(alias_command.child_by_field_name('name'), local, context, read_context)
                        if len(alias_name) == 1:
                            expanded = alias_name[0][0]
                            bound_arguments = tuple(values(child, local, context, read_context)
                                                    for child in alias_command.children_by_field_name('argument')) + arguments
                if expanded in local.functions:
                    # Only a real invocation enters the function's argument frame.
                    result = invoke(expanded, bound_arguments, local,
                        context + f' function {expanded} called at line {node.start_point.row+1}', capture, prefix)
                    outputs.append(result.stdout)
                    for key in set(state) | set(local):
                        if key not in prefix:
                            if key in local:state[key] = local[key]
                            else:state.pop(key, None)
                    state.exported = (local.exported-prefix) | (state.exported & prefix)
                    state.conditional = (local.conditional-prefix) | (state.conditional & prefix)
                    state.aliases, state.functions, state.expand_aliases = local.aliases, local.functions, local.expand_aliases
                    continue
                commands.append(ShellCommand(expanded, node.start_point.row + 1,
                                node.start_point.column, node.text.decode(), bound_arguments,
                                child_environment(local, prefix), context))
                if posixpath.basename(expanded) == 'timeout':
                    literals = [options[0][0] if len(options) == 1 and options[0][1] else None for options in bound_arguments]
                    index = timeout_command_index(literals)
                    if index is not None and literals[index] is not None:
                        commands.append(ShellCommand(literals[index], node.start_point.row+1, node.start_point.column,
                            node.text.decode(), bound_arguments[index+1:], child_environment(local, prefix),
                            context+' timeout executable operand'))
                if name == "alias":
                    for options in arguments:
                        for value, known in options:
                            if known and "=" in value:
                                key, replacement = value.split("=", 1)
                                state.aliases[key] = replacement
                if name == 'unalias':
                    for options in arguments:
                        for value, known in options:
                            if known:state.aliases.pop(value, None)
                if name == 'shopt':
                    literals = [value for options in arguments for value, known in options if known]
                    if 'expand_aliases' in literals:state.expand_aliases = '-s' in literals
                if name == "unset":
                    for options in arguments:
                        for value, known in options:
                            if known:
                                state.pop(value, None)
                if name == 'shift' and state.positionals is not None:
                    count = int(arguments[0][0][0]) if arguments and arguments[0][0][1] and arguments[0][0][0].isdigit() else 1
                    positional(state, state.positionals[count:])
                if name in {'return', 'exit', 'break', 'continue'}:
                    return _Flow((), (), empty)
                if capture:
                    if name == 'echo':
                        parts = [part for index, argument in enumerate(bound_arguments)
                                 for part in ([((" ", True),), argument] if index else [argument])]
                        outputs.append(combine([*parts, (("\n", True),)]))
                    elif name in {'dirname', 'basename'} and len(bound_arguments) == 1:
                        operation = posixpath.dirname if name == 'dirname' else posixpath.basename
                        outputs.append(tuple((operation(value) or '.', known) for value, known in bound_arguments[0]))
                    elif name not in {'alias', 'shopt', 'unset', 'cd', 'set', 'true', 'false', 'test', '[', 'shift'}:
                        outputs.append(((node.text.decode(), False),))
            output = choices(outputs) if outputs else empty
            if name in {'echo', 'printf', 'true', ':'}:
                return _Flow(output, ())
            if name == 'false':
                return _Flow((), output)
            return _Flow(output, output)
        if node.type == "file_redirect":
            destination = node.child_by_field_name("destination")
            if destination is not None:
                prefix = node.text.decode().split(destination.text.decode(), 1)[0]
                name = ">" if ">" in prefix and ">&" not in prefix else "<"
                commands.append(ShellCommand(name, node.start_point.row + 1,
                                node.start_point.column, node.text.decode(),
                                (values(destination, state, context, read_context),), child_environment(state), context))
            return _Flow()
        if node.type == 'redirected_statement':
            result = walk(node.child_by_field_name('body'), state, context, read_context, capture)
            for redirect in node.children_by_field_name('redirect'):
                walk(redirect, state, context, read_context)
                if b'>' in redirect.text and not redirect.text.lstrip().startswith(b'2>'):
                    return _Flow(empty if result.success else (), empty if result.failure else (),
                                 empty if result.terminated else ())
            return result
        if node.type == "heredoc_body":
            for child in node.named_children:
                if child.type in {"command_substitution", "process_substitution"}:
                    walk(child, state.copy(), context, read_context)
            return _Flow()
        return sequence(node.named_children, state, context, read_context, capture)

    incoming = entry_bindings or {}
    state = _ShellBindings(incoming, [name for name in incoming if re.fullmatch(r"[A-Za-z_]\w*", name)])
    if script_path is not None:
        state['0'] = state['BASH_SOURCE[0]'] = ((script_path, True),)
    sequence(root.named_children, state, '')
    # A sourceable helper without a visible caller is still observed, but its
    # unknown entry frame cannot mutate the script's or another helper's state.
    for identifier, (name, definition, read_context) in list(definitions.items()):
        if identifier not in visited:
            local = state.copy()
            local.functions[name] = (definition, read_context)
            invoke(name, None, local, 'unresolved function entry '+name, False)
    return commands
