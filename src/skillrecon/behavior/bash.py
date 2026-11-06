"""Bash command observations and parsed variable provenance."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_bash

from skillrecon.behavior.normalize import RawObservation
from skillrecon.core.types import PackageManifest
from skillrecon.behavior.shell import network_target_bindings, shell_commands

logger = logging.getLogger(__name__)

# Bash maintains these frame arrays and user IDs; they are not inherited environment
# inputs. https://www.gnu.org/software/bash/manual/html_node/Bash-Variables.html
_SHELL_INTERNAL_VARIABLES = {'BASH_SOURCE', 'BASH_LINENO', 'FUNCNAME', 'UID', 'EUID'}


def _file_targets(command, capability):
    operands=[choices for choices in command.arguments
              if not all(value.startswith('-') and known for value,known in choices)]
    if capability=='file_permission_change':
        operands=operands[1:]  # The first positional operand is the mode/owner/group.
    return tuple(dict.fromkeys((value,known) for choices in operands for value,known in choices))


def extract_bash_observations(
    manifest: PackageManifest,
    unit_paths: dict[str, str],
    staged_root: Path,
    pattern_path: Path,
) -> list[RawObservation]:
    """Extract capability observations with parsed Shell input provenance."""
    config = json.loads(pattern_path.read_text(encoding="utf-8"))
    pattern_map: dict[str, dict[str, list[str]]] = config.get("patterns", {})

    observations: list[RawObservation] = []
    bash_units = [unit for unit in manifest.code_units if unit.language == "bash"]

    for unit in bash_units:
        relative_path = unit_paths[unit.unit_id]
        file_path = staged_root / relative_path
        if not file_path.is_file():
            continue

        content = file_path.read_text(encoding="utf-8")
        environment_reads, heredoc_lines, heredoc_commands = _environment_references(content)
        for line_no, names in sorted(environment_reads.items()):
            observations.append(RawObservation(
                language="bash", relative_path=relative_path, line=line_no,
                message="capType=env_var_read | detail=shell_expansion | tier=bash_ast",
                fields={"capType": "env_var_read", "detail": "shell_expansion", "tier": "bash_ast",
                        "environmentNames": json.dumps(sorted(set(names)))},
                source_text=content.splitlines()[line_no - 1],
            ))
        logical_lines = [(command.line, command) for command in shell_commands(content, script_path=relative_path)]
        for index, (line_no, command) in enumerate(logical_lines):
            stripped = command.source
            if not stripped or stripped.startswith("#"):
                continue
            end_line = line_no + 1
            names = sorted({name for row in range(line_no, end_line)
                            for name in environment_reads.get(row, [])})

            for cap_type, spec in pattern_map.items():
                if cap_type == "env_var_read":
                    continue
                operation_text = stripped
                if not operation_text:
                    continue
                matched = False
                matched_command = _best_matching_command(command, spec.get("commands", []))
                if matched_command is not None:
                    bindings = (network_target_bindings(command) if cap_type == 'http_request'
                                and Path(command.name).name in {'curl', 'wget'} else
                                _file_targets(command,cap_type) if cap_type in {'file_delete','file_permission_change'} else ())
                    for binding in bindings or (None,):
                        fields = {
                            "capType": cap_type,
                            "detail": matched_command,
                            "tier": "bash_ast",
                            "environmentNames": json.dumps(names),
                            "shellArguments": json.dumps(command.arguments),
                            "branchContext": command.context,
                        }
                        if binding is not None:
                            fields.update(resourceType='url' if cap_type=='http_request' else 'path', resourceValue=binding[0],
                                resourceResolved=str(binding[1]).lower(), resourceOriginKind='shell_binding',
                                resourceOriginHint=command.context)
                        observations.append(
                            RawObservation(
                                language="bash",
                                relative_path=relative_path,
                                line=line_no,
                                message=(
                                    f"capType={cap_type} | detail={matched_command} | "
                                    "tier=bash_ast"
                                ),
                                fields=fields,
                                source_text=operation_text,
                            )
                        )
                    matched = True
                if matched:
                    continue

                for pattern in spec.get("patterns", []):
                    if cap_type == "file_write" and command.name in {">", "tee"}:
                        for binding in _file_targets(command,cap_type):
                            observations.append(RawObservation(
                                language="bash",
                                relative_path=relative_path,
                                line=line_no,
                                message=(
                                    f"capType={cap_type} | detail=shell_pattern | "
                                    f"tier=bash_ast | matchedPattern={pattern}"
                                ),
                                fields={
                                    "capType": cap_type,
                                    "detail": "shell_pattern",
                                    "matchedPattern": pattern,
                                    "tier": "bash_ast",
                                    "environmentNames": json.dumps(names),
                                    "shellArguments": json.dumps(command.arguments),
                                    "branchContext": command.context,
                                    "resourceType": "path", "resourceValue": binding[0],
                                    "resourceResolved": str(binding[1]).lower(),
                                    "resourceOriginKind": "shell_binding", "resourceOriginHint": command.context,
                                },
                                source_text=operation_text,
                            ))
                        break

    logger.info("Extracted %d Bash observations", len(observations))
    return observations


def _environment_references(content: str) -> tuple[dict[int, list[str]], set[int], dict[int, list[str]]]:
    """Locate external-name expansions using ordered assignments and Shell scopes."""
    tree = Parser(Language(tree_sitter_bash.language())).parse(content.encode("utf-8"))
    reads: list[tuple[int, str]] = []
    heredoc_lines: set[int] = set()
    heredoc_commands: dict[int, list[str]] = {}

    def collect_heredoc_commands(node) -> None:
        if node.type == "command":
            heredoc_commands.setdefault(node.start_point.row + 1, []).append(node.text.decode())
        for child in node.named_children:
            collect_heredoc_commands(child)

    def walk(node, bindings: dict[str, bool]) -> None:
        if node.type in {"simple_expansion", "expansion"}:
            operand = next((c for c in node.named_children if c.type in {"variable_name", "subscript"}), None)
            if operand is not None and operand.type == "subscript":
                operand = operand.child_by_field_name("name")
            name = operand.text.decode() if operand is not None else ""
            if name not in _SHELL_INTERNAL_VARIABLES and re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) and bindings.get(name, True):
                reads.append((node.start_point.row + 1, name))
            for child in node.named_children:
                if child.type != "variable_name":
                    walk(child, bindings)
            return
        if node.type == "variable_assignment":
            name = node.child_by_field_name("name").text.decode()
            start = len(reads)
            value = node.child_by_field_name("value")
            if value is not None:
                walk(value, bindings)
            # X="${X:-default}" keeps its same-name external configuration;
            # X="$HOME/path", X="$1" and X=$(command) create a local value.
            bindings[name] = any(read_name == name for _, read_name in reads[start:])
            return
        if node.type == "function_definition":
            walk(node.child_by_field_name("body"), bindings.copy())
            return
        if node.type in {"subshell", "command_substitution", "process_substitution"}:
            scope = bindings.copy()
            for child in node.named_children:
                walk(child, scope)
            return
        if node.type == "if_statement":
            branches = []
            branch = bindings.copy()
            has_else = False
            for child in node.named_children:
                if child.type in {"else_clause", "elif_clause"}:
                    branches.append(branch)
                    branch = bindings.copy()
                    has_else |= child.type == "else_clause"
                walk(child, branch)
            branches.append(branch)
            if not has_else:
                branches.append(bindings.copy())
            for name in set().union(*(b.keys() for b in branches)):
                bindings[name] = any(b.get(name, True) for b in branches)
            return
        if node.type in {"for_statement", "while_statement", "case_statement"}:
            for child in node.named_children:
                scope = bindings.copy() if child.type in {"do_group", "case_item"} else bindings
                if node.type == "for_statement" and child.type == "do_group":
                    scope[node.child_by_field_name("variable").text.decode()] = False
                walk(child, scope)
            return
        if node.type == "list":
            # The first side executes; &&/|| do not guarantee the second side.
            for index, child in enumerate(node.named_children):
                walk(child, bindings if index == 0 else bindings.copy())
            return
        if node.type == "heredoc_body":
            heredoc_lines.update(range(node.start_point.row + 1, node.end_point.row + 1))
            collect_heredoc_commands(node)
        if node.type == "command":
            for child in node.named_children:
                walk(child, bindings.copy() if child.type == "variable_assignment" else bindings)
            command = node.child_by_field_name("name").text.decode()
            arguments = [c.text.decode() for i, c in enumerate(node.named_children)
                         if node.field_name_for_named_child(i) == "argument"]
            if command == "printenv":
                names = [name for name in arguments if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name)]
                reads.extend((node.start_point.row + 1, name) for name in names or ["<dynamic-env-var>"])
            if command == "read":
                for name in arguments:
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                        bindings[name] = False
            return
        if node.type == "declaration_command":
            for child in node.named_children:
                walk(child, bindings)
                if child.type == "variable_name" and node.children[0].type == "local":
                    bindings[child.text.decode()] = False
            return
        for child in node.named_children:
            walk(child, bindings)

    walk(tree.root_node, {})
    grouped: dict[int, list[str]] = {}
    for line, name in reads:
        grouped.setdefault(line, []).append(name)
    return grouped, heredoc_lines, heredoc_commands


def _logical_lines(content: str):
    """Keep a backslash-continued command and its original starting line."""
    parts: list[str] = []
    start_line = 1
    for line_no, line in enumerate(content.splitlines(), start=1):
        if not parts:
            start_line = line_no
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            parts.append(stripped[:-1])
            continue
        parts.append(line)
        yield start_line, " ".join(parts)
        parts = []
    if parts:
        yield start_line, " ".join(parts)


def _best_matching_command(operation, commands: list[str]) -> str | None:
    # The parser already separated executable and argument positions. Searching
    # flattened argv text would treat quoted data (e.g. ". Download") as code.
    executable = operation.name if operation.name == '.' else Path(operation.name).name
    matches = []
    for command in commands:
        name, *arguments = command.split()
        if executable == name and len(operation.arguments) >= len(arguments) and all(
            any(value == token and resolved for value, resolved in choices)
            for token, choices in zip(arguments, operation.arguments)
        ):
            matches.append(command)
    if not matches:
        return None
    return max(matches, key=len)
