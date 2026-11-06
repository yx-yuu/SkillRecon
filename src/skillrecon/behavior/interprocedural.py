"""Dispatch proven argv/env payloads across resolved package-local commands."""
from collections import deque
from pathlib import Path

from skillrecon.behavior.flow import recover_ast_flows
from skillrecon.behavior.javascript_flow import recover_javascript_flows
from skillrecon.behavior.shell_flow import recover_shell_flows


def recover_native_flows(events, resources, unit_paths, staged_root, bridges):
    pending, paths = deque(), []
    analyzers = [recover_ast_flows, recover_javascript_flows, recover_shell_flows]
    for analyze in analyzers:
        paths.extend(analyze(events, resources, unit_paths, staged_root, bridges, invocations=pending))
    by_bridge = {bridge.bridge_id: bridge for bridge in bridges}
    seen = set()
    while pending:
        payload = pending.popleft()
        key = (payload.bridge_id, payload.arguments, frozenset(payload.environment.items()))
        if key in seen:
            continue
        seen.add(key)
        if not any(payload.arguments) and not any(payload.environment.values()):
            continue  # No sensitive value crosses this invocation; local roots were analyzed above.
        target = by_bridge[payload.bridge_id].target_unit_id
        suffix = Path(unit_paths[target]).suffix
        analyze = (recover_ast_flows if suffix == '.py' else
                   recover_shell_flows if suffix in {'.sh', '.bash'} else
                   recover_javascript_flows if suffix in {'.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx'} else None)
        if analyze is not None:
            paths.extend(analyze(events, resources, unit_paths, staged_root, bridges,
                                 invocations=pending, entry=payload))
    unique = {}
    for path in paths:
        key = (tuple((segment.unit_id, segment.location, segment.event_id,
                      segment.slot_kind, segment.expr_hint) for segment in path.segments),
               tuple(path.bridges_used))
        unique.setdefault(key, path)
    return list(unique.values())
