"""Bridge recovery, risk-path construction, and G_C materialization."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath
import re

from skillrecon.behavior.normalize_helpers import _normalize_relpath, _read_source_line
from skillrecon.behavior.normalize_types import RawObservation, RawPathResult
from skillrecon.core.enums import BridgeKind
from skillrecon.core.sensitivity import event_requires_authorization
from skillrecon.core.types import (
    Bridge,
    CapabilityEvent,
    DataFlowEdge,
    DataObject,
    GraphEdge,
    GraphNode,
    GraphObject,
    LocationNodeRecord,
    Operation,
    OrchestrationHypothesis,
    PackageManifest,
    PathSegment,
    ResourceUse,
    RiskPath,
    SinkEndpoint,
    SourceEndpoint,
)

_DEFAULT_WEAK_PATH_MAX_LINE_GAP = 8
_WEAK_PATH_MAX_LINE_GAP_BY_PAIR = {
    ("env_var_read", "http_request"): 16,
    ("file_read", "http_request"): 16,
    ("token_file_read", "http_request"): 16,
}
_SOURCE_CAPABILITIES = {
    "env_var_read",
    "token_file_read",
    "file_read",
    "file_write",
    "deserialization",
    "sql_exec",
}
_SINK_CAPABILITIES = {
    "http_request",
    "websocket",
    "smtp_send",
    "ssh_connect",
    "data_encode_send",
    "file_write",
    "shell_exec",
    "subprocess_spawn",
}

_NETWORK_CAPABILITIES = {
    "http_request",
    "websocket",
    "smtp_send",
    "ssh_connect",
    "data_encode_send",
    "dns_lookup",
    "socket_connect",
    "ftp_transfer",
}
_FILESYSTEM_CAPABILITIES = {
    "file_read",
    "file_write",
    "file_delete",
    "file_execute",
    "file_permission_change",
    "temp_file_create",
    "token_file_read",
}
_PROCESS_CAPABILITIES = {
    "shell_exec",
    "subprocess_spawn",
    "dynamic_import",
    "process_kill",
}
_CREDENTIAL_CAPABILITIES = {
    "env_var_read",
    "api_key_use",
    "credential_store_access",
    "keychain_access",
}


def detect_bridges(
    unit_paths: dict[str, str],
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    bridge_hints: list[RawObservation],
    staged_root: Path,
) -> list[Bridge]:
    """Recover static, command, and artifact bridges across code units."""
    bridges: dict[tuple[str, str, BridgeKind, str], Bridge] = {}
    unit_by_path = {
        _normalize_relpath(relative_path): unit_id
        for unit_id, relative_path in unit_paths.items()
    }

    for hint in bridge_hints:
        imported = hint.fields.get("imported")
        if not imported:
            continue
        source_path = _normalize_relpath(hint.relative_path)
        source_unit = unit_by_path.get(source_path)
        target_path = _resolve_import_target(hint.language, source_path, imported, unit_by_path)
        target_unit = unit_by_path.get(target_path) if target_path else None
        if source_unit and target_unit and source_unit != target_unit:
            evidence = (
                f"{hint.fields.get('import_type', 'import')} "
                f"{imported} at {source_path}:{hint.line}"
            )
            key = (source_unit, target_unit, BridgeKind.STATIC, evidence)
            bridges[key] = Bridge(
                bridge_id=f"b{len(bridges)}",
                source_unit_id=source_unit,
                target_unit_id=target_unit,
                kind=BridgeKind.STATIC,
                evidence=evidence,
            )

    events_by_unit: dict[str, list[CapabilityEvent]] = defaultdict(list)
    for event in events:
        events_by_unit[event.unit_id].append(event)

    known_paths = sorted(unit_paths.items(), key=lambda item: len(item[1]), reverse=True)
    for unit_id, unit_events in events_by_unit.items():
        for event in unit_events:
            if event.capability not in {"shell_exec", "subprocess_spawn"}:
                continue
            line_text = _read_source_line(staged_root, event.file_path, event.line)
            for target_unit_id, relative_path in known_paths:
                if target_unit_id == unit_id:
                    continue
                basename = PurePosixPath(relative_path).name
                if relative_path in line_text or basename in line_text:
                    evidence = f"command references {relative_path} at {event.location}"
                    key = (unit_id, target_unit_id, BridgeKind.COMMAND, evidence)
                    bridges[key] = Bridge(
                        bridge_id=f"b{len(bridges)}",
                        source_unit_id=unit_id,
                        target_unit_id=target_unit_id,
                        kind=BridgeKind.COMMAND,
                        evidence=evidence,
                        source_event_id=event.event_id,
                    )

    event_by_id = {event.event_id: event for event in events}
    writes: dict[str, list[tuple[str, str]]] = defaultdict(list)
    reads: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for resource in resources:
        if resource.resource_type != "path" or resource.event_id is None:
            continue
        matched_event = event_by_id.get(resource.event_id)
        if matched_event is None:
            continue
        if matched_event.capability == "file_write":
            writes[resource.value].append((resource.unit_id, matched_event.event_id))
        elif matched_event.capability == "file_read":
            reads[resource.value].append((resource.unit_id, matched_event.event_id))

    for path_value, write_units in writes.items():
        for source_unit, source_event_id in write_units:
            for target_unit, target_event_id in reads.get(path_value, []):
                if source_unit == target_unit:
                    continue
                evidence = f"shared artifact {path_value}"
                key = (source_unit, target_unit, BridgeKind.ARTIFACT, evidence)
                bridges[key] = Bridge(
                    bridge_id=f"b{len(bridges)}",
                    source_unit_id=source_unit,
                    target_unit_id=target_unit,
                    kind=BridgeKind.ARTIFACT,
                    evidence=evidence,
                    source_event_id=source_event_id,
                    target_event_id=target_event_id,
                )

    return list(bridges.values())


def build_risk_paths(
    events: list[CapabilityEvent],
    bridges: list[Bridge],
    skip_event_pairs: set[tuple[str, str]] | None = None,
    a_req: set[str] | None = None,
) -> list[RiskPath]:
    """Build conservative source-to-sink paths from normalized events."""
    unit_events: dict[str, list[CapabilityEvent]] = defaultdict(list)
    for event in events:
        if event.tier == "instruction":
            continue
        unit_events[event.unit_id].append(event)
    for event_list in unit_events.values():
        event_list.sort(key=lambda item: (item.line, item.location, item.event_id))
    event_by_id = {event.event_id: event for event in events}

    paths: list[RiskPath] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    skipped_pairs = skip_event_pairs or set()

    for _unit_id, event_list in unit_events.items():
        for source, sink in _select_intra_unit_pairs(event_list, skipped_pairs, a_req):
            bridge_ids: tuple[str, ...] = ()
            key = (source.event_id, sink.event_id, bridge_ids)
            if key in seen:
                continue
            seen.add(key)
            paths.append(
                RiskPath(
                    path_id=f"p{len(paths)}",
                    source=_event_segment(source),
                    sink=_event_segment(sink),
                    segments=[_event_segment(source), _event_segment(sink)],
                    bridges_used=list(bridge_ids),
                    evidence_level="weak",
                    path_kind="weak:intra_unit",
                )
            )

    for bridge_group in _group_bridges_for_paths(bridges):
        sample_bridge = bridge_group[0]
        bridge_ids = tuple(sorted(bridge.bridge_id for bridge in bridge_group))
        source_candidates = [
            event
            for event in unit_events.get(sample_bridge.source_unit_id, [])
            if event.capability in _SOURCE_CAPABILITIES
        ]
        sink_candidates = [
            event
            for event in unit_events.get(sample_bridge.target_unit_id, [])
            if event.capability in _SINK_CAPABILITIES
            and _sink_event_requires_attention(event, a_req)
        ]
        for source, sink in _select_bridge_pairs(
            source_candidates,
            sink_candidates,
            skipped_pairs,
            bridge_group=bridge_group,
            events_by_id=event_by_id,
        ):
            key = (source.event_id, sink.event_id, bridge_ids)
            if key in seen:
                continue
            seen.add(key)
            paths.append(
                RiskPath(
                    path_id=f"p{len(paths)}",
                    source=_event_segment(source),
                    sink=_event_segment(sink),
                    segments=[_event_segment(source), _event_segment(sink)],
                    bridges_used=list(bridge_ids),
                    evidence_level="bridge",
                    path_kind=f"bridge:{sample_bridge.kind.value}",
                )
            )

    return paths


def attach_orchestration_hypotheses(
    paths: list[RiskPath],
    orchestrations: list[OrchestrationHypothesis],
) -> list[RiskPath]:
    """Attach instruction-conditioned orchestration hints to matching paths."""
    if not paths or not orchestrations:
        return paths

    attached: list[RiskPath] = []
    for path in paths:
        matched_ids = _matching_orchestration_ids(path, orchestrations)
        if not matched_ids:
            attached.append(path)
            continue

        conditioned_units = {
            unit_id
            for orchestration in orchestrations
            if orchestration.hypothesis_id in matched_ids
            for unit_id in orchestration.target_unit_ids
        }
        attached.append(
            path.model_copy(
                update={
                    "source": _condition_segment(path.source, conditioned_units),
                    "sink": _condition_segment(path.sink, conditioned_units),
                    "segments": [
                        _condition_segment(segment, conditioned_units)
                        for segment in path.segments
                    ],
                    "orchestration_hypotheses": matched_ids,
                }
            )
        )

    return attached


def _select_intra_unit_pairs(
    event_list: list[CapabilityEvent],
    skipped_pairs: set[tuple[str, str]],
    a_req: set[str] | None,
) -> list[tuple[CapabilityEvent, CapabilityEvent]]:
    selected: dict[tuple[str, str], tuple[CapabilityEvent, CapabilityEvent]] = {}
    sources = [event for event in event_list if event.capability in _SOURCE_CAPABILITIES]
    sinks = [
        event
        for event in event_list
        if event.capability in _SINK_CAPABILITIES
        and _sink_event_requires_attention(event, a_req)
    ]
    for source in sources:
        for sink in sinks:
            if not _valid_intra_unit_pair(source, sink, skipped_pairs):
                continue
            key = (source.capability, sink.capability)
            current = selected.get(key)
            if current is None or _path_pair_rank(source, sink) < _path_pair_rank(*current):
                selected[key] = (source, sink)
    return list(selected.values())


def _group_bridges_for_paths(bridges: list[Bridge]) -> list[list[Bridge]]:
    grouped: dict[tuple[str, str, BridgeKind], list[Bridge]] = defaultdict(list)
    for bridge in bridges:
        grouped[(bridge.source_unit_id, bridge.target_unit_id, bridge.kind)].append(bridge)
    return list(grouped.values())


def _matching_orchestration_ids(
    path: RiskPath,
    orchestrations: list[OrchestrationHypothesis],
) -> list[str]:
    matched_ids = list(path.orchestration_hypotheses)
    seen = set(matched_ids)

    for orchestration in orchestrations:
        if (
            path.sink.unit_id in orchestration.target_unit_ids
            and orchestration.hypothesis_id not in seen
        ):
            seen.add(orchestration.hypothesis_id)
            matched_ids.append(orchestration.hypothesis_id)

    return matched_ids


def _select_bridge_pairs(
    source_candidates: list[CapabilityEvent],
    sink_candidates: list[CapabilityEvent],
    skipped_pairs: set[tuple[str, str]],
    *,
    bridge_group: list[Bridge],
    events_by_id: dict[str, CapabilityEvent],
) -> list[tuple[CapabilityEvent, CapabilityEvent]]:
    selected: dict[tuple[str, str], tuple[CapabilityEvent, CapabilityEvent]] = {}
    source_anchor_lines = _bridge_anchor_lines(bridge_group, events_by_id, use_target=False)
    target_anchor_lines = _bridge_anchor_lines(bridge_group, events_by_id, use_target=True)
    bridge_kind = bridge_group[0].kind
    for source in source_candidates:
        for sink in sink_candidates:
            if (source.event_id, sink.event_id) in skipped_pairs:
                continue
            if not _bridge_pair_allowed(
                source,
                sink,
                bridge_kind=bridge_kind,
                source_anchor_lines=source_anchor_lines,
                target_anchor_lines=target_anchor_lines,
            ):
                continue
            key = (source.capability, sink.capability)
            current = selected.get(key)
            if current is None or _bridge_pair_rank(
                source,
                sink,
                source_anchor_lines=source_anchor_lines,
                target_anchor_lines=target_anchor_lines,
            ) < _bridge_pair_rank(
                *current,
                source_anchor_lines=source_anchor_lines,
                target_anchor_lines=target_anchor_lines,
            ):
                selected[key] = (source, sink)
    return list(selected.values())


def _bridge_anchor_lines(
    bridge_group: list[Bridge],
    events_by_id: dict[str, CapabilityEvent],
    *,
    use_target: bool,
) -> list[int]:
    lines: list[int] = []
    for bridge in bridge_group:
        event_id = bridge.target_event_id if use_target else bridge.source_event_id
        if not event_id:
            continue
        event = events_by_id.get(event_id)
        if event is None or event.line <= 0:
            continue
        lines.append(event.line)
    return sorted(lines)


def _bridge_pair_allowed(
    source: CapabilityEvent,
    sink: CapabilityEvent,
    *,
    bridge_kind: BridgeKind,
    source_anchor_lines: list[int],
    target_anchor_lines: list[int],
) -> bool:
    if source_anchor_lines and source.line > 0:
        if not any(source.line <= anchor for anchor in source_anchor_lines):
            return False
        if bridge_kind == BridgeKind.COMMAND:
            best_source_gap = min(
                anchor - source.line
                for anchor in source_anchor_lines
                if source.line <= anchor
            )
            if best_source_gap > 80:
                return False
    if target_anchor_lines and sink.line > 0:
        if not any(sink.line >= anchor for anchor in target_anchor_lines):
            return False
    return True


def _bridge_pair_rank(
    source: CapabilityEvent,
    sink: CapabilityEvent,
    *,
    source_anchor_lines: list[int],
    target_anchor_lines: list[int],
) -> tuple[int, int, int, int, str, str]:
    if source_anchor_lines and source.line > 0:
        source_gap = min(
            anchor - source.line
            for anchor in source_anchor_lines
            if source.line <= anchor
        )
    else:
        source_gap = 10**9
    if target_anchor_lines and sink.line > 0:
        sink_gap = min(
            sink.line - anchor
            for anchor in target_anchor_lines
            if sink.line >= anchor
        )
    else:
        sink_gap = 10**9
    source_line = source.line if source.line > 0 else 10**9
    sink_line = sink.line if sink.line > 0 else 10**9
    return (source_gap, sink_gap, source_line, sink_line, source.event_id, sink.event_id)


def _valid_intra_unit_pair(
    source: CapabilityEvent,
    sink: CapabilityEvent,
    skipped_pairs: set[tuple[str, str]],
) -> bool:
    if source.event_id == sink.event_id:
        return False
    if source.line and sink.line and source.line > sink.line:
        return False
    if (
        source.line
        and sink.line
        and sink.line - source.line
        > _weak_path_max_line_gap(source.capability, sink.capability)
    ):
        return False
    return (source.event_id, sink.event_id) not in skipped_pairs


def _path_pair_rank(
    source: CapabilityEvent,
    sink: CapabilityEvent,
) -> tuple[int, int, str, str]:
    source_line = source.line if source.line > 0 else 10**9
    sink_line = sink.line if sink.line > 0 else 10**9
    return (source_line, sink_line, source.event_id, sink.event_id)


def normalize_codeql_paths(
    raw_paths: list[RawPathResult],
    unit_paths: dict[str, str],
    events: list[CapabilityEvent],
    a_req: set[str] | None = None,
) -> list[RiskPath]:
    """Convert CodeQL path-problem SARIF results into normalized risk paths."""
    if not raw_paths:
        return []

    path_to_units: dict[str, list[str]] = defaultdict(list)
    for unit_id, relative_path in unit_paths.items():
        path_to_units[_normalize_relpath(relative_path)].append(unit_id)

    event_index: dict[tuple[str, int, str], CapabilityEvent] = {}
    events_by_id: dict[str, CapabilityEvent] = {}
    for event in events:
        event_index[(event.file_path, event.line, event.capability)] = event
        events_by_id[event.event_id] = event

    paths: list[RiskPath] = []
    seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()

    for raw_path in raw_paths:
        source_capability = raw_path.fields.get("pathSource", "source")
        sink_capability = raw_path.fields.get("pathSink", "sink")
        path_kind = raw_path.fields.get("pathKind", f"codeql:{raw_path.language}")

        segments: list[PathSegment] = []
        segment_key: list[tuple[str, int]] = []

        for index, step in enumerate(raw_path.steps):
            normalized_path = _normalize_relpath(step.relative_path)
            unit_ids = path_to_units.get(normalized_path)
            if not unit_ids:
                continue
            unit_id = unit_ids[0]
            location = f"{normalized_path}:{step.line}" if step.line else normalized_path
            label = step.message or "flow"
            event_id: str | None = None

            if index == 0:
                label = source_capability
                event = event_index.get((normalized_path, step.line, source_capability))
                if event is not None:
                    event_id = event.event_id
                slot_kind = raw_path.fields.get("sourceSlotKind", "")
                symbol_hint = raw_path.fields.get("sourceSymbolHint", "")
                expr_hint = raw_path.fields.get("sourceExprHint", "")
            elif index == len(raw_path.steps) - 1:
                label = sink_capability
                event = event_index.get((normalized_path, step.line, sink_capability))
                if event is not None:
                    event_id = event.event_id
                slot_kind = raw_path.fields.get("sinkSlotKind", "")
                symbol_hint = raw_path.fields.get("sinkSymbolHint", "")
                expr_hint = raw_path.fields.get("sinkExprHint", "")
            else:
                slot_kind = ""
                symbol_hint = ""
                expr_hint = step.message or ""

            segments.append(
                PathSegment(
                    unit_id=unit_id,
                    location=location,
                    label=label,
                    event_id=event_id,
                    slot_kind=slot_kind,
                    symbol_hint=symbol_hint,
                    expr_hint=expr_hint,
                )
            )
            segment_key.append((normalized_path, step.line))

        if len(segments) < 2:
            continue

        dedupe_key = (path_kind, tuple(segment_key))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        path = RiskPath(
                path_id=f"p{len(paths)}",
                source=segments[0],
                sink=segments[-1],
                segments=segments,
                bridges_used=[],
                evidence_level="codeql",
                path_kind=path_kind,
            )
        if not _path_requires_attention(path, events_by_id, a_req):
            continue
        paths.append(path)

    return paths


def build_gc_artifact(
    manifest: PackageManifest,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    bridges: list[Bridge],
    paths: list[RiskPath],
    staged_root: Path | None = None,
) -> dict[str, object]:
    """Build a lightweight JSON-serializable behavior graph artifact."""
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    known_unit_ids: set[str] = set()
    operations, locations, sources, sinks = derive_gc_enrichment(events, resources, paths)
    data_objects, data_flow_edges = derive_data_object_graph(
        paths,
        events,
        resources,
        staged_root=staged_root,
    )

    for unit in manifest.code_units:
        known_unit_ids.add(unit.unit_id)
        nodes.append(
            GraphNode(
                node_id=unit.unit_id,
                kind="code_unit",
                attrs={"language": unit.language},
            )
        )
    for event in events:
        if event.unit_id not in known_unit_ids:
            known_unit_ids.add(event.unit_id)
            nodes.append(
                GraphNode(
                    node_id=event.unit_id,
                    kind="code_unit",
                    attrs={"language": event.language or "instruction"},
                )
            )
    for event in events:
        nodes.append(
            GraphNode(
                node_id=event.event_id,
                kind="capability_event",
                attrs={
                    "capability": event.capability,
                    "unit_id": event.unit_id,
                },
            )
        )
        edges.append(
            GraphEdge(
                kind="emits",
                source=event.unit_id,
                target=event.event_id,
            )
        )
    for operation in operations:
        nodes.append(
            GraphNode(
                node_id=operation.operation_id,
                kind="operation",
                attrs={
                    "operation_type": operation.operation_type,
                    "summary": operation.summary,
                    "unit_id": operation.unit_id,
                    "event_id": operation.event_id,
                },
            )
        )
        edges.append(
            GraphEdge(
                kind="contains_operation",
                source=operation.unit_id,
                target=operation.operation_id,
            )
        )
        edges.append(
            GraphEdge(
                kind="realizes_event",
                source=operation.operation_id,
                target=operation.event_id,
            )
        )
    for data_object in data_objects:
        nodes.append(
            GraphNode(
                node_id=data_object.object_id,
                kind="data_object",
                attrs={
                    "object_kind": data_object.object_kind,
                    "label": data_object.label,
                    "abstraction_level": data_object.abstraction_level,
                    "slot_kind": data_object.slot_kind,
                    "symbol_hint": data_object.symbol_hint,
                    "expr_hint": data_object.expr_hint,
                    "origin_kind": data_object.origin_kind,
                    "path_id": data_object.path_id,
                    "event_id": data_object.event_id,
                },
            )
        )
        if data_object.event_id is not None:
            if data_object.object_id.endswith("::source"):
                edge_kind = "produces"
            elif data_object.object_id.endswith("::sink"):
                edge_kind = "consumes"
            else:
                edge_kind = (
                    "produces"
                    if data_object.object_id.endswith("::step::0")
                    else "consumes"
                )
            edges.append(
                GraphEdge(
                    kind=edge_kind,
                    source=data_object.event_id,
                    target=data_object.object_id,
                )
            )
        if data_object.location_id:
            edges.append(
                GraphEdge(
                    kind="located_at",
                    source=data_object.object_id,
                    target=data_object.location_id,
                )
            )
    for resource in resources:
        nodes.append(
            GraphNode(
                node_id=resource.resource_id,
                kind="resource_use",
                attrs={
                    "resource_type": resource.resource_type,
                    "value": resource.value,
                    "unit_id": resource.unit_id,
                    "origin_kind": resource.origin_kind,
                },
            )
        )
        if resource.event_id:
            edges.append(
                GraphEdge(
                    kind="uses_resource",
                    source=resource.event_id,
                    target=resource.resource_id,
                )
            )
    for location in locations:
        nodes.append(
            GraphNode(
                node_id=location.location_id,
                kind="location",
                attrs={
                    "file_path": location.file_path,
                    "line": location.line,
                    "raw_location": location.raw_location,
                },
            )
        )
    for operation in operations:
        if operation.location_id:
            edges.append(
                GraphEdge(
                    kind="located_at",
                    source=operation.operation_id,
                    target=operation.location_id,
                )
            )
    for resource in resources:
        location_id = _location_id(resource.location)
        if location_id:
            edges.append(
                GraphEdge(
                    kind="located_at",
                    source=resource.resource_id,
                    target=location_id,
                )
            )
    for source_endpoint in sources:
        nodes.append(
            GraphNode(
                node_id=source_endpoint.source_id,
                kind="source",
                attrs={
                    "capability": source_endpoint.capability,
                    "label": source_endpoint.label,
                    "event_id": source_endpoint.event_id,
                    "object_id": source_endpoint.object_id,
                },
            )
        )
        edges.append(
            GraphEdge(
                kind="classified_as_source",
                source=source_endpoint.object_id or source_endpoint.event_id,
                target=source_endpoint.source_id,
            )
        )
    for sink_endpoint in sinks:
        nodes.append(
            GraphNode(
                node_id=sink_endpoint.sink_id,
                kind="sink",
                attrs={
                    "capability": sink_endpoint.capability,
                    "label": sink_endpoint.label,
                    "event_id": sink_endpoint.event_id,
                    "object_id": sink_endpoint.object_id,
                },
            )
        )
        edges.append(
            GraphEdge(
                kind="classified_as_sink",
                source=sink_endpoint.object_id or sink_endpoint.event_id,
                target=sink_endpoint.sink_id,
            )
        )
    for flow_edge in data_flow_edges:
        edges.append(
            GraphEdge(
                kind="flows_to",
                source=flow_edge.source_object_id,
                target=flow_edge.target_object_id,
                attrs={
                    "edge_id": flow_edge.edge_id,
                    "flow_kind": flow_edge.flow_kind,
                    "evidence_level": flow_edge.evidence_level,
                    "path_id": flow_edge.path_id,
                },
            )
        )
    for bridge in bridges:
        edges.append(
            GraphEdge(
                kind=f"bridge:{bridge.kind.value}",
                source=bridge.source_unit_id,
                target=bridge.target_unit_id,
                attrs={"bridge_id": bridge.bridge_id},
            )
        )
    for path in paths:
        edges.append(
            GraphEdge(
                kind=f"path:{path.path_kind}",
                source=path.source.event_id or path.source.unit_id,
                target=path.sink.event_id or path.sink.unit_id,
                attrs={
                    "path_id": path.path_id,
                    "evidence_level": path.evidence_level,
                },
            )
        )
        source_id = f"src::{path.source.event_id}" if path.source.event_id else None
        sink_id = f"sink::{path.sink.event_id}" if path.sink.event_id else None
        if source_id and sink_id:
            edges.append(
                GraphEdge(
                    kind="reaches",
                    source=source_id,
                    target=sink_id,
                    attrs={
                        "path_id": path.path_id,
                        "evidence_level": path.evidence_level,
                    },
                )
            )

    return GraphObject(nodes=nodes, edges=edges).to_wire()


def derive_gc_enrichment(
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    paths: list[RiskPath],
) -> tuple[list[Operation], list[LocationNodeRecord], list[SourceEndpoint], list[SinkEndpoint]]:
    """Derive first-stage richer G_C objects from existing behavior observations."""
    operations: list[Operation] = []
    locations_by_id: dict[str, LocationNodeRecord] = {}
    sources: list[SourceEndpoint] = []
    sinks: list[SinkEndpoint] = []
    source_event_ids: set[str] = set()
    sink_event_ids: set[str] = set()
    source_index_by_event: dict[str, int] = {}
    sink_index_by_event: dict[str, int] = {}

    for event in events:
        location_id = _location_id(event.location)
        if location_id:
            locations_by_id.setdefault(
                location_id,
                _location_record(location_id, event.location, fallback_path=event.file_path, fallback_line=event.line),
            )
        operations.append(
            Operation(
                operation_id=f"op::{event.event_id}",
                unit_id=event.unit_id,
                event_id=event.event_id,
                operation_type=_operation_type(event),
                summary=event.api_call or event.detail or event.capability,
                location_id=location_id,
            )
        )
        if event.capability in _SOURCE_CAPABILITIES and event.event_id not in source_event_ids:
            source_event_ids.add(event.event_id)
            source_index_by_event[event.event_id] = len(sources)
            sources.append(
                SourceEndpoint(
                    source_id=f"src::{event.event_id}",
                    event_id=event.event_id,
                    capability=event.capability,
                    label=event.capability,
                )
            )
        if event.capability in _SINK_CAPABILITIES and event.event_id not in sink_event_ids:
            sink_event_ids.add(event.event_id)
            sink_index_by_event[event.event_id] = len(sinks)
            sinks.append(
                SinkEndpoint(
                    sink_id=f"sink::{event.event_id}",
                    event_id=event.event_id,
                    capability=event.capability,
                    label=event.capability,
                )
            )

    for resource in resources:
        location_id = _location_id(resource.location)
        if location_id:
            locations_by_id.setdefault(
                location_id,
                _location_record(location_id, resource.location),
            )

    object_id_by_segment_key: dict[tuple[str, str, str | None], str] = {}
    for path in paths:
        for index, segment in enumerate(path.segments):
            object_id_by_segment_key[(path.path_id, segment.location, segment.event_id)] = (
                f"obj::{path.path_id}::step::{index}"
            )

    for path in paths:
        for segment in path.segments:
            location_id = _location_id(segment.location)
            if location_id:
                locations_by_id.setdefault(
                    location_id,
                    _location_record(location_id, segment.location),
                )
            if segment.event_id and segment.label in _SOURCE_CAPABILITIES and segment.event_id not in source_event_ids:
                source_event_ids.add(segment.event_id)
                source_index_by_event[segment.event_id] = len(sources)
                sources.append(
                    SourceEndpoint(
                        source_id=f"src::{segment.event_id}",
                        event_id=segment.event_id,
                        object_id=object_id_by_segment_key.get(
                            (path.path_id, segment.location, segment.event_id)
                        ),
                        capability=segment.label,
                        label=segment.label,
                    )
                )
            elif segment.event_id and segment.label in _SOURCE_CAPABILITIES:
                existing_index = source_index_by_event.get(segment.event_id)
                if existing_index is not None and sources[existing_index].object_id is None:
                    sources[existing_index] = sources[existing_index].model_copy(
                        update={
                            "object_id": object_id_by_segment_key.get(
                                (path.path_id, segment.location, segment.event_id)
                            )
                        }
                    )
            if segment.event_id and segment.label in _SINK_CAPABILITIES and segment.event_id not in sink_event_ids:
                sink_event_ids.add(segment.event_id)
                sink_index_by_event[segment.event_id] = len(sinks)
                sinks.append(
                    SinkEndpoint(
                        sink_id=f"sink::{segment.event_id}",
                        event_id=segment.event_id,
                        object_id=object_id_by_segment_key.get(
                            (path.path_id, segment.location, segment.event_id)
                        ),
                        capability=segment.label,
                        label=segment.label,
                    )
                )
            elif segment.event_id and segment.label in _SINK_CAPABILITIES:
                existing_index = sink_index_by_event.get(segment.event_id)
                if existing_index is not None and sinks[existing_index].object_id is None:
                    sinks[existing_index] = sinks[existing_index].model_copy(
                        update={
                            "object_id": object_id_by_segment_key.get(
                                (path.path_id, segment.location, segment.event_id)
                            )
                        }
                    )

    return operations, list(locations_by_id.values()), sources, sinks


def derive_data_object_graph(
    paths: list[RiskPath],
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    *,
    staged_root: Path | None = None,
) -> tuple[list[DataObject], list[DataFlowEdge]]:
    """Lift existing RiskPath objects into an explicit data-object graph."""
    events_by_id = {event.event_id: event for event in events}
    resources_by_event: dict[str, list[ResourceUse]] = defaultdict(list)
    for resource in resources:
        if resource.event_id is not None:
            resources_by_event[resource.event_id].append(resource)

    data_objects: list[DataObject] = []
    data_flow_edges: list[DataFlowEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    for path in paths:
        object_ids: list[str] = []
        for index, segment in enumerate(path.segments):
            object_id = f"obj::{path.path_id}::step::{index}"
            object_ids.append(object_id)
            if object_id in seen_nodes:
                continue
            seen_nodes.add(object_id)
            event = events_by_id.get(segment.event_id or "")
            segment_resources = resources_by_event.get(segment.event_id or "", [])
            origin_kind, origin_hint = _segment_origin(segment_resources)
            abstraction_level, slot_kind, symbol_hint, expr_hint = _path_step_refinement(
                path_segments=path.segments,
                segment=segment,
                event=event,
                resources=segment_resources,
                staged_root=staged_root,
                segment_index=index,
                segment_count=len(path.segments),
            )
            data_objects.append(
                DataObject(
                    object_id=object_id,
                    unit_id=segment.unit_id,
                    location_id=_location_id(segment.location),
                    event_id=segment.event_id,
                    path_id=path.path_id,
                    object_kind=_data_object_kind_for_segment(
                        segment=segment,
                        event=event,
                        segment_index=index,
                        segment_count=len(path.segments),
                    ),
                    label=segment.label,
                    abstraction_level=abstraction_level,
                    slot_kind=slot_kind,
                    symbol_hint=symbol_hint,
                    expr_hint=expr_hint,
                    origin_kind=origin_kind,
                    origin_hint=origin_hint,
                )
            )
        if not object_ids:
            continue

        source_segment = path.segments[0]
        source_event = events_by_id.get(source_segment.event_id or "")
        source_resources = resources_by_event.get(source_segment.event_id or "", [])
        source_object = _semantic_source_object(
            path=path,
            segment=source_segment,
            event=source_event,
            resources=source_resources,
            staged_root=staged_root,
        )
        if source_object is not None and source_object.object_id not in seen_nodes:
            seen_nodes.add(source_object.object_id)
            data_objects.append(source_object)
            data_flow_edges.append(
                DataFlowEdge(
                    edge_id=f"flow::{path.path_id}::source",
                    source_object_id=source_object.object_id,
                    target_object_id=object_ids[0],
                    flow_kind="endpoint_binding",
                    evidence_level=path.evidence_level,
                    path_id=path.path_id,
                )
            )

        sink_segment = path.segments[-1]
        sink_event = events_by_id.get(sink_segment.event_id or "")
        sink_resources = resources_by_event.get(sink_segment.event_id or "", [])
        sink_object = _semantic_sink_object(
            path=path,
            segment=sink_segment,
            event=sink_event,
            resources=sink_resources,
            staged_root=staged_root,
        )
        if sink_object is not None and sink_object.object_id not in seen_nodes:
            seen_nodes.add(sink_object.object_id)
            data_objects.append(sink_object)
            data_flow_edges.append(
                DataFlowEdge(
                    edge_id=f"flow::{path.path_id}::sink",
                    source_object_id=object_ids[-1],
                    target_object_id=sink_object.object_id,
                    flow_kind="endpoint_binding",
                    evidence_level=path.evidence_level,
                    path_id=path.path_id,
                )
            )

        for source_object_id, target_object_id in zip(object_ids, object_ids[1:], strict=False):
            edge_key = (source_object_id, target_object_id, path.path_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            data_flow_edges.append(
                DataFlowEdge(
                    edge_id=f"flow::{path.path_id}::{len(data_flow_edges)}",
                    source_object_id=source_object_id,
                    target_object_id=target_object_id,
                    flow_kind="path_step",
                    evidence_level=path.evidence_level,
                    path_id=path.path_id,
                )
            )

    return data_objects, data_flow_edges


def _sink_event_requires_attention(
    event: CapabilityEvent,
    a_req: set[str] | None,
) -> bool:
    if not a_req:
        return True
    return event_requires_authorization(event, a_req)


def _path_requires_attention(
    path: RiskPath,
    events_by_id: dict[str, CapabilityEvent],
    a_req: set[str] | None,
) -> bool:
    if not a_req:
        return True
    sink_event_id = path.sink.event_id
    if sink_event_id and sink_event_id in events_by_id:
        return event_requires_authorization(events_by_id[sink_event_id], a_req)
    return path.sink.label in a_req


def _weak_path_max_line_gap(source_capability: str, sink_capability: str) -> int:
    return _WEAK_PATH_MAX_LINE_GAP_BY_PAIR.get(
        (source_capability, sink_capability),
        _DEFAULT_WEAK_PATH_MAX_LINE_GAP,
    )


def _resolve_import_target(
    language: str,
    source_path: str,
    imported: str,
    unit_by_path: dict[str, str],
) -> str | None:
    if language == "python":
        module_name = imported.split(".")[-1]
        candidates = {
            path for path in unit_by_path if PurePosixPath(path).stem == module_name
        }
        return sorted(candidates)[0] if candidates else None

    source_parent = PurePosixPath(source_path).parent
    raw = PurePosixPath(imported)
    if not imported.startswith("."):
        candidates = {
            path for path in unit_by_path if PurePosixPath(path).stem == raw.name
        }
        return sorted(candidates)[0] if candidates else None

    base = (source_parent / raw).as_posix()
    import_candidates = [
        base,
        f"{base}.js",
        f"{base}.ts",
        f"{base}.mjs",
        f"{base}/index.js",
        f"{base}/index.ts",
    ]
    for candidate in import_candidates:
        normalized = _normalize_relpath(candidate)
        if normalized in unit_by_path:
            return normalized
    return None


def _event_segment(event: CapabilityEvent) -> PathSegment:
    return PathSegment(
        unit_id=event.unit_id,
        location=event.location,
        label=event.capability,
        event_id=event.event_id,
    )


def _data_object_kind_for_segment(
    *,
    segment: PathSegment,
    event: CapabilityEvent | None,
    segment_index: int,
    segment_count: int,
) -> str:
    if event is None:
        return "flow_step"
    capability = event.capability
    if segment_index == 0:
        if capability == "env_var_read":
            return "env_value"
        if capability in {"file_read", "token_file_read"}:
            return "file_content"
    if segment_index == segment_count - 1:
        if capability in _NETWORK_CAPABILITIES:
            return "request_payload"
        if capability in {"file_write", "temp_file_create"}:
            return "written_content"
        if capability in {"shell_exec", "subprocess_spawn"}:
            return "command_payload"
    if capability == "config_file":
        return "config_value"
    return "flow_step"


def _segment_origin(resources: list[ResourceUse]) -> tuple[str, str]:
    for resource in resources:
        if resource.origin_kind != "unknown":
            return resource.origin_kind, resource.origin_hint
    return "derived", ""


def _semantic_source_object(
    *,
    path: RiskPath,
    segment: PathSegment,
    event: CapabilityEvent | None,
    resources: list[ResourceUse],
    staged_root: Path | None,
) -> DataObject | None:
    if event is None:
        return None
    object_kind = _semantic_source_kind(event, resources)
    if object_kind is None:
        return None
    origin_kind, origin_hint = _segment_origin(resources)
    _abstraction_level, slot_kind, symbol_hint, expr_hint = _path_step_refinement(
        path_segments=path.segments,
        segment=segment,
        event=event,
        resources=resources,
        staged_root=staged_root,
        segment_index=0,
        segment_count=1,
    )
    return DataObject(
        object_id=f"obj::{path.path_id}::source",
        unit_id=segment.unit_id,
        location_id=_location_id(segment.location),
        event_id=segment.event_id,
        path_id=path.path_id,
        object_kind=object_kind,
        label=segment.label,
        abstraction_level="semantic_endpoint",
        slot_kind=slot_kind,
        symbol_hint=symbol_hint,
        expr_hint=expr_hint,
        origin_kind=origin_kind,
        origin_hint=origin_hint,
    )


def _semantic_sink_object(
    *,
    path: RiskPath,
    segment: PathSegment,
    event: CapabilityEvent | None,
    resources: list[ResourceUse],
    staged_root: Path | None,
) -> DataObject | None:
    if event is None:
        return None
    object_kind = _semantic_sink_kind(event, resources)
    if object_kind is None:
        return None
    origin_kind, origin_hint = _segment_origin(resources)
    _abstraction_level, slot_kind, symbol_hint, expr_hint = _path_step_refinement(
        path_segments=path.segments,
        segment=segment,
        event=event,
        resources=resources,
        staged_root=staged_root,
        segment_index=0,
        segment_count=1,
    )
    return DataObject(
        object_id=f"obj::{path.path_id}::sink",
        unit_id=segment.unit_id,
        location_id=_location_id(segment.location),
        event_id=segment.event_id,
        path_id=path.path_id,
        object_kind=object_kind,
        label=segment.label,
        abstraction_level="semantic_endpoint",
        slot_kind=slot_kind,
        symbol_hint=symbol_hint,
        expr_hint=expr_hint,
        origin_kind=origin_kind,
        origin_hint=origin_hint,
    )


def _semantic_source_kind(
    event: CapabilityEvent,
    resources: list[ResourceUse],
) -> str | None:
    capability = event.capability
    if capability == "env_var_read":
        return "env_value"
    if capability in {"file_read", "token_file_read"}:
        if any(resource.origin_kind == "config_file" for resource in resources):
            return "config_value"
        return "file_content"
    if capability in {"api_key_use", "credential_store_access"}:
        return "credential_value"
    return None


def _semantic_sink_kind(
    event: CapabilityEvent,
    resources: list[ResourceUse],
) -> str | None:
    capability = event.capability
    if capability in _NETWORK_CAPABILITIES:
        if any(resource.resource_type in {"url", "domain"} for resource in resources):
            return "request_payload"
        return "network_payload"
    if capability in {"file_write", "temp_file_create"}:
        return "written_content"
    if capability in {"shell_exec", "subprocess_spawn"}:
        return "command_payload"
    return None


def _path_step_refinement(
    *,
    path_segments: list[PathSegment],
    segment: PathSegment,
    event: CapabilityEvent | None,
    resources: list[ResourceUse],
    staged_root: Path | None,
    segment_index: int,
    segment_count: int,
) -> tuple[str, str, str, str]:
    if segment.slot_kind or segment.symbol_hint or segment.expr_hint:
        symbol_hint = segment.symbol_hint or _path_successor_symbol_hint(
            path_segments=path_segments,
            segment_index=segment_index,
            current_location=segment.location,
        )
        return (
            _abstraction_from_segment_metadata(segment.slot_kind, symbol_hint),
            segment.slot_kind,
            symbol_hint,
            segment.expr_hint,
        )

    if event is None or event.language != "python":
        return "path_step", "", "", ""

    source_line = _event_source_line(event, staged_root)
    if not source_line:
        return "path_step", "", "", ""

    expr_hint = source_line.strip()
    capability = event.capability

    if segment_index == 0:
        symbol_hint = _assigned_symbol(source_line)
        if not symbol_hint:
            symbol_hint = _path_successor_symbol_hint(
                path_segments=path_segments,
                segment_index=segment_index,
                current_location=segment.location,
            )
        if capability == "env_var_read":
            return (
                "exact_symbol" if symbol_hint else "expression_slot",
                "env_access",
                symbol_hint,
                expr_hint,
            )
        if capability in {"file_read", "token_file_read"}:
            return (
                "exact_symbol" if symbol_hint else "expression_slot",
                "file_read_result",
                symbol_hint,
                expr_hint,
            )
        if capability in {"api_key_use", "credential_store_access"}:
            return (
                "exact_symbol" if symbol_hint else "expression_slot",
                "credential_access",
                symbol_hint,
                expr_hint,
            )

    if segment_index == segment_count - 1:
        slot_kind, symbol_hint = _sink_slot_refinement(capability, source_line)
        if slot_kind:
            abstraction_level = "argument_slot" if symbol_hint else "expression_slot"
            return abstraction_level, slot_kind, symbol_hint, expr_hint

    return "path_step", "", "", expr_hint


def _abstraction_from_segment_metadata(slot_kind: str, symbol_hint: str) -> str:
    if slot_kind.endswith("_arg"):
        return "argument_slot" if symbol_hint else "expression_slot"
    if slot_kind in {"env_access", "file_read_result", "credential_access"}:
        return "exact_symbol" if symbol_hint else "expression_slot"
    if slot_kind:
        return "expression_slot"
    return "path_step"


def _path_successor_symbol_hint(
    *,
    path_segments: list[PathSegment],
    segment_index: int,
    current_location: str,
) -> str:
    current_line = _line_from_location(current_location)
    fallback = ""
    for offset in range(1, min(len(path_segments) - segment_index, 5)):
        label = path_segments[segment_index + offset].label
        match = re.fullmatch(r"ControlFlowNode for ([A-Za-z_][A-Za-z0-9_]*)", label)
        if match is None:
            continue
        candidate = match.group(1)
        if not fallback:
            fallback = candidate
        candidate_line = _line_from_location(path_segments[segment_index + offset].location)
        if current_line is not None and candidate_line is not None and candidate_line > current_line:
            return candidate
    return fallback


def _line_from_location(location: str) -> int | None:
    if not location:
        return None
    path_part, _, line_part = location.rpartition(":")
    if path_part and line_part.isdigit():
        return int(line_part)
    return None


def _event_source_line(
    event: CapabilityEvent,
    staged_root: Path | None,
) -> str:
    if staged_root is not None and event.file_path and event.line:
        return _read_source_line(staged_root, event.file_path, event.line)
    parts = [event.detail, event.api_call, *event.arguments]
    return " ".join(part for part in parts if part)


def _assigned_symbol(text: str) -> str:
    match = re.search(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", text)
    if match is None:
        return ""
    return match.group(1)


def _sink_slot_refinement(
    capability: str,
    text: str,
) -> tuple[str, str]:
    if capability in _NETWORK_CAPABILITIES:
        for slot_name in ("data", "json", "content", "body", "params"):
            match = re.search(rf"{slot_name}\s*=\s*([A-Za-z_][A-Za-z0-9_]*)", text)
            if match is not None:
                return "http_body_arg", match.group(1)
        match = re.search(
            r"(?:requests|httpx)\.[A-Za-z_][A-Za-z0-9_]*\([^,\n]+,\s*([A-Za-z_][A-Za-z0-9_]*)",
            text,
        )
        if match is not None:
            return "http_body_arg", match.group(1)
        return "http_request_call", ""
    if capability in {"shell_exec", "subprocess_spawn"}:
        match = re.search(
            r"(?:system|run|Popen|call|check_call|check_output)\(\s*([A-Za-z_][A-Za-z0-9_]*)",
            text,
        )
        if match is not None:
            return "command_arg", match.group(1)
        return "command_call", ""
    if capability in {"file_write", "temp_file_create"}:
        match = re.search(r"\.write\(\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        if match is not None:
            return "file_write_arg", match.group(1)
        match = re.search(r"dump\(\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        if match is not None:
            return "file_write_arg", match.group(1)
        return "file_write_call", ""
    return "", ""


def _location_id(raw_location: str) -> str | None:
    if not raw_location:
        return None
    return f"loc::{raw_location}"


def _location_record(
    location_id: str,
    raw_location: str,
    *,
    fallback_path: str = "",
    fallback_line: int = 0,
) -> LocationNodeRecord:
    path_part, _, line_part = raw_location.rpartition(":")
    if path_part and line_part.isdigit():
        return LocationNodeRecord(
            location_id=location_id,
            file_path=path_part,
            line=int(line_part),
            raw_location=raw_location,
        )
    return LocationNodeRecord(
        location_id=location_id,
        file_path=fallback_path or raw_location,
        line=fallback_line or None,
        raw_location=raw_location,
    )


def _operation_type(event: CapabilityEvent) -> str:
    capability = event.capability
    if capability in _NETWORK_CAPABILITIES:
        return "network_operation"
    if capability in _FILESYSTEM_CAPABILITIES:
        return "filesystem_operation"
    if capability in _PROCESS_CAPABILITIES:
        return "process_operation"
    if capability in _CREDENTIAL_CAPABILITIES:
        return "credential_operation"
    return "capability_operation"


def _condition_segment(
    segment: PathSegment,
    conditioned_units: set[str],
) -> PathSegment:
    if segment.unit_id not in conditioned_units:
        return segment
    return segment.model_copy(update={"is_orchestration_conditioned": True})
