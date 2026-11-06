"""Normalize CodeQL SARIF and Bash observations into behavior-side objects."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path, PurePosixPath

from skillrecon.behavior.normalize_helpers import _normalize_relpath, _read_source_line
from skillrecon.behavior.normalize_types import RawObservation, RawPathResult, RawPathStep
from skillrecon.behavior.path_graph import (
    build_gc_artifact,
    build_risk_paths,
    detect_bridges,
    derive_data_object_graph,
    derive_gc_enrichment,
    normalize_codeql_paths,
)
from skillrecon.behavior.resource_extract import _extract_resources, _refine_capability
from skillrecon.core.types import CapabilityEvent, PackageManifest, ResourceUse

logger = logging.getLogger(__name__)

_TIER_PRIORITY = {
    "concepts": 0,
    "api_graph": 1,
    "bash_pattern": 2,
    "bash_ast": 2,
    "instruction": 3,
}


def load_taxonomy_atoms(taxonomy_path: Path) -> set[str]:
    """Load all capability atoms from taxonomy v2."""
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    atoms: set[str] = set()
    for category in data.get("categories", {}).values():
        atoms.update(category.get("atoms", []))
    return atoms


def parse_structured_message(message: str) -> dict[str, str]:
    """Parse CodeQL/Bash message fields formatted as k=v | k=v."""
    fields: dict[str, str] = {}
    for chunk in message.split("|"):
        piece = chunk.strip()
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def load_sarif_observations(sarif_path: Path, language: str) -> list[RawObservation]:
    """Load CodeQL SARIF results into raw observations."""
    data = json.loads(sarif_path.read_text(encoding="utf-8"))
    observations: list[RawObservation] = []

    for run in data.get("runs", []):
        artifacts = run.get("artifacts", [])
        for result in run.get("results", []):
            message = result.get("message", {}).get("text", "")
            fields = parse_structured_message(message)
            if result.get("ruleId") == "skillrecon/javascript/unsafe-command-construction":
                # This query refines shell execution with an input-flow proof;
                # it does not introduce an independent eval permission demand.
                fields["capType"] = "shell_exec"
                fields["unsafeCommandConstruction"] = "true"
            if "pathKind" in fields:
                continue
            location = _first_location(result)
            if location is None:
                continue
            relative_path = _resolve_artifact_uri(location, artifacts)
            physical = location.get("physicalLocation", {})
            if isinstance(physical, dict):
                region = physical.get("region", {})
                line = region.get("startLine", 0) if isinstance(region, dict) else 0
            else:
                line = 0
            observations.append(
                RawObservation(
                    language=language,
                    relative_path=relative_path,
                    line=line,
                    message=message,
                    fields=fields,
                )
            )

    logger.info("Loaded %d SARIF observations from %s", len(observations), sarif_path)
    return observations


def load_sarif_paths(sarif_path: Path, language: str) -> list[RawPathResult]:
    """Read every CodeQL flow with its own row's endpoint metadata."""
    data = json.loads(sarif_path.read_text(encoding="utf-8"))
    paths = []

    def site(location, artifacts):
        physical = location['physicalLocation']
        region = physical['region']
        return f"{_normalize_relpath(_resolve_artifact_uri(location, artifacts))}:{region['startLine']}:{region.get('startColumn', 1)}"

    identity = ['pathSource', 'pathSink', 'pathKind', 'sourceSlotKind', 'sinkSlotKind']
    for run in data.get('runs', []):
        artifacts = run.get('artifacts', [])
        for result in run.get('results', []):
            message = result.get('message', {}).get('text', '')
            rows = [parse_structured_message(line) for line in message.splitlines()
                    if 'pathKind=' in line]
            if not rows:
                continue
            located = all('sourceSite' in row and 'sinkSite' in row for row in rows)
            if not located and len({tuple(row.get(key) for key in identity) for row in rows}) != 1:
                raise ValueError('Grouped SARIF has ambiguous path metadata; rerun the current query pack with sourceSite/sinkSite')
            for code_flow in result.get('codeFlows', []):
                for thread in code_flow.get('threadFlows', []):
                    locations = [item['location'] for item in thread.get('locations', []) if 'location' in item]
                    if len(locations) < 2:
                        continue
                    matching = ([row for row in rows if row['sourceSite'] == site(locations[0], artifacts)
                                 and row['sinkSite'] == site(locations[-1], artifacts)] if located else rows)
                    if not matching:
                        raise ValueError('CodeQL flow endpoint locations do not match any result metadata row')
                    steps = []
                    for location in locations:
                        relative = _resolve_artifact_uri(location, artifacts)
                        if relative:
                            steps.append(RawPathStep(relative_path=relative,
                                line=location['physicalLocation']['region'].get('startLine', 0),
                                message=location.get('message', {}).get('text', '')))
                    if len(steps) < 2:
                        continue
                    for fields in matching:
                        paths.append(RawPathResult(language=language, message=message,
                                                   fields=fields, steps=steps))
    logger.info("Loaded %d SARIF paths from %s", len(paths), sarif_path)
    return paths


def normalize_observations(
    observations: list[RawObservation],
    manifest: PackageManifest,
    unit_paths: dict[str, str],
    taxonomy_atoms: set[str],
    staged_root: Path,
) -> tuple[list[CapabilityEvent], list[ResourceUse], list[RawObservation]]:
    """Convert raw observations into normalized events/resources."""
    path_to_units: dict[str, list[str]] = defaultdict(list)
    for unit_id, relative_path in unit_paths.items():
        path_to_units[_normalize_relpath(relative_path)].append(unit_id)

    grouped: dict[tuple[str, str, int, str], list[RawObservation]] = defaultdict(list)
    bridge_hints: list[RawObservation] = []
    for obs in observations:
        if "capType" not in obs.fields:
            bridge_hints.append(obs)
            continue
        capability = obs.fields["capType"]
        if capability not in taxonomy_atoms:
            raise ValueError(f"Unknown capability atom from analysis: {capability}")
        unit_id = obs.unit_id or _resolve_unit_id(path_to_units, obs.relative_path)
        if unit_id is None:
            logger.warning(
                "Skipping observation with unmapped analysis path: %s",
                obs.relative_path,
            )
            continue
        key = (
            _normalize_relpath(obs.relative_path),
            capability,
            obs.line,
            unit_id,
        )
        grouped[key].append(obs)

    events: list[CapabilityEvent] = []
    resources: list[ResourceUse] = []
    resource_keys: set[tuple[str, str | None, str, str, bool, str, str]] = set()

    for key in sorted(grouped):
        obs = _pick_best_observation(grouped[key])
        unit_id = key[3]
        detail = obs.fields.get("detail", "")
        tier = obs.fields.get("tier", "")
        relative_path = _normalize_relpath(obs.relative_path)
        source_line = obs.source_text or _read_source_line(staged_root, relative_path, obs.line)
        location = f"{relative_path}:{obs.line}" if obs.line else relative_path
        capability = _refine_capability(obs.fields["capType"], source_line, detail)
        environment_names = json.loads(obs.fields["environmentNames"]) if tier in {"bash_pattern", "bash_ast"} else None
        extracted = _extract_resources(source_line, detail, capability, language=obs.language,
                                       environment_names=environment_names)
        if "shellArguments" in obs.fields:
            arguments = json.loads(obs.fields["shellArguments"])
            alternatives = [tuple(option) for choices in arguments for option in choices]
            if capability in {"http_request", "websocket", "upload_exfil", "remote_sync", "telemetry_post"}:
                targets = [(value, resolved) for value, resolved in alternatives
                           if value.startswith(("http://", "https://", "ws://", "wss://")) or not resolved]
                if targets:
                    extracted = [item for item in extracted if item[0] not in {"url", "domain"}]
                    extracted.extend(("url", value, resolved, "shell_binding", "AST argument binding")
                                     for value, resolved in targets)
            if capability in {"file_write", "file_delete", "file_permission_change"} and arguments:
                targets = [tuple(option) for option in arguments[-1] if not option[0].startswith("-")]
                if targets:
                    extracted = [item for item in extracted if item[0] != "path"]
                    extracted.extend(("path", value, resolved, "shell_binding", "AST argument binding")
                                     for value, resolved in targets)
        authoritative = [item for item in grouped[key]
                         if _TIER_PRIORITY.get(item.fields.get("tier", ""), 99) == _TIER_PRIORITY.get(tier, 99)]
        explicit_resources = list(dict.fromkeys(hint for item in authoritative
                                               if (hint := _explicit_resource_hint(item)) is not None))
        if explicit_resources:
            explicit_types = {hint[0] for hint in explicit_resources}
            extracted = [
                item
                for item in extracted
                if item[0] not in explicit_types
            ]
            extracted = [*explicit_resources, *extracted]

        resource_groups = [extracted]
        for resource_group in resource_groups:
            event_id = f"e{len(events)}"
            events.append(
                CapabilityEvent(
                    event_id=event_id,
                    unit_id=unit_id,
                    capability=capability,
                    api_call="environment variable read" if capability == "env_var_read" else detail,
                    location=location,
                    arguments=[value for _, value, _, _, _ in resource_group],
                    tier=tier,
                    language=obs.language,
                    file_path=relative_path,
                    line=obs.line,
                    detail=detail,
                )
            )
            for resource_type, value, resolved, origin_kind, origin_hint in resource_group:
                context = origin_hint if origin_kind in {"native_ast", "shell_binding"} else obs.fields.get("branchContext", "")
                resource_key = (unit_id, event_id, resource_type, value, resolved, location, context)
                if resource_key in resource_keys:
                    continue
                resource_keys.add(resource_key)
                resources.append(
                    ResourceUse(
                        resource_id=f"r{len(resources)}",
                        unit_id=unit_id,
                        event_id=event_id,
                        resource_type=resource_type,
                        value=value,
                        resolved=resolved,
                        location=location,
                        origin_kind=origin_kind,
                        origin_hint=origin_hint,
                        context=context,
                    )
                )
            unsafe = [item for item in grouped[key]
                      if item.fields.get("unsafeCommandConstruction") == "true"]
            if unsafe:
                resources.append(ResourceUse(
                    resource_id=f"r{len(resources)}", unit_id=unit_id, event_id=event_id,
                    resource_type="registered_operation", value="shell_exec", resolved=True,
                    location=location, origin_kind="codeql_dataflow",
                    origin_hint=json.dumps({"model":"codeql_unsafe_command_construction_v1",
                        "location":location, "detail":unsafe[0].fields["detail"]}),
                ))

    logger.info(
        "Normalized %d events and %d resources",
        len(events),
        len(resources),
    )
    return events, resources, bridge_hints


def canonicalize_behavior(events, resources, manifest, unit_paths=None, doc_contents=None):
    """Apply one K_e and strict tier precedence, including document unit provenance."""
    unit_paths = unit_paths or {}
    doc_contents = doc_contents or {}
    synthetic_units = {unit.source_block_id:unit for unit in manifest.code_units if unit.source_doc_id is not None}
    normalized = []
    operations = {resource.event_id:resource for resource in resources if resource.resource_type == "registered_operation"}
    for event in events:
        proof = operations.get(event.event_id)
        if event.unit_id.startswith("doc::") and proof is not None:
            fact = json.loads(proof.origin_hint)
            unit = synthetic_units.get(fact.get("source_block_id"))
            if unit is not None and unit.unit_id in unit_paths and unit.source_doc_id in doc_contents:
                content = doc_contents[unit.source_doc_id]
                first_body_line = content[:unit.source_offset].count(chr(10)) + 2
                line = event.line - first_body_line + 1
                file_path = unit_paths[unit.unit_id]
                event = event.model_copy(update={"unit_id":unit.unit_id, "file_path":file_path,
                                                "line":line, "location":f"{file_path}:{line}"})
        normalized.append(event)
    groups = defaultdict(list)
    for event in normalized:
        groups[(event.file_path, event.line, event.unit_id, event.capability)].append(event)
    rank = lambda event: _TIER_PRIORITY.get(event.tier, 1 if event.tier == "registered_source" else 99)
    retained, identifiers, winner_ranks = [], {}, {}
    normalized_by_id = {event.event_id:event for event in normalized}
    for group in groups.values():
        winner = min(group, key=lambda event: (rank(event), event.event_id))
        # Deterministic documentary clauses use these source-bound IDs. Keep that ID
        # while taking all semantics and resource bindings from the highest tier.
        stable_id = min((event.event_id for event in group if event.event_id.startswith("operation-doc::")), default=winner.event_id)
        config_units = {u.unit_id for u in synthetic_units.values() if u.source_pointer is not None}
        tier = (3 if winner.unit_id in config_units else
                4 if winner.unit_id in {u.unit_id for u in synthetic_units.values()} or winner.unit_id.startswith("doc::")
                else rank(winner) + 1)
        retained.append(winner.model_copy(update={"event_id":stable_id,"analysis_tier":tier}))
        winner_ranks[stable_id] = rank(winner)
        identifiers.update((event.event_id, stable_id) for event in group)
    target_types = {"url", "domain", "path", "env_var", "command", "import_target"}
    types_by_winner = defaultdict(set)
    for resource in resources:
        winner_id = identifiers.get(resource.event_id, resource.event_id)
        original = normalized_by_id.get(resource.event_id)
        if original and rank(original) == winner_ranks[winner_id]:
            types_by_winner[winner_id].add(resource.resource_type)
    events_by_id = {event.event_id:event for event in retained}
    revised = []
    for resource in resources:
        winner_id = identifiers.get(resource.event_id, resource.event_id)
        original = normalized_by_id.get(resource.event_id)
        if original and resource.resource_type in target_types and rank(original) > winner_ranks[winner_id] and resource.resource_type in types_by_winner[winner_id]:
            continue
        winner = events_by_id.get(winner_id)
        revised.append(resource.model_copy(update={"event_id":winner_id,
                       **({"unit_id":winner.unit_id} if winner else {})}))
    return retained, revised, identifiers


def _first_location(result: dict[str, object]) -> dict[str, object] | None:
    locations = result.get("locations", [])
    if not isinstance(locations, list) or not locations:
        return None
    location = locations[0]
    return location if isinstance(location, dict) else None


def _first_thread_flow_locations(result: dict[str, object]) -> list[dict[str, object]]:
    code_flows = result.get("codeFlows", [])
    if not isinstance(code_flows, list) or not code_flows:
        return []
    code_flow = code_flows[0]
    if not isinstance(code_flow, dict):
        return []
    thread_flows = code_flow.get("threadFlows", [])
    if not isinstance(thread_flows, list) or not thread_flows:
        return []
    thread_flow = thread_flows[0]
    if not isinstance(thread_flow, dict):
        return []
    locations = thread_flow.get("locations", [])
    if not isinstance(locations, list):
        return []
    return [location for location in locations if isinstance(location, dict)]


def _resolve_artifact_uri(location: dict[str, object], artifacts: list[dict[str, object]]) -> str:
    physical = location.get("physicalLocation", {})
    if not isinstance(physical, dict):
        return ""
    artifact_location = physical.get("artifactLocation", {})
    if not isinstance(artifact_location, dict):
        return ""
    uri = artifact_location.get("uri")
    if isinstance(uri, str) and uri:
        if uri.startswith("file://"):
            return Path(uri[7:]).as_posix()
        return PurePosixPath(uri).as_posix()
    index = artifact_location.get("index")
    if isinstance(index, int) and 0 <= index < len(artifacts):
        artifact = artifacts[index]
        if isinstance(artifact, dict):
            art_location = artifact.get("location", {})
            if isinstance(art_location, dict):
                art_uri = art_location.get("uri", "")
                if isinstance(art_uri, str):
                    return PurePosixPath(art_uri).as_posix()
    return ""


def _pick_best_observation(options: list[RawObservation]) -> RawObservation:
    return min(
        options,
        key=lambda item: (
            _TIER_PRIORITY.get(item.fields.get("tier", ""), 99),
            item.message,
        ),
    )


def _resolve_unit_id(
    path_to_units: dict[str, list[str]],
    relative_path: str,
) -> str | None:
    normalized = _normalize_relpath(relative_path)
    candidates = path_to_units.get(normalized, [])
    if not candidates:
        return None
    return candidates[0]


def _explicit_resource_hint(obs: RawObservation) -> tuple[str, str, bool, str, str] | None:
    resource_type = obs.fields.get("resourceType")
    resource_value = obs.fields.get("resourceValue")
    if not resource_type or not resource_value:
        return None
    resolved = obs.fields.get("resourceResolved", "true").lower() != "false"
    origin_kind = obs.fields.get("resourceOriginKind", "unknown")
    origin_hint = obs.fields.get("resourceOriginHint", "")
    return resource_type, resource_value, resolved, origin_kind, origin_hint


__all__ = [
    "RawObservation",
    "RawPathResult",
    "RawPathStep",
    "build_gc_artifact",
    "build_risk_paths",
    "detect_bridges",
    "derive_data_object_graph",
    "derive_gc_enrichment",
    "load_sarif_observations",
    "load_sarif_paths",
    "load_taxonomy_atoms",
    "normalize_codeql_paths",
    "normalize_observations",
    "parse_structured_message",
]
