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
    """Load CodeQL path-problem SARIF results into structured raw paths."""
    data = json.loads(sarif_path.read_text(encoding="utf-8"))
    paths: list[RawPathResult] = []

    for run in data.get("runs", []):
        artifacts = run.get("artifacts", [])
        for result in run.get("results", []):
            message = result.get("message", {}).get("text", "")
            fields = parse_structured_message(message)
            if "pathKind" not in fields:
                continue
            thread_locations = _first_thread_flow_locations(result)
            if not thread_locations:
                continue

            steps: list[RawPathStep] = []
            for thread_location in thread_locations:
                if not isinstance(thread_location, dict):
                    continue
                location = thread_location.get("location", {})
                if not isinstance(location, dict):
                    continue
                relative_path = _resolve_artifact_uri(location, artifacts)
                if not relative_path:
                    continue
                physical = location.get("physicalLocation", {})
                if isinstance(physical, dict):
                    region = physical.get("region", {})
                    line = region.get("startLine", 0) if isinstance(region, dict) else 0
                else:
                    line = 0
                step_message = location.get("message", {}).get("text", "")
                steps.append(
                    RawPathStep(
                        relative_path=relative_path,
                        line=line,
                        message=step_message,
                    )
                )

            if len(steps) < 2:
                continue
            paths.append(
                RawPathResult(
                    language=language,
                    message=message,
                    fields=fields,
                    steps=steps,
                )
            )

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
    resource_keys: set[tuple[str, str | None, str, str, str]] = set()

    for idx, key in enumerate(sorted(grouped)):
        obs = _pick_best_observation(grouped[key])
        unit_id = key[3]
        detail = obs.fields.get("detail", "")
        tier = obs.fields.get("tier", "")
        relative_path = _normalize_relpath(obs.relative_path)
        source_line = obs.source_text or _read_source_line(staged_root, relative_path, obs.line)
        location = f"{relative_path}:{obs.line}" if obs.line else relative_path
        event_id = f"e{idx}"
        capability = _refine_capability(obs.fields["capType"], source_line, detail)
        extracted = _extract_resources(source_line, detail, capability)
        explicit_resource = _explicit_resource_hint(obs)
        if explicit_resource is not None:
            extracted = [
                item
                for item in extracted
                if item[2] or item[0] != explicit_resource[0]
            ]
            extracted = [explicit_resource, *extracted]

        events.append(
            CapabilityEvent(
                event_id=event_id,
                unit_id=unit_id,
                capability=capability,
                api_call=detail,
                location=location,
                arguments=[value for _, value, _, _, _ in extracted],
                tier=tier,
                language=obs.language,
                file_path=relative_path,
                line=obs.line,
                detail=detail,
            )
        )

        for resource_type, value, resolved, origin_kind, origin_hint in extracted:
            resource_key = (unit_id, event_id, resource_type, value, location)
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
                )
            )

    logger.info(
        "Normalized %d events and %d resources",
        len(events),
        len(resources),
    )
    return events, resources, bridge_hints


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
