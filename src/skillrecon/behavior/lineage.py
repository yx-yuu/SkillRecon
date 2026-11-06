"""Value provenance shared by the native language analyzers.

A command bridge transports individual argv/env slots, not every source in the
calling file. Its invocation event and target entry stay in each value's trail.
"""
from __future__ import annotations

from dataclasses import dataclass

from skillrecon.core.types import PathSegment, RiskPath


@dataclass(frozen=True)
class PayloadOrigin:
    event_id: str
    # unit, location, label, event ID, slot kind, expression
    trail: tuple[tuple[str, str, str, str | None, str, str], ...]
    bridges: tuple[str, ...] = ()


@dataclass
class InvocationPayload:
    bridge_id: str
    # argv[0] is the package-relative script, as in Bridge.cli_arguments.
    arguments: tuple[frozenset[PayloadOrigin], ...]
    environment: dict[str, frozenset[PayloadOrigin]]


def source_origin(event):
    return PayloadOrigin(event.event_id, ((event.unit_id, event.location,
        event.capability, event.event_id, 'read_result', event.api_call),))


def invocation_payload(bridge, event, arguments, environment, target_path):
    def transport(origins, slot):
        return frozenset(PayloadOrigin(origin.event_id, (*origin.trail,
            (event.unit_id, event.location, event.capability, event.event_id,
             'invocation_payload', slot),
            (bridge.target_unit_id, f'{target_path}:1', 'Script entry', None,
             slot, bridge.evidence)), (*origin.bridges, bridge.bridge_id))
            for origin in origins if bridge.bridge_id not in origin.bridges)

    return InvocationPayload(bridge.bridge_id,
        tuple(transport(origins, f'argv:{index}') for index, origins in enumerate(arguments)),
        {name: transport(origins, f'env:{name}') for name, origins in environment.items()})


def payload_path(origin, event, expression, slot, path_kind, path_id):
    trail = (*origin.trail, (event.unit_id, event.location, event.capability,
                            event.event_id, slot, expression))
    segments = [PathSegment(unit_id=unit, location=location, label=label,
        event_id=identifier, slot_kind=kind, expr_hint=expr)
        for unit, location, label, identifier, kind, expr in trail]
    return RiskPath(path_id=path_id, source=segments[0], sink=segments[-1],
        segments=segments, bridges_used=list(origin.bridges),
        evidence_level='bridge' if origin.bridges else 'registered_source',
        path_kind=('bridge:' if origin.bridges else 'registered_source:') + path_kind)
