"""Candidate generation for reconciliation graph construction.

Produces CandidatePair objects from deterministic matching and embedding-backed
semantic retrieval, without making any relation judgments.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from skillrecon.core.enums import BehaviorKind, CandidateSource
from skillrecon.core.sensitivity import event_requires_authorization
from skillrecon.core.types import (
    Bridge,
    CandidatePair,
    CapabilityEvent,
    Clause,
    ContractTable,
    OrchestrationHypothesis,
    PackageLink,
    PackageManifest,
    ResourceUse,
    RiskPath,
    Step,
    StepUnitCandidate,
)
from skillrecon.reconcile.predicate import _OverlapMap, load_overlap_policy

logger = logging.getLogger(__name__)
_DEFAULT_OVERLAP_POLICY_PATH = Path("experiments/configs/overlap_policy_v1.json")


# ---------------------------------------------------------------------------
# Index structures (internal)
# ---------------------------------------------------------------------------


@dataclass
class _BehaviorIndex:
    """Lookup indices over behavior objects for fast candidate matching."""

    events_by_id: dict[str, CapabilityEvent] = field(default_factory=dict)
    events_by_unit: dict[str, list[CapabilityEvent]] = field(default_factory=dict)
    events_by_cap: dict[str, list[CapabilityEvent]] = field(default_factory=dict)
    paths_by_id: dict[str, RiskPath] = field(default_factory=dict)
    resources_by_value: dict[str, list[ResourceUse]] = field(default_factory=dict)
    resources_by_event: dict[str, list[ResourceUse]] = field(default_factory=dict)
    paths_by_unit: dict[str, list[RiskPath]] = field(default_factory=dict)
    bridges_by_unit: dict[str, list[Bridge]] = field(default_factory=dict)
    orch_by_unit: dict[str, list[OrchestrationHypothesis]] = field(
        default_factory=dict
    )

    @classmethod
    def build(
        cls,
        events: list[CapabilityEvent],
        resources: list[ResourceUse],
        paths: list[RiskPath],
        bridges: list[Bridge],
        orchestrations: list[OrchestrationHypothesis],
    ) -> _BehaviorIndex:
        idx = cls()
        for ev in events:
            idx.events_by_id[ev.event_id] = ev
            idx.events_by_unit.setdefault(ev.unit_id, []).append(ev)
            idx.events_by_cap.setdefault(ev.capability, []).append(ev)
        for res in resources:
            idx.resources_by_value.setdefault(res.value.lower(), []).append(res)
            if res.event_id:
                idx.resources_by_event.setdefault(res.event_id, []).append(res)
        for p in paths:
            idx.paths_by_id[p.path_id] = p
            idx.paths_by_unit.setdefault(p.source.unit_id, []).append(p)
            if p.sink.unit_id != p.source.unit_id:
                idx.paths_by_unit.setdefault(p.sink.unit_id, []).append(p)
        for br in bridges:
            idx.bridges_by_unit.setdefault(br.source_unit_id, []).append(br)
            idx.bridges_by_unit.setdefault(br.target_unit_id, []).append(br)
        for oh in orchestrations:
            for uid in oh.target_unit_ids:
                idx.orch_by_unit.setdefault(uid, []).append(oh)
        return idx


@dataclass
class _PackageIndex:
    """Lookup indices over package provenance."""

    links_by_unit: dict[str, list[PackageLink]] = field(default_factory=dict)
    units_by_file: dict[str, list[str]] = field(default_factory=dict)
    file_by_id: dict[str, str] = field(default_factory=dict)
    doc_by_id: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(cls, manifest: PackageManifest) -> _PackageIndex:
        idx = cls()
        for link in manifest.links:
            if link.target_unit_id:
                idx.links_by_unit.setdefault(link.target_unit_id, []).append(link)
        for cu in manifest.code_units:
            idx.units_by_file.setdefault(cu.file_id, []).append(cu.unit_id)
        for f in manifest.files:
            idx.file_by_id[f.file_id] = f.relative_path
        for doc in manifest.documents:
            idx.doc_by_id[doc.doc_id] = doc.file_id
        return idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_candidates(
    contract_table: ContractTable,
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    paths: list[RiskPath],
    bridges: list[Bridge],
    orchestrations: list[OrchestrationHypothesis],
    manifest: PackageManifest,
    overlap_map: _OverlapMap | None = None,
    a_req: set[str] | None = None,
    max_candidates_per_behavior: int | None = 8,
    max_alignment_fallbacks_per_step: int = 3,
    max_semantic_event_fallbacks: int = 3,
    max_semantic_path_fallbacks: int = 2,
) -> list[CandidatePair]:
    """Generate candidate pairs from deterministic sources.

    The returned list is deduplicated by (clause_id, behavior_kind, object_id).
    ``max_candidates_per_behavior`` implements the paper's top-k candidate
    retrieval bound over each behavior object; set it to ``None`` to disable
    this cap for diagnostic runs.
    """
    bidx = _BehaviorIndex.build(events, resources, paths, bridges, orchestrations)
    pidx = _PackageIndex.build(manifest)
    effective_overlap_map = overlap_map
    if effective_overlap_map is None:
        effective_overlap_map = (
            load_overlap_policy(_DEFAULT_OVERLAP_POLICY_PATH)
            if _DEFAULT_OVERLAP_POLICY_PATH.is_file()
            else {}
        )

    aggregated_hits: dict[
        tuple[str, str, str],
        tuple[BehaviorKind, str, list[CandidateSource], dict[str, str], list[str]],
    ] = {}
    alignment_hits_by_clause, aligned_units_by_clause = _alignment_backfill_candidates(
        contract_table=contract_table,
        bidx=bidx,
        manifest=manifest,
        overlap_map=effective_overlap_map,
        a_req=a_req,
        max_fallbacks_per_step=max_alignment_fallbacks_per_step,
    )

    for clause in contract_table.clauses:
        pairs = _deterministic_candidates(
            clause,
            bidx,
            pidx,
            manifest,
            effective_overlap_map,
            a_req,
            max_semantic_event_fallbacks=max_semantic_event_fallbacks,
            max_semantic_path_fallbacks=max_semantic_path_fallbacks,
        )
        pairs.extend(alignment_hits_by_clause.get(clause.clause_id, []))
        for bkind, obj_id, sources, signals, route_refs in pairs:
            sources, signals = _annotate_alignment_context(
                clause=clause,
                behavior_kind=bkind,
                object_id=obj_id,
                candidate_sources=sources,
                shared_signals=signals,
                bidx=bidx,
                aligned_units_by_clause=aligned_units_by_clause,
            )
            key = (clause.clause_id, bkind.value, obj_id)
            if key in aggregated_hits:
                existing = aggregated_hits[key]
                aggregated_hits[key] = (
                    existing[0],
                    existing[1],
                    _merge_candidate_sources([*existing[2], *sources]),
                    {**existing[3], **signals},
                    list(dict.fromkeys([*existing[4], *route_refs])),
                )
                continue
            aggregated_hits[key] = (
                bkind,
                obj_id,
                list(sources),
                dict(signals),
                list(route_refs),
            )

    capped_hits = _cap_candidates_per_behavior(
        aggregated_hits,
        max_candidates_per_behavior=max_candidates_per_behavior,
    )

    candidates: list[CandidatePair] = []
    for counter, ((clause_id, _bkind, _obj_id), (bkind, obj_id, sources, signals, route_refs)) in enumerate(
        capped_hits.items(),
        start=1,
    ):
        cp = CandidatePair(
            candidate_id=f"cand-{counter:04d}",
            clause_id=clause_id,
            behavior_kind=bkind,
            event_id=obj_id if bkind == BehaviorKind.EVENT else None,
            resource_id=obj_id if bkind == BehaviorKind.RESOURCE else None,
            path_id=obj_id if bkind == BehaviorKind.PATH else None,
            candidate_sources=sources,
            shared_signals=signals,
            route_refs=route_refs,
        )
        candidates.append(cp)

    logger.info(
        "Generated %d candidate pairs for %d clauses",
        len(candidates),
        len(contract_table.clauses),
    )
    return candidates


def generate_alignment_candidates(
    *,
    steps: list[Step],
    manifest: PackageManifest,
    max_fallbacks_per_step: int = 3,
) -> list[StepUnitCandidate]:
    """Generate auditable step-to-code-unit alignment candidates."""
    file_by_id = {entry.file_id: entry.relative_path for entry in manifest.files}
    code_units_by_id = {unit.unit_id: unit for unit in manifest.code_units}
    links_by_doc: dict[str, list[PackageLink]] = {}
    for link in manifest.links:
        links_by_doc.setdefault(link.source_doc_id, []).append(link)

    seen: set[tuple[str, str]] = set()
    candidates: list[StepUnitCandidate] = []

    for step in steps:
        step_hits_by_unit: dict[str, tuple[int, list[CandidateSource], dict[str, str]]] = {}
        step_text_lower = step.text.lower()
        step_tokens = set(_tokenize_text(" ".join([step.heading_context, step.text])))

        def record_hit(
            unit_id: str,
            priority: int,
            sources: list[CandidateSource],
            signals: dict[str, str],
        ) -> None:
            existing = step_hits_by_unit.get(unit_id)
            if existing is None:
                step_hits_by_unit[unit_id] = (priority, list(sources), dict(signals))
                return
            existing_priority, existing_sources, existing_signals = existing
            step_hits_by_unit[unit_id] = (
                min(priority, existing_priority),
                _merge_candidate_sources([*existing_sources, *sources]),
                {**existing_signals, **signals},
            )

        for link in links_by_doc.get(step.doc_id, []):
            if link.target_unit_id is None:
                continue
            if link.source_span and link.source_span.lower() in step_text_lower:
                record_hit(
                    link.target_unit_id,
                    0,
                    [CandidateSource.PACKAGE_LINK],
                    {"link_id": link.link_id, "source_span": link.source_span},
                )

        for unit in manifest.code_units:
            unit_path = file_by_id.get(unit.file_id, "")
            basename = PurePosixPath(unit_path).name.lower() if unit_path else ""
            relative_lower = unit_path.lower()

            if unit.source_doc_id == step.doc_id and (
                (unit.source_block_id and unit.source_block_id in step.block_ids)
                or (
                    unit.binding_instruction
                    and unit.binding_instruction.strip().lower() == step.text.strip().lower()
                )
            ):
                record_hit(
                    unit.unit_id,
                    0,
                    [CandidateSource.FILE_MENTION],
                    {"binding_instruction": unit.binding_instruction or ""},
                )
                continue

            if unit_path and (
                relative_lower in step_text_lower
                or (basename and basename in step_text_lower)
            ):
                record_hit(
                    unit.unit_id,
                    1,
                    [CandidateSource.FILE_MENTION],
                    {"unit_path": unit_path},
                )
                continue

            section_relation = _section_alignment_relation(step, unit)
            if section_relation is not None:
                section_signals = {
                    "section_alignment": section_relation,
                }
                distance = _step_unit_distance(step, unit)
                if distance is not None:
                    section_signals["section_distance"] = str(distance)
                record_hit(
                    unit.unit_id,
                    1 if distance is not None and distance <= 4 else 2,
                    [CandidateSource.SECTION_ALIGNMENT],
                    section_signals,
                )

            if step_tokens:
                unit_tokens = _code_unit_semantic_tokens(unit, unit_path)
                score, shared, embedding_score = _semantic_score(step_tokens, unit_tokens)
                if _passes_semantic_threshold(score, shared, embedding_score):
                    record_hit(
                        unit.unit_id,
                        3,
                        [CandidateSource.SEMANTIC_RETRIEVAL],
                        {
                            "semantic_score": f"{score:.3f}",
                            "embedding_score": f"{embedding_score:.3f}",
                            "semantic_shared_tokens": ",".join(sorted(shared)),
                        },
                    )

        step_hits = [
            (priority, unit_id, sources, signals)
            for unit_id, (priority, sources, signals) in step_hits_by_unit.items()
        ]
        step_hits.sort(key=lambda item: (item[0], item[1]))
        selected_per_step = 0
        for _priority, unit_id, sources, signals in step_hits:
            key = (step.step_id, unit_id)
            if key in seen:
                continue
            seen.add(key)
            selected_per_step += 1
            candidates.append(
                StepUnitCandidate(
                    candidate_id=f"stepcand-{len(candidates) + 1:04d}",
                    step_id=step.step_id,
                    code_unit_id=unit_id,
                    candidate_sources=sources,
                    shared_signals=signals,
                )
            )
            if (
                selected_per_step >= max_fallbacks_per_step
                and CandidateSource.SEMANTIC_RETRIEVAL in sources
            ):
                break

    logger.info(
        "Generated %d step-unit alignment candidates for %d steps",
        len(candidates),
        len(steps),
    )
    return candidates


def _section_alignment_relation(step: Step, unit) -> str | None:
    if unit.source_doc_id != step.doc_id:
        return None
    if not step.heading_context or not unit.heading_context:
        return None
    step_heading = step.heading_context.strip().lower()
    unit_heading = unit.heading_context.strip().lower()
    if step_heading == unit_heading:
        return "exact_heading"
    if step_heading.startswith(unit_heading) or unit_heading.startswith(step_heading):
        return "nested_heading"
    return None


def _step_unit_distance(step: Step, unit) -> int | None:
    step_index = _block_index(step.step_id)
    block_index = _block_index(unit.source_block_id)
    if step_index is None or block_index is None:
        return None
    return abs(step_index - block_index)


def _block_index(block_id: str | None) -> int | None:
    if not block_id:
        return None
    matches = re.findall(r"\d+", block_id)
    if not matches:
        return None
    return int(matches[-1])


def _alignment_backfill_candidates(
    *,
    contract_table: ContractTable,
    bidx: _BehaviorIndex,
    manifest: PackageManifest,
    overlap_map: _OverlapMap,
    a_req: set[str] | None,
    max_fallbacks_per_step: int = 3,
) -> tuple[dict[str, list[_Hit]], dict[str, set[str]]]:
    """Lift step-unit alignment into clause-event/path candidates."""
    if not contract_table.steps or not manifest.code_units:
        return {}, {}

    clause_ids_by_step: dict[str, list[str]] = {}
    for clause in contract_table.clauses:
        for step_id in clause.step_ids:
            clause_ids_by_step.setdefault(step_id, []).append(clause.clause_id)
    if not clause_ids_by_step:
        return {}, {}

    clauses_by_id = {clause.clause_id: clause for clause in contract_table.clauses}
    hits_by_clause: dict[str, list[_Hit]] = {}
    aligned_units_by_clause: dict[str, set[str]] = {}
    alignment_candidates = generate_alignment_candidates(
        steps=contract_table.steps,
        manifest=manifest,
        max_fallbacks_per_step=max_fallbacks_per_step,
    )

    for candidate in alignment_candidates:
        clause_ids = clause_ids_by_step.get(candidate.step_id, [])
        if not clause_ids:
            continue
        base_sources = _merge_candidate_sources(
            [CandidateSource.STEP_UNIT_ALIGNMENT, *candidate.candidate_sources]
        )
        base_signals = {
            **candidate.shared_signals,
            "aligned_step_id": candidate.step_id,
            "aligned_code_unit_id": candidate.code_unit_id,
        }

        for clause_id in clause_ids:
            clause = clauses_by_id[clause_id]
            aligned_units_by_clause.setdefault(clause_id, set()).add(candidate.code_unit_id)
            for event in bidx.events_by_unit.get(candidate.code_unit_id, []):
                if not _capability_family_match(clause.capability, event.capability, overlap_map):
                    continue
                hits_by_clause.setdefault(clause_id, []).append(
                    (
                        BehaviorKind.EVENT,
                        event.event_id,
                        base_sources,
                        {
                            **base_signals,
                            "clause_capability": clause.capability,
                            "behavior_capability": event.capability,
                        },
                        [],
                    )
                )

            for path in bidx.paths_by_unit.get(candidate.code_unit_id, []):
                if not _path_requires_candidate_attention(path, bidx.events_by_id, a_req):
                    continue
                comparable_caps = (
                    (path.sink.label,)
                    if a_req
                    else (path.source.label, path.sink.label)
                )
                if not any(
                    _capability_family_match(clause.capability, behavior_cap, overlap_map)
                    for behavior_cap in comparable_caps
                ):
                    continue
                hits_by_clause.setdefault(clause_id, []).append(
                    (
                        BehaviorKind.PATH,
                        path.path_id,
                        base_sources,
                        {
                            **base_signals,
                            "clause_capability": clause.capability,
                            "path_kind": path.path_kind,
                        },
                        list(path.bridges_used) + list(path.orchestration_hypotheses),
                    )
                )

    return hits_by_clause, aligned_units_by_clause


def _annotate_alignment_context(
    *,
    clause: Clause,
    behavior_kind: BehaviorKind,
    object_id: str,
    candidate_sources: list[CandidateSource],
    shared_signals: dict[str, str],
    bidx: _BehaviorIndex,
    aligned_units_by_clause: dict[str, set[str]],
) -> tuple[list[CandidateSource], dict[str, str]]:
    aligned_units = aligned_units_by_clause.get(clause.clause_id, set())
    if not clause.step_ids or not aligned_units:
        return candidate_sources, shared_signals

    candidate_unit_ids: set[str] = set()
    if behavior_kind == BehaviorKind.EVENT:
        event = bidx.events_by_id.get(object_id)
        if event is not None:
            candidate_unit_ids.add(event.unit_id)
    elif behavior_kind == BehaviorKind.PATH:
        path = bidx.paths_by_id.get(object_id)
        if path is not None:
            candidate_unit_ids.update(segment.unit_id for segment in path.segments)

    if not candidate_unit_ids:
        return candidate_sources, shared_signals

    if candidate_unit_ids & aligned_units:
        return (
            _merge_candidate_sources([*candidate_sources, CandidateSource.STEP_UNIT_ALIGNMENT]),
            {
                **shared_signals,
                "alignment_status": "matched",
            },
        )

    return (
        candidate_sources,
        {
            **shared_signals,
            "alignment_status": "mismatch",
            "alignment_expected_units": ",".join(sorted(aligned_units)),
        },
    )


# ---------------------------------------------------------------------------
# Deterministic candidate sources
# ---------------------------------------------------------------------------

# Type alias for a single candidate hit before dedup
_Hit = tuple[
    BehaviorKind,  # behavior_kind
    str,  # object_id
    list[CandidateSource],  # sources
    dict[str, str],  # shared_signals
    list[str],  # route_refs
]


def _cap_candidates_per_behavior(
    aggregated_hits: dict[
        tuple[str, str, str],
        tuple[BehaviorKind, str, list[CandidateSource], dict[str, str], list[str]],
    ],
    *,
    max_candidates_per_behavior: int | None,
) -> dict[
    tuple[str, str, str],
    tuple[BehaviorKind, str, list[CandidateSource], dict[str, str], list[str]],
]:
    if max_candidates_per_behavior is None:
        return aggregated_hits
    if max_candidates_per_behavior <= 0:
        return {}

    grouped: dict[tuple[str, str], list[tuple[
        tuple[str, str, str],
        tuple[BehaviorKind, str, list[CandidateSource], dict[str, str], list[str]],
    ]]] = {}
    for key, hit in aggregated_hits.items():
        _clause_id, behavior_kind, object_id = key
        grouped.setdefault((behavior_kind, object_id), []).append((key, hit))

    selected_keys: set[tuple[str, str, str]] = set()
    for group in grouped.values():
        group.sort(key=lambda item: _candidate_rank(item[1], item[0][0]))
        selected_keys.update(
            key for key, _hit in group[:max_candidates_per_behavior]
        )

    return {
        key: hit
        for key, hit in aggregated_hits.items()
        if key in selected_keys
    }


def _candidate_rank(
    hit: tuple[BehaviorKind, str, list[CandidateSource], dict[str, str], list[str]],
    clause_id: str,
) -> tuple[int, float, str]:
    _bkind, obj_id, sources, signals, route_refs = hit
    return (
        -_candidate_source_strength(sources, signals, route_refs),
        -float(signals.get("semantic_score", "0") or 0),
        f"{clause_id}:{obj_id}",
    )


def _candidate_source_strength(
    sources: list[CandidateSource],
    signals: dict[str, str],
    route_refs: list[str],
) -> int:
    strength = 0
    if CandidateSource.LITERAL_OVERLAP in sources:
        strength += 40
    if CandidateSource.PACKAGE_LINK in sources:
        strength += 35
    if CandidateSource.FILE_MENTION in sources:
        strength += 30
    if CandidateSource.STEP_UNIT_ALIGNMENT in sources:
        strength += 25
    if CandidateSource.TYPED_RESOURCE_FAMILY in sources:
        strength += 20
    if CandidateSource.SEMANTIC_RETRIEVAL in sources:
        strength += 10
    if signals.get("alignment_status") == "matched":
        strength += 8
    if route_refs:
        strength += 4
    return strength


def _deterministic_candidates(
    clause: Clause,
    bidx: _BehaviorIndex,
    pidx: _PackageIndex,
    manifest: PackageManifest,
    overlap_map: _OverlapMap,
    a_req: set[str] | None,
    *,
    max_semantic_event_fallbacks: int = 3,
    max_semantic_path_fallbacks: int = 2,
) -> list[_Hit]:
    hits: list[_Hit] = []
    hits.extend(_capability_atom_match(clause, bidx, overlap_map))
    hits.extend(_literal_overlap(clause, bidx))
    hits.extend(_package_link_provenance(clause, bidx, pidx, overlap_map))
    hits.extend(_path_candidates(clause, bidx, overlap_map, a_req))
    hits.extend(
        _semantic_fallback_candidates(
            clause,
            bidx,
            a_req,
            max_event_fallbacks=max_semantic_event_fallbacks,
            max_path_fallbacks=max_semantic_path_fallbacks,
        )
    )
    return hits


def _capability_atom_match(
    clause: Clause,
    bidx: _BehaviorIndex,
    overlap_map: _OverlapMap,
) -> list[_Hit]:
    """Match clause capability to events sharing the same atom."""
    cap = clause.capability
    hits: list[_Hit] = []
    comparable_caps = overlap_map.get(cap, {cap})
    for comparable_cap in sorted(comparable_caps):
        for ev in bidx.events_by_cap.get(comparable_cap, []):
            hits.append((
                BehaviorKind.EVENT,
                ev.event_id,
                [CandidateSource.TYPED_RESOURCE_FAMILY],
                {
                    "clause_capability": cap,
                    "behavior_capability": ev.capability,
                },
                [],
            ))
    return hits


def _literal_overlap(
    clause: Clause,
    bidx: _BehaviorIndex,
) -> list[_Hit]:
    """Match clause target string against resource values."""
    literal_surfaces = [clause.target] if clause.target else []
    literal_surfaces.extend(
        constraint.value
        for constraint in clause.constraints
        if constraint.constraint_type.lower()
        in {"path", "path_glob", "file_glob", "domain", "domain_glob", "url", "url_glob", "env_var", "env_glob", "command", "command_glob"}
    )
    literal_surfaces = [surface for surface in literal_surfaces if surface]
    if not literal_surfaces:
        return []

    hits: list[_Hit] = []

    for val, res_list in bidx.resources_by_value.items():
        matched_surface = next(
            (surface for surface in literal_surfaces if _string_overlap(surface.lower(), val)),
            None,
        )
        if matched_surface is None:
            continue
        for res in res_list:
            hits.append((
                BehaviorKind.RESOURCE,
                res.resource_id,
                [CandidateSource.LITERAL_OVERLAP],
                {"clause_target": matched_surface, "resource_value": res.value},
                [],
            ))
    return hits


def _package_link_provenance(
    clause: Clause,
    bidx: _BehaviorIndex,
    pidx: _PackageIndex,
    overlap_map: _OverlapMap,
) -> list[_Hit]:
    """Match via package links: clause source doc → linked code unit → events."""
    hits: list[_Hit] = []
    for doc_id in clause.source_doc_ids:
        file_id = pidx.doc_by_id.get(doc_id)
        if not file_id:
            continue
        for link in pidx.links_by_unit.values():
            for lk in link:
                if lk.source_doc_id != doc_id or not lk.target_unit_id:
                    continue
                unit_id = lk.target_unit_id
                for ev in bidx.events_by_unit.get(unit_id, []):
                    if not _capability_family_match(
                        clause.capability,
                        ev.capability,
                        overlap_map,
                    ):
                        continue
                    hits.append((
                        BehaviorKind.EVENT,
                        ev.event_id,
                        [CandidateSource.PACKAGE_LINK],
                        {"link_id": lk.link_id, "unit_id": unit_id},
                        [],
                    ))
    return hits


def _path_candidates(
    clause: Clause,
    bidx: _BehaviorIndex,
    overlap_map: _OverlapMap,
    a_req: set[str] | None,
) -> list[_Hit]:
    """Match clause capability against path source/sink capabilities."""
    cap = clause.capability
    hits: list[_Hit] = []

    for path in bidx.paths_by_id.values():
        if not _path_requires_candidate_attention(path, bidx.events_by_id, a_req):
            continue
        sink_cap = path.sink.label
        comparable_caps = (sink_cap,) if a_req else (path.source.label, sink_cap)
        if any(
            _capability_family_match(cap, behavior_cap, overlap_map)
            for behavior_cap in comparable_caps
        ):
            route_refs = list(path.bridges_used) + list(path.orchestration_hypotheses)
            hits.append((
                BehaviorKind.PATH,
                path.path_id,
                [CandidateSource.TYPED_RESOURCE_FAMILY],
                {"capability": cap, "path_kind": path.path_kind},
                route_refs,
            ))
    return hits


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SEMANTIC_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "only",
    "this",
    "that",
    "your",
    "user",
    "api",
    "http",
    "https",
    "request",
    "requests",
    "main",
    "true",
    "false",
    "none",
    "com",
}
_MIN_SHARED_TOKENS = 2
_MIN_SEMANTIC_SCORE = 0.3
_MIN_EMBEDDING_SCORE = 0.55
_EMBEDDING_DIMENSIONS = 128


def _semantic_fallback_candidates(
    clause: Clause,
    bidx: _BehaviorIndex,
    a_req: set[str] | None,
    *,
    max_event_fallbacks: int = 3,
    max_path_fallbacks: int = 2,
) -> list[_Hit]:
    clause_tokens = _clause_semantic_tokens(clause)
    if not clause_tokens:
        return []

    hits: list[_Hit] = []
    hits.extend(
        _top_semantic_event_hits(
            clause,
            clause_tokens,
            bidx,
            max_fallbacks=max_event_fallbacks,
        )
    )
    hits.extend(
        _top_semantic_path_hits(
            clause,
            clause_tokens,
            bidx,
            a_req,
            max_fallbacks=max_path_fallbacks,
        )
    )
    return hits


def _merge_candidate_sources(
    sources: list[CandidateSource],
) -> list[CandidateSource]:
    merged: list[CandidateSource] = []
    seen: set[CandidateSource] = set()
    for source in sources:
        if source in seen:
            continue
        seen.add(source)
        merged.append(source)
    return merged


def _top_semantic_event_hits(
    clause: Clause,
    clause_tokens: set[str],
    bidx: _BehaviorIndex,
    *,
    max_fallbacks: int = 3,
) -> list[_Hit]:
    scored: list[tuple[float, list[_Hit]]] = []
    for event in bidx.events_by_id.values():
        event_tokens = _event_semantic_tokens(
            event,
            bidx.resources_by_event.get(event.event_id, []),
        )
        score, shared_tokens, embedding_score = _semantic_score(
            clause_tokens,
            event_tokens,
        )
        if not _passes_semantic_threshold(score, shared_tokens, embedding_score):
            continue
        scored.append((
            max(score, embedding_score),
            [(
                BehaviorKind.EVENT,
                event.event_id,
                [CandidateSource.SEMANTIC_RETRIEVAL],
                {
                    "semantic_score": f"{score:.3f}",
                    "embedding_score": f"{embedding_score:.3f}",
                    "semantic_shared_tokens": ",".join(sorted(shared_tokens)),
                    "clause_capability": clause.capability,
                    "behavior_capability": event.capability,
                },
                [],
            )],
        ))
    scored.sort(key=lambda item: (-item[0], item[1][0][1]))
    result: list[_Hit] = []
    for _, hit_group in scored[:max_fallbacks]:
        result.extend(hit_group)
    return result


def _top_semantic_path_hits(
    clause: Clause,
    clause_tokens: set[str],
    bidx: _BehaviorIndex,
    a_req: set[str] | None,
    *,
    max_fallbacks: int = 2,
) -> list[_Hit]:
    scored: list[tuple[float, _Hit]] = []
    for path in bidx.paths_by_id.values():
        if not _path_requires_candidate_attention(path, bidx.events_by_id, a_req):
            continue
        path_tokens = _path_semantic_tokens(path, sink_only=bool(a_req))
        score, shared_tokens, embedding_score = _semantic_score(
            clause_tokens,
            path_tokens,
        )
        if not _passes_semantic_threshold(score, shared_tokens, embedding_score):
            continue
        route_refs = list(path.bridges_used) + list(path.orchestration_hypotheses)
        scored.append((
            max(score, embedding_score),
            (
                BehaviorKind.PATH,
                path.path_id,
                [CandidateSource.SEMANTIC_RETRIEVAL],
                {
                    "semantic_score": f"{score:.3f}",
                    "embedding_score": f"{embedding_score:.3f}",
                    "semantic_shared_tokens": ",".join(sorted(shared_tokens)),
                    "path_kind": path.path_kind,
                },
                route_refs,
            ),
        ))
    scored.sort(key=lambda item: (-item[0], item[1][1]))
    return [hit for _, hit in scored[:max_fallbacks]]


def _path_requires_candidate_attention(
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


# ---------------------------------------------------------------------------
# String matching helpers
# ---------------------------------------------------------------------------


def _string_overlap(clause_target: str, resource_value: str) -> bool:
    """Check if clause target and resource value have meaningful overlap.

    Supports exact match, basename match, and prefix containment.
    """
    if clause_target == resource_value:
        return True

    ct_path = PurePosixPath(clause_target)
    rv_path = PurePosixPath(resource_value)
    if ct_path.name and ct_path.name == rv_path.name:
        return True

    if clause_target.upper() == resource_value.upper():
        return True

    return bool(
        resource_value.startswith(clause_target)
        or clause_target.startswith(resource_value)
    )


def _capability_family_match(
    clause_capability: str,
    behavior_capability: str,
    overlap_map: _OverlapMap,
) -> bool:
    if clause_capability == behavior_capability:
        return True
    return behavior_capability in overlap_map.get(clause_capability, set())


def _clause_semantic_tokens(clause: Clause) -> set[str]:
    tokens = set(_tokenize_text(clause.capability))
    if clause.target:
        tokens.update(_tokenize_text(clause.target))
    for constraint in clause.constraints:
        tokens.update(_tokenize_text(constraint.constraint_type))
        tokens.update(_tokenize_text(constraint.value))
    for evidence in clause.evidence_spans:
        tokens.update(_tokenize_text(evidence.text))
    return tokens


def _event_semantic_tokens(
    event: CapabilityEvent,
    resources: list[ResourceUse],
) -> set[str]:
    tokens = set(_tokenize_text(event.capability))
    tokens.update(_tokenize_text(event.api_call))
    tokens.update(_tokenize_text(event.detail))
    for argument in event.arguments:
        tokens.update(_tokenize_text(argument))
    for resource in resources:
        tokens.update(_tokenize_text(resource.resource_type))
        tokens.update(_tokenize_text(resource.value))
    return tokens


def _path_semantic_tokens(path: RiskPath, *, sink_only: bool = False) -> set[str]:
    tokens = set(_tokenize_text(path.path_kind))
    tokens.update(_tokenize_text(path.sink.label))
    if sink_only:
        return tokens
    tokens.update(_tokenize_text(path.source.label))
    for segment in path.segments:
        tokens.update(_tokenize_text(segment.label))
    return tokens


def _code_unit_semantic_tokens(unit, unit_path: str) -> set[str]:
    tokens = set(_tokenize_text(unit.language))
    tokens.update(_tokenize_text(unit_path))
    if unit.heading_context:
        tokens.update(_tokenize_text(unit.heading_context))
    if unit.binding_instruction:
        tokens.update(_tokenize_text(unit.binding_instruction))
    return tokens


def _semantic_score(left: set[str], right: set[str]) -> tuple[float, set[str], float]:
    if not left or not right:
        return 0.0, set(), 0.0
    shared = left & right
    lexical_score = (
        0.0
        if not shared
        else len(shared) / math.sqrt(len(left) * len(right))
    )
    embedding_score = _hashed_embedding_cosine(left, right)
    return lexical_score, shared, embedding_score


def _passes_semantic_threshold(
    score: float,
    shared_tokens: set[str],
    embedding_score: float,
) -> bool:
    if len(shared_tokens) >= _MIN_SHARED_TOKENS and score >= _MIN_SEMANTIC_SCORE:
        return True
    return embedding_score >= _MIN_EMBEDDING_SCORE


def _tokenize_text(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower().replace("_", " "))
        if len(token) >= 3 and token not in _SEMANTIC_STOPWORDS
    ]


def _hashed_embedding_cosine(left: set[str], right: set[str]) -> float:
    """Compute deterministic embedding-style cosine via signed feature hashing.

    This gives the retrieval fallback a real vector-space ranking without adding
    a network dependency to the core reconciliation path. The feature space uses
    word tokens plus token character trigrams, which makes near-surface matches
    rank above unrelated text even when exact token overlap is sparse.
    """
    left_vec = _hashed_embedding(left)
    right_vec = _hashed_embedding(right)
    if not left_vec or not right_vec:
        return 0.0
    dot = sum(value * right_vec.get(index, 0.0) for index, value in left_vec.items())
    left_norm = math.sqrt(sum(value * value for value in left_vec.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vec.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _hashed_embedding(tokens: set[str]) -> dict[int, float]:
    vector: dict[int, float] = {}
    for feature in _embedding_features(tokens):
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % _EMBEDDING_DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] = vector.get(bucket, 0.0) + sign
    return vector


def _embedding_features(tokens: set[str]) -> list[str]:
    features: list[str] = []
    for token in sorted(tokens):
        features.append(f"tok:{token}")
        padded = f"^{token}$"
        for index in range(max(0, len(padded) - 2)):
            features.append(f"tri:{padded[index:index + 3]}")
    return features
