#!/usr/bin/env python3
"""Build an RQ1 edge-validity gold-label benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.core.enums import ClauseOperator, JudgmentKind
from skillrecon.core.types import (
    CandidatePair,
    CapabilityEvent,
    Clause,
    ReconciliationEdge,
    ReconciliationJudgment,
    ResourceUse,
    Step,
    StepUnitCandidate,
)
from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    EdgeAnnotation,
    GoldLabel,
    load_gold_label_records,
    sanitize_gold_metadata,
    write_jsonl_models,
)
from skillrecon.evaluation.metrics import compute_edge_validity_by_type
from skillrecon.evaluation.runner import _load_edges
from skillrecon.llm.cache import CachedLLMClient

_EDGE_TYPES = (
    "supports",
    "potentially_supports",
    "contradicts",
    "relates_to",
    "scope_matches",
    "scope_violates",
    "aligns",
)
_CRITICAL_EDGE_TYPES = (
    "supports",
    "contradicts",
    "scope_matches",
    "scope_violates",
)


class _EdgeJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)

    signature: str
    edge_type: Literal[
        "supports",
        "potentially_supports",
        "contradicts",
        "relates_to",
        "scope_matches",
        "scope_violates",
        "aligns",
    ]
    expected_correct: bool
    rationale: str = ""


class _EdgeValidityBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    edges: list[_EdgeJudgment]


class _SampledEdgeItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    signature: str
    edge_type: str
    clause_id: str | None = None
    constraint_id: str | None = None
    event_id: str | None = None
    resource_id: str | None = None
    step_id: str | None = None
    code_unit_id: str | None = None
    predicted: bool
    candidate_sources: list[str] = []


_SYSTEM_PROMPT = """You are an exacting evaluator for cross-modal reconciliation relations in agent skills.

Task: For each candidate relation item, judge whether the stated relation should hold.

Interpretation rules:
- supports: the documented clause genuinely authorizes the observed event.
- potentially_supports: there is meaningful but incomplete support; weaker than supports.
- contradicts: the documentation meaningfully conflicts with the observed event.
- relates_to: an unknown clause is stably related to the event without clearly authorizing it.
- scope_matches: the constraint is satisfied by the observed resource use.
- scope_violates: the resource use is definitively out of scope for an otherwise authorized allowed clause.
- aligns: the documentation step and code unit are evidentially grounded as corresponding.

General rules:
- Judge the exact relation, not general topical similarity.
- Be conservative.
- Ignore whether the item was sampled from a predicted edge or a withheld candidate.
- Return only the structured result."""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an RQ1 edge-validity gold-label benchmark"
    )
    parser.add_argument(
        "--base-gold-labels",
        default="data/evaluation/rq1_gold/gold_labels.jsonl",
        help="Existing RQ1 gold-label file whose clause labels will be reused",
    )
    parser.add_argument(
        "--artifact-root",
        default="derived/rq1_gold",
        help="Artifact root containing per-skill reconciliation artifacts",
    )
    parser.add_argument(
        "--output-data-dir",
        default="data/evaluation/rq1_edge_validity_gold",
        help="Directory for generated edge-validity gold labels and summaries",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for the edge-validity gold builder",
    )
    parser.add_argument("--base-url", help="LLM API base URL override")
    parser.add_argument("--model", help="LLM model name override")
    parser.add_argument("--api-key-env", help="API key env var or literal key override")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
    parser.add_argument(
        "--positive-quota",
        type=int,
        default=2,
        help="Maximum positive samples per edge type per skill",
    )
    parser.add_argument(
        "--negative-quota",
        type=int,
        default=2,
        help="Maximum negative samples per edge type per skill",
    )
    args = parser.parse_args()

    env_config_path = Path(args.env_config)
    env_config = load_env_config(env_config_path) if env_config_path.is_file() else None
    try:
        llm_config = resolve_llm_config(
            llm_config_path=Path(args.llm_config),
            env_config=env_config,
            base_url=args.base_url,
            model=args.model,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    client = CachedLLMClient.from_config(llm_config, "evaluation_rq1_edge_validity_v1")

    base_records = load_gold_label_records(Path(args.base_gold_labels))
    artifact_root = Path(args.artifact_root)
    output_data_dir = Path(args.output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    output_records: list[GoldLabelRecord] = []
    manifest: list[dict[str, object]] = []

    for base_record in base_records:
        artifact_dir = artifact_root / base_record.skill_id
        if not artifact_dir.exists():
            continue

        contract_payload = json.loads((artifact_dir / "contract_table.json").read_text(encoding="utf-8"))
        clauses = [Clause.model_validate(item) for item in contract_payload.get("clauses", [])]
        steps = [Step.model_validate(item) for item in contract_payload.get("steps", [])]
        edges = _load_edges(artifact_dir / "reconciliation_edges.json")
        events = _load_model_list(artifact_dir / "event_table.json", CapabilityEvent)
        resources = _load_model_list(artifact_dir / "resource_table.json", ResourceUse)
        candidates = _load_model_list(artifact_dir / "candidate_pairs.json", CandidatePair)
        alignment_candidates = _load_model_list(
            artifact_dir / "alignment_candidates.json",
            StepUnitCandidate,
        )
        judgments = _load_model_list(artifact_dir / "judgment_table.json", ReconciliationJudgment)
        code_pack = json.loads((artifact_dir / "code_pack.json").read_text(encoding="utf-8"))
        unit_paths = {
            str(unit_id): str(path)
            for unit_id, path in code_pack.get("unit_paths", {}).items()
        }

        sampled_items = _sample_edge_items(
            clauses=clauses,
            edges=edges,
            candidates=candidates,
            alignment_candidates=alignment_candidates,
            judgments=judgments,
            positive_quota=args.positive_quota,
            negative_quota=args.negative_quota,
        )
        if not sampled_items:
            continue

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_prompt(
                    skill_id=base_record.skill_id,
                    skill_root=Path("data/skill_dataset") / base_record.skill_id,
                    sampled_items=sampled_items,
                    clauses=clauses,
                    steps=steps,
                    events=events,
                    resources=resources,
                    unit_paths=unit_paths,
                ),
            },
        ]
        judgment = client.structured_complete(
            messages,
            _EdgeValidityBundle,
            skill_id=base_record.skill_id,
            call_key=_call_key_for_sampled_items(sampled_items),
        )

        output_records.append(
            GoldLabelRecord(
                skill_id=base_record.skill_id,
                gold=GoldLabel(
                    label=base_record.gold.label,
                    violation_subtype=base_record.gold.violation_subtype,
                    rationale=base_record.gold.rationale,
                ),
                risk_stratum=base_record.risk_stratum,
                bucket=base_record.bucket,
                clause_labels=base_record.clause_labels,
                edge_labels=[
                    EdgeAnnotation(
                        edge_type=item.edge_type,
                        signature=item.signature,
                        expected_correct=item.expected_correct,
                        evidence_refs=[],
                    )
                    for item in judgment.edges
                ],
                expected_sites=base_record.expected_sites,
                metadata={
                    **sanitize_gold_metadata(base_record.metadata),
                    "edge_validity_benchmark": True,
                    "positive_quota": args.positive_quota,
                    "negative_quota": args.negative_quota,
                },
            )
        )
        manifest.append(
            {
                "skill_id": base_record.skill_id,
                "sampled_items": len(sampled_items),
                "sampled_positive": sum(1 for item in sampled_items if item.predicted),
                "sampled_negative": sum(1 for item in sampled_items if not item.predicted),
                "sampled_by_type": {
                    edge_type: {
                        "positive": sum(
                            1
                            for item in sampled_items
                            if item.edge_type == edge_type and item.predicted
                        ),
                        "negative": sum(
                            1
                            for item in sampled_items
                            if item.edge_type == edge_type and not item.predicted
                        ),
                    }
                    for edge_type in _EDGE_TYPES
                },
            }
        )

    gold_labels_path = output_data_dir / "gold_labels.jsonl"
    write_jsonl_models(gold_labels_path, output_records)

    predicted_edges_by_skill = {
        record.skill_id: _load_edges(artifact_root / record.skill_id / "reconciliation_edges.json")
        for record in output_records
    }
    validity = compute_edge_validity_by_type(output_records, predicted_edges_by_skill)
    summary = {
        "edge_validity_by_type": validity,
        "macro_bal_valid": _macro_average(validity, _EDGE_TYPES),
        "critical_macro_bal_valid": _macro_average(validity, _CRITICAL_EDGE_TYPES),
        "judged_positive": sum(int(metric["judged_positive"]) for metric in validity.values()),
        "judged_negative": sum(int(metric["judged_negative"]) for metric in validity.values()),
        "skills": len(output_records),
    }
    (output_data_dir / "manifest.json").write_text(
        json.dumps({"items": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_data_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gold_labels": str(gold_labels_path),
                "summary": str(output_data_dir / "summary.json"),
                "skills": len(output_records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_model_list(path: Path, model_cls: type[BaseModel]) -> list[BaseModel]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [model_cls.model_validate(item) for item in payload]


def _sample_edge_items(
    *,
    clauses: list[Clause],
    edges: list[ReconciliationEdge],
    candidates: list[CandidatePair],
    alignment_candidates: list[StepUnitCandidate],
    judgments: list[ReconciliationJudgment],
    positive_quota: int,
    negative_quota: int,
) -> list[_SampledEdgeItem]:
    clause_by_id = {clause.clause_id: clause for clause in clauses}
    predicted_signatures_by_type: dict[str, set[str]] = {
        edge_type: set()
        for edge_type in _EDGE_TYPES
    }
    predicted_items_by_type: dict[str, list[_SampledEdgeItem]] = {
        edge_type: []
        for edge_type in _EDGE_TYPES
    }
    for edge in sorted(edges, key=lambda item: item.edge_id):
        edge_type = edge.relation.value
        if edge_type not in predicted_items_by_type:
            continue
        signature = _edge_signature(
            edge_type=edge_type,
            clause_id=edge.clause_id,
            constraint_id=edge.constraint_id,
            event_id=edge.event_id,
            resource_id=edge.resource_id,
            step_id=edge.step_id,
            code_unit_id=edge.code_unit_id,
        )
        predicted_signatures_by_type[edge_type].add(signature)
        predicted_items_by_type[edge_type].append(
            _SampledEdgeItem(
                signature=signature,
                edge_type=edge_type,
                clause_id=edge.clause_id,
                constraint_id=edge.constraint_id,
                event_id=edge.event_id,
                resource_id=edge.resource_id,
                step_id=edge.step_id,
                code_unit_id=edge.code_unit_id,
                predicted=True,
                candidate_sources=[source.value for source in edge.candidate_sources],
            )
        )

    items: list[_SampledEdgeItem] = []
    seen_signatures: set[str] = set()
    for edge_type in _EDGE_TYPES:
        for item in predicted_items_by_type[edge_type][:positive_quota]:
            if item.signature in seen_signatures:
                continue
            items.append(item)
            seen_signatures.add(item.signature)
        for item in _negative_pool_for_type(
            edge_type=edge_type,
            clauses=clauses,
            clause_by_id=clause_by_id,
            candidates=candidates,
            alignment_candidates=alignment_candidates,
            judgments=judgments,
            predicted_signatures=predicted_signatures_by_type[edge_type],
        )[:negative_quota]:
            if item.signature in seen_signatures:
                continue
            items.append(item)
            seen_signatures.add(item.signature)
    return items


def _negative_pool_for_type(
    *,
    edge_type: str,
    clauses: list[Clause],
    clause_by_id: dict[str, Clause],
    candidates: list[CandidatePair],
    alignment_candidates: list[StepUnitCandidate],
    judgments: list[ReconciliationJudgment],
    predicted_signatures: set[str],
) -> list[_SampledEdgeItem]:
    if edge_type in {"supports", "potentially_supports", "contradicts", "relates_to"}:
        operator = {
            "supports": ClauseOperator.ALLOWED,
            "potentially_supports": ClauseOperator.ALLOWED,
            "contradicts": ClauseOperator.PROHIBITED,
            "relates_to": ClauseOperator.UNKNOWN,
        }[edge_type]
        pool: list[tuple[tuple[float, int], _SampledEdgeItem]] = []
        for candidate in candidates:
            if candidate.behavior_kind.value != "event" or not candidate.event_id:
                continue
            clause = clause_by_id.get(candidate.clause_id)
            if clause is None or clause.operator != operator:
                continue
            signature = _edge_signature(
                edge_type=edge_type,
                clause_id=candidate.clause_id,
                event_id=candidate.event_id,
            )
            if signature in predicted_signatures:
                continue
            pool.append(
                (
                    _candidate_score(candidate),
                    _SampledEdgeItem(
                        signature=signature,
                        edge_type=edge_type,
                        clause_id=candidate.clause_id,
                        event_id=candidate.event_id,
                        predicted=False,
                        candidate_sources=[source.value for source in candidate.candidate_sources],
                    ),
                )
            )
        return [item for _score, item in sorted(pool, key=lambda pair: pair[0], reverse=True)]

    if edge_type in {"scope_matches", "scope_violates"}:
        owner_clause_by_constraint = {
            constraint.constraint_id: clause
            for clause in clauses
            for constraint in clause.constraints
        }
        pool = []
        for judgment in judgments:
            if judgment.kind != JudgmentKind.SCOPE_CHECK:
                continue
            if len(judgment.subject_refs) < 2:
                continue
            constraint_id = judgment.subject_refs[0]
            resource_id = judgment.subject_refs[1]
            owner_clause = owner_clause_by_constraint.get(constraint_id)
            if owner_clause is None:
                continue
            if edge_type == "scope_violates" and owner_clause.operator != ClauseOperator.ALLOWED:
                continue
            signature = _edge_signature(
                edge_type=edge_type,
                constraint_id=constraint_id,
                resource_id=resource_id,
            )
            if signature in predicted_signatures:
                continue
            pool.append(
                (
                    _scope_judgment_score(judgment),
                    _SampledEdgeItem(
                        signature=signature,
                        edge_type=edge_type,
                        clause_id=owner_clause.clause_id,
                        constraint_id=constraint_id,
                        resource_id=resource_id,
                        predicted=False,
                    ),
                )
            )
        return [item for _score, item in sorted(pool, key=lambda pair: pair[0], reverse=True)]

    if edge_type == "aligns":
        pool = []
        for candidate in alignment_candidates:
            signature = _edge_signature(
                edge_type=edge_type,
                step_id=candidate.step_id,
                code_unit_id=candidate.code_unit_id,
            )
            if signature in predicted_signatures:
                continue
            pool.append(
                (
                    _alignment_candidate_score(candidate),
                    _SampledEdgeItem(
                        signature=signature,
                        edge_type=edge_type,
                        step_id=candidate.step_id,
                        code_unit_id=candidate.code_unit_id,
                        predicted=False,
                        candidate_sources=[source.value for source in candidate.candidate_sources],
                    ),
                )
            )
        return [item for _score, item in sorted(pool, key=lambda pair: pair[0], reverse=True)]

    return []


def _candidate_score(candidate: CandidatePair) -> tuple[float, int]:
    semantic_score = float(candidate.shared_signals.get("semantic_score", "0") or 0.0)
    return semantic_score, len(candidate.candidate_sources)


def _scope_judgment_score(judgment: ReconciliationJudgment) -> tuple[int, int]:
    result_order = {
        "false": 2,
        "abstain": 1,
        "true": 0,
    }
    route_bonus = 1 if judgment.route_support_type is not None else 0
    return result_order.get(judgment.result.value, 0), route_bonus


def _alignment_candidate_score(candidate: StepUnitCandidate) -> tuple[float, int]:
    semantic_score = float(candidate.shared_signals.get("semantic_score", "0") or 0.0)
    return semantic_score, len(candidate.candidate_sources)


def _build_prompt(
    *,
    skill_id: str,
    skill_root: Path,
    sampled_items: list[_SampledEdgeItem],
    clauses: list[Clause],
    steps: list[Step],
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    unit_paths: dict[str, str],
) -> str:
    docs_text: list[str] = []
    for rel in ["SKILL.md", "README.md"]:
        path = skill_root / rel
        if path.exists():
            docs_text.append(
                f"## {rel}\n{path.read_text(encoding='utf-8', errors='ignore')[:7000]}"
            )

    clause_by_id = {clause.clause_id: clause for clause in clauses}
    step_by_id = {step.step_id: step for step in steps}
    event_by_id = {event.event_id: event for event in events}
    resource_by_id = {resource.resource_id: resource for resource in resources}
    owner_clause_by_constraint = {
        constraint.constraint_id: clause
        for clause in clauses
        for constraint in clause.constraints
    }

    item_text = "\n\n".join(
        _format_sampled_item(
            item,
            clause_by_id=clause_by_id,
            step_by_id=step_by_id,
            event_by_id=event_by_id,
            resource_by_id=resource_by_id,
            owner_clause_by_constraint=owner_clause_by_constraint,
            unit_paths=unit_paths,
        )
        for item in sampled_items
    )
    return "\n\n".join(
        [
            f"Skill: {skill_id}",
            "Documentation:",
            *docs_text,
            "Candidate reconciliation items to judge:",
            item_text,
        ]
    )


def _format_sampled_item(
    item: _SampledEdgeItem,
    *,
    clause_by_id: dict[str, Clause],
    step_by_id: dict[str, Step],
    event_by_id: dict[str, CapabilityEvent],
    resource_by_id: dict[str, ResourceUse],
    owner_clause_by_constraint: dict[str, Clause],
    unit_paths: dict[str, str],
) -> str:
    parts = [f"signature={item.signature}", f"relation={item.edge_type}"]
    clause = clause_by_id.get(item.clause_id or "")
    if clause is not None:
        parts.append(f"clause={_format_clause(clause)}")
    if item.constraint_id:
        owner_clause = owner_clause_by_constraint.get(item.constraint_id)
        constraint = next(
            (
                constraint
                for constraint in (owner_clause.constraints if owner_clause is not None else [])
                if constraint.constraint_id == item.constraint_id
            ),
            None,
        )
        if constraint is not None:
            parts.append(f"constraint={_format_constraint(constraint)}")
        if owner_clause is not None and clause is None:
            parts.append(f"owner_clause={_format_clause(owner_clause)}")
    if item.event_id and item.event_id in event_by_id:
        parts.append(f"event={_format_event(event_by_id[item.event_id], unit_paths)}")
    if item.resource_id and item.resource_id in resource_by_id:
        parts.append(f"resource={_format_resource(resource_by_id[item.resource_id], unit_paths)}")
    if item.step_id and item.step_id in step_by_id:
        parts.append(f"step={_format_step(step_by_id[item.step_id])}")
    if item.code_unit_id:
        parts.append(
            f"code_unit={item.code_unit_id} path={unit_paths.get(item.code_unit_id, 'unknown')}"
        )
    return "\n".join(parts)


def _format_clause(clause: Clause) -> str:
    evidence = " | ".join(
        span.text.strip().replace("\n", " ")[:160]
        for span in clause.evidence_spans[:2]
    )
    constraints = "; ".join(constraint.value for constraint in clause.constraints[:2])
    return (
        f"{clause.clause_id} operator={clause.operator.value} capability={clause.capability} "
        f"target={clause.target or 'none'} constraints={constraints or 'none'} "
        f"evidence={evidence or 'none'}"
    )


def _format_constraint(constraint: Constraint) -> str:
    evidence = constraint.evidence.text.replace("\n", " ")[:160] if constraint.evidence else "none"
    return (
        f"{constraint.constraint_id} type={constraint.constraint_type} "
        f"value={constraint.value} evidence={evidence}"
    )


def _format_step(step: Step) -> str:
    evidence = step.evidence.text.replace("\n", " ")[:160] if step.evidence else "none"
    return (
        f"{step.step_id} type={step.step_type} heading={step.heading_context or 'none'} "
        f"text={step.text[:180]} evidence={evidence}"
    )


def _format_event(event: CapabilityEvent, unit_paths: dict[str, str]) -> str:
    return (
        f"{event.event_id} capability={event.capability} unit={event.unit_id} "
        f"path={unit_paths.get(event.unit_id, event.file_path or 'unknown')} "
        f"location={event.location} detail={(event.detail or event.api_call)[:180]}"
    )


def _format_resource(resource: ResourceUse, unit_paths: dict[str, str]) -> str:
    return (
        f"{resource.resource_id} type={resource.resource_type} value={resource.value} "
        f"unit={resource.unit_id} path={unit_paths.get(resource.unit_id, 'unknown')} "
        f"resolved={resource.resolved} location={resource.location}"
    )


def _edge_signature(
    *,
    edge_type: str,
    clause_id: str | None = None,
    constraint_id: str | None = None,
    event_id: str | None = None,
    resource_id: str | None = None,
    step_id: str | None = None,
    code_unit_id: str | None = None,
) -> str:
    return "|".join(
        [
            edge_type,
            clause_id or "",
            constraint_id or "",
            event_id or "",
            resource_id or "",
            "",
            step_id or "",
            code_unit_id or "",
        ]
    )


def _macro_average(
    validity: dict[str, dict[str, float | int | None]],
    edge_types: tuple[str, ...],
) -> float | None:
    values = [
        float(metric["bal_valid"])
        for edge_type in edge_types
        if edge_type in validity and metric_has_bal_valid(validity[edge_type])
        for metric in [validity[edge_type]]
    ]
    if not values:
        return None
    return sum(values) / len(values)


def metric_has_bal_valid(metric: dict[str, float | int | None]) -> bool:
    return metric.get("bal_valid") is not None


def _call_key_for_sampled_items(sampled_items: list[_SampledEdgeItem]) -> str:
    signature_blob = "\n".join(item.signature for item in sampled_items)
    digest = hashlib.sha256(signature_blob.encode("utf-8")).hexdigest()[:10]
    return f"rq1_edge_validity_judge_{digest}"


if __name__ == "__main__":
    main()
