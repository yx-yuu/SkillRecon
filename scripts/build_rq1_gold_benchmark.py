#!/usr/bin/env python3
"""Build an AI-generated RQ1 gold-label benchmark from explicit items."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.core.types import (
    CapabilityEvent,
    Clause,
    ReconciliationEdge,
    ResourceUse,
    RiskPath,
)
from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    ClauseAnnotation,
    EdgeAnnotation,
    GoldLabel,
    sanitize_gold_metadata,
    write_jsonl_models,
)
from skillrecon.llm.cache import CachedLLMClient


class _ClauseJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)

    clause_id: str
    operator: Literal["allowed", "prohibited", "unknown"]
    rationale: str = ""


class _EdgeJudgment(BaseModel):
    model_config = ConfigDict(frozen=True)

    signature: str
    edge_type: Literal["supports", "contradicts", "scope_matches", "scope_violates"]
    expected_correct: bool
    rationale: str = ""


class _RQ1JudgmentBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    clauses: list[_ClauseJudgment]
    edges: list[_EdgeJudgment]


class _BenchmarkItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    skill_id: str
    artifact_dir: str


_SYSTEM_PROMPT = """You are an exacting evaluator for documentation-derived authorization semantics in AI agent skills.

Task 1: For each provided predicted clause, judge its gold operator as exactly one of:
- allowed
- prohibited
- unknown

Task 2: For each provided sampled reconciliation edge, judge whether the edge relation is correct.

Rules:
- Be conservative.
- Use unknown when the documentation mentions a capability but does not clearly authorize the exact target or scope.
- Do not reward a clause merely because the implementation exists.
- Judge only the provided clauses and provided sampled edges.
- Return only the structured result."""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build an AI-generated RQ1 gold-label benchmark"
    )
    parser.add_argument(
        "--items-json",
        required=True,
        help="JSON file listing benchmark items with skill_id and artifact_dir",
    )
    parser.add_argument(
        "--output-data-dir",
        default="data/evaluation/rq1_gold",
        help="Directory for generated RQ1 gold-label files",
    )
    parser.add_argument(
        "--output-artifact-root",
        default="derived/rq1_gold",
        help="Artifact root containing copied benchmark artifacts",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for the AI RQ1 gold builder",
    )
    parser.add_argument("--base-url", help="LLM API base URL override")
    parser.add_argument("--model", help="LLM model name override")
    parser.add_argument("--api-key-env", help="API key env var or literal key override")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument("--max-tokens", type=int, help="LLM max_tokens override")
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
    client = CachedLLMClient.from_config(llm_config, "evaluation_rq1_gold_v1")
    output_data_dir = Path(args.output_data_dir)
    output_artifact_root = Path(args.output_artifact_root)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_artifact_root.mkdir(parents=True, exist_ok=True)

    benchmark_items = _load_items(Path(args.items_json))
    records: list[GoldLabelRecord] = []
    manifest_items: list[dict[str, object]] = []
    for item in benchmark_items:
        skill_id = item.skill_id
        artifact_dir = Path(item.artifact_dir)
        copied_dir = output_artifact_root / skill_id
        if copied_dir.exists():
            shutil.rmtree(copied_dir)
        copied_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(artifact_dir, copied_dir)

        skill_path = _resolve_skill_path(skill_id)
        clauses = _load_model_list(copied_dir / "contract_table.json", Clause, key="clauses")
        edges = _load_model_list(copied_dir / "reconciliation_edges.json", ReconciliationEdge)
        sampled_edges = _sample_edges(edges)
        events = _load_model_list(copied_dir / "event_table.json", CapabilityEvent)
        resources = _load_model_list(copied_dir / "resource_table.json", ResourceUse)
        paths = _load_model_list(copied_dir / "path_table.json", RiskPath)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_prompt(
                    skill_id=skill_id,
                    skill_path=skill_path,
                    clauses=clauses,
                    sampled_edges=sampled_edges,
                    events=events,
                    resources=resources,
                    paths=paths,
                ),
            },
        ]
        judgment = client.structured_complete(
            messages,
            _RQ1JudgmentBundle,
            skill_id=skill_id,
            call_key="rq1_gold_judge",
        )

        clause_map = {clause.clause_id: clause for clause in clauses}
        clause_labels = [
            _to_clause_annotation(clause_map[item.clause_id], item.operator)
            for item in judgment.clauses
            if item.clause_id in clause_map
        ]
        edge_labels = [
            EdgeAnnotation(
                edge_type=item.edge_type,
                signature=item.signature,
                expected_correct=item.expected_correct,
                evidence_refs=[],
            )
            for item in judgment.edges
        ]
        records.append(
            GoldLabelRecord(
                skill_id=skill_id,
                gold=GoldLabel(label="benign"),
                risk_stratum="rq1_gold",
                bucket="rq1_gold",
                clause_labels=clause_labels,
                edge_labels=edge_labels,
                metadata=sanitize_gold_metadata(
                    {
                        "assessment": "ai_generated_rq1_gold_benchmark",
                    }
                ),
            )
        )
        manifest_items.append(
            {
                "skill_id": skill_id,
                "clause_count": len(clause_labels),
                "edge_count": len(edge_labels),
            }
        )

    gold_labels_path = output_data_dir / "gold_labels.jsonl"
    write_jsonl_models(gold_labels_path, records)
    (output_data_dir / "manifest.json").write_text(
        json.dumps({"items": manifest_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gold_labels": str(gold_labels_path),
                "artifact_root": str(output_artifact_root),
                "skills": len(records),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_model_list(path: Path, model_cls: type, *, key: str | None = None) -> list:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if key is not None:
        payload = payload.get(key, [])
    return [model_cls.model_validate(item) for item in payload]


def _load_items(path: Path) -> list[_BenchmarkItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [_BenchmarkItem.model_validate(item) for item in payload]


def _resolve_skill_path(skill_id: str) -> Path:
    nested = Path("data/skill_dataset") / skill_id
    flat = Path("data/skill_dataset") / skill_id.split("/")[-1]
    if nested.exists():
        return nested
    return flat


def _sample_edges(edges: list[ReconciliationEdge]) -> list[ReconciliationEdge]:
    quotas = {
        "supports": 4,
        "contradicts": 4,
        "scope_matches": 4,
        "scope_violates": 6,
    }
    selected: list[ReconciliationEdge] = []
    for relation_name, limit in quotas.items():
        relation_edges = [
            edge
            for edge in edges
            if edge.relation.value == relation_name
        ]
        relation_edges.sort(key=lambda edge: edge.edge_id)
        selected.extend(relation_edges[:limit])
    return selected


def _build_prompt(
    *,
    skill_id: str,
    skill_path: Path,
    clauses: list[Clause],
    sampled_edges: list[ReconciliationEdge],
    events: list[CapabilityEvent],
    resources: list[ResourceUse],
    paths: list[RiskPath],
) -> str:
    docs_text = []
    for rel in ["SKILL.md", "README.md"]:
        path = skill_path / rel
        if path.exists():
            docs_text.append(f"## {rel}\n{path.read_text(encoding='utf-8', errors='ignore')[:7000]}")
    clauses_text = "\n".join(_format_clause(clause) for clause in clauses)
    clause_by_id = {clause.clause_id: clause for clause in clauses}
    event_by_id = {event.event_id: event for event in events}
    resource_by_id = {resource.resource_id: resource for resource in resources}
    path_by_id = {path.path_id: path for path in paths}
    edges_text = "\n\n".join(
        _format_edge(
            edge,
            clause_by_id=clause_by_id,
            event_by_id=event_by_id,
            resource_by_id=resource_by_id,
            path_by_id=path_by_id,
        )
        for edge in sampled_edges
    )
    return "\n\n".join(
        [
            f"Skill: {skill_id}",
            "Documentation:",
            *docs_text,
            "Predicted clauses to judge:",
            clauses_text,
            "Sampled reconciliation edges to judge:",
            edges_text,
        ]
    )


def _format_clause(clause: Clause) -> str:
    evidence = " | ".join(span.text.strip().replace("\n", " ")[:180] for span in clause.evidence_spans[:2])
    constraints = "; ".join(constraint.value for constraint in clause.constraints)
    return (
        f"- clause_id={clause.clause_id} predicted_operator={clause.operator.value} "
        f"capability={clause.capability} target={clause.target or 'none'} "
        f"constraints={constraints or 'none'} evidence={evidence or 'none'}"
    )


def _format_edge(
    edge: ReconciliationEdge,
    *,
    clause_by_id: dict[str, Clause],
    event_by_id: dict[str, CapabilityEvent],
    resource_by_id: dict[str, ResourceUse],
    path_by_id: dict[str, RiskPath],
) -> str:
    parts = [f"signature={_edge_signature(edge)}", f"relation={edge.relation.value}"]
    clause = clause_by_id.get(edge.clause_id or "")
    if clause is not None:
        parts.append(f"clause={_format_clause(clause)}")
    if edge.event_id and edge.event_id in event_by_id:
        event = event_by_id[edge.event_id]
        parts.append(
            "event="
            f"{event.event_id} capability={event.capability} "
            f"location={event.location} detail={(event.detail or event.api_call)[:160]}"
        )
    if edge.resource_id and edge.resource_id in resource_by_id:
        resource = resource_by_id[edge.resource_id]
        parts.append(
            "resource="
            f"{resource.resource_id} type={resource.resource_type} "
            f"value={resource.value} location={resource.location}"
        )
    if edge.path_id and edge.path_id in path_by_id:
        path = path_by_id[edge.path_id]
        parts.append(
            "path="
            f"{path.path_id} {path.source.label} -> {path.sink.label} kind={path.path_kind}"
        )
    return "\n".join(parts)


def _to_clause_annotation(clause: Clause, operator: str) -> ClauseAnnotation:
    return ClauseAnnotation(
        signature=_clause_signature(clause, operator),
        operator=operator,
        capability=clause.capability,
        target=clause.target,
        constraints=[constraint.value for constraint in clause.constraints],
        evidence_refs=[span.text for span in clause.evidence_spans[:2]],
    )


def _clause_signature(clause: Clause, operator: str) -> str:
    return "|".join(
        [
            operator,
            clause.capability,
            (clause.target or "").strip().lower(),
            ";".join(sorted(constraint.value.strip().lower() for constraint in clause.constraints)),
        ]
    )


def _edge_signature(edge: ReconciliationEdge) -> str:
    return "|".join(
        [
            edge.relation.value,
            edge.clause_id or "",
            edge.constraint_id or "",
            edge.event_id or "",
            edge.resource_id or "",
            edge.path_id or "",
            edge.step_id or "",
            edge.code_unit_id or "",
        ]
    )


if __name__ == "__main__":
    main()
