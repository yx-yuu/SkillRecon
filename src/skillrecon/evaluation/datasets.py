"""Input annotation and normalized prediction schemas."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, TypeVar
from pydantic import BaseModel, ConfigDict, model_validator

T = TypeVar("T", bound=BaseModel)

class ClauseAnnotation(BaseModel):
    """A gold clause record used for RQ1 contract-induction evaluation."""

    model_config = ConfigDict(frozen=True)

    signature: str | None = None
    operator: str
    capability: str
    target: str | None = None
    constraints: list[str] = []
    evidence_refs: list[str] = []

class EdgeAnnotation(BaseModel):
    """A gold reconciliation-edge record used for RQ1 sampled edge accuracy."""

    model_config = ConfigDict(frozen=True)

    edge_type: str
    signature: str
    expected_correct: bool = True
    evidence_refs: list[str] = []

class GoldLabel(BaseModel):
    """The single supervision field used by evaluation datasets."""

    model_config = ConfigDict(frozen=True)

    label: Literal["violation", "exposure-only", "benign"]
    violation_subtype: str | None = None
    violation_subtypes: list[str] = []
    rationale: str = ""

    @model_validator(mode="after")
    def _validate_subtype(self) -> "GoldLabel":
        if self.label != "violation" and (self.violation_subtype is not None or self.violation_subtypes):
            raise ValueError("violation_subtype must be null unless gold.label is violation")
        return self

class GoldLabelRecord(BaseModel):
    """One skill-level record whose supervision is stored only in ``gold``."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    gold: GoldLabel
    risk_stratum: str | None = None
    bucket: str | None = None
    clause_labels: list[ClauseAnnotation] = []
    edge_labels: list[EdgeAnnotation] = []
    expected_sites: list[str] = []
    metadata: dict[str, object] = {}

class BaselinePredictionRecord(BaseModel):
    """A normalized baseline prediction for one skill."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    system_id: str
    main_label: Literal["violation", "exposure-only", "benign"]
    subtype: str | None = None
    rationale: str = ""
    score: float | None = None
    metadata: dict[str, object] = {}

def load_jsonl_models(path: Path, model_cls: type[T]) -> list[T]:
    """Load a JSONL file into validated Pydantic models."""
    payload: list[T] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload.append(model_cls.model_validate(json.loads(line)))
    return payload

def load_json_models(path: Path, model_cls: type[T]) -> list[T]:
    """Load a JSON array file into validated Pydantic models."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [model_cls.model_validate(item) for item in payload]

def load_gold_label_records(path: Path) -> list[GoldLabelRecord]:
    """Load gold-label records from .json or .jsonl."""
    return _load_records(path, GoldLabelRecord)

def load_baseline_prediction_records(path: Path) -> list[BaselinePredictionRecord]:
    """Load baseline predictions from .json or .jsonl."""
    return _load_records(path, BaselinePredictionRecord)

def _load_records(path: Path, model_cls: type[T]) -> list[T]:
    if path.suffix == ".jsonl":
        return load_jsonl_models(path, model_cls)
    if path.suffix == ".json":
        return load_json_models(path, model_cls)
    raise ValueError(f"Unsupported dataset extension for {path}")

def write_jsonl_models(path: Path, models: list[BaseModel]) -> None:
    """Write validated Pydantic records to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(model.model_dump(exclude_none=False), ensure_ascii=False) + "\n"
            for model in models
        ),
        encoding="utf-8",
    )
