"""Dataset schemas and loaders for corpus-level evaluation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from random import Random
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from skillrecon.loader.path_resolver import resolve_skill_path_from_index

T = TypeVar("T", bound=BaseModel)

_DEFAULT_PILOT_RISK_WEIGHTS = {
    "high_risk": 0.4,
    "medium_risk": 0.4,
    "review_needed": 0.2,
}
_DEFAULT_PILOT_BUCKET_ORDER = (
    "pure_py",
    "pure_bash",
    "mixed",
    "pure_js",
    "pure_ts",
    "skill_md_only",
)
_PAPER_SLICE_NAMES = ("high", "medium", "low")

_SAFE_GOLD_METADATA_KEYS = (
    "display_name",
    "skill_version",
    "dataset_bucket",
    "script_types",
    "virus_total_status",
    "openclaw_status",
    "scan_confidence",
    "scan_summary",
    "scan_sections",
    "purpose_capability",
    "instruction_scope",
    "install_mechanism",
    "credentials",
    "persistence_privilege",
    "assessment",
    "evidence_refs",
    "source_content",
    "workflow",
    "payloads",
    "policy_mappings",
    "dependence_judgments",
    "authorization_judgments",
    "gold_label_generation",
    "gold_label_rationale",
)

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
    rationale: str = ""

    @model_validator(mode="after")
    def _validate_subtype(self) -> "GoldLabel":
        if self.label != "violation" and self.violation_subtype is not None:
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


class SeededBenchmarkRecord(BaseModel):
    """One seeded-injection variant used for RQ2/RQ3 controlled evaluation."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    base_skill_id: str
    injection_family: str
    gold: GoldLabel
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


class SecurityScanSnapshot(BaseModel):
    """Minimal security-scan summary kept from the crawled corpus index."""

    model_config = ConfigDict(frozen=True)

    virus_total_status: str | None = None
    openclaw_status: str | None = None
    confidence: str | None = None
    summary: str | None = None
    sections: dict[str, str] = {}
    virus_total_report_url: str | None = None


class SkillIndexMetadata(BaseModel):
    """Compact per-skill metadata loaded from the crawl index."""

    model_config = ConfigDict(frozen=True)

    owner: str | None = None
    slug: str | None = None
    display_name: str | None = None
    page_url: str | None = None
    description: str | None = None
    list_rank_by_downloads: int | None = None
    security_scan: SecurityScanSnapshot = SecurityScanSnapshot()


class FlaggedSkillRecord(BaseModel):
    """One crawled skill entry from ``flagged_security_scan_skills*.jsonl``."""

    model_config = ConfigDict(frozen=True)

    dataset_bucket: str
    owner: str
    slug: str
    version: str
    script_types: list[str] = []
    extract_root: str
    manifest_path: str | None = None
    detail_html: str | None = None
    risk_tier: str | None = None
    purpose_capability: str | None = None
    instruction_scope: str | None = None
    install_mechanism: str | None = None
    credentials: str | None = None
    persistence_privilege: str | None = None
    assessment: str | None = None
    skill: SkillIndexMetadata = SkillIndexMetadata()

    @property
    def skill_id(self) -> str:
        """Return the owner-qualified local skill identifier."""
        return f"{self.owner}/{self.slug}"

    @property
    def openclaw_status(self) -> str | None:
        """Return the OpenClaw scan status if present."""
        return self.skill.security_scan.openclaw_status

    @property
    def virus_total_status(self) -> str | None:
        """Return the VirusTotal scan status if present."""
        return self.skill.security_scan.virus_total_status

    def resolve_skill_path(
        self,
        dataset_root: Path,
        *,
        windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
    ) -> Path:
        """Resolve the local skill directory inside the repository dataset root."""
        return resolve_skill_path_from_index(
            dataset_root,
            self.owner,
            self.slug,
            extract_root=self.extract_root,
            windows_drive_map=windows_drive_map,
        )


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


def load_seeded_benchmark_records(path: Path) -> list[SeededBenchmarkRecord]:
    """Load seeded-injection benchmark records from .json or .jsonl."""
    return _load_records(path, SeededBenchmarkRecord)


def load_baseline_prediction_records(path: Path) -> list[BaselinePredictionRecord]:
    """Load baseline predictions from .json or .jsonl."""
    return _load_records(path, BaselinePredictionRecord)


def load_flagged_skill_records(path: Path) -> list[FlaggedSkillRecord]:
    """Load crawled flagged-skill corpus records from .json or .jsonl."""
    return _load_records(path, FlaggedSkillRecord)


def load_paper_sample_records(
    dataset_root: Path,
    *,
    slice_names: tuple[str, ...] = _PAPER_SLICE_NAMES,
) -> dict[str, list[FlaggedSkillRecord]]:
    """Load the paper-facing High/Medium/Low sample-index dataset."""
    records_by_slice: dict[str, list[FlaggedSkillRecord]] = {}
    for slice_name in slice_names:
        sample_path = dataset_root / slice_name / "sample_index.jsonl"
        if not sample_path.is_file():
            raise FileNotFoundError(
                f"Missing paper sample index for slice {slice_name!r}: {sample_path}"
            )
        records_by_slice[slice_name] = load_flagged_skill_records(sample_path)
    return records_by_slice


def flatten_paper_sample_records(
    dataset_root: Path,
    *,
    slice_names: tuple[str, ...] = _PAPER_SLICE_NAMES,
) -> list[FlaggedSkillRecord]:
    """Load all paper sample-index records in stable slice order."""
    records_by_slice = load_paper_sample_records(
        dataset_root,
        slice_names=slice_names,
    )
    return [
        record
        for slice_name in slice_names
        for record in records_by_slice[slice_name]
    ]


def paper_sample_skill_ids(
    dataset_root: Path,
    *,
    slice_names: tuple[str, ...] = _PAPER_SLICE_NAMES,
) -> set[str]:
    """Return the skill ids present in the paper sample-index dataset."""
    return {
        record.skill_id
        for record in flatten_paper_sample_records(
            dataset_root,
            slice_names=slice_names,
        )
    }


def compare_gold_labels_to_paper_sample(
    gold_labels: list[GoldLabelRecord],
    dataset_root: Path,
) -> dict[str, list[str]]:
    """Compare generated gold labels with the paper sample index."""
    expected = paper_sample_skill_ids(dataset_root)
    actual = {record.skill_id for record in gold_labels}
    return {
        "missing": sorted(expected - actual),
        "extra": sorted(actual - expected),
    }


def _load_records(path: Path, model_cls: type[T]) -> list[T]:
    if path.suffix == ".jsonl":
        return load_jsonl_models(path, model_cls)
    if path.suffix == ".json":
        return load_json_models(path, model_cls)
    raise ValueError(f"Unsupported dataset extension for {path}")


def select_pilot_sample(
    records: list[FlaggedSkillRecord],
    *,
    sample_size: int = 20,
    seed: int = 20260326,
    resolved_openclaw_only: bool = True,
    risk_weights: dict[str, float] | None = None,
    bucket_order: tuple[str, ...] = _DEFAULT_PILOT_BUCKET_ORDER,
) -> list[FlaggedSkillRecord]:
    """Select a deterministic, stratified pilot sample from the flagged corpus."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    weights = risk_weights or _DEFAULT_PILOT_RISK_WEIGHTS
    filtered = [
        record
        for record in records
        if not resolved_openclaw_only
        or _map_openclaw_status_to_label(record.openclaw_status) is not None
    ]
    quotas = _allocate_counts(sample_size, weights)
    rng = Random(seed)

    grouped: dict[str, dict[str, list[FlaggedSkillRecord]]] = {
        risk_tier: {bucket: [] for bucket in bucket_order}
        for risk_tier in quotas
    }
    leftovers_by_risk: dict[str, list[FlaggedSkillRecord]] = {
        risk_tier: []
        for risk_tier in quotas
    }
    for record in filtered:
        risk_tier = record.risk_tier
        if risk_tier not in grouped:
            continue
        if record.dataset_bucket in grouped[risk_tier]:
            grouped[risk_tier][record.dataset_bucket].append(record)
        else:
            leftovers_by_risk[risk_tier].append(record)

    for risk_tier in grouped:
        for bucket in bucket_order:
            rng.shuffle(grouped[risk_tier][bucket])
        rng.shuffle(leftovers_by_risk[risk_tier])

    selected: list[FlaggedSkillRecord] = []
    for risk_tier, quota in quotas.items():
        risk_selected = _take_round_robin(
            grouped_by_bucket=grouped[risk_tier],
            bucket_order=bucket_order,
            quota=quota,
        )
        if len(risk_selected) < quota:
            remaining = quota - len(risk_selected)
            risk_selected.extend(leftovers_by_risk[risk_tier][:remaining])
            leftovers_by_risk[risk_tier] = leftovers_by_risk[risk_tier][remaining:]
        if len(risk_selected) < quota:
            raise ValueError(
                f"Not enough samples for risk_tier={risk_tier!r}: need {quota}, got {len(risk_selected)}"
            )
        selected.extend(risk_selected)

    return sorted(
        selected,
        key=lambda record: (
            record.risk_tier or "",
            record.dataset_bucket,
            record.skill_id,
        ),
    )


def select_bucket_stratified_sample(
    records: list[FlaggedSkillRecord],
    *,
    sample_size: int,
    seed: int = 20260327,
    bucket_order: tuple[str, ...] = _DEFAULT_PILOT_BUCKET_ORDER,
    minimum_bucket_counts: dict[str, int] | None = None,
) -> list[FlaggedSkillRecord]:
    """Select a deterministic one-slice paper sample stratified by bucket."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(records) < sample_size:
        raise ValueError(
            f"Not enough records to sample {sample_size} items; only {len(records)} available"
        )

    minimums = {bucket: 0 for bucket in bucket_order}
    for bucket, count in (minimum_bucket_counts or {}).items():
        if bucket not in minimums:
            raise ValueError(f"Unknown dataset bucket: {bucket!r}")
        if count < 0:
            raise ValueError("minimum bucket counts must be non-negative")
        minimums[bucket] = count
    if sum(minimums.values()) > sample_size:
        raise ValueError("minimum bucket counts exceed sample_size")

    rng = Random(seed)
    grouped: dict[str, list[FlaggedSkillRecord]] = {
        bucket: []
        for bucket in bucket_order
    }
    leftovers: list[FlaggedSkillRecord] = []
    for record in records:
        if record.dataset_bucket in grouped:
            grouped[record.dataset_bucket].append(record)
        else:
            leftovers.append(record)

    for bucket in bucket_order:
        rng.shuffle(grouped[bucket])
    rng.shuffle(leftovers)

    reserved: dict[str, int] = {}
    for bucket in bucket_order:
        available = len(grouped[bucket])
        requested = minimums[bucket]
        if requested > available:
            raise ValueError(
                f"Bucket {bucket!r} requires at least {requested} samples, "
                f"but only {available} are available"
            )
        reserved[bucket] = requested

    remaining_quota = sample_size - sum(reserved.values())
    remaining_availability = {
        bucket: len(grouped[bucket]) - reserved[bucket]
        for bucket in bucket_order
    }
    additional = _allocate_counts_with_cap(
        total=remaining_quota,
        capacities=remaining_availability,
    )

    selected: list[FlaggedSkillRecord] = []
    for bucket in bucket_order:
        take = reserved[bucket] + additional.get(bucket, 0)
        for _ in range(take):
            selected.append(grouped[bucket].pop())

    if len(selected) < sample_size:
        spill_needed = sample_size - len(selected)
        spill_pool: list[FlaggedSkillRecord] = []
        for bucket in bucket_order:
            spill_pool.extend(grouped[bucket])
        spill_pool.extend(leftovers)
        if len(spill_pool) < spill_needed:
            raise ValueError(
                f"Unable to satisfy sample_size={sample_size}; "
                f"only {len(selected) + len(spill_pool)} selectable"
            )
        selected.extend(spill_pool[:spill_needed])

    return sorted(
        selected,
        key=lambda record: (
            record.dataset_bucket,
            record.skill_id,
        ),
    )


def sanitize_gold_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    """Keep stable metadata fields for generated gold-label artifacts."""
    if not metadata:
        return {}

    sanitized: dict[str, object] = {}
    for key in _SAFE_GOLD_METADATA_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, list):
            sanitized[key] = list(value)
        elif isinstance(value, dict):
            sanitized[key] = dict(value)
        else:
            sanitized[key] = value
    return sanitized


def build_openclaw_predictions(
    records: list[FlaggedSkillRecord],
    *,
    unresolved_label: Literal["violation", "exposure-only", "benign"] | None = None,
) -> list[BaselinePredictionRecord]:
    """Convert sampled corpus records into B4 OpenClaw baseline predictions."""
    predictions: list[BaselinePredictionRecord] = []
    for record in records:
        main_label = _map_openclaw_status_to_label(record.openclaw_status)
        if main_label is None:
            if unresolved_label is None:
                continue
            main_label = unresolved_label
        predictions.append(
            BaselinePredictionRecord(
                skill_id=record.skill_id,
                system_id="baseline_openclaw",
                main_label=main_label,
                rationale=(
                    "Mapped directly from corpus index field skill.security_scan.openclaw_status."
                ),
                metadata={
                    "openclaw_status": record.openclaw_status,
                    "virus_total_status": record.virus_total_status,
                    "risk_tier": record.risk_tier,
                },
            )
        )
    return predictions


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


def _allocate_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    if not weights:
        raise ValueError("weights must be non-empty")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    exact = {
        key: total * (value / weight_sum)
        for key, value in weights.items()
    }
    counts = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        weights,
        key=lambda key: (exact[key] - counts[key], key),
        reverse=True,
    )
    for key in remainders[:remaining]:
        counts[key] += 1
    return counts


def _allocate_counts_with_cap(total: int, capacities: dict[str, int]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if total > sum(capacities.values()):
        raise ValueError("total exceeds available capacity")
    if total == 0:
        return {key: 0 for key in capacities}

    active = {key: value for key, value in capacities.items() if value > 0}
    counts = {key: 0 for key in capacities}
    remaining = total
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for key in list(active):
            take = min(share, active[key], remaining)
            if take <= 0:
                continue
            counts[key] += take
            active[key] -= take
            remaining -= take
            progressed = True
            if active[key] == 0:
                del active[key]
            if remaining == 0:
                break
        if not progressed:
            break
    if remaining != 0:
        raise ValueError("Unable to allocate requested total under capacities")
    return counts


def _take_round_robin(
    *,
    grouped_by_bucket: dict[str, list[FlaggedSkillRecord]],
    bucket_order: tuple[str, ...],
    quota: int,
) -> list[FlaggedSkillRecord]:
    selected: list[FlaggedSkillRecord] = []
    while len(selected) < quota:
        progressed = False
        for bucket in bucket_order:
            candidates = grouped_by_bucket.get(bucket, [])
            if not candidates:
                continue
            selected.append(candidates.pop())
            progressed = True
            if len(selected) == quota:
                break
        if not progressed:
            break
    return selected


def _map_openclaw_status_to_label(status: str | None) -> Literal["violation", "benign"] | None:
    if status in {"Suspicious", "Malicious"}:
        return "violation"
    if status == "Benign":
        return "benign"
    return None
