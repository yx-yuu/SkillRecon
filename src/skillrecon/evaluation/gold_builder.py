Gold-label construction for paper evaluation datasets."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from skillrecon.evaluation.datasets import (
    GoldLabelRecord,
    FlaggedSkillRecord,
    GoldLabel,
    sanitize_gold_metadata,
)
from skillrecon.loader.path_resolver import iter_skill_path_candidates
from skillrecon.llm.cache import CachedLLMClient

PROMPT_VERSION = "paper500_gold_v2"
_SHORT_TEXT = StringConstraints(max_length=300)
_MEDIUM_TEXT = StringConstraints(max_length=600)
ShortText = Annotated[str, _SHORT_TEXT]
MediumText = Annotated[str, _MEDIUM_TEXT]

_DOC_FILENAMES = (
    "SKILL.md",
    "README.md",
    "README_CN.md",
    "README.zh-CN.md",
    "INSTALLATION.md",
    "USAGE.md",
    "_manifest.json",
    "_meta.json",
    "skill.json",
    "package.json",
)
_CODE_SUFFIXES = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
}


class GoldBuildConfig(BaseModel):
    """Prompt and source-pack limits for gold-label generation."""

    model_config = ConfigDict(frozen=True)

    prompt_version: str = PROMPT_VERSION
    max_file_chars: int = 6000
    max_source_files: int = 18
    max_total_chars: int = 36000
    max_tokens: int | None = None


class SourceExcerpt(BaseModel):
    """One source excerpt passed to the builder."""

    model_config = ConfigDict(frozen=True)

    path: str
    kind: Literal["doc", "code", "metadata"]
    text: str


class SourceBundle(BaseModel):
    """Skill source pack used for one gold-label generation request."""

    model_config = ConfigDict(frozen=True)

    skill_id: str
    skill_path: str
    excerpts: list[SourceExcerpt]
    source_hash: str


class SourceCollectionError(RuntimeError):
    """Raised when gold label construction cannot access required skill source."""


class PolicyMapping(BaseModel):
    """LLM-produced mapping from source/policy text to candidate behavior."""

    model_config = ConfigDict(frozen=True)

    policy: ShortText
    behavior: ShortText
    status: Literal["allowed", "prohibited", "unknown"]
    evidence_refs: list[ShortText] = Field(default_factory=list, max_length=4)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"allowed", "authorized", "declared", "documented"}:
            return "allowed"
        if normalized in {
            "prohibited",
            "forbidden",
            "disallowed",
            "denied",
            "unauthorized",
            "unsupported",
            "contradicted",
            "out_of_scope",
            "scope_violation",
        }:
            return "prohibited"
        if normalized in {"unknown", "unclear", "ambiguous", "not_enough_evidence"}:
            return "unknown"
        return value


class DependenceJudgment(BaseModel):
    """LLM-produced judgment about workflow/payload dependence."""

    model_config = ConfigDict(frozen=True)

    source: ShortText
    target: ShortText
    relation: ShortText
    rationale: MediumText


class AuthorizationJudgment(BaseModel):
    """LLM-produced judgment about whether a behavior is authorized."""

    model_config = ConfigDict(frozen=True)

    behavior: ShortText
    authorized: bool
    violation_type: ShortText | None = None
    rationale: MediumText


class GoldLabelDecision(BaseModel):
    """Structured output for one generated gold label."""

    model_config = ConfigDict(frozen=True)

    source_content: list[MediumText] = Field(max_length=8)
    workflow: list[MediumText] = Field(max_length=8)
    payloads: list[ShortText] = Field(max_length=8)
    policy_mappings: list[PolicyMapping] = Field(max_length=12)
    dependence_judgments: list[DependenceJudgment] = Field(max_length=12)
    authorization_judgments: list[AuthorizationJudgment] = Field(max_length=12)
    gold: GoldLabel

    @field_validator(
        "policy_mappings",
        "dependence_judgments",
        "authorization_judgments",
        mode="before",
    )
    @classmethod
    def _drop_invalid_audit_items(cls, value: object, info: ValidationInfo) -> object:
        if not isinstance(value, list):
            return value
        required_fields = {
            "policy_mappings": ("policy", "behavior", "status"),
            "dependence_judgments": ("source", "target", "relation", "rationale"),
            "authorization_judgments": ("behavior", "authorized", "rationale"),
        }[info.field_name]
        return [
            item
            for item in value
            if not (
                isinstance(item, dict)
                and not any(
                    field_value not in (None, "", [], {})
                    for field_value in item.values()
                )
            )
            and not (
                isinstance(item, dict)
                and any(item.get(field) in (None, "", [], {}) for field in required_fields)
            )
        ]

    @model_validator(mode="after")
    def _validate_gold_subtype(self) -> "GoldLabelDecision":
        if self.gold.label == "violation" and not self.gold.violation_subtype:
            raise ValueError("violation gold labels must include violation_subtype")
        return self


def build_gold_label_records(
    records: Sequence[FlaggedSkillRecord],
    *,
    dataset_root: Path,
    client: CachedLLMClient,
    build_config: GoldBuildConfig | None = None,
    existing_records: Sequence[GoldLabelRecord] | None = None,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> list[GoldLabelRecord]:
    """Generate complete gold-label records with an LLM builder."""
    config = build_config or GoldBuildConfig()
    existing_by_skill = {
        record.skill_id: record
        for record in (existing_records or [])
    }
    output: list[GoldLabelRecord] = []
    for record in records:
        if record.skill_id in existing_by_skill:
            output.append(existing_by_skill[record.skill_id])
            continue

        source_bundle = collect_source_bundle(
            record,
            dataset_root=dataset_root,
            build_config=config,
            windows_drive_map=windows_drive_map,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_prompt(record, source_bundle),
            },
        ]
        decision = client.structured_complete(
            messages,
            GoldLabelDecision,
            skill_id=record.skill_id,
            call_key=f"{config.prompt_version}_{source_bundle.source_hash}",
            max_tokens=config.max_tokens,
        )
        output.append(
            _to_gold_label_record(
                record=record,
                source_bundle=source_bundle,
                decision=decision,
                client=client,
                build_config=config,
            )
        )
    return output


def collect_source_bundle(
    record: FlaggedSkillRecord,
    *,
    dataset_root: Path,
    build_config: GoldBuildConfig | None = None,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None = None,
) -> SourceBundle:
    """Collect bounded source excerpts for one skill."""
    config = build_config or GoldBuildConfig()
    skill_path = record.resolve_skill_path(
        dataset_root,
        windows_drive_map=windows_drive_map,
    )
    if not skill_path.is_dir():
        raise SourceCollectionError(
            _missing_source_message(
                record,
                dataset_root,
                windows_drive_map=windows_drive_map,
            )
        )

    excerpts: list[SourceExcerpt] = []

    for rel_path in _candidate_source_paths(skill_path):
        kind = _source_kind(rel_path)
        if kind is None:
            continue
        text = _read_text_excerpt(rel_path, max_chars=config.max_file_chars)
        if not text:
            continue
        excerpts.append(
            SourceExcerpt(
                path=_safe_relative_path(rel_path, skill_path),
                kind=kind,
                text=text,
            )
        )
        if len(excerpts) >= config.max_source_files:
            break

    excerpts = _trim_total_chars(excerpts, config.max_total_chars)
    if not excerpts:
        raise SourceCollectionError(
            f"No readable source excerpts found for {record.skill_id} at {skill_path}. "
            "Expected at least one documentation, metadata, or code file."
        )

    source_hash = _source_hash(record, excerpts)
    return SourceBundle(
        skill_id=record.skill_id,
        skill_path=skill_path.as_posix(),
        excerpts=excerpts,
        source_hash=source_hash,
    )


def _to_gold_label_record(
    *,
    record: FlaggedSkillRecord,
    source_bundle: SourceBundle,
    decision: GoldLabelDecision,
    client: CachedLLMClient,
    build_config: GoldBuildConfig,
) -> GoldLabelRecord:
    generation = {
        "generator": "ai",
        "prompt_version": build_config.prompt_version,
        "model": client.config.model,
        "base_url": client.config.base_url,
        "temperature": client.config.temperature,
        "max_tokens": client.config.max_tokens,
        "source_hash": source_bundle.source_hash,
        "source_files": [excerpt.path for excerpt in source_bundle.excerpts],
    }
    metadata = sanitize_gold_metadata(
        {
            "display_name": record.skill.display_name or record.slug,
            "skill_version": record.version,
            "dataset_bucket": record.dataset_bucket,
            "script_types": list(record.script_types),
            "virus_total_status": record.virus_total_status,
            "openclaw_status": record.openclaw_status,
            "scan_confidence": record.skill.security_scan.confidence,
            "scan_summary": record.skill.security_scan.summary,
            "scan_sections": dict(record.skill.security_scan.sections),
            "purpose_capability": record.purpose_capability,
            "instruction_scope": record.instruction_scope,
            "install_mechanism": record.install_mechanism,
            "credentials": record.credentials,
            "persistence_privilege": record.persistence_privilege,
            "assessment": record.assessment,
            "evidence_refs": [excerpt.path for excerpt in source_bundle.excerpts],
            "source_content": list(decision.source_content),
            "workflow": list(decision.workflow),
            "payloads": list(decision.payloads),
            "policy_mappings": [
                item.model_dump(exclude_none=False)
                for item in decision.policy_mappings
            ],
            "dependence_judgments": [
                item.model_dump(exclude_none=False)
                for item in decision.dependence_judgments
            ],
            "authorization_judgments": [
                item.model_dump(exclude_none=False)
                for item in decision.authorization_judgments
            ],
            "gold_label_generation": generation,
            "gold_label_rationale": decision.gold.rationale,
        }
    )
    return GoldLabelRecord(
        skill_id=record.skill_id,
        gold=decision.gold,
        risk_stratum=record.risk_tier,
        bucket=record.dataset_bucket,
        clause_labels=[],
        edge_labels=[],
        expected_sites=[],
        metadata=metadata,
    )


def _candidate_source_paths(skill_path: Path) -> list[Path]:
    candidates: list[Path] = []
    for filename in _DOC_FILENAMES:
        path = skill_path / filename
        if path.is_file():
            candidates.append(path)

    for path in sorted(skill_path.rglob("*")):
        if not path.is_file() or path in candidates:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _CODE_SUFFIXES:
            continue
        candidates.append(path)
    return candidates


def _source_kind(path: Path) -> Literal["doc", "code", "metadata"] | None:
    name = path.name
    if name in {"_manifest.json", "_meta.json", "skill.json", "package.json"}:
        return "metadata"
    if name.lower().startswith("readme") or name == "SKILL.md" or path.suffix == ".md":
        return "doc"
    if path.suffix.lower() in _CODE_SUFFIXES:
        return "code"
    return None


def _missing_source_message(
    record: FlaggedSkillRecord,
    dataset_root: Path,
    *,
    windows_drive_map: Mapping[str, str | os.PathLike[str]] | None,
) -> str:
    candidates = [
        candidate.as_posix()
        for candidate in iter_skill_path_candidates(
            dataset_root,
            record.owner,
            record.slug,
            extract_root=record.extract_root,
            windows_drive_map=windows_drive_map,
        )
    ]
    return (
        f"Missing local source directory for {record.skill_id}. "
        f"Checked: {candidates}. "
        "If the corpus archive lives on a Windows drive, mount that drive in WSL "
        "or set SKILLRECON_DRIVE_E_ROOT / pass --drive-map E=/path/to/root to "
        "the WSL-readable root that contains 'clawhub_skills'."
    )


def _read_text_excerpt(path: Path, *, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    return text[:max_chars]


def _trim_total_chars(
    excerpts: list[SourceExcerpt],
    max_total_chars: int,
) -> list[SourceExcerpt]:
    remaining = max_total_chars
    trimmed: list[SourceExcerpt] = []
    for excerpt in excerpts:
        if remaining <= 0:
            break
        text = excerpt.text[:remaining]
        if text:
            trimmed.append(excerpt.model_copy(update={"text": text}))
            remaining -= len(text)
    return trimmed


def _source_hash(record: FlaggedSkillRecord, excerpts: list[SourceExcerpt]) -> str:
    payload = {
        "skill_id": record.skill_id,
        "version": record.version,
        "risk_tier": record.risk_tier,
        "excerpts": [excerpt.model_dump() for excerpt in excerpts],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _build_prompt(record: FlaggedSkillRecord, source_bundle: SourceBundle) -> str:
    metadata = {
        "skill_id": record.skill_id,
        "display_name": record.skill.display_name,
        "version": record.version,
        "risk_tier": record.risk_tier,
        "dataset_bucket": record.dataset_bucket,
        "script_types": record.script_types,
        "virus_total_status": record.virus_total_status,
        "openclaw_status": record.openclaw_status,
        "scan_confidence": record.skill.security_scan.confidence,
        "scan_summary": record.skill.security_scan.summary,
        "purpose_capability": record.purpose_capability,
        "instruction_scope": record.instruction_scope,
        "install_mechanism": record.install_mechanism,
        "credentials": record.credentials,
        "persistence_privilege": record.persistence_privilege,
        "assessment": record.assessment,
    }
    source_text = "\n\n".join(
        [
            f"### {excerpt.kind}: {excerpt.path}\n{excerpt.text}"
            for excerpt in source_bundle.excerpts
        ]
    )
    return "\n\n".join(
        [
            f"Skill id: {record.skill_id}",
            "Corpus metadata:",
            json.dumps(metadata, ensure_ascii=False, indent=2),
            "Source excerpts:",
            source_text or "No local source excerpts were available.",
            "Return the structured dataset record now.",
        ]
    )


_SYSTEM_PROMPT = """You construct the gold-label dataset for a paper evaluation of agent skills.

Hard constraints:
- The dataset construction step is fully automatically generated. Do not ask for human input.
- The final action label must be represented only as the gold label.
- Do not create any additional label-like fields or alternate labeling channels.
- Human inspection, if any, happens offline after this dataset is frozen and edits the gold field directly.

For the given skill, infer all dataset contents from the provided source and metadata:
1. source_content: up to 8 concise source facts that matter for security behavior.
2. workflow: up to 8 runtime steps the skill asks or enables an agent to perform.
3. payloads: up to 8 files, commands, URLs, credentials, prompts, scripts, or data flows.
4. policy_mappings: up to 12 documentation/policy statements mapped to candidate behavior.
5. dependence_judgments: up to 12 workflow/payload dependence judgments.
6. authorization_judgments: up to 12 authorization judgments for important behavior.
7. gold: the final gold label.

Output constraints:
- Return compact JSON only. No Markdown fences.
- Keep each string under 80 words.
- Keep gold.rationale under 120 words.
- Prefer evidence-grounded phrases over long narrative.

Gold label policy:
- violation: at least one behavior is unauthorized, contradicted by policy, outside declared scope, or only becomes dangerous through an unjustified composition.
- exposure-only: security-sensitive behavior exists but is sufficiently declared and scoped.
- benign: no material security-sensitive inconsistency is supported by the source.
- If gold.label is violation, gold.violation_subtype must be exactly one of:
  unsupported_behavior, contradicted_behavior, scope_violation, unjustified_composition.
- If gold.label is exposure-only or benign, gold.violation_subtype must be null.
- Be conservative and evidence-grounded. Use scanner metadata as context, not as the gold label itself.
"""
