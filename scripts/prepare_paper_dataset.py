#!/usr/bin/env python3
"""Generate gold labels and baseline files for the paper500 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from skillrecon.core.config import (
    DEFAULT_ENV_CONFIG_PATH,
    DEFAULT_LLM_CONFIG_PATH,
    load_env_config,
    resolve_llm_config,
)
from skillrecon.evaluation.datasets import (
    build_openclaw_predictions,
    load_gold_label_records,
    load_paper_sample_records,
    write_jsonl_models,
)
from skillrecon.evaluation.gold_builder import (
    GoldBuildConfig,
    PROMPT_VERSION,
    SourceCollectionError,
    build_gold_label_records,
)
from skillrecon.loader.path_resolver import parse_windows_drive_map
from skillrecon.llm.cache import CachedLLMClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build gold labels and baseline predictions for paper500"
    )
    parser.add_argument(
        "--paper-dataset",
        default="data/evaluation/skill_paper500_dataset",
        help="Paper High/Medium/Low sample dataset directory",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/skill_dataset",
        help="Root directory containing local skill packages",
    )
    parser.add_argument(
        "--openclaw-unresolved-label",
        choices=["violation", "exposure-only", "benign"],
        help="Optional fallback label for unresolved OpenClaw statuses",
    )
    parser.add_argument(
        "--gold-labels-out",
        default=None,
        help="Output gold-label JSONL path (defaults to <paper-dataset>/gold_labels.jsonl)",
    )
    parser.add_argument(
        "--env-config",
        default=str(DEFAULT_ENV_CONFIG_PATH),
        help="Path to env_config.json for non-LLM local defaults",
    )
    parser.add_argument(
        "--llm-config",
        default=str(DEFAULT_LLM_CONFIG_PATH),
        help="Path to llm_config.json for the gold-label generator",
    )
    parser.add_argument(
        "--drive-map",
        action="append",
        default=[],
        metavar="DRIVE=ROOT",
        help=(
            "Override Windows drive mapping for source paths, e.g. "
            "E=/mnt/e or E=/data/clawhub-drive"
        ),
    )
    parser.add_argument("--base-url", help="LLM API base URL override")
    parser.add_argument("--model", help="LLM model name override")
    parser.add_argument("--api-key-env", help="API key env var or literal key override")
    parser.add_argument("--temperature", type=float, help="LLM temperature override")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing records in the gold-label output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of sample records to generate in this invocation",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many sample records before applying --limit; useful for shards",
    )
    parser.add_argument(
        "--max-source-files",
        type=int,
        default=18,
        help="Maximum source files included in each gold-label request",
    )
    parser.add_argument(
        "--max-file-chars",
        type=int,
        default=6000,
        help="Maximum characters read from each source file",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=36000,
        help="Maximum total source characters included in each request",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Optional per-call max_tokens override for gold-label generation",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help=(
            "Write gold-label progress after this many newly generated records; "
            "set 0 to write only at the end"
        ),
    )
    args = parser.parse_args()
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.offset < 0:
        parser.error("--offset must be non-negative")

    paper_dataset = Path(args.paper_dataset)
    dataset_root = Path(args.dataset_root)
    gold_label_path = (
        Path(args.gold_labels_out)
        if args.gold_labels_out
        else paper_dataset / "gold_labels.jsonl"
    )
    output_data_dir = (
        paper_dataset
        if gold_label_path.parent == paper_dataset
        else gold_label_path.parent
    )
    try:
        drive_map = parse_windows_drive_map(args.drive_map)
    except ValueError as exc:
        parser.error(str(exc))

    records_by_slice = load_paper_sample_records(paper_dataset)
    sample = [
        record
        for slice_name in ("high", "medium", "low")
        for record in records_by_slice[slice_name]
    ]
    if args.offset:
        sample = sample[args.offset:]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        sample = sample[: args.limit]

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
    client = CachedLLMClient.from_config(llm_config, PROMPT_VERSION)
    existing_records = (
        load_gold_label_records(gold_label_path)
        if args.resume and gold_label_path.is_file()
        else []
    )
    try:
        gold_labels = _build_gold_labels_with_checkpoints(
            sample,
            dataset_root=dataset_root,
            client=client,
            build_config=GoldBuildConfig(
                max_file_chars=args.max_file_chars,
                max_source_files=args.max_source_files,
                max_total_chars=args.max_total_chars,
                max_tokens=args.max_tokens,
            ),
            existing_records=existing_records,
            windows_drive_map=drive_map,
            output_path=gold_label_path,
            checkpoint_every=args.checkpoint_every,
        )
    except SourceCollectionError as exc:
        parser.error(str(exc))
    openclaw_predictions = build_openclaw_predictions(
        sample,
        unresolved_label=args.openclaw_unresolved_label,
    )

    write_jsonl_models(gold_label_path, gold_labels)
    openclaw_prediction_path = output_data_dir / "baseline_predictions" / "openclaw.jsonl"
    skill_list_path = output_data_dir / "all_skills.txt"
    summary_path = output_data_dir / "paper_dataset_summary.json"
    write_jsonl_models(
        openclaw_prediction_path,
        openclaw_predictions,
    )
    _write_skill_list(skill_list_path, sample)
    _write_summary(
        summary_path,
        paper_dataset=paper_dataset,
        dataset_root=dataset_root,
        records_by_slice=records_by_slice,
        sample=sample,
        gold_label_path=gold_label_path,
        gold_labels=gold_labels,
        openclaw_predictions=openclaw_predictions,
        output_root=output_data_dir,
    )

    sample_skill_ids = {record.skill_id for record in sample}
    print(
        json.dumps(
            {
                "paper_dataset": str(paper_dataset),
                "dataset_root": str(dataset_root),
                "offset": args.offset,
                "sample_size": len(sample),
                "slices": {
                    slice_name: sum(
                        1
                        for record in records
                        if record.skill_id in sample_skill_ids
                    )
                    for slice_name, records in records_by_slice.items()
                },
                "gold_labels": str(gold_label_path),
                "openclaw_predictions": str(openclaw_prediction_path),
                "skill_list": str(skill_list_path),
                "summary": str(summary_path),
                "gold_label_count": len(gold_labels),
                "openclaw_prediction_coverage": len(openclaw_predictions),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _build_gold_labels_with_checkpoints(
    sample,
    *,
    dataset_root: Path,
    client: CachedLLMClient,
    build_config: GoldBuildConfig,
    existing_records,
    windows_drive_map,
    output_path: Path,
    checkpoint_every: int,
):
    existing_by_skill = {
        record.skill_id: record
        for record in existing_records
    }
    gold_labels = []
    generated_since_checkpoint = 0

    for index, record in enumerate(sample, start=1):
        if record.skill_id in existing_by_skill:
            gold_labels.append(existing_by_skill[record.skill_id])
            continue

        generated = build_gold_label_records(
            [record],
            dataset_root=dataset_root,
            client=client,
            build_config=build_config,
            existing_records=[],
            windows_drive_map=windows_drive_map,
        )[0]
        existing_by_skill[record.skill_id] = generated
        gold_labels.append(generated)
        generated_since_checkpoint += 1

        if checkpoint_every and generated_since_checkpoint >= checkpoint_every:
            write_jsonl_models(output_path, gold_labels)
            generated_since_checkpoint = 0
            print(
                f"checkpoint: wrote {len(gold_labels)}/{len(sample)} gold labels",
                file=sys.stderr,
                flush=True,
            )

    return gold_labels


def _write_skill_list(path: Path, sample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{record.skill_id}\n" for record in sample),
        encoding="utf-8",
    )


def _write_summary(
    path: Path,
    *,
    paper_dataset: Path,
    dataset_root: Path,
    records_by_slice,
    sample,
    gold_label_path: Path,
    gold_labels,
    openclaw_predictions,
    output_root: Path,
) -> None:
    prediction_skills = {record.skill_id for record in openclaw_predictions}
    sample_skill_ids = {record.skill_id for record in sample}
    summary = {
        "paper_dataset": str(paper_dataset),
        "dataset_root": str(dataset_root),
        "sample_size": len(sample),
        "slice_counts": {
            slice_name: sum(
                1 for record in records if record.skill_id in sample_skill_ids
            )
            for slice_name, records in records_by_slice.items()
        },
        "risk_tier_counts": dict(Counter(record.risk_tier for record in sample)),
        "dataset_bucket_counts": dict(
            Counter(record.dataset_bucket for record in sample)
        ),
        "openclaw_status_counts": dict(
            Counter(record.openclaw_status for record in sample)
        ),
        "openclaw_prediction_coverage": len(prediction_skills),
        "openclaw_prediction_missing": sorted(
            record.skill_id
            for record in sample
            if record.skill_id not in prediction_skills
        ),
        "gold_label_count": len(gold_labels),
        "gold_label_missing": sorted(
            record.skill_id
            for record in sample
            if record.skill_id not in {item.skill_id for item in gold_labels}
        ),
        "artifacts": {
            "gold_labels": _relative_to(gold_label_path, output_root),
            "openclaw_predictions": "baseline_predictions/openclaw.jsonl",
            "skill_list": "all_skills.txt",
        },
        "notes": [
            "sample_index.jsonl files are the paper sample list.",
            "gold_labels.jsonl stores the gold label in each record's gold field.",
            "Offline calibration edits the gold field directly; downstream experiments read the same field.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    main()
