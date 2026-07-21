#!/usr/bin/env python3
"""Run gold-label dataset construction in resumable shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from skillrecon.evaluation.datasets import load_gold_label_records, load_paper_sample_records


@dataclass(frozen=True)
class Shard:
    index: int
    offset: int
    limit: int
    output_dir: Path

    @property
    def gold_labels_path(self) -> Path:
        return self.output_dir / "gold_labels.jsonl"

    @property
    def log_path(self) -> Path:
        return self.output_dir / "run.log"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scripts/prepare_paper_dataset.py in resumable shards"
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
        "--output-root",
        default="temp/paper500_gold_shards_v2/full",
        help="Directory where shard outputs are stored",
    )
    parser.add_argument("--shard-size", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--llm-config", default="experiments/configs/llm_config.json")
    parser.add_argument("--api-key-env", default="SKILLRECON_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.shard_size <= 0:
        parser.error("--shard-size must be positive")
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.start < 0:
        parser.error("--start must be non-negative")
    if args.stop is not None and args.stop <= args.start:
        parser.error("--stop must be greater than --start")

    paper_dataset = Path(args.paper_dataset)
    records_by_slice = load_paper_sample_records(paper_dataset)
    total = sum(len(records) for records in records_by_slice.values())
    stop = min(args.stop if args.stop is not None else total, total)
    shards = _make_shards(
        output_root=Path(args.output_root),
        start=args.start,
        stop=stop,
        shard_size=args.shard_size,
    )
    pending = [
        shard
        for shard in shards
        if _gold_count(shard.gold_labels_path) != shard.limit
    ]

    summary = {
        "paper_dataset": str(paper_dataset),
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "total_records": total,
        "range": [args.start, stop],
        "shard_size": args.shard_size,
        "max_workers": args.max_workers,
        "shard_count": len(shards),
        "pending_count": len(pending),
        "pending": [
            {
                "index": shard.index,
                "offset": shard.offset,
                "limit": shard.limit,
                "output": str(shard.gold_labels_path),
            }
            for shard in pending
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run or not pending:
        return

    env = dict(os.environ)
    if args.api_key_env not in env:
        parser.error(f"{args.api_key_env} is not set")
    env["PYTHONPATH"] = "src"

    failures: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_shard = {
            executor.submit(_run_shard, shard, args=args, env=env): shard
            for shard in pending
        }
        for future in as_completed(future_to_shard):
            shard = future_to_shard[future]
            result = future.result()
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["returncode"] != 0 or result["records"] != shard.limit:
                failures.append(result)

    report = {
        "completed": len(pending) - len(failures),
        "failed": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


def _make_shards(
    *,
    output_root: Path,
    start: int,
    stop: int,
    shard_size: int,
) -> list[Shard]:
    shards: list[Shard] = []
    shard_index = 0
    for offset in range(start, stop, shard_size):
        limit = min(shard_size, stop - offset)
        output_dir = output_root / f"shard_{offset:04d}_{offset + limit:04d}"
        shards.append(Shard(index=shard_index, offset=offset, limit=limit, output_dir=output_dir))
        shard_index += 1
    return shards


def _gold_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(load_gold_label_records(path))
    except Exception:
        return 0


def _run_shard( shard: Shard, *, args: argparse.Namespace, env: dict[str, str]) -> dict[str, object]:
    shard.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/prepare_paper_dataset.py",
        "--paper-dataset",
        args.paper_dataset,
        "--dataset-root",
        args.dataset_root,
        "--offset",
        str(shard.offset),
        "--limit",
        str(shard.limit),
        "--gold-labels-out",
        str(shard.gold_labels_path),
        "--resume",
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--llm-config",
        args.llm_config,
        "--max-tokens",
        str(args.max_tokens),
    ]
    with shard.log_path.open("a", encoding="utf-8") as log:
        log.write("\n=== run shard ===\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log,
            stderr=log,
            text=True,
            check=False,
        )
    return {
        "index": shard.index,
        "offset": shard.offset,
        "limit": shard.limit,
        "returncode": completed.returncode,
        "records": _gold_count(shard.gold_labels_path),
        "output": str(shard.gold_labels_path),
        "log": str(shard.log_path),
    }


if __name__ == "__main__":
    main()
