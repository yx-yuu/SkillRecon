#!/usr/bin/env python3
"""Copy corpus index files into this repository and rewrite stored paths."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from skillrecon.loader.path_resolver import (
    parse_windows_drive_map,
    rewrite_dataset_path_string,
)

_ALL_SUBSETS = ("dataset_index", "test_index", "evaluation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy dataset index metadata from the full SkillRecon data directory "
            "and rewrite Windows/WSL paths for the current runtime."
        )
    )
    parser.add_argument(
        "--source-data-root",
        required=True,
        help="External data directory used as the read-only index source",
    )
    parser.add_argument(
        "--output-data-root",
        default="data",
        help="Current-repository data directory to write rewritten metadata into",
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=_ALL_SUBSETS,
        help=(
            "Subdirectory to copy; may be passed multiple times. "
            "Defaults to all dataset index metadata subsets."
        ),
    )
    parser.add_argument(
        "--drive-map",
        action="append",
        default=[],
        metavar="DRIVE=ROOT",
        help="Override Windows drive mapping, e.g. E=/mnt/e or E=/data/clawhub",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copy operations without writing files",
    )
    args = parser.parse_args()

    source_root = Path(args.source_data_root)
    output_root = Path(args.output_data_root)
    subsets = tuple(args.subset) if args.subset else _ALL_SUBSETS
    try:
        drive_map = parse_windows_drive_map(args.drive_map)
    except ValueError as exc:
        parser.error(str(exc))

    summary = {
        "source_data_root": str(source_root),
        "output_data_root": str(output_root),
        "subsets": list(subsets),
        "dry_run": args.dry_run,
        "files": [],
    }

    for subset in subsets:
        source_dir = source_root / subset
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source subset: {source_dir}")
        for source_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
            relative_path = source_path.relative_to(source_root)
            output_path = output_root / relative_path
            action = _copy_or_rewrite_file(
                source_path,
                output_path,
                windows_drive_map=drive_map,
                dry_run=args.dry_run,
            )
            summary["files"].append(
                {
                    "source": str(source_path),
                    "output": str(output_path),
                    "action": action,
                }
            )

    summary["file_count"] = len(summary["files"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _copy_or_rewrite_file(
    source_path: Path,
    output_path: Path,
    *,
    windows_drive_map: Mapping[str, str | Path],
    dry_run: bool,
) -> str:
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open(encoding="utf-8") as source, output_path.open(
                "w",
                encoding="utf-8",
            ) as output:
                for line in source:
                    stripped = line.strip()
                    if not stripped:
                        output.write(line)
                        continue
                    payload = _rewrite_json_paths(
                        json.loads(stripped),
                        windows_drive_map=windows_drive_map,
                    )
                    output.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return "rewrite_jsonl"

    if suffix == ".json":
        if not dry_run:
            payload = _rewrite_json_paths(
                json.loads(source_path.read_text(encoding="utf-8")),
                windows_drive_map=windows_drive_map,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return "rewrite_json"

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
    return "copy"


def _rewrite_json_paths(
    value: Any,
    *,
    windows_drive_map: Mapping[str, str | Path],
) -> Any:
    if isinstance(value, str):
        return rewrite_dataset_path_string(value, windows_drive_map=windows_drive_map)
    if isinstance(value, list):
        return [
            _rewrite_json_paths(item, windows_drive_map=windows_drive_map)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _rewrite_json_paths(item, windows_drive_map=windows_drive_map)
            for key, item in value.items()
        }
    return value


if __name__ == "__main__":
    main()
