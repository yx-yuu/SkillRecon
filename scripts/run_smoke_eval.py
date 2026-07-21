#!/usr/bin/env python3
"""Run configured smoke-evaluation pytest suites."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object config in {path}")
    suites = payload.get("suites")
    if not isinstance(suites, dict) or not suites:
        raise ValueError(f"Smoke config {path} must define non-empty suites")
    return payload


def _select_suite(config: dict, suite_name: str | None) -> tuple[str, dict]:
    suites = config["suites"]
    selected = suite_name or config.get("default_suite")
    if not selected:
        raise ValueError("Smoke config must define default_suite or pass --suite")
    suite = suites.get(selected)
    if not isinstance(suite, dict):
        raise ValueError(f"Unknown smoke suite: {selected}")
    nodeids = suite.get("nodeids")
    if not isinstance(nodeids, list) or not nodeids:
        raise ValueError(f"Smoke suite {selected} must define non-empty nodeids")
    return selected, suite


def _build_pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path if not existing else os.pathsep.join([src_path, existing])
    )
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a configured SkillRecon smoke-evaluation suite"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to smoke evaluation config JSON",
    )
    parser.add_argument("--suite", help="Named suite in the config to execute")
    parser.add_argument(
        "--json-out",
        help="Optional path to write machine-readable summary JSON",
    )
    parser.add_argument(
        "--list-suites",
        action="store_true",
        help="List available suites and exit",
    )

    args = parser.parse_args()
    config_path = Path(args.config)

    try:
        config = _load_config(config_path)
        suites = config["suites"]
        if args.list_suites:
            for suite_name, suite in suites.items():
                description = suite.get("description", "")
                print(f"{suite_name}: {description}".rstrip())
            return

        suite_name, suite = _select_suite(config, args.suite)
        nodeids = list(suite["nodeids"])
        cmd = ["pytest", "-q", *nodeids]
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=_build_pytest_env(),
            check=False,
        )
    except Exception as exc:
        print(f"Smoke harness failed to start: {exc}", file=sys.stderr)
        sys.exit(1)

    summary = {
        "config_path": str(config_path),
        "suite": suite_name,
        "description": suite.get("description", ""),
        "nodeids": nodeids,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
    }

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    status = "passed" if completed.returncode == 0 else "failed"
    print(f"Smoke suite {suite_name} {status} ({len(nodeids)} checks).")

    if completed.returncode != 0:
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
