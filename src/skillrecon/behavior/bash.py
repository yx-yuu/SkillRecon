"""Tier 3 conservative Bash extraction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from skillrecon.behavior.normalize import RawObservation
from skillrecon.core.types import PackageManifest

logger = logging.getLogger(__name__)


def extract_bash_observations(
    manifest: PackageManifest,
    unit_paths: dict[str, str],
    staged_root: Path,
    pattern_path: Path,
) -> list[RawObservation]:
    """Extract conservative capability observations from Bash scripts."""
    config = json.loads(pattern_path.read_text(encoding="utf-8"))
    pattern_map: dict[str, dict[str, list[str]]] = config.get("patterns", {})

    observations: list[RawObservation] = []
    bash_units = [unit for unit in manifest.code_units if unit.language == "bash"]

    for unit in bash_units:
        relative_path = unit_paths[unit.unit_id]
        file_path = staged_root / relative_path
        if not file_path.is_file():
            continue

        for line_no, line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            for cap_type, spec in pattern_map.items():
                matched = False
                matched_command = _best_matching_command(
                    stripped,
                    spec.get("commands", []),
                )
                if matched_command is not None:
                    observations.append(
                        RawObservation(
                            language="bash",
                            relative_path=relative_path,
                            line=line_no,
                            message=(
                                f"capType={cap_type} | detail={matched_command} | "
                                "tier=bash_pattern"
                            ),
                            fields={
                                "capType": cap_type,
                                "detail": matched_command,
                                "tier": "bash_pattern",
                            },
                        )
                    )
                    matched = True
                if matched:
                    continue

                for pattern in spec.get("patterns", []):
                    if re.search(pattern, stripped):
                        observations.append(
                            RawObservation(
                                language="bash",
                                relative_path=relative_path,
                                line=line_no,
                                message=(
                                    f"capType={cap_type} | detail={pattern} | "
                                    "tier=bash_pattern"
                                ),
                                fields={
                                    "capType": cap_type,
                                    "detail": pattern,
                                    "tier": "bash_pattern",
                                },
                            )
                        )
                        break

    logger.info("Extracted %d Bash observations", len(observations))
    return observations


def _contains_shell_command(line: str, command: str) -> bool:
    escaped = re.escape(command)
    return re.search(rf"(^|[;&|]\s*|\s){escaped}(\s|$)", line) is not None


def _best_matching_command(line: str, commands: list[str]) -> str | None:
    matches = [command for command in commands if _contains_shell_command(line, command)]
    if not matches:
        return None
    return max(matches, key=len)
