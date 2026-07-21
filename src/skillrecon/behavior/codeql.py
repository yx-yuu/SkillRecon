"""Helpers for staging source code and running CodeQL."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from skillrecon.contract.parser import parse_document
from skillrecon.core.types import PackageManifest

logger = logging.getLogger(__name__)

_RECOVERABLE_JS_TS_FAILURE_MARKERS = (
    "No JavaScript or TypeScript code found",
    "no-source-code-seen-during-build",
)

_LANGUAGE_GROUPS: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "javascript",
}

_SYNTHETIC_EXTENSIONS: dict[str, str] = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "bash": ".sh",
    "html": ".html",
    "css": ".css",
    "sql": ".sql",
    "ruby": ".rb",
    "go": ".go",
    "rust": ".rs",
    "java": ".java",
    "php": ".php",
}


def normalize_codeql_language(language: str) -> str | None:
    """Map a unit language to the CodeQL extractor language."""
    return _LANGUAGE_GROUPS.get(language)


def resolve_codeql_bin(codeql_bin: str | None = None) -> str:
    """Resolve the CodeQL CLI binary path."""
    candidates = [
        codeql_bin,
        os.environ.get("CODEQL_BIN"),
        "codeql",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate == "codeql":
            return candidate
        if Path(candidate).is_file():
            return candidate
    return "codeql"


def compute_source_fingerprint(skill_path: Path) -> str:
    """Compute a stable content fingerprint for a skill package."""
    digest = hashlib.sha256()
    for file_path in sorted(p for p in skill_path.rglob("*") if p.is_file()):
        rel = file_path.relative_to(skill_path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stage_skill_sources(
    skill_path: Path,
    manifest: PackageManifest,
    staged_root: Path,
) -> dict[str, str]:
    """Copy skill sources into a staging directory and materialize synthetic units."""
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)

    file_map = {entry.file_id: entry for entry in manifest.files}
    for entry in manifest.files:
        source = skill_path / entry.relative_path
        target = staged_root / entry.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    unit_paths: dict[str, str] = {}
    block_cache: dict[str, dict[str, str]] = {}

    for unit in manifest.code_units:
        if unit.source_doc_id is None:
            unit_paths[unit.unit_id] = PurePosixPath(
                file_map[unit.file_id].relative_path
            ).as_posix()
            continue

        doc_entry = file_map[unit.file_id]
        block_map = block_cache.get(unit.source_doc_id)
        if block_map is None:
            doc_content = (skill_path / doc_entry.relative_path).read_text(encoding="utf-8")
            block_map = {
                block.block_id: block.content
                for block in parse_document(unit.source_doc_id, doc_content)
                if block.block_type == "code_block"
            }
            block_cache[unit.source_doc_id] = block_map

        if unit.source_block_id is None or unit.source_block_id not in block_map:
            raise ValueError(
                f"Cannot materialize synthetic unit {unit.unit_id}: "
                f"missing block {unit.source_block_id}"
            )

        extension = _SYNTHETIC_EXTENSIONS.get(unit.language, ".txt")
        relative_path = PurePosixPath(
            ".skillrecon_synthetic", f"{unit.unit_id}{extension}"
        ).as_posix()
        destination = staged_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(block_map[unit.source_block_id], encoding="utf-8")
        unit_paths[unit.unit_id] = relative_path

    logger.info(
        "Staged %d code units for %s at %s",
        len(manifest.code_units),
        manifest.skill_id,
        staged_root,
    )
    return unit_paths


def ensure_codeql_database(
    codeql_bin: str,
    language: str,
    source_root: Path,
    db_path: Path,
    source_fingerprint: str,
) -> None:
    """Create or reuse a language-specific CodeQL database."""
    meta_path = db_path / ".skillrecon_meta.json"
    if db_path.exists() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_fingerprint") == source_fingerprint:
            logger.info("Reusing CodeQL database: %s", db_path)
            return

    if db_path.exists():
        shutil.rmtree(db_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        codeql_bin,
        "database",
        "create",
        str(db_path),
        f"--language={language}",
        f"--source-root={source_root}",
        "--overwrite",
    ]
    if language == "python":
        cmd.append("--command=true")
    _run_command(cmd, cwd=source_root.parent)
    meta_path.write_text(
        json.dumps(
            {
                "language": language,
                "source_root": str(source_root),
                "source_fingerprint": source_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run_codeql_analysis(
    codeql_bin: str,
    db_path: Path,
    query_dir: Path,
    output_path: Path,
    source_fingerprint: str,
) -> None:
    """Run a CodeQL query pack directory and emit SARIF output."""
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_fingerprint") == source_fingerprint:
            logger.info("Reusing SARIF output: %s", output_path)
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        codeql_bin,
        "database",
        "analyze",
        str(db_path),
        str(query_dir),
        "--format=sarif-latest",
        f"--output={output_path}",
    ]
    _run_command(cmd, cwd=query_dir)
    meta_path.write_text(
        json.dumps(
            {
                "database": str(db_path),
                "query_dir": str(query_dir),
                "source_fingerprint": source_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class CodeQLCommandError(RuntimeError):
    """Structured CodeQL command failure with stdout/stderr preserved."""

    def __init__(
        self,
        *,
        cmd: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        self.cmd = list(cmd)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(
            "Command failed with exit code "
            f"{returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )


def is_recoverable_codeql_failure(language: str, error: Exception) -> bool:
    """Whether a CodeQL failure should degrade to empty behavior output."""
    if language != "javascript":
        return False
    if not isinstance(error, CodeQLCommandError):
        return False
    combined = f"{error.stdout}\n{error.stderr}"
    return any(marker in combined for marker in _RECOVERABLE_JS_TS_FAILURE_MARKERS)


def write_empty_sarif(output_path: Path, *, reason: str) -> None:
    """Emit an empty SARIF file for degraded analysis runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {
                            "driver": {
                                "name": "skillrecon-codeql-degraded",
                            }
                        },
                        "invocations": [
                            {
                                "executionSuccessful": False,
                                "stderr": reason,
                            }
                        ],
                        "results": [],
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_command(cmd: list[str], cwd: Path) -> None:
    """Run a subprocess command and surface stderr on failure."""
    logger.info("Running command: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    raise CodeQLCommandError(
        cmd=cmd,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
