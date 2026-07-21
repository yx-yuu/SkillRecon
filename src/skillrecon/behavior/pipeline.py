"""End-to-end behavior observation pipeline."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from skillrecon.behavior.bash import extract_bash_observations
from skillrecon.behavior.codeql import (
    CodeQLCommandError,
    compute_source_fingerprint,
    ensure_codeql_database,
    is_recoverable_codeql_failure,
    normalize_codeql_language,
    resolve_codeql_bin,
    run_codeql_analysis,
    stage_skill_sources,
    write_empty_sarif,
)
from skillrecon.behavior.instruction import (
    extract_instruction_observations,
    resolve_instruction_evidence,
)
from skillrecon.behavior.normalize import (
    build_gc_artifact,
    build_risk_paths,
    detect_bridges,
    derive_data_object_graph,
    derive_gc_enrichment,
    load_sarif_observations,
    load_sarif_paths,
    load_taxonomy_atoms,
    normalize_codeql_paths,
    normalize_observations,
)
from skillrecon.behavior.path_graph import attach_orchestration_hypotheses
from skillrecon.core.config import PROJECT_ROOT, AnalyzerConfig
from skillrecon.core.enums import HypothesisStatus
from skillrecon.core.types import (
    CodeDatabaseRef,
    CodePack,
    EvidenceSpan,
    OrchestrationHypothesis,
    PackageManifest,
)
from skillrecon.loader.manifest import build_manifest

logger = logging.getLogger(__name__)

_ORCHESTRATION_CONFIRM_CUE_RE = re.compile(
    r"(?i)执行|运行|调用|启动|运行脚本|执行脚本|"
    r"\brun\b|\bexecute\b|\binvoke\b|\bcall\b|\blaunch\b|\bstart\b"
)


def _load_a_req(taxonomy_path: Path) -> set[str]:
    data = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    return set(data.get("a_req", []))


class BehaviorObservationPipeline:
    """Construct behavior-side artifacts for a single skill package."""

    def __init__(
        self,
        analyzer_config: AnalyzerConfig,
        skill_data_root: Path,
        output_dir: Path,
        codeql_bin: str | None = None,
        taxonomy_path: Path | None = None,
        bash_pattern_path: Path | None = None,
        query_root: Path | None = None,
        codeql_db_root: Path | None = None,
    ) -> None:
        self._config = analyzer_config
        self._data_root = skill_data_root
        self._output_dir = output_dir
        self._codeql_bin = resolve_codeql_bin(codeql_bin)
        self._taxonomy_path = taxonomy_path or analyzer_config.taxonomy_path
        self._a_req = _load_a_req(self._taxonomy_path)
        self._bash_pattern_path = (
            bash_pattern_path or PROJECT_ROOT / "experiments" / "configs" / "bash_patterns.json"
        )
        self._query_root = query_root or PROJECT_ROOT / "experiments" / "codeql" / "queries"
        self._codeql_db_root = codeql_db_root or Path("derived/codeql_db")

    def run(self, skill_id: str) -> CodePack:
        """Run behavior observation and save intermediate artifacts."""
        skill_path = (self._data_root / skill_id).resolve()
        artifact_dir = (self._output_dir / skill_id).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sarif_dir = (artifact_dir / "sarif").resolve()
        staged_root = (artifact_dir / "staged_source").resolve()

        manifest, _, _ = build_manifest(skill_path, skill_id)
        source_fingerprint = compute_source_fingerprint(skill_path)
        taxonomy_atoms = load_taxonomy_atoms(self._taxonomy_path)
        unit_paths = stage_skill_sources(skill_path, manifest, staged_root)

        databases: list[CodeDatabaseRef] = []
        observations = []
        raw_paths = []

        codeql_languages = sorted(
            {
                grouped
                for unit in manifest.code_units
                if (grouped := normalize_codeql_language(unit.language)) is not None
            }
        )
        for language in codeql_languages:
            db_path = (self._codeql_db_root / skill_id / language).resolve()
            query_dir = (self._query_root / language).resolve()
            sarif_path = (sarif_dir / f"{language}.sarif").resolve()
            try:
                ensure_codeql_database(
                    self._codeql_bin,
                    language,
                    staged_root,
                    db_path,
                    source_fingerprint,
                )
                run_codeql_analysis(
                    self._codeql_bin,
                    db_path,
                    query_dir,
                    sarif_path,
                    source_fingerprint,
                )
            except CodeQLCommandError as error:
                if not is_recoverable_codeql_failure(language, error):
                    raise
                logger.warning(
                    "[%s] Degrading %s CodeQL analysis to empty SARIF: %s",
                    skill_id,
                    language,
                    error.stderr.strip() or error.stdout.strip() or str(error),
                )
                write_empty_sarif(
                    sarif_path,
                    reason=error.stderr.strip() or error.stdout.strip() or str(error),
                )
            databases.append(
                CodeDatabaseRef(
                    language=language,
                    db_path=str(db_path),
                    sarif_path=str(sarif_path),
                    source_fingerprint=source_fingerprint,
                )
            )
            observations.extend(load_sarif_observations(sarif_path, language))
            raw_paths.extend(load_sarif_paths(sarif_path, language))

        observations.extend(
            extract_bash_observations(
                manifest,
                unit_paths,
                staged_root,
                self._bash_pattern_path,
            )
        )
        observations.extend(extract_instruction_observations(skill_path, manifest))

        events, resources, bridge_hints = normalize_observations(
            observations,
            manifest,
            unit_paths,
            taxonomy_atoms,
            staged_root,
        )
        bridges = detect_bridges(unit_paths, events, resources, bridge_hints, staged_root)
        codeql_paths = normalize_codeql_paths(
            raw_paths,
            unit_paths,
            events,
            a_req=self._a_req,
        )
        codeql_pairs = {
            (path.source.event_id, path.sink.event_id)
            for path in codeql_paths
            if path.source.event_id is not None and path.sink.event_id is not None
        }
        heuristic_paths = build_risk_paths(
            events,
            bridges,
            skip_event_pairs=codeql_pairs,
            a_req=self._a_req,
        )
        orchestration = self._build_orchestration_hypotheses(skill_path, manifest)
        paths = attach_orchestration_hypotheses(codeql_paths + heuristic_paths, orchestration)
        operations, locations, sources, sinks = derive_gc_enrichment(events, resources, paths)
        data_objects, data_flow_edges = derive_data_object_graph(paths, events, resources)

        code_pack = CodePack(
            skill_id=skill_id,
            staged_root=str(staged_root),
            unit_paths=unit_paths,
            databases=databases,
        )

        self._save_artifact(artifact_dir / "code_pack.json", code_pack.model_dump())
        self._save_artifact(
            artifact_dir / "doc_ref_graph.json",
            [ref.model_dump() for ref in manifest.document_references],
        )
        self._save_artifact(
            artifact_dir / "package_links.json",
            [link.model_dump() for link in manifest.links],
        )
        self._save_artifact(
            artifact_dir / "event_table.json",
            [event.model_dump() for event in events],
        )
        self._save_artifact(
            artifact_dir / "resource_table.json",
            [resource.model_dump() for resource in resources],
        )
        self._save_artifact(
            artifact_dir / "bridge_table.json",
            [bridge.model_dump() for bridge in bridges],
        )
        self._save_artifact(
            artifact_dir / "path_table.json",
            [path.model_dump() for path in paths],
        )
        self._save_artifact(
            artifact_dir / "operation_table.json",
            [operation.model_dump() for operation in operations],
        )
        self._save_artifact(
            artifact_dir / "location_table.json",
            [location.model_dump() for location in locations],
        )
        self._save_artifact(
            artifact_dir / "source_table.json",
            [source.model_dump() for source in sources],
        )
        self._save_artifact(
            artifact_dir / "sink_table.json",
            [sink.model_dump() for sink in sinks],
        )
        self._save_artifact(
            artifact_dir / "data_object_table.json",
            [data_object.model_dump() for data_object in data_objects],
        )
        self._save_artifact(
            artifact_dir / "data_flow_table.json",
            [edge.model_dump() for edge in data_flow_edges],
        )
        self._save_artifact(
            artifact_dir / "g_c.json",
            build_gc_artifact(manifest, events, resources, bridges, paths, staged_root=staged_root),
        )
        self._save_artifact(
            artifact_dir / "orchestration_table.json",
            [item.model_dump() for item in orchestration],
        )

        logger.info(
            "[%s] Behavior observation done: %d events, %d resources, %d bridges, "
            "%d paths, %d orchestration hypotheses",
            skill_id,
            len(events),
            len(resources),
            len(bridges),
            len(paths),
            len(orchestration),
        )
        return code_pack

    def _save_artifact(self, path: Path, data: object) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _build_orchestration_hypotheses(
        self,
        skill_path: Path,
        manifest: PackageManifest,
    ) -> list[OrchestrationHypothesis]:
        hypotheses: list[OrchestrationHypothesis] = []
        seen: set[tuple[str, int, int, tuple[str, ...]]] = set()

        for link in manifest.links:
            if link.target_unit_id is None:
                continue
            evidence = resolve_instruction_evidence(skill_path, manifest, link)
            if evidence is None:
                continue
            target_unit_ids = [link.target_unit_id]
            key = (
                evidence.doc_id,
                evidence.start_offset,
                evidence.end_offset,
                tuple(target_unit_ids),
            )
            if key in seen:
                continue
            seen.add(key)
            hypotheses.append(
                OrchestrationHypothesis(
                    hypothesis_id=f"oh{len(hypotheses)}",
                    instruction_evidence=evidence,
                    target_unit_ids=target_unit_ids,
                    status=self._initial_orchestration_status(link, evidence),
                )
            )

        return self._mark_competing_hypotheses(hypotheses)

    @staticmethod
    def _initial_orchestration_status(
        link,
        evidence: EvidenceSpan,
    ) -> HypothesisStatus:
        target_path = (getattr(link, "target_path", "") or "").lower()
        evidence_text = evidence.text.lower()
        if _ORCHESTRATION_CONFIRM_CUE_RE.search(evidence_text):
            return HypothesisStatus.CONFIRMED
        if _looks_like_direct_command_reference(evidence_text, target_path):
            return HypothesisStatus.CONFIRMED
        return HypothesisStatus.UNRESOLVED

    @staticmethod
    def _mark_competing_hypotheses(
        hypotheses: list[OrchestrationHypothesis],
    ) -> list[OrchestrationHypothesis]:
        grouped: dict[tuple[str, int, int, str], list[OrchestrationHypothesis]] = {}
        for hypothesis in hypotheses:
            evidence = hypothesis.instruction_evidence
            grouped.setdefault(
                (evidence.doc_id, evidence.start_offset, evidence.end_offset, evidence.text),
                [],
            ).append(hypothesis)

        updated: list[OrchestrationHypothesis] = []
        for group in grouped.values():
            target_sets = {tuple(hypothesis.target_unit_ids) for hypothesis in group}
            if len(target_sets) > 1:
                updated.extend(
                    hypothesis.model_copy(update={"status": HypothesisStatus.COMPETING})
                    for hypothesis in group
                )
                continue
            updated.extend(group)

        return sorted(updated, key=lambda hypothesis: hypothesis.hypothesis_id)


def _looks_like_direct_command_reference(evidence_text: str, target_path: str) -> bool:
    """Return whether a doc span directly presents the target as a command."""
    if not target_path:
        return False
    basename = Path(target_path).name.lower()
    if not basename or basename not in evidence_text:
        return False
    executable_suffixes = (".py", ".js", ".ts", ".sh", ".bash")
    if not basename.endswith(executable_suffixes):
        return False
    command_cues = (
        "python",
        "python3",
        "node",
        "npx",
        "bash",
        "sh",
        "uv run",
        "npm run",
        "pnpm run",
    )
    return any(cue in evidence_text for cue in command_cues)
