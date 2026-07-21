"""Witness engine pipeline for Module 04."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from skillrecon.core.config import AnalyzerConfig
from skillrecon.core.types import (
    CapabilityEvent,
    Certificate,
    Clause,
    Finding,
    ResourceUse,
    ReconciliationEdge,
    ReconciliationJudgment,
    RiskPath,
    Witness,
)
from skillrecon.detect.findings import (
    materialize_diagnostics,
    materialize_exposures,
    materialize_findings,
)
from skillrecon.detect.witness import assemble_witnesses, build_witness_contexts

logger = logging.getLogger(__name__)


class WitnessPipeline:
    """Orchestrates finding materialization and witness construction."""

    def __init__(self, analyzer_config: AnalyzerConfig, output_dir: Path) -> None:
        self._config = analyzer_config
        self._output_dir = output_dir

    def run(
        self,
        skill_id: str,
        clauses: list[Clause] | None,
        judgments: list[ReconciliationJudgment],
        certificates: list[Certificate],
        projection_edges: list[ReconciliationEdge],
        events: list[CapabilityEvent],
        resources: list[ResourceUse],
        paths: list[RiskPath],
    ) -> tuple[list[Finding], list[Witness]]:
        logger.info("Starting witness pipeline for skill %s", skill_id)
        skill_dir = self._output_dir / skill_id
        skill_dir.mkdir(parents=True, exist_ok=True)

        findings = materialize_findings(
            judgments=judgments,
            certificates=certificates,
            events=events,
            paths=paths,
            taxonomy_path=self._config.taxonomy_path,
            projection_edges=projection_edges,
        )
        diagnostics = materialize_diagnostics(
            judgments=judgments,
            certificates=certificates,
            events=events,
            paths=paths,
            taxonomy_path=self._config.taxonomy_path,
            clauses=clauses,
            projection_edges=projection_edges,
        )
        exposures = materialize_exposures(
            judgments=judgments,
            certificates=certificates,
            events=events,
            resources=resources,
            paths=paths,
            taxonomy_path=self._config.taxonomy_path,
            projection_edges=projection_edges,
        )
        bundle = assemble_witnesses(
            findings=findings,
            judgments=judgments,
            certificates=certificates,
            projection_edges=projection_edges,
        )
        valid_witnesses = [
            witness
            for witness, validation in zip(bundle.witnesses, bundle.validations, strict=True)
            if validation.passed
        ]
        invalid_witnesses = [
            witness
            for witness, validation in zip(bundle.witnesses, bundle.validations, strict=True)
            if not validation.passed
        ]
        witness_contexts = build_witness_contexts(
            witnesses=valid_witnesses,
            diagnostics=diagnostics,
        )

        self._save_artifact(skill_dir / "findings.json", [f.model_dump() for f in findings])
        self._save_artifact(
            skill_dir / "diagnostics.json",
            [diagnostic.model_dump() for diagnostic in diagnostics],
        )
        self._save_artifact(
            skill_dir / "exposures.json",
            [exposure.model_dump() for exposure in exposures],
        )
        self._save_artifact(
            skill_dir / "witnesses.json",
            [witness.model_dump() for witness in valid_witnesses],
        )
        self._save_artifact(
            skill_dir / "witness_contexts.json",
            [context.model_dump() for context in witness_contexts],
        )
        self._save_artifact(
            skill_dir / "rejected_witnesses.json",
            [witness.model_dump() for witness in invalid_witnesses],
        )
        self._save_artifact(
            skill_dir / "witness_validation.json",
            [validation.model_dump() for validation in bundle.validations],
        )
        self._save_artifact(
            skill_dir / "permission_manifest.json",
            [entry.model_dump() for entry in bundle.permission_manifest],
        )

        logger.info(
            "Witness pipeline complete: %d findings, %d exposures, %d diagnostics -> %d valid witnesses, %d witness contexts (%d rejected)",
            len(findings),
            len(exposures),
            len(diagnostics),
            len(valid_witnesses),
            len(witness_contexts),
            len(invalid_witnesses),
        )
        return findings, valid_witnesses

    @staticmethod
    def _save_artifact(path: Path, data: object) -> None:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
