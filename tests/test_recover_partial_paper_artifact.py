from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from skillrecon.core.config import AnalyzerConfig, LLMConfig
from skillrecon.evaluation.artifacts import REQUIRED_ARTIFACTS, STATUS_ARTIFACT


SCRIPT_PATH = Path("scripts/recover_partial_paper_artifact.py")


def _load_recover_module():
    spec = importlib.util.spec_from_file_location("recover_partial_paper_artifact", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recover_partial_artifact_runs_post_behavior_stages(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    recover = _load_recover_module()
    skill_id = "owner/skill"
    output_dir = tmp_path / "artifacts"
    artifact_dir = output_dir / skill_id
    artifact_dir.mkdir(parents=True)
    _write_partial_artifacts(artifact_dir, skill_id=skill_id)

    class FakeContractPipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.kwargs = kwargs

        def recall_for_behavior(self, skill_id_arg, contract_table, events, resources):  # noqa: ANN001
            assert skill_id_arg == skill_id
            assert events == []
            assert resources == []
            return contract_table

    class FakeReconciliationPipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.kwargs = kwargs

        def run(self, **kwargs):  # noqa: ANN001
            assert kwargs["skill_id"] == skill_id
            _write_json(artifact_dir / "judgment_table.json", [])
            _write_json(artifact_dir / "certificate_table.json", [])
            _write_json(artifact_dir / "reconciliation_edges.json", [])
            _write_json(artifact_dir / "g_x.json", {"nodes": [], "edges": []})
            return []

    class FakeWitnessPipeline:
        def __init__(self, **kwargs) -> None:  # noqa: ANN001
            self.kwargs = kwargs

        def run(self, **kwargs):  # noqa: ANN001
            assert kwargs["skill_id"] == skill_id
            for name in (
                "findings.json",
                "diagnostics.json",
                "exposures.json",
                "witnesses.json",
                "rejected_witnesses.json",
                "witness_validation.json",
                "permission_manifest.json",
            ):
                _write_json(artifact_dir / name, [])
            return [], []

    monkeypatch.setattr(recover, "ContractObservationPipeline", FakeContractPipeline)
    monkeypatch.setattr(recover, "ReconciliationPipeline", FakeReconciliationPipeline)
    monkeypatch.setattr(recover, "WitnessPipeline", FakeWitnessPipeline)
    analyzer_config = AnalyzerConfig(
        llm=LLMConfig(
            base_url="https://example.test/v1",
            model="unit-test-model",
            api_key_env="literal-key",
        )
    )

    result = recover.recover_partial_artifact(
        skill_id=skill_id,
        data_root=tmp_path / "sources",
        output_dir=output_dir,
        analyzer_config=analyzer_config,
        status_metadata={"recovered_from": "unit-test"},
    )

    assert result["status"] == "ok"
    assert result["missing_required_artifacts"] == []
    status_payload = json.loads(
        (artifact_dir / STATUS_ARTIFACT).read_text(encoding="utf-8")
    )
    assert status_payload["artifact_kind"] == "full"
    assert status_payload["metadata"]["recovered_from"] == "unit-test"


def test_batch_recovery_dry_run_reports_only_post_behavior_ready_skills(
    tmp_path: Path,
) -> None:
    recover = _load_recover_module()
    output_dir = tmp_path / "artifacts"
    _write_partial_artifacts(output_dir / "owner" / "ready", skill_id="owner/ready")
    (output_dir / "owner" / "early").mkdir(parents=True)
    _write_json(
        output_dir / "owner" / "early" / "document_pack.json",
        {"skill_id": "owner/early", "admitted_docs": [], "doc_blocks": []},
    )

    result = recover.recover_partial_artifacts(
        skill_ids=["owner/ready", "owner/early"],
        data_root=tmp_path / "sources",
        output_dir=output_dir,
        analyzer_config=None,
        prompt_version="v1",
        taxonomy_version="v2",
        dry_run=True,
    )

    assert result["status"] == "dry_run"
    assert result["status_counts"] == {
        "dry_run_recoverable": 1,
        "skipped_missing_inputs": 1,
    }
    rows = {row["skill_id"]: row for row in result["rows"]}
    assert rows["owner/ready"]["status"] == "dry_run_recoverable"
    assert rows["owner/early"]["status"] == "skipped_missing_inputs"
    assert "contract_table.json" in rows["owner/early"]["missing_inputs"]


def _write_partial_artifacts(artifact_dir: Path, *, skill_id: str) -> None:
    for relative in REQUIRED_ARTIFACTS:
        if relative == STATUS_ARTIFACT:
            continue
        _write_json(artifact_dir / relative, [])
    _write_json(
        artifact_dir / "package_manifest.json",
        {
            "skill_id": skill_id,
            "root_doc": "SKILL.md",
            "files": [],
            "documents": [],
            "code_units": [],
            "links": [],
        },
    )
    _write_json(
        artifact_dir / "document_pack.json",
        {
            "skill_id": skill_id,
            "admitted_docs": [],
            "doc_blocks": [],
        },
    )
    _write_json(
        artifact_dir / "contract_table.json",
        {
            "skill_id": skill_id,
            "clauses": [],
            "steps": [],
            "step_order_edges": [],
        },
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
