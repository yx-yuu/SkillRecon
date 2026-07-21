"""Build a PackageManifest for a skill package."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath

from skillrecon.contract.parser import parse_document
from skillrecon.core.enums import FileKind, ReferenceResolutionStatus
from skillrecon.core.types import CodeUnit, PackageLink, PackageManifest
from skillrecon.loader.inline import extract_synthetic_code_units
from skillrecon.loader.reference import build_document_closure
from skillrecon.loader.scanner import infer_language, scan_skill_directory

logger = logging.getLogger(__name__)

# File kinds that contain executable code.
_CODE_KINDS: set[FileKind] = {
    FileKind.PYTHON,
    FileKind.JAVASCRIPT,
    FileKind.TYPESCRIPT,
    FileKind.BASH,
}

# Extensions typical of runtime output artifacts, not source documents.
_ARTIFACT_EXTENSIONS: set[str] = {
    ".html", ".css", ".csv", ".xml", ".svg", ".png", ".jpg",
    ".jpeg", ".gif", ".pdf", ".xlsx", ".docx", ".zip",
}


def _is_artifact_target(path: str) -> bool:
    """Return whether an unresolved path looks like a declared output artifact."""
    p = PurePosixPath(path)
    return "/" not in path and p.suffix.lower() in _ARTIFACT_EXTENSIONS


def build_manifest(
    skill_path: Path, skill_id: str | None = None
) -> tuple[PackageManifest, list[str], list[str]]:
    """Build the manifest, unresolved document refs, and artifact targets."""
    if skill_id is None:
        skill_id = skill_path.name

    files = scan_skill_directory(skill_path)
    documents, document_references = build_document_closure(skill_path, files)
    root_doc = documents[0].doc_id if documents else ""
    raw_unresolved = [
        ref.target_path
        for ref in document_references
        if ref.resolution_status == ReferenceResolutionStatus.UNRESOLVED
    ]

    artifact_targets: list[str] = []
    doc_refs: list[str] = []
    for path in raw_unresolved:
        if _is_artifact_target(path):
            artifact_targets.append(path)
        else:
            doc_refs.append(path)

    file_map = {f.file_id: f for f in files}
    code_units: list[CodeUnit] = []
    unit_idx = 0

    for entry in files:
        if entry.kind not in _CODE_KINDS:
            continue
        lang = infer_language(entry)
        if lang is None:
            continue
        code_units.append(
            CodeUnit(
                unit_id=f"u{unit_idx}",
                file_id=entry.file_id,
                language=lang,
            )
        )
        unit_idx += 1

    for doc in documents:
        entry = file_map[doc.file_id]
        file_path = skill_path / entry.relative_path
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Cannot read for inline detection: %s", file_path)
            continue

        blocks = parse_document(doc.doc_id, content)
        synthetic = extract_synthetic_code_units(
            doc_id=doc.doc_id,
            file_id=doc.file_id,
            blocks=blocks,
            unit_id_start=unit_idx,
        )
        code_units.extend(synthetic)
        unit_idx += len(synthetic)

    standalone_unit_by_file_id = {
        unit.file_id: unit.unit_id
        for unit in code_units
        if unit.source_doc_id is None
    }
    package_links = [
        PackageLink(
            link_id=f"pl{idx}",
            source_doc_id=ref.source_doc_id,
            source_span=ref.source_span,
            target_path=ref.target_path,
            target_file_id=ref.target_file_id,
            target_unit_id=(
                standalone_unit_by_file_id.get(ref.target_file_id)
                if ref.target_file_id is not None
                else None
            ),
        )
        for idx, ref in enumerate(document_references)
        if ref.target_file_id is not None
    ]

    manifest = PackageManifest(
        skill_id=skill_id,
        root_doc=root_doc,
        files=files,
        documents=documents,
        document_references=document_references,
        code_units=code_units,
        links=package_links,
        declared_artifact_targets=artifact_targets,
    )

    standalone_count = sum(1 for u in code_units if u.source_doc_id is None)
    synthetic_count = len(code_units) - standalone_count
    logger.info(
        "Manifest: skill=%s, files=%d, docs=%d, "
        "code_units=%d (standalone=%d, synthetic=%d), "
        "doc_refs=%d, package_links=%d, "
        "artifact_targets=%d, unresolved_doc_refs=%d",
        skill_id,
        len(files),
        len(documents),
        len(code_units),
        standalone_count,
        synthetic_count,
        len(document_references),
        len(package_links),
        len(artifact_targets),
        len(doc_refs),
    )
    return manifest, doc_refs, artifact_targets
