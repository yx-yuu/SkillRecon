"""Artifact Normalization: file scanning, reference closure, manifest."""

from skillrecon.loader.inline import extract_synthetic_code_units
from skillrecon.loader.manifest import build_manifest
from skillrecon.loader.reference import build_document_closure, extract_references
from skillrecon.loader.scanner import infer_language, scan_skill_directory

__all__ = [
    "build_document_closure",
    "build_manifest",
    "extract_references",
    "extract_synthetic_code_units",
    "infer_language",
    "scan_skill_directory",
]
