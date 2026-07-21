"""Artifact Normalization: file scanning, reference closure, manifest."""

from skillrecon.loader.inline import extract_synthetic_code_units
from skillrecon.loader.manifest import build_manifest
from skillrecon.loader.reference import build_document_closure, extract_references
from skillrecon.loader.scanner import infer_language, scan_skill_directory
from skillrecon.loader.test_index import TestIndexEntry, load_test_index, load_test_index_dir

__all__ = [
    "TestIndexEntry",
    "build_document_closure",
    "build_manifest",
    "extract_references",
    "extract_synthetic_code_units",
    "infer_language",
    "load_test_index",
    "load_test_index_dir",
    "scan_skill_directory",
]
