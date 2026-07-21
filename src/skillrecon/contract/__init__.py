"""Contract Observation: deterministic extraction, ICCM, voting, and pipeline."""

from __future__ import annotations

from skillrecon.contract.deterministic import extract_from_frontmatter
from skillrecon.contract.parser import parse_document
from skillrecon.contract.steps import build_steps
from skillrecon.contract.voting import (
    aggregate_samples,
    apply_authorization_guard,
    build_contract_table,
)

__all__ = [
    "ContractObservationPipeline",
    "ICCMExtractor",
    "aggregate_samples",
    "apply_authorization_guard",
    "build_steps",
    "build_contract_table",
    "extract_from_frontmatter",
    "parse_document",
]


def __getattr__(name: str) -> object:
    if name == "ICCMExtractor":
        from skillrecon.contract.iccm import ICCMExtractor

        return ICCMExtractor
    if name == "ContractObservationPipeline":
        from skillrecon.contract.pipeline import ContractObservationPipeline

        return ContractObservationPipeline
    raise AttributeError(name)
