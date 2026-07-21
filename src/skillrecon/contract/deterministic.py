"""Deterministic clause extraction from structured document fields."""

from __future__ import annotations

import logging
import re

from skillrecon.contract.normalize import normalize_clause_surface
from skillrecon.core.enums import ClauseOperator, ClauseRole
from skillrecon.core.types import Clause, EvidenceSpan

logger = logging.getLogger(__name__)

_TOOL_PATTERN = re.compile(r"(\w+)\(([^)]*)\)")

_EXCLUSION_PATTERNS = [
    re.compile(r"[Nn]ot\s+for\s+(.+?)(?:\.|$)", re.MULTILINE),
    re.compile(r"不(?:用于|适用于|支持)\s*(.+?)(?:。|$)", re.MULTILINE),
]

_TOOL_CAPABILITY_MAP: dict[str, str] = {
    "Bash": "shell_exec",
    "Read": "file_read",
    "Edit": "file_write",
    "Write": "file_write",
    "WebSearch": "http_request",
    "WebFetch": "http_request",
}


def extract_from_frontmatter(
    frontmatter: dict[str, str],
    doc_id: str,
    frontmatter_text: str,
) -> list[Clause]:
    """Extract deterministic clauses from parsed frontmatter."""
    clauses: list[Clause] = []
    clause_idx = 0

    allowed_tools = frontmatter.get("allowed-tools", "")
    if isinstance(allowed_tools, str) and allowed_tools.strip():
        tool_clauses = _extract_allowed_tools(
            allowed_tools, doc_id, frontmatter_text, clause_idx
        )
        clauses.extend(tool_clauses)
        clause_idx += len(tool_clauses)

    description = frontmatter.get("description", "")
    if isinstance(description, str) and description.strip():
        exclusion_clauses = _extract_exclusions(
            description, doc_id, frontmatter_text, clause_idx
        )
        clauses.extend(exclusion_clauses)

    logger.info(
        "Deterministic extraction from %s: %d clauses", doc_id, len(clauses)
    )
    return clauses


def _extract_allowed_tools(
    tools_str: str,
    doc_id: str,
    full_text: str,
    start_idx: int,
) -> list[Clause]:
    """Parse ``allowed-tools`` into deterministic allowed clauses."""
    clauses: list[Clause] = []
    evidence_offset = full_text.find(tools_str)

    for i, match in enumerate(_TOOL_PATTERN.finditer(tools_str)):
        tool_name = match.group(1)
        pattern = match.group(2).strip()

        capability = _TOOL_CAPABILITY_MAP.get(tool_name, "shell_exec")
        evidence = EvidenceSpan(
            doc_id=doc_id,
            start_offset=max(0, evidence_offset),
            end_offset=max(0, evidence_offset) + len(tools_str),
            text=f"allowed-tools: {tools_str}",
        )
        target, constraints = normalize_clause_surface(
            capability,
            None,
            [pattern] if pattern else [],
            clause_id=f"det_{start_idx + i}",
            evidence=evidence,
        )

        clause = Clause(
            clause_id=f"det_{start_idx + i}",
            capability=capability,
            operator=ClauseOperator.ALLOWED,
            role=ClauseRole.POLICY,
            target=target,
            constraints=constraints,
            evidence_spans=[evidence],
            vote_agreement=1.0,
            step_ids=[f"{doc_id}_frontmatter"],
            source_doc_ids=[doc_id],
        )
        clauses.append(clause)
        logger.debug("  allowed-tools: %s(%s) -> %s", tool_name, pattern, capability)

    return clauses


def _extract_exclusions(
    description: str,
    doc_id: str,
    full_text: str,
    start_idx: int,
) -> list[Clause]:
    """Extract explicit scope exclusions from the description field."""
    clauses: list[Clause] = []
    desc_offset = full_text.find(description)

    for pattern in _EXCLUSION_PATTERNS:
        for j, match in enumerate(pattern.finditer(description)):
            excluded_scope = match.group(1).strip()
            if not excluded_scope:
                continue

            evidence = EvidenceSpan(
                doc_id=doc_id,
                start_offset=max(0, desc_offset + match.start()),
                end_offset=max(0, desc_offset + match.end()),
                text=match.group(0).strip(),
            )
            target, constraints = normalize_clause_surface(
                "file_read",
                excluded_scope,
                [],
                clause_id=f"det_excl_{start_idx + j}",
                evidence=evidence,
            )

            clause = Clause(
                clause_id=f"det_excl_{start_idx + j}",
                capability="file_read",
                operator=ClauseOperator.PROHIBITED,
                role=ClauseRole.POLICY,
                target=target,
                constraints=constraints,
                evidence_spans=[evidence],
                vote_agreement=1.0,
                step_ids=[f"{doc_id}_frontmatter"],
                source_doc_ids=[doc_id],
            )
            clauses.append(clause)
            logger.debug("  exclusion: Not for %s", excluded_scope)

    return clauses
