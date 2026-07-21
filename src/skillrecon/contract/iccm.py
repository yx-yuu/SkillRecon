"""ICCM (Intent-Conditioned Contract Mining) extraction via LLM."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from skillrecon.core.types import ClauseSample, ClauseSampleList, DocBlock, Step
from skillrecon.llm.cache import CachedLLMClient

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path("experiments/prompts/iccm")
_MAX_BLOCKS_PER_CALL = 24
_MAX_RECALL_BLOCKS_PER_CALL = 16
_URL_RE = re.compile(r"https?://[^\s)\]}>\"']+")
_INLINE_CODE_RE = re.compile(r"`[^`\n]{1,240}`")
_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_ABSOLUTE_PATH_RE = re.compile(r"(?:~|\$HOME|/)[A-Za-z0-9._~/$-]+")
_COMMAND_LINE_RE = re.compile(
    r"(?im)^\s*(?:bash|sh|python3?|node|npm|npx|curl|wget|git|sudo|yt-dlp)\b.*$"
)
_SENSITIVE_TERM_REPLACEMENTS = (
    (re.compile(r"(?i)\byoutube\b|\byoutu\.be\b"), "video-platform"),
    (re.compile(r"(?i)\byt[-_\s]?downloader\b"), "media-retrieval-tool"),
    (re.compile(r"(?i)\byt-dlp\b"), "media-retrieval-cli"),
    (re.compile(r"(?i)\bvideos?\b|\bmp4\b"), "media-file"),
    (re.compile(r"(?i)\btelegram\b"), "messaging-platform"),
    (re.compile(r"(?i)\bphone\s*(?:number)?\b"), "contact-identifier"),
    (re.compile(r"(?i)\bdownload(?:ing|ed|s)?\b"), "retrieve"),
    (re.compile(r"(?i)\bprivate/deleted\b|\bprivate\b|\bdeleted\b"), "restricted"),
    (re.compile(r"(?i)\bage-restricted\b"), "age-gated"),
    (re.compile(r"(?i)\bcookies?\b"), "browser-state"),
)


class ICCMExtractionError(RuntimeError):
    """Raised when ICCM cannot obtain a required LLM extraction result."""


def _load_prompt(prompt_version: str, name: str) -> str:
    """Load a prompt template file."""
    path = _PROMPT_DIR / prompt_version / name
    return path.read_text(encoding="utf-8")


def _load_optional_prompt(prompt_version: str, name: str) -> str | None:
    """Load an optional prompt template file."""
    path = _PROMPT_DIR / prompt_version / name
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _format_doc_blocks(doc_blocks: list[DocBlock] | list[Step]) -> str:
    """Format doc steps/blocks for inclusion in the LLM prompt."""
    parts: list[str] = []
    for block in doc_blocks:
        if isinstance(block, Step):
            header = f"[{block.step_id}] type=step step_type={block.step_type}"
            if block.heading_context:
                header += f" section=\"{block.heading_context}\""
            header += f" doc={block.doc_id}"
            parts.append(f"{header}\n{block.text}")
            continue

        header = f"[{block.block_id}] type={block.block_type}"
        if block.heading_context:
            header += f" section=\"{block.heading_context}\""
        header += f" doc={block.doc_id}"
        parts.append(f"{header}\n{block.content}")
    return "\n\n---\n\n".join(parts)


def _load_taxonomy_atoms(taxonomy_path: str = "experiments/configs/taxonomy_v2.json") -> list[str]:
    """Load capability atom names from taxonomy JSON."""
    with open(taxonomy_path, encoding="utf-8") as f:
        data = json.load(f)
    atoms: list[str] = []
    for cat in data.get("categories", {}).values():
        atoms.extend(cat.get("atoms", []))
    return sorted(atoms)


class ICCMExtractor:
    """Extract contract clauses from document blocks with an LLM."""

    def __init__(
        self,
        client: CachedLLMClient,
        taxonomy_atoms: list[str] | None = None,
        prompt_version: str = "v1",
    ) -> None:
        self._client = client
        self._atoms = taxonomy_atoms or _load_taxonomy_atoms()
        self._prompt_version = prompt_version
        self._system_prompt = _load_prompt(prompt_version, "system.txt")
        self._extraction_template = _load_prompt(prompt_version, "extraction.txt")
        self._recall_system_prompt = (
            _load_optional_prompt(prompt_version, "recall_system.txt") or self._system_prompt
        )
        self._recall_extraction_template = (
            _load_optional_prompt(prompt_version, "recall_extraction.txt")
            or self._extraction_template
        )

    def extract_single(
        self,
        skill_id: str,
        doc_blocks: list[DocBlock] | list[Step],
        sample_index: int,
        call_key_prefix: str = "iccm",
    ) -> list[ClauseSample]:
        """Run one ICCM extraction pass."""
        chunks = _chunk_blocks(doc_blocks, _MAX_BLOCKS_PER_CALL)
        all_clauses: list[ClauseSample] = []
        for chunk_index, chunk in enumerate(chunks):
            user_prompt = self._extraction_template.format(
                skill_id=skill_id,
                taxonomy_atoms=", ".join(self._atoms),
                doc_blocks=_format_doc_blocks(chunk),
            )

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            call_key = f"{call_key_prefix}_sample_{sample_index}_chunk_{chunk_index}"

            try:
                result = self._client.structured_complete(
                    messages,
                    ClauseSampleList,
                    skill_id=skill_id,
                    call_key=call_key,
                )
            except Exception as exc:
                if not _is_content_filter_error(exc):
                    raise ICCMExtractionError(
                        "ICCM chunk failed for "
                        f"{skill_id} sample={sample_index} chunk={chunk_index}: {exc}"
                    ) from exc
                result = self._extract_filter_safe_chunk(
                    skill_id=skill_id,
                    chunk=chunk,
                    sample_index=sample_index,
                    chunk_index=chunk_index,
                    call_key=call_key,
                    original_error=exc,
                    focus_context=None,
                )
            all_clauses.extend(result.clauses)

        logger.info(
            "ICCM sample %d for %s: %d clauses extracted",
            sample_index,
            skill_id,
            len(all_clauses),
        )
        return list(all_clauses)

    def extract_recall_single(
        self,
        skill_id: str,
        doc_blocks: list[DocBlock] | list[Step],
        focus_context: str,
        sample_index: int,
        call_key_prefix: str = "iccm_recall",
    ) -> list[ClauseSample]:
        """Run a focused ICCM extraction pass for observed sensitive behavior."""
        chunks = _chunk_blocks(doc_blocks, _MAX_RECALL_BLOCKS_PER_CALL)
        all_clauses: list[ClauseSample] = []
        for chunk_index, chunk in enumerate(chunks):
            user_prompt = self._recall_extraction_template.format(
                skill_id=skill_id,
                focus_context=focus_context,
                taxonomy_atoms=", ".join(self._atoms),
                doc_blocks=_format_doc_blocks(chunk),
            )
            messages = [
                {"role": "system", "content": self._recall_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            call_key = f"{call_key_prefix}_sample_{sample_index}_chunk_{chunk_index}"
            try:
                result = self._client.structured_complete(
                    messages,
                    ClauseSampleList,
                    skill_id=skill_id,
                    call_key=call_key,
                )
            except Exception as exc:
                if not _is_content_filter_error(exc):
                    raise ICCMExtractionError(
                        "ICCM recall chunk failed for "
                        f"{skill_id} sample={sample_index} chunk={chunk_index}: {exc}"
                    ) from exc
                result = self._extract_filter_safe_chunk(
                    skill_id=skill_id,
                    chunk=chunk,
                    sample_index=sample_index,
                    chunk_index=chunk_index,
                    call_key=call_key,
                    original_error=exc,
                    focus_context=focus_context,
                )
            all_clauses.extend(result.clauses)
        logger.info(
            "ICCM recall sample %d for %s: %d clauses extracted",
            sample_index,
            skill_id,
            len(all_clauses),
        )
        return list(all_clauses)

    def extract_all(
        self,
        skill_id: str,
        doc_blocks: list[DocBlock] | list[Step],
        n_samples: int = 5,
        call_key_prefix: str = "iccm",
    ) -> list[list[ClauseSample]]:
        """Run repeated ICCM passes for self-consistency voting."""
        all_samples: list[list[ClauseSample]] = []
        for i in range(n_samples):
            samples = self.extract_single(
                skill_id,
                doc_blocks,
                i,
                call_key_prefix=call_key_prefix,
            )
            all_samples.append(samples)
        return all_samples

    def _extract_filter_safe_chunk(
        self,
        *,
        skill_id: str,
        chunk: list[DocBlock] | list[Step],
        sample_index: int,
        chunk_index: int,
        call_key: str,
        original_error: Exception,
        focus_context: str | None,
    ) -> ClauseSampleList:
        """Retry ICCM with an abstracted evidence view after provider filtering."""
        safe_doc_blocks, evidence_map = _format_filter_safe_doc_blocks(chunk)
        safe_skill_id = _abstract_filter_sensitive_text(skill_id)
        if focus_context is None:
            user_prompt = self._extraction_template.format(
                skill_id=safe_skill_id,
                taxonomy_atoms=", ".join(self._atoms),
                doc_blocks=safe_doc_blocks,
            )
            system_prompt = self._system_prompt
            error_prefix = "ICCM content-filter-safe chunk failed"
        else:
            user_prompt = self._recall_extraction_template.format(
                skill_id=safe_skill_id,
                focus_context=_abstract_filter_sensitive_text(focus_context),
                taxonomy_atoms=", ".join(self._atoms),
                doc_blocks=safe_doc_blocks,
            )
            system_prompt = self._recall_system_prompt
            error_prefix = "ICCM recall content-filter-safe chunk failed"

        user_prompt = (
            "Provider content filtering blocked the raw documentation view. "
            "The following documentation view is safety-preserving but evidence-grounded. "
            "Each source unit has an [EVIDENCE_REF:<id>] marker. When a clause is supported "
            "by a source unit, set evidence_span exactly to that marker instead of inventing "
            "or expanding source text; the local pipeline will restore the original quote. "
            "Continue extracting authorization/prohibition/scope clauses from the abstracted "
            "view without giving operational instructions.\n\n"
            f"{user_prompt}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        logger.warning(
            "ICCM content filter for %s sample=%d chunk=%d; retrying with evidence-safe view",
            skill_id,
            sample_index,
            chunk_index,
        )
        try:
            result = self._client.structured_complete(
                messages,
                ClauseSampleList,
                skill_id=skill_id,
                call_key=call_key,
            )
        except Exception as exc:
            raise ICCMExtractionError(
                f"{error_prefix} for "
                f"{skill_id} sample={sample_index} chunk={chunk_index}: "
                f"raw_error={original_error}; retry_error={exc}"
            ) from exc
        return _restore_filter_safe_evidence(result, evidence_map)

    def extract_recall_all(
        self,
        skill_id: str,
        doc_blocks: list[DocBlock] | list[Step],
        focus_context: str,
        n_samples: int = 5,
        call_key_prefix: str = "iccm_recall",
    ) -> list[list[ClauseSample]]:
        """Run focused recall extraction passes for observed sensitive behavior."""
        all_samples: list[list[ClauseSample]] = []
        for i in range(n_samples):
            samples = self.extract_recall_single(
                skill_id,
                doc_blocks,
                focus_context,
                i,
                call_key_prefix=call_key_prefix,
            )
            all_samples.append(samples)
        return all_samples


def _chunk_blocks(
    doc_blocks: list[DocBlock] | list[Step],
    chunk_size: int,
) -> list[list[DocBlock] | list[Step]]:
    if len(doc_blocks) <= chunk_size:
        return [doc_blocks]
    return [
        doc_blocks[index : index + chunk_size]
        for index in range(0, len(doc_blocks), chunk_size)
    ]


def _format_filter_safe_doc_blocks(
    doc_blocks: list[DocBlock] | list[Step],
) -> tuple[str, dict[str, str]]:
    """Format doc blocks as a provider-safe view while retaining restore markers."""
    parts: list[str] = []
    restore_map: dict[str, str] = {}

    for index, block in enumerate(doc_blocks):
        evidence_id = f"E{index:03d}"
        evidence_marker = f"[EVIDENCE_REF:{evidence_id}]"
        if isinstance(block, Step):
            original_text = block.text
            restore_map[evidence_marker] = original_text
            header = f"[{block.step_id}] type=step step_type={block.step_type}"
            if block.heading_context:
                header += (
                    f" section=\"{_abstract_filter_sensitive_text(block.heading_context)}\""
                )
            header += f" doc={block.doc_id}"
            safe_text = _abstract_filter_sensitive_text(
                original_text,
                ref_prefix=evidence_id,
                restore_map=restore_map,
            )
            parts.append(f"{header}\n{evidence_marker}\n{safe_text}")
            continue

        original_text = block.content
        restore_map[evidence_marker] = original_text
        header = f"[{block.block_id}] type={block.block_type}"
        if block.heading_context:
            header += f" section=\"{_abstract_filter_sensitive_text(block.heading_context)}\""
        header += f" doc={block.doc_id}"
        safe_text = _abstract_filter_sensitive_text(
            original_text,
            ref_prefix=evidence_id,
            restore_map=restore_map,
        )
        parts.append(f"{header}\n{evidence_marker}\n{safe_text}")

    return "\n\n---\n\n".join(parts), restore_map


def _abstract_filter_sensitive_text(
    text: str,
    *,
    ref_prefix: str = "CTX",
    restore_map: dict[str, str] | None = None,
) -> str:
    """Replace operationally sensitive literals with local restore markers."""
    restore = restore_map if restore_map is not None else {}
    counters: dict[str, int] = {}

    def marker(kind: str, value: str) -> str:
        counters[kind] = counters.get(kind, 0) + 1
        ref = f"[{kind}:{ref_prefix}_{counters[kind] - 1:02d}]"
        restore[ref] = value
        return ref

    def replace_fenced_code(match: re.Match[str]) -> str:
        return f"{marker('CODE_BLOCK_REF', match.group(0))} code_block_present"

    def replace_command(match: re.Match[str]) -> str:
        return f"{marker('COMMAND_REF', match.group(0).strip())} command_line_present"

    def replace_inline_code(match: re.Match[str]) -> str:
        return f"{marker('CODE_REF', match.group(0))} inline_literal_present"

    def replace_url(match: re.Match[str]) -> str:
        value = match.group(0)
        return f"{marker('URL_REF', value)} {_url_category(value)}"

    def replace_path(match: re.Match[str]) -> str:
        return f"{marker('PATH_REF', match.group(0))} filesystem_path"

    safe = _FENCED_CODE_RE.sub(replace_fenced_code, text)
    safe = _COMMAND_LINE_RE.sub(replace_command, safe)
    safe = _INLINE_CODE_RE.sub(replace_inline_code, safe)
    safe = _URL_RE.sub(replace_url, safe)
    safe = _ABSOLUTE_PATH_RE.sub(replace_path, safe)
    for pattern, replacement in _SENSITIVE_TERM_REPLACEMENTS:
        safe = pattern.sub(replacement, safe)
    safe = re.sub(r"\n{3,}", "\n\n", safe).strip()
    return safe


def _url_category(value: str) -> str:
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if "youtu" in host or "video" in host:
        return "external_video_platform_url"
    if host:
        return "external_http_url"
    return "http_url"


def _restore_filter_safe_evidence(
    result: ClauseSampleList,
    restore_map: dict[str, str],
) -> ClauseSampleList:
    restored: list[ClauseSample] = []
    for sample in result.clauses:
        evidence_span = _restore_filter_safe_text(
            sample.evidence_span,
            restore_map,
            prefer_full_evidence=True,
        )
        restored.append(
            sample.model_copy(
                update={
                    "target": _restore_filter_safe_optional_text(sample.target, restore_map),
                    "constraint": _restore_filter_safe_optional_text(
                        sample.constraint,
                        restore_map,
                    ),
                    "evidence_span": evidence_span,
                    "confidence_note": _restore_filter_safe_text(
                        sample.confidence_note,
                        restore_map,
                    ),
                }
            )
        )
    return result.model_copy(update={"clauses": restored})


def _restore_filter_safe_optional_text(
    value: str | None,
    restore_map: dict[str, str],
) -> str | None:
    if value is None:
        return None
    return _restore_filter_safe_text(value, restore_map)


def _restore_filter_safe_text(
    value: str,
    restore_map: dict[str, str],
    *,
    prefer_full_evidence: bool = False,
) -> str:
    stripped = value.strip()
    if prefer_full_evidence and stripped in restore_map:
        return restore_map[stripped]
    restored = value
    for marker, original in sorted(restore_map.items(), key=lambda item: -len(item[0])):
        restored = restored.replace(marker, original)
    return restored


def _is_content_filter_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code != 400:
        return False
    text = str(exc).lower()
    return (
        "contentfilter" in text
        or "content filter" in text
        or "content-filter" in text
        or "1301" in text
    )
