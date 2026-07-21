"""Detect bound inline code blocks and promote them to synthetic code units."""

from __future__ import annotations

import logging
import re

from skillrecon.core.types import CodeUnit, DocBlock

logger = logging.getLogger(__name__)

_RECOGNIZED_LANGUAGES: set[str] = {
    "python", "py",
    "javascript", "js",
    "typescript", "ts",
    "bash", "sh", "shell", "zsh",
    "html",
    "css",
    "ruby", "rb",
    "go", "golang",
    "rust", "rs",
    "java",
    "c", "cpp", "c++",
    "sql",
    "r",
    "php",
    "perl",
    "lua",
    "swift",
    "kotlin",
}

_HINT_NORMALIZE: dict[str, str] = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "golang": "go",
    "rs": "rust",
    "c++": "cpp",
    "rb": "ruby",
}


def has_recognized_language(hint: str) -> bool:
    """Return whether a fenced code language should be treated as executable."""
    return hint.lower() in _RECOGNIZED_LANGUAGES


def normalize_language(hint: str) -> str:
    """Normalize a language hint to canonical form."""
    lower = hint.lower()
    return _HINT_NORMALIZE.get(lower, lower)


_BINDING_KEYWORDS_ZH = re.compile(
    r"复制|执行|运行|输出|保存|写入|查阅|读取|调用|使用以下模板|使用下面模板",
)
_BINDING_KEYWORDS_EN = re.compile(
    r"(?i)\bcopy\b|\bexecute\b|\brun\b|\boutput\b|\bsave\b"
    r"|\bwrite\b|\bwrite_file\b|\bread\b|\binvoke\b|\buse\s+template\b",
)
_HEADING_POSITIVE = re.compile(r"(?i)模板|template|脚本|script|代码|code")
_HEADING_NEGATIVE = re.compile(r"(?i)示例|example|schema|规范|spec")


def has_binding_keyword(text: str) -> bool:
    """Return whether prose contains an operational binding cue."""
    return bool(_BINDING_KEYWORDS_ZH.search(text) or _BINDING_KEYWORDS_EN.search(text))


def check_binding_signal(
    preceding_block: DocBlock | None,
    code_block: DocBlock,
) -> str | None:
    """Return the instruction text that binds a code block, if any."""
    if _HEADING_NEGATIVE.search(code_block.heading_context):
        return None

    if preceding_block is not None and preceding_block.block_type in {"paragraph", "list_item"}:
        text = preceding_block.content
        if has_binding_keyword(text):
            return text

    if _HEADING_POSITIVE.search(code_block.heading_context):
        return f"[heading] {code_block.heading_context}"

    return None


def extract_synthetic_code_units(
    doc_id: str,
    file_id: str,
    blocks: list[DocBlock],
    unit_id_start: int = 0,
) -> list[CodeUnit]:
    """Promote bound inline code blocks into synthetic code units."""
    units: list[CodeUnit] = []
    unit_idx = unit_id_start

    for i, block in enumerate(blocks):
        if block.block_type != "code_block":
            continue

        if not block.language_hint or not has_recognized_language(
            block.language_hint
        ):
            continue

        preceding = blocks[i - 1] if i > 0 else None
        binding_text = check_binding_signal(preceding, block)
        if binding_text is None:
            continue

        units.append(
            CodeUnit(
                unit_id=f"syn_u{unit_idx}",
                file_id=file_id,
                language=normalize_language(block.language_hint),
                entry_point=False,
                source_doc_id=doc_id,
                source_block_id=block.block_id,
                source_offset=block.start_offset,
                heading_context=block.heading_context,
                binding_instruction=binding_text,
            )
        )
        logger.info(
            "Synthetic CodeUnit: syn_u%d (lang=%s, block=%s)",
            unit_idx,
            block.language_hint,
            block.block_id,
        )
        unit_idx += 1

    return units
