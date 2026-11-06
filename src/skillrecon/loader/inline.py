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
    r"(?:^|[，。；:：]\s*)(?:请|先|然后|接着|务必|直接|立即|需要|必须|可以|应当)?"
    r"(?:(?:执行|运行|安装|调用)(?!后|前|时|的|结果|返回)|使用(?:以下|此|该)?(?:命令|脚本))",
)
_BINDING_KEYWORDS_EN = re.compile(
    r"(?i)(?:^|[,;:.]\s*)(?:please\s+|then\s+|next\s+|first\s+|now\s+|you\s+(?:must|should|can)\s+)?"
    r"(?:(?:execute|run|install|invoke)\b|use\s+(?:the\s+)?(?:following\s+)?(?:command|script)\b)",
)
_HEADING_POSITIVE = re.compile(r"(?i)模板|template|脚本|script|代码|code")
_HEADING_NEGATIVE = re.compile(r"(?i)示例|example|schema|规范|spec")
_PROHIBITIVE_BINDING = re.compile(r"(?i)^(?:please\s+)?(?:do not\b|don't\b|must not\b|never\b|禁止|不得|不要|切勿)")


def is_prohibitive_binding(text: str) -> bool:
    """A prohibition introduces a constraint, rather than an execution request."""
    text = re.sub(r'^(?:(?:[-+*]|\d+[.)]|>)\s+)+','',text.strip())
    return bool(_PROHIBITIVE_BINDING.search(text.lstrip('*_ ')))


def has_binding_keyword(text: str) -> bool:
    """Return whether prose contains an operational binding cue."""
    text = re.sub(r'^(?:(?:[-+*]|\d+[.)]|>)\s+)+','',text.strip())
    text = re.sub(r'[*_]','',text)
    return bool(_BINDING_KEYWORDS_ZH.search(text) or _BINDING_KEYWORDS_EN.search(text))


def check_binding_signal(
    preceding_block: DocBlock | None,
    code_block: DocBlock,
) -> str | None:
    """Return the instruction text that binds a code block, if any."""
    if (preceding_block is not None and preceding_block.doc_id == code_block.doc_id
            and preceding_block.block_type in {"paragraph", "list_item"}):
        text = preceding_block.content
        if not is_prohibitive_binding(text) and has_binding_keyword(text):
            return text

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
