"""Parse markdown documents into structured DocBlock sequences."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from markdown_it import MarkdownIt

from skillrecon.core.types import DocBlock

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?\n)---\n", re.DOTALL)


def parse_yaml_frontmatter(content: str) -> tuple[str, int]:
    """Return frontmatter text and its end offset."""
    match = _FRONTMATTER_RE.match(content)
    if match:
        return match.group(1), match.end()
    return "", 0


def parse_frontmatter_fields(frontmatter_text: str) -> dict[str, str]:
    """Parse the small SKILL.md frontmatter subset used by the pipeline."""
    fields: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    for line in frontmatter_text.split("\n"):
        kv_match = re.match(r"^(\w[\w-]*)\s*:\s*(.*)", line)
        if kv_match:
            if current_key is not None:
                fields[current_key] = "\n".join(current_lines).strip()
            current_key = kv_match.group(1)
            value = kv_match.group(2).strip()
            current_lines = [value] if value and value != "|" else []
        elif current_key is not None and (line.startswith("  ") or line.startswith("\t")):
            current_lines.append(line.strip())
        elif not line.strip() and current_key is not None:
            current_lines.append("")
        elif not line.strip():
            continue
        else:
            if current_key is None and line.strip():
                fields["name"] = line.strip()

    if current_key is not None:
        fields[current_key] = "\n".join(current_lines).strip()

    return fields


def parse_document(doc_id: str, content: str) -> list[DocBlock]:
    """Parse one markdown document into offset-aware blocks."""
    blocks: list[DocBlock] = []
    block_idx = 0

    fm_text, fm_end = parse_yaml_frontmatter(content)
    if fm_text:
        blocks.append(
            DocBlock(
                block_id=f"{doc_id}_b{block_idx}",
                doc_id=doc_id,
                block_type="frontmatter",
                content=fm_text.strip(),
                start_offset=0,
                end_offset=fm_end,
                heading_context="",
            )
        )
        block_idx += 1

    md = MarkdownIt()
    body = content[fm_end:] if fm_end > 0 else content
    tokens = md.parse(body)
    body_offset = fm_end

    heading_stack: list[str] = []

    for token in tokens:
        if token.type == "heading_open":
            continue

        if token.type == "heading_close":
            continue

        if token.type == "inline" and token.map is not None:
            parent_type = _get_parent_type(tokens, token)
            start_line, end_line = token.map
            start_off = _line_to_offset(body, start_line) + body_offset
            end_off = _line_to_offset(body, end_line) + body_offset
            text = token.content

            if parent_type == "heading":
                level = _get_heading_level(tokens, token)
                while len(heading_stack) >= level:
                    heading_stack.pop()
                heading_stack.append(text)
                block_type = "heading"
            elif parent_type == "list_item":
                block_type = "list_item"
            else:
                block_type = "paragraph"

            heading_ctx = " > ".join(heading_stack) if heading_stack else ""
            blocks.append(
                DocBlock(
                    block_id=f"{doc_id}_b{block_idx}",
                    doc_id=doc_id,
                    block_type=block_type,
                    content=text,
                    start_offset=start_off,
                    end_offset=end_off,
                    heading_context=heading_ctx,
                )
            )
            block_idx += 1

        elif token.type == "fence" and token.map is not None:
            start_line, end_line = token.map
            start_off = _line_to_offset(body, start_line) + body_offset
            end_off = _line_to_offset(body, end_line) + body_offset
            heading_ctx = " > ".join(heading_stack) if heading_stack else ""

            blocks.append(
                DocBlock(
                    block_id=f"{doc_id}_b{block_idx}",
                    doc_id=doc_id,
                    block_type="code_block",
                    content=token.content,
                    start_offset=start_off,
                    end_offset=end_off,
                    heading_context=heading_ctx,
                    language_hint=token.info.strip(),
                )
            )
            block_idx += 1

        elif token.type == "html_block" and token.map is not None:
            start_line, end_line = token.map
            start_off = _line_to_offset(body, start_line) + body_offset
            end_off = _line_to_offset(body, end_line) + body_offset
            heading_ctx = " > ".join(heading_stack) if heading_stack else ""
            text = token.content or ""

            blocks.append(
                DocBlock(
                    block_id=f"{doc_id}_b{block_idx}",
                    doc_id=doc_id,
                    block_type="html_block",
                    content=text,
                    start_offset=start_off,
                    end_offset=end_off,
                    heading_context=heading_ctx,
                )
            )
            block_idx += 1

    logger.debug("Parsed %d blocks from doc %s", len(blocks), doc_id)
    return blocks


def _line_to_offset(content: str, line_num: int) -> int:
    """Convert a 0-based line number to a character offset."""
    offset = 0
    for i, line in enumerate(content.split("\n")):
        if i == line_num:
            return offset
        offset += len(line) + 1
    return len(content)


def _get_parent_type(tokens: Sequence[object], inline_token: object) -> str:
    """Determine the parent block type of an inline token."""
    idx = tokens.index(inline_token)
    if idx > 0:
        prev = tokens[idx - 1]
        if hasattr(prev, "type") and prev.type == "heading_open":
            return "heading"
        if hasattr(prev, "type") and prev.type == "paragraph_open":
            for lookback in range(idx - 2, -1, -1):
                candidate = tokens[lookback]
                if not hasattr(candidate, "type"):
                    continue
                if candidate.type == "list_item_open":
                    return "list_item"
                if candidate.type in {
                    "ordered_list_open",
                    "bullet_list_open",
                    "heading_open",
                    "paragraph_open",
                }:
                    break
    return "paragraph"


def _get_heading_level(tokens: Sequence[object], inline_token: object) -> int:
    """Return the heading level implied by the preceding heading token."""
    idx = tokens.index(inline_token)
    if idx > 0:
        prev = tokens[idx - 1]
        if hasattr(prev, "tag") and prev.tag.startswith("h"):
            try:
                return int(prev.tag[1:])
            except ValueError:
                pass
    return 1
