"""Markdown AST segmentation with character and UTF-8 byte provenance."""

from __future__ import annotations

import re
from collections.abc import Sequence

import yaml
from markdown_it import MarkdownIt

from skillrecon.core.types import DocBlock

_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)^---[ \t]*(?:\r?\n|$)", re.DOTALL | re.MULTILINE)


def parse_yaml_frontmatter(content: str) -> tuple[str, int]:
    """Return the YAML source and the first character after its delimiter."""
    match = _FRONTMATTER_RE.match(content)
    return (match.group(1), match.end()) if match else ("", 0)


def parse_frontmatter_fields(frontmatter_text: str) -> dict[str, object]:
    """Decode the actual YAML, preserving lists and nested permission rules."""
    fields = yaml.safe_load(frontmatter_text)
    if fields is None:
        return {}
    if not isinstance(fields, dict):
        raise ValueError("Skill frontmatter must be a YAML mapping")
    return fields


def parse_document(doc_id: str, content: str) -> list[DocBlock]:
    """Segment AST blocks; keep offsets into the original unmodified document.

    start_offset/end_offset are Python character positions for existing source
    consumers. start_byte/end_byte are the paper's UTF-8 byte positions.
    """
    blocks: list[DocBlock] = []
    headings: list[tuple[int, str]] = []
    fm_text, fm_end = parse_yaml_frontmatter(content)
    body = content[fm_end:]
    offsets = [0]
    for line in body.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def add(kind, text, start, end, **fields):
        blocks.append(DocBlock(
            block_id=f"{doc_id}_b{len(blocks)}", doc_id=doc_id,
            block_type=kind, content=text, start_offset=start, end_offset=end,
            source_text=content[start:end],
            start_byte=len(content[:start].encode("utf-8")),
            end_byte=len(content[:end].encode("utf-8")),
            heading_context=" > ".join(label for _, label in headings), **fields,
        ))

    if fm_end:
        add("frontmatter", fm_text.strip(), 0, fm_end)
    tokens = MarkdownIt("commonmark").enable("table").parse(body)
    headers: list[str] = []
    row_cells: list[str] = []
    row_map = None
    in_head = False
    in_row = False
    list_depth = 0
    for index, token in enumerate(tokens):
        if token.type == "list_item_open":
            list_depth += 1
        elif token.type == "list_item_close":
            list_depth -= 1
        elif token.type == "thead_open":
            in_head = True
        elif token.type == "thead_close":
            in_head = False
        elif token.type == "table_open":
            headers = []
        elif token.type == "tr_open":
            in_row, row_cells, row_map = True, [], token.map
        elif token.type == "tr_close":
            if in_head:
                headers = row_cells
            elif row_map is not None:
                start, end = (fm_end + offsets[line] for line in row_map)
                add("table_row", content[start:end], start, end,
                    table_headers=list(headers), table_cells=list(row_cells))
            in_row = False
        elif token.type == "inline" and in_row:
            row_cells.append(token.content)
        elif token.type == "inline" and token.map is not None:
            start, end = (fm_end + offsets[line] for line in token.map)
            previous = tokens[index - 1]
            if previous.type == "heading_open":
                level = int(previous.tag[1:])
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, token.content))
                add("heading", token.content, start, end)
            else:
                add("list_item" if list_depth else "paragraph", token.content, start, end)
        elif token.type in {"fence", "html_block"} and token.map is not None:
            start, end = (fm_end + offsets[line] for line in token.map)
            add("code_block" if token.type == "fence" else "html_block",
                token.content, start, end,
                language_hint=token.info.strip() if token.type == "fence" else "")
    return blocks


def _line_to_offset(content: str, line_num: int) -> int:
    return sum(len(line) for line in content.splitlines(keepends=True)[:line_num])


def _get_parent_type(tokens: Sequence[object], inline_token: object) -> str:
    index = next(i for i, token in enumerate(tokens) if token is inline_token)
    previous = tokens[index - 1]
    return "heading" if previous.type == "heading_open" else "paragraph"


def _get_heading_level(tokens: Sequence[object], inline_token: object) -> int:
    index = next(i for i, token in enumerate(tokens) if token is inline_token)
    return int(tokens[index - 1].tag[1:])
