"""Heuristic step extraction over parsed document blocks."""

from __future__ import annotations

import re
from collections import defaultdict

from skillrecon.core.types import DocBlock, EvidenceSpan, Step, StepOrderEdge
from skillrecon.loader.inline import check_binding_signal

_ACTION_CUE_RE = re.compile(
    r"(?i)必须|应当|需要|先|后|检查|选择|读取|查阅|执行|运行|生成|保存|升级|安装|"
    r"点击|打开|访问|导航|过滤|搜索|添加|移除|匹配|复用|上传|下载|同步|"
    r"\bmust\b|\bneed\b|\bcheck\b|\bselect\b|\bread\b|\bconsult\b|\brun\b|"
    r"\bexecute\b|\bgenerate\b|\bsave\b|\binstall\b|\buse\b|\bopen\b|\bvisit\b|"
    r"\bnavigate\b|\bfilter\b|\bsearch\b|\badd\b|\bremove\b|\breuse\b|"
    r"\bupload\b|\bdownload\b|\bsync\b"
)
_POLICY_CUE_RE = re.compile(
    r"(?i)禁止|不得|仅|只|始终|总是|永远|不要|不能|不可|"
    r"\bnever\b|\balways\b|\bonly\b|\bdo not\b|\bdon't\b|\bcannot\b|\bcan't\b|"
    r"\bwithout\b|\bunless\b|\bwhen\b|\bif\b"
)
_MAX_LLM_STEPS = 160
_INITIAL_LLM_CONTEXT_STEPS = 8
_CORE_LLM_PRIORITY_STEPS = 96
_CONTEXT_NEIGHBOR_RADIUS = 1


def build_steps(
    doc_blocks: list[DocBlock],
) -> tuple[list[Step], list[StepOrderEdge]]:
    """Build stable step objects and precedence edges from parsed blocks."""
    steps: list[Step] = []
    step_order_edges: list[StepOrderEdge] = []
    steps_by_doc: dict[str, list[Step]] = defaultdict(list)

    for index, block in enumerate(doc_blocks):
        step = _step_from_block(
            block=block,
            preceding_block=doc_blocks[index - 1] if index > 0 else None,
            local_index=len(steps_by_doc[block.doc_id]),
        )
        if step is None:
            continue
        steps.append(step)
        steps_by_doc[block.doc_id].append(step)

    for doc_steps in steps_by_doc.values():
        for edge_index, (source, target) in enumerate(zip(doc_steps, doc_steps[1:], strict=False)):
            step_order_edges.append(
                StepOrderEdge(
                    edge_id=f"precedes-{len(step_order_edges)}",
                    source_step_id=source.step_id,
                    target_step_id=target.step_id,
                )
            )

    return steps, step_order_edges


def select_steps_for_llm(
    steps: list[Step],
    *,
    max_steps: int = _MAX_LLM_STEPS,
    initial_context_steps: int = _INITIAL_LLM_CONTEXT_STEPS,
    core_priority_steps: int = _CORE_LLM_PRIORITY_STEPS,
    context_neighbor_radius: int = _CONTEXT_NEIGHBOR_RADIUS,
) -> list[Step]:
    """Trim long step sequences before ICCM extraction.

    The goal is to keep policy- and action-bearing steps, preserve a small
    amount of leading context, and retain local neighbors around high-signal
    steps. Short documents are returned unchanged.
    """
    llm_steps = [step for step in steps if step.step_type != "frontmatter"]
    if len(llm_steps) <= max_steps:
        return llm_steps

    selected_indices: set[int] = set(range(min(initial_context_steps, len(llm_steps))))
    scored_steps = [
        (_llm_priority_score(step), index)
        for index, step in enumerate(llm_steps)
    ]
    scored_steps.sort(key=lambda item: (-item[0], item[1]))

    for _score, index in scored_steps:
        if len(selected_indices) >= min(core_priority_steps, max_steps):
            break
        selected_indices.add(index)

    expanded_indices = set(selected_indices)
    for index in sorted(selected_indices):
        for offset in range(-context_neighbor_radius, context_neighbor_radius + 1):
            neighbor = index + offset
            if 0 <= neighbor < len(llm_steps):
                expanded_indices.add(neighbor)
            if len(expanded_indices) >= max_steps:
                break
        if len(expanded_indices) >= max_steps:
            break

    ordered_indices = sorted(expanded_indices)
    if len(ordered_indices) > max_steps:
        ordered_indices = ordered_indices[:max_steps]
    return [llm_steps[index] for index in ordered_indices]


def _step_from_block(
    *,
    block: DocBlock,
    preceding_block: DocBlock | None,
    local_index: int,
) -> Step | None:
    if block.block_type == "heading":
        return None

    if block.block_type == "frontmatter":
        return Step(
            step_id=f"{block.doc_id}_frontmatter",
            doc_id=block.doc_id,
            order_index=local_index,
            local_index=local_index,
            step_type="frontmatter",
            text=block.content,
            block_ids=[block.block_id],
            heading_context=block.heading_context,
            evidence=_evidence_for_block(block),
        )

    if block.block_type == "code_block":
        binding = check_binding_signal(preceding_block, block)
        if binding is None:
            return None
        return Step(
            step_id=block.block_id,
            doc_id=block.doc_id,
            order_index=local_index,
            local_index=local_index,
            step_type="bound_code_block",
            text=block.content,
            block_ids=[block.block_id],
            heading_context=block.heading_context,
            evidence=_evidence_for_block(block),
        )

    if not block.content.strip():
        return None

    return Step(
        step_id=block.block_id,
        doc_id=block.doc_id,
        order_index=local_index,
        local_index=local_index,
        step_type=_step_type_for_block(block),
        text=block.content,
        block_ids=[block.block_id],
        heading_context=block.heading_context,
        evidence=_evidence_for_block(block),
    )


def _step_type_for_block(block: DocBlock) -> str:
    if block.block_type == "list_item":
        return "list_item"
    if _POLICY_CUE_RE.search(block.content):
        return "policy_paragraph"
    if _ACTION_CUE_RE.search(block.content):
        return "instruction_paragraph"
    if block.block_type == "html_block":
        return "html_fragment"
    return "context_paragraph"


def _llm_priority_score(step: Step) -> int:
    score_by_type = {
        "policy_paragraph": 12,
        "instruction_paragraph": 10,
        "bound_code_block": 9,
        "list_item": 7,
        "context_paragraph": 2,
        "html_fragment": 1,
    }
    score = score_by_type.get(step.step_type, 1)
    text = " ".join(part for part in (step.heading_context, step.text) if part)
    score += len(_ACTION_CUE_RE.findall(text))
    score += len(_POLICY_CUE_RE.findall(text)) * 2
    if "`" in step.text:
        score += 2
    if len(step.text.strip()) >= 160:
        score += 1
    return score


def _evidence_for_block(block: DocBlock) -> EvidenceSpan:
    return EvidenceSpan(
        doc_id=block.doc_id,
        start_offset=block.start_offset,
        end_offset=block.end_offset,
        text=block.content,
    )
