"""Heuristic step extraction over parsed document blocks."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import PurePosixPath

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
_STEP_LABEL_RE = re.compile(
    r"(?i)^\s*(?:(?:step|步骤)\s*(\d+)|第\s*(\d+)\s*步)\s*[:.：、]\s*"
)
_STEP_BEFORE_AFTER_RE = re.compile(r"(?i)\b(before|after)\s+step\s+(\d+)\b")
_STEP_BEFORE_AFTER_ZH_RE = re.compile(
    r"(?:步骤\s*(\d+)|第\s*(\d+)\s*步)\s*(之前|之后|前|后)"
)
_SEQUENCE_DECLARATION_RE = re.compile(
    r"(?i)\b(?:run|execute|perform|follow)\b[^\n]*\b(?:steps?|procedure)\b"
    r"[^\n]*\b(?:in order|in sequence|sequentially)\b|"
    r"(?:按顺序|依次)(?:执行|运行)(?:以下|下列)?(?:步骤|操作)"
)
_ORDER_REQUIREMENT_RE = re.compile(
    r"(?i)^\s*(?:(?:please|only|always|must)\s+)?"
    r"(?:run|execute|read|write|send|upload|download|validate|check|save|delete|install|fetch)\b|"
    r"\b(?:must|shall|required|only after|only before)\b|"
    r"^\s*(?:执行|运行|读取|写入|上传|下载|发送|检查|验证|保存|删除|安装|必须|应当|仅)"
)
_ORDER_CUE_RE = re.compile(
    r"(?i)\b(?:before|after|until|then)\b|\bin (?:order|sequence)\b(?!\s+to\b)|"
    r"之前|之后|然后|按顺序|依次"
)


def build_steps(
    doc_blocks: list[DocBlock],
) -> tuple[list[Step], list[StepOrderEdge]]:
    """Keep document order separate from source-declared execution order."""
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
        for source, target in zip(doc_steps, doc_steps[1:], strict=False):
            step_order_edges.append(
                StepOrderEdge(
                    edge_id=f"document-order-{len(step_order_edges)}",
                    source_step_id=source.step_id,
                    target_step_id=target.step_id,
                    relation="document_precedes",
                )
            )

    semantic_edges, unresolved = extract_step_order(steps)
    steps = [step.model_copy(update={"unresolved_order":step.step_id in unresolved}) for step in steps]
    return steps, [*step_order_edges, *semantic_edges]


def extract_step_order(steps: list[Step]) -> tuple[list[StepOrderEdge], set[str]]:
    """Recover bounded explicit ordering and retain unresolved requirements.

    Supported forms are imperative/normative before/after references to a
    labeled step in the same document section, and a list explicitly declared
    to run in order. Paragraph position alone never supplies execution order.
    """
    labeled: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    ordered = sorted(steps, key=lambda step: (step.doc_id, step.order_index))
    for step in ordered:
        match = _STEP_LABEL_RE.match(step.text)
        if match:
            label = int(match.group(1) or match.group(2))
            labeled[(step.doc_id, step.heading_context, label)].append(step.step_id)

    edges: list[StepOrderEdge] = []
    unresolved: set[str] = set()
    seen: set[tuple[str, str]] = set()
    workflow: tuple[str, str] | None = None
    previous: str | None = None

    def add_edge(source: str, target: str) -> None:
        if (source, target) in seen:
            return
        seen.add((source, target))
        edges.append(
            StepOrderEdge(
                edge_id=f"semantic-precedes-{len(edges)}",
                source_step_id=source,
                target_step_id=target,
            )
        )

    for step in ordered:
        section = (step.doc_id, step.heading_context)
        label = _STEP_LABEL_RE.match(step.text)
        body = step.text[label.end():] if label else step.text
        if _SEQUENCE_DECLARATION_RE.search(body):
            workflow, previous = section, None
            continue
        if _SEQUENCE_DECLARATION_RE.search(step.heading_context) and workflow != section:
            workflow, previous = section, None
        in_workflow = workflow == section and (
            step.step_type in {"list_item", "bound_code_block"} or label is not None
        )
        if in_workflow:
            if previous is not None:
                add_edge(previous, step.step_id)
            previous = step.step_id
        else:
            workflow, previous = None, None

        if not _ORDER_REQUIREMENT_RE.search(body):
            continue
        references = [
            (match.group(1).lower(), int(match.group(2)))
            for match in _STEP_BEFORE_AFTER_RE.finditer(body)
        ]
        references.extend(
            (
                "before" if match.group(3) in {"之前", "前"} else "after",
                int(match.group(1) or match.group(2)),
            )
            for match in _STEP_BEFORE_AFTER_ZH_RE.finditer(body)
        )
        for direction, number in references:
            targets = labeled.get((step.doc_id, step.heading_context, number), [])
            if len(targets) != 1:
                unresolved.add(step.step_id)
                continue
            source, target = (
                (step.step_id, targets[0])
                if direction == "before"
                else (targets[0], step.step_id)
            )
            add_edge(source, target)
        remainder = _STEP_BEFORE_AFTER_RE.sub("", body)
        remainder = _STEP_BEFORE_AFTER_ZH_RE.sub("", remainder)
        if in_workflow:
            remainder = re.sub(r"(?i)^\s*then\b|^\s*然后", "", remainder)
        if _ORDER_CUE_RE.search(remainder):
            unresolved.add(step.step_id)

    temporal = re.compile(r"(?i)(.+?)\s+\b(before|after|prior to)\b\s+(.+?)[.!;]?$" )
    action = re.compile(r"(?i)\b(authenticat(?:e|ing)|validat(?:e|ing)|verif(?:y|ying)|"
                        r"upload(?:ing)?|download(?:ing)?|send(?:ing)?|read(?:ing)?|"
                        r"writ(?:e|ing)|post(?:ing)?|sanitiz(?:e|ing)|dispatch(?:ing)?)\b")

    def operation_tokens(text: str) -> set[str]:
        words = set()
        for match in action.finditer(text):
            word = match.group().lower()
            if word.endswith("ing"):
                word = word[:-3]
            words.add(word.rstrip("e"))
        return words

    def resolve_phrase(phrase: str, directive: Step) -> Step | None:
        words = operation_tokens(phrase)
        if not words:
            return None
        referenced_files = re.findall(r"[\w./-]+\.(?:md|json|ya?ml)", phrase)
        targets = [candidate for candidate in ordered
                   if candidate.step_id != directive.step_id
                   and words <= operation_tokens(candidate.text)
                   and (candidate.doc_id == directive.doc_id if not referenced_files else
                        candidate.document_path is not None and any(
                            candidate.document_path == path or PurePosixPath(candidate.document_path).name == path
                            for path in referenced_files))]
        return targets[0] if len(targets) == 1 else None

    for directive in ordered:
        if not _ORDER_REQUIREMENT_RE.search(directive.text):
            continue
        match = temporal.search(directive.text)
        if match is None or _STEP_BEFORE_AFTER_RE.search(directive.text):
            continue
        first, second = resolve_phrase(match[1], directive), resolve_phrase(match[3], directive)
        if first is not None and second is not None:
            source, target = (second, first) if match[2].lower() == "after" else (first, second)
            add_edge(source.step_id, target.step_id)
            unresolved.discard(directive.step_id)

    return edges, unresolved


def select_steps_for_llm(steps: list[Step]) -> list[Step]:
    """Send every non-frontmatter step to ICCM's bounded-size chunks."""
    return [step for step in steps if step.step_type != "frontmatter"]


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
            start_byte=block.start_byte,
            end_byte=block.end_byte,
        )

    if block.block_type == "code_block":
        binding = check_binding_signal(preceding_block, block)
        return Step(
            step_id=block.block_id,
            doc_id=block.doc_id,
            order_index=local_index,
            local_index=local_index,
            step_type="bound_code_block" if binding is not None else "example_code_block",
            text=block.content,
            block_ids=[block.block_id],
            heading_context=block.heading_context,
            evidence=_evidence_for_block(block),
            exec_bound=binding is not None,
            start_byte=block.start_byte,
            end_byte=block.end_byte,
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
        start_byte=block.start_byte,
        end_byte=block.end_byte,
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


def _evidence_for_block(block: DocBlock) -> EvidenceSpan:
    return EvidenceSpan(
        doc_id=block.doc_id,
        start_offset=block.start_offset,
        end_offset=block.end_offset,
        text=block.source_text or block.content,
        start_byte=block.start_byte,
        end_byte=block.end_byte,
    )
