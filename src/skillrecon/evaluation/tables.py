"""LaTeX table emitters for the paper evaluation section."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


_RQ2_MAIN_ROWS = [
    ("baseline_rule_scanner", "Rule-based scanner"),
    ("baseline_llm_judge", "LLM-as-judge"),
    ("baseline_capability_lattice", "Capability-lattice"),
    ("baseline_openclaw", "OpenClaw Scan"),
    ("baseline_doc_code_consistency", "Doc-code consistency"),
    ("baseline_spec_containment", "Spec containment"),
    ("baseline_instruction_constraints", "Instruction constraints"),
    ("baseline_skillfortify", "SkillFortify"),
    ("baseline_cisco_skill_scanner", "Cisco Skill Scanner"),
    ("baseline_skillspector", "SkillSpector"),
    ("baseline_snyk_agent_scan", "Snyk Agent Scan"),
    ("skillrecon", r"\toolname"),
]

_RQ2_ABLATION_ROWS = [
    ("skillrecon", r"\toolname (full)"),
    ("ablation_no_iccm", r"\quad w/o ICCM (keyword extraction)"),
    ("ablation_no_scope_constraints", r"\quad w/o scope constraints"),
    ("ablation_no_composition_analysis", r"\quad w/o composition analysis"),
    ("ablation_no_authorization_guard", r"\quad w/o authorization guard"),
]

_VIOLATION_SUBTYPES = [
    ("unsupported_behavior", "Unsupported behavior (V1)"),
    ("scope_violation", "Scope violation (V2)"),
    ("unjustified_composition", "Unjustified composition (V3)"),
]


def render_baseline_table() -> str:
    """Render T1 baseline capability table as LaTeX."""
    return "\n".join(
        [
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{@{}lccccc@{}}",
            r"\toprule",
            (
                r"\textbf{Method} & \textbf{Documentation} "
                r"& \textbf{Implementation} & \textbf{Graph reconciliation} "
                r"& \textbf{Scope reasoning} & \textbf{Witness extraction} \\"
            ),
            r"\midrule",
            r"Rule-based scanner & \xmark & \cmark & \xmark & \xmark & \xmark \\",
            r"LLM-as-judge & \cmark & \cmark & \xmark & \xmark & \xmark \\",
            r"Capability-lattice & \cmark & \cmark & \xmark & \xmark & \xmark \\",
            r"OpenClaw Scan & \cmark & \cmark & \xmark & \xmark & \xmark \\",
            r"Doc-code consistency & \cmark & \cmark & \xmark & partial & \xmark \\",
            r"Spec containment & \cmark & \cmark & \xmark & \xmark & \xmark \\",
            r"Instruction constraints & \cmark & \cmark & \xmark & partial & \xmark \\",
            r"SkillFortify & \cmark & \cmark & \xmark & partial & \xmark \\",
            r"Cisco Skill Scanner & \cmark & \cmark & \xmark & partial & \xmark \\",
            r"SkillSpector & \cmark & \cmark & \xmark & partial & \xmark \\",
            r"Snyk Agent Scan & \cmark & \cmark & \xmark & \xmark & \xmark \\",
            r"\midrule",
            r"\toolname & \cmark & \cmark & \cmark & \cmark & \cmark \\",
            r"\bottomrule",
            r"\end{tabular}}",
        ]
    )


def write_table(path: Path, content: str) -> None:
    """Write a generated table to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def render_rq1_clause_table(rq1: Mapping[str, object]) -> str:
    """Render T2: contract clause/operator validity."""
    clause_metrics = _mapping(rq1.get("clause_metrics"))
    false_authorization_rate = rq1.get("false_authorization_rate")
    rows: list[str] = []
    for operator in ["allowed", "prohibited", "unknown"]:
        metric = _mapping(clause_metrics.get(operator))
        false_auth = (
            _fmt_pct(false_authorization_rate)
            if operator == "allowed"
            else "N/A"
        )
        rows.append(
            " & ".join(
                [
                    rf"\textsc{{{operator}}}",
                    _fmt_pct(metric.get("precision")),
                    _fmt_pct(metric.get("recall")),
                    _fmt_pct(metric.get("f1")),
                    false_auth,
                ]
            )
            + r" \\"
        )
    overall = _mapping(rq1.get("overall_clause_metrics"))
    rows.append(
        " & ".join(
            [
                r"\textbf{Overall}",
                _fmt_pct(overall.get("precision")),
                _fmt_pct(overall.get("recall")),
                _fmt_pct(overall.get("f1")),
                _fmt_pct(false_authorization_rate),
            ]
        )
        + r" \\"
    )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            (
                r"\textbf{Clause operator} & \textbf{Precision} & \textbf{Recall} "
                r"& \textbf{F1} & \textbf{False authorization} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def render_rq1_edge_table(rq1: Mapping[str, object]) -> str:
    """Render T3: reconciliation edge validity."""
    edge_validity = _mapping(rq1.get("edge_validity_by_type"))
    rows: list[str] = []
    for edge_type in ["supports", "contradicts", "scope_matches", "scope_violates"]:
        metric = _mapping(edge_validity.get(edge_type))
        rows.append(
            " & ".join(
                [
                    _tex_label(edge_type),
                    _fmt_pct(metric.get("pos_valid")),
                    _fmt_pct(metric.get("neg_rej")),
                    _fmt_pct(metric.get("bal_valid")),
                ]
            )
            + r" \\"
        )
    overall = _mapping(rq1.get("overall_edge_validity"))
    rows.append(
        " & ".join(
            [
                r"\textbf{Overall}",
                _fmt_pct(overall.get("pos_valid")),
                _fmt_pct(overall.get("neg_rej")),
                _fmt_pct(overall.get("bal_valid")),
            ]
        )
        + r" \\"
    )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            (
                r"\textbf{Edge type} & \textbf{Positive-edge validity} "
                r"& \textbf{Negative-edge rejection} "
                r"& \textbf{Balanced validity} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def render_rq2_slice_table(rq2: Mapping[str, object]) -> str:
    """Render T4: high/medium risk detection."""
    rows: list[str] = []
    for system_id, label in _ordered_main_rows(rq2):
        by_slice = _mapping(_mapping(rq2.get(system_id)).get("by_slice"))
        high = _mapping(by_slice.get("high_risk"))
        medium = _mapping(by_slice.get("medium_risk"))
        rows.append(
            " & ".join(
                [
                    label,
                    _fmt_pct(high.get("precision")),
                    _fmt_pct(high.get("recall")),
                    _fmt_pct(high.get("f1")),
                    _fmt_pct(medium.get("precision")),
                    _fmt_pct(medium.get("recall")),
                    _fmt_pct(medium.get("f1")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{@{}lccccccc@{}}",
            r"\toprule",
            (
                r"& \multicolumn{3}{c}{\textbf{High}} "
                r"& \multicolumn{3}{c}{\textbf{Medium}} \\"
            ),
            r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
            (
                r"\textbf{Method} & \textbf{Precision} & \textbf{Recall} "
                r"& \textbf{F1} & \textbf{Precision} & \textbf{Recall} "
                r"& \textbf{F1} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
        ]
    )


def render_rq2_type_table(
    rq2: Mapping[str, object],
    rq2_meta: Mapping[str, object],
) -> str:
    """Render T5: detection F1 by violation type V1/V2/V3."""
    support = _mapping(rq2_meta.get("subtype_support"))
    rows: list[str] = []
    for system_id, label in _ordered_main_rows(rq2):
        system_metrics = _mapping(rq2.get(system_id))
        by_subtype = _mapping(
            system_metrics.get("paper_by_subtype") or system_metrics.get("by_subtype")
        )
        row = [label]
        for subtype, _label in _VIOLATION_SUBTYPES:
            metric = _mapping(by_subtype.get(subtype))
            row.append(_fmt_pct(metric.get("f1")))
        rows.append(" & ".join(row) + r" \\")

    support_row = [r"\textit{Support}"]
    for subtype, _label in _VIOLATION_SUBTYPES:
        support_row.append(_fmt_int(support.get(subtype)))
    rows.append(" & ".join(support_row) + r" \\")

    return "\n".join(
        [
            r"\begin{tabular}{@{}lccc@{}}",
            r"\toprule",
            (
                r"\textbf{Method} & \textbf{Unsupported behavior (V1)} "
                r"& \textbf{Scope violation (V2)} "
                r"& \textbf{Unjustified composition (V3)} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def render_rq2_ablation_table(rq2: Mapping[str, object]) -> str:
    """Render T6: ablations on High and disputed Medium cases."""
    rows: list[str] = []
    for system_id, label in _ordered_ablation_rows(rq2):
        system_metrics = _mapping(rq2.get(system_id))
        by_slice = _mapping(system_metrics.get("by_slice"))
        disputed = _mapping(system_metrics.get("medium_disputed"))
        rows.append(
            " & ".join(
                [
                    label,
                    _fmt_pct(_mapping(by_slice.get("high_risk")).get("f1")),
                    _fmt_pct(_mapping(by_slice.get("medium_risk")).get("f1")),
                    _fmt_int(disputed.get("corrected_fp")),
                    _fmt_pct(disputed.get("recovery")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lcccc@{}}",
            r"\toprule",
            (
                r"\textbf{Variant} & \textbf{High-slice F1} "
                r"& \textbf{Medium-slice F1} "
                r"& \textbf{Corrected false positives} "
                r"& \textbf{Recovery} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )


def render_rq3_discovery_table(rq3: Mapping[str, object]) -> str:
    """Render T7: scanner-benign/low-slice discovery yield."""
    rows: list[str] = []
    for system_id, label in _ordered_main_rows(rq3):
        metrics = _mapping(rq3.get(system_id))
        confirmed_by_type = _mapping(metrics.get("confirmed_by_type"))
        rows.append(
            " & ".join(
                [
                    label,
                    _fmt_int(metrics.get("flagged")),
                    _fmt_int(metrics.get("checked") or metrics.get("audited")),
                    _fmt_int(metrics.get("confirmed")),
                    _fmt_pct(metrics.get("confirmation_rate")),
                    _fmt_int(confirmed_by_type.get("unsupported_behavior")),
                    _fmt_int(confirmed_by_type.get("scope_violation")),
                    _fmt_int(confirmed_by_type.get("unjustified_composition")),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{@{}lccccccc@{}}",
            r"\toprule",
            (
                r"& \multicolumn{4}{c}{\textbf{Discovery Yield}} "
                r"& \multicolumn{3}{c}{\textbf{Confirmed by Type}} \\"
            ),
            r"\cmidrule(lr){2-5} \cmidrule(l){6-8}",
            (
                r"\textbf{Method} & \textbf{Flagged} & \textbf{Audited} "
                r"& \textbf{Confirmed} & \textbf{Confirmation rate} "
                r"& \textbf{Unsupported behavior (V1)} "
                r"& \textbf{Scope violation (V2)} "
                r"& \textbf{Unjustified composition (V3)} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
        ]
    )


def render_rq4_witness_table(rq4: Mapping[str, object]) -> str:
    """Render T8: minimal witness fidelity and auditability."""
    skillrecon = _mapping(rq4.get("skillrecon"))
    by_subtype = _mapping(skillrecon.get("by_subtype"))
    rows: list[str] = []
    for subtype, label in _VIOLATION_SUBTYPES:
        metrics = _mapping(by_subtype.get(subtype))
        rows.append(_witness_row(label, metrics))
    rows.append(_witness_row(r"\textbf{Overall}", _mapping(skillrecon.get("overall"))))
    return "\n".join(
        [
            r"\resizebox{\columnwidth}{!}{%",
            r"\begin{tabular}{@{}lcccccc@{}}",
            r"\toprule",
            (
                r"\textbf{Subtype} & \textbf{N} & \textbf{Coverage} "
                r"& \textbf{Mechanical revalidation} "
                r"& \textbf{Independent human revalidation} "
                r"& \textbf{Irreducibility} "
                r"& \textbf{Exact-mode minimality} "
                r"& \textbf{Cross-modal} \\"
            ),
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}}",
        ]
    )


def render_all_tables(bundle: Mapping[str, object]) -> dict[str, str]:
    """Render all paper-facing LaTeX tables from an experiment bundle."""
    rq1 = _mapping(bundle.get("rq1"))
    rq2 = _mapping(bundle.get("rq2"))
    return {
        "t1_baselines.tex": render_baseline_table(),
        "t2_rq1_clause.tex": render_rq1_clause_table(rq1),
        "t3_rq1_edges.tex": render_rq1_edge_table(rq1),
        "t4_rq2_slices.tex": render_rq2_slice_table(rq2),
        "t5_rq2_types.tex": render_rq2_type_table(
            rq2,
            _mapping(bundle.get("rq2_meta")),
        ),
        "t6_rq2_ablation.tex": render_rq2_ablation_table(rq2),
        "t7_rq3_discovery.tex": render_rq3_discovery_table(
            _mapping(bundle.get("rq3")),
        ),
        "t8_rq4_witness.tex": render_rq4_witness_table(
            _mapping(bundle.get("rq4")),
        ),
    }


def _witness_row(label: str, metrics: Mapping[str, object]) -> str:
    return (
        " & ".join(
            [
                label,
                _fmt_int(metrics.get("n")),
                _fmt_pct(metrics.get("coverage")),
                _fmt_pct(metrics.get("revalidation")),
                _fmt_pct(metrics.get("independent_revalidation")),
                _fmt_pct(metrics.get("irreducibility")),
                _fmt_pct(metrics.get("exact")),
                _fmt_pct(metrics.get("cross_modal")),
            ]
        )
        + r" \\"
    )


def _ordered_main_rows(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    rows = [row for row in _RQ2_MAIN_ROWS if row[0] in payload or row[0] == "skillrecon"]
    known = {system_id for system_id, _label in rows}
    extras = [
        (system_id, _tex_label(system_id))
        for system_id in sorted(payload)
        if system_id not in known and not system_id.startswith("ablation_")
    ]
    return [*rows, *extras]


def _ordered_ablation_rows(payload: Mapping[str, object]) -> list[tuple[str, str]]:
    rows = [row for row in _RQ2_ABLATION_ROWS if row[0] in payload or row[0] == "skillrecon"]
    known = {system_id for system_id, _label in rows}
    extras = [
        (system_id, _tex_label(system_id))
        for system_id in sorted(payload)
        if system_id not in known and system_id.startswith("ablation_")
    ]
    return [*rows, *extras]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _tex_label(value: object) -> str:
    return str(value).replace("_", r"\_")


def _fmt_pct(value: object | None) -> str:
    if value is None:
        return r"\tbd"
    return f"{float(value) * 100:.1f}"


def _fmt_int(value: object | None) -> str:
    if value is None:
        return r"\tbd"
    return str(int(value))
