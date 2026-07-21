"""Figure emitters for journal-facing evaluation results."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skillrecon-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_MAIN_SYSTEM_LABELS = {
    "baseline_rule_scanner": "Rule",
    "baseline_llm_judge": "LLM",
    "baseline_capability_lattice": "Cap",
    "baseline_openclaw": "OpenClaw",
    "baseline_doc_code_consistency": "Doc-code",
    "baseline_spec_containment": "Spec",
    "baseline_instruction_constraints": "Instr.",
    "baseline_skillfortify": "SkillFortify",
    "baseline_cisco_skill_scanner": "Cisco",
    "baseline_skillspector": "SkillSpector",
    "baseline_snyk_agent_scan": "Snyk",
    "skillrecon": "SkillRecon",
}
_SUBTYPE_LABELS = {
    "unsupported_behavior": "V1",
    "scope_violation": "V2",
    "unjustified_composition": "V3",
}
_GRANULARITY_LABELS = {
    "capability_only": "Capability",
    "resource_aware": "Resource",
    "full_clause_scope": "Full",
}


def render_all_figures(bundle: Mapping[str, object], output_dir: Path) -> dict[str, str]:
    """Render all paper-facing evaluation figures and return filename map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig-rq2-type-f1.pdf": _render_rq2_type_f1,
        "fig-granularity-benefit.pdf": _render_granularity_benefit,
        "fig-generalization.pdf": _render_generalization,
        "fig-robustness-cost.pdf": _render_robustness_cost,
    }
    written: dict[str, str] = {}
    for filename, renderer in figures.items():
        path = output_dir / filename
        renderer(bundle, path)
        written[filename] = str(path)
    return written


def _render_rq2_type_f1(bundle: Mapping[str, object], path: Path) -> None:
    rq2 = _mapping(bundle.get("rq2"))
    subtypes = list(_SUBTYPE_LABELS)
    systems = [
        system_id
        for system_id in _MAIN_SYSTEM_LABELS
        if system_id in rq2
    ]
    values = [
        [
            _metric(
                _mapping(
                    _mapping(rq2.get(system_id)).get("paper_by_subtype")
                    or _mapping(rq2.get(system_id)).get("by_subtype")
                ),
                subtype,
                "f1",
            )
            for subtype in subtypes
        ]
        for system_id in systems
    ]
    _grouped_bar(
        path=path,
        group_labels=[_SUBTYPE_LABELS[item] for item in subtypes],
        series_labels=[_MAIN_SYSTEM_LABELS[item] for item in systems],
        values=values,
        ylabel="F1",
        title="Violation-type diagnostic accuracy",
    )


def _render_granularity_benefit(bundle: Mapping[str, object], path: Path) -> None:
    granularity = _mapping(bundle.get("rq3_granularity"))
    subtypes = list(_SUBTYPE_LABELS)
    systems = [system_id for system_id in _GRANULARITY_LABELS if system_id in granularity]
    values = [
        [
            _metric(_mapping(_mapping(granularity.get(system_id)).get("by_subtype")), subtype, "f1")
            for subtype in subtypes
        ]
        for system_id in systems
    ]
    _grouped_bar(
        path=path,
        group_labels=[_SUBTYPE_LABELS[item] for item in subtypes],
        series_labels=[_GRANULARITY_LABELS[item] for item in systems],
        values=values,
        ylabel="F1",
        title="Incremental benefit of reconciliation granularity",
    )


def _render_generalization(bundle: Mapping[str, object], path: Path) -> None:
    generalization = _mapping(bundle.get("rq5_generalization"))
    rows: list[tuple[str, float, int]] = []
    for section in ("language_profile", "package_structure", "documentation_closure"):
        for label, metrics in _mapping(generalization.get(section)).items():
            metric = _mapping(metrics)
            rows.append((f"{section.replace('_', ' ')}\n{label}", float(metric.get("f1") or 0.0), int(metric.get("n") or 0)))
    if not rows:
        rows = [("no data", 0.0, 0)]
    rows = sorted(rows, key=lambda item: (item[0], item[2]))
    fig_height = max(3.2, 0.42 * len(rows))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    labels = [label for label, _f1, _n in rows]
    values = [f1 for _label, f1, _n in rows]
    y_positions = range(len(rows))
    ax.barh(list(y_positions), values, color="#2f6f9f")
    ax.set_yticks(list(y_positions), labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("SkillRecon F1")
    ax.set_title("Performance by language and package structure")
    for idx, (_label, value, n) in enumerate(rows):
        ax.text(min(value + 0.02, 0.98), idx, f"n={n}", va="center", fontsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_robustness_cost(bundle: Mapping[str, object], path: Path) -> None:
    robustness = _mapping(bundle.get("rq6_robustness_cost"))
    drops = _mapping(robustness.get("ablation_drops"))
    witness_modes = _mapping(robustness.get("witness_modes"))
    graph_size = _mapping(robustness.get("graph_size"))

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2))
    ablation_labels = []
    ablation_values = []
    for system_id, metrics in drops.items():
        metric = _mapping(metrics)
        ablation_labels.append(str(system_id).replace("ablation_", "").replace("_", " "))
        ablation_values.append(float(metric.get("medium_f1_drop") or 0.0))
    if not ablation_labels:
        ablation_labels = ["no ablation"]
        ablation_values = [0.0]
    axes[0].barh(ablation_labels, ablation_values, color="#9a4f2f")
    axes[0].set_xlabel("Medium F1 drop")
    axes[0].set_title("Ablation impact")
    axes[0].grid(axis="x", alpha=0.25)

    modes = ["exact", "greedy"]
    mode_counts = [int(witness_modes.get(mode) or 0) for mode in modes]
    axes[1].bar(modes, mode_counts, color="#4f7d45")
    axes[1].set_ylabel("Witnesses")
    axes[1].set_title("Witness search mode")
    axes[1].grid(axis="y", alpha=0.25)

    graph_labels = ["G_D", "G_C", "G_X"]
    medians = [
        float(_mapping(graph_size.get("g_d_nodes")).get("median") or 0.0),
        float(_mapping(graph_size.get("g_c_nodes")).get("median") or 0.0),
        float(_mapping(graph_size.get("g_x_nodes")).get("median") or 0.0),
    ]
    axes[2].bar(graph_labels, medians, color="#5f5f8f")
    axes[2].set_ylabel("Median nodes")
    axes[2].set_title("Graph size proxy")
    axes[2].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _grouped_bar(
    *,
    path: Path,
    group_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    ylabel: str,
    title: str,
) -> None:
    if not series_labels:
        series_labels = ["no data"]
        values = [[0.0 for _ in group_labels]]
    width = 0.8 / max(len(series_labels), 1)
    x_positions = list(range(len(group_labels)))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    palette = ["#2f6f9f", "#9a4f2f", "#4f7d45", "#5f5f8f", "#8a6a2f", "#7c4f8f", "#327a75", "#444444"]
    for series_index, label in enumerate(series_labels):
        offset = (series_index - (len(series_labels) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_positions],
            values[series_index],
            width=width,
            label=label,
            color=palette[series_index % len(palette)],
        )
    ax.set_xticks(x_positions, group_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncols=min(4, len(series_labels)), loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _metric(section: Mapping[str, object], subtype: str, field: str) -> float:
    return float(_mapping(section.get(subtype)).get(field) or 0.0)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
