"""Standalone renderers for Evaluation Section figures embedded in main.tex."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/skillrecon-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CHAPTER4_FIGURES = {
    "fig:cost": "fig-cost",
    "fig:error": "fig-error",
    "fig:language": "fig-language",
    "fig:sensitivity": "fig-sensitivity",
}


def render_chapter4_figures(
    *,
    bundle: Mapping[str, object],
    output_dir: Path,
    figure_spec: Mapping[str, object] | None = None,
    figures: Iterable[str] | None = None,
    file_format: str = "pdf",
) -> dict[str, str]:
    """Render paper Section 4 figures and return a label-to-path manifest.

    The figure spec is optional. When present, it carries experiment outputs
    that are not stored in the core RQ bundle, such as hyperparameter sweeps
    or per-backbone sensitivity measurements.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _resolve_requested_figures(figures)
    spec = _mapping(figure_spec)
    renderers = {
        "fig:cost": _render_cost,
        "fig:error": _render_error,
        "fig:language": _render_language,
        "fig:sensitivity": _render_sensitivity,
    }
    written: dict[str, str] = {}
    for label in selected:
        filename = f"{CHAPTER4_FIGURES[label]}.{file_format}"
        path = output_dir / filename
        renderers[label](bundle, spec, path)
        written[label] = str(path)
    return written


def _render_cost(
    bundle: Mapping[str, object],
    figure_spec: Mapping[str, object],
    path: Path,
) -> None:
    data = _figure_data(bundle, figure_spec, "fig_cost")
    stage_share = _cost_stage_share(bundle, data)
    scaling = _mapping(data.get("scaling"))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.4))
    stage_labels = list(stage_share)
    stage_values = [_as_percent(stage_share[label]) for label in stage_labels]
    axes[0].bar(stage_labels, stage_values, color="#2f6f9f")
    axes[0].set_ylabel("% of total time")
    axes[0].set_title("(a) Wall-clock breakdown")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    for index, value in enumerate(stage_values):
        axes[0].text(index, value + 1.0, f"{value:.1f}", ha="center", fontsize=7)

    x_values = _float_list(scaling.get("files_per_package"))
    series = [
        ("SkillRecon", _float_list(scaling.get("skillrecon")), "#2f6f9f", "-"),
        ("OpenClaw", _float_list(scaling.get("openclaw")), "#9a4f2f", "--"),
        ("Capability", _float_list(scaling.get("capability_lattice")), "#8a6a2f", ":"),
        ("Rule", _float_list(scaling.get("rule_based")), "#444444", "-."),
    ]
    if x_values and any(values for _name, values, _color, _style in series):
        for name, values, color, style in series:
            if len(values) == len(x_values):
                axes[1].plot(x_values, values, marker="o", label=name, color=color, linestyle=style)
        axes[1].set_xlabel("Files per package")
        axes[1].set_ylabel("Time (s)")
        axes[1].set_title("(b) End-to-end scaling")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=7)
    else:
        _empty_panel(axes[1], "(b) End-to-end scaling", "No scaling sweep data")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_error(
    bundle: Mapping[str, object],
    figure_spec: Mapping[str, object],
    path: Path,
) -> None:
    data = _figure_data(bundle, figure_spec, "fig_error")
    residue = _normalize_shares(
        _float_mapping(
            data.get("false_authorization_residue")
            or _mapping(_mapping(bundle.get("rq1")).get("false_authorization_residue"))
        )
    )
    backbone = _mapping(
        data.get("backbone_sensitivity")
        or _mapping(_mapping(bundle.get("rq1")).get("backbone_sensitivity"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6))
    if residue:
        labels = [_display_label(label) for label in residue]
        values = [_as_percent(value) for value in residue.values()]
        positions = list(range(len(labels)))
        axes[0].barh(positions, values, color="#9a4f2f")
        axes[0].set_yticks(positions, labels, fontsize=8)
        axes[0].set_xlabel("% of false-authorization clauses")
        axes[0].set_title("(a) False-authorization residue")
        axes[0].grid(axis="x", alpha=0.25)
        for index, value in enumerate(values):
            axes[0].text(value + 0.8, index, f"{value:.1f}", va="center", fontsize=7)
    else:
        false_authorization = _mapping(bundle.get("rq1")).get("false_authorization_rate")
        if isinstance(false_authorization, int | float):
            value = _as_percent(float(false_authorization))
            axes[0].barh([0], [value], color="#9a4f2f")
            axes[0].set_yticks([0], ["Overall false authorization"], fontsize=8)
            axes[0].set_xlim(0, max(100.0, value + 5.0))
            axes[0].set_xlabel("%")
            axes[0].set_title("(a) False authorization")
        else:
            _empty_panel(axes[0], "(a) False-authorization residue", "No residue labels")

    if backbone:
        labels = [_display_label(label) for label in backbone]
        f1_values = [_as_percent(_metric(row, "f1")) for row in backbone.values()]
        fa_values = [
            _as_percent(_metric(row, "false_authorization"))
            for row in backbone.values()
        ]
        _grouped_bars(
            axes[1],
            labels,
            [("F1", f1_values, "#2f6f9f"), ("False authorization", fa_values, "#d05a55")],
        )
        axes[1].set_ylabel("Score (%)")
        axes[1].set_title("(b) LLM backbone sensitivity")
        axes[1].legend(fontsize=7)
    else:
        _empty_panel(axes[1], "(b) LLM backbone sensitivity", "No backbone sweep data")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_language(
    bundle: Mapping[str, object],
    figure_spec: Mapping[str, object],
    path: Path,
) -> None:
    data = _figure_data(bundle, figure_spec, "fig_language")
    sample_sizes = _int_mapping(data.get("sample_sizes"))
    series_by_key = {
        "skillrecon_f1": ("SkillRecon", "#2f6f9f"),
        "openclaw_f1": ("OpenClaw", "#777777"),
        "cisco_skill_scanner_f1": ("Cisco", "#d05a55"),
        "capability_lattice_f1": ("Capability", "#8a6a2f"),
    }

    if not data:
        language_profile = _mapping(
            _mapping(bundle.get("rq5_generalization")).get("language_profile")
        )
        sample_sizes = {
            label: int(_mapping(metrics).get("n") or 0)
            for label, metrics in language_profile.items()
        }
        data = {
            "skillrecon_f1": {
                label: _mapping(metrics).get("f1")
                for label, metrics in language_profile.items()
            },
            "sample_sizes": sample_sizes,
        }

    labels = list(sample_sizes) or sorted(
        {
            label
            for key in series_by_key
            for label in _float_mapping(data.get(key))
        }
    )
    if not labels:
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        _empty_panel(ax, "Per-language skill-level F1", "No language-profile data")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    series = []
    for key, (display, color) in series_by_key.items():
        values = _float_mapping(data.get(key))
        if values:
            series.append(
                (display, [_as_percent(values.get(label, 0.0)) for label in labels], color)
            )

    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    _grouped_bars(ax, [_display_label(label) for label in labels], series)
    ax.set_ylabel("Skill-level F1 (%)")
    ax.set_title("Per-language-profile skill-level F1")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    for index, label in enumerate(labels):
        n = sample_sizes.get(label)
        if n is not None:
            ax.text(index, -8.0, f"n={n}", ha="center", va="top", fontsize=7)
    ax.set_ylim(-12, 105)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_sensitivity(
    bundle: Mapping[str, object],
    figure_spec: Mapping[str, object],
    path: Path,
) -> None:
    data = _figure_data(bundle, figure_spec, "fig_sensitivity")
    if not data:
        data = _mapping(bundle.get("hyperparameter_sensitivity"))

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.3))
    _line_panel(
        axes[0],
        _mapping(data.get("iccm_samples")),
        title="(a) ICCM samples",
        xlabel="n",
        series=(("F1", "f1", "#2f6f9f", "-"), ("False auth.", "false_authorization", "#d05a55", "--")),
    )
    _line_panel(
        axes[1],
        _mapping(data.get("majority_threshold")),
        title="(b) Majority threshold",
        xlabel="theta",
        series=(("F1", "f1", "#2f6f9f", "-"), ("False auth.", "false_authorization", "#d05a55", "--")),
    )
    _line_panel(
        axes[2],
        _mapping(data.get("candidate_budget")),
        title="(c) Candidate budget",
        xlabel="k_c",
        series=(("F1", "f1", "#2f6f9f", "-"), ("Event recall", "event_recall", "#4f7d45", ":")),
    )
    _line_panel(
        axes[3],
        _mapping(data.get("proof_neighborhood")),
        title="(d) Proof neighborhood",
        xlabel="threshold",
        series=(("Exact minimality", "exact_minimality", "#2f6f9f", "-"), ("Greedy share", "greedy_share", "#8a6a2f", "--")),
    )
    axes[0].set_ylabel("Score (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def _cost_stage_share(
    bundle: Mapping[str, object],
    data: Mapping[str, object],
) -> dict[str, float]:
    explicit = _float_mapping(data.get("stage_share"))
    if explicit:
        return _normalize_shares(explicit)

    runtime = _mapping(_mapping(bundle.get("rq6_robustness_cost")).get("runtime"))
    stage_share = _float_mapping(runtime.get("stage_share"))
    grouped: dict[str, float] = {}
    for key, value in stage_share.items():
        label = _stage_label(key)
        grouped[label] = grouped.get(label, 0.0) + value
    return _normalize_shares(grouped)


def _figure_data(
    bundle: Mapping[str, object],
    figure_spec: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    aliases = {
        key,
        key.replace("_", "-"),
        key.replace("_", ":"),
        f"fig:{key.removeprefix('fig_')}",
    }
    for source in (
        figure_spec,
        _mapping(bundle.get("paper_figures")),
        _mapping(bundle.get("embedded_figures")),
    ):
        for alias in aliases:
            value = source.get(alias)
            if isinstance(value, Mapping):
                return value
    return {}


def _resolve_requested_figures(figures: Iterable[str] | None) -> list[str]:
    if figures is None:
        return list(CHAPTER4_FIGURES)
    requested: list[str] = []
    for item in figures:
        if item == "all":
            requested.extend(CHAPTER4_FIGURES)
            continue
        label = item if item.startswith("fig:") else f"fig:{item.removeprefix('fig-')}"
        if label not in CHAPTER4_FIGURES:
            raise ValueError(f"unknown Section 4 figure label: {item}")
        requested.append(label)
    return list(dict.fromkeys(requested))


def _line_panel(
    ax,
    panel: Mapping[str, object],
    *,
    title: str,
    xlabel: str,
    series: Sequence[tuple[str, str, str, str]],
) -> None:
    x_raw = _list(panel.get("x"))
    if not x_raw:
        _empty_panel(ax, title, "No sweep data")
        return
    x_positions = list(range(len(x_raw)))
    has_series = False
    for display, key, color, style in series:
        values = [_as_percent(value) for value in _float_list(panel.get(key))]
        if len(values) != len(x_positions):
            continue
        has_series = True
        ax.plot(x_positions, values, marker="o", label=display, color=color, linestyle=style)
    if not has_series:
        _empty_panel(ax, title, "No series data")
        return
    ax.set_xticks(x_positions, [str(item) for item in x_raw], rotation=35, fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)


def _grouped_bars(
    ax,
    labels: Sequence[str],
    series: Sequence[tuple[str, Sequence[float], str]],
) -> None:
    if not series:
        _empty_panel(ax, "", "No series data")
        return
    width = 0.78 / len(series)
    x_positions = list(range(len(labels)))
    for index, (name, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(
            [x + offset for x in x_positions],
            list(values),
            width=width,
            label=name,
            color=color,
        )
    ax.set_xticks(x_positions, labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)


def _empty_panel(ax, title: str, message: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _stage_label(key: str) -> str:
    cleaned = key.lower().replace("_share", "").replace("_seconds", "")
    if cleaned in {"contract", "targeted_recall"}:
        return "ICCM"
    if cleaned in {"codeql"}:
        return "CodeQL"
    if cleaned in {"behavior"}:
        return "Behavior"
    if cleaned in {"reconciliation", "reconcile"}:
        return "Reconcile"
    if cleaned in {"witness"}:
        return "Witness"
    return "I/O+metrics"


def _normalize_shares(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}
    total = sum(value for value in values.values() if value > 0)
    if total == 0:
        return {}
    if 0.98 <= total <= 1.02:
        return dict(values)
    if 98.0 <= total <= 102.0:
        return {key: value / 100.0 for key, value in values.items()}
    return {key: value / total for key, value in values.items()}


def _as_percent(value: float) -> float:
    return value * 100.0 if abs(value) <= 1.0 else value


def _metric(value: object, key: str) -> float:
    payload = _mapping(value)
    raw = payload.get(key)
    return float(raw) if isinstance(raw, int | float) else 0.0


def _display_label(value: object) -> str:
    return str(value).replace("_", " ").replace("-", " ").title()


def _mapping(value: object | None) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _float_mapping(value: object | None) -> dict[str, float]:
    payload = _mapping(value)
    return {
        str(key): float(item)
        for key, item in payload.items()
        if isinstance(item, int | float)
    }


def _int_mapping(value: object | None) -> dict[str, int]:
    payload = _mapping(value)
    return {
        str(key): int(item)
        for key, item in payload.items()
        if isinstance(item, int | float)
    }


def _float_list(value: object | None) -> list[float]:
    return [float(item) for item in _list(value) if isinstance(item, int | float)]


def _list(value: object | None) -> list[object]:
    return list(value) if isinstance(value, list | tuple) else []
