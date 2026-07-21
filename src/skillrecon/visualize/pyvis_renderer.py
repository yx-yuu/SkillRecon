"""PyVis-based rendering for SkillRecon graph artifacts."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillrecon.core.types import Finding, Witness


_EXPLANATORY_RELATIONS = {
    "supports",
    "contradicts",
    "scope_violates",
    "scope_matches",
    "aligns",
}

_EXPLANATORY_CONTEXT_KINDS = {
    "declares",
    "constrains",
    "targets",
    "supported_by",
    "about",
    "reaches",
    "flows_to",
    "produces",
    "consumes",
    "emits",
    "uses_resource",
    "contains_operation",
    "realizes_event",
    "located_at",
    "classified_as_source",
    "classified_as_sink",
    "subject_of",
    "premise_for",
    "certified_by",
}


@dataclass(frozen=True)
class WireGraph:
    """A lightweight graph loaded from JSON artifacts."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def node_map(self) -> dict[str, dict[str, Any]]:
        return {
            str(node["id"]): node
            for node in self.nodes
            if isinstance(node, dict) and "id" in node
        }


@dataclass(frozen=True)
class RenderResult:
    """A rendered HTML file and its label for index generation."""

    label: str
    relative_path: str
    description: str


_NODE_STYLE_BY_KIND: dict[str, dict[str, Any]] = {
    "step": {
        "shape": "box",
        "level": 0,
        "color": {"background": "#e0f2fe", "border": "#0284c7"},
    },
    "clause": {
        "shape": "box",
        "level": 1,
        "color": {"background": "#dbeafe", "border": "#2563eb"},
    },
    "constraint": {
        "shape": "box",
        "level": 2,
        "color": {"background": "#dbeafe", "border": "#1d4ed8"},
    },
    "capability": {
        "shape": "ellipse",
        "level": 2,
        "color": {"background": "#ede9fe", "border": "#7c3aed"},
    },
    "evidence_span": {
        "shape": "box",
        "level": 3,
        "color": {"background": "#f3e8ff", "border": "#9333ea"},
    },
    "code_unit": {
        "shape": "box",
        "level": 4,
        "color": {"background": "#dcfce7", "border": "#16a34a"},
    },
    "event": {
        "shape": "ellipse",
        "level": 5,
        "color": {"background": "#bbf7d0", "border": "#16a34a"},
    },
    "capability_event": {
        "shape": "ellipse",
        "level": 5,
        "color": {"background": "#bbf7d0", "border": "#15803d"},
    },
    "resource": {
        "shape": "diamond",
        "level": 6,
        "color": {"background": "#a7f3d0", "border": "#0f766e"},
    },
    "resource_use": {
        "shape": "diamond",
        "level": 6,
        "color": {"background": "#a7f3d0", "border": "#0f766e"},
    },
    "data_object": {
        "shape": "dot",
        "level": 6,
        "size": 16,
        "color": {"background": "#fde68a", "border": "#d97706"},
    },
    "operation": {
        "shape": "box",
        "level": 5,
        "color": {"background": "#ecfccb", "border": "#65a30d"},
    },
    "location": {
        "shape": "box",
        "level": 6,
        "color": {"background": "#fef3c7", "border": "#d97706"},
    },
    "source": {
        "shape": "triangle",
        "level": 6,
        "color": {"background": "#bbf7d0", "border": "#15803d"},
    },
    "sink": {
        "shape": "triangleDown",
        "level": 6,
        "color": {"background": "#fecaca", "border": "#dc2626"},
    },
    "path": {
        "shape": "hexagon",
        "level": 6,
        "color": {"background": "#86efac", "border": "#15803d"},
    },
    "judgment": {
        "shape": "dot",
        "level": 7,
        "size": 18,
        "color": {"background": "#fde68a", "border": "#d97706"},
    },
    "certificate": {
        "shape": "diamond",
        "level": 8,
        "color": {"background": "#fecaca", "border": "#dc2626"},
    },
    "unknown": {
        "shape": "box",
        "level": 9,
        "color": {"background": "#e5e7eb", "border": "#6b7280"},
    },
}

_EDGE_STYLE_BY_RELATION: dict[str, dict[str, Any]] = {
    "supports": {"color": "#0f9f6e", "width": 5},
    "potentially_supports": {"color": "#33b07a", "width": 4, "dashes": True},
    "contradicts": {"color": "#d62839", "width": 7},
    "scope_matches": {"color": "#2f6fed", "width": 4},
    "scope_violates": {"color": "#f05a28", "width": 7},
    "aligns": {"color": "#7c3aed", "width": 4},
    "relates_to": {"color": "#8b9bb4", "width": 2, "dashes": True},
}

_EDGE_STYLE_BY_KIND: dict[str, dict[str, Any]] = {
    "subject_of": {"color": "#b87a2c", "width": 2, "dashes": True, "opacity": 0.62},
    "premise_for": {"color": "#9b6b2f", "width": 2, "dashes": True, "opacity": 0.82},
    "certified_by": {"color": "#c2185b", "width": 4, "dashes": True, "opacity": 0.92},
    "about": {"color": "#a78bfa", "width": 1, "opacity": 0.45},
    "targets": {"color": "#60a5fa", "width": 1, "opacity": 0.45},
    "constrains": {"color": "#818cf8", "width": 1, "opacity": 0.45},
    "supported_by": {"color": "#c4b5fd", "width": 1, "opacity": 0.45},
    "emits": {"color": "#86efac", "width": 1, "opacity": 0.45},
    "uses_resource": {"color": "#5eead4", "width": 1, "opacity": 0.45},
    "contains_operation": {"color": "#bef264", "width": 1, "opacity": 0.45},
    "realizes_event": {"color": "#a3e635", "width": 1, "opacity": 0.45},
    "located_at": {"color": "#fdba74", "width": 1, "dashes": True, "opacity": 0.45},
    "classified_as_source": {"color": "#4ade80", "width": 1, "opacity": 0.45},
    "classified_as_sink": {"color": "#f87171", "width": 1, "opacity": 0.45},
    "reaches": {"color": "#64748b", "width": 2, "opacity": 0.7},
    "produces": {"color": "#facc15", "width": 1, "opacity": 0.45},
    "consumes": {"color": "#fb923c", "width": 1, "opacity": 0.45},
    "flows_to": {"color": "#f59e0b", "width": 2, "opacity": 0.75},
}

_NETWORK_OPTIONS = """
const options = {
  "configure": {
    "enabled": true,
    "filter": ["layout", "physics", "edges"]
  },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "LR",
      "sortMethod": "directed",
      "levelSeparation": 220,
      "nodeSpacing": 200,
      "treeSpacing": 240,
      "blockShifting": true,
      "edgeMinimization": true,
      "parentCentralization": true
    }
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true,
    "dragNodes": true,
    "dragView": true,
    "zoomView": true,
    "hideEdgesOnDrag": false,
    "hideNodesOnDrag": false
  },
  "physics": {
    "enabled": false,
    "stabilization": false
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 1.15
      }
    },
    "smooth": {
      "enabled": true,
      "type": "cubicBezier",
      "roundness": 0.18
    },
    "font": {
      "size": 0,
      "face": "Helvetica"
    },
    "shadow": {
      "enabled": false
    },
    "selectionWidth": 1,
    "hoverWidth": 0.5
  },
  "nodes": {
    "font": {
      "face": "Helvetica",
      "size": 20,
      "color": "#0f172a"
    },
    "margin": {
      "top": 14,
      "right": 18,
      "bottom": 14,
      "left": 18
    },
    "borderWidth": 2,
    "shadow": {
      "enabled": true,
      "color": "rgba(15,23,42,0.08)",
      "size": 10,
      "x": 0,
      "y": 2
    }
  }
}
"""


def load_wire_graph(path: Path) -> WireGraph:
    """Load a graph artifact serialized with GraphObject.to_wire()."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError(f"Expected nodes/edges arrays in {path}")
    return WireGraph(nodes=nodes, edges=edges)


def load_model_list(path: Path, model_cls: type[Witness] | type[Finding]) -> list[Any]:
    """Load a list of Pydantic models from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {path}")
    return [model_cls.model_validate(item) for item in payload]


def enrich_wire_graph(graph: WireGraph, artifact_dir: Path) -> WireGraph:
    """Backfill node fields from artifact tables for paper-friendly rendering."""
    node_map = {str(node.get("id", "")): dict(node) for node in graph.nodes if str(node.get("id", ""))}
    unit_paths: dict[str, str] = {}
    code_pack_path = artifact_dir / "code_pack.json"
    if code_pack_path.is_file():
        code_pack = json.loads(code_pack_path.read_text(encoding="utf-8"))
        if isinstance(code_pack, dict):
            raw_unit_paths = code_pack.get("unit_paths", {})
            if isinstance(raw_unit_paths, dict):
                unit_paths = {str(key): str(value) for key, value in raw_unit_paths.items()}
                for unit_id, unit_path in unit_paths.items():
                    if unit_id not in node_map:
                        node_map[unit_id] = {"id": unit_id, "type": "code_unit"}
                    node_map[unit_id]["file_path"] = unit_path
                    node_map[unit_id].setdefault("type", "code_unit")

    path_table_path = artifact_dir / "path_table.json"
    if path_table_path.is_file():
        path_rows = json.loads(path_table_path.read_text(encoding="utf-8"))
        if isinstance(path_rows, list):
            for row in path_rows:
                if not isinstance(row, dict):
                    continue
                path_id = str(row.get("path_id", ""))
                if path_id not in node_map:
                    node_map[path_id] = {"id": path_id, "type": "path"}
                source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
                sink = row.get("sink", {}) if isinstance(row.get("sink"), dict) else {}
                node_map[path_id]["source_label"] = source.get("label", "")
                node_map[path_id]["sink_label"] = sink.get("label", "")
                node_map[path_id]["source_location"] = source.get("location", "")
                node_map[path_id]["sink_location"] = sink.get("location", "")
                node_map[path_id]["path_kind"] = row.get("path_kind", "")
                source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
                sink = row.get("sink", {}) if isinstance(row.get("sink"), dict) else {}
                source_event_id = str(source.get("event_id", ""))
                sink_event_id = str(sink.get("event_id", ""))
                if source_event_id in node_map:
                    node_map[source_event_id]["path_role"] = "source"
                    node_map[source_event_id]["unit_path"] = unit_paths.get(str(source.get("unit_id", "")), node_map[source_event_id].get("unit_path", ""))
                if sink_event_id in node_map:
                    node_map[sink_event_id]["path_role"] = "sink"
                    node_map[sink_event_id]["unit_path"] = unit_paths.get(str(sink.get("unit_id", "")), node_map[sink_event_id].get("unit_path", ""))

    table_specs = [
        (artifact_dir / "event_table.json", "event_id", {"type": "event"}),
        (artifact_dir / "resource_table.json", "resource_id", {"type": "resource"}),
        (artifact_dir / "canonical_clauses.json", "clause_id", {"type": "clause"}),
        (artifact_dir / "judgment_table.json", "judgment_id", {"type": "judgment"}),
        (artifact_dir / "certificate_table.json", "certificate_id", {"type": "certificate"}),
        (artifact_dir / "location_table.json", "location_id", {"type": "location"}),
        (artifact_dir / "data_object_table.json", "object_id", {"type": "data_object"}),
        (artifact_dir / "operation_table.json", "operation_id", {"type": "operation"}),
    ]
    for path, key_field, defaults in table_specs:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            node_id = str(row.get(key_field, ""))
            if not node_id:
                continue
            enriched = dict(node_map.get(node_id, {"id": node_id}))
            for key, value in defaults.items():
                enriched.setdefault(key, value)
            for key, value in row.items():
                if key in {key_field} and key != "id":
                    continue
                enriched[key] = value
            if key_field == "event_id":
                enriched.setdefault("unit_path", unit_paths.get(str(row.get("unit_id", "")), ""))
            enriched.setdefault("id", node_id)
            node_map[node_id] = enriched
    return WireGraph(nodes=sorted(node_map.values(), key=_node_sort_key), edges=graph.edges)


def extract_witness_subgraph(graph: WireGraph, witness: Witness, *, artifact_dir: Path | None = None) -> WireGraph:
    """Extract the replayable witness neighborhood from G_X."""
    node_ids = set(witness.anchor_ids)
    node_ids.update(witness.fact_node_ids)
    node_ids.update(witness.judgment_ids)
    node_ids.update(witness.certificate_ids)
    edge_ids = set(witness.projection_edge_ids)

    for edge in graph.edges:
        edge_id = str(edge.get("edge_id", ""))
        if edge_id and edge_id in edge_ids:
            node_ids.add(str(edge.get("source", "")))
            node_ids.add(str(edge.get("target", "")))

    context_node_ids = _expand_witness_context(graph, node_ids)
    node_ids.update(context_node_ids)

    node_map = graph.node_map()
    missing_nodes = _load_missing_nodes(artifact_dir, node_ids - set(node_map)) if artifact_dir else {}
    selected_nodes = {
        node_id: node_map.get(node_id, missing_nodes.get(node_id, _placeholder_node(node_id)))
        for node_id in node_ids
        if node_id
    }

    selected_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    for edge in graph.edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        edge_id = str(edge.get("edge_id", ""))
        if not (
            (edge_id and edge_id in edge_ids)
            or (source in selected_nodes and target in selected_nodes)
        ):
            continue
        edge_key = (edge_id, source, target, str(edge.get("kind", "")))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        selected_edges.append(edge)

    return _complete_graph(
        WireGraph(
        nodes=sorted(selected_nodes.values(), key=_node_sort_key),
        edges=selected_edges,
        )
    )


def extract_explanatory_subgraph(graph: WireGraph, witness: Witness, *, artifact_dir: Path | None = None) -> WireGraph:
    """Extract a case-focused explanatory subgraph from the full audit graph."""
    witness_graph = extract_witness_subgraph(graph, witness, artifact_dir=artifact_dir)
    core_node_ids = set(witness.anchor_ids)
    core_node_ids.update(witness.fact_node_ids)
    core_node_ids.update(witness.judgment_ids)
    core_node_ids.update(witness.certificate_ids)
    core_node_ids.update(_non_empty_node_ids(witness_graph.nodes))

    selected_node_ids = set(core_node_ids)

    for edge in graph.edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        relation = str(edge.get("relation", ""))
        kind = str(edge.get("kind", ""))
        if source in core_node_ids or target in core_node_ids:
            if relation in _EXPLANATORY_RELATIONS or kind in _EXPLANATORY_CONTEXT_KINDS:
                selected_node_ids.add(source)
                selected_node_ids.add(target)

    expanded = True
    while expanded:
        expanded = False
        for edge in graph.edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            relation = str(edge.get("relation", ""))
            kind = str(edge.get("kind", ""))
            if source in selected_node_ids or target in selected_node_ids:
                if relation in _EXPLANATORY_RELATIONS or kind in _EXPLANATORY_CONTEXT_KINDS:
                    if source not in selected_node_ids:
                        selected_node_ids.add(source)
                        expanded = True
                    if target not in selected_node_ids:
                        selected_node_ids.add(target)
                        expanded = True

    node_map = graph.node_map()
    selected_nodes = [
        node_map.get(node_id, _placeholder_node(node_id))
        for node_id in selected_node_ids
        if node_id
    ]
    selected_edges = [
        edge
        for edge in graph.edges
        if str(edge.get("source", "")) in selected_node_ids
        and str(edge.get("target", "")) in selected_node_ids
        and (
            str(edge.get("relation", "")) in _EXPLANATORY_RELATIONS
            or str(edge.get("kind", "")) in _EXPLANATORY_CONTEXT_KINDS
            or str(edge.get("edge_id", "")) in set(witness.projection_edge_ids)
        )
    ]
    return _complete_graph(
        WireGraph(
            nodes=sorted(selected_nodes, key=_node_sort_key),
            edges=selected_edges,
        )
    )


def render_artifact_dir(
    artifact_dir: Path,
    *,
    skill_id: str | None = None,
    output_dir: Path | None = None,
    witness_ids: set[str] | None = None,
    render_full_graph: bool = True,
) -> list[Path]:
    """Render PyVis HTML pages for one artifact directory."""
    graph = load_wire_graph(artifact_dir / "g_x.json")
    graph = enrich_wire_graph(graph, artifact_dir)
    skill_label = skill_id or artifact_dir.name
    resolved_output_dir = output_dir or artifact_dir / "viz"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    findings_path = artifact_dir / "findings.json"
    findings_by_id: dict[str, Finding] = {}
    if findings_path.is_file():
        findings_by_id = {
            finding.finding_id: finding
            for finding in load_model_list(findings_path, Finding)
        }

    rendered_files: list[Path] = []
    index_entries: list[RenderResult] = []

    if render_full_graph:
        full_graph_path = resolved_output_dir / "g_x.html"
        render_graph_html(
            graph,
            full_graph_path,
            title=f"{skill_label} / G_X",
            subtitle="Full reconciliation graph",
        )
        rendered_files.append(full_graph_path)
        index_entries.append(
            RenderResult(
                label="Full G_X",
                relative_path=full_graph_path.name,
                description=(
                    f"{len(graph.nodes)} nodes, {len(graph.edges)} edges "
                    "from the full reconciliation graph"
                ),
            )
        )

    witnesses_path = artifact_dir / "witnesses.json"
    if witnesses_path.is_file():
        witnesses = load_model_list(witnesses_path, Witness)
        for witness in witnesses:
            if witness_ids and witness.witness_id not in witness_ids:
                continue
            witness_graph = extract_witness_subgraph(graph, witness, artifact_dir=artifact_dir)
            explanatory_graph = extract_explanatory_subgraph(graph, witness, artifact_dir=artifact_dir)
            finding = findings_by_id.get(witness.finding_id)
            witness_path = resolved_output_dir / f"{witness.witness_id}.html"
            overlay_path = resolved_output_dir / f"{witness.witness_id}-overlay.html"
            explanatory_path = resolved_output_dir / f"{witness.witness_id}-explanatory.html"
            highlighted_node_ids = set(witness.anchor_ids)
            highlighted_node_ids.update(witness.fact_node_ids)
            highlighted_node_ids.update(witness.judgment_ids)
            highlighted_node_ids.update(witness.certificate_ids)
            highlighted_edge_ids = set(witness.projection_edge_ids)
            highlighted_edge_ids.update(_collect_witness_edge_ids(witness_graph, highlighted_node_ids))
            explanatory_highlighted_edge_ids = _collect_witness_edge_ids(explanatory_graph, highlighted_node_ids)
            render_graph_html(
                witness_graph,
                witness_path,
                title=f"{skill_label} / {witness.witness_id}",
                subtitle=_witness_subtitle(witness, finding),
                anchor_ids=set(witness.anchor_ids),
                highlighted_edge_ids=highlighted_edge_ids,
                highlighted_node_ids=highlighted_node_ids,
            )
            render_graph_html(
                graph,
                overlay_path,
                title=f"{skill_label} / {witness.witness_id} / overlay",
                subtitle=_witness_subtitle(witness, finding) + " | full G_X with witness highlighted",
                anchor_ids=set(witness.anchor_ids),
                highlighted_edge_ids=highlighted_edge_ids,
                highlighted_node_ids=highlighted_node_ids,
            )
            render_graph_html(
                explanatory_graph,
                explanatory_path,
                title=f"{skill_label} / {witness.witness_id} / explanatory",
                subtitle=_witness_subtitle(witness, finding) + " | case-focused explanatory subgraph",
                anchor_ids=set(witness.anchor_ids),
                highlighted_edge_ids=explanatory_highlighted_edge_ids,
                highlighted_node_ids=highlighted_node_ids,
            )
            rendered_files.append(witness_path)
            rendered_files.append(overlay_path)
            rendered_files.append(explanatory_path)
            index_entries.append(
                RenderResult(
                    label=witness.witness_id,
                    relative_path=witness_path.name,
                    description=_witness_subtitle(witness, finding),
                )
            )
            index_entries.append(
                RenderResult(
                    label=f"{witness.witness_id} overlay",
                    relative_path=overlay_path.name,
                    description=_witness_subtitle(witness, finding) + "; full G_X with witness highlighted",
                )
            )
            index_entries.append(
                RenderResult(
                    label=f"{witness.witness_id} explanatory",
                    relative_path=explanatory_path.name,
                    description=_witness_subtitle(witness, finding) + "; case-focused explanatory subgraph",
                )
            )

    index_path = resolved_output_dir / "index.html"
    _write_index_html(
        index_path,
        title=f"{skill_label} / PyVis Views",
        entries=index_entries,
    )
    rendered_files.append(index_path)
    return rendered_files


def render_graph_html(
    graph: WireGraph,
    output_path: Path,
    *,
    title: str,
    subtitle: str = "",
    anchor_ids: set[str] | None = None,
    highlighted_edge_ids: set[str] | None = None,
    highlighted_node_ids: set[str] | None = None,
) -> None:
    """Render one graph view to a standalone HTML file."""
    graph = _complete_graph(graph)
    anchor_ids = anchor_ids or set()
    highlighted_edge_ids = highlighted_edge_ids or set()
    highlighted_node_ids = highlighted_node_ids or set()
    network = _build_pyvis_network()
    network.heading = html.escape(title)
    network.show_buttons(filter_=["physics", "layout", "interaction", "edges"])
    network.set_options(_NETWORK_OPTIONS)

    for node in graph.nodes:
        payload = _build_node_payload(
            node,
            anchor_ids=anchor_ids,
            highlighted_node_ids=highlighted_node_ids,
        )
        network.add_node(**payload)
    for edge in graph.edges:
        payload = _build_edge_payload(edge, highlighted_edge_ids=highlighted_edge_ids)
        network.add_edge(**payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        network.write_html(str(output_path), notebook=False, open_browser=False)
    except TypeError:
        network.save_graph(str(output_path))
    _inject_caption_block(output_path, title=title, subtitle=subtitle)


def _build_pyvis_network() -> Any:
    try:
        from pyvis.network import Network
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyVis is not installed in the current .venv. "
            "Install project dependencies before rendering."
        ) from exc
    return Network(
        height="920px",
        width="100%",
        directed=True,
        bgcolor="#f8fafc",
        font_color="#0f172a",
    )


def _build_node_payload(
    node: dict[str, Any],
    *,
    anchor_ids: set[str],
    highlighted_node_ids: set[str],
) -> dict[str, Any]:
    node_id = str(node.get("id", ""))
    kind = _node_kind(node)
    style = dict(_NODE_STYLE_BY_KIND.get(kind, _NODE_STYLE_BY_KIND["unknown"]))
    color = dict(style.pop("color"))
    is_anchor = node_id in anchor_ids
    is_highlighted = node_id in highlighted_node_ids
    if is_highlighted:
        highlight_border = "#0f172a"
        highlight_shadow = "rgba(15,23,42,0.18)"
        if kind == "clause":
            highlight_border = "#f97316"
            highlight_shadow = "rgba(249,115,22,0.28)"
        elif kind in {"event", "capability_event"}:
            highlight_border = "#2563eb"
            highlight_shadow = "rgba(37,99,235,0.22)"
        elif kind in {"resource", "resource_use", "path"}:
            highlight_border = "#0f766e"
            highlight_shadow = "rgba(15,118,110,0.22)"
        elif kind == "certificate":
            highlight_border = "#dc2626"
            highlight_shadow = "rgba(220,38,38,0.22)"
        if is_anchor and kind not in {"clause", "event", "capability_event", "resource", "resource_use", "path", "certificate"}:
            highlight_border = "#f97316"
            highlight_shadow = "rgba(249,115,22,0.28)"
        color["border"] = highlight_border
        style["borderWidth"] = 4 if is_anchor else 3
        style["shadow"] = {
            "enabled": True,
            "color": highlight_shadow,
            "size": 18,
        }
        style["font"] = {"size": 22, "color": "#020617", "face": "Helvetica"}
    else:
        color["background"] = _fade_hex_color(color["background"], 0.72)
        color["border"] = _fade_hex_color(color["border"], 0.6)
        style["font"] = {"size": 17, "color": "#475569", "face": "Helvetica"}

    payload = {
        "n_id": node_id,
        "label": _node_label(node),
        "title": _node_title(node),
        "color": color,
        **style,
    }
    return payload


def _build_edge_payload(
    edge: dict[str, Any],
    *,
    highlighted_edge_ids: set[str],
) -> dict[str, Any]:
    relation = str(edge.get("relation", ""))
    kind = str(edge.get("kind", ""))
    style = dict(_EDGE_STYLE_BY_RELATION.get(relation, _EDGE_STYLE_BY_KIND.get(kind, {})))
    edge_id = str(edge.get("edge_id", ""))
    is_highlighted = bool(edge_id and edge_id in highlighted_edge_ids)
    if is_highlighted:
        style["width"] = max(int(style.get("width", 2)), 7)
        style["opacity"] = 1.0
        style["color"] = {
            "color": style.get("color", "#0f172a"),
            "highlight": style.get("color", "#0f172a"),
            "hover": style.get("color", "#0f172a"),
            "opacity": 1.0,
        }
        style["font"] = {
            "size": 20,
            "color": "#0f172a",
            "strokeWidth": 0,
            "background": "rgba(255,255,255,0.96)",
            "align": "middle",
        }
    else:
        base_color = style.get("color", "#94a3b8")
        style["color"] = {
            "color": base_color,
            "highlight": base_color,
            "hover": base_color,
            "opacity": float(style.pop("opacity", 0.34)),
        }
        style["font"] = {
            "size": 15,
            "color": "#64748b",
            "strokeWidth": 0,
            "background": "rgba(255,255,255,0.9)",
            "align": "middle",
        }

    label = relation or kind
    title = _edge_title(edge)
    return {
        "source": str(edge.get("source", "")),
        "to": str(edge.get("target", "")),
        "label": label,
        "title": title,
        **style,
    }


def _node_label(node: dict[str, Any]) -> str:
    kind = _node_kind(node)
    if kind == "clause":
        capability = _truncate(_display_text(node.get("capability", "")).replace("_", " "), 20)
        operator = _display_text(node.get("operator", ""))
        target = _truncate(_display_text(node.get("target", "")), 26)
        evidence = _truncate(_primary_evidence_text(node), 32)
        if target:
            return f"Clause\n{capability} {operator}\n{target}\n{evidence}"
        return f"Clause\n{capability} {operator}\n{evidence}"
    if kind == "constraint":
        constraint_type = _truncate(str(node.get("constraint_type", "")).replace("_", " "), 18)
        value = _truncate(str(node.get("value", "")), 34)
        return f"Constraint\n{constraint_type}\n{value}"
    if kind in {"event", "capability_event"}:
        role = _display_text(node.get("path_role", "")).strip()
        role_prefix = "Source event" if role == "source" else "Sink event" if role == "sink" else "Event"
        capability = _truncate(_display_text(node.get("capability", "")).replace("_", " "), 20)
        location = _truncate(_display_text(node.get("location", node.get("unit_path", ""))), 34)
        detail = _truncate(_display_text(node.get("detail", "")), 24)
        if location and detail:
            return f"{role_prefix}\n{capability}\n{location}\n{detail}"
        if location:
            return f"{role_prefix}\n{capability}\n{location}"
        unit_id = _truncate(_display_text(node.get("unit_id", "")), 18)
        return f"{role_prefix}\n{capability}\n{unit_id}"
    if kind in {"resource", "resource_use"}:
        resource_type = _truncate(str(node.get("resource_type", "")).replace("_", " "), 18)
        value = _truncate(str(node.get("value", "")), 34)
        location = _truncate(str(node.get("location", "")), 28)
        if location:
            return f"Resource\n{resource_type}\n{value}\n{location}"
        return f"Resource\n{resource_type}\n{value}"
    if kind == "data_object":
        object_kind = _truncate(str(node.get("object_kind", "")), 18)
        label = _truncate(str(node.get("label", "")), 26)
        return f"Data object\n{object_kind}\n{label}"
    if kind == "operation":
        operation_type = _truncate(str(node.get("operation_type", "")), 18)
        summary = _truncate(str(node.get("summary", "")), 28)
        return f"Operation\n{operation_type}\n{summary}"
    if kind == "location":
        file_path = _truncate(str(node.get("file_path", "")), 24)
        line = str(node.get("line", ""))
        return f"Location\n{file_path}:{line}"
    if kind == "source":
        label = _truncate(str(node.get("label", "")), 24)
        return f"Source\n{label}"
    if kind == "sink":
        label = _truncate(str(node.get("label", "")), 24)
        return f"Sink\n{label}"
    if kind == "path":
        source_label = _truncate(_display_text(node.get("source_label", "")).replace("_", " "), 16)
        sink_label = _truncate(_display_text(node.get("sink_label", "")).replace("_", " "), 16)
        path_kind = _truncate(_display_text(node.get("path_kind", "")).replace("_", " "), 22)
        source_location = _truncate(_display_text(node.get("source_location", "")), 24)
        sink_location = _truncate(_display_text(node.get("sink_location", "")), 24)
        if source_location and sink_location:
            return f"Path\n{source_label} → {sink_label}\n{source_location}\n{sink_location}\n{path_kind}"
        return f"Path\n{source_label} → {sink_label}\n{path_kind}"
    if kind == "judgment":
        judgment_kind = _truncate(str(node.get("kind", "")).replace("_", " ").lower(), 22)
        result = str(node.get("result", "")).lower()
        return f"Judgment\n{judgment_kind}\n{result}"
    if kind == "certificate":
        certificate_kind = _truncate(str(node.get("kind", "")).replace("_", " ").lower(), 28)
        notes = _truncate(str(node.get("notes", "")), 24)
        if notes:
            return f"Certificate\n{certificate_kind}\n{notes}"
        return f"Certificate\n{certificate_kind}"
    if kind == "code_unit":
        file_path = _truncate(_display_text(node.get('file_path', node.get('id', ''))), 28)
        language = _truncate(_display_text(node.get('language', '')), 14)
        if language:
            return f"Code unit\n{file_path}\n{language}"
        return f"Code unit\n{file_path}"
    if kind == "capability":
        return f"Capability\n{_truncate(str(node.get('value', '')), 24)}"
    if kind == "evidence_span":
        return f"Evidence\n{_truncate(str(node.get('doc_id', '')), 20)}"
    if kind == "step":
        return f"Step\n{_truncate(str(node.get('id', '')), 24)}"
    return f"{kind.title()}\n{_truncate(str(node.get('id', '')), 24)}"


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.strip().lower() == "none":
        return ""
    return str(value)


def _primary_evidence_text(node: dict[str, Any]) -> str:
    evidence_spans = node.get("evidence_spans")
    if isinstance(evidence_spans, list) and evidence_spans:
        first = evidence_spans[0]
        if isinstance(first, dict):
            return _display_text(first.get("text", ""))
    evidence = node.get("evidence")
    if isinstance(evidence, dict):
        return _display_text(evidence.get("text", ""))
    return ""


def _node_title(node: dict[str, Any]) -> str:
    return html.escape(json.dumps(node, indent=2, ensure_ascii=False))


def _edge_title(edge: dict[str, Any]) -> str:
    return html.escape(json.dumps(edge, indent=2, ensure_ascii=False))


def _inject_caption_block(output_path: Path, *, title: str, subtitle: str) -> None:
    content = output_path.read_text(encoding="utf-8")
    caption = (
        "<div style=\"max-width:1200px;margin:0 auto 12px auto;"
        "padding:14px 18px;border:1px solid #e5e7eb;border-radius:14px;"
        "background:#ffffff;font-family:Helvetica,Arial,sans-serif;\">"
        f"<div style=\"font-size:24px;font-weight:700;color:#0f172a;\">{html.escape(title)}</div>"
        f"<div style=\"margin-top:6px;font-size:14px;color:#475569;\">{html.escape(subtitle)}</div>"
        "<div style=\"margin-top:10px;font-size:12px;color:#64748b;display:flex;gap:18px;flex-wrap:wrap;\">"
        "<span><strong style=\"color:#111827;\">Bold relation edge</strong> = primary evidence link</span>"
        "<span><strong style=\"color:#111827;\">Dashed edge</strong> = proof dependency</span>"
        "<span><strong style=\"color:#111827;\">Blue border</strong> = observed behavior</span>"
        "<span><strong style=\"color:#111827;\">Orange border</strong> = contract clause</span>"
        "<span><strong style=\"color:#111827;\">Red diamond</strong> = certificate</span>"
        "</div>"
        "</div>"
    )
    if "<body>" in content:
        content = content.replace("<body>", f"<body>\n{caption}\n", 1)
    output_path.write_text(content, encoding="utf-8")


def _write_index_html(path: Path, *, title: str, entries: list[RenderResult]) -> None:
    items = "\n".join(
        (
            "<li style=\"margin:0 0 14px 0;\">"
            f"<a href=\"{html.escape(entry.relative_path)}\" "
            "style=\"font-size:18px;font-weight:700;color:#0f172a;text-decoration:none;\">"
            f"{html.escape(entry.label)}</a>"
            f"<div style=\"margin-top:4px;color:#475569;\">{html.escape(entry.description)}</div>"
            "</li>"
        )
        for entry in entries
    )
    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
</head>
<body style="margin:0;background:#f8fafc;font-family:Helvetica,Arial,sans-serif;color:#0f172a;">
  <main style="max-width:960px;margin:40px auto;padding:0 24px;">
    <h1 style="margin:0 0 10px 0;">{html.escape(title)}</h1>
    <p style="margin:0 0 24px 0;color:#475569;">
      PyVis views for SkillRecon graph artifacts.
      Open a witness page for the most paper-friendly view.
    </p>
    <section style="padding:24px;border:1px solid #e5e7eb;border-radius:18px;background:#ffffff;">
      <ul style="list-style:none;padding:0;margin:0;">
        {items or '<li>No graph views were generated.</li>'}
      </ul>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _witness_subtitle(witness: Witness, finding: Finding | None) -> str:
    details = [
        f"finding={witness.finding_id}",
        f"anchors={len(witness.anchor_ids)}",
        f"facts={len(witness.fact_node_ids)}",
        f"judgments={len(witness.judgment_ids)}",
        f"certificates={len(witness.certificate_ids)}",
        f"projection_edges={len(witness.projection_edge_ids)}",
    ]
    if finding is not None:
        details.insert(0, f"type={finding.finding_type.value}")
    return " | ".join(details)


def _complete_graph(graph: WireGraph) -> WireGraph:
    """Fill in missing edge endpoints so strict renderers can accept the graph."""
    node_map = graph.node_map()
    completed_nodes = dict(node_map)
    completed_edges: list[dict[str, Any]] = []

    for edge in graph.edges:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if not source or not target:
            continue
        normalized_edge = dict(edge)
        normalized_edge["source"] = source
        normalized_edge["target"] = target
        completed_edges.append(normalized_edge)
        for node_id in (source, target):
            if node_id not in completed_nodes:
                completed_nodes[node_id] = _placeholder_node(node_id)

    return WireGraph(
        nodes=sorted(completed_nodes.values(), key=_node_sort_key),
        edges=completed_edges,
    )


def _node_sort_key(node: dict[str, Any]) -> tuple[int, str]:
    kind = _node_kind(node)
    level = int(_NODE_STYLE_BY_KIND.get(kind, _NODE_STYLE_BY_KIND["unknown"]).get("level", 99))
    return (level, str(node.get("id", "")))


def _node_kind(node: dict[str, Any]) -> str:
    raw_kind = node.get("type", node.get("kind", "unknown"))
    return str(raw_kind or "unknown")


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)] + "..."


def _fade_hex_color(color: str, factor: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return f"#{color}"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    faded = (
        int(255 - (255 - red) * factor),
        int(255 - (255 - green) * factor),
        int(255 - (255 - blue) * factor),
    )
    return "#{:02x}{:02x}{:02x}".format(*faded)


def _collect_witness_edge_ids(graph: WireGraph, highlighted_node_ids: set[str]) -> set[str]:
    edge_ids: set[str] = set()
    for edge in graph.edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        edge_id = str(edge.get("edge_id", ""))
        if source in highlighted_node_ids and target in highlighted_node_ids and edge_id:
            edge_ids.add(edge_id)
    return edge_ids


def _expand_witness_context(graph: WireGraph, core_node_ids: set[str], *, limit: int = 8) -> set[str]:
    node_map = graph.node_map()
    candidates: list[tuple[int, str]] = []
    preferred_kinds = {
        "path": 0,
        "event": 1,
        "capability_event": 1,
        "resource": 2,
        "resource_use": 2,
        "data_object": 2,
        "code_unit": 3,
        "location": 4,
        "clause": 5,
        "constraint": 6,
    }
    seen_node_ids: set[str] = set()
    seen_signatures: set[tuple[str, str, str, str]] = set()
    for edge in graph.edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        if source in core_node_ids and target not in core_node_ids:
            other = target
        elif target in core_node_ids and source not in core_node_ids:
            other = source
        else:
            continue
        if other in seen_node_ids:
            continue
        node = node_map.get(other, {})
        kind = _node_kind(node)
        if kind not in preferred_kinds:
            continue
        signature = _context_signature(node, kind)
        if signature in seen_signatures:
            continue
        seen_node_ids.add(other)
        seen_signatures.add(signature)
        candidates.append((preferred_kinds[kind], other))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return {node_id for _, node_id in candidates[:limit]}


def _non_empty_node_ids(nodes: list[dict[str, Any]]) -> set[str]:
    return {str(node.get("id", "")) for node in nodes if str(node.get("id", ""))}


def _load_missing_nodes(artifact_dir: Path, missing_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not missing_ids:
        return {}
    enriched = enrich_wire_graph(WireGraph(nodes=[], edges=[]), artifact_dir)
    enriched_map = enriched.node_map()
    return {node_id: enriched_map[node_id] for node_id in missing_ids if node_id in enriched_map}


def _context_signature(node: dict[str, Any], kind: str) -> tuple[str, str, str, str]:
    if kind == "clause":
        return (
            kind,
            str(node.get("capability", "")),
            str(node.get("operator", "")),
            str(node.get("target", "")),
        )
    if kind in {"event", "capability_event"}:
        return (
            kind,
            str(node.get("capability", "")),
            str(node.get("location", "")),
            str(node.get("unit_id", "")),
        )
    if kind in {"resource", "resource_use", "data_object", "path", "code_unit", "location", "constraint"}:
        return (
            kind,
            str(node.get("resource_type", node.get("constraint_type", node.get("path_kind", "")))),
            str(node.get("value", node.get("location", node.get("file_path", "")))),
            str(node.get("unit_id", node.get("id", ""))),
        )
    return (kind, str(node.get("id", "")), "", "")


def _placeholder_node(node_id: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": _infer_placeholder_kind(node_id),
        "missing": True,
    }


def _infer_placeholder_kind(node_id: str) -> str:
    if node_id.startswith("cst"):
        return "constraint"
    if node_id.startswith("cert"):
        return "certificate"
    if node_id.startswith("judge"):
        return "judgment"
    if node_id.startswith("path-") or node_id.startswith("p"):
        return "path"
    if node_id.startswith("e"):
        return "event"
    if node_id.startswith("r"):
        return "resource"
    if node_id.startswith("c"):
        return "clause"
    return "unknown"
