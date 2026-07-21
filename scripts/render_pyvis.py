#!/usr/bin/env python3
"""Render SkillRecon graph artifacts into standalone PyVis HTML pages."""

from __future__ import annotations

import argparse
from pathlib import Path

from skillrecon.visualize import render_artifact_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render G_X and witness subgraphs as standalone PyVis HTML pages"
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="Directory containing g_x.json and witnesses.json",
    )
    parser.add_argument("--skill", help="Skill identifier for page titles")
    parser.add_argument(
        "--output-dir",
        help="Output directory for HTML pages (defaults to <artifact-dir>/viz)",
    )
    parser.add_argument(
        "--witness-id",
        action="append",
        default=[],
        help="Render only specific witness ids; may be passed multiple times",
    )
    parser.add_argument(
        "--no-full-graph",
        action="store_true",
        help="Skip the full G_X page and render witness pages only",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None
    rendered_files = render_artifact_dir(
        artifact_dir=artifact_dir,
        skill_id=args.skill,
        output_dir=output_dir,
        witness_ids=set(args.witness_id),
        render_full_graph=not args.no_full_graph,
    )

    print(f"Rendered {len(rendered_files)} HTML files:")
    for path in rendered_files:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
