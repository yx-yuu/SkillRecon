#!/usr/bin/env python3
"""Build dry-run command plans for OpenAI-compatible LLM replacements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_TASKS = ("artifacts", "bundle")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-model configs and command plans for future LLM replacement "
            "experiments. The script does not call any model endpoint."
        )
    )
    parser.add_argument(
        "--models-json",
        required=True,
        help=(
            "JSON object with a 'models' list, or a JSON list. Each model uses "
            "OpenAI-compatible fields: id, base_url, model, api_key_env, "
            "temperature, max_tokens, structured_output_mode."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="derived/experiments/llm_replacement_plan",
        help="Directory for generated config files and plan.json",
    )
    parser.add_argument(
        "--paper-dataset",
        default="data/evaluation/skill_paper500_dataset",
        help="Paper dataset path used in planned commands",
    )
    parser.add_argument(
        "--source-root",
        default="derived/paper500_source_links",
        help="Source root used by planned ablation/bundle commands",
    )
    parser.add_argument(
        "--artifact-output-root",
        default="derived/llm_replacement/paper500",
        help="Root for planned per-model paper artifact outputs",
    )
    parser.add_argument(
        "--experiment-output-root",
        default="derived/llm_replacement/experiments",
        help="Root for planned per-model experiment bundle outputs",
    )
    parser.add_argument(
        "--external-predictions",
        help=(
            "Normalized external baseline predictions for planned experiment/stat "
            "commands. Defaults to paper_method_baselines.jsonl when present, "
            "otherwise OpenClaw only."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=("artifacts", "bundle", "experiments", "stats", "ablation"),
        help=(
            "Command family to include. Defaults to artifacts and bundle. "
            "May be repeated."
        ),
    )
    parser.add_argument("--limit", type=int, help="Optional planned artifact-run limit")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    models = _load_model_specs(Path(args.models_json))
    output_dir = Path(args.output_dir)
    config_dir = output_dir / "llm_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    tasks = tuple(args.task or DEFAULT_TASKS)
    planned = []
    for model_spec in models:
        config = _normalize_model_spec(model_spec)
        config_path = config_dir / f"{config['id']}.json"
        config_path.write_text(
            json.dumps(_config_payload(config), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        planned.append(
            {
                "id": config["id"],
                "llm_config": str(config_path),
                "commands": _commands_for_model(config, config_path, tasks, args),
            }
        )

    manifest = {
        "models": planned,
        "tasks": list(tasks),
        "status": "ok",
        "note": "dry-run plan only; no LLM endpoints were called",
    }
    (output_dir / "plan.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _load_model_specs(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("models")
    if not isinstance(payload, list):
        raise ValueError("--models-json must contain a list or a {'models': [...]} object")
    specs = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each model spec must be a JSON object")
        specs.append(item)
    return specs


def _normalize_model_spec(spec: dict[str, object]) -> dict[str, object]:
    base_url = _required_string(spec, "base_url")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url must be a full HTTP(S) URL: {base_url}")
    model_name = _required_string(spec, "model")
    model_id = str(spec.get("id") or _slug(model_name))
    api_key_env = str(spec.get("api_key_env") or "SKILLRECON_API_KEY")
    structured_output_mode = str(spec.get("structured_output_mode") or "json_prompt")
    if structured_output_mode not in {"json_schema", "json_prompt"}:
        raise ValueError(
            "structured_output_mode must be json_schema or json_prompt "
            f"for model {model_id}"
        )
    return {
        "id": _slug(model_id),
        "base_url": base_url,
        "model": model_name,
        "api_key_env": api_key_env,
        "temperature": float(spec.get("temperature") or 0.0),
        "max_tokens": int(spec.get("max_tokens") or 8192),
        "structured_output_mode": structured_output_mode,
    }


def _commands_for_model(
    config: dict[str, object],
    config_path: Path,
    tasks: tuple[str, ...],
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    commands = []
    artifact_root = Path(args.artifact_output_root) / str(config["id"])
    experiment_dir = Path(args.experiment_output_root) / str(config["id"])
    paper_dataset = Path(args.paper_dataset)
    gold_labels = paper_dataset / "gold_labels.jsonl"
    external_predictions = _default_external_predictions(args.external_predictions, paper_dataset)
    skills_file = paper_dataset / "all_skills.txt"
    if "artifacts" in tasks:
        command = [
            "python3",
            "scripts/run_paper_artifacts.py",
            "--paper-dataset",
            args.paper_dataset,
            "--output-dir",
            str(artifact_root),
            "--llm-config",
            str(config_path),
            "--base-url",
            str(config["base_url"]),
            "--model",
            str(config["model"]),
            "--api-key-env",
            str(config["api_key_env"]),
            "--temperature",
            str(config["temperature"]),
            "--max-tokens",
            str(config["max_tokens"]),
            "--max-workers",
            str(args.max_workers),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        commands.append({"task": "artifacts", "command": command})
    if "ablation" in tasks:
        commands.append(
            {
                "task": "ablation",
                "command": [
                    "python3",
                    "scripts/build_ablation_artifacts.py",
                    "--skills-file",
                    str(skills_file),
                    "--data-root",
                    args.source_root,
                    "--artifact-root",
                    str(artifact_root),
                    "--output-root",
                    str(Path(args.artifact_output_root) / "ablations" / str(config["id"])),
                    "--llm-config",
                    str(config_path),
                    "--base-url",
                    str(config["base_url"]),
                    "--model",
                    str(config["model"]),
                    "--api-key-env",
                    str(config["api_key_env"]),
                    "--temperature",
                    str(config["temperature"]),
                    "--max-tokens",
                    str(config["max_tokens"]),
                ],
            }
        )
    if "experiments" in tasks:
        commands.append(
            {
                "task": "experiments",
                "command": [
                    "python3",
                    "scripts/run_experiments.py",
                    "--paper-dataset",
                    args.paper_dataset,
                    "--artifact-root",
                    str(artifact_root),
                    "--external-predictions",
                    str(external_predictions),
                    "--output-dir",
                    str(experiment_dir),
                    "--llm-config",
                    str(config_path),
                    "--llm-base-url",
                    str(config["base_url"]),
                    "--llm-model",
                    str(config["model"]),
                    "--llm-api-key-env",
                    str(config["api_key_env"]),
                    "--llm-temperature",
                    str(config["temperature"]),
                    "--llm-max-tokens",
                    str(config["max_tokens"]),
                ],
            }
        )
    if "stats" in tasks:
        commands.append(
            {
                "task": "stats",
                "command": [
                    "python3",
                    "scripts/compute_rq2_statistics.py",
                    "--gold-labels",
                    str(gold_labels),
                    "--external-predictions",
                    str(external_predictions),
                    "--artifact-root",
                    str(artifact_root),
                    "--output-dir",
                    str(experiment_dir),
                    "--llm-config",
                    str(config_path),
                    "--llm-base-url",
                    str(config["base_url"]),
                    "--llm-model",
                    str(config["model"]),
                    "--llm-api-key-env",
                    str(config["api_key_env"]),
                    "--llm-temperature",
                    str(config["temperature"]),
                    "--llm-max-tokens",
                    str(config["max_tokens"]),
                ],
            }
        )
    if "bundle" in tasks:
        commands.append(
            {
                "task": "bundle",
                "command": [
                    "python3",
                    "scripts/run_paper_experiment_bundle.py",
                    "--paper-dataset",
                    args.paper_dataset,
                    "--artifact-root",
                    str(artifact_root),
                    "--source-root",
                    args.source_root,
                    "--external-predictions",
                    str(external_predictions),
                    "--output-dir",
                    str(experiment_dir),
                    "--llm-config",
                    str(config_path),
                    "--llm-base-url",
                    str(config["base_url"]),
                    "--llm-model",
                    str(config["model"]),
                    "--llm-api-key-env",
                    str(config["api_key_env"]),
                    "--llm-temperature",
                    str(config["temperature"]),
                    "--llm-max-tokens",
                    str(config["max_tokens"]),
                    "--dry-run",
                ],
            }
        )
    return commands


def _config_payload(config: dict[str, object]) -> dict[str, object]:
    return {
        "base_url": config["base_url"],
        "model": config["model"],
        "api_key_env": config["api_key_env"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "structured_output_mode": config["structured_output_mode"],
    }


def _default_external_predictions(value: str | None, paper_dataset: Path) -> Path:
    if value:
        return Path(value)
    merged = paper_dataset / "baseline_predictions" / "paper_method_baselines.jsonl"
    if merged.is_file():
        return merged
    return paper_dataset / "baseline_predictions" / "openclaw.jsonl"


def _required_string(spec: dict[str, object], key: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model spec missing non-empty {key}")
    return value.strip()


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "model"


if __name__ == "__main__":
    main()
