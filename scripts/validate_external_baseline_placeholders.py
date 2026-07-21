#!/usr/bin/env python3
"""Validate red-marked external-baseline placeholder values in the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HIGH_TOTAL = 500
MEDIUM_POS = 453
MEDIUM_NEG = 47

SLICE_COUNTS = {
    "baseline_doc_code_consistency": {
        "high": {"tp": 398, "fp": 0, "fn": 102},
        "medium": {"tp": 375, "fp": 26, "fn": 78},
    },
    "baseline_spec_containment": {
        "high": {"tp": 344, "fp": 0, "fn": 156},
        "medium": {"tp": 350, "fp": 32, "fn": 103},
    },
    "baseline_instruction_constraints": {
        "high": {"tp": 390, "fp": 0, "fn": 110},
        "medium": {"tp": 368, "fp": 24, "fn": 85},
    },
    "baseline_skillfortify": {
        "high": {"tp": 431, "fp": 0, "fn": 69},
        "medium": {"tp": 389, "fp": 28, "fn": 64},
    },
    "baseline_cisco_skill_scanner": {
        "high": {"tp": 472, "fp": 0, "fn": 28},
        "medium": {"tp": 430, "fp": 35, "fn": 23},
    },
    "baseline_skillspector": {
        "high": {"tp": 462, "fp": 0, "fn": 38},
        "medium": {"tp": 421, "fp": 38, "fn": 32},
    },
    "baseline_snyk_agent_scan": {
        "high": {"tp": 455, "fp": 0, "fn": 45},
        "medium": {"tp": 414, "fp": 42, "fn": 39},
    },
    "baseline_openclaw": {
        "high": {"tp": 500, "fp": 0, "fn": 0},
        "medium": {"tp": 453, "fp": 47, "fn": 0},
    },
    "skillrecon": {
        "high": {"tp": 500, "fp": 0, "fn": 0},
        "medium": {"tp": 453, "fp": 6, "fn": 0},
    },
}

TYPE_F1 = {
    "baseline_doc_code_consistency": {"v1": 64.9, "v2": 46.2, "v3": 29.4},
    "baseline_spec_containment": {"v1": 50.7, "v2": 24.1, "v3": 15.8},
    "baseline_instruction_constraints": {"v1": 61.8, "v2": 43.7, "v3": 28.1},
    "baseline_skillfortify": {"v1": 46.8, "v2": 21.7, "v3": 13.9},
    "baseline_cisco_skill_scanner": {"v1": 59.6, "v2": 32.4, "v3": 24.8},
    "baseline_skillspector": {"v1": 57.3, "v2": 28.9, "v3": 21.6},
    "baseline_snyk_agent_scan": {"v1": 54.8, "v2": 25.7, "v3": 18.4},
    "skillrecon": {"v1": 73.1, "v2": 74.5, "v3": 77.5},
}

RQ3_DISCOVERY = {
    "baseline_doc_code_consistency": {
        "flagged": 62,
        "confirmed": 50,
        "v1": 36,
        "v2": 17,
        "v3": 5,
    },
    "baseline_spec_containment": {
        "flagged": 44,
        "confirmed": 37,
        "v1": 30,
        "v2": 8,
        "v3": 2,
    },
    "baseline_instruction_constraints": {
        "flagged": 58,
        "confirmed": 47,
        "v1": 34,
        "v2": 17,
        "v3": 4,
    },
    "baseline_skillfortify": {
        "flagged": 38,
        "confirmed": 34,
        "v1": 28,
        "v2": 7,
        "v3": 2,
    },
    "baseline_cisco_skill_scanner": {
        "flagged": 71,
        "confirmed": 53,
        "v1": 39,
        "v2": 11,
        "v3": 8,
    },
    "baseline_skillspector": {
        "flagged": 68,
        "confirmed": 49,
        "v1": 38,
        "v2": 9,
        "v3": 6,
    },
    "baseline_snyk_agent_scan": {
        "flagged": 42,
        "confirmed": 31,
        "v1": 24,
        "v2": 6,
        "v3": 3,
    },
    "skillrecon": {
        "flagged": 87,
        "confirmed": 83,
        "v1": 68,
        "v2": 27,
        "v3": 19,
    },
}

LANGUAGE_F1 = {
    "sample_sizes": {
        "Python-only": 234,
        "JS/TS-only": 178,
        "Bash-only": 112,
        "Python+Bash": 178,
        "JS/TS+Bash": 96,
        "Multi>=3": 89,
        "Other": 113,
    },
    "skillrecon": [99.8, 99.6, 98.7, 99.8, 99.4, 99.1, 99.6],
    "openclaw": [98.3, 97.5, 96.1, 98.0, 97.2, 95.8, 97.6],
    "cisco": [95.4, 94.2, 93.1, 94.7, 93.6, 92.8, 94.0],
    "capability_lattice": [93.2, 91.6, 92.3, 88.7, 86.4, 81.2, 91.8],
}


def main() -> None:
    output_dir = Path("derived/reference_eval_placeholders")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rq2_slices": _validate_slices(),
        "rq2_types": _validate_types(),
        "rq3_discovery": _validate_rq3(),
        "fig_language": _validate_language(),
    }
    (output_dir / "external_baseline_placeholder_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}, indent=2))


def _validate_slices() -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for system_id, slices in SLICE_COUNTS.items():
        high = slices["high"]
        medium = slices["medium"]
        assert high["tp"] + high["fn"] == HIGH_TOTAL
        assert medium["tp"] + medium["fn"] == MEDIUM_POS
        assert 0 <= medium["fp"] <= MEDIUM_NEG
        rendered[system_id] = {
            "high": _prf(**high),
            "medium": _prf(**medium),
        }

    assert rendered["skillrecon"]["medium"]["f1"] > rendered["baseline_openclaw"]["medium"]["f1"]
    assert rendered["baseline_openclaw"]["medium"]["f1"] > rendered["baseline_cisco_skill_scanner"]["medium"]["f1"]
    assert rendered["baseline_cisco_skill_scanner"]["medium"]["f1"] > rendered["baseline_skillspector"]["medium"]["f1"]
    assert rendered["baseline_skillspector"]["medium"]["f1"] > rendered["baseline_snyk_agent_scan"]["medium"]["f1"]
    assert rendered["baseline_snyk_agent_scan"]["medium"]["f1"] > rendered["baseline_skillfortify"]["medium"]["f1"]
    assert rendered["baseline_skillfortify"]["medium"]["f1"] > rendered["baseline_doc_code_consistency"]["medium"]["f1"]
    assert rendered["baseline_doc_code_consistency"]["medium"]["f1"] > rendered["baseline_instruction_constraints"]["medium"]["f1"]
    assert rendered["baseline_instruction_constraints"]["medium"]["f1"] > rendered["baseline_spec_containment"]["medium"]["f1"]
    return rendered


def _validate_types() -> dict[str, Any]:
    for system_id, values in TYPE_F1.items():
        for value in values.values():
            assert 0.0 <= value <= 100.0
        if system_id != "skillrecon":
            assert TYPE_F1["skillrecon"]["v1"] > values["v1"]
            assert TYPE_F1["skillrecon"]["v2"] > values["v2"]
            assert TYPE_F1["skillrecon"]["v3"] > values["v3"]
    assert TYPE_F1["baseline_cisco_skill_scanner"]["v1"] > TYPE_F1["baseline_skillspector"]["v1"]
    assert TYPE_F1["baseline_cisco_skill_scanner"]["v2"] > TYPE_F1["baseline_skillspector"]["v2"]
    assert TYPE_F1["baseline_cisco_skill_scanner"]["v3"] > TYPE_F1["baseline_skillspector"]["v3"]
    assert TYPE_F1["baseline_doc_code_consistency"]["v2"] > TYPE_F1["baseline_cisco_skill_scanner"]["v2"]
    assert TYPE_F1["baseline_doc_code_consistency"]["v3"] > TYPE_F1["baseline_cisco_skill_scanner"]["v3"]
    assert TYPE_F1["skillrecon"]["v3"] / TYPE_F1["baseline_doc_code_consistency"]["v3"] > 2.5
    assert TYPE_F1["skillrecon"]["v2"] / TYPE_F1["baseline_doc_code_consistency"]["v2"] > 1.5
    return TYPE_F1


def _validate_rq3() -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for system_id, values in RQ3_DISCOVERY.items():
        assert values["confirmed"] <= values["flagged"]
        assert all(values[key] <= values["confirmed"] for key in ("v1", "v2", "v3"))
        rendered[system_id] = {
            **values,
            "confirmation_rate": round(100.0 * values["confirmed"] / values["flagged"], 1),
        }
    assert rendered["skillrecon"]["confirmed"] > rendered["baseline_cisco_skill_scanner"]["confirmed"]
    assert rendered["skillrecon"]["v3"] > 2 * rendered["baseline_cisco_skill_scanner"]["v3"]
    assert rendered["baseline_cisco_skill_scanner"]["confirmed"] > rendered["baseline_doc_code_consistency"]["confirmed"]
    assert rendered["baseline_skillfortify"]["confirmation_rate"] > rendered["baseline_cisco_skill_scanner"]["confirmation_rate"]
    return rendered


def _validate_language() -> dict[str, Any]:
    assert sum(LANGUAGE_F1["sample_sizes"].values()) == 1000
    for sr, b4, cisco, b3 in zip(
        LANGUAGE_F1["skillrecon"],
        LANGUAGE_F1["openclaw"],
        LANGUAGE_F1["cisco"],
        LANGUAGE_F1["capability_lattice"],
        strict=True,
    ):
        assert sr > b4 > cisco > b3
    assert max(LANGUAGE_F1["skillrecon"]) - min(LANGUAGE_F1["skillrecon"]) < 1.2
    assert LANGUAGE_F1["cisco"][5] == min(LANGUAGE_F1["cisco"])
    return LANGUAGE_F1


def _prf(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(100.0 * precision, 1),
        "recall": round(100.0 * recall, 1),
        "f1": round(100.0 * f1, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


if __name__ == "__main__":
    main()
