# Experiment inputs

Each annotation or prediction file is JSONL: one JSON object per line. Experiment configurations are JSON objects. Paths in an experiment configuration are relative to that configuration file.

| Schema | Input |
|---|---|
| `skill-label.schema.json` | Skill identity, `gold` label, optional clause/edge labels, grouping metadata |
| `prediction.schema.json` | System identity, skill identity, binary/typed prediction and metadata |
| `experiment.schema.json` | Annotation paths, system output paths, model/parameter comparisons and perturbation groups |
| `rq1-clause.schema.json` | Located clause ID with its operator and POLICY/KNOWLEDGE role |
| `rq1-object.schema.json` | Complete reference keys for one intermediate-object group |
| `perturbation.schema.json` | Located document edit and optional variant label |
| `witness-grounding.schema.json` | Category and grounding judgment |
| `perturbation-mechanism.schema.json` | Clause/constraint/event IDs and mechanism judgments |

## Skill labels and group metadata

`gold.label` is `violation`, `exposure-only`, or `benign`. Category aliases accepted by evaluation are:

| Category | Subtype values |
|---|---|
| V1 | `unsupported_behavior`, `contradicted_behavior`, `undeclared_high_impact_capability` |
| V2 | `scope_violation` |
| V3 | `unjustified_composition`, `unjustified_dangerous_composition` |

`gold.violation_subtypes` accepts overlapping categories. `risk_stratum` selects `high`, `medium`, or `low`. `metadata.platform` selects `claude_code` or `opencode`. Optional metadata includes `documentation_words`, `language_profile`, `benign_high_privilege`, and `false_negative_reason`.

Language profiles are `Python`, `JS/TS`, `Bash`, `Py+Bash`, `JS/TS+Bash`, `Multi>=3`, and `Other`.

## Evaluation configuration

`gold` names the skill-label file. `main_system` selects the reference system. Each `systems` entry supplies one of:

- `artifacts`: analysis output root, with `<skill_id>/findings.json` and the analysis records.
- `reports`: normalized report root, with `<skill_id>/report.json`.
- `predictions`: normalized prediction JSONL; `system_id` selects a method from a shared file.

`native_types` selects type-specific evaluation for a system. Prediction types are stored in `subtype` or `metadata.violation_subtypes`; the system's binary decision is `main_label`.

Optional inputs:

| Field | Purpose |
|---|---|
| `rq1_clauses` | Operator/role annotations keyed by skill and clause ID |
| `rq1_objects` | Intermediate-object and effective-support reference keys |
| `witness_grounding` | Grounding judgments |
| `backbone_systems` | System names for model comparisons |
| `sensitivity` | Rows of `parameter`, `value`, `system` for n, θ, k_c and witness bounds |
| `robustness` | Named variant groups, their `gold` path and system output source |
| `robustness[].mechanism_annotations` | Variant clause/constraint/event judgments |
| `error_annotations` | Error-attribution rows with `reason` |

RQ1 object rows specify `artifact`, `key_fields`, and `gold_keys`. Object groups include clauses, events, resources, bridges, paths, edges, certificates, and effective `supports` relations. Use `relation` to select an edge relation. Keys use the actual fields in the corresponding analysis record.

## Ablations and variants

Ablation output roots use the system names `ablation_no_iccm`, `ablation_no_scope_constraints`, `ablation_no_composition_analysis`, and `ablation_no_authorization_guard`; add them to `systems` as artifact sources.

A perturbation row identifies `skill_id`, `kind`, `file`, `original`, and `replacement`. `start_byte` locates a repeated UTF-8 source span. A supplied `gold` object becomes the variant label. Supported kinds are `missing_authorization`, `stale_resource`, `ambiguous_scope`, and `conflicting_allow_prohibit`.

Mechanism annotations identify the variant's `clause_ids`, `constraint_ids`, and `event_ids`. Their `gold` object can contain `guard_downgrade`, `scope_abstain`, and `prohibition_block` judgments.
