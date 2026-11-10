from __future__ import annotations

from collections import defaultdict

from transfer_eval.statistics.confidence import summarize_seed_values


NOT_IMPLEMENTED_ROWS = [
    {"method_id": "domain_transfer", "transfer_axis": "simulation_domain_generalization", "evidence_status": "not_implemented"},
    {"method_id": "physical_capability_heterogeneity", "transfer_axis": "physical_capability_heterogeneity", "evidence_status": "not_implemented"},
]

NO_LOCAL_RUN_ROWS = [
    {"method_id": "maptf", "category": "map_tf", "transfer_axis": "intra_population_transfer", "evidence_status": "implemented",
     "native_simulator": "MPE/Pacman", "missing_data_flags": ["no current local run artifacts discovered"]},
    {"method_id": "portal", "category": "portal", "transfer_axis": "task_transfer", "evidence_status": "partial",
     "native_simulator": "SMAC/StarCraft II", "missing_data_flags": ["local folder has no native PORTAL run export"]},
    {"method_id": "gct_marl", "category": "gct_marl", "transfer_axis": "population_size_transfer", "evidence_status": "external_reference",
     "native_simulator": "SMAC/StarCraft II", "missing_data_flags": ["upstream reference only; no local checkout or run"]},
    {"method_id": "bayes_tomop", "category": "bayes_tomop", "transfer_axis": "competitive_opponent_heterogeneity", "evidence_status": "implemented",
     "native_simulator": "native competitive games", "missing_data_flags": ["checkpoints exist but no parseable evaluation exports"]},
    {"method_id": "epirl_wumpus", "category": "epirl_wumpus", "transfer_axis": "single_agent_epistemic_planning", "evidence_status": "external_reference",
     "native_simulator": "Rust Wumpus World", "missing_data_flags": ["external reference only; no local checkout or results"]},
]


def cross_category_rows(run_records: list[dict], metric_records: list[dict], cost_records: list[dict],
                        seed_rows: list[dict] | None = None, curve_points: list[dict] | None = None,
                        metric_specs: dict | None = None) -> list[dict]:
    """One evidence row per method/benchmark/target; never compare raw rewards."""
    seed_rows = seed_rows or []
    curve_points = curve_points or []
    metric_specs = metric_specs or {}
    grouped_runs: dict[tuple, list[dict]] = defaultdict(list)
    for run in run_records:
        key = (run.get("method_id"), run.get("benchmark_id"), run.get("target_condition"),
               run.get("transfer_axis"), run.get("implementation_status"), run.get("simulator"))
        grouped_runs[key].append(run)
    rows = []
    for key, runs in sorted(grouped_runs.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        ids = {run.get("run_id") for run in runs}
        metrics = [m for m in metric_records if m.get("run_id") in ids]
        costs = [c for c in cost_records if c.get("run_id") in ids]
        primary = [m for m in metrics if m.get("metric_name") == "vmas_assigned_goal_team_success"
                   and m.get("value") is not None and m.get("aggregation_level") == "evaluation_episode"
                   and m.get("extensions", {}).get("split") == "test"]
        per_run_checkpoint = defaultdict(list)
        for metric in primary:
            if metric.get("seed") is not None:
                per_run_checkpoint[(metric.get("run_id"), metric.get("seed"), metric.get("checkpoint_step"))].append(float(metric["value"]))
        final_values = defaultdict(list)
        latest_step = {}
        for (run_id, seed, step), values in per_run_checkpoint.items():
            ident = (run_id, seed)
            if ident not in latest_step or (step is not None and (latest_step[ident] is None or step > latest_step[ident])):
                latest_step[ident] = step
                final_values[ident] = [sum(values) / len(values)]
            elif step == latest_step[ident]:
                final_values[ident].append(sum(values) / len(values))
        seed_values = [sum(values) / len(values) for values in final_values.values()]
        confidence = summarize_seed_values(seed_values)
        curve_group = [p for p in curve_points if p.get("target_condition") == key[2]
                       and p.get("method_id") == key[0] and p.get("benchmark_id") == key[1]
                       and p.get("metric_name") == "vmas_assigned_goal_team_success"
                       and p.get("split") == "validation"]
        budget = metric_specs.get("vmas_assigned_goal_team_success", {}).get("normalized_budget")
        auc = None
        threshold = None
        if budget:
            from transfer_eval.analysis.learning_curves import normalized_auc, time_to_threshold
            auc = normalized_auc(curve_group, budget)
            threshold = time_to_threshold(curve_group, 0.8, budget) if curve_group else None
        units = _units(runs)
        phases = {cost.get("phase") for cost in costs}
        has_safety = any(m.get("metric_name", "").startswith(("contact", "collision", "minimum_surface", "minimum_center")) for m in metrics)
        missing = []
        if not primary:
            missing.append("held-out target competence episodes unavailable")
        if len(seed_values) < 2:
            missing.append("fewer than two independent training seeds; no seed-level confidence interval")
        if not any(m.get("metric_name") == "vmas_assigned_goal_team_success" and
                   m.get("aggregation_level") == "evaluation_episode" and
                   m.get("extensions", {}).get("split") == "test" and
                   m.get("condition") == "zero_shot" for m in metrics):
            missing.append("zero-shot target baseline not registered")
        if "source_pretraining" not in phases or "target_adaptation" not in phases:
            missing.append("phase-wise cost coverage incomplete")
        missing.append("matched scratch baseline not registered for this benchmark/target")
        if not any(c.get("wall_clock_seconds") is not None for c in costs):
            missing.append("phase and end-to-end wall-clock measurements unavailable")
        rows.append({
            "method_id": key[0], "benchmark_id": key[1], "target_condition": key[2],
            "transfer_axis": key[3], "evidence_status": key[4], "native_simulator": key[5],
            "native_interaction_unit": units,
            "zero_shot_retention_or_degradation": None,
            "matched_budget_advantage_over_scratch": None,
            "normalized_within_benchmark_learning_auc": auc,
            "interactions_to_competence": threshold,
            "end_to_end_wall_clock_seconds": None,
            "target_only_wall_clock_seconds": None,
            "robustness_safety_evidence": "available" if has_safety else "missing",
            "n_independent_training_seeds": len(seed_values),
            "confidence_interval": [confidence["ci_low"], confidence["ci_high"]],
            "primary_competence_mean": confidence["mean"] if primary else None,
            "primary_competence_std": confidence["std"] if primary else None,
            "missing_data_flags": missing,
            "reproducibility_flags": ["dirty root checkout" if any(r.get("dirty_tree") for r in runs) else "clean root checkout"],
            "benchmark_semantics_id": next((m.get("benchmark_semantics_id") for m in primary), None),
            "run_count": len(runs), "run_ids": sorted(ids),
        })
    rows.extend({**item, "benchmark_id": None, "target_condition": None, "native_simulator": None,
                 "native_interaction_unit": None, "zero_shot_retention_or_degradation": None,
                 "matched_budget_advantage_over_scratch": None, "normalized_within_benchmark_learning_auc": None,
                 "interactions_to_competence": None, "end_to_end_wall_clock_seconds": None,
                 "target_only_wall_clock_seconds": None, "robustness_safety_evidence": "missing",
                 "n_independent_training_seeds": 0, "confidence_interval": None,
                 "primary_competence_mean": None, "primary_competence_std": None,
                 "missing_data_flags": ["no local reproducible runs"], "reproducibility_flags": ["no local manifest"],
                 "benchmark_semantics_id": None, "run_count": 0, "run_ids": []}
                for item in NOT_IMPLEMENTED_ROWS)
    for item in NO_LOCAL_RUN_ROWS:
        if any(run.get("extensions", {}).get("category") == item["category"] for run in run_records):
            continue
        row = {**item, "benchmark_id": None, "target_condition": None,
                     "native_interaction_unit": None, "zero_shot_retention_or_degradation": None,
                     "matched_budget_advantage_over_scratch": None, "normalized_within_benchmark_learning_auc": None,
                     "interactions_to_competence": None, "end_to_end_wall_clock_seconds": None,
                     "target_only_wall_clock_seconds": None, "robustness_safety_evidence": "unavailable",
                     "n_independent_training_seeds": 0, "confidence_interval": None,
                     "primary_competence_mean": None, "primary_competence_std": None,
                     "missing_data_flags": item["missing_data_flags"], "reproducibility_flags": ["no local manifest"],
                     "benchmark_semantics_id": None, "run_count": 0, "run_ids": []}
        row.pop("category", None)
        rows.append(row)
    return rows


def _units(runs: list[dict]) -> str | None:
    units = sorted({v.get("unit") for run in runs for v in run.get("phase_interactions", {}).values()
                    if isinstance(v, dict) and v.get("unit")})
    return ", ".join(units) or None
