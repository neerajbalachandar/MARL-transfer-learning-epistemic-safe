from __future__ import annotations

from collections import defaultdict

from transfer_eval.statistics.confidence import summarize_seed_values


# Only explicit manifest condition labels are assigned roles. Unrecognized labels stay unclassified.
CONDITION_ROLES = {
    "scratch": {"scratch", "from_scratch", "target_from_scratch"},
    "zero_shot": {"zero_shot", "source_policy_zero_shot"},
    "transfer": {"transfer", "full_transfer", "target_adaptation"},
}


def _role(condition: str | None) -> str | None:
    if condition is None:
        return None
    value = condition.strip().lower()
    return next((role for role, labels in CONDITION_ROLES.items() if value in labels), None)


def condition_comparisons(seed_rows: list[dict], runs: list[dict]) -> list[dict]:
    """Summarize explicit scratch/zero-shot/transfer conditions without pooling semantics."""
    run_info = {row.get("run_id"): row for row in runs}
    by_group = defaultdict(lambda: defaultdict(dict))
    by_budget = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in seed_rows:
        if row.get("split") != "test" or row.get("seed") is None:
            continue
        run = run_info.get(row.get("run_id"), {})
        role = _role(run.get("condition"))
        if role is None:
            continue
        group = (run.get("method_id"), run.get("benchmark_id"), row.get("source_condition"), row.get("target_condition"),
                 row.get("metric_name"), row.get("unit"), row.get("native_unit"), row.get("benchmark_semantics_id"))
        identity = (row.get("run_id"), row.get("seed"))
        step = row.get("checkpoint_step")
        by_budget[group][role][step][identity] = row["seed_value"]
        prior = by_group[group][role].get(identity)
        if prior is None or (row.get("checkpoint_step") is not None and
                             (prior.get("checkpoint_step") is None or row["checkpoint_step"] > prior["checkpoint_step"])):
            by_group[group][role][identity] = row

    output = []
    for group, roles in sorted(by_group.items(), key=lambda item: tuple(str(value) for value in item[0])):
        values = {role: [row["seed_value"] for row in runs_by_seed.values()]
                  for role, runs_by_seed in roles.items()}
        summaries = {role: summarize_seed_values(role_values) for role, role_values in values.items()}
        zero = summaries.get("zero_shot", {}).get("mean")
        transfer = summaries.get("transfer", {}).get("mean")
        scratch = summaries.get("scratch", {}).get("mean")
        common_budgets = set(by_budget[group].get("transfer", {})) & set(by_budget[group].get("scratch", {}))
        common_budgets.discard(None)
        matched_budget = max(common_budgets) if common_budgets else None
        transfer_at_budget = list(by_budget[group].get("transfer", {}).get(matched_budget, {}).values()) if matched_budget is not None else []
        scratch_at_budget = list(by_budget[group].get("scratch", {}).get(matched_budget, {}).values()) if matched_budget is not None else []
        missing = []
        if zero is None:
            missing.append("no explicit zero_shot condition in run manifests")
        if transfer is None:
            missing.append("no explicit transfer condition in run manifests")
        if scratch is None:
            missing.append("no explicit scratch condition in run manifests")
        output.append({
            "method_id": group[0], "benchmark_id": group[1], "source_condition": group[2],
            "target_condition": group[3], "metric_name": group[4], "unit": group[5],
            "native_unit": group[6], "benchmark_semantics_id": group[7],
            "zero_shot_mean": zero, "transfer_final_mean": transfer, "scratch_final_mean": scratch,
            "adaptation_gain": transfer - zero if transfer is not None and zero is not None else None,
            "transfer_vs_scratch_final": transfer - scratch if transfer is not None and scratch is not None else None,
            "matched_budget": matched_budget,
            "matched_budget_advantage": (sum(transfer_at_budget) / len(transfer_at_budget) -
                                         sum(scratch_at_budget) / len(scratch_at_budget))
                                        if transfer_at_budget and scratch_at_budget else None,
            "n_zero_shot_seeds": summaries.get("zero_shot", {}).get("n_seeds", 0),
            "n_transfer_seeds": summaries.get("transfer", {}).get("n_seeds", 0),
            "n_scratch_seeds": summaries.get("scratch", {}).get("n_seeds", 0),
            "zero_shot_ci": [summaries.get("zero_shot", {}).get("ci_low"), summaries.get("zero_shot", {}).get("ci_high")],
            "transfer_ci": [summaries.get("transfer", {}).get("ci_low"), summaries.get("transfer", {}).get("ci_high")],
            "scratch_ci": [summaries.get("scratch", {}).get("ci_low"), summaries.get("scratch", {}).get("ci_high")],
            "paired_comparison": False,
            "missing_data_flags": missing,
        })
    return output
