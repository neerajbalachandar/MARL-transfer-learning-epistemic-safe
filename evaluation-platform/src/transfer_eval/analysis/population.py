from __future__ import annotations

import json
from collections import defaultdict

from transfer_eval.statistics.confidence import summarize_seed_values


def population_heterogeneity_summary(seed_rows: list[dict], runs: list[dict]) -> list[dict]:
    """Summarize explicit target population metadata within matching benchmarks."""
    run_info = {row.get("run_id"): row for row in runs}
    groups = defaultdict(dict)
    for row in seed_rows:
        if row.get("split") != "test" or row.get("seed") is None:
            continue
        run = run_info.get(row.get("run_id"), {})
        n_agents = run.get("n_agents_target")
        agent_types = run.get("agent_types")
        if n_agents is None and agent_types is None:
            continue
        agent_types_key = json.dumps(agent_types, sort_keys=True, separators=(",", ":")) if agent_types is not None else None
        key = (run.get("method_id"), run.get("benchmark_id"), row.get("metric_name"),
               row.get("unit"), row.get("native_unit"), row.get("benchmark_semantics_id"), n_agents,
               agent_types_key,
               row.get("source_condition"), row.get("target_condition"))
        groups[key][(row.get("run_id"), row.get("seed"))] = row.get("seed_value")

    output = []
    for key, by_seed in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        summary = summarize_seed_values([value for value in by_seed.values() if value is not None])
        output.append({
            "method_id": key[0], "benchmark_id": key[1], "metric_name": key[2],
            "unit": key[3], "native_unit": key[4], "benchmark_semantics_id": key[5],
            "n_agents_target": key[6], "agent_types": json.loads(key[7]) if key[7] is not None else None,
            "source_condition": key[8], "target_condition": key[9],
            "mean": summary["mean"], "std": summary["std"], "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"], "n_independent_seeds": summary["n_seeds"],
            "paired_comparison": False,
            "interpretation": "Descriptive population stratum; cross-stratum effects require a controlled matched protocol.",
        })
    return output
