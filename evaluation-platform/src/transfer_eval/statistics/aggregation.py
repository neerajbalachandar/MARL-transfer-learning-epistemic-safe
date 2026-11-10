from __future__ import annotations

from collections import defaultdict

from transfer_eval.statistics.confidence import summarize_seed_values


def aggregate_episodes_to_seed(records: list[dict], metric_name: str) -> list[dict]:
    """Average evaluation episodes within each run/seed/checkpoint first."""
    groups = defaultdict(list)
    for row in records:
        if row.get("metric_name") != metric_name or row.get("value") is None:
            continue
        key = (row.get("run_id"), row.get("seed"), row.get("source_condition"),
               row.get("target_condition"), row.get("checkpoint_step"),
               row.get("native_unit"), row.get("unit"), row.get("benchmark_semantics_id"),
               row.get("extensions", {}).get("split"))
        groups[key].append(float(row["value"]))
    result = []
    for key, values in groups.items():
        result.append({
            "run_id": key[0], "seed": key[1], "source_condition": key[2],
            "target_condition": key[3], "checkpoint_step": key[4],
            "native_unit": key[5], "unit": key[6], "benchmark_semantics_id": key[7],
            "split": key[8],
            "metric_name": metric_name, "seed_value": sum(values) / len(values),
            "episodes_within_seed": len(values),
        })
    return result


def summarize_groups(seed_rows: list[dict], keys: tuple[str, ...] = ("target_condition", "metric_name")) -> list[dict]:
    groups = defaultdict(dict)
    for row in seed_rows:
        key = tuple(row.get(field) for field in keys)
        groups[key][(row.get("run_id"), row.get("seed"))] = row["seed_value"]
    output = []
    for key, seed_map in groups.items():
        summary = summarize_seed_values(list(seed_map.values()))
        output.append({**dict(zip(keys, key)), **summary, "per_seed": seed_map})
    return output
