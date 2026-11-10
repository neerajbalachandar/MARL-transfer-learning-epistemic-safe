from __future__ import annotations

from collections import defaultdict

from transfer_eval.statistics.bootstrap import bootstrap_mean_interval


def curve_summary(seed_rows: list[dict], confidence: float = 0.95, resamples: int = 10000) -> list[dict]:
    groups = defaultdict(dict)
    for row in seed_rows:
        step = row.get("checkpoint_step")
        if step is None:
            continue
        if row.get("seed") is None:
            continue
        key = (row.get("metric_name"), row.get("source_condition"), row.get("target_condition"),
               row.get("native_unit"), step, row.get("split"), row.get("method_id"),
               row.get("benchmark_id"), row.get("unit"), row.get("benchmark_semantics_id"))
        groups[key].setdefault((row.get("run_id"), row.get("seed")), []).append(float(row["seed_value"]))
    output = []
    for key, by_seed in sorted(groups.items(), key=lambda pair: (str(pair[0][0]), str(pair[0][1]), str(pair[0][2]), pair[0][4])):
        values = [sum(samples) / len(samples) for samples in by_seed.values()]
        low, high = bootstrap_mean_interval(values, confidence, resamples, seed=int(key[4]) % 2147483647)
        output.append({"metric_name": key[0], "source_condition": key[1], "target_condition": key[2],
                       "native_unit": key[3], "checkpoint_step": key[4], "split": key[5], "n_seeds": len(values), "mean": sum(values) / len(values),
                       "ci_low": low, "ci_high": high,
                       "per_seed": [{"run_id": identity[0], "seed": identity[1], "value": sum(samples) / len(samples)}
                                    for identity, samples in by_seed.items()],
                       "method_id": key[6], "benchmark_id": key[7], "unit": key[8],
                       "benchmark_semantics_id": key[9]})
    return output


def normalized_auc(points: list[dict], budget: float) -> float | None:
    points = sorted((p for p in points if p.get("checkpoint_step") is not None and p.get("mean") is not None and p["checkpoint_step"] <= budget), key=lambda p: p["checkpoint_step"])
    if len(points) < 2 or budget <= 0:
        return None
    area = 0.0
    for left, right in zip(points, points[1:]):
        area += (right["checkpoint_step"] - left["checkpoint_step"]) * (left["mean"] + right["mean"]) / 2
    return area / budget


def time_to_threshold(points: list[dict], threshold: float, budget: float) -> dict:
    points = sorted((p for p in points if p.get("mean") is not None and p.get("checkpoint_step") is not None), key=lambda p: p["checkpoint_step"])
    for row in points:
        if row["mean"] >= threshold:
            return {"value": row["checkpoint_step"], "native_unit": row.get("native_unit"), "right_censored": False, "threshold": threshold}
    return {"value": budget, "native_unit": points[0].get("native_unit") if points else None, "right_censored": True,
            "status": "not_reached_within_declared_budget", "threshold": threshold}
