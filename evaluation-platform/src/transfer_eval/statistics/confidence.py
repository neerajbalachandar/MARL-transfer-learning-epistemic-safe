from __future__ import annotations

import math
from transfer_eval.statistics.bootstrap import bootstrap_mean_interval


def summarize_seed_values(values: list[float], confidence: float = 0.95,
                          n_resamples: int = 10000, seed: int = 1729) -> dict:
    if not values:
        return {"n_seeds": 0, "mean": None, "std": None, "ci_low": None, "ci_high": None}
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else None
    low, high = bootstrap_mean_interval(values, confidence, n_resamples, seed)
    return {"n_seeds": len(values), "mean": mean, "std": std, "ci_low": low, "ci_high": high}
