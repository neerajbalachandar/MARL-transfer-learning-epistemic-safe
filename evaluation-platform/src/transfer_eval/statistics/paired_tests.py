from __future__ import annotations

import itertools
import math


def paired_sign_flip_test(left: list[float], right: list[float]) -> dict:
    """Exact paired randomization test; only appropriate for documented seed pairs."""
    if len(left) != len(right) or not left:
        raise ValueError("Paired tests require equal non-empty seed lists")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    observed = abs(sum(differences) / len(differences))
    if len(differences) <= 20:
        extremes = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
            statistic = abs(sum(d * s for d, s in zip(differences, signs)) / len(differences))
            extremes += statistic >= observed - 1e-12
            total += 1
        p = extremes / total
    else:
        sd = math.sqrt(sum((d - sum(differences) / len(differences)) ** 2 for d in differences) / (len(differences) - 1))
        z = abs(sum(differences) / len(differences)) / (sd / math.sqrt(len(differences)) + 1e-12)
        p = math.erfc(z / math.sqrt(2))
    mean = sum(differences) / len(differences)
    sd = math.sqrt(sum((d - mean) ** 2 for d in differences) / (len(differences) - 1)) if len(differences) > 1 else 0.0
    return {"n_pairs": len(differences), "mean_difference": mean, "cohens_dz": mean / sd if sd else None, "p_value": p}


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, (index, p_value) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p_value))
        adjusted[index] = running
    return adjusted
