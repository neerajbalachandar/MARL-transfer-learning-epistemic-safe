from __future__ import annotations

import random


def bootstrap_mean_interval(values: list[float], confidence: float = 0.95,
                            n_resamples: int = 10000, seed: int = 1729) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return None, None
    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        samples.append(sum(rng.choice(values) for _ in values) / len(values))
    samples.sort()
    alpha = 1 - confidence
    return samples[int(alpha / 2 * (n_resamples - 1))], samples[int((1 - alpha / 2) * (n_resamples - 1))]
