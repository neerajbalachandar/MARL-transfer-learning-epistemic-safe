from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SAFETY_METRICS = {"collision_event_count", "contact_episode", "geometric_overlap_episode",
                  "minimum_surface_clearance", "minimum_center_separation"}


def plot_safety_distributions(metrics: list[dict], runs: list[dict], output: Path) -> list[Path]:
    run_info = {row.get("run_id"): row for row in runs}
    grouped = defaultdict(list)
    for row in metrics:
        name, value = row.get("metric_name"), row.get("value")
        if name not in SAFETY_METRICS or value is None or row.get("aggregation_level") != "evaluation_episode":
            continue
        run = run_info.get(row.get("run_id"), {})
        key = (run.get("method_id"), run.get("benchmark_id"), row.get("target_condition"), name, row.get("unit"))
        grouped[key].append(float(value))
    if not grouped:
        return []
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in sorted({key[3] for key in grouped}):
        entries = [(key, values) for key, values in grouped.items() if key[3] == name]
        labels = [f"{key[0]}\n{key[2] or 'unspecified'}" for key, _ in entries]
        values = [sample for _, sample in entries]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
        ax.boxplot(values, labels=labels, showfliers=False)
        unit = next((key[4] for key, _ in entries if key[4]), "native unit unavailable")
        ax.set_ylabel(f"{name} ({unit})")
        ax.set_title(f"Native VMAS episode safety metric: {name}")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"safety_{name}.{suffix}"
            fig.savefig(path, dpi=160)
            files.append(path)
        plt.close(fig)
    return files
