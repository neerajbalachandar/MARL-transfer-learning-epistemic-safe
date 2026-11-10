from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_experiment_summary(rows: list[dict], output: Path) -> list[Path]:
    """Consolidate available primary competence summaries in benchmark-local panels."""
    available = [row for row in rows if row.get("primary_competence_mean") is not None
                 and row.get("benchmark_id") and row.get("benchmark_semantics_id")]
    panels = defaultdict(list)
    for row in available:
        panels[(row["benchmark_id"], row["benchmark_semantics_id"])].append(row)
    if not panels:
        return []
    output.mkdir(parents=True, exist_ok=True)
    count = len(panels)
    fig, axes = plt.subplots(count, 1, figsize=(8, max(4, 3.3 * count)), squeeze=False)
    for ax, ((benchmark, semantics), items) in zip(axes[:, 0], sorted(panels.items())):
        items.sort(key=lambda row: (str(row.get("method_id")), str(row.get("target_condition"))))
        labels = [f"{row.get('method_id')} / {row.get('target_condition') or 'unspecified'}" for row in items]
        means = [float(row["primary_competence_mean"]) for row in items]
        low = [row.get("confidence_interval", [None, None])[0] for row in items]
        high = [row.get("confidence_interval", [None, None])[1] for row in items]
        ax.scatter(means, range(len(items)), color="#287d8e", zorder=3)
        for index, (mean, lo, hi) in enumerate(zip(means, low, high)):
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [index, index], color="#287d8e", linewidth=1.4)
            ax.text(mean, index + 0.13, f"n={items[index].get('n_independent_training_seeds', 0)}", fontsize=8)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Native primary competence fraction")
        ax.set_title(f"{benchmark} · {semantics}")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Experiment summary by benchmark semantics (panels are not cross-comparable)", y=1.0)
    fig.tight_layout()
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"experiment_summary.{suffix}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
