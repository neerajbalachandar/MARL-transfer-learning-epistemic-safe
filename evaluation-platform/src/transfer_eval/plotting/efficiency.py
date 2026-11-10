from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transfer_eval.analysis.learning_curves import curve_summary, normalized_auc, time_to_threshold


def efficiency_table(seed_rows: list[dict], budget_by_metric: dict[str, float], threshold: float = 0.8) -> list[dict]:
    points = curve_summary(seed_rows)
    preferred = "validation" if any(p.get("split") == "validation" for p in points) else "test"
    points = [p for p in points if p.get("split") in (preferred, None)]
    groups = {}
    for p in points:
        groups.setdefault((p.get("method_id"), p.get("benchmark_id"), p["metric_name"],
                           p["target_condition"], p.get("unit"), p["native_unit"],
                           p.get("benchmark_semantics_id")), []).append(p)
    output = []
    for (method, benchmark, metric, target, value_unit, unit, semantics), rows in groups.items():
        budget = budget_by_metric.get(metric)
        output.append({"method_id": method, "benchmark_id": benchmark, "metric_name": metric,
                       "target_condition": target, "unit": value_unit, "native_unit": unit,
                       "benchmark_semantics_id": semantics,
                       "declared_budget": budget, "normalized_auc": normalized_auc(rows, budget) if budget else None,
                       "time_to_threshold": time_to_threshold(rows, threshold, budget) if budget else None,
                       "threshold": threshold})
    return output


def plot_efficiency(rows: list[dict], output: Path) -> list[Path]:
    available = [r for r in rows if r.get("normalized_auc") is not None]
    if not available:
        return []
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    grouped = defaultdict(list)
    for row in available:
        grouped[(row.get("method_id"), row.get("benchmark_id"), row.get("metric_name"),
                 row.get("benchmark_semantics_id"))].append(row)
    for (method, benchmark, metric, semantics), items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        labels = [str(row.get("target_condition") or "unknown") for row in items]
        fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.1), 4))
        ax.bar(labels, [row["normalized_auc"] for row in items])
        ax.set_ylabel("Normalized AUC (within declared benchmark budget)")
        ax.set_title(f"Learning efficiency: {method} / {benchmark}\n{metric}; {semantics}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{benchmark}_{metric}")
        for suffix in ("png", "pdf"):
            path = output / f"learning_efficiency_{slug}.{suffix}"
            fig.savefig(path, dpi=160)
            paths.append(path)
        plt.close(fig)
    return paths
