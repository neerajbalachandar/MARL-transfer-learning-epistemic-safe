from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_condition_comparisons(rows: list[dict], output: Path) -> list[Path]:
    created = []
    valid = [row for row in rows if row.get("adaptation_gain") is not None]
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row.get("method_id"), row.get("benchmark_id"), row.get("metric_name"))].append(row)
    for (method, benchmark, metric), items in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        sources = sorted({row.get("source_condition") or "unspecified" for row in items})
        targets = sorted({row.get("target_condition") or "unspecified" for row in items})
        matrix = [[math.nan for _ in targets] for _ in sources]
        for row in items:
            i = sources.index(row.get("source_condition") or "unspecified")
            j = targets.index(row.get("target_condition") or "unspecified")
            matrix[i][j] = float(row["adaptation_gain"])
        finite = [value for line in matrix for value in line if math.isfinite(value)]
        bound = max(abs(min(finite)), abs(max(finite)), 1e-9)
        fig, ax = plt.subplots(figsize=(max(5, len(targets) * 1.15), max(4, len(sources) * 0.8)))
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
        ax.set_xticks(range(len(targets)), targets, rotation=35, ha="right")
        ax.set_yticks(range(len(sources)), sources)
        ax.set_xlabel("Target condition")
        ax.set_ylabel("Source condition recorded in the run manifest")
        ax.set_title(f"Transfer adaptation gain: {method} / {benchmark}\n{metric}; transfer minus explicit zero-shot")
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if math.isfinite(matrix[i][j]):
                    ax.text(j, i, f"{matrix[i, j]:.3g}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label=f"Native metric difference ({metric})")
        fig.tight_layout()
        output.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{benchmark}_{metric}")
        for suffix in ("png", "pdf"):
            path = output / f"adaptation_gain_{slug}.{suffix}"
            fig.savefig(path, dpi=160)
            created.append(path)
        plt.close(fig)

    scratch_groups = defaultdict(list)
    for row in rows:
        if row.get("matched_budget_advantage") is not None:
            scratch_groups[(row.get("method_id"), row.get("benchmark_id"), row.get("metric_name"),
                            row.get("native_unit"))].append(row)
    for (method, benchmark, metric, unit), scratch_rows in sorted(scratch_groups.items(), key=lambda pair: str(pair[0])):
        labels = [str(row.get("target_condition") or "unspecified") for row in scratch_rows]
        values = [float(row["matched_budget_advantage"]) for row in scratch_rows]
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 4.5))
        ax.bar(range(len(labels)), values, color=["#b24a4a" if value < 0 else "#3c8d70" for value in values])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_ylabel("Transfer minus scratch (benchmark-local metric)")
        ax.set_title(f"Matched-budget comparison: {method} / {benchmark}\n{metric}; budget unit: {unit or 'unknown'}")
        fig.tight_layout()
        output.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{benchmark}_{metric}")
        for suffix in ("png", "pdf"):
            path = output / f"matched_budget_transfer_vs_scratch_{slug}.{suffix}"
            fig.savefig(path, dpi=160)
            created.append(path)
        plt.close(fig)
    return created
