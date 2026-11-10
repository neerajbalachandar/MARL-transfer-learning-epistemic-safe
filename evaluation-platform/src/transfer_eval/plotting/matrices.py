from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_source_target_matrix(records: list[dict], output: Path, metric_name: str) -> list[Path]:
    rows = [r for r in records if r.get("metric_name") == metric_name
            and r.get("seed_value", r.get("value")) is not None
            and r.get("source_condition") and r.get("target_condition")
            and r.get("split", "test") == "test"]
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("method_id"), row.get("benchmark_id"), row.get("benchmark_semantics_id"))].append(row)
    if not groups:
        return []
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for (method, benchmark, semantics), group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        sources = sorted({row["source_condition"] for row in group})
        targets = sorted({row["target_condition"] for row in group})
        values, counts = [], []
        for source in sources:
            val_row, cnt_row = [], []
            for target in targets:
                cell = [float(row.get("seed_value", row.get("value"))) for row in group
                        if row["source_condition"] == source and row["target_condition"] == target]
                val_row.append(sum(cell) / len(cell) if cell else float("nan"))
                cnt_row.append(len(cell))
            values.append(val_row)
            counts.append(cnt_row)
        fig, ax = plt.subplots(figsize=(max(5, len(targets) * 1.1), max(4, len(sources) * 0.7)))
        im = ax.imshow(values, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(targets)), targets, rotation=35, ha="right")
        ax.set_yticks(range(len(sources)), sources)
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                ax.text(j, i, "—" if val != val else f"{val:.2f}\n(n={counts[i][j]})", ha="center", va="center", color="white", fontsize=8)
        ax.set_xlabel("Target condition")
        ax.set_ylabel("Source condition")
        ax.set_title(f"{method}: source-target competence\n{benchmark} / {semantics}")
        fig.colorbar(im, ax=ax, label="Competence fraction")
        fig.tight_layout()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{benchmark}_{semantics}")
        for suffix in ("png", "pdf"):
            path = output / f"source_target_matrix_{slug}.{suffix}"
            fig.savefig(path, dpi=160)
            paths.append(path)
        plt.close(fig)
    return paths
