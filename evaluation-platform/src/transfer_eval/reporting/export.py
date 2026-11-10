from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_pdf_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    ax.axis("off")
    cols = ["method_id", "benchmark_id", "evidence_status", "native_simulator", "n_independent_training_seeds"]
    values = [[str(row.get(col, "—") if row.get(col) is not None else "—")[:42] for col in cols] for row in rows]
    table = ax.table(cellText=values, colLabels=cols, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)
    ax.set_title("MARL Transfer Evidence Summary\nNative rewards and interaction units are not pooled across benchmarks", pad=20)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
