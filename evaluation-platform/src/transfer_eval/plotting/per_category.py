from __future__ import annotations

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transfer_eval.analysis.learning_curves import curve_summary


def plot_learning_curves(seed_rows: list[dict], output: Path, category: str) -> list[Path]:
    all_points = curve_summary(seed_rows)
    groups = {}
    for point in all_points:
        if point.get("metric_name"):
            key = (point.get("method_id"), point.get("benchmark_id"), point.get("benchmark_semantics_id"), point["metric_name"])
            groups.setdefault(key, []).append(point)
    if not groups:
        return []
    output.mkdir(parents=True, exist_ok=True)
    created = []
    for (method, benchmark, semantics, metric_name), points in sorted(groups.items(), key=lambda pair: str(pair[0])):
        preferred = ("validation" if any(p.get("split") == "validation" for p in points) else
                     "test" if any(p.get("split") == "test" for p in points) else None)
        points = [p for p in points if p.get("split") == preferred] if preferred is not None else points
        if not points:
            continue
        by_target = {}
        for row in points:
            label = f"{row.get('condition') or 'condition unspecified'} · {row.get('source_condition') or 'source unspecified'} → {row.get('target_condition') or 'target unspecified'}"
            by_target.setdefault(label, []).append(row)
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for target, rows in sorted(by_target.items()):
            rows.sort(key=lambda r: r["checkpoint_step"])
            xs = [r["checkpoint_step"] for r in rows]
            ys = [r["mean"] for r in rows]
            ax.plot(xs, ys, marker="o", label=f"{target} (n≤{max(r['n_seeds'] for r in rows)} seeds)")
            if all(r.get("ci_low") is not None and r.get("ci_high") is not None for r in rows):
                ax.fill_between(xs, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], alpha=0.2)
        counter_unit = next((row.get("native_unit") for row in points if row.get("native_unit")), "native unit unavailable")
        value_unit = next((row.get("unit") for row in points if row.get("unit")), "native value")
        ax.set_xlabel(f"Checkpoint counter ({counter_unit})")
        ax.set_ylabel(f"{metric_name} ({value_unit})")
        if metric_name == "vmas_assigned_goal_team_success":
            ax.set_ylim(-0.03, 1.03)
        split_label = preferred or "native split unknown"
        ax.set_title(f"{method or category} / {benchmark}\n{metric_name} ({semantics}); {split_label}")
        ax.legend(loc="best", fontsize="small")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{method}_{benchmark}_{metric_name}")
        for extension in ("png", "pdf"):
            path = output / f"{category}_{slug}_learning_curve.{extension}"
            fig.savefig(path, dpi=160)
            created.append(path)
        plt.close(fig)
    return created
