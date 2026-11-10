from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _score(row: dict, key: str):
    value = row.get("metrics", {}).get(key)
    if isinstance(value, list) and value:
        try:
            return sum(float(v) for v in value) / len(value)
        except (TypeError, ValueError):
            return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _positions(row: dict):
    values = row.get("extensions", {}).get("trajectory_positions")
    if not isinstance(values, list):
        return []
    tracks = defaultdict(list)
    for frame in values:
        if not isinstance(frame, list):
            continue
        for agent, point in enumerate(frame):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    tracks[agent].append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    pass
    return tracks


def plot_representative_episodes(episodes: list[dict], runs: list[dict], output: Path) -> tuple[list[Path], list[dict]]:
    run_by_id = {row["run_id"]: row for row in runs}
    groups = defaultdict(list)
    for row in episodes:
        if row.get("split") != "test":
            continue
        run = run_by_id.get(row.get("run_id"), {})
        trajectories = row.get("extensions", {}).get("trajectory_positions")
        if trajectories:
            groups[(run.get("method_id", "unknown"), row.get("target_condition") or "unspecified")].append(row)

    files, selections = [], []
    for (method, target), rows in sorted(groups.items()):
        scored = [(row, _score(row, "team_success")) for row in rows]
        successes = [row for row, value in scored if value is not None and value >= 1.0]
        failures = [row for row, value in scored if value is not None and value < 1.0]
        choices = []
        if successes:
            lengths = [(row, _score(row, "episode_length")) for row in successes]
            valid = [(row, val) for row, val in lengths if val is not None]
            median_length = sorted(val for _, val in valid)[len(valid) // 2] if valid else None
            row = min(successes, key=lambda r: abs((_score(r, "episode_length") or 0) - median_length)) if median_length is not None else successes[len(successes) // 2]
            choices.append(("median_success", row, "successful episode with episode length nearest the median among successful test episodes"))
            clearance_rows = [(r, _score(r, "minimum_surface_clearance")) for r in successes]
            clearance_rows = [(r, val) for r, val in clearance_rows if val is not None]
            if clearance_rows:
                row = min(clearance_rows, key=lambda pair: pair[1])[0]
                choices.append(("boundary_success", row, "successful episode with the smallest recorded minimum surface clearance"))
        progress_rows = [(r, _score(r, "individual_success")) for r in failures]
        progress_rows = [(r, val) for r, val in progress_rows if val is not None]
        if progress_rows:
            row = max(progress_rows, key=lambda pair: pair[1])[0]
            choices.append(("near_miss", row, "failed episode with the largest mean individual goal-success fraction; this is a proxy, not a geometric distance-to-goal"))

        for panel, row, criterion in choices:
            record = {"method_id": method, "target_condition": target, "panel": panel,
                      "run_id": row.get("run_id"), "seed": row.get("seed"),
                      "episode_index": row.get("episode_index"), "checkpoint_step": row.get("checkpoint_step"),
                      "selection_criterion": criterion}
            selections.append(record)
            tracks = _positions(row)
            if not tracks:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            for agent, points in sorted(tracks.items()):
                if points:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, linewidth=1.2, label=f"agent {agent}")
                    ax.scatter(xs[0], ys[0], s=18, marker="o")
                    ax.scatter(xs[-1], ys[-1], s=24, marker="x")
            goals = row.get("extensions", {}).get("trajectory_goals", [])
            if goals and isinstance(goals[-1], list):
                for goal in goals[-1]:
                    if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                        ax.scatter(float(goal[0]), float(goal[1]), marker="*", s=65, color="black")
            ax.set_title(f"{method} / {target}: {panel}\nseed={row.get('seed')} episode={row.get('episode_index')}")
            ax.set_xlabel("native world x")
            ax.set_ylabel("native world y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize="small")
            fig.tight_layout()
            path = output / "representative_episodes" / f"{method}_{target}_{panel}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=160)
            plt.close(fig)
            files.append(path)
            pdf_path = path.with_suffix(".pdf")
            fig, ax = plt.subplots(figsize=(6.2, 5.2))
            for agent, points in sorted(tracks.items()):
                if points:
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, linewidth=1.2, label=f"agent {agent}")
                    ax.scatter(xs[0], ys[0], s=18, marker="o")
                    ax.scatter(xs[-1], ys[-1], s=24, marker="x")
            goals = row.get("extensions", {}).get("trajectory_goals", [])
            if goals and isinstance(goals[-1], list):
                for goal in goals[-1]:
                    if isinstance(goal, (list, tuple)) and len(goal) >= 2:
                        ax.scatter(float(goal[0]), float(goal[1]), marker="*", s=65, color="black")
            ax.set_title(f"{method} / {target}: {panel}\nseed={row.get('seed')} episode={row.get('episode_index')}")
            ax.set_xlabel("native world x")
            ax.set_ylabel("native world y")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize="small")
            fig.tight_layout()
            fig.savefig(pdf_path)
            plt.close(fig)
            files.append(pdf_path)
    return files, selections


def plot_phase_interactions(costs: list[dict], output: Path, runs: list[dict] | None = None) -> list[Path]:
    values = [row for row in costs if row.get("native_interactions") is not None and row.get("interaction_unit")]
    units = {row["interaction_unit"] for row in values}
    if not values or len(units) != 1:
        return []
    groups = defaultdict(dict)
    for row in values:
        groups[row.get("run_id")][row.get("phase")] = float(row["native_interactions"])
    phases = ["source_pretraining", "source_probe", "curriculum", "target_adaptation", "validation", "final_test"]
    labels = sorted(groups)
    run_map = {row.get("run_id"): row for row in (runs or [])}
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 4.8))
    bottom = [0.0] * len(labels)
    for phase in phases:
        heights = [groups[label].get(phase, 0.0) for label in labels]
        if any(heights):
            ax.bar(range(len(labels)), heights, bottom=bottom, label=phase)
            bottom = [a + b for a, b in zip(bottom, heights)]
    tick_labels = [f"{run_map.get(run_id, {}).get('method_id', run_id)}\nseed={run_map.get(run_id, {}).get('seed_train')}" for run_id in labels]
    ax.set_xticks(range(len(labels)), tick_labels, rotation=45, ha="right")
    ax.set_ylabel(f"Native interactions ({next(iter(units))})")
    ax.set_title("Phase interaction accounting (wall-clock time unavailable)")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"phase_interactions.{suffix}"
        fig.savefig(path, dpi=160)
        paths.append(path)
    plt.close(fig)
    return paths
