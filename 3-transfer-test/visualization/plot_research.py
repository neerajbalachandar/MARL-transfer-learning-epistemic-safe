#!/usr/bin/env python3
"""Plot actual validation, held-out and transfer records found under a run folder."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = {
    "return": ("undiscounted_team_return", "Mean team return", None),
    "success": ("team_success", "Team success fraction", (0, 1)),
    "collision": ("collision_episode", "Collision-reward condition fraction", (0, 1)),
    "contact": ("contact_episode", "Physical contact fraction", (0, 1)),
    "clearance": ("minimum_surface_clearance", "Minimum surface clearance (m)", None),
}


def read_jsonl(path):
    rows = []
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Bad JSON at {path}:{line_number}: {error}") from error
    return rows


def load_records(root):
    records = []
    for path in sorted(Path(root).rglob("*.jsonl")):
        if path.name not in ("validation_episodes.jsonl", "test_episodes.jsonl", "source_target_episodes.jsonl") and not path.name.endswith("_episodes.jsonl"):
            continue
        try:
            rows = read_jsonl(path)
        except ValueError as error:
            print(f"Skipping {path}: {error}")
            continue
        manifest_path = path.parents[1] / "manifest.json" if path.parent.name == "evaluation" else path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        for row in rows:
            row["_file"] = str(path)
            row.setdefault("method", manifest.get("method_id", manifest.get("method", "unknown")))
            row.setdefault("task", manifest.get("task", manifest.get("target_task")))
            row.setdefault("training_seed", manifest.get("training_seed", manifest.get("seed")))
            row.setdefault(
                "environment_frames",
                manifest.get("observed_environment_frames", manifest.get("total_environment_frames", row.get("training_frames"))),
            )
            if path.name.endswith("_episodes.jsonl") and path.name != "validation_episodes.jsonl":
                row.setdefault("split", "test")
            records.append(row)
    return records


def _save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _curves(records, metric, title, output, filename, task=None):
    key, ylabel, limits = METRICS[metric]
    selected = [
        row
        for row in records
        if key in row
        and row.get("split", "validation") != "test"
        and (task is None or row.get("task") == task)
    ]
    if not selected:
        print(f"Unavailable: {filename} (no records for {task or 'any task'})")
        return False
    by_method_seed_frame = defaultdict(list)
    for row in selected:
        if row.get("environment_frames") is None or row.get("training_seed") is None:
            continue
        method = row.get("method", "unknown")
        by_method_seed_frame[(method, int(row["training_seed"]), int(row["environment_frames"]))].append(float(row[key]))
    if not by_method_seed_frame:
        print(f"Unavailable: {filename} (records lack environment frame or seed fields)")
        return False
    grouped = defaultdict(dict)
    for (method, seed, frame), values in by_method_seed_frame.items():
        grouped[method].setdefault(seed, {})[frame] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method, seed_runs in sorted(grouped.items()):
        common_frames = sorted(set.intersection(*(set(run) for run in seed_runs.values()))) if seed_runs else []
        if not common_frames:
            print(f"Skipping {method} in {filename}: seed runs do not share evaluation frames")
            continue
        matrix = np.asarray([[run[frame] for frame in common_frames] for run in seed_runs.values()])
        mean = matrix.mean(axis=0)
        ax.plot(common_frames, mean, marker="o", markersize=3, label=f"{method} (n={len(matrix)})")
        if len(matrix) > 1:
            error = 1.96 * matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
            ax.fill_between(common_frames, mean - error, mean + error, alpha=.18)
            for row in matrix:
                ax.plot(common_frames, row, color=ax.lines[-1].get_color(), alpha=.18, linewidth=.7)
    ax.set_xlabel("Environment frames")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=.25)
    if limits:
        ax.set_ylim(*limits)
    ax.legend(fontsize=8)
    _save(fig, output, filename)
    return True


def source_target_matrix(root, output, gain=False):
    path = Path(root) / "source_target_episodes.jsonl"
    if not path.is_file():
        candidates = list(Path(root).rglob("source_target_episodes.jsonl"))
        if not candidates:
            print(f"Unavailable: {'transfer_gain_matrix' if gain else 'source_target_matrix'} (no source_target_episodes.jsonl)")
            return False
        path = candidates[0]
    rows = read_jsonl(path)
    source_rows = [r for r in rows if not r.get("is_scratch_baseline")]
    tasks = sorted({r["target_task"] for r in source_rows} | {r["source_task"] for r in source_rows})
    matrix = np.full((len(tasks), len(tasks)), np.nan)
    for i, source in enumerate(tasks):
        for j, target in enumerate(tasks):
            cell = [r for r in source_rows if r["source_task"] == source and r["target_task"] == target]
            if cell:
                matrix[i, j] = np.mean([float(r["team_success"]) for r in cell])
    if gain:
        scratch_rows = [r for r in rows if r.get("is_scratch_baseline")]
        if not scratch_rows:
            print("Unavailable: transfer_gain_matrix (rerun matrix evaluation with --scratch-checkpoint for every task)")
            return False
        scratch = {task: np.mean([float(r["team_success"]) for r in scratch_rows if r["target_task"] == task]) for task in tasks}
        if any(not np.isfinite(value) for value in scratch.values()):
            print("Unavailable: transfer_gain_matrix (scratch baseline missing for at least one task)")
            return False
        matrix = matrix - np.asarray([scratch[t] for t in tasks])[None, :]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, vmin=-1 if gain else 0, vmax=1, cmap="coolwarm" if gain else "viridis")
    ax.set_xticks(range(len(tasks)), tasks)
    ax.set_yticks(range(len(tasks)), tasks)
    ax.set_xlabel("Evaluation task")
    ax.set_ylabel("Source policy task")
    ax.set_title("Zero-shot gain vs target scratch baseline" if gain else "Zero-shot team success")
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if abs(matrix[i, j]) > .45 else "black")
    fig.colorbar(image, ax=ax, label="Success difference" if gain else "Team success fraction")
    _save(fig, output, "transfer_gain_matrix" if gain else "source_target_matrix")
    return True


def transfer_efficiency(records, output, threshold=0.8, task="T2"):
    """Plot frames to validation competence and normalized target-curve area."""
    by_method_seed_frame = defaultdict(list)
    for row in records:
        if row.get("task") != task or "team_success" not in row:
            continue
        if row.get("split", "validation") == "test":
            continue
        if row.get("environment_frames") is None or row.get("training_seed") is None:
            continue
        by_method_seed_frame[(row.get("method", "unknown"), int(row["training_seed"]), int(row["environment_frames"]))].append(float(row["team_success"]))
    if not by_method_seed_frame:
        print("Unavailable: transfer_efficiency (need validation learning curves with frames and seeds)")
        return False
    runs = defaultdict(list)
    max_frames = max(frame for _, _, frame in by_method_seed_frame)
    for (method, seed, frame), values in by_method_seed_frame.items():
        runs[(method, seed)].append((frame, float(np.mean(values))))
    data = defaultdict(list)
    for (method, seed), curve in runs.items():
        curve = sorted(curve)
        frames = np.asarray([point[0] for point in curve], dtype=float)
        success = np.asarray([point[1] for point in curve], dtype=float)
        reached = np.flatnonzero(success >= threshold)
        steps = float(frames[reached[0]]) if len(reached) else float(max_frames)
        auc = float(np.trapz(success, frames) / max(frames[-1] - frames[0], 1.0)) if len(frames) > 1 else float(success[0])
        data[method].append((seed, steps, auc, not len(reached)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    names = sorted(data)
    for index, method in enumerate(names):
        values = data[method]
        steps = np.asarray([item[1] for item in values])
        aucs = np.asarray([item[2] for item in values])
        axes[0].scatter(np.full(len(steps), index), steps, label=f"{method} (failed {sum(item[3] for item in values)}/{len(values)})")
        axes[1].scatter(np.full(len(aucs), index), aucs)
        axes[0].plot(index, steps.mean(), marker="_", markersize=18, color="black")
        axes[1].plot(index, aucs.mean(), marker="_", markersize=18, color="black")
    axes[0].set(xticks=range(len(names)), xticklabels=names, ylabel="Environment frames", title=f"Frames to success ≥ {threshold:.2f}")
    axes[1].set(xticks=range(len(names)), xticklabels=names, ylabel="Normalized area (success × frames)", ylim=(0, 1), title="Target learning-curve area")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=.25)
    _save(fig, output, "transfer_efficiency")
    print(f"Transfer efficiency uses max observed frame ({max_frames}) for runs that did not reach threshold {threshold:.2f}.")
    return True


def diagnostic_plots(root, output, records):
    # These figures are emitted only when the raw required fields exist.
    manifests = [json.loads(p.read_text()) for p in Path(root).rglob("manifest.json")]
    belief_rows = [(m.get("method", "BPR"), row) for m in manifests for row in m.get("probe_history", []) if "posterior" in row]
    if belief_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        methods = sorted(set(name for name, _ in belief_rows))
        for method in methods:
            subset = [r for name, r in belief_rows if name == method]
            ax.plot([r["probe_budget_episodes"] for r in subset], [r["posterior"][0] for r in subset], marker=".", label=f"{method}: source 0")
        ax.set(xlabel="Probe episodes", ylabel="Posterior probability", ylim=(0, 1), title="Bayesian belief evolution")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        _save(fig, output, "bayesian_belief_evolution")
    else:
        print("Unavailable: bayesian_belief_evolution (no posterior probe history)")

    portal = [(m, stage) for m in manifests for stage in m.get("stages", []) if stage.get("jumpstart_return") or stage.get("similarity")]
    if portal:
        fig, ax = plt.subplots(figsize=(7, 4))
        similarity_values = []
        for index, (manifest, stage) in enumerate(portal):
            candidates = sorted(set(stage.get("jumpstart_return", {})) | set(stage.get("similarity", {})))
            if not candidates:
                continue
            y = [stage.get("jumpstart_return", {}).get(candidate, np.nan) for candidate in candidates]
            similarities = [stage.get("similarity", {}).get(candidate, np.nan) for candidate in candidates]
            similarity_values.extend(value for value in similarities if np.isfinite(value))
            scatter = ax.scatter([index] * len(candidates), y, c=similarities, cmap="viridis", s=55, edgecolors="black", linewidths=.4)
            selected = stage.get("selected_task")
            if selected in candidates:
                ax.annotate(f"selected {selected}", (index, y[candidates.index(selected)]), xytext=(4, 4), textcoords="offset points", fontsize=7)
        if similarity_values:
            fig.colorbar(scatter, ax=ax, label="Aligned mean-distance similarity proxy")
        ax.set(xlabel="Recorded curriculum stage", ylabel="Candidate mean return (difficulty signal)", title="PORTAL pilot candidate scores")
        ax.grid(alpha=.2)
        _save(fig, output, "portal_task_selection")
    else:
        print("Unavailable: portal_task_selection (no candidate scores in manifests)")

    maptf = [m for m in manifests if m.get("option_selection_frequency")]
    if maptf:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        labels, frequencies = [], []
        durations = []
        for manifest in maptf:
            for receiver, peers in manifest.get("option_selection_frequency", {}).items():
                for peer, count in peers.items():
                    labels.append(f"{receiver}←{peer}")
                    frequencies.append(count)
            durations.extend(manifest.get("option_duration_updates", []))
        if labels:
            axes[0].bar(labels, frequencies)
            axes[0].set(ylabel="Selections", title="MAPTF pilot peer selections")
        if durations:
            axes[1].hist(durations, bins=min(15, max(1, len(set(durations)))))
            axes[1].set(xlabel="Updates", ylabel="Options", title="Recorded option duration")
        else:
            axes[1].text(.5, .5, "Duration data unavailable", ha="center", va="center", transform=axes[1].transAxes)
        _save(fig, output, "maptf_option_diagnostics")
    else:
        print("Unavailable: maptf_option_diagnostics (no option log in manifests)")

    episodes = [r for r in records if "minimum_surface_clearance" in r]
    if episodes:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].hist([r["minimum_surface_clearance"] for r in episodes], bins=20)
        axes[0].set_xlabel("Minimum surface clearance (m)")
        axes[1].hist([r.get("collision_event_count", 0) for r in episodes], bins=range(0, max(2, max(r.get("collision_event_count", 0) for r in episodes) + 2)))
        axes[1].set_xlabel("Collision-reward events")
        axes[2].hist([r.get("episode_length", 0) for r in episodes], bins=20)
        axes[2].set_xlabel("Episode length (steps)")
        axes[0].set_ylabel("Episodes")
        fig.suptitle("Safety outcome distributions")
        _save(fig, output, "safety_distribution")
    else:
        print("Unavailable: safety_distribution (no raw episode metrics)")

    trajectories = [r for r in episodes if r.get("trajectory_positions") and r.get("trajectory_goals")]
    if trajectories:
        fig, ax = plt.subplots(figsize=(6, 6))
        for row in trajectories[:12]:
            path = np.asarray(row["trajectory_positions"])
            is_success = bool(row.get("team_success"))
            color = "tab:green" if is_success else ("tab:red" if row.get("contact_episode") else "tab:blue")
            for agent in range(path.shape[1]):
                ax.plot(path[:, agent, 0], path[:, agent, 1], color=color, alpha=.5)
                goals = np.asarray(row["trajectory_goals"])
                ax.scatter(goals[-1, agent, 0], goals[-1, agent, 1], marker="x", color=color, s=20)
        ax.set(aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Recorded trajectories (green success, red contact)")
        ax.grid(alpha=.2)
        _save(fig, output, "trajectory_comparison")
    else:
        print("Unavailable: trajectory_comparison (no trajectory records)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="Root directory containing run folders")
    parser.add_argument("--output", type=Path, default=Path("results/figures/research"))
    args = parser.parse_args()
    records = load_records(args.runs)
    if not records:
        raise SystemExit(f"No episode JSONL records found under {args.runs}")
    target_tasks = sorted({row.get("task") for row in records if row.get("task")})
    for metric, name in (("return", "baseline_learning_curves"), ("success", "baseline_success_curves")):
        _curves(records, metric, "Learning curve", args.output, name)
    for metric, name in (("collision", "safety_collision_curves"), ("contact", "safety_contact_curves"), ("clearance", "safety_clearance_curves")):
        _curves(records, metric, "Safety during training", args.output, name)
    if "T2" in target_tasks:
        for metric, filename in (("return", "transfer_learning_return"), ("success", "transfer_learning_success")):
            _curves(records, metric, "T2 transfer learning", args.output, filename, task="T2")
        transfer_efficiency(records, args.output, task="T2")
    source_target_matrix(args.runs, args.output)
    source_target_matrix(args.runs, args.output, gain=True)
    diagnostic_plots(args.runs, args.output, records)
    print(f"Saved available figures to {args.output}")


if __name__ == "__main__":
    main()
