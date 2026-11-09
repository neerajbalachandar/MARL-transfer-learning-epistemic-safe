#!/usr/bin/env python3
"""Plot held-out transfer results from one or more seed output folders."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("team_success_rate", "Team success"),
    ("mean_return", "Mean return"),
    ("contact_episode_fraction", "Episodes with contact"),
)


def load_results(root):
    grouped = defaultdict(list)
    completed_runs = set()
    for evaluation_path in Path(root).rglob("evaluation.json"):
        manifest_path = evaluation_path.with_name("manifest.json")
        if not manifest_path.exists():
            continue
        with manifest_path.open() as source:
            manifest = json.load(source)
        with evaluation_path.open() as source:
            evaluation = json.load(source)
        if all(metric in evaluation for metric, _ in METRICS):
            grouped[manifest["method"]].append(evaluation)
            completed_runs.add(evaluation_path.parent.resolve())

    # Source-task baselines are evaluated by evaluate_checkpoint.py and saved
    # as per-episode JSONL rather than evaluation.json.
    for episode_path in Path(root).rglob("test_episodes.jsonl"):
        run_dir = episode_path.parent.parent
        if run_dir.resolve() in completed_runs:
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        with manifest_path.open() as source:
            manifest = json.load(source)
        if manifest.get("method") != "mappo" or manifest.get("task") != "T2":
            continue
        with episode_path.open() as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if rows:
            count = len(rows)
            grouped["scratch_t2"].append(
                {
                    "team_success_rate": sum(row["team_success"] for row in rows) / count,
                    "mean_return": sum(row["undiscounted_team_return"] for row in rows) / count,
                    "contact_episode_fraction": sum(row["contact_episode"] for row in rows) / count,
                }
            )
    return grouped


def plot_results(results, output_dir):
    methods = sorted(results)
    figure, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    for axis, (key, title) in zip(axes, METRICS):
        means, errors = [], []
        for method in methods:
            values = np.asarray([row[key] for row in results[method]], dtype=float)
            means.append(values.mean())
            errors.append(values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0)
        axis.bar(range(len(methods)), means, yerr=errors, capsize=4, color="#4c78a8")
        axis.set_title(title)
        axis.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"transfer_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="Folder containing transfer seed runs")
    parser.add_argument("--figures", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    results = load_results(args.output_root)
    if not results:
        raise SystemExit(f"No transfer evaluation.json files found under {args.output_root}")
    plot_results(results, args.figures)
    print(f"Saved transfer comparison plots to {args.figures}")


if __name__ == "__main__":
    main()
