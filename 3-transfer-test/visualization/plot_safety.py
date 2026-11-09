"""Plot logged collision penalty and measured evaluation collision frequency."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common import evaluation_rows, find_scalars, frame_lookup, frame_series, save_figure


def plot(experiment_dir):
    experiment_dir = Path(experiment_dir)
    output_dir = Path("results/figures") / experiment_dir.name
    scalars = find_scalars(experiment_dir)
    frames = frame_lookup(scalars)
    penalty = scalars.get("collection_agents_info_agent_collisions")
    points = frame_series(penalty, frames) if penalty and frames else []
    if points:
        x, values = zip(*points)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, [abs(value) for value in values], label="Mean collision penalty magnitude")
        ax.set(title="Logged collision penalty", xlabel="Environment frames", ylabel="Penalty magnitude (reward units)")
        ax.legend()
        ax.grid(alpha=0.25)
        save_figure(fig, output_dir, "collision_penalty")
        plt.close(fig)
        print(f"Saved {output_dir / 'collision_penalty.png'}")
    else:
        print("Unavailable: collision penalty scalar or frame counter is missing")

    rows = evaluation_rows(experiment_dir)
    if not rows or "episodes_with_collision_fraction" not in rows[0]:
        print("Unavailable: actual collision frequency requires checkpoint evaluation metrics")
        return
    grouped = {}
    for row in rows:
        frame = int(float(row["training_frames"]))
        grouped.setdefault(frame, []).append(float(row["episodes_with_collision_fraction"]))
    points = sorted((frame, sum(values) / len(values)) for frame, values in grouped.items())
    x, y = zip(*points)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label="Episodes with a collision")
    ax.set(title="Measured collision frequency", xlabel="Environment frames", ylabel="Episode fraction")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, "collision_frequency")
    plt.close(fig)
    print(f"Saved {output_dir / 'collision_frequency.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    plot(parser.parse_args().experiment_dir)
