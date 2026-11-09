"""Plot team and individual success from saved evaluation metrics, if present."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common import evaluation_rows, save_figure


def plot(experiment_dir):
    experiment_dir = Path(experiment_dir)
    rows = evaluation_rows(experiment_dir)
    if not rows or "team_success_rate" not in rows[0]:
        print("Unavailable: no checkpoint evaluation metrics found")
        return
    grouped = {}
    for row in rows:
        frame = int(float(row["training_frames"]))
        grouped.setdefault(frame, []).append(float(row["team_success_rate"]))
    points = sorted((frame, sum(values) / len(values)) for frame, values in grouped.items())
    x, y = zip(*points)
    output_dir = Path("results/figures") / experiment_dir.name
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", label="Team success rate")
    ax.set(title="Team success over training", xlabel="Environment frames", ylabel="Success rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, "team_success")
    plt.close(fig)
    print(f"Saved {output_dir / 'team_success.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    plot(parser.parse_args().experiment_dir)
