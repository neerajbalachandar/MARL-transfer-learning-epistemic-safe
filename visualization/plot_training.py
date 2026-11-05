"""Plot episode reward against actual environment frames."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from common import find_scalars, frame_lookup, frame_series, save_figure


def plot(experiment_dir):
    experiment_dir = Path(experiment_dir)
    scalars = find_scalars(experiment_dir)
    frames = frame_lookup(scalars)
    output_dir = Path("results/figures") / experiment_dir.name
    plots = [
        ("collection_reward_episode_reward_mean", "Training episode reward", "training_reward"),
        ("eval_reward_episode_reward_mean", "Evaluation episode reward", "evaluation_reward"),
    ]
    for key, title, filename in plots:
        path = scalars.get(key)
        points = frame_series(path, frames) if path and frames else []
        if not points:
            print(f"Unavailable: {title} or its environment-frame counter is missing")
            continue
        x, y = zip(*points)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, y, label=title)
        ax.set(title=title, xlabel="Environment frames", ylabel="Mean episode return")
        ax.legend()
        ax.grid(alpha=0.25)
        save_figure(fig, output_dir, filename)
        plt.close(fig)
        print(f"Saved {output_dir / (filename + '.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    plot(parser.parse_args().experiment_dir)
