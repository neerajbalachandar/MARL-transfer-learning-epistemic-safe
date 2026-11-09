"""Render one deterministic checkpoint rollout to an MP4 video."""

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "../BenchMARL"))

from experiments.evaluate_checkpoint import evaluate_seed, load_experiment, save_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="benchmarl-render-") as temp_dir:
        experiment = load_experiment(args.checkpoint.resolve(), temp_dir)
        try:
            _, _, frames = evaluate_seed(experiment, args.seed, 1, render=True)
        finally:
            experiment.close()
    output = args.output or ROOT / "results" / "videos" / f"{args.checkpoint.stem}_seed_{args.seed}.mp4"
    save_video(frames, output.resolve())
    print(f"Video saved to {output.resolve()}")


if __name__ == "__main__":
    main()
