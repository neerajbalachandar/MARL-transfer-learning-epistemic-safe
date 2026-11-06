#!/usr/bin/env python3
"""Train the shared fixed-layout navigation task for transfer experiments."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transfer.experiment import make_experiment, write_run_manifest
from transfer.navigation import TASK_GOALS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASK_GOALS, help="Task layout to train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=600_000)
    parser.add_argument("--eval-interval", type=int, default=120_000)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = args.output or Path("outputs") / args.task / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    experiment = make_experiment(
        args.task,
        args.seed,
        args.frames,
        output,
        args.device,
        evaluation_interval=args.eval_interval,
        evaluation_episodes=args.eval_episodes,
        validation_seed=args.validation_seed,
    )
    experiment.run()
    write_run_manifest(experiment, completed=True)


if __name__ == "__main__":
    main()
