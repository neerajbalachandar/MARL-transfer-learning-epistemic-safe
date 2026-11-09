#!/usr/bin/env python3
"""Render one deterministic episode for each named checkpoint."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.evaluate_checkpoint import evaluate_seed, load_experiment, save_video


def parse_case(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("use NAME=CHECKPOINT_PATH")
    name, path = value.split("=", 1)
    if not name or Path(name).name != name:
        raise argparse.ArgumentTypeError("case name must be a simple folder name")
    return name, Path(path)


def parse_steps(value):
    try:
        steps = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated step numbers") from error
    if not steps or steps[0] < 1:
        raise argparse.ArgumentTypeError("snapshot steps must be positive")
    return steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        required=True,
        metavar="NAME=CHECKPOINT",
        help="Named checkpoint to render; repeat for more cases",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--snapshot-steps", type=parse_steps, default=parse_steps("1,25,50,75,100"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/rendered"))
    args = parser.parse_args()

    names = [name for name, _ in args.case]
    if len(names) != len(set(names)):
        raise SystemExit("Case names must be unique")
    if args.fps < 1:
        raise SystemExit("--fps must be positive")
    for name, checkpoint in args.case:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint for {name!r} not found: {checkpoint}")

    try:
        from PIL import Image
    except ImportError as error:
        raise SystemExit("Install Pillow to save time snapshots") from error

    for name, checkpoint in args.case:
        case_output = args.output / name
        with tempfile.TemporaryDirectory(prefix="benchmarl-render-") as temp_dir:
            experiment = load_experiment(checkpoint.resolve(), temp_dir)
            try:
                metrics, _, frames = evaluate_seed(
                    experiment, args.seed, 1, render=True
                )
            finally:
                experiment.close()

        video_path = case_output / f"episode_seed_{args.seed}.mp4"
        save_video(frames, video_path, fps=args.fps)
        snapshot_dir = case_output / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        saved_steps = []
        for step in args.snapshot_steps:
            if step > len(frames):
                print(f"{name}: step {step} not saved; episode ended at step {len(frames)}")
                continue
            Image.fromarray(frames[step - 1]).save(snapshot_dir / f"step_{step:03d}.png")
            saved_steps.append(step)

        summary = {
            "case": name,
            "checkpoint": str(checkpoint.resolve()),
            "evaluation_seed": args.seed,
            "team_success_rate": metrics["team_success_rate"],
            "mean_undiscounted_team_return": metrics["mean_undiscounted_team_return"],
            "episode_length": metrics["mean_episode_length"],
            "video": str(video_path),
            "snapshot_steps_saved": saved_steps,
        }
        (case_output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"{name}: success={metrics['team_success_rate']:.3f}, "
            f"return={metrics['mean_undiscounted_team_return']:.3f}"
        )
        print(f"  video: {video_path}")
        print(f"  snapshots: {snapshot_dir}")


if __name__ == "__main__":
    main()
