#!/usr/bin/env python3
"""Evaluate source policies on every fixed navigation task before fine-tuning."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transfer.experiment import load_source_experiment, probe_policy
from transfer.navigation import TASK_GOALS


def parse_checkpoint(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("use TASK=CHECKPOINT_PATH")
    task, checkpoint = value.split("=", 1)
    if task not in TASK_GOALS:
        raise argparse.ArgumentTypeError(f"unknown source task {task!r}")
    return task, Path(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        metavar="TASK=PATH",
        help="Source policy checkpoint; repeat for tasks and independent training seeds",
    )
    parser.add_argument(
        "--scratch-checkpoint",
        action="append",
        type=parse_checkpoint,
        default=[],
        metavar="TASK=PATH",
        help="Target-task scratch policy for a valid transfer-gain matrix; repeat for each task/seed",
    )
    parser.add_argument("--evaluation-seeds", default="900001,900002,900003")
    parser.add_argument("--episodes-per-seed", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("outputs/source_target_matrix"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seeds = [int(value) for value in args.evaluation_seeds.split(",") if value.strip()]
    if not seeds or args.episodes_per_seed < 1:
        raise SystemExit("Provide evaluation seeds and at least one episode per seed")
    for task, checkpoint in args.checkpoint + args.scratch_checkpoint:
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found for {task}: {checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "source_target_episodes.jsonl"
    matrix = {}
    with records_path.open("w") as output:
        for checkpoint_index, (source_task, checkpoint) in enumerate(args.checkpoint):
            with tempfile.TemporaryDirectory(prefix="source-target-matrix-") as temp:
                source = load_source_experiment(checkpoint, temp)
                try:
                    if source.task.name != source_task:
                        raise SystemExit(
                            f"Checkpoint task mismatch: labelled {source_task}, "
                            f"checkpoint contains {source.task.name}"
                        )
                    training_seed = int(source.seed)
                    for target_task in TASK_GOALS:
                        cell = []
                        for evaluation_seed in seeds:
                            rows, _ = probe_policy(
                                source,
                                target_task,
                                evaluation_seed,
                                args.episodes_per_seed,
                            )
                            for episode_index, row in enumerate(rows):
                                record = {
                                    "source_task": source_task,
                                    "source_training_seed": training_seed,
                                    "source_checkpoint": str(checkpoint.resolve()),
                                    "target_task": target_task,
                                    "evaluation_seed": evaluation_seed,
                                    "episode_index": episode_index,
                                    "training_frames": int(source.total_frames),
                                    **row,
                                }
                                output.write(json.dumps(record) + "\n")
                                cell.append(row)
                        matrix.setdefault(source_task, {}).setdefault(target_task, []).append(
                            {
                                "source_training_seed": training_seed,
                                "episodes": len(cell),
                                "team_success_rate": sum(row["success"] for row in cell)
                                / len(cell),
                                "mean_return": sum(row["return"] for row in cell)
                                / len(cell),
                                "contact_episode_fraction": sum(
                                    row["contact"] for row in cell
                                )
                                / len(cell),
                            }
                        )
                finally:
                    source.close()
            print(f"Evaluated source {source_task} ({checkpoint_index + 1}/{len(args.checkpoint)})")

        scratch_tasks = {task for task, _ in args.scratch_checkpoint}
        if args.scratch_checkpoint and scratch_tasks != set(TASK_GOALS):
            raise SystemExit("For transfer gain, provide a scratch checkpoint for every target task")
        for scratch_index, (scratch_task, checkpoint) in enumerate(args.scratch_checkpoint):
            with tempfile.TemporaryDirectory(prefix="scratch-target-matrix-") as temp:
                scratch = load_source_experiment(checkpoint, temp)
                try:
                    if scratch.task.name != scratch_task:
                        raise SystemExit(f"Scratch checkpoint labelled {scratch_task} contains {scratch.task.name}")
                    for evaluation_seed in seeds:
                        rows, _ = probe_policy(scratch, scratch_task, evaluation_seed, args.episodes_per_seed)
                        for episode_index, row in enumerate(rows):
                            output.write(json.dumps({
                                "source_task": "scratch",
                                "source_training_seed": int(scratch.seed),
                                "target_task": scratch_task,
                                "evaluation_seed": evaluation_seed,
                                "episode_index": episode_index,
                                "training_frames": int(scratch.total_frames),
                                "is_scratch_baseline": True,
                                **row,
                            }) + "\n")
                finally:
                    scratch.close()
            print(f"Evaluated scratch baseline {scratch_task} ({scratch_index + 1}/{len(args.scratch_checkpoint)})")

    manifest = {
        "method": "zero_shot_source_target_evaluation",
        "tasks": list(TASK_GOALS),
        "evaluation_seeds": seeds,
        "episodes_per_seed": args.episodes_per_seed,
        "scratch_baseline_checkpoints": {task: str(path.resolve()) for task, path in args.scratch_checkpoint},
        "test_split": "dedicated 900000+ evaluation seeds; same sets for every source-target pair",
        "records": str(records_path),
        "matrix": matrix,
        "scope": "Source policies are evaluated before target fine-tuning. A gain matrix is available only when scratch checkpoints are supplied for every target task; post-fine-tuning transfer is not included.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Per-episode records: {records_path}")
    print(f"Matrix summary: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
