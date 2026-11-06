#!/usr/bin/env python3
"""Train T2 with a small MAPTF-style frozen-peer imitation pilot."""

import argparse
import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transfer.experiment import (
    checkpoint_hash,
    load_source_experiment,
    make_experiment,
    probe_policy,
    summarize_probe,
    write_episode_records,
    write_run_manifest,
)
from transfer.maptf import MAPTFExperiment
from transfer.navigation import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint", type=Path, help="Competent T2 MAPPO checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=600_000)
    parser.add_argument("--transfer-weight", type=float, default=0.1)
    parser.add_argument("--condition", choices=("adaptive", "fixed"), default="adaptive")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=120_000)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--source-eval-episodes", type=int, default=100)
    parser.add_argument("--source-success-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output = args.output or Path("outputs/maptf") / f"seed_{args.seed}"
    method = EXPERIMENTS["maptf"]
    checkpoint = args.source_checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Source checkpoint not found: {checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="maptf-source-") as temp_dir:
        source = load_source_experiment(checkpoint, temp_dir)
        try:
            if source.task.name != method["source_task"]:
                raise SystemExit(
                    "MAPTF source checkpoint must be trained on "
                    f"{method['source_task']}, found {source.task.name}"
                )
            source_rows, _ = probe_policy(
                source, method["source_task"], 700_000 + args.seed, args.source_eval_episodes
            )
            source_evaluation = summarize_probe(source_rows)
            if source_evaluation["team_success_rate"] < args.source_success_threshold:
                raise SystemExit(
                    "T2 source policy did not reach the predeclared success threshold "
                    f"({source_evaluation['team_success_rate']:.3f} < {args.source_success_threshold:.3f}); "
                    "train a competent source checkpoint first."
                )
            experiment = make_experiment(
                method["target_task"],
                args.seed,
                args.frames,
                args.output,
                args.device,
                experiment_class=MAPTFExperiment,
                experiment_kwargs={
                    "source_policy": source.policy,
                    "transfer_weight": args.transfer_weight,
                    "adaptive": args.condition == "adaptive",
                },
                method=f"maptf_pilot_{args.condition}",
                source_tasks=[method["source_task"]],
                policy_initialization="random; frozen source actors used as peer library",
                evaluation_interval=args.eval_interval,
                evaluation_episodes=args.validation_episodes,
            )
            manifest = {
                "method": f"MAPTF-style {args.condition} peer imitation pilot",
                "task": method["target_task"],
                "source_checkpoint": str(checkpoint),
                "source_sha256": checkpoint_hash(checkpoint),
                "source_agents": [0, 1, 2],
                "receiver_agents": [0, 1, 2],
                "transfer_weight": args.transfer_weight,
                "condition": args.condition,
                "sources_frozen": True,
                "option_value": "online moving-average team-return bandit",
                "termination": "learned logistic reward-change classifier, checked per update batch",
                "simplification": "MAPTF-style pilot; paper SRO/value updates and temporally extended environment options are not implemented",
                "seed": args.seed,
                "frames": args.frames,
                "source_pretraining_frames": int(source.total_frames),
                "source_validation": source_evaluation,
                "source_validation_frames": args.source_eval_episodes * 100,
                "source_success_threshold": args.source_success_threshold,
            }
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            run_completed = False
            try:
                experiment.run()
                run_completed = True
                manifest["selected_peer_by_frame"] = {
                    str(frame): peers for frame, peers in experiment.selected_sources.items()
                }
                selection_rows = [row for row in experiment.option_log if "termination_probability" in row]
                duration_rows = [row["duration_updates"] for row in experiment.option_log if "duration_updates" in row]
                manifest["option_selection_frequency"] = {
                    str(receiver): {
                        str(source): sum(
                            row["receiver"] == receiver and row["source"] == source
                            for row in selection_rows
                        )
                        for source in range(3) if source != receiver
                    }
                    for receiver in range(3)
                }
                manifest["mean_termination_probability"] = (
                    sum(row["termination_probability"] for row in selection_rows) / len(selection_rows)
                    if selection_rows else None
                )
                manifest["option_duration_updates"] = duration_rows
                manifest["option_values"] = experiment.option_values.detach().cpu().tolist()
                manifest["termination_head"] = {
                    "weight": experiment.term_weight.detach().cpu().tolist(),
                    "bias": experiment.term_bias.detach().cpu().tolist(),
                    "trained": args.condition == "adaptive",
                }
                evaluation_rows, _ = probe_policy(
                    experiment, method["target_task"], 900_000 + args.seed, args.eval_episodes
                )
                manifest["final_evaluation"] = summarize_probe(evaluation_rows)
                write_episode_records(
                    args.output / "evaluation" / "test_episodes.jsonl",
                    evaluation_rows,
                    method=f"maptf_pilot_{args.condition}",
                    task=method["target_task"],
                    training_seed=args.seed,
                    evaluation_seed=900_000 + args.seed,
                    environment_frames=int(experiment.total_frames),
                    split="test",
                )
                write_run_manifest(experiment, completed=True, extra=manifest)
                manifest["target_training_frames"] = int(experiment.total_frames)
                manifest["held_out_evaluation_frames"] = args.eval_episodes * 100
                manifest["total_target_interaction_frames"] = (
                    manifest["target_training_frames"]
                    + manifest["held_out_evaluation_frames"]
                )
                manifest["total_end_to_end_frames"] = (
                    manifest["source_pretraining_frames"]
                    + manifest["source_validation_frames"]
                    + manifest["total_target_interaction_frames"]
                )
                (args.output / "evaluation.json").write_text(
                    json.dumps(manifest["final_evaluation"], indent=2) + "\n"
                )
                if checkpoint_hash(checkpoint) != manifest["source_sha256"]:
                    raise RuntimeError("Source checkpoint changed during the transfer run")
                (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            finally:
                if not run_completed:
                    experiment.close()
        finally:
            source.close()


if __name__ == "__main__":
    main()
