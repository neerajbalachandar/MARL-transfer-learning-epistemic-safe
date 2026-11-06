#!/usr/bin/env python3
"""Probe, select, and fine-tune a source policy using a BPR-style posterior."""

import argparse
import json
import random
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from transfer.experiment import (
    checkpoint_hash,
    copy_actor_weights,
    load_source_experiment,
    make_experiment,
    probe_policy,
    summarize_probe,
    summarize_signals,
    write_episode_records,
    write_run_manifest,
)
from transfer.methods import (
    bayes_update,
    belief_information_gain,
    bpr_expected_improvement,
    calibrated_bayes_update,
    joint_signal_log_likelihood,
    signal_log_likelihood,
    summarize_joint_signals,
)
from transfer.navigation import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_t1a", type=Path)
    parser.add_argument("source_t1b", type=Path)
    parser.add_argument("--selection", choices=("bayes", "random", "fixed-a", "fixed-b", "oracle"), default="bayes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-episodes", type=int, default=20)
    parser.add_argument("--frames", type=int, default=600_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=120_000)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--validation-seed", type=int, default=None)
    parser.add_argument("--likelihood", choices=("factorized", "joint"), default="factorized")
    parser.add_argument("--source-eval-episodes", type=int, default=100)
    parser.add_argument("--source-success-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    method = EXPERIMENTS["bayesian_reuse"]
    source_tasks = method["source_tasks"]
    args.output = args.output or Path("outputs/bayesian_reuse") / f"seed_{args.seed}"
    checkpoints = [args.source_t1a.resolve(), args.source_t1b.resolve()]
    if any(not path.is_file() for path in checkpoints):
        raise SystemExit("Both source checkpoint files must exist")
    if args.probe_episodes < 1:
        raise SystemExit("--probe-episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    experiments, temp_dirs = [], []
    try:
        for index, checkpoint in enumerate(checkpoints):
            temp = tempfile.TemporaryDirectory(prefix=f"bpr-source-{index}-")
            temp_dirs.append(temp)
            experiment = load_source_experiment(checkpoint, temp.name)
            experiments.append(experiment)
            expected_task = source_tasks[index]
            if experiment.task.name != expected_task:
                raise SystemExit(
                    f"Source {index} must be trained on {expected_task}, found {experiment.task.name}"
                )

        # Estimate P(signal | transfer affinity, selected source policy) from
        # held-out T1a/T1b episodes. The final target probes use separate seeds.
        models = []
        joint_models = []
        for source_index, experiment in enumerate(experiments):
            policy_models = []
            policy_joint_models = []
            for hypothesis_index, task in enumerate(source_tasks):
                rows, _ = probe_policy(
                    experiment, task, 100_000 + args.seed + hypothesis_index,
                    args.probe_episodes,
                )
                policy_models.append(summarize_signals(rows))
                policy_joint_models.append(summarize_joint_signals(rows))
            models.append(policy_models)
            joint_models.append(policy_joint_models)
        source_success = []
        for source_index, task in enumerate(source_tasks):
            rows, _ = probe_policy(
                experiments[source_index], task, 700_000 + args.seed + source_index,
                args.source_eval_episodes,
            )
            source_success.append(summarize_probe(rows)["team_success_rate"])
        if min(source_success) < args.source_success_threshold:
            raise SystemExit(
                "A source policy failed its source-task competence check: "
                f"success rates={source_success}, required={args.source_success_threshold:.3f}. "
                "Train that source task longer before probing transfer."
            )

        probes = []
        belief = torch.tensor([0.5, 0.5], dtype=torch.float64)
        for source_index, experiment in enumerate(experiments):
            rows, _ = probe_policy(
                experiment, method["target_task"], 200_000 + args.seed, args.probe_episodes
            )
            for episode_index, row in enumerate(rows):
                likelihood_fn = (
                    signal_log_likelihood
                    if args.likelihood == "factorized"
                    else joint_signal_log_likelihood
                )
                selected_models = models if args.likelihood == "factorized" else joint_models
                likelihood = torch.stack(
                    [likelihood_fn(row, selected_models[source_index][h]) for h in range(2)]
                ).to(torch.float64)
                prior = belief.clone()
                belief = calibrated_bayes_update(belief, likelihood)
                probes.append(
                    {
                        "source": source_index,
                        "episode": episode_index,
                        "signal": row,
                        "prior": prior.tolist(),
                        "log_likelihood": likelihood.tolist(),
                        "posterior": belief.tolist(),
                        "information_gain_nats": belief_information_gain(prior, belief),
                        "probe_budget_episodes": len(probes) + 1,
                    }
                )

        mean_utility = torch.tensor(
            [[models[policy][hypothesis]["return_mean"] for policy in range(2)] for hypothesis in range(2)],
            dtype=torch.float64,
        )
        utility_std = torch.tensor(
            [[models[policy][hypothesis]["return_variance"] ** 0.5 for policy in range(2)] for hypothesis in range(2)],
            dtype=torch.float64,
        )
        if args.selection == "bayes":
            selected = bpr_expected_improvement(belief, mean_utility, utility_std)
        elif args.selection == "random":
            selected = random.Random(args.seed).randrange(2)
        elif args.selection == "fixed-a":
            selected = 0
        elif args.selection == "fixed-b":
            selected = 1
        else:
            oracle_returns = []
            for experiment in experiments:
                oracle_rows, _ = probe_policy(
                    experiment, method["target_task"], 500_000 + args.seed, args.probe_episodes
                )
                oracle_returns.append(
                    sum(row["return"] for row in oracle_rows) / len(oracle_rows)
                )
            selected = max(range(2), key=lambda index: oracle_returns[index])

        target = make_experiment(
            method["target_task"],
            args.seed,
            args.frames,
            args.output,
            args.device,
            method=f"bpr_{args.selection}_{args.likelihood}",
            source_tasks=source_tasks,
            policy_initialization=f"actor_only:{source_tasks[selected]}",
            evaluation_interval=args.eval_interval,
            evaluation_episodes=args.validation_episodes,
            validation_seed=args.validation_seed,
        )
        run_completed = False
        try:
            copy_actor_weights(experiments[selected], target)
            manifest = {
                "method": f"bpr_{args.selection}_{args.likelihood}",
                "selection_condition": args.selection,
                "task": method["target_task"],
                "selected_source": source_tasks[selected],
                "transfer_mapping": {
                    "source_actors": [0, 1, 2],
                    "target_actors": [0, 1, 2],
                    "parameters": "actor only; target critic and optimizer start fresh",
                },
                "source_checkpoints": [str(path) for path in checkpoints],
                "source_sha256": [checkpoint_hash(path) for path in checkpoints],
                "source_critics_transferred": False,
                "prior": [0.5, 0.5],
                "final_posterior": belief.tolist(),
                "validation_models": models,
                "joint_validation_models": joint_models,
                "likelihood_model": args.likelihood,
                "likelihood_misspecification_note": (
                    "factorized Gaussian/Bernoulli baseline assumes conditional independence"
                    if args.likelihood == "factorized"
                    else "joint categorical success/contact and conditional Gaussian return model"
                ),
                "probe_episodes_per_candidate": args.probe_episodes,
                "probe_frames": 2 * args.probe_episodes * 100,
                "oracle_validation_frames": (
                    2 * args.probe_episodes * 100 if args.selection == "oracle" else 0
                ),
                "oracle_validation_returns": oracle_returns if args.selection == "oracle" else None,
                "source_validation_frames": (
                    2 * 2 * args.probe_episodes * 100
                    + 2 * args.source_eval_episodes * 100
                ),
                "source_task_success_rates": source_success,
                "source_success_threshold": args.source_success_threshold,
                "source_pretraining_frames": [int(exp.total_frames) for exp in experiments],
                "target_training_frames": args.frames,
                "probe_history": probes,
                "seed": args.seed,
                "oracle_note": "Oracle uses a separate T2 validation seed split and is not a deployable selector." if args.selection == "oracle" else None,
            }
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            target.run()
            run_completed = True
            final_records, _ = probe_policy(
                target, method["target_task"], 900_000 + args.seed, args.eval_episodes
            )
            manifest["final_evaluation"] = summarize_probe(final_records)
            write_episode_records(
                args.output / "evaluation" / "test_episodes.jsonl",
                final_records,
                method=manifest["method"],
                task=method["target_task"],
                training_seed=args.seed,
                evaluation_seed=900_000 + args.seed,
                environment_frames=int(target.total_frames),
                split="test",
            )
            write_run_manifest(target, completed=True, extra=manifest)
            manifest["target_training_frames"] = int(target.total_frames)
            manifest["held_out_evaluation_frames"] = args.eval_episodes * 100
            manifest["total_target_interaction_frames"] = (
                manifest["probe_frames"]
                + manifest["oracle_validation_frames"]
                + manifest["target_training_frames"]
                + manifest["held_out_evaluation_frames"]
            )
            manifest["posterior_entropy"] = float(
                -(belief * belief.clamp_min(1e-12).log()).sum()
            )
            manifest["total_end_to_end_frames"] = (
                sum(manifest["source_pretraining_frames"])
                + manifest["source_validation_frames"]
                + manifest["total_target_interaction_frames"]
            )
            (args.output / "evaluation.json").write_text(
                json.dumps(manifest["final_evaluation"], indent=2) + "\n"
            )
            (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if [checkpoint_hash(path) for path in checkpoints] != manifest["source_sha256"]:
                raise RuntimeError("A source checkpoint changed during policy reuse")
        finally:
            if not run_completed:
                target.close()
    finally:
        for experiment in experiments:
            experiment.close()
        for temp in temp_dirs:
            temp.cleanup()


if __name__ == "__main__":
    main()
