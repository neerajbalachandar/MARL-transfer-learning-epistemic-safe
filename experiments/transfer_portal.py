#!/usr/bin/env python3
"""Train a compact PORTAL-style automatic curriculum on fixed navigation tasks."""

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transfer.experiment import (
    checkpoint_hash,
    copy_actor_weights,
    make_experiment,
    probe_policy,
    summarize_probe,
)
from transfer.methods import moderate_candidates, portal_gmm_similarity
from transfer.navigation import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("portal", "difficulty", "manual", "scratch"), default="portal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames-per-stage", type=int, default=600_000)
    parser.add_argument("--probe-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--competence-episodes", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument(
        "--difficulty-band",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Use pilot-frozen moderate return bounds for main seeds",
    )
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    method = EXPERIMENTS["portal"]
    args.output = args.output or Path("outputs/portal") / f"seed_{args.seed}"
    if args.probe_episodes < args.components:
        raise SystemExit("Use at least as many probe episodes as GMM components")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "scratch":
        task_sequence = list(method["scratch_sequence"])
    elif args.mode == "manual":
        task_sequence = list(method["manual_sequence"])
    else:
        task_sequence = [method["initial_task"]]

    history = []
    candidates = list(method["candidate_tasks"])
    previous = None
    stage = 0
    previous_checkpoint = None
    while task_sequence:
        task_name = task_sequence.pop(0)
        stage_dir = args.output / f"stage_{stage}_{task_name}"
        experiment = make_experiment(
            task_name, args.seed, args.frames_per_stage, stage_dir, args.device
        )
        if previous is not None:
            copy_actor_weights(previous, experiment)
        experiment.run()
        stage_info = {
            "task": task_name,
            "training_frames": int(experiment.total_frames),
            "source_checkpoint": str(previous_checkpoint) if previous_checkpoint else None,
            "source_checkpoint_sha256": checkpoint_hash(previous_checkpoint) if previous_checkpoint else None,
        }
        competence_rows, _ = probe_policy(
            experiment, task_name, 700_000 + args.seed + stage, args.competence_episodes
        )
        stage_info["competence_evaluation"] = summarize_probe(competence_rows)
        stage_info["competent_at_cap"] = (
            stage_info["competence_evaluation"]["team_success_rate"] >= args.success_threshold
        )
        stage_info["success_threshold"] = args.success_threshold
        stage_info["competence_evaluation_frames"] = args.competence_episodes * 100

        if task_name == method["target_task"]:
            previous_checkpoint = experiment.folder_name / "checkpoints" / f"checkpoint_{experiment.total_frames}.pt"
            stage_info["checkpoint"] = str(previous_checkpoint)
            stage_info["checkpoint_sha256"] = checkpoint_hash(previous_checkpoint)
            previous = experiment
            history.append(stage_info)
            break

        if args.mode in ("manual", "scratch"):
            previous = experiment
            history.append(stage_info)
            stage += 1
            continue

        scores, success_rates, confidence_intervals, observation_sets = {}, {}, {}, {}
        for candidate in candidates:
            rows, observations = probe_policy(
                experiment,
                candidate,
                300_000 + args.seed,
                args.probe_episodes,
                observation_limit=10_000,
            )
            scores[candidate] = sum(row["return"] for row in rows) / len(rows)
            success_rates[candidate] = sum(row["success"] for row in rows) / len(rows)
            returns = [row["return"] for row in rows]
            if len(returns) > 1:
                variance = sum((value - scores[candidate]) ** 2 for value in returns) / (len(returns) - 1)
                margin = 1.96 * math.sqrt(variance / len(returns))
            else:
                margin = 0.0
            confidence_intervals[candidate] = [scores[candidate] - margin, scores[candidate] + margin]
            observation_sets[candidate] = observations
        moderate, interval = moderate_candidates(scores, args.difficulty_band)
        similarity = {}
        similarity_by_components = {}
        if args.mode == "portal":
            _, target_observation_samples = probe_policy(
                experiment,
                method["target_task"],
                400_000 + args.seed,
                args.probe_episodes,
                observation_limit=10_000,
            )
            target_observations = target_observation_samples.numpy()
            for candidate in moderate:
                similarity_by_components[candidate] = {}
                for count in sorted(set((3, 5, 8, args.components))):
                    if min(len(observation_sets[candidate]), len(target_observations)) < count:
                        continue
                    score = portal_gmm_similarity(
                        observation_sets[candidate].numpy(),
                        target_observations,
                        components=count,
                    )
                    similarity_by_components[candidate][str(count)] = score
                    if count == args.components:
                        similarity[candidate] = score
            selected = max(moderate, key=lambda name: similarity[name])
        else:
            midpoint = (interval["lower"] + interval["upper"]) / 2
            selected = min(moderate, key=lambda name: abs(scores[name] - midpoint))

        stage_info.update(
            {
                "jumpstart_return": scores,
                "team_success_rate": success_rates,
                "jumpstart_return_95pct_ci": confidence_intervals,
                "moderate_candidates": moderate,
                "difficulty_interval": interval,
                "similarity": similarity,
                "similarity_sensitivity_by_gmm_components": similarity_by_components,
                "selected_task": selected,
                "probe_episodes_per_task": args.probe_episodes,
                "probe_frames": len(candidates) * args.probe_episodes * 100,
                "target_distribution_probe_frames": (
                    args.probe_episodes * 100 if args.mode == "portal" else 0
                ),
                "gmm_components": args.components if args.mode == "portal" else None,
            }
        )
        history.append(stage_info)
        previous_checkpoint = experiment.folder_name / "checkpoints" / f"checkpoint_{experiment.total_frames}.pt"
        stage_info["checkpoint"] = str(previous_checkpoint)
        stage_info["checkpoint_sha256"] = checkpoint_hash(previous_checkpoint)
        previous = experiment
        if selected == method["target_task"]:
            task_sequence = [method["target_task"]]
        else:
            candidates.remove(selected)
            task_sequence = [selected]
        stage += 1

    manifest = {
        "method": args.mode,
        "seed": args.seed,
        "stage_frame_cap": args.frames_per_stage,
        "difficulty_band": args.difficulty_band,
        "target_task": method["target_task"],
        "stages": history,
        "total_training_frames": sum(row["training_frames"] for row in history),
        "total_probe_frames": sum(
            row.get("probe_frames", 0) + row.get("target_distribution_probe_frames", 0)
            for row in history
        ),
        "total_competence_evaluation_frames": sum(
            row.get("competence_evaluation_frames", 0) for row in history
        ),
        "scope": "Fixed observation/action dimensions; ERFN is not used or claimed.",
        "limitations": "This pilot runs fixed stage budgets and records task scores; it does not implement the paper's two-consecutive-evaluation competence stopping rule or aligned-component adaptation procedure beyond Hungarian mean alignment.",
    }
    if previous is not None:
        final_rows, _ = probe_policy(
            previous, method["target_task"], 900_000 + args.seed, args.eval_episodes
        )
        manifest["final_evaluation"] = summarize_probe(final_rows)
        manifest["held_out_evaluation_frames"] = args.eval_episodes * 100
        manifest["total_target_training_frames"] = sum(
            row["training_frames"] for row in history if row["task"] == method["target_task"]
        )
        manifest["total_interaction_frames"] = (
            manifest["total_training_frames"]
            + manifest["total_probe_frames"]
            + manifest["total_competence_evaluation_frames"]
            + manifest["held_out_evaluation_frames"]
        )
        (args.output / "evaluation.json").write_text(
            json.dumps(manifest["final_evaluation"], indent=2) + "\n"
        )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
