#!/usr/bin/env python3
"""Evaluate a BenchMARL checkpoint on independently seeded VMAS episodes."""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import torch
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarl.experiment.experiment import Experiment
from evaluation.metrics import summarize_episodes
from evaluation.success_rate import goal_hits


def load_experiment(checkpoint, save_folder):
    """Restore the checkpoint into an isolated folder, leaving its run untouched."""
    return Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "save_folder": str(save_folder),
            "loggers": [],
            "create_json": False,
        },
    )


def make_evaluation_env(experiment, episodes, seed):
    """Build the same BenchMARL task environment with a requested batch size."""
    raw_env = experiment.task.get_env_fun(
        num_envs=episodes,
        continuous_actions=experiment.continuous_actions,
        seed=seed,
        device=experiment.config.sampling_device,
    )()
    transforms = Compose(*experiment.task.get_env_transforms(raw_env))
    env = TransformedEnv(raw_env, transforms).to(experiment.config.sampling_device)
    env = experiment.algorithm.process_env_fun(lambda: env)()
    env.set_seed(seed)
    return env


def first_end_index(trajectory):
    for key in (("next", "terminated"), ("next", "truncated"), ("next", "done")):
        ended = trajectory.get(key, None)
        if ended is not None and ended.any():
            return int(ended.squeeze(-1).nonzero()[0].item()) + 1
    return trajectory.batch_size[0]


def evaluate_seed(experiment, seed, episode_count, render=False):
    env = make_evaluation_env(experiment, episode_count, seed)
    scenario = env.base_env.scenario
    # Navigation.done() uses each agent's sphere radius as its goal threshold.
    radii = [agent.shape.radius for agent in scenario.world.agents]
    positions_for_video = []
    trails = [[] for _ in scenario.world.agents]

    def render_frame(render_env, data):
        from PIL import Image, ImageDraw
        import numpy as np

        frame = experiment.task.__class__.render_callback(experiment, render_env, data)
        image = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
        image = Image.fromarray(image.astype("uint8"), mode="RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        world = scenario.world
        x_limit = world.x_semidim or 1.1
        y_limit = world.y_semidim or 1.1

        def pixel(position):
            x, y = position
            return (
                (float(x) / (2 * x_limit) + 0.5) * width,
                (0.5 - float(y) / (2 * y_limit)) * height,
            )

        current_positions = []
        colors = [(56, 126, 184), (230, 126, 34), (65, 155, 85), (180, 90, 160)]
        for index, agent in enumerate(world.agents):
            position = agent.state.pos[0].detach().cpu().tolist()
            point = pixel(position)
            trails[index].append(point)
            current_positions.append(position)
            if len(trails[index]) > 1:
                draw.line(trails[index], fill=colors[index % len(colors)], width=2)

        for first in range(len(current_positions)):
            for second in range(first + 1, len(current_positions)):
                separation = torch.linalg.vector_norm(
                    torch.tensor(current_positions[first])
                    - torch.tensor(current_positions[second])
                ).item()
                if separation <= scenario.min_collision_distance:
                    middle = pixel(
                        [
                            (a + b) / 2
                            for a, b in zip(
                                current_positions[first], current_positions[second]
                            )
                        ]
                    )
                    draw.ellipse(
                        (
                            middle[0] - 8,
                            middle[1] - 8,
                            middle[0] + 8,
                            middle[1] + 8,
                        ),
                        outline="red",
                        width=3,
                    )

        positions_for_video.append(np.asarray(image))
        return frame

    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            trajectory = env.rollout(
                max_steps=experiment.max_steps,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
                callback=render_frame if render else None,
            )

        episodes = []
        for episode_trajectory in trajectory.unbind(0):
            length = first_end_index(episode_trajectory)
            data = episode_trajectory[:length]
            observations = data.get(("next", "agents", "observation"))
            positions = observations[..., :2]
            goals = positions - observations[..., 4:6]
            hits = goal_hits(positions, goals, radii)
            initial_observation = data.get(("agents", "observation"))[:1]
            initial_position = initial_observation[..., :2]
            initial_goal = initial_position - initial_observation[..., 4:6]
            positions = torch.cat((initial_position, positions), dim=0)
            hits = torch.cat((goal_hits(initial_position, initial_goal, radii), hits), dim=0)
            rewards = data.get(("next", "agents", "reward"))
            truncated = data.get(("next", "truncated"), None)
            explicit_timeout = (
                bool(truncated[-1].any().item()) if truncated is not None else False
            )
            team_success = bool(hits.all(dim=-1).any().item())
            timed_out = explicit_timeout or (
                length >= experiment.max_steps and not team_success
            )
            episodes.append(
                {
                    "positions": positions.detach().cpu(),
                    "goal_hits": hits.detach().cpu(),
                    "team_success": team_success,
                    "return": float(rewards.mean(dim=-2).sum().item()),
                    "length": length,
                    "timeout": timed_out,
                }
            )

        if positions_for_video:
            from PIL import Image, ImageDraw
            import numpy as np

            final = Image.fromarray(positions_for_video[-1])
            draw = ImageDraw.Draw(final)
            result = "Episode complete" if episodes[0]["team_success"] else "Episode timeout"
            draw.rectangle((5, 5, 180, 28), fill="white")
            draw.text((10, 9), result, fill="black")
            positions_for_video[-1] = np.asarray(final)

        metrics = summarize_episodes(episodes, scenario.min_collision_distance)
        metrics["seed"] = seed
        metrics["training_frames"] = int(experiment.total_frames)
        return metrics, positions_for_video
    finally:
        env.close()


def save_metrics(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_video(frames, path):
    if not frames:
        raise RuntimeError("VMAS returned no frames; cannot save the requested video")
    import av
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=20)
        for frame in frames:
            image = frame.detach().cpu().numpy() if torch.is_tensor(frame) else frame
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="BenchMARL checkpoint .pt file")
    parser.add_argument(
        "--seeds", default="0", help="Comma-separated independent evaluation seeds"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Episodes per seed (default: saved config)",
    )
    parser.add_argument("--render", action="store_true", help="Save a VMAS rollout video")
    parser.add_argument(
        "--output", type=Path, default=None, help="Metrics CSV output path"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise SystemExit("Provide at least one seed with --seeds")

    run_dir = checkpoint.parents[2]
    output = (
        args.output or run_dir / "evaluation" / f"{checkpoint.stem}_metrics.csv"
    ).resolve()
    with tempfile.TemporaryDirectory(prefix="benchmarl-evaluation-") as temp_dir:
        experiment = load_experiment(checkpoint, temp_dir)
        episode_count = (
            args.episodes
            if args.episodes is not None
            else experiment.config.evaluation_episodes
        )
        if episode_count < 1:
            raise SystemExit("--episodes must be at least 1")

        rows = []
        try:
            for index, seed in enumerate(seeds):
                result, frames = evaluate_seed(
                    experiment, seed, episode_count, render=args.render and index == 0
                )
                rows.append(result)
                print(
                    f"seed={seed}: team_success={result['team_success_rate']:.3f}, "
                    f"collisions={result['collision_events']}, "
                    f"mean_return={result['mean_episode_return']:.3f}"
                )
                if args.render and index == 0:
                    video_path = (
                        ROOT
                        / "results"
                        / "videos"
                        / f"{checkpoint.stem}_seed_{seed}.mp4"
                    )
                    save_video(frames, video_path)
                    print(f"Video saved to {video_path}")
        finally:
            experiment.close()

    save_metrics(rows, output)
    print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
