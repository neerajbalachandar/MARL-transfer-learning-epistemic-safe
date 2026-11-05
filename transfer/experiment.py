"""Small BenchMARL helpers for training and actor-only checkpoint transfer."""

from pathlib import Path

import torch
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from transfer.navigation import make_task
from evaluation.success_rate import goal_hits


def make_experiment(
    task_name,
    seed,
    frames,
    output_dir,
    device="cpu",
    callbacks=None,
    experiment_class=Experiment,
    experiment_kwargs=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = device
    config.train_device = device
    config.buffer_device = device
    config.share_policy_params = False
    config.prefer_continuous_actions = True
    config.max_n_frames = frames
    config.evaluation = False
    config.render = False
    config.loggers = ["csv"]
    config.create_json = False
    config.save_folder = str(Path(output_dir).resolve())
    config.checkpoint_at_end = True
    config.checkpoint_interval = 0

    algorithm = MappoConfig.get_from_yaml()
    algorithm.share_param_critic = True
    model = MlpConfig.get_from_yaml()
    critic = MlpConfig.get_from_yaml()
    return experiment_class(
        task=make_task(task_name),
        algorithm_config=algorithm,
        model_config=model,
        critic_model_config=critic,
        seed=seed,
        config=config,
        callbacks=callbacks,
        **(experiment_kwargs or {}),
    )


def copy_actor_weights(source_experiment, target_experiment):
    """Copy only policy parameters after checking their names and dimensions."""
    source_policy = source_experiment.policy
    target_policy = target_experiment.policy
    assert_policy_compatible(source_policy, target_policy)
    target_policy.load_state_dict(source_policy.state_dict())


def assert_policy_compatible(source_policy, target_policy):
    """Check actor parameter names and dimensions before reusing a policy."""
    source = source_policy.state_dict()
    target = target_policy.state_dict()
    if source.keys() != target.keys():
        missing = sorted(source.keys() ^ target.keys())
        raise ValueError(f"Actor policy structures differ: {missing[:5]}")
    for name in source:
        source_value, target_value = source[name], target[name]
        if torch.is_tensor(source_value) and torch.is_tensor(target_value):
            compatible = source_value.shape == target_value.shape
        else:
            compatible = type(source_value) is type(target_value) and source_value == target_value
        if not compatible:
            raise ValueError(
                f"Actor state entry {name} differs: {source_value!r} vs {target_value!r}"
            )


def load_source_experiment(checkpoint, temporary_dir):
    """Load a BenchMARL checkpoint into a temporary folder for actor reuse."""
    Path(temporary_dir).mkdir(parents=True, exist_ok=True)
    return Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "save_folder": str(temporary_dir),
            "loggers": [],
            "create_json": False,
            "evaluation": False,
            "render": False,
        },
    )


def checkpoint_hash(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_policy(experiment, task_name, seed, episodes, observation_limit=None):
    """Run a frozen policy on a task and return simple episode signals and observations."""
    task = make_task(task_name)
    device = experiment.config.sampling_device
    raw = task.get_env_fun(episodes, True, seed, device)()
    env = TransformedEnv(raw, Compose(*task.get_env_transforms(raw))).to(device)
    env = experiment.algorithm.process_env_fun(lambda: env)()
    env.set_seed(seed)
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            rollout = env.rollout(
                max_steps=100,
                policy=experiment.policy,
                auto_cast_to_device=True,
                break_when_any_done=False,
            )
        records, observations = [], []
        for episode in rollout.unbind(0):
            obs = episode.get(("next", "agents", "observation"))
            positions = obs[..., :2]
            goals = positions - obs[..., 4:6]
            hits = goal_hits(positions, goals, [0.1, 0.1, 0.1])
            reward = episode.get(("next", "agents", "reward"))
            contact = False
            collision_onsets = 0
            minimum_clearance = float("inf")
            for first in range(positions.shape[-2]):
                for second in range(first + 1, positions.shape[-2]):
                    clearance = torch.linalg.vector_norm(
                        positions[..., first, :] - positions[..., second, :], dim=-1
                    ) - 0.2
                    contact = contact or bool((clearance <= 0).any().item())
                    minimum_clearance = min(minimum_clearance, float(clearance.min().item()))
                    overlaps = clearance <= 0
                    collision_onsets += int(
                        (overlaps & ~torch.nn.functional.pad(overlaps[:-1], (1, 0))).sum().item()
                    )
            records.append(
                {
                    "return": float(reward.mean(dim=-2).sum().item()),
                    "success": float(hits.all(dim=-1).any().item()),
                    "agent_success": hits.any(dim=0).to(torch.float32).mean(dim=0).tolist(),
                    "contact": float(contact),
                    "collision_onsets": collision_onsets,
                    "minimum_surface_clearance": minimum_clearance,
                    "length": int(obs.shape[0]),
                }
            )
            observations.append(obs.reshape(-1, obs.shape[-1]).detach().cpu())
        all_observations = torch.cat(observations, dim=0)
        if observation_limit is not None:
            all_observations = all_observations[:observation_limit]
        return records, all_observations
    finally:
        env.close()


def summarize_signals(records):
    """Mean and sample variance model for return plus Bernoulli success/contact."""
    returns = torch.tensor([row["return"] for row in records], dtype=torch.float64)
    return {
        "return_mean": float(returns.mean()),
        "return_variance": float(returns.var(unbiased=False).clamp_min(1e-3)),
        "success_probability": float(sum(row["success"] for row in records) / len(records)),
        "contact_probability": float(sum(row["contact"] for row in records) / len(records)),
    }


def summarize_probe(records):
    """Return small evaluation metrics suitable for a run manifest."""
    return {
        "team_success_rate": sum(row["success"] for row in records) / len(records),
        "agent_goal_success_rate": [
            sum(row["agent_success"][agent] for row in records) / len(records)
            for agent in range(len(records[0]["agent_success"]))
        ],
        "mean_return": sum(row["return"] for row in records) / len(records),
        "mean_episode_length": sum(row["length"] for row in records) / len(records),
        "contact_episode_fraction": sum(row["contact"] for row in records) / len(records),
        "mean_collision_onsets": sum(row["collision_onsets"] for row in records) / len(records),
        "mean_minimum_surface_clearance": sum(row["minimum_surface_clearance"] for row in records) / len(records),
        "collision_measure": "physical contact when pairwise surface clearance <= 0",
        "vmas_penalty_threshold": "Navigation min_collision_distance=0.005 center distance",
        "episodes": len(records),
    }
