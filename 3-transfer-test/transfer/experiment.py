"""Shared BenchMARL training, evaluation, and checkpoint helpers."""

import json
import importlib
from pathlib import Path
import subprocess

import torch
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.algorithms import MappoConfig
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.experiment.callback import Callback
from benchmarl.models.mlp import MlpConfig

from transfer.navigation import TASK_CONFIG, make_task
from evaluation.metrics import episode_metrics, summarize_episodes


def make_experiment(
    task_name,
    seed,
    frames,
    output_dir,
    device="cpu",
    callbacks=None,
    experiment_class=Experiment,
    experiment_kwargs=None,
    method="mappo",
    source_tasks=None,
    policy_initialization="random",
    evaluation_interval=120_000,
    evaluation_episodes=32,
    validation_seed=None,
    learning_rate=None,
    ppo_epochs=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig.get_from_yaml()
    config.sampling_device = device
    config.train_device = device
    config.buffer_device = device
    config.share_policy_params = False
    config.prefer_continuous_actions = True
    config.max_n_frames = frames
    config.evaluation = True
    config.render = False
    config.loggers = ["csv"]
    config.create_json = False
    config.evaluation_interval = evaluation_interval
    config.evaluation_episodes = evaluation_episodes
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        config.lr = learning_rate
    if ppo_epochs is not None:
        if ppo_epochs < 1:
            raise ValueError("ppo_epochs must be at least one")
        config.on_policy_n_minibatch_iters = ppo_epochs
    config.evaluation_deterministic_actions = True
    config.evaluation_static = False
    config.save_folder = str(Path(output_dir).resolve())
    config.checkpoint_at_end = True
    config.checkpoint_interval = 0

    algorithm = MappoConfig.get_from_yaml()
    algorithm.share_param_critic = True
    model = MlpConfig.get_from_yaml()
    critic = MlpConfig.get_from_yaml()
    validation_seed = 100_000 + seed if validation_seed is None else validation_seed
    evaluation_callback = EpisodeEvaluationCallback(
        Path(output_dir) / "evaluation" / "validation_episodes.jsonl",
        task_name=task_name,
        validation_seed=validation_seed,
    )
    callbacks = list(callbacks or []) + [evaluation_callback]
    experiment = experiment_class(
        task=make_task(task_name),
        algorithm_config=algorithm,
        model_config=model,
        critic_model_config=critic,
        seed=seed,
        config=config,
        callbacks=callbacks,
        **(experiment_kwargs or {}),
    )
    experiment.test_env.set_seed(validation_seed)
    experiment.run_manifest_path = Path(output_dir) / "manifest.json"
    experiment.run_metadata = {
        "method": method,
        "task": task_name,
        "source_tasks": list(source_tasks or []),
        "target_task": task_name,
        "training_seed": int(seed),
        "validation_seed": int(validation_seed),
        "final_test_seed_base": 900_000,
        "total_environment_frames": int(frames),
        "evaluation_episode_count": int(evaluation_episodes),
        "evaluation_interval_environment_frames": int(evaluation_interval),
        "optimizer_configuration": {
            "learning_rate": float(config.lr),
            "ppo_epochs": int(config.on_policy_n_minibatch_iters),
        },
        "checkpoint_path": str(
            experiment.folder_name / "checkpoints" / f"checkpoint_{frames}.pt"
        ),
        "policy_initialization": policy_initialization,
        "model_configuration": {
            "actor": _plain_config(model),
            "critic": _plain_config(critic),
            "algorithm": _plain_config(algorithm),
        },
        "dependency_revisions": dependency_revisions(),
    }
    write_run_manifest(experiment, completed=False)
    return experiment


def _plain_config(config):
    return {key: str(value) for key, value in vars(config).items()}


def dependency_revisions():
    """Return installed source revisions without changing dependencies."""
    revisions = {"repository": _git_revision(Path(__file__).resolve().parents[1])}
    for name in ("benchmarl", "vmas"):
        module_path = Path(importlib.import_module(name).__file__).resolve()
        revisions[name] = _git_revision(module_path)
    return revisions


def _git_revision(path):
    root = path if path.is_dir() else path.parent
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_run_manifest(experiment, completed=True, extra=None):
    """Write run settings and the final observed environment-frame count."""
    metadata = dict(getattr(experiment, "run_metadata", {}))
    metadata["status"] = "complete" if completed else "running"
    metadata["observed_environment_frames"] = int(experiment.total_frames)
    if extra:
        metadata.update(extra)
    path = Path(getattr(experiment, "run_manifest_path", experiment.folder_name / "manifest.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return path


def write_episode_records(path, records, **metadata):
    """Write raw episode dictionaries as JSON Lines with common run fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as output:
        for record in records:
            output.write(json.dumps({**metadata, **record}, default=str) + "\n")
    return path


class EpisodeEvaluationCallback(Callback):
    """Save one raw record per validation episode at the actual frame count."""

    def __init__(self, output_path, task_name, validation_seed):
        super().__init__()
        self.output_path = Path(output_path)
        self.task_name = task_name
        self.validation_seed = int(validation_seed)
        self.episode_number = 0

    def on_evaluation_end(self, rollouts):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        env = self.experiment.test_env
        radii = [agent.shape.radius for agent in env.base_env.scenario.world.agents]
        with self.output_path.open("a") as output:
            for rollout in rollouts:
                record = episode_metrics(
                    rollout,
                    radii,
                    self.experiment.max_steps,
                    gamma=self.experiment.config.gamma,
                )
                record.update(
                    {
                        "task": self.task_name,
                        "method": getattr(self.experiment, "run_metadata", {}).get(
                            "method", "mappo"
                        ),
                        "training_seed": int(self.experiment.seed),
                        "evaluation_seed": self.validation_seed,
                        "episode_index": self.episode_number,
                        "environment_frames": int(self.experiment.total_frames),
                    }
                )
                output.write(json.dumps(record) + "\n")
                self.episode_number += 1


def copy_actor_weights(source_experiment, target_experiment):
    """Copy only policy parameters after checking their names and dimensions."""
    source_policy = source_experiment.policy
    target_policy = target_experiment.policy
    assert_policy_compatible(source_policy, target_policy)
    target_policy.load_state_dict(source_policy.state_dict())


def copy_actor_mean_weights(source_policy, target_policy, action_dim):
    """Transfer actor features and action means, resetting target scales to 1.

    The final MLP projection is split as ``[loc, scale]`` by BenchMARL's
    NormalParamExtractor. Reusing a trained source scale can make a TanhNormal
    policy nearly deterministic on a changed task; its saturated actions then
    produce extreme PPO log probabilities. Keep source features/means and start
    the target scale projection at the neutral biased-softplus value of 1.
    """
    assert_policy_compatible(source_policy, target_policy)
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    output_layers = []
    for weight_key, weight in source_state.items():
        if not weight_key.endswith(".weight") or weight.ndim < 2:
            continue
        if weight.shape[-2] != action_dim * 2:
            continue
        bias_key = weight_key[:-len("weight")] + "bias"
        bias = source_state.get(bias_key)
        if bias is not None and bias.shape[-1] == action_dim * 2:
            output_layers.append((weight_key, bias_key))
    if len(output_layers) != 1:
        raise ValueError(
            "Could not uniquely identify the actor's final [loc, scale] projection; "
            f"found {len(output_layers)} candidates"
        )
    weight_key, bias_key = output_layers[0]
    with torch.no_grad():
        for key, source_value in source_state.items():
            if key not in (weight_key, bias_key):
                if torch.is_tensor(target_state[key]) and torch.is_tensor(source_value):
                    target_state[key].copy_(source_value)
                else:
                    target_state[key] = source_value
        target_state[weight_key][..., :action_dim, :].copy_(
            source_state[weight_key][..., :action_dim, :]
        )
        target_state[weight_key][..., action_dim:, :].zero_()
        target_state[bias_key][..., :action_dim].copy_(
            source_state[bias_key][..., :action_dim]
        )
        target_state[bias_key][..., action_dim:].zero_()
    target_policy.load_state_dict(target_state)


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
        radii = [agent.shape.radius for agent in env.base_env.scenario.world.agents]
        # This pinned TorchRL/VMAS build returns [parallel_envs, time].
        for episode in rollout.unbind(0):
            obs = episode.get(("next", "agents", "observation"))
            record = episode_metrics(
                episode,
                radii,
                int(TASK_CONFIG["environment"]["max_steps"]),
                gamma=experiment.config.gamma,
            )
            # Keep concise aliases used by the pilot selector manifests.
            record.update(
                {
                    "return": record["undiscounted_team_return"],
                    "success": float(record["team_success"]),
                    "agent_success": record["individual_success"],
                    "contact": float(record["contact_episode"]),
                    "collision_onsets": record["collision_event_count"],
                    "minimum_surface_clearance": record["minimum_surface_clearance"],
                    "length": record["episode_length"],
                }
            )
            records.append(record)
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
    summary = summarize_episodes(records)
    summary.update(
        {
            "agent_goal_success_rate": summary["individual_success_rate"],
            "mean_return": summary["mean_undiscounted_team_return"],
            "mean_minimum_surface_clearance": summary[
                "mean_minimum_surface_clearance"
            ],
            "mean_collision_onsets": summary["collision_events_per_episode"],
        }
    )
    return summary
