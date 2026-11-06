"""Per-episode and aggregate metrics for VMAS navigation evaluations."""

import torch

from evaluation.collision_metrics import navigation_collision_metrics
from evaluation.success_rate import goal_hits


def episode_metrics(rollout, agent_radii, max_steps, gamma=0.99):
    """Convert one VMAS rollout (time, agents, observation) into raw metrics."""
    if gamma < 0 or gamma > 1:
        raise ValueError("gamma must be in [0, 1]")
    current = rollout.get(("agents", "observation"))
    following = rollout.get(("next", "agents", "observation"))
    rewards = rollout.get(("next", "agents", "reward"))
    if current is None or following is None or rewards is None:
        raise ValueError("rollout is missing observations or rewards")
    if current.ndim != 3 or following.ndim != 3:
        raise ValueError("one rollout must have [time, agents, features] observations")

    length = int(following.shape[0])
    positions = torch.cat((current[:1, :, :2], following[:, :, :2]), dim=0)
    goals = torch.cat(
        (
            current[:1, :, :2] - current[:1, :, 4:6],
            following[:, :, :2] - following[:, :, 4:6],
        ),
        dim=0,
    )
    hits = goal_hits(positions, goals, agent_radii)
    team_success = bool(hits.all(dim=-1).any().item())
    per_agent_success = hits.any(dim=0)

    team_rewards = rewards.squeeze(-1).mean(dim=-1)
    discount = torch.pow(
        torch.as_tensor(gamma, dtype=team_rewards.dtype, device=team_rewards.device),
        torch.arange(length, dtype=team_rewards.dtype, device=team_rewards.device),
    )
    collision = navigation_collision_metrics(positions, agent_radii)
    collision_info = rollout.get(("next", "agents", "info", "agent_collisions"), None)
    if collision_info is None:
        reward_steps = 0
    else:
        reward_steps = int((collision_info < 0).any(dim=(-1, -2)).sum().item())
    truncated = rollout.get(("next", "truncated"), None)
    explicit_timeout = bool(truncated[-1].any().item()) if truncated is not None else False
    return {
        "team_success": team_success,
        "individual_success": [bool(value) for value in per_agent_success.tolist()],
        "undiscounted_team_return": float(team_rewards.sum().item()),
        "discounted_team_return": float((team_rewards * discount).sum().item()),
        "discount_gamma": float(gamma),
        "episode_length": length,
        "timeout": bool(explicit_timeout or (length >= max_steps and not team_success)),
        "trajectory_positions": positions.detach().cpu().tolist(),
        "trajectory_goals": goals.detach().cpu().tolist(),
        "simulator_collision_reward_steps": reward_steps,
        **collision,
    }


def summarize_episodes(episodes):
    """Aggregate raw episode records without discarding the source records."""
    if not episodes:
        raise ValueError("At least one episode is required")
    count = len(episodes)
    agent_count = len(episodes[0]["individual_success"])
    return {
        "episodes": count,
        "team_success_rate": sum(row["team_success"] for row in episodes) / count,
        "individual_success_rate": [
            sum(row["individual_success"][agent] for row in episodes) / count
            for agent in range(agent_count)
        ],
        "mean_undiscounted_team_return": sum(
            row["undiscounted_team_return"] for row in episodes
        )
        / count,
        "mean_discounted_team_return": sum(
            row["discounted_team_return"] for row in episodes
        )
        / count,
        "discount_gamma": episodes[0]["discount_gamma"],
        "mean_episode_length": sum(row["episode_length"] for row in episodes) / count,
        "timeout_fraction": sum(row["timeout"] for row in episodes) / count,
        "collision_events_per_episode": sum(
            row["collision_event_count"] for row in episodes
        )
        / count,
        "collision_episode_fraction": sum(
            row["collision_episode"] for row in episodes
        )
        / count,
        "contact_episode_fraction": sum(
            row["contact_episode"] for row in episodes
        )
        / count,
        "geometric_overlap_episode_fraction": sum(
            row["geometric_overlap_episode"] for row in episodes
        )
        / count,
        "simulator_collision_reward_episode_fraction": sum(
            row["simulator_collision_reward_steps"] > 0 for row in episodes
        )
        / count,
        "mean_minimum_center_separation": sum(
            row["minimum_center_separation"] for row in episodes
        )
        / count,
        "mean_minimum_surface_clearance": sum(
            row["minimum_surface_clearance"] for row in episodes
        )
        / count,
    }
