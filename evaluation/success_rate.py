"""Success calculations using VMAS navigation's goal-distance rule."""

import torch


def goal_hits(positions, goals, goal_radii):
    """Return a [time, agents] mask for the VMAS navigation goal test."""
    distances = torch.linalg.vector_norm(positions - goals, dim=-1)
    radii = torch.as_tensor(goal_radii, device=positions.device, dtype=positions.dtype)
    return distances < radii


def success_rates(episode_goal_hits):
    """Use the scenario's simultaneous all-agent done condition and per-agent hits."""
    reached = torch.stack([hits.any(dim=0) for hits in episode_goal_hits])
    team_success = torch.stack([hits.all(dim=-1).any() for hits in episode_goal_hits])
    results = {"team_success_rate": float(team_success.float().mean().item())}
    results.update(
        {
            f"agent_{index}_success_rate": float(value)
            for index, value in enumerate(reached.float().mean(dim=0).tolist())
        }
    )
    return results
