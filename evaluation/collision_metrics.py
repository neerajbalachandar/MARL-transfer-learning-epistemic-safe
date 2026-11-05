"""Collision and separation calculations for navigation rollouts."""

import torch


def pairwise_distances(positions):
    """Return distances for unique agent pairs; positions ends in [agents, xy]."""
    distances = torch.cdist(positions, positions)
    indices = torch.triu_indices(
        positions.shape[-2], positions.shape[-2], offset=1, device=positions.device
    )
    return distances[..., indices[0], indices[1]]


def collision_episode_metrics(episode_positions, collision_distance):
    """Count unique pairs that collide at least once in an episode."""
    distances = pairwise_distances(episode_positions)
    pair_collisions = distances <= collision_distance
    return {
        "collision_events": int(pair_collisions.any(dim=0).sum().item()),
        "has_collision": bool(pair_collisions.any().item()),
        "minimum_pairwise_separation": float(distances.min().item()),
    }
