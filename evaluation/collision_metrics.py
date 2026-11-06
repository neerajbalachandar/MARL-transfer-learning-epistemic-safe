"""Pairwise distance and collision measurements for navigation episodes."""

import torch


def pairwise_center_distances(positions):
    """Return each unique pair's centre distance; input ends in [agents, xy]."""
    if positions.ndim < 2 or positions.shape[-1] != 2:
        raise ValueError("positions must end in [agents, 2]")
    if positions.shape[-2] < 2:
        return positions.new_empty((*positions.shape[:-2], 0))
    distances = torch.cdist(positions, positions)
    indices = torch.triu_indices(
        positions.shape[-2], positions.shape[-2], offset=1, device=positions.device
    )
    return distances[..., indices[0], indices[1]]


def pairwise_surface_clearances(positions, radii):
    """Return centre distance minus both sphere radii for each unique pair."""
    distances = pairwise_center_distances(positions)
    radii = torch.as_tensor(radii, dtype=positions.dtype, device=positions.device)
    if radii.ndim != 1 or radii.numel() != positions.shape[-2]:
        raise ValueError("radii must contain one value per agent")
    indices = torch.triu_indices(
        positions.shape[-2], positions.shape[-2], offset=1, device=positions.device
    )
    pair_radii = radii[indices[0]] + radii[indices[1]]
    return distances - pair_radii


def navigation_collision_metrics(
    episode_positions, agent_radii, vmas_reward_clearance=0.005
):
    """Measure contact, overlap, and VMAS's navigation collision-reward rule.

    VMAS navigation applies its collision reward when signed surface clearance is
    at most ``min_collision_distance`` (0.005 in the pinned scenario), after its
    broad-phase collision check. Geometric overlap is independently clearance < 0.
    Events count each false-to-true transition for each pair, including a collision
    already present in the first observed frame.
    """
    clearances = pairwise_surface_clearances(episode_positions, agent_radii)
    centers = pairwise_center_distances(episode_positions)
    reward_condition = clearances <= vmas_reward_clearance
    contact = clearances <= 0
    overlap = clearances < 0
    previous = torch.cat(
        (torch.zeros_like(reward_condition[:1]), reward_condition[:-1]), dim=0
    )
    event_count = int((reward_condition & ~previous).sum().item())
    return {
        "collision_event_count": event_count,
        "collision_pair_count": int(reward_condition.any(dim=0).sum().item()),
        "collision_episode": bool(reward_condition.any().item()),
        "contact_episode": bool(contact.any().item()),
        "geometric_overlap_episode": bool(overlap.any().item()),
        "minimum_center_separation": (
            float(centers.min().item()) if centers.numel() else float("inf")
        ),
        "minimum_surface_clearance": (
            float(clearances.min().item()) if clearances.numel() else float("inf")
        ),
        "vmas_collision_reward_clearance": float(vmas_reward_clearance),
    }


def collision_episode_metrics(episode_positions, collision_distance):
    """Backward-compatible centre-distance threshold summary.

    New navigation evaluation should call :func:`navigation_collision_metrics`
    with agent radii so physical overlap and VMAS reward conditions stay distinct.
    """
    distances = pairwise_center_distances(episode_positions)
    collisions = distances <= collision_distance
    previous = torch.cat((torch.zeros_like(collisions[:1]), collisions[:-1]), dim=0)
    return {
        "collision_events": int((collisions & ~previous).sum().item()),
        "has_collision": bool(collisions.any().item()),
        "minimum_pairwise_separation": float(distances.min().item()),
    }
