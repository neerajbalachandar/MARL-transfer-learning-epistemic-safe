"""Combine per-episode VMAS navigation measurements into simple metrics."""

from evaluation.collision_metrics import collision_episode_metrics
from evaluation.success_rate import success_rates


def summarize_episodes(episodes, collision_distance):
    """Summarize episodes with positions, goal_hits, returns, and lengths."""
    if not episodes:
        raise ValueError("At least one episode is required")

    successes = success_rates([episode["goal_hits"] for episode in episodes])
    collisions = [
        collision_episode_metrics(episode["positions"], collision_distance)
        for episode in episodes
    ]
    return {
        **successes,
        "collision_events": sum(item["collision_events"] for item in collisions),
        "episodes_with_collision_fraction": sum(item["has_collision"] for item in collisions)
        / len(episodes),
        "mean_minimum_pairwise_separation": sum(
            item["minimum_pairwise_separation"] for item in collisions
        )
        / len(collisions),
        "mean_episode_return": sum(episode["return"] for episode in episodes)
        / len(episodes),
        "mean_episode_length": sum(episode["length"] for episode in episodes)
        / len(episodes),
        "timeout_episodes": sum(episode["timeout"] for episode in episodes),
        "episodes": len(episodes),
    }
