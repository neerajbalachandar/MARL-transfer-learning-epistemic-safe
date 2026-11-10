from __future__ import annotations

from pathlib import Path

from transfer_eval.reporting.tables import write_jsonl


def export_trajectories(episodes: list[dict], output: Path, overwrite: bool = False) -> int:
    rows = []
    for episode in episodes:
        trajectory = episode.get("extensions", {}).get("trajectory_positions")
        goals = episode.get("extensions", {}).get("trajectory_goals")
        if trajectory is None:
            continue
        rows.append({
            "run_id": episode["run_id"], "seed": episode.get("seed"),
            "episode_index": episode.get("episode_index"), "split": episode.get("split"),
            "checkpoint_step": episode.get("checkpoint_step"), "native_unit": episode.get("native_unit"),
            "source_file": episode.get("source_file"), "source_sha256": episode.get("source_sha256"),
            "trajectory_positions": trajectory, "trajectory_goals": goals,
            "selection_criterion": "native evaluation episode; median/boundary/near-miss selection not inferred",
        })
    write_jsonl(output, rows, overwrite)
    return len(rows)
