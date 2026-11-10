from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from transfer_eval.ingestion.base import BaseAdapter, IngestionBundle, RunCandidate, ValidationIssue, read_json, read_jsonl, relative
from transfer_eval.manifest import normalize_manifest, sha256
from transfer_eval.normalization.metrics import finite_number
from transfer_eval.registry import RepositorySpec


def _first(data: dict, *keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


class PilotAdapter(BaseAdapter):
    """Ingests the local 3-transfer-test manifest and per-episode JSONL exports."""

    def parse_manifest(self, candidate: RunCandidate, repository_root: Path) -> dict[str, Any]:
        raw = read_json(candidate.path)
        method = str(_first(raw, "method_id", "method", "algorithm") or "pilot_unknown")
        target = _first(raw, "target_task", "task")
        source = _first(raw, "selected_source", "source_task")
        seed = _first(raw, "training_seed", "seed", "seed_train")
        method_lower = method.lower()
        axis = "task_transfer" if "portal" in method_lower or "bayes" in method_lower else "intra_population_transfer" if "maptf" in method_lower else "none"
        native = raw.get("dependency_revisions", {})
        vm_version = native.get("vmas") or raw.get("vmas_version")
        if isinstance(vm_version, dict):
            vm_version = vm_version.get("version") or vm_version.get("commit")
        phase_interactions = {}
        for phase, key in (("source_pretraining", "source_pretraining_frames"), ("source_probe", "probe_frames"), ("target_adaptation", "target_training_frames")):
            value = raw.get(key)
            if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
                value = sum(value)
            if isinstance(value, (int, float)):
                phase_interactions[phase] = {"value": float(value), "unit": "environment_frames"}
        if not phase_interactions and isinstance(raw.get("total_target_training_frames"), (int, float)):
            phase_interactions["target_adaptation"] = {"value": float(raw["total_target_training_frames"]), "unit": "environment_frames"}
        checkpoint = raw.get("checkpoint_path")
        checkpoint_path = Path(checkpoint) if checkpoint else None
        checkpoint_sha = sha256(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None
        metadata = {
            "method_id": method,
            "transfer_axis": axis,
            "condition": _first(raw, "selection_condition", "condition", "policy_initialization"),
            "ablation": raw.get("ablation"),
            "native_framework": "BenchMARL/TorchRL" if candidate.repository_path == "3-transfer-test" else None,
            "simulator": raw.get("simulator", "VMAS"),
            "simulator_version": vm_version,
            "scenario_id": _first(raw, "scenario_id", "scenario", "task", "target_task"),
            "map_id": raw.get("map_id"),
            "benchmark_id": "vmas_assigned_goal_navigation_pilot" if target else "vmas_navigation_pilot",
            "source_condition": source,
            "target_condition": target,
            "changed_factors": raw.get("changed_factors", {}),
            "n_agents_source": raw.get("n_agents_source"),
            "n_agents_target": raw.get("n_agents_target"),
            "agent_types": raw.get("agent_types"),
            "observation_schema": raw.get("observation_schema"),
            "action_schema": raw.get("action_schema"),
            "reward_definition_id": raw.get("reward_definition_id"),
            "termination_definition_id": raw.get("termination_definition_id"),
            "information_model": raw.get("information_model", {}),
            "algorithm": _first(raw, "algorithm", "method"),
            "architecture": raw.get("model_configuration"),
            "optimizer": raw.get("optimizer_configuration"),
            "hyperparameters_path": raw.get("hyperparameters_path"),
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "seed_train": seed,
            "seed_env": _first(raw, "seed_env", "validation_seed"),
            "validation_seed_set": raw.get("validation_seed_set") or ([raw["validation_seed"]] if raw.get("validation_seed") is not None else None),
            "test_seed_set": raw.get("test_seed_set") or ([raw["final_test_seed_base"]] if raw.get("final_test_seed_base") is not None else None),
            "checkpoint_selection_rule": raw.get("checkpoint_selection_rule"),
            "phase_interactions": phase_interactions,
            "phase_wall_time_seconds": raw.get("phase_wall_time_seconds", {}),
            "hardware_software": raw.get("hardware_software", {"dependency_revisions": native}),
            "extensions": {
                "native_manifest_path": relative(candidate.path, repository_root),
                "native_run_path": relative(candidate.path.parent, repository_root),
                "selected_source": raw.get("selected_source"),
                "source_checkpoints": raw.get("source_checkpoints"),
                "source_sha256": raw.get("source_sha256"),
                "limitations": raw.get("limitations"),
                "method_metadata": {key: raw.get(key) for key in (
                    "selection_condition", "likelihood_model", "probe_episodes_per_candidate",
                    "probe_frames", "total_probe_frames", "source_validation_frames",
                    "target_training_frames", "final_posterior", "stages", "policy_initialization",
                ) if key in raw},
            },
        }
        identity = {"native_path": relative(candidate.path, repository_root), "method": method, "task": target, "seed": seed}
        return normalize_manifest(
            root=repository_root, category=candidate.category, repo_path=candidate.repository_path,
            upstream_url=self.spec.upstream_url,
            status="partial" if "maptf" in method_lower or "portal" in method_lower else "implemented",
            candidate_path=candidate.path, identity=identity, metadata=metadata,
            implementation_type=self.spec.implementation_type,
        )

    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict) -> IngestionBundle:
        run = self.parse_manifest(candidate, repository_root)
        bundle = IngestionBundle(run=run)
        native_dir = candidate.path.parent
        artifact_paths = [candidate.path]
        eval_dir = native_dir / "evaluation"
        episode_files = []
        if eval_dir.is_dir():
            episode_files.extend(sorted(eval_dir.glob("*_episodes.jsonl")))
        # A legacy pilot may expose one summary file without episode rows.
        summary = native_dir / "evaluation.json"
        if summary.is_file():
            artifact_paths.append(summary)
        for path in episode_files:
            artifact_paths.append(path)
            provenance = {"source_file": relative(path, repository_root), "source_sha256": sha256(path)}
            split = path.stem.replace("_episodes", "")
            try:
                rows = read_jsonl(path)
            except (ValueError, json.JSONDecodeError) as exc:
                bundle.issues.append(ValidationIssue("error", "episode_jsonl_corrupt", str(exc), provenance["source_file"]))
                continue
            checkpoint_groups: dict[float | None, list[dict[str, Any]]] = defaultdict(list)
            for index, row in enumerate(rows):
                step = finite_number(row.get("environment_frames"))
                checkpoint_groups[step].append(row)
                metrics = {k: v for k, v in row.items() if isinstance(v, (bool, int, float)) or (isinstance(v, list) and all(isinstance(x, (bool, int, float)) for x in v))}
                extension = {k: v for k, v in row.items() if k not in metrics and k not in {"method", "task", "training_seed", "evaluation_seed", "split"}}
                episode = {
                    "run_id": run["run_id"],
                    "seed": row.get("training_seed", row.get("seed", run.get("seed_train"))),
                    "episode_index": row.get("episode_index", index),
                    "split": row.get("split", split),
                    "source_condition": row.get("source_task", run.get("source_condition")),
                    "target_condition": row.get("task", run.get("target_condition")),
                    "checkpoint_step": step,
                    "native_unit": "environment_frames" if step is not None else None,
                    **provenance,
                    "metrics": metrics,
                    "extensions": {**extension, "source_row_index": index},
                }
                bundle.episodes.append(episode)
                semantic = "vmas_navigation_assigned_goal_team_success_v1"
                definitions = {
                    "team_success": ("vmas_assigned_goal_team_success", "fraction", [0.0, 1.0], "higher"),
                    "individual_success": ("vmas_individual_goal_success", "fraction", [0.0, 1.0], "higher"),
                    "undiscounted_team_return": ("native_team_return", "native_reward", None, "unknown"),
                    "discounted_team_return": ("native_discounted_team_return", "native_reward", None, "unknown"),
                    "episode_length": ("episode_length", "policy_decisions", None, "lower"),
                    "timeout": ("timeout", "boolean", [0.0, 1.0], "lower"),
                    "collision_event_count": ("collision_event_count", "pair_events", None, "lower"),
                    "contact_episode": ("contact_episode", "boolean", [0.0, 1.0], "lower"),
                    "geometric_overlap_episode": ("geometric_overlap_episode", "boolean", [0.0, 1.0], "lower"),
                    "minimum_surface_clearance": ("minimum_surface_clearance", "world_distance", None, "higher"),
                    "minimum_center_separation": ("minimum_center_separation", "world_distance", None, "higher"),
                    "simulator_collision_reward_steps": ("collision_reward_steps", "policy_decisions", None, "lower"),
                }
                for source_key, (name, unit, value_range, direction) in definitions.items():
                    val = row.get(source_key)
                    if source_key == "individual_success" and isinstance(val, list):
                        values = {f"agent_{i}": float(bool(x)) for i, x in enumerate(val)}
                        for agent, agent_value in values.items():
                            bundle.metrics.append({
                                "run_id": run["run_id"], "seed": episode["seed"],
                                "metric_name": f"vmas_individual_goal_success_{agent}", "value": agent_value,
                                "unit": "fraction", "range": [0.0, 1.0], "direction": "higher",
                                "aggregation_level": "evaluation_episode", "benchmark_semantics_id": semantic,
                                **provenance, "source_condition": episode["source_condition"],
                                "target_condition": episode["target_condition"], "checkpoint_step": step,
                                "native_unit": "environment_frames" if step is not None else None,
                                "missing_reason": None, "extensions": {"source_field": source_key, "agent_id": agent,
                                                                           "episode_index": episode["episode_index"], "source_row_index": index,
                                                                           "split": episode["split"]},
                            })
                    else:
                        num = finite_number(val)
                        values = {"value": num} if num is not None else {}
                    if not values:
                        continue
                    bundle.metrics.append({
                        "run_id": run["run_id"], "seed": episode["seed"], "metric_name": name,
                        "value": values.get("value"), "unit": unit, "range": value_range,
                        "direction": direction, "aggregation_level": "evaluation_episode",
                        "benchmark_semantics_id": semantic, **provenance,
                        "source_condition": episode["source_condition"], "target_condition": episode["target_condition"],
                        "checkpoint_step": step, "native_unit": "environment_frames" if step is not None else None,
                        "missing_reason": None, "extensions": {"source_field": source_key,
                                                                  "per_agent_values": values if source_key == "individual_success" else None,
                                                                  "episode_index": episode["episode_index"], "source_row_index": index,
                                                                  "split": episode["split"]},
                    })
            for step, group in checkpoint_groups.items():
                successes = [finite_number(row.get("team_success")) for row in group]
                successes = [x for x in successes if x is not None]
                if not successes:
                    continue
                bundle.checkpoints.append({
                    "run_id": run["run_id"], "seed": run.get("seed_train"), "checkpoint_step": step,
                    "native_unit": "environment_frames" if step is not None else None,
                    "source_condition": run.get("source_condition"), "target_condition": run.get("target_condition"),
                    **provenance,
                    "metrics": {"vmas_assigned_goal_team_success": sum(successes) / len(successes)},
                    "extensions": {"split": split, "episode_count": len(group)},
                })
        if summary.is_file():
            raw_summary = read_json(summary)
            provenance = {"source_file": relative(summary, repository_root), "source_sha256": sha256(summary)}
            for source_key, name, unit, direction in (
                ("team_success_rate", "vmas_assigned_goal_team_success", "fraction", "higher"),
                ("mean_return", "native_team_return", "native_reward", "unknown"),
                ("mean_episode_length", "episode_length", "policy_decisions", "lower"),
            ):
                number = finite_number(raw_summary.get(source_key))
                if number is not None:
                    bundle.metrics.append({
                        "run_id": run["run_id"], "seed": run.get("seed_train"), "metric_name": name,
                        "value": number, "unit": unit, "range": [0.0, 1.0] if name.endswith("success") else None,
                        "direction": direction, "aggregation_level": "evaluation_summary",
                        "benchmark_semantics_id": "vmas_navigation_assigned_goal_team_success_v1",
                        **provenance, "source_condition": run.get("source_condition"),
                        "target_condition": run.get("target_condition"), "checkpoint_step": None,
                        "native_unit": "environment_frames", "missing_reason": None,
                        "extensions": {"source_field": source_key, "episode_count": raw_summary.get("episodes")},
                    })
        # Method diagnostics that the native pilot explicitly records.
        method_meta = run["extensions"].get("method_metadata", {})
        manifest_provenance = {"source_file": relative(candidate.path, repository_root), "source_sha256": sha256(candidate.path)}
        posterior = method_meta.get("final_posterior")
        if isinstance(posterior, list):
            for source_index, probability in enumerate(posterior):
                number = finite_number(probability)
                if number is not None:
                    bundle.metrics.append({
                        "run_id": run["run_id"], "seed": run.get("seed_train"),
                        "metric_name": f"source_policy_posterior_{source_index}", "value": number,
                        "unit": "probability", "range": [0.0, 1.0], "direction": "unknown",
                        "aggregation_level": "final_source_selection", "benchmark_semantics_id": "bayesian_source_policy_selection",
                        **manifest_provenance, "source_condition": run.get("source_condition"),
                        "target_condition": run.get("target_condition"), "checkpoint_step": None,
                        "native_unit": "probe_episodes", "missing_reason": None,
                        "extensions": {"source_index": source_index},
                    })
        for key in ("probe_episodes_per_candidate", "probe_frames", "total_probe_frames"):
            number = finite_number(method_meta.get(key))
            if number is not None:
                bundle.metrics.append({
                    "run_id": run["run_id"], "seed": run.get("seed_train"), "metric_name": key,
                    "value": number, "unit": "native_interactions", "range": None, "direction": "lower",
                    "aggregation_level": "selection_cost", "benchmark_semantics_id": "vmas_navigation_assigned_goal_team_success_v1",
                    **manifest_provenance, "source_condition": run.get("source_condition"),
                    "target_condition": run.get("target_condition"), "checkpoint_step": None,
                    "native_unit": "environment_frames" if "frames" in key else "episodes",
                    "missing_reason": None, "extensions": {"source_field": key},
                })
        phase_names = ("source_pretraining", "source_probe", "curriculum", "target_adaptation", "validation", "final_test")
        phase_interactions = run.get("phase_interactions", {})
        phase_times = run.get("phase_wall_time_seconds", {})
        for phase in phase_names:
            interaction = phase_interactions.get(phase)
            interaction_value = finite_number(interaction.get("value")) if isinstance(interaction, dict) else None
            interaction_unit = interaction.get("unit") if isinstance(interaction, dict) else None
            seconds = finite_number(phase_times.get(phase)) if isinstance(phase_times, dict) else None
            if interaction_value is not None or seconds is not None or phase in phase_interactions or phase in phase_times:
                bundle.costs.append({
                    "run_id": run["run_id"], "seed": run.get("seed_train"), "phase": phase,
                    "native_interactions": interaction_value, "interaction_unit": interaction_unit,
                    "wall_clock_seconds": seconds, "source_file": manifest_provenance["source_file"],
                    "source_sha256": manifest_provenance["source_sha256"],
                    "cpu_utilization": None, "gpu_utilization": None, "peak_ram_bytes": None,
                    "peak_vram_bytes": None,
                    "missing_reason": None if interaction_value is not None or seconds is not None else "phase is listed but its interaction and wall-time values are absent",
                })
        native_plots = sorted((repository_root / "3-transfer-test/results/figures").glob("**/*")) if candidate.repository_path == "3-transfer-test" else []
        for path in dict.fromkeys(artifact_paths):
            bundle.artifacts.append({
                "run_id": run["run_id"], "artifact_type": "native_evaluation_data" if "evaluation" in path.parts else "native_run_manifest",
                "path": relative(path, repository_root), "format": path.suffix.lstrip("."),
                "sha256": sha256(path), "aggregation_level": "run_or_episode", "exists": True,
            })
        for path in native_plots:
            if path.is_file() and path.suffix.lower() in {".png", ".pdf", ".svg"}:
                bundle.artifacts.append({
                    "run_id": run["run_id"], "artifact_type": "native_plot", "path": relative(path, repository_root),
                    "format": path.suffix.lstrip("."), "sha256": sha256(path),
                    "aggregation_level": "cross_run_native_figure", "exists": True,
                })
        return bundle
