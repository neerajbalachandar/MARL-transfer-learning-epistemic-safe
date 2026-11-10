from __future__ import annotations

from pathlib import Path
from typing import Any

from transfer_eval.ingestion.base import BaseAdapter, IngestionBundle, RunCandidate, ValidationIssue, read_json
from transfer_eval.manifest import normalize_manifest, sha256
from transfer_eval.normalization.metrics import finite_number
from transfer_eval.registry import RepositorySpec


class MapTFAdapter(BaseAdapter):
    """Read-only adapter for the legacy args.json + output JSON run convention."""

    def parse_manifest(self, candidate: RunCandidate, repository_root: Path) -> dict[str, Any]:
        raw = read_json(candidate.path)
        run_dir = candidate.path.parent
        game = raw.get("game")
        scenario = raw.get("game_name")
        algorithm = raw.get("algorithm")
        identity = {"native_path": candidate.path.relative_to(repository_root).as_posix(), "algorithm": algorithm, "game": game, "scenario": scenario, "seed": raw.get("seed")}
        metadata = {
            "method_id": f"maptf_{algorithm or 'unknown'}",
            "transfer_axis": "intra_population_transfer",
            "condition": ("adversary_option_enabled" if raw.get("adv_use_option") is True else
                          "adversary_option_disabled" if raw.get("adv_use_option") is False else "native_configured_run"),
            "ablation": None,
            "native_framework": "TensorFlow 1.x",
            "simulator": {"particle": "MPE particle world", "pacman": "Pacman"}.get(game, game),
            "simulator_version": None,
            "scenario_id": scenario,
            "map_id": scenario if game == "pacman" else None,
            "benchmark_id": f"legacy_{game}_{scenario}" if game and scenario else None,
            "source_condition": raw.get("source_condition"),
            "target_condition": scenario,
            "changed_factors": {"adv_use_option": raw.get("adv_use_option")},
            "n_agents_source": None,
            "n_agents_target": raw.get("n_agents"),
            "agent_types": None,
            "observation_schema": None,
            "action_schema": None,
            "reward_definition_id": f"native_{game}_{scenario}_reward" if game and scenario else None,
            "termination_definition_id": f"native_{game}_{scenario}_termination" if game and scenario else None,
            "information_model": {"source": "native args.json and scenario code"},
            "algorithm": algorithm,
            "architecture": None,
            "optimizer": raw.get("optimizer"),
            "hyperparameters_path": raw.get("algorithm_config"),
            "checkpoint_path": str(run_dir / "model") if (run_dir / "model").exists() else None,
            "checkpoint_sha256": None,
            "seed_train": raw.get("seed"),
            "seed_env": raw.get("seed"),
            "validation_seed_set": None,
            "test_seed_set": None,
            "checkpoint_selection_rule": None,
            "phase_interactions": {},
            "phase_wall_time_seconds": {},
            "hardware_software": {"python": "3.6 per README", "tensorflow": "1.14 per requirements.txt"},
            "extensions": {"native_args": raw, "native_run_dir": run_dir.relative_to(repository_root).as_posix()},
        }
        return normalize_manifest(
            root=repository_root, category=candidate.category, repo_path="Intra-population-agent-transfer-MAPTF",
            upstream_url=self.spec.upstream_url, status="implemented", candidate_path=candidate.path,
            identity=identity, metadata=metadata, implementation_type=self.spec.implementation_type,
        )

    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict) -> IngestionBundle:
        run = self.parse_manifest(candidate, repository_root)
        bundle = IngestionBundle(run=run)
        output = candidate.path.parent / "output" / "out.json"
        candidates = [output] if output.is_file() else sorted(candidate.path.parent.glob("output/*.json"))
        if not candidates:
            bundle.issues.append(ValidationIssue("warning", "native_output_missing", "args.json exists but no output/*.json metrics file was found", candidate.path.relative_to(repository_root).as_posix()))
        for path in candidates:
            raw = read_json(path)
            checksum = sha256(path)
            names = list(raw)
            lengths = {len(v) for v in raw.values() if isinstance(v, list)}
            if len(lengths) > 1:
                bundle.issues.append(ValidationIssue("error", "legacy_metric_length_mismatch", "MAPTF output arrays have different episode counts", path.relative_to(repository_root).as_posix()))
                continue
            n = next(iter(lengths), 0)
            for episode_index in range(n):
                payload = {key: values[episode_index] for key, values in raw.items() if isinstance(values, list) and episode_index < len(values)}
                scalar_metrics: dict[str, Any] = {}
                for key, value in payload.items():
                    # These are record structure or success flags, not reward
                    # curves. Preserve them in native_record, but do not plot
                    # episode indices or episode lengths as performance.
                    if key in {"episode", "step", "win"}:
                        continue
                    if isinstance(value, list):
                        for i, component in enumerate(value):
                            num = finite_number(component)
                            if num is not None:
                                scalar_metrics[f"native_{key}_agent_{i}"] = num
                    else:
                        num = finite_number(value)
                        if num is not None:
                            scalar_metrics[f"native_{key}"] = num
                bundle.episodes.append({
                    "run_id": run["run_id"], "seed": run.get("seed_train"), "episode_index": episode_index,
                    "split": "training_or_native_eval_unknown", "source_condition": run.get("source_condition"),
                    "target_condition": run.get("target_condition"), "checkpoint_step": episode_index,
                    "native_unit": "episodes", "source_file": path.relative_to(repository_root).as_posix(),
                    "source_sha256": checksum, "metrics": scalar_metrics,
                    "extensions": {"native_record": payload, "semantics_unresolved": True,
                                   "condition": run.get("condition")},
                })
                for metric_name, value in scalar_metrics.items():
                    bundle.metrics.append({
                        "run_id": run["run_id"], "seed": run.get("seed_train"), "metric_name": metric_name,
                        "value": value, "unit": "native_value", "range": None, "direction": "unknown",
                        "aggregation_level": "native_episode_record", "benchmark_semantics_id": run.get("benchmark_id"),
                        "source_file": path.relative_to(repository_root).as_posix(), "source_sha256": checksum,
                        "source_condition": run.get("source_condition"), "target_condition": run.get("target_condition"),
                        "checkpoint_step": episode_index, "native_unit": "episodes", "missing_reason": None,
                        "extensions": {"native_field": metric_name.removeprefix("native_"),
                                       "episode_index": episode_index,
                                       "split": "training_or_native_eval_unknown",
                                       "condition": run.get("condition")},
                    })
            bundle.artifacts.append({"run_id": run["run_id"], "artifact_type": "native_metric_export", "path": path.relative_to(repository_root).as_posix(), "format": "json", "sha256": checksum, "aggregation_level": "run", "exists": True})
        return bundle
