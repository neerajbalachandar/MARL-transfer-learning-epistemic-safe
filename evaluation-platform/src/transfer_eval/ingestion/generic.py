from __future__ import annotations

from pathlib import Path

from transfer_eval.ingestion.base import BaseAdapter, IngestionBundle, RunCandidate, ValidationIssue, read_json
from transfer_eval.manifest import normalize_manifest, sha256
from transfer_eval.registry import RepositorySpec


class GenericRunAdapter(BaseAdapter):
    """Strictly reads a future canonical manifest/episode JSONL, never model files."""

    def parse_manifest(self, candidate: RunCandidate, repository_root: Path) -> dict:
        raw = read_json(candidate.path)
        method = raw.get("method_id", self.spec.category)
        identity = {"path": candidate.path.relative_to(repository_root).as_posix(), "method": method, "seed": raw.get("seed_train")}
        metadata = {
            "method_id": method,
            "transfer_axis": raw.get("transfer_axis"),
            "condition": raw.get("condition"),
            "ablation": raw.get("ablation"),
            "native_framework": raw.get("native_framework"),
            "simulator": raw.get("simulator"),
            "simulator_version": raw.get("simulator_version"),
            "scenario_id": raw.get("scenario_id"),
            "map_id": raw.get("map_id"),
            "benchmark_id": raw.get("benchmark_id"),
            "source_condition": raw.get("source_condition"),
            "target_condition": raw.get("target_condition"),
            "changed_factors": raw.get("changed_factors", {}),
            "information_model": raw.get("information_model", {}),
            "algorithm": raw.get("algorithm"),
            "seed_train": raw.get("seed_train"),
            "seed_env": raw.get("seed_env"),
            "phase_interactions": raw.get("phase_interactions", {}),
            "phase_wall_time_seconds": raw.get("phase_wall_time_seconds", {}),
            "extensions": {"native_manifest": raw},
        }
        return normalize_manifest(root=repository_root, category=candidate.category, repo_path=self.spec.path,
                                  upstream_url=self.spec.upstream_url, status=self.spec.status,
                                  candidate_path=candidate.path, identity=identity, metadata=metadata,
                                  implementation_type=self.spec.implementation_type)

    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict) -> IngestionBundle:
        run = self.parse_manifest(candidate, repository_root)
        bundle = IngestionBundle(run=run)
        bundle.artifacts.append({"run_id": run["run_id"], "artifact_type": "native_run_manifest",
                                 "path": candidate.path.relative_to(repository_root).as_posix(),
                                 "format": "json", "sha256": sha256(candidate.path),
                                 "aggregation_level": "run", "exists": True})
        episode_path = candidate.path.parent / "evaluation" / "test_episodes.jsonl"
        if episode_path.exists():
            bundle.issues.append(ValidationIssue("warning", "generic_episode_parser_needed",
                "Found canonical-looking episodes, but this category's native metric mapping is not registered; values were not guessed.",
                episode_path.relative_to(repository_root).as_posix()))
        return bundle


class PortalAdapter(GenericRunAdapter):
    """Future explicit-manifest adapter; no native PORTAL run export is present locally."""


class GCTMarlAdapter(GenericRunAdapter):
    """Future explicit-manifest adapter; upstream SMAC results are not cloned locally."""


class BayesToMoPAdapter(GenericRunAdapter):
    """Native result parser is unavailable; checkpoint and performance arrays are not evaluation rows."""


class EPIRLWumpusAdapter(GenericRunAdapter):
    """Future adapter for original single-agent Rust Wumpus exports; no local run is present."""


class DisabledAdapter(BaseAdapter):
    def discover(self, repository_root: Path, allowlist: set[str] | None = None):
        return []

    def parse_manifest(self, candidate: RunCandidate, repository_root: Path):
        raise RuntimeError("This adapter is intentionally disabled until local reproducible runs exist")

    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict):
        raise RuntimeError("This adapter is intentionally disabled until local reproducible runs exist")

    def validate(self, bundle: IngestionBundle):
        return []
