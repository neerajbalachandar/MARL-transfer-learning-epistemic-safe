from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from transfer_eval.manifest import validate_manifest
from transfer_eval.registry import RepositorySpec


@dataclass(frozen=True)
class RunCandidate:
    category: str
    path: Path
    repository_path: str

    @property
    def candidate_id(self) -> str:
        return f"{self.category}:{self.path.as_posix()}"


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    source_file: str | None = None

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class IngestionBundle:
    run: dict[str, Any]
    episodes: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    costs: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)


class Adapter(Protocol):
    def discover(self, repository_root: Path, allowlist: set[str] | None = None) -> list[RunCandidate]: ...
    def parse_manifest(self, candidate: RunCandidate, repository_root: Path) -> dict[str, Any]: ...
    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict) -> IngestionBundle: ...
    def validate(self, bundle: IngestionBundle) -> list[ValidationIssue]: ...


class BaseAdapter:
    def __init__(self, spec: RepositorySpec):
        self.spec = spec

    def discover(self, repository_root: Path, allowlist: set[str] | None = None) -> list[RunCandidate]:
        if not self.spec.enabled or not self.spec.path:
            return []
        found: dict[Path, RunCandidate] = {}
        for pattern in self.spec.run_patterns:
            for path in repository_root.glob(pattern):
                if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"}:
                    resolved = path.resolve()
                    found[resolved] = RunCandidate(self.spec.category, resolved, self.spec.path)
        candidates = sorted(found.values(), key=lambda c: c.path.as_posix())
        if allowlist is None:
            return candidates
        return [c for c in candidates if c.path.relative_to(repository_root.resolve()).as_posix() in allowlist or c.candidate_id in allowlist]

    def validate(self, bundle: IngestionBundle) -> list[ValidationIssue]:
        issues = [ValidationIssue("error", "manifest_invalid", msg) for msg in validate_manifest(bundle.run)]
        record_requirements = {
            "episode": {"run_id", "seed", "episode_index", "split", "source_condition", "target_condition",
                        "checkpoint_step", "native_unit", "source_file", "source_sha256", "metrics", "extensions"},
            "metric": {"run_id", "seed", "metric_name", "value", "unit", "range", "direction",
                       "aggregation_level", "benchmark_semantics_id", "source_file", "source_sha256",
                       "source_condition", "target_condition", "checkpoint_step", "native_unit",
                       "original_metric_name", "canonical_category", "definition",
                       "environment_applicability", "task_applicability"},
            "checkpoint": {"run_id", "seed", "checkpoint_step", "native_unit", "source_condition",
                           "target_condition", "source_file", "source_sha256", "metrics", "extensions"},
            "cost": {"run_id", "seed", "phase", "native_interactions", "interaction_unit", "wall_clock_seconds",
                     "source_file", "source_sha256", "cpu_utilization", "gpu_utilization", "peak_ram_bytes",
                     "peak_vram_bytes", "missing_reason"},
            "artifact": {"run_id", "artifact_type", "path", "format", "sha256", "aggregation_level", "exists"},
        }
        for name, rows in (("episode", bundle.episodes), ("metric", bundle.metrics),
                           ("checkpoint", bundle.checkpoints), ("cost", bundle.costs),
                           ("artifact", bundle.artifacts)):
            for row in rows:
                absent = record_requirements[name] - row.keys()
                if absent:
                    issues.append(ValidationIssue("error", f"{name}_schema_invalid",
                        f"Missing required fields: {', '.join(sorted(absent))}", row.get("source_file") or row.get("path")))
        for episode in bundle.episodes:
            if not episode.get("run_id") or not episode.get("source_sha256"):
                issues.append(ValidationIssue("error", "episode_provenance_missing", "Episode lacks run ID or source checksum", episode.get("source_file")))
        for metric in bundle.metrics:
            metric.setdefault("original_metric_name", metric.get("extensions", {}).get("native_field", metric.get("metric_name")))
            metric.setdefault("canonical_category", "unclassified_native_metric")
            metric.setdefault("definition", None)
            metric.setdefault("environment_applicability", [])
            metric.setdefault("task_applicability", [])
            missing = metric.setdefault("missing_reason", {})
            if not isinstance(missing, dict):
                missing = metric["missing_reason"] = {"record": "invalid prior missing_reason; replaced during validation"}
            for field in ("value", "unit", "range", "benchmark_semantics_id", "source_condition", "target_condition",
                          "checkpoint_step", "native_unit", "seed", "definition"):
                if metric.get(field) is None:
                    missing.setdefault(field, "Not recorded or not defined by the native result format")
            value = metric.get("value")
            if value is not None and not isinstance(value, (int, float)):
                issues.append(ValidationIssue("error", "metric_not_numeric", "Normalized metric value must be numeric or null", metric.get("source_file")))
        return issues


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{number}")
            rows.append(value)
    return rows


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
