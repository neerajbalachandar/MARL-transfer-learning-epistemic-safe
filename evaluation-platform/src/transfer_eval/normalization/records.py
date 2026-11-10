from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunRecord:
    manifest: dict[str, Any]

    def to_dict(self):
        return self.manifest


@dataclass
class TrainingCheckpointRecord:
    run_id: str
    seed: int | None
    checkpoint_step: float | None
    native_unit: str | None
    source_condition: str | None
    target_condition: str | None
    source_file: str
    source_sha256: str
    metrics: dict[str, float | None] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class EpisodeRecord:
    run_id: str
    seed: int | None
    episode_index: int | None
    split: str
    source_condition: str | None
    target_condition: str | None
    checkpoint_step: float | None
    native_unit: str | None
    source_file: str
    source_sha256: str
    metrics: dict[str, Any]
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class MetricRecord:
    run_id: str
    seed: int | None
    metric_name: str
    value: float | None
    unit: str | None
    range: list[float] | None
    direction: str
    aggregation_level: str
    benchmark_semantics_id: str | None
    source_file: str
    source_sha256: str
    source_condition: str | None
    target_condition: str | None
    checkpoint_step: float | None
    native_unit: str | None
    missing_reason: str | None = None
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class CostRecord:
    run_id: str
    seed: int | None
    phase: str
    native_interactions: float | None
    interaction_unit: str | None
    wall_clock_seconds: float | None
    source_file: str
    source_sha256: str
    cpu_utilization: float | None = None
    gpu_utilization: float | None = None
    peak_ram_bytes: int | None = None
    peak_vram_bytes: int | None = None
    missing_reason: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ArtifactRecord:
    run_id: str
    artifact_type: str
    path: str
    format: str | None
    sha256: str | None
    aggregation_level: str | None
    exists: bool
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
