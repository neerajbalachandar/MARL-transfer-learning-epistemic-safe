from __future__ import annotations

import math
from typing import Any


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def metric_record(*, registry: dict, metric_key: str, value: Any, run_id: str,
                  seed: int | None, source_file: str, source_sha256: str,
                  source_condition: str | None, target_condition: str | None,
                  checkpoint_step: float | None, native_unit: str | None,
                  benchmark_semantics_id: str | None = None, aggregation_level: str | None = None,
                  extensions: dict | None = None) -> dict:
    spec = registry.get(metric_key, {})
    number = finite_number(value)
    return {
        "run_id": run_id,
        "seed": seed,
        "metric_name": metric_key,
        "value": number,
        "unit": spec.get("unit"),
        "range": spec.get("range"),
        "direction": spec.get("direction", "unknown"),
        "aggregation_level": aggregation_level or spec.get("aggregation_level", "episode"),
        "benchmark_semantics_id": benchmark_semantics_id or spec.get("benchmark_semantics_id"),
        "source_file": source_file,
        "source_sha256": source_sha256,
        "source_condition": source_condition,
        "target_condition": target_condition,
        "checkpoint_step": checkpoint_step,
        "native_unit": native_unit,
        "missing_reason": None if number is not None else "value absent or non-finite in native source",
        "extensions": extensions or {},
    }
