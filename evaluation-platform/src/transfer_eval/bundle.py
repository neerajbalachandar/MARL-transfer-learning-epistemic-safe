from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from transfer_eval.manifest import validate_manifest


BUNDLE_REQUIRED = {"schema_version", "run_id", "method_id", "implementation_status", "implementation_type",
                   "environment", "task", "transfer_setting", "agent_population", "configuration", "seeds",
                   "training_budget", "provenance", "metric_definitions", "files"}
EPISODE_CSV_COLUMNS = ["run_id", "seed", "episode_index", "split", "source_condition", "target_condition",
                       "checkpoint_step", "native_unit", "source_file", "source_sha256", "metrics", "extensions"]
METRIC_CSV_COLUMNS = ["run_id", "seed", "metric_name", "original_metric_name", "canonical_category", "definition",
                      "environment_applicability", "task_applicability", "value", "unit", "range", "direction",
                      "aggregation_level", "benchmark_semantics_id", "source_file", "source_sha256",
                      "source_condition", "target_condition", "checkpoint_step", "native_unit", "missing_reason", "extensions"]
EPISODE_COLUMNS = {"run_id", "seed", "episode_index", "split", "source_condition", "target_condition",
                   "checkpoint_step", "native_unit", "source_file", "source_sha256", "metrics"}
METRIC_COLUMNS = {"run_id", "seed", "metric_name", "value", "unit", "range", "direction", "aggregation_level",
                  "benchmark_semantics_id", "source_file", "source_sha256", "source_condition", "target_condition",
                  "checkpoint_step", "native_unit", "canonical_category", "definition"}


def validate_experiment_bundle(path: Path) -> list[str]:
    issues: list[str] = []
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return [f"Missing bundle manifest: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Invalid manifest JSON at {manifest_path}: {exc}"]
    if not isinstance(manifest, dict):
        return ["manifest.json must contain an object"]
    missing = BUNDLE_REQUIRED - manifest.keys()
    if missing:
        issues.append("Manifest missing fields: " + ", ".join(sorted(missing)))
    if manifest.get("schema_version") != "1.0.0":
        issues.append(f"Unsupported schema_version: {manifest.get('schema_version')!r}")
    for field in ("run_id", "method_id", "implementation_type"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            issues.append(f"{field} must be a non-empty string")
    for field in ("environment", "task", "transfer_setting", "agent_population", "configuration", "seeds",
                  "training_budget", "provenance", "metric_definitions", "files"):
        if not isinstance(manifest.get(field), dict):
            issues.append(f"{field} must be an object")
    run_record = manifest.get("normalized_run")
    if run_record is not None:
        issues.extend(f"Normalized run: {issue}" for issue in validate_manifest(run_record))
    files = manifest.get("files", {})
    if not isinstance(files, dict):
        return issues + ["files must be an object"]
    for logical_name, required_columns in (("episodes", EPISODE_COLUMNS), ("metrics", METRIC_COLUMNS)):
        filename = files.get(logical_name)
        if not isinstance(filename, str):
            issues.append(f"files.{logical_name} must name a CSV file")
            continue
        relative_filename = Path(filename)
        if relative_filename.is_absolute() or ".." in relative_filename.parts:
            issues.append(f"files.{logical_name} must stay inside the bundle directory")
            continue
        csv_path = path / relative_filename
        if not csv_path.is_file():
            issues.append(f"Missing {logical_name} CSV: {csv_path}")
            continue
        try:
            with csv_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                header = set(reader.fieldnames or [])
                for row_number, row in enumerate(reader, 2):
                    if row.get("run_id") != manifest.get("run_id"):
                        issues.append(f"{logical_name}.csv row {row_number} has a run_id that differs from the manifest")
                    if logical_name == "metrics":
                        if not row.get("metric_name"):
                            issues.append(f"metrics.csv row {row_number} has an empty metric_name")
                        if row.get("direction") not in {"higher", "lower", "target", "unknown"}:
                            issues.append(f"metrics.csv row {row_number} has an invalid direction")
                        if row.get("value") not in {None, ""}:
                            try:
                                value = float(row["value"])
                                if not math.isfinite(value):
                                    raise ValueError("non-finite")
                            except (TypeError, ValueError):
                                issues.append(f"metrics.csv row {row_number} value must be a finite number or empty")
                        for field in ("environment_applicability", "task_applicability"):
                            try:
                                applicability = json.loads(row.get(field) or "[]")
                            except json.JSONDecodeError:
                                applicability = None
                            if not isinstance(applicability, list) or any(not isinstance(item, str) for item in applicability):
                                issues.append(f"metrics.csv row {row_number} {field} must be a JSON array of strings")
                    else:
                        try:
                            metric_values = json.loads(row.get("metrics") or "{}")
                        except json.JSONDecodeError:
                            metric_values = None
                        if not isinstance(metric_values, dict):
                            issues.append(f"episodes.csv row {row_number} metrics must be a JSON object")
        except (OSError, csv.Error) as exc:
            issues.append(f"Cannot read {logical_name} CSV: {exc}")
            continue
        absent = required_columns - header
        if absent:
            issues.append(f"{logical_name}.csv missing columns: " + ", ".join(sorted(absent)))
    artifacts = files.get("artifacts")
    if artifacts:
        relative_artifacts = Path(artifacts) if isinstance(artifacts, str) else None
        if relative_artifacts is None or relative_artifacts.is_absolute() or ".." in relative_artifacts.parts:
            issues.append("files.artifacts must be a bundle-relative path")
        elif not (path / relative_artifacts).is_file():
            issues.append(f"Missing artifact index: {path / relative_artifacts}")
    expected_hashes = manifest.get("file_sha256", {})
    if not isinstance(expected_hashes, dict):
        issues.append("file_sha256 must be an object")
        expected_hashes = {}
    for logical_name, filename in files.items():
        if not isinstance(filename, str):
            continue
        relative_filename = Path(filename)
        if relative_filename.is_absolute() or ".." in relative_filename.parts:
            continue
        file_path = path / relative_filename
        expected = expected_hashes.get(logical_name)
        if not file_path.is_file():
            continue
        if not expected:
            issues.append(f"Missing checksum for {logical_name}: {filename}")
            continue
        digest_builder = hashlib.sha256()
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest_builder.update(block)
        digest = digest_builder.hexdigest()
        if digest != expected:
            issues.append(f"Checksum mismatch for {logical_name}: {filename}")
    return issues


def _cell(value: str, field: str) -> Any:
    if value == "":
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value[0] in "[{\"":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if field in {"seed", "episode_index", "checkpoint_step", "value", "native_interactions", "wall_clock_seconds",
                 "episode_length", "environment_frames"}:
        try:
            return float(value) if any(mark in value for mark in ".eE") else int(value)
        except ValueError:
            return value
    return value


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{key: _cell(value, key) for key, value in row.items()} for row in csv.DictReader(stream)]


def load_experiment_bundle(path: Path) -> dict[str, list[dict]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    run = manifest.get("normalized_run")
    if not isinstance(run, dict):
        run = {
            "schema_version": manifest["schema_version"], "run_id": manifest["run_id"],
            "method_id": manifest["method_id"], "implementation_status": manifest["implementation_status"],
            "implementation_type": manifest["implementation_type"],
            "native_framework": manifest.get("environment", {}).get("native_framework"),
            "simulator": manifest.get("environment", {}).get("simulator"),
            "benchmark_id": manifest.get("task", {}).get("benchmark_id"),
            "scenario_id": manifest.get("task", {}).get("scenario_id"),
            "transfer_axis": manifest.get("transfer_setting", {}).get("transfer_axis"),
            "condition": manifest.get("transfer_setting", {}).get("condition"),
            "source_condition": manifest.get("transfer_setting", {}).get("source_condition"),
            "target_condition": manifest.get("transfer_setting", {}).get("target_condition"),
            "seed_train": manifest.get("seeds", {}).get("seed_train"),
            "phase_interactions": manifest.get("training_budget", {}).get("phase_interactions", {}),
            "phase_wall_time_seconds": manifest.get("training_budget", {}).get("phase_wall_time_seconds", {}),
            "extensions": {"category": manifest.get("method_id"), "implementation_type": manifest.get("implementation_type")},
        }
    else:
        run.setdefault("extensions", {})
    return {
        "runs": [run], "episodes": _read_csv(path / files["episodes"]),
        "metrics": _read_csv(path / files["metrics"]), "checkpoints": [], "costs": [], "artifacts": [], "validation": [],
    }
