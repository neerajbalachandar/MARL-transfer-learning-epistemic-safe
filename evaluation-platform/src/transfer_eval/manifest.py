from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

STATUS = {"implemented", "partial", "planned", "external_reference", "not_implemented"}
REQUIRED = {
    "schema_version", "run_id", "method_id", "transfer_axis", "condition", "ablation", "implementation_status", "implementation_type",
    "repository_url", "repository_path", "commit_sha", "dirty_tree", "patch_id", "submodule_shas",
    "upstream_commit_sha", "native_framework", "simulator", "simulator_version", "scenario_id", "map_id",
    "benchmark_id", "source_condition", "target_condition", "changed_factors", "n_agents_source",
    "n_agents_target", "agent_types", "observation_schema", "action_schema", "reward_definition_id",
    "termination_definition_id", "information_model", "algorithm", "architecture", "optimizer",
    "hyperparameters_path", "checkpoint_path", "checkpoint_sha256", "seed_train", "seed_env",
    "validation_seed_set", "test_seed_set", "checkpoint_selection_rule", "phase_interactions",
    "phase_wall_time_seconds", "hardware_software", "artifacts", "missing_reason", "extensions",
}


@lru_cache(maxsize=512)
def git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_run_id(category: str, relative_path: str, identity: dict[str, Any]) -> str:
    raw = json.dumps([category, relative_path, identity], sort_keys=True, separators=(",", ":"))
    return f"{category}-{hashlib.sha256(raw.encode()).hexdigest()[:20]}"


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    issues = [f"missing required field: {field}" for field in sorted(REQUIRED - manifest.keys())]
    if manifest.get("schema_version") != "1.0.0":
        issues.append("schema_version must be '1.0.0'")
    if not isinstance(manifest.get("implementation_type"), str) or not manifest.get("implementation_type"):
        issues.append("implementation_type must identify the implementation evidence class")
    if not isinstance(manifest.get("run_id"), str) or not manifest.get("run_id"):
        issues.append("run_id must be a non-empty string")
    if manifest.get("implementation_status") not in STATUS:
        issues.append("implementation_status must be one of " + ", ".join(sorted(STATUS)))
    reasons = manifest.get("missing_reason", {})
    if not isinstance(reasons, dict):
        issues.append("missing_reason must be a mapping from field path to reason")
    else:
        for key, value in manifest.items():
            if value is None and key not in reasons:
                issues.append(f"null field {key!r} needs missing_reason[{key!r}]")
    for field in ("submodule_shas", "changed_factors", "information_model", "phase_interactions", "phase_wall_time_seconds", "hardware_software", "extensions"):
        if not isinstance(manifest.get(field), dict):
            issues.append(f"{field} must be an object")
    if not isinstance(manifest.get("artifacts"), list):
        issues.append("artifacts must be a list")
    return issues


def normalize_manifest(
    *, root: Path, category: str, repo_path: str, upstream_url: str | None,
    status: str, candidate_path: Path, identity: dict[str, Any], metadata: dict[str, Any],
    implementation_type: str = "unknown",
) -> dict[str, Any]:
    rel = candidate_path.resolve().relative_to(root.resolve()).as_posix()
    head = git_value(root, "rev-parse", "HEAD")
    dirty_output = git_value(root, "status", "--porcelain")
    dirty = bool(dirty_output)
    submodules: dict[str, str] = {}
    for checkout in (root / "venv/src/vmas", root / "venv/src/benchmarl"):
        sha = git_value(checkout, "rev-parse", "HEAD") if checkout.exists() else None
        if sha:
            submodules[checkout.relative_to(root).as_posix()] = sha

    missing_reason: dict[str, str] = {}
    output: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": stable_run_id(category, rel, identity),
        "method_id": metadata.get("method_id") or category,
        "transfer_axis": metadata.get("transfer_axis"),
        "condition": metadata.get("condition"),
        "ablation": metadata.get("ablation"),
        "implementation_status": status if status in STATUS else "partial",
        "implementation_type": implementation_type,
        "repository_url": upstream_url,
        "repository_path": repo_path,
        "commit_sha": head,
        "dirty_tree": dirty,
        "patch_id": None,
        "submodule_shas": submodules,
        "upstream_commit_sha": None,
        "native_framework": metadata.get("native_framework"),
        "simulator": metadata.get("simulator"),
        "simulator_version": metadata.get("simulator_version"),
        "scenario_id": metadata.get("scenario_id"),
        "map_id": metadata.get("map_id"),
        "benchmark_id": metadata.get("benchmark_id"),
        "source_condition": metadata.get("source_condition"),
        "target_condition": metadata.get("target_condition"),
        "changed_factors": metadata.get("changed_factors", {}),
        "n_agents_source": metadata.get("n_agents_source"),
        "n_agents_target": metadata.get("n_agents_target"),
        "agent_types": metadata.get("agent_types"),
        "observation_schema": metadata.get("observation_schema"),
        "action_schema": metadata.get("action_schema"),
        "reward_definition_id": metadata.get("reward_definition_id"),
        "termination_definition_id": metadata.get("termination_definition_id"),
        "information_model": metadata.get("information_model", {}),
        "algorithm": metadata.get("algorithm"),
        "architecture": metadata.get("architecture"),
        "optimizer": metadata.get("optimizer"),
        "hyperparameters_path": metadata.get("hyperparameters_path"),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "seed_train": metadata.get("seed_train"),
        "seed_env": metadata.get("seed_env"),
        "validation_seed_set": metadata.get("validation_seed_set"),
        "test_seed_set": metadata.get("test_seed_set"),
        "checkpoint_selection_rule": metadata.get("checkpoint_selection_rule"),
        "phase_interactions": metadata.get("phase_interactions", {}),
        "phase_wall_time_seconds": metadata.get("phase_wall_time_seconds", {}),
        "hardware_software": metadata.get("hardware_software", {}),
        "artifacts": metadata.get("artifacts", []),
        "missing_reason": missing_reason,
        "extensions": {"category": category, "candidate_identity": identity, **metadata.get("extensions", {})},
    }
    for key, value in output.items():
        if value is None:
            if key == "commit_sha":
                missing_reason[key] = "Git metadata unavailable from this checkout"
            elif key == "repository_url":
                missing_reason[key] = "No upstream URL recorded in repository registry"
            elif key == "upstream_commit_sha":
                missing_reason[key] = "No nested upstream checkout or upstream revision recorded"
            elif key == "patch_id":
                missing_reason[key] = "Dirty-tree patch artifact is not registered; working tree is preserved"
            else:
                missing_reason[key] = "Not recorded in the native run output"
    return output
