from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # JSON is a valid YAML subset, and keeps the CLI usable without PyYAML.
    yaml = None


@dataclass(frozen=True)
class RepositorySpec:
    category: str
    path: str
    upstream_url: str | None
    status: str
    adapter: str
    run_patterns: tuple[str, ...]
    enabled: bool = True
    implementation_type: str = "unknown"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        if yaml is None:
            import json
            value = json.load(stream) or {}
        else:
            value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value


def load_repositories(config_path: Path) -> dict[str, RepositorySpec]:
    raw = load_yaml(config_path).get("repositories", {})
    return {
        category: RepositorySpec(
            category=category,
            path=value.get("path", ""),
            upstream_url=value.get("upstream_url"),
            status=value.get("status", "external_reference"),
            adapter=value["adapter"],
            run_patterns=tuple(value.get("run_patterns", ())),
            enabled=value.get("enabled", True),
            implementation_type=value.get("implementation_type", "unknown"),
        )
        for category, value in raw.items()
    }


def load_metrics(config_path: Path) -> dict[str, dict[str, Any]]:
    return load_yaml(config_path).get("metrics", {})
