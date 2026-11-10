from __future__ import annotations

from pathlib import Path

from transfer_eval.ingestion import make_adapter
from transfer_eval.registry import load_metrics, load_repositories


def load_context(repository_root: Path, repository_config: Path, metric_config: Path):
    specs = load_repositories(repository_config)
    metrics = load_metrics(metric_config)
    adapters = {key: make_adapter(spec.adapter, spec) for key, spec in specs.items()}
    return specs, adapters, metrics


def discover_all(repository_root: Path, specs: dict, adapters: dict, allowlist: set[str] | None = None):
    candidates = []
    for category, spec in sorted(specs.items()):
        candidates.extend(adapters[category].discover(repository_root, allowlist))
    # Stable ordering and path-level deduplication; a run belongs to its configured adapter.
    unique = {(candidate.category, candidate.path.resolve()): candidate for candidate in candidates}
    return sorted(unique.values(), key=lambda item: (item.category, item.path.as_posix()))
