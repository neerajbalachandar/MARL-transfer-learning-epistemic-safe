"""Read-only skeleton for adding a native result adapter.

Copy this file into ``transfer_eval/ingestion/`` and replace the example
filename and field mappings only after documenting the native format. This
template deliberately does not import a simulator, training framework, or
checkpoint loader.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from transfer_eval.ingestion.base import BaseAdapter, IngestionBundle, RunCandidate
from transfer_eval.manifest import normalize_manifest, sha256
from transfer_eval.registry import RepositorySpec


class ExampleAdapter(BaseAdapter):
    """Parser skeleton; the filename below is illustrative, not a data claim."""

    RESULT_FILENAME = "documented_result.json"

    def __init__(self, spec: RepositorySpec):
        super().__init__(spec)

    def discover(self, repository_root: Path, allowlist: set[str] | None = None) -> list[RunCandidate]:
        if not self.spec.enabled or not self.spec.path:
            return []
        excluded = {".git", ".venv", "venv", "checkpoints", "models", "events", "tensorboard"}
        found = []
        for base, dirs, filenames in os.walk(repository_root / self.spec.path):
            dirs[:] = [name for name in dirs if name.lower() not in excluded]
            if self.RESULT_FILENAME not in filenames:
                continue
            path = (Path(base) / self.RESULT_FILENAME).resolve()
            candidate = RunCandidate(self.spec.category, path, self.spec.path)
            relative_path = path.relative_to(repository_root.resolve()).as_posix()
            if allowlist is None or relative_path in allowlist or candidate.candidate_id in allowlist:
                found.append(candidate)
        return sorted(found, key=lambda item: item.path.as_posix())

    def parse_manifest(self, candidate: RunCandidate, repository_root: Path) -> dict[str, Any]:
        """Map explicit native metadata; unknown values must remain missing."""
        # TODO: read candidate.path as data and record the exact native fields.
        identity = {"native_path": candidate.path.relative_to(repository_root).as_posix()}
        metadata = {
            "method_id": self.spec.category,
            "transfer_axis": None,
            "condition": None,
            "native_framework": None,
            "simulator": None,
            "scenario_id": None,
            "benchmark_id": None,
            "information_model": {},
            "extensions": {"native_format": "documented_result.json"},
        }
        return normalize_manifest(
            root=repository_root,
            category=candidate.category,
            repo_path=candidate.repository_path,
            upstream_url=self.spec.upstream_url,
            status=self.spec.status,
            candidate_path=candidate.path,
            identity=identity,
            metadata=metadata,
            implementation_type=self.spec.implementation_type,
        )

    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict) -> IngestionBundle:
        """Read saved results and preserve source provenance; never execute training code."""
        run = self.parse_manifest(candidate, repository_root)
        source = candidate.path.relative_to(repository_root).as_posix()
        checksum = sha256(candidate.path)
        bundle = IngestionBundle(run=run)

        # TODO: parse only documented records. For each native scalar, add a
        # metric row with its original name, actual value/scope, source path,
        # checksum, benchmark semantics, and missing_reason for unknown fields.
        # Do not fabricate episodes from an aggregate or assign seed/counter
        # values unless the native file explicitly records them.
        bundle.artifacts.append({
            "run_id": run["run_id"],
            "artifact_type": "native_result",
            "path": source,
            "format": candidate.path.suffix.lstrip(".") or "unknown",
            "sha256": checksum,
            "aggregation_level": "native file",
            "exists": True,
        })
        return bundle
