from __future__ import annotations

from pathlib import Path

from transfer_eval.ingestion.base import BaseAdapter, IngestionBundle, RunCandidate, read_json
from transfer_eval.manifest import normalize_manifest, sha256
from transfer_eval.normalization.metrics import finite_number
from transfer_eval.registry import RepositorySpec


class ZSCEvalAdapter(BaseAdapter):
    """Preserves official ZSC-Eval scalar exports without assigning opaque semantics."""

    def parse_manifest(self, candidate: RunCandidate, repository_root: Path) -> dict:
        parts = candidate.path.parts
        try:
            idx = parts.index("ZSC-Eval-master")
            zsc_rel = Path(*parts[idx + 1:]).as_posix()
        except ValueError:
            zsc_rel = candidate.path.name
        selection = "/prep/results/" in zsc_rel
        family = "zsc_eval_br_div_selection" if selection else "zsc_eval_partner_conditioned"
        identity = {"native_path": candidate.path.relative_to(repository_root).as_posix()}
        metadata = {
            "method_id": family,
            "transfer_axis": "partner_diversity_zero_shot_coordination",
            "condition": candidate.path.parent.name,
            "native_framework": "PyTorch",
            "simulator": "ZSC-Eval native benchmark (task not identified by export schema)",
            "scenario_id": None,
            "benchmark_id": "zsc_eval_partner_conditioned_native_export",
            "target_condition": None,
            "information_model": {"scope": "partner-conditioned evaluation"},
            "algorithm": candidate.path.parent.parent.name,
            "architecture": None,
            "seed_train": None,
            "seed_env": None,
            "extensions": {"zsc_relative_path": zsc_rel, "recorded_parent_directory": candidate.path.parent.name,
                           "native_filename": candidate.path.name,
                           "export_kind": "partner_selection" if selection else "evaluation_scalar_map"},
        }
        return normalize_manifest(
            root=repository_root, category=candidate.category, repo_path=self.spec.path,
            upstream_url=self.spec.upstream_url, status="implemented", candidate_path=candidate.path,
            identity=identity, metadata=metadata, implementation_type=self.spec.implementation_type,
        )

    def ingest(self, candidate: RunCandidate, repository_root: Path, metric_registry: dict) -> IngestionBundle:
        run = self.parse_manifest(candidate, repository_root)
        raw = read_json(candidate.path)
        checksum = sha256(candidate.path)
        rel = candidate.path.relative_to(repository_root).as_posix()
        bundle = IngestionBundle(run=run)
        if "selection" in raw and "det" in raw:
            value = finite_number(raw.get("det"))
            bundle.metrics.append({
                "run_id": run["run_id"], "seed": None, "metric_name": "zsc_br_div_determinant",
                "value": value, "unit": "determinant_score", "range": None, "direction": "higher",
                "aggregation_level": "selection_run", "benchmark_semantics_id": "zsc_eval_br_div_selection",
                "source_file": rel, "source_sha256": checksum, "source_condition": None,
                "target_condition": run.get("target_condition"), "checkpoint_step": None,
                "native_unit": "partner_candidates", "missing_reason": None if value is not None else "det absent",
                "extensions": {"selected_partner_ids": raw.get("selection"), "native_field": "det"},
            })
        else:
            # Preserve the dense partner-conditioned map as a native export
            # artifact and retain one opaque metric row per native scalar key.
            # These rows have export-level aggregation; they are not episodes,
            # seeds, or evidence of independent replications.
            for native_name, native_value in raw.items():
                number = finite_number(native_value)
                bundle.metrics.append({
                    "run_id": run["run_id"], "seed": None, "metric_name": str(native_name),
                    "value": number, "unit": "unknown", "range": None, "direction": "unknown",
                    "aggregation_level": "native_export_scalar", "benchmark_semantics_id": "zsc_eval_native_key",
                    "source_file": rel, "source_sha256": checksum, "source_condition": None,
                    "target_condition": None, "checkpoint_step": None, "native_unit": None,
                    "missing_reason": {"unit": "native export does not declare a unit",
                                       "definition": "native field semantics have not been mapped in the registry",
                                       **({"value": "native scalar is not numeric or finite"} if number is None else {})},
                    "extensions": {"native_field": str(native_name), "export_parent": candidate.path.parent.name},
                })
            bundle.artifacts.append({"run_id": run["run_id"], "artifact_type": "native_zsc_scalar_map",
                                     "path": rel, "format": "json", "sha256": checksum,
                                     "aggregation_level": "one native export scalar map", "exists": True,
                                     "native_payload": raw, "native_key_count": len(raw),
                                     "missing_reason": {"seed": "seed is not explicitly recorded in the export payload",
                                                        "native_unit": "no interaction counter in export"}})
            return bundle
        bundle.artifacts.append({"run_id": run["run_id"], "artifact_type": "native_zsc_selection_export", "path": rel,
                                 "format": "json", "sha256": checksum,
                                 "aggregation_level": "one partner-selection record", "exists": True})
        return bundle
