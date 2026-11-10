from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from transfer_eval.analysis.cross_category import cross_category_rows
from transfer_eval.analysis.condition_comparisons import condition_comparisons
from transfer_eval.analysis.population import population_heterogeneity_summary
from transfer_eval.analysis.learning_curves import curve_summary, normalized_auc, time_to_threshold
from transfer_eval.bundle import (EPISODE_CSV_COLUMNS, METRIC_CSV_COLUMNS,
                                  load_experiment_bundle, validate_experiment_bundle)
from transfer_eval.discovery import discover_all, load_context
from transfer_eval.ingestion import make_adapter
from transfer_eval.ingestion.base import RunCandidate
from transfer_eval.normalization.trajectories import export_trajectories
from transfer_eval.plotting.diagnostics import diagnostic_availability
from transfer_eval.plotting.episodes import plot_representative_episodes, plot_phase_interactions
from transfer_eval.plotting.efficiency import efficiency_table, plot_efficiency
from transfer_eval.plotting.matrices import plot_source_target_matrix
from transfer_eval.plotting.per_category import plot_learning_curves
from transfer_eval.plotting.safety import plot_safety_distributions
from transfer_eval.plotting.summary import plot_experiment_summary
from transfer_eval.plotting.transfers import plot_condition_comparisons
from transfer_eval.reporting.build_report import build_report
from transfer_eval.reporting.tables import write_csv, write_json, write_jsonl
from transfer_eval.statistics.aggregation import aggregate_episodes_to_seed


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PACKAGE_ROOT / "config/repositories.yaml"
DEFAULT_METRICS = PACKAGE_ROOT / "config/metric_registry.yaml"
DEFAULT_PROFILES = PACKAGE_ROOT / "config/plot_profiles.yaml"
DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs"
DEFAULT_RUNS = PACKAGE_ROOT / "runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval-platform", description="Platform-agnostic MARL result ingestion and analysis")
    parser.add_argument("--root", type=Path, default=PACKAGE_ROOT.parent, help="project repository root")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="repositories.yaml")
    parser.add_argument("--metric-config", type=Path, default=DEFAULT_METRICS, help="metric registry YAML")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="normalized output directory")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS, help="run index directory")
    parser.add_argument("--category", default="all", help="category key or all")
    parser.add_argument("--allowlist", action="append", default=[], help="candidate path/ID or text file containing paths (repeatable)")
    parser.add_argument("--run-id", action="append", default=[], help="normalized run ID filter")
    parser.add_argument("--overwrite", action="store_true", help="replace non-identical normalized/report outputs")
    parser.add_argument("--dry-run", action="store_true", help="validate/preview without writing outputs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "discover", "validate", "ingest", "analyze", "plot", "report", "dashboard", "compare"):
        child = sub.add_parser(name)
        if name == "ingest":
            child.add_argument("adapter_name", nargs="?", help="adapter key from repositories.yaml")
            child.add_argument("native_path", nargs="?", type=Path, help="native run file or directory, anywhere on disk")
        elif name == "validate":
            child.add_argument("bundle_path", nargs="?", type=Path, help="experiment bundle directory; omit to validate configured native runs")
        elif name in ("analyze", "plot"):
            child.add_argument("dataset_path", nargs="?", type=Path, help="experiment bundle or normalized dataset directory")
        elif name == "compare":
            child.add_argument("experiment_bundles", nargs="*", type=Path, help="one or more portable experiment bundle directories")
        if name in ("report", "dashboard", "compare"):
            child.add_argument("--format", default="html,csv", help="comma-separated html,pdf,csv")
        if name == "analyze":
            child.add_argument("--threshold", type=float, default=0.8)
    return parser


def _allowlist(values: list[str]) -> set[str] | None:
    if not values:
        return None
    output = set()
    for value in values:
        path = Path(value)
        if path.is_file():
            output.update(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#"))
        else:
            output.add(value)
    return output


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Corrupt normalized JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _write_bundle(bundles: list, out: Path, runs_dir: Path, overwrite: bool, metric_specs: dict | None = None):
    types = {"runs": [], "episodes": [], "checkpoints": [], "metrics": [], "costs": [], "artifacts": [], "validation": []}
    for bundle in bundles:
        types["runs"].append(bundle.run)
        for field in ("episodes", "checkpoints", "metrics", "costs", "artifacts"):
            types[field].extend(getattr(bundle, field))
        types["validation"].extend(issue.to_dict() for issue in bundle.issues)
    keys = {
        "runs": ("run_id",), "episodes": ("run_id", "source_file", "episode_index", "extensions"),
        "checkpoints": ("run_id", "source_file", "checkpoint_step"),
        "metrics": ("run_id", "source_file", "metric_name", "checkpoint_step", "seed", "extensions"),
        "costs": ("run_id", "phase", "source_file"), "artifacts": ("run_id", "path"),
        "validation": ("severity", "code", "source_file", "message"),
    }
    for record_type, rows in types.items():
        path = out / f"{record_type}.jsonl"
        previous = _read_jsonl(path)
        if overwrite and bundles:
            selected_categories = {bundle.run.get("extensions", {}).get("category") for bundle in bundles}
            selected_ids = {run.get("run_id") for run in _read_jsonl(out / "runs.jsonl")
                            if run.get("extensions", {}).get("category") in selected_categories}
            previous = [row for row in previous if row.get("run_id") not in selected_ids]
        merged = {}
        for row in previous + rows:
            key = tuple(json.dumps(row.get(field), sort_keys=True, separators=(",", ":")) for field in keys[record_type])
            if key in merged and merged[key] != row and not overwrite:
                raise FileExistsError(f"Conflicting {record_type} record without --overwrite: {row.get('run_id')}")
            merged[key] = row
        write_jsonl(path, list(merged.values()), overwrite or bool(previous))
    index_path = runs_dir / "index.json"
    old_index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    index = {row["run_id"]: row for row in old_index}
    index.update({row["run_id"]: {"run_id": row["run_id"], "category": row["extensions"].get("category"),
                                          "method_id": row.get("method_id"), "candidate_identity": row["extensions"].get("candidate_identity"),
                                          "source_file": row["extensions"].get("native_manifest_path")} for row in types["runs"]})
    write_json(index_path, sorted(index.values(), key=lambda row: row["run_id"]), overwrite or bool(old_index))
    _write_experiment_bundles(bundles, out, metric_specs or {}, overwrite)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_experiment_bundles(bundles: list, out: Path, metric_specs: dict, overwrite: bool):
    """Write portable per-run folders; native input files are never written here."""
    for bundle in bundles:
        run = bundle.run
        run_dir = out / "bundles" / run["run_id"]
        metric_rows = bundle.metrics
        metric_names = sorted({row.get("metric_name") for row in metric_rows if row.get("metric_name")})
        definitions = {}
        for name in metric_names:
            declared = dict(metric_specs.get(name, {}))
            sample = next(row for row in metric_rows if row.get("metric_name") == name)
            definitions[name] = {
                **declared,
                "original_metric_name": sample.get("extensions", {}).get("native_field", name),
                "canonical_category": declared.get("canonical_category", "unclassified_native_metric"),
                "unit": declared.get("unit", sample.get("unit")),
                "definition": declared.get("definition", declared.get("description")),
                "aggregation_scope": declared.get("aggregation_scope", sample.get("aggregation_level")),
                "direction": declared.get("direction", sample.get("direction", "unknown")),
                "environment_applicability": declared.get("environment_applicability") or ([run["simulator"]] if run.get("simulator") else []),
                "task_applicability": declared.get("task_applicability") or ([run.get("scenario_id") or run.get("benchmark_id")] if run.get("scenario_id") or run.get("benchmark_id") else []),
            }
        manifest = {
            "schema_version": "1.0.0", "run_id": run["run_id"], "method_id": run["method_id"],
            "implementation_status": run["implementation_status"], "implementation_type": run["implementation_type"],
            "environment": {key: run.get(key) for key in ("native_framework", "simulator", "simulator_version")},
            "task": {key: run.get(key) for key in ("scenario_id", "map_id", "benchmark_id", "reward_definition_id", "termination_definition_id")},
            "transfer_setting": {key: run.get(key) for key in ("transfer_axis", "condition", "source_condition", "target_condition", "changed_factors", "ablation")},
            "agent_population": {key: run.get(key) for key in ("n_agents_source", "n_agents_target", "agent_types", "observation_schema", "action_schema")},
            "configuration": {key: run.get(key) for key in ("algorithm", "architecture", "optimizer", "hyperparameters_path", "checkpoint_path", "checkpoint_sha256", "checkpoint_selection_rule")},
            "seeds": {key: run.get(key) for key in ("seed_train", "seed_env", "validation_seed_set", "test_seed_set")},
            "training_budget": {"phase_interactions": run.get("phase_interactions"), "phase_wall_time_seconds": run.get("phase_wall_time_seconds")},
            "provenance": {key: run.get(key) for key in ("repository_url", "repository_path", "commit_sha", "dirty_tree", "patch_id", "submodule_shas", "upstream_commit_sha", "hardware_software", "artifacts")},
            "metric_definitions": definitions,
            "files": {"episodes": "episodes.csv", "metrics": "metrics.csv", "artifacts": "artifacts.json"},
            "missing_reason": run.get("missing_reason", {}), "extensions": run.get("extensions", {}),
        }
        write_csv(run_dir / "episodes.csv", bundle.episodes, overwrite, EPISODE_CSV_COLUMNS)
        write_csv(run_dir / "metrics.csv", metric_rows, overwrite, METRIC_CSV_COLUMNS)
        write_json(run_dir / "artifacts.json", bundle.artifacts, overwrite)
        checksums = {name: _file_sha256(run_dir / filename) for name, filename in manifest["files"].items()}
        manifest["file_sha256"] = checksums
        manifest["normalized_run"] = run
        write_json(run_dir / "manifest.json", manifest, overwrite)


def _ingest_candidates(candidates, adapters, root, metric_specs):
    bundles = []
    issues = []
    for candidate in candidates:
        adapter = adapters[candidate.category]
        try:
            bundle = adapter.ingest(candidate, root, metric_specs)
            for metric in bundle.metrics:
                spec = metric_specs.get(metric.get("metric_name"), {})
                metric.setdefault("original_metric_name", metric.get("extensions", {}).get("native_field", metric.get("metric_name")))
                metric.setdefault("canonical_category", spec.get("canonical_category", "unclassified_native_metric"))
                metric.setdefault("definition", spec.get("definition", spec.get("description")))
                metric.setdefault("environment_applicability", spec.get("environment_applicability") or ([bundle.run["simulator"]] if bundle.run.get("simulator") else []))
                metric.setdefault("task_applicability", spec.get("task_applicability") or ([bundle.run.get("scenario_id") or bundle.run.get("benchmark_id")] if bundle.run.get("scenario_id") or bundle.run.get("benchmark_id") else []))
            local_issues = adapter.validate(bundle)
            bundle.issues.extend(local_issues)
            bundles.append(bundle)
            issues.extend(bundle.issues)
        except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            issues.append(type("Issue", (), {"severity": "error", "code": "candidate_parse_error", "message": str(exc), "source_file": candidate.path.as_posix()})())
    return bundles, issues


def _external_candidates(adapter_name: str, native_path: Path, repository_path: str) -> tuple[Path, list[RunCandidate]]:
    path = native_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Native result path does not exist: {path}")
    category = adapter_name
    if path.is_file():
        if adapter_name == "map_tf" and path.name != "args.json":
            args_path = path.parent / "args.json"
            paths = [args_path] if args_path.is_file() else []
        elif adapter_name == "pilot" and path.name != "manifest.json":
            manifest_path = path.parent.parent / "manifest.json" if path.parent.name == "evaluation" else path.parent / "manifest.json"
            paths = [manifest_path] if manifest_path.is_file() else []
        else:
            paths = [path]
        root = path.parent
    else:
        root = path
        expected_name = {"map_tf": "args.json", "pilot": "manifest.json", "portal": "manifest.json",
                         "gct_marl": "manifest.json", "bayes_tomop": "manifest.json",
                         "epirl_wumpus": "manifest.json"}.get(adapter_name)
        paths = []
        prune = {".git", ".venv", "venv", "checkpoints", "checkpoint", "models", "model", "graph", "tensorboard", "events"}
        for base, dirs, filenames in os.walk(path):
            dirs[:] = sorted(name for name in dirs if name.lower() not in prune)
            for filename in sorted(filenames):
                candidate = Path(base) / filename
                if adapter_name == "zsc_eval":
                    parts = candidate.parts
                    if candidate.suffix.lower() == ".json" and (("eval" in parts and "results" in parts) or ("prep" in parts and "results" in parts)):
                        paths.append(candidate)
                elif expected_name and filename == expected_name:
                    paths.append(candidate)
        paths.sort(key=lambda candidate: candidate.as_posix())
    if not paths:
        raise ValueError(f"No supported native result files found for adapter {adapter_name!r} under {path}")
    return root, [RunCandidate(category, candidate.resolve(), repository_path) for candidate in paths]


def _load_dataset_path(path: Path) -> dict[str, list[dict]]:
    path = path.resolve()
    if path.is_file() and path.name == "manifest.json":
        path = path.parent
    if (path / "manifest.json").is_file():
        issues = validate_experiment_bundle(path)
        if issues:
            raise ValueError("Invalid experiment bundle:\n- " + "\n- ".join(issues))
        return load_experiment_bundle(path)
    if (path / "runs.jsonl").is_file():
        return _load_normalized(path)
    if path == DEFAULT_OUTPUT.resolve():
        return _load_normalized(path)
    raise FileNotFoundError(f"Expected an experiment bundle (manifest.json) or normalized dataset directory: {path}")


def _load_normalized(out: Path) -> dict[str, list[dict]]:
    return {name: _read_jsonl(out / f"{name}.jsonl") for name in ("runs", "episodes", "checkpoints", "metrics", "costs", "artifacts", "validation")}


def _make_analysis(data: dict[str, list[dict]], metric_specs: dict, threshold: float):
    metrics = data["metrics"]
    primary = "vmas_assigned_goal_team_success"
    seed_rows = []
    for metric_name in sorted({row.get("metric_name") for row in metrics if row.get("metric_name")}):
        seed_rows.extend(aggregate_episodes_to_seed(metrics, metric_name))
    runs = {run["run_id"]: run for run in data["runs"]}
    for row in seed_rows:
        run = runs.get(row.get("run_id"), {})
        row["method_id"] = run.get("method_id")
        row["benchmark_id"] = run.get("benchmark_id")
        row["condition"] = run.get("condition")
    curves = curve_summary(seed_rows)
    efficiency = efficiency_table(seed_rows, {k: v.get("normalized_budget") for k, v in metric_specs.items() if v.get("normalized_budget")}, threshold)
    for row in efficiency:
        row["native_semantics_warning"] = "AUC is only meaningful within the metric's declared benchmark and budget."
    comparison = cross_category_rows(data["runs"], metrics, data["costs"], seed_rows, curves, metric_specs)
    within_conditions = condition_comparisons(seed_rows, data["runs"])
    population = population_heterogeneity_summary(seed_rows, data["runs"])
    diagnostics = diagnostic_availability(metrics)
    return {"seed_rows": seed_rows, "curves": curves, "efficiency": efficiency,
            "cross_category": comparison, "condition_comparisons": within_conditions,
            "population_heterogeneity": population,
            "diagnostic_availability": diagnostics}


def _write_analysis(analysis: dict, out: Path, overwrite: bool):
    analysis_dir = out / "analysis"
    for name, rows in (("seed_aggregates", analysis["seed_rows"]), ("learning_curve_points", analysis["curves"]),
                       ("efficiency", analysis["efficiency"]), ("cross_category_summary", analysis["cross_category"])):
        write_json(analysis_dir / f"{name}.json", rows, overwrite)
        write_csv(analysis_dir / f"{name}.csv", rows, overwrite)
    write_json(analysis_dir / "condition_comparisons.json", analysis["condition_comparisons"], overwrite)
    write_csv(analysis_dir / "condition_comparisons.csv", analysis["condition_comparisons"], overwrite)
    write_json(analysis_dir / "population_heterogeneity.json", analysis["population_heterogeneity"], overwrite)
    write_csv(analysis_dir / "population_heterogeneity.csv", analysis["population_heterogeneity"], overwrite)
    write_json(analysis_dir / "diagnostic_availability.json", analysis["diagnostic_availability"], overwrite)


def _do_plots(data: dict[str, list[dict]], analysis: dict, output: Path, overwrite: bool) -> list[str]:
    plot_root = output / "plots"
    generated: list[str] = []
    by_category = defaultdict(list)
    for run in data["runs"]:
        by_category[run.get("extensions", {}).get("category", "unknown")].append(run["run_id"])
    for category, ids in sorted(by_category.items()):
        idset = set(ids)
        curves = [row for row in analysis["seed_rows"] if row.get("run_id") in idset]
        curve_keys = {(row.get("method_id"), row.get("benchmark_id"), row.get("metric_name"))
                      for row in curves if row.get("metric_name")}
        expected = [plot_root / category / f"{category}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{method}_{benchmark}_{metric}')}_learning_curve.{suffix}"
                    for method, benchmark, metric in curve_keys for suffix in ("png", "pdf")]
        paths = expected if expected and not overwrite and all(path.exists() for path in expected) else plot_learning_curves(curves, plot_root / category, category)
        generated.extend(str(path.relative_to(output)) for path in paths)
    final_rows: dict[tuple, dict] = {}
    for row in analysis["seed_rows"]:
        if row.get("split") != "test" or row.get("source_condition") is None or row.get("target_condition") is None:
            continue
        key = (row.get("run_id"), row.get("seed"), row.get("source_condition"), row.get("target_condition"))
        prior = final_rows.get(key)
        step = row.get("checkpoint_step")
        if prior is None or (step is not None and (prior.get("checkpoint_step") is None or step > prior["checkpoint_step"])):
            final_rows[key] = row
    matrix_keys = {(row.get("method_id"), row.get("benchmark_id"), row.get("benchmark_semantics_id"))
                   for row in final_rows.values() if row.get("metric_name") == "vmas_assigned_goal_team_success"
                   and row.get("source_condition") and row.get("target_condition")}
    expected_matrix_paths = [plot_root / f"source_target_matrix_{re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{method}_{benchmark}_{semantics}')}.{suffix}"
                             for method, benchmark, semantics in matrix_keys for suffix in ("png", "pdf")]
    matrix_paths = (expected_matrix_paths if expected_matrix_paths and not overwrite and all(path.exists() for path in expected_matrix_paths)
                    else plot_source_target_matrix(list(final_rows.values()), plot_root, "vmas_assigned_goal_team_success"))
    generated.extend(str(path.relative_to(output)) for path in matrix_paths)
    efficiency_keys = {(row.get("method_id"), row.get("benchmark_id"), row.get("metric_name"))
                       for row in analysis["efficiency"] if row.get("normalized_auc") is not None}
    expected_efficiency_paths = [plot_root / f"learning_efficiency_{re.sub(r'[^A-Za-z0-9_.-]+', '_', f'{method}_{benchmark}_{metric}')}.{suffix}"
                                 for method, benchmark, metric in efficiency_keys for suffix in ("png", "pdf")]
    efficiency_paths = (expected_efficiency_paths if expected_efficiency_paths and not overwrite and all(path.exists() for path in expected_efficiency_paths)
                        else plot_efficiency(analysis["efficiency"], plot_root))
    generated.extend(str(path.relative_to(output)) for path in efficiency_paths)
    representative_dir = plot_root / "representative_episodes"
    selections_path = output / "analysis/representative_episode_selections.json"
    if representative_dir.exists() and not overwrite:
        representative_paths = sorted(representative_dir.glob("*.png"))
        selections = json.loads(selections_path.read_text(encoding="utf-8")) if selections_path.exists() else []
    else:
        representative_paths, selections = plot_representative_episodes(data["episodes"], data["runs"], plot_root)
        write_json(selections_path, selections, overwrite)
    generated.extend(str(path.relative_to(output)) for path in representative_paths)
    phase_paths = ([plot_root / f"phase_interactions.{suffix}" for suffix in ("png", "pdf")]
                   if not overwrite and all((plot_root / f"phase_interactions.{suffix}").exists() for suffix in ("png", "pdf"))
                   else plot_phase_interactions(data["costs"], plot_root, data["runs"]))
    generated.extend(str(path.relative_to(output)) for path in phase_paths)
    safety_dir = plot_root / "safety"
    if safety_dir.exists() and not overwrite:
        safety_paths = sorted(safety_dir.glob("*.png")) + sorted(safety_dir.glob("*.pdf"))
    else:
        safety_paths = plot_safety_distributions(data["metrics"], data["runs"], safety_dir)
    generated.extend(str(path.relative_to(output)) for path in safety_paths)
    transfer_dir = plot_root / "transfers"
    transfer_paths = sorted(transfer_dir.glob("*.png")) + sorted(transfer_dir.glob("*.pdf")) if transfer_dir.exists() and not overwrite else plot_condition_comparisons(analysis["condition_comparisons"], transfer_dir)
    generated.extend(str(path.relative_to(output)) for path in transfer_paths)
    summary_dir = plot_root / "summary"
    summary_paths = ([summary_dir / f"experiment_summary.{suffix}" for suffix in ("png", "pdf")]
                     if not overwrite and all((summary_dir / f"experiment_summary.{suffix}").exists() for suffix in ("png", "pdf"))
                     else plot_experiment_summary(analysis["cross_category"], summary_dir))
    generated.extend(str(path.relative_to(output)) for path in summary_paths)
    export_trajectories(data["episodes"], output / "normalized/trajectories.jsonl", overwrite)
    return generated


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = args.root.resolve()
    root = project_root
    config_path = args.config if args.config.is_absolute() else (project_root / args.config)
    metric_path = args.metric_config if args.metric_config.is_absolute() else (project_root / args.metric_config)
    # Defaults are installed with the package, while user-provided configs are root-relative.
    if not config_path.exists() and args.config == DEFAULT_CONFIG:
        config_path = DEFAULT_CONFIG
    if not metric_path.exists() and args.metric_config == DEFAULT_METRICS:
        metric_path = DEFAULT_METRICS
    specs, adapters, metric_specs = load_context(project_root, config_path, metric_path)
    allow = _allowlist(args.allowlist)
    direct_ingest = args.command == "ingest" and (args.adapter_name is not None or args.native_path is not None)
    if direct_ingest:
        if not args.adapter_name or not args.native_path:
            print("Direct ingestion requires both <adapter> and <native-result-path>.", file=sys.stderr)
            return 2
        category = args.adapter_name if args.adapter_name in specs else next(
            (key for key, spec in specs.items() if spec.adapter == args.adapter_name), None)
        if category is None:
            print(f"Unknown adapter {args.adapter_name!r}; configured adapters: {', '.join(sorted(specs))}", file=sys.stderr)
            return 2
        spec = specs[category]
        root, candidates = _external_candidates(spec.adapter, args.native_path, spec.path)
        adapters = {category: adapters[category]}
        selected_specs = {category: spec}
    else:
        selected_specs = specs if args.category == "all" else {args.category: specs[args.category]}
        candidates = discover_all(root, selected_specs, adapters, allow)
    if args.command in ("audit", "discover"):
        payload = {
            "root": str(root), "root_commit": _root_commit(root),
            "repositories": [{"category": spec.category, "path": spec.path, "status": spec.status,
                              "adapter": spec.adapter, "enabled": spec.enabled,
                              "candidate_count": sum(c.category == spec.category for c in candidates)}
                             for spec in selected_specs.values()],
            "candidate_examples": ([{"category": c.category, "path": c.path.relative_to(root).as_posix()} for c in candidates]
                                   if args.command == "discover" and (len(candidates) <= 200 or args.category != "all" or allow) else []),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "validate":
        if args.bundle_path is not None:
            issues = validate_experiment_bundle(args.bundle_path.resolve())
            for issue in issues:
                print(f"ERROR bundle_schema_invalid: {issue}", file=sys.stderr)
            print(f"validated bundle={args.bundle_path} issues={len(issues)}")
            return 2 if issues else 0
        bundles, issues = _ingest_candidates(candidates, adapters, root, metric_specs)
        for issue in issues:
            print(f"{issue.severity.upper()} {issue.code}: {issue.message} [{issue.source_file}]", file=sys.stderr)
        print(f"validated candidates={len(candidates)} bundles={len(bundles)} issues={len(issues)}")
        return 2 if any(issue.severity == "error" for issue in issues) else 0
    if args.command == "ingest":
        bundles, issues = _ingest_candidates(candidates, adapters, root, metric_specs)
        if args.run_id:
            selected = set(args.run_id)
            bundles = [bundle for bundle in bundles if bundle.run["run_id"] in selected]
        for issue in issues:
            print(f"{issue.severity.upper()} {issue.code}: {issue.message} [{issue.source_file}]", file=sys.stderr)
        if any(issue.severity == "error" for issue in issues):
            print("Ingestion stopped because one or more inputs failed validation; no normalized outputs were written.", file=sys.stderr)
            return 2
        print(f"ingested bundles={len(bundles)} episodes={sum(len(b.episodes) for b in bundles)} metrics={sum(len(b.metrics) for b in bundles)}")
        if not args.dry_run:
            _write_bundle(bundles, args.out, args.runs_dir, args.overwrite, metric_specs)
        return 0
    if args.command == "compare":
        if not args.experiment_bundles:
            print("compare requires one or more experiment bundle directories.", file=sys.stderr)
            return 2
        loaded = [_load_dataset_path(path) for path in args.experiment_bundles]
        names = ("runs", "episodes", "checkpoints", "metrics", "costs", "artifacts", "validation")
        data = {name: [row for item in loaded for row in item.get(name, [])] for name in names}
    elif getattr(args, "dataset_path", None) is not None:
        data = _load_dataset_path(args.dataset_path)
    else:
        data = _load_normalized(args.out)
    if args.category != "all" or args.run_id:
        selected_runs = {row["run_id"] for row in data["runs"]
                         if (args.category == "all" or row.get("extensions", {}).get("category") == args.category)
                         and (not args.run_id or row.get("run_id") in set(args.run_id))}
        for name in data:
            if name != "validation":
                data[name] = [row for row in data[name] if row.get("run_id") in selected_runs]
    analysis = _make_analysis(data, metric_specs, getattr(args, "threshold", 0.8))
    if args.command == "analyze":
        if not args.dry_run:
            _write_analysis(analysis, args.out, args.overwrite)
        print(f"analyzed runs={len(data['runs'])} metrics={len(data['metrics'])} seed_points={len(analysis['seed_rows'])}")
        return 0
    plots = _do_plots(data, analysis, args.out, args.overwrite)
    if args.command == "plot":
        print(json.dumps({"generated": plots, "diagnostics": analysis["diagnostic_availability"]}, indent=2))
        return 0
    formats = set(args.format.split(",")) if args.command in ("report", "dashboard", "compare") else {"html", "csv"}
    if not formats <= {"html", "pdf", "csv"}:
        print("Supported report formats are html,pdf,csv", file=sys.stderr)
        return 2
    if not args.dry_run:
        _write_analysis(analysis, args.out, args.overwrite)
        native_plot_paths = sorted({f"../../{item['path']}" for item in data["artifacts"]
                                    if item.get("artifact_type") == "native_plot" and item.get("path")})
        report_plots = plots + native_plot_paths
        paths = build_report(analysis["cross_category"], args.out / "reports", formats, report_plots, args.overwrite)
    else:
        paths = []
    print(json.dumps({"reports": [str(p) for p in paths], "plots": plots}, indent=2))
    return 0


def _root_commit(root: Path) -> str | None:
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
