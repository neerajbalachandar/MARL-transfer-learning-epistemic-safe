from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from transfer_eval.analysis.learning_curves import time_to_threshold
from transfer_eval.ingestion import make_adapter
from transfer_eval.ingestion.base import RunCandidate
from transfer_eval.manifest import REQUIRED, validate_manifest, sha256
from transfer_eval.normalization.units import assert_same_semantics, safe_native_sum
from transfer_eval.plotting.per_category import plot_learning_curves
from transfer_eval.reporting.tables import write_jsonl
from transfer_eval.registry import RepositorySpec, load_metrics
from transfer_eval.statistics.aggregation import aggregate_episodes_to_seed


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
FIXTURES = PACKAGE / "tests/fixtures"


def spec(category, adapter, path="fixture", patterns=(), url=None, status="partial", enabled=True):
    return RepositorySpec(category, path, url, status, adapter, tuple(patterns), enabled)


def valid_manifest():
    value = {key: None for key in REQUIRED}
    value.update({
        "schema_version": "1.0.0", "run_id": "fixture-run", "method_id": "fixture", "implementation_status": "implemented",
        "implementation_type": "synthetic_test_fixture",
        "dirty_tree": False, "submodule_shas": {}, "changed_factors": {}, "information_model": {},
        "phase_interactions": {}, "phase_wall_time_seconds": {}, "hardware_software": {},
        "artifacts": [], "missing_reason": {}, "extensions": {},
    })
    for key, field in list(value.items()):
        if field is None:
            value["missing_reason"][key] = "synthetic fixture deliberately omits this field"
    return value


class PlatformTests(unittest.TestCase):
    def test_manifest_required_fields_missing_reasons_and_status(self):
        value = valid_manifest()
        self.assertEqual(validate_manifest(value), [])
        del value["scenario_id"]
        self.assertTrue(any("scenario_id" in issue for issue in validate_manifest(value)))
        value = valid_manifest()
        value["simulator"] = None
        value["missing_reason"].pop("simulator")
        self.assertTrue(any("missing_reason" in issue for issue in validate_manifest(value)))
        value = valid_manifest()
        value["implementation_status"] = "completed-ish"
        self.assertTrue(any("implementation_status" in issue for issue in validate_manifest(value)))

    def test_native_map_tf_adapter_reads_legacy_dict_of_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "results/multi_ppo/particle/simple_spread/run_0"
            (run_dir / "output").mkdir(parents=True)
            shutil.copy(FIXTURES / "map_tf/args.json", run_dir / "args.json")
            shutil.copy(FIXTURES / "map_tf/output/out.json", run_dir / "output/out.json")
            adapter = make_adapter("map_tf", spec("map_tf", "map_tf", "Intra-population-agent-transfer-MAPTF", ["results/**/args.json"]))
            candidate = adapter.discover(root)[0]
            bundle = adapter.ingest(candidate, root, {})
            self.assertEqual(len(bundle.episodes), 2)
            self.assertEqual(bundle.episodes[0]["extensions"]["native_record"]["win"], "False")
            self.assertEqual(bundle.episodes[0]["native_unit"], "episodes")
            self.assertEqual(bundle.run["algorithm"], "multi_ppo")

    def test_corrupt_native_input_is_an_explicit_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "results/multi_ppo/particle/simple_spread/run_0"
            (run_dir / "output").mkdir(parents=True)
            shutil.copy(FIXTURES / "map_tf/args.json", run_dir / "args.json")
            (run_dir / "output/out.json").write_text('{"win": [0,\n', encoding="utf-8")
            adapter = make_adapter("map_tf", spec("map_tf", "map_tf", "fixture", ["results/**/args.json"]))
            candidate = adapter.discover(root)[0]
            with self.assertRaises(json.JSONDecodeError):
                adapter.ingest(candidate, root, {})

    def test_pilot_episode_parser_and_native_file_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "3-transfer-test/outputs/T2/seed_7"
            (run_dir / "evaluation").mkdir(parents=True)
            shutil.copy(FIXTURES / "pilot/manifest.json", run_dir / "manifest.json")
            shutil.copy(FIXTURES / "pilot/validation_episodes.jsonl", run_dir / "evaluation/validation_episodes.jsonl")
            manifest = run_dir / "manifest.json"
            episodes = run_dir / "evaluation/validation_episodes.jsonl"
            before = (sha256(manifest), sha256(episodes))
            adapter = make_adapter("pilot", spec("pilot", "pilot", "3-transfer-test", ["3-transfer-test/outputs/**/manifest.json"]))
            candidate = adapter.discover(root)[0]
            bundle = adapter.ingest(candidate, root, load_metrics(PACKAGE / "config/metric_registry.yaml"))
            repeated = adapter.ingest(candidate, root, load_metrics(PACKAGE / "config/metric_registry.yaml"))
            after = (sha256(manifest), sha256(episodes))
            self.assertEqual(before, after)
            self.assertEqual(len(bundle.episodes), 2)
            self.assertEqual(bundle.episodes, repeated.episodes)
            issues = adapter.validate(bundle)
            self.assertFalse([issue for issue in issues if issue.severity == "error"])
            self.assertTrue(all(isinstance(metric["missing_reason"], dict) for metric in bundle.metrics))
            success = [m for m in bundle.metrics if m["metric_name"] == "vmas_assigned_goal_team_success"]
            self.assertEqual([m["value"] for m in success], [1.0, 0.0])
            self.assertTrue(all(m["native_unit"] == "environment_frames" for m in success))
            self.assertTrue(all(len(m["source_sha256"]) == 64 for m in success))

    def test_zsc_native_export_and_selection_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_path = root / "Heterogeneity-transfer-zsc-tomop/ZSC-Eval-master/zsceval/scripts/overcooked/eval/results/random1/hsp/eval-hsp-S2-s1-1.json"
            eval_path.parent.mkdir(parents=True)
            shutil.copy(FIXTURES / "zsc_eval/evaluation.json", eval_path)
            selection = root / "Heterogeneity-transfer-zsc-tomop/ZSC-Eval-master/zsceval/scripts/prep/results/random1/hsp-s1.json"
            selection.parent.mkdir(parents=True)
            shutil.copy(FIXTURES / "zsc_eval/selection.json", selection)
            adapter = make_adapter("zsc_eval", spec("zsc_eval", "zsc_eval", "Heterogeneity-transfer-zsc-tomop/ZSC-Eval-master", []))
            eval_candidate = RunCandidate("zsc_eval", eval_path, "Heterogeneity-transfer-zsc-tomop/ZSC-Eval-master")
            selected = adapter.ingest(eval_candidate, root, {})
            self.assertEqual(len(selected.metrics), 0)
            self.assertEqual(len(selected.episodes), 0)
            self.assertEqual(selected.artifacts[0]["native_payload"]["hsp-s1-bias1_final-eval_ep_shaped_r"], 2.5)
            self.assertEqual(selected.artifacts[0]["aggregation_level"], "one native export scalar map")
            selection_candidate = RunCandidate("zsc_eval", selection, "Heterogeneity-transfer-zsc-tomop/ZSC-Eval-master")
            selected = adapter.ingest(selection_candidate, root, {})
            self.assertEqual(selected.metrics[0]["metric_name"], "zsc_br_div_determinant")
            self.assertEqual(selected.metrics[0]["extensions"]["selected_partner_ids"], [2, 5, 8])

    def test_generic_adapter_families_have_fixture_parser(self):
        for category, filename in (("portal", "portal/manifest.json"), ("gct_marl", "gct_marl/manifest.json"), ("bayes_tomop", "bayes_tomop/evaluation.json"), ("epirl_wumpus", "epirl_wumpus/evaluation.json")):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "fixture" / Path(filename).name
                path.parent.mkdir(parents=True)
                shutil.copy(FIXTURES / filename, path)
                adapter = make_adapter(category, spec(category, category, "fixture", ["fixture/**/*.json"]))
                candidate = RunCandidate(category, path, "fixture")
                manifest = adapter.parse_manifest(candidate, root)
                self.assertIn("run_id", manifest)
                self.assertEqual(validate_manifest(manifest), [])

    def test_disabled_future_adapter_creates_no_records(self):
        adapter = make_adapter("autodrive_future", spec("autodrive_future", "autodrive_future", "Domain-transfer", [], status="not_implemented", enabled=False))
        self.assertEqual(adapter.discover(ROOT), [])

    def test_semantics_and_native_units_block_invalid_comparisons(self):
        with self.assertRaises(ValueError):
            assert_same_semantics([{"benchmark_semantics_id": "vmas"}, {"benchmark_semantics_id": "smac"}])
        with self.assertRaises(ValueError):
            safe_native_sum([{"n": 5, "u": "environment_frames"}, {"n": 5, "u": "environment_steps"}], "n", "u")

    def test_episodes_are_aggregated_within_seed_not_counted_as_seeds(self):
        rows = [
            {"run_id": "r0", "seed": 0, "metric_name": "success", "value": 1.0, "checkpoint_step": 100, "native_unit": "frames", "benchmark_semantics_id": "b"},
            {"run_id": "r0", "seed": 0, "metric_name": "success", "value": 0.0, "checkpoint_step": 100, "native_unit": "frames", "benchmark_semantics_id": "b"},
            {"run_id": "r1", "seed": 1, "metric_name": "success", "value": 1.0, "checkpoint_step": 100, "native_unit": "frames", "benchmark_semantics_id": "b"},
        ]
        result = aggregate_episodes_to_seed(rows, "success")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["seed_value"], 0.5)

    def test_threshold_is_right_censored(self):
        result = time_to_threshold([{"checkpoint_step": 10, "mean": 0.2, "native_unit": "frames"}], 0.8, 100)
        self.assertTrue(result["right_censored"])
        self.assertEqual(result["value"], 100)

    def test_outputs_are_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            rows = [{"run_id": "fixture", "value": 1}]
            write_jsonl(path, rows)
            before = path.read_bytes()
            write_jsonl(path, rows)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                write_jsonl(path, [{"run_id": "fixture", "value": 2}])

    def test_plots_regenerate_from_normalized_seed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"run_id": "r0", "seed": 0, "metric_name": "vmas_assigned_goal_team_success", "target_condition": "T2", "native_unit": "environment_frames", "checkpoint_step": 100, "split": "validation", "seed_value": 0.4},
                {"run_id": "r1", "seed": 1, "metric_name": "vmas_assigned_goal_team_success", "target_condition": "T2", "native_unit": "environment_frames", "checkpoint_step": 100, "split": "validation", "seed_value": 0.6},
                {"run_id": "r0", "seed": 0, "metric_name": "vmas_assigned_goal_team_success", "target_condition": "T2", "native_unit": "environment_frames", "checkpoint_step": 200, "split": "validation", "seed_value": 0.7},
                {"run_id": "r1", "seed": 1, "metric_name": "vmas_assigned_goal_team_success", "target_condition": "T2", "native_unit": "environment_frames", "checkpoint_step": 200, "split": "validation", "seed_value": 0.9},
            ]
            paths = plot_learning_curves(rows, Path(tmp), "fixture")
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
