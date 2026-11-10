# MARL evaluation and visualization platform

`evaluation-platform/` is a standalone ingestion, analysis, comparison and visualization package for saved MARL experiment outputs. It is not a training or simulation framework. It does not port algorithms to VMAS or BenchMARL, launch policies, import training environments, or modify native repositories.

## Install

From this directory, install the package with `python -m pip install -e .`. The core requires Python 3.10+, PyYAML and Matplotlib. It does not require VMAS, BenchMARL, TensorFlow, Unity, SMAC or method-specific training dependencies.

## CLI

```bash
# Inspect configured local repositories
eval-platform --root .. audit
eval-platform --root .. discover --category pilot

# Ingest either a specific external native run or configured local results
eval-platform ingest map_tf /path/to/native/run-or-args.json
eval-platform ingest pilot /path/to/3-transfer-test/outputs/T2/seed_0
eval-platform --root .. --category zsc_eval ingest

# Validate and analyze portable experiment bundles or normalized datasets
eval-platform validate outputs/bundles/<run_id>
eval-platform analyze outputs/bundles/<run_id>
eval-platform plot outputs/bundles/<run_id>
eval-platform compare outputs/bundles/<run_id_a> outputs/bundles/<run_id_b>
eval-platform --root .. report --format html,pdf,csv
```

`ingest <adapter> <native-result-path>` accepts a file or directory outside this repository. Directory discovery only selects known lightweight result filenames and skips checkpoint/model and event directories. `--out`, `--overwrite`, `--dry-run`, `--allowlist`, `--run-id`, `--config` and `--metric-config` control output and discovery. Run `eval-platform --help` for the complete syntax. When omitting an explicit path, `--root` selects the repository tree described by `repositories.yaml`.

## Interchange bundle

Each ingested run is exported under `outputs/bundles/<run_id>/`:

```text
manifest.json       # schema_version 1.0.0, run identity, conditions, config, seeds and provenance
episodes.csv        # one row per native evaluation episode where available
metrics.csv         # flexible native metric records and explicit semantic definitions
artifacts.json      # paths and checksums for saved native/generated artifacts
```

The versioned schema is in `schemas/experiment_bundle.schema.json`. `metrics.csv` retains the original field name and value as well as a canonical category, unit, definition, aggregation scope, direction, applicability and benchmark semantics ID. Unknown native metrics remain unclassified. CSV cells containing arrays or objects are JSON encoded. Run-level JSONL and analysis tables are also written for portable dataset-wide processing. Raw inputs are never copied over or edited.

The machine-readable source audit is [docs/input_audit.md](docs/input_audit.md). It distinguishes upstream code variants, paper-inspired pilots, external references and not-implemented work. No code or checkpoint artifact is labeled as a paper reproduction without run evidence.

## Analysis rules

Training seeds, not episodes, are the independent unit. The package reports sample counts and does not make inferential claims from insufficient seeds. Paired comparisons are not assumed from matching numeric seed labels. Scratch, zero-shot and transfer roles are recognized only from explicit condition labels; unknown labels remain unclassified. Comparisons require matching benchmark semantics and native units.

Raw returns are never compared across unrelated environments. Missing or incomplete experiments remain missing, never zero. Opaque ZSC-Eval keys keep their native names and export-level aggregation. ZSC partner diversity, Bayes-ToMoP competitive opponent reasoning, physical capability heterogeneity and simulation-domain transfer are separate categories. Domain transfer and physical capability heterogeneity remain `NOT_IMPLEMENTED` until local runs exist.

Figures are generated only when the required records exist. Safety measures are plotted under their native definitions; trajectories require saved trajectory data. The HTML report links native figures without scraping them. Large normalized outputs and generated figures/reports are ignored by Git by default.

## Add a method

Implement `Adapter` in `src/transfer_eval/ingestion/`, document the native format and semantic limits, add its registry/config entry, and emit the versioned run/episode/metric/artifact records. A template and checklist are in [docs/adding_an_adapter.md](docs/adding_an_adapter.md). The central analysis and plotting code should remain method-agnostic; only the adapter and metric registry should need method-specific changes.
