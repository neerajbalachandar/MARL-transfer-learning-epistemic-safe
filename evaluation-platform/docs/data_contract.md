# Experiment interchange contract

The versioned interchange format is `schemas/experiment_bundle.schema.json`, version `1.0.0`. Every run bundle is a directory with `manifest.json`, `episodes.csv`, `metrics.csv` and `artifacts.json`. The manifest references checksums of the three data files. `eval-platform validate <bundle-dir>` checks required fields, schema version, file presence, CSV headers, and content checksums.

## `manifest.json`

The manifest describes method identity and evidence type, native environment/framework, task and reward/termination semantics, transfer conditions, source/target agent populations, architecture/configuration, seeds, phase budgets, repository revision/dirty state, hardware/software information, metric definitions, artifacts, and missing-field reasons. Unknown values stay null and carry a machine-readable reason in provenance or the normalized run record. `implementation_type` separates upstream code variants, paper-inspired local pilots, native evaluation exports, incomplete code and external references. It must not be used as a claim of paper reproduction.

## `episodes.csv`

One row represents one native evaluation episode when the source provides episode-level data. It includes `run_id`, seed, episode identity, split, source/target condition, checkpoint counter and native unit, raw source path/checksum, a JSON-encoded benchmark-specific `metrics` cell, and extension metadata. Dense exports with no episode semantics remain metric exports/artifacts, not pseudo-episodes.

## `metrics.csv`

One row represents a native scalar at its actual aggregation scope. Required semantics include original metric name, value, canonical category (or `unclassified_native_metric`), unit, mathematical/operational definition, range, direction, aggregation scope, benchmark semantics ID, source-file checksum, conditions, checkpoint counter/unit, and environment/task applicability. Arrays and objects are JSON encoded inside CSV cells. Native metrics with unknown meanings are retained under their original names and are not silently mapped to success, reward, safety or another category.

## Record hierarchy and comparisons

Run manifests, checkpoints, evaluation episodes, metric rows, phase costs, artifacts and validation issues remain separate record types. Training seeds are independent experimental units; episodes within a seed/checkpoint are first aggregated within that seed. Paired analysis requires explicit protocol support, not merely matching integer seed labels. Raw rewards and unlike native counters are never pooled. Comparisons require a shared benchmark semantics ID and compatible units. Missing baselines and small sample counts are shown as missing-data flags, not zeros or inferential conclusions.

Native files remain in their original repositories and are opened read-only. Normalized JSONL tables support dataset-wide portability; the per-run CSV bundle is the stable interchange interface. Generated outputs are kept under the selected output directory and ignored by Git by default.
