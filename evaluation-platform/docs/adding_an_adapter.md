# Add a method adapter

1. Audit its actual result files and document the native field definitions, units, task/configuration identifiers, seed hierarchy and missing fields in `docs/input_audit.md`.
2. Classify the local implementation honestly in `config/repositories.yaml`: upstream/native variant, paper-inspired pilot, incomplete, external reference, or another precise evidence class. Do not label it a paper reproduction without evidence.
3. Implement `Adapter` from `transfer_eval.ingestion.base`. Discovery should accept an allowlist, select only known result filenames, and skip model/checkpoint/event directories. Parsing saved output must not import or execute the training environment.
4. Emit normalized `RunRecord`, episode, metric, checkpoint, cost and artifact rows. Preserve every native metric name. Register a canonical metric category only when its definition, unit, direction, aggregation scope, environment/task applicability and benchmark semantics are supported by native documentation.
5. Add the adapter key, implementation type and filename patterns to `config/repositories.yaml`; add metric definitions to `config/metric_registry.yaml`. If no native runs exist, use a clear not-yet-parseable status rather than manufacturing examples.
6. Add clearly marked synthetic fixtures strictly for software tests. Keep fixtures outside repository discovery roots and out of reports. Test provenance, corruption reporting, repeated ingestion and native file immutability as appropriate.

For local development the package can be installed with `python -m pip install -e .`, then the new adapter accepts arbitrary native inputs via `eval-platform ingest <adapter> /path/to/result-or-run-directory`. The resulting bundle is written under `outputs/bundles/<run_id>/`.

See [`templates/adapter_template.py`](../templates/adapter_template.py) for a minimal parser skeleton. The adapter must preserve unknown values and add missing reasons; it must never infer semantics from an artifact filename.
