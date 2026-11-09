# Implementation audit

This repository remains a VMAS three-agent navigation study using BenchMARL MAPPO. The work here adds reproducible per-episode measurement and small, testable transfer/safety components. It does **not** make every experiment in the research brief a faithful reproduction.

## Implemented

- Shared raw episode metrics for team and per-agent completion, undiscounted and discounted return, length, timeout, collision-reward condition, physical contact, overlap, and clearance.
- BenchMARL validation records at actual environment-frame counts, run manifests with settings and dependency revisions, and separate final-evaluation episode records.
- Zero-shot source-to-target evaluation with optional target-task scratch checkpoints for valid gain calculations.
- Factorized and joint rollout likelihoods, posterior probability floor, posterior/probe history, and BPR-inspired source choice.
- BPR target fine-tuning uses a conservative transfer-specific PPO schedule (4 passes and learning rate `1e-5`); these settings are recorded in its manifest and configurable from the CLI.
- PORTAL pilot labels and explicit `portal_aligned_mean_distance_similarity` naming. It is a mean-distance proxy, not a GMM KL divergence.
- Finite possible-world knowledge predicate in `transfer/epistemic.py`.
- A one-step nominal geometric action filter in `transfer/safety.py` using pinned navigation integration settings.
- Data-driven PNG/PDF curves, matrices, safety distributions, trajectories, and Bayesian belief plots. Missing inputs are reported as unavailable.

## Not implemented or not yet validated

- Full MAPTF SRO representation, successor-representation learner, option-value update and persistent option execution. `transfer/maptf.py` remains a pilot.
- Exact PORTAL competence stopping rule, full reference similarity statistic, and post-fine-tuning transfer matrix.
- Integrated safety filter in the BenchMARL policy action path; epistemic action preconditions are not connected to the filter.
- Continuous possible-world construction from VMAS partial observations, noisy/occluded sensors, or uncertainty calibration.
- Multi-seed research results, inferential statistics, wall-clock comparisons, and all plots requiring the corresponding run data.
- Paired statistical comparisons and selection accuracy/calibration across repeated source policies.

The safety utility is deliberately **not called a formal guarantee**. It checks a nominal one-step prediction and has documented model, sensing, and contact limitations. The epistemic utility checks only the finite worlds supplied by the caller.

## Focused verification

The focused checks can run without adding a test-runner dependency:

```bash
venv/bin/python tests/run_focused.py
```

It covers small synthetic cases for actor compatibility, task observation shapes, posterior math, PORTAL proxy behavior, possible-world knowledge, shield feasibility, and plot file generation. `pytest` is not installed in the current project environment. No full training run is part of this audit.

## Main commands

```bash
python experiments/train_transfer_task.py T2 --seed 0 --frames 600000
python experiments/evaluate_source_target_matrix.py \
  --checkpoint T0=PATH --checkpoint T1a=PATH --checkpoint T1b=PATH \
  --checkpoint T1c=PATH --checkpoint T2=PATH \
  --scratch-checkpoint T0=PATH --scratch-checkpoint T1a=PATH \
  --scratch-checkpoint T1b=PATH --scratch-checkpoint T1c=PATH \
  --scratch-checkpoint T2=PATH
python visualization/plot_research.py outputs/source_target_matrix
```

Use multiple independent checkpoints per source and provide the same evaluation seed list for all comparisons. These commands can be expensive; plots can be generated only after the runs exist.
