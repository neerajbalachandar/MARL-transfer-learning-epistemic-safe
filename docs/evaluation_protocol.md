# Evaluation protocol

## Data split

- Training seeds control model initialization and task randomization.
- Validation uses the per-run `validation_seed` (by default `100000 + training_seed`) and runs at the configured environment-frame interval.
- Final test evaluation uses a separate seed base (`900000 + training_seed`) and writes one JSON line per episode under `evaluation/test_episodes.jsonl`.
- Transfer source probing and oracle diagnostics use distinct seed ranges recorded in their manifest. Do not use final test episodes for policy selection.

The current scripts default to one training seed for a pilot. Research comparisons should run at least five independent seeds and use identical seed lists and held-out episode sets across methods. This repo does not claim completed five-seed results.

## Metrics

Raw episode rows keep success, individual success, undiscounted and discounted team return, gamma, length, timeout, collision event count, VMAS collision-reward condition, physical contact, geometric overlap, center separation, surface clearance, and (where requested) trajectories. Aggregate rows are summaries of episodes, not independent training runs.

Learning-curve x-values are `environment_frames` from BenchMARL. Aggregate intervals in `visualization/plot_research.py` use variation across training seeds when multiple seeds exist. Individual seed traces are shown. The plotted 95% normal interval is descriptive and imprecise for small seed counts; it is not a significance test. A single seed has no seed-level interval.

Timeout is a full horizon without team completion. Frames to competence, target-task AUC, paired tests, effect sizes, calibration, and wall-clock time are not yet computed consistently; do not report them as available from the present outputs.

## Collision terms

- `contact_episode`: any pair reaches zero or negative surface clearance.
- `geometric_overlap_episode`: any pair has negative surface clearance.
- `collision_episode`: any pair reaches the pinned navigation reward threshold, signed surface clearance `<= 0.005`.
- `simulator_collision_reward_episode_fraction`: derived from the environment's collision info channel when available.

These are distinct observations. The scenario uses a broad-phase collision check before applying its near-contact penalty; a geometric threshold alone does not reproduce every simulator collision response in every state.

## Source-target matrices

Run `experiments/evaluate_source_target_matrix.py` with source checkpoints for all five tasks. The initial matrix is zero-shot. For transfer gain, supply a task-specific scratch checkpoint for every target. All use the same explicit evaluation seed list. This runner does not yet produce a post-fine-tuning matrix.
