# Plot catalog

Generate figures from recorded JSONL and manifest files:

```bash
python visualization/plot_research.py outputs --output results/figures/research
```

The script writes PNG and PDF for the following figures when the raw inputs exist:

- `baseline_learning_curves`, `baseline_success_curves`: team return and success against actual environment frames.
- `safety_collision_curves`, `safety_contact_curves`, `safety_clearance_curves`: reward-threshold collision, physical contact, and clearance against frames. Fractions use [0, 1].
- `transfer_learning_return`, `transfer_learning_success`: T2 curves grouped by method and training seed.
- `source_target_matrix`: zero-shot team success for source and evaluation task pairs.
- `transfer_gain_matrix`: zero-shot gain over target scratch success. Requires scratch-checkpoint rows for every target; it is omitted otherwise.
- `transfer_efficiency`: validation frames to a predeclared success threshold and normalized area over observed T2 frames. Runs that miss the threshold are retained at the maximum observed frame and labeled as failures in the console output.
- `bayesian_belief_evolution`: posterior probability by cumulative probe episodes when BPR manifests contain probe histories.
- `portal_task_selection`: candidate difficulty and the recorded similarity proxy, with the chosen candidate annotated.
- `maptf_option_diagnostics`: peer selection counts and option durations when those pilot diagnostics exist.
- `safety_distribution`: measured clearance, event count, and episode length distributions.
- `trajectory_comparison`: actual paths from episode records; green indicates team success and red indicates contact.

Unavailable analyses such as post-finetuning gain, Bayesian selection accuracy/calibration, full PORTAL GMM component plots, and MAPTF representation/SR/transfer losses are reported as unavailable or omitted; no values are fabricated. The current script plots recorded trajectories from available episodes and does not guarantee one successful, failed, and colliding episode of each type will exist.
