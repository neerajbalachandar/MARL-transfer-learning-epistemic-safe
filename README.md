# MARL transfer learning baseline

This repository contains a cooperative MAPPO baseline for VMAS navigation. Training and checkpoint loading use the installed BenchMARL and VMAS packages.

## Install

Create and activate a Python environment, then install the pinned project dependencies:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

`scripts/train_navigation.sh` expects a BenchMARL checkout in `../BenchMARL`. Set `BENCHMARL_ROOT` if it is elsewhere.

## Run the VMAS smoke test

```bash
source venv/bin/activate
python experiments/test_vmas.py
```

The smoke test uses random actions and reports VMAS collision information.

## Train MAPPO

```bash
SEED=0 FRAMES=120000 EVAL_EPISODES=5 scripts/train_navigation.sh
```

CSV and TensorBoard logs are written under `outputs/<date>/<run>/`. Evaluation rendering runs during scheduled evaluation only. The CSV logger saves the rendered clip to `outputs/<date>/<run>/<experiment>/videos/eval_video_<step>.mp4`; TensorBoard records the same clip under the `eval_video` tag. Open that run in TensorBoard's **Videos** tab. The small training launcher uses PyAV to supply `write_video` for this installed torchvision build, which does not provide that function. CSV scalars remain enabled. Checkpoints are written with the experiment outputs.

The environment variables `SEED`, `FRAMES`, and `EVAL_EPISODES` configure the training seed, frame budget, and episodes per evaluation. Rendering uses VMAS's RGB renderer and may require a graphical display or X forwarding on Linux.

## Evaluate a checkpoint

```bash
python experiments/evaluate_checkpoint.py \
  outputs/<date>/<run>/checkpoints/checkpoint_120000.pt \
  --seeds 1,2,3 --episodes 5
```

The script reloads the saved BenchMARL experiment and evaluates its policy without training. It writes one CSV row per seed to a separate `evaluation/` folder alongside that run's logs. Add `--render` to save the first seed's rollout as an MP4 under `results/videos/`. Use `--output path.csv` to choose the metrics file. The original checkpoint and training logs are left in place.

For a video only:

```bash
python visualization/render_rollout.py \
  outputs/<date>/<run>/checkpoints/checkpoint_120000.pt --seed 1
```

The video uses VMAS's scene rendering with agent paths overlaid, collision markers when detected, and a completion or timeout label. A graphical display or X forwarding may be required by VMAS's renderer.

## Metrics

Evaluation uses the navigation scenario's agent-radius goal test and `min_collision_distance` from the installed VMAS source. Team success uses VMAS's simultaneous all-agent completion condition; individual success records whether each agent reached its assigned goal during the episode. Collision events count each unique agent pair once per episode; the collision frequency is the fraction of episodes with any such pair. The reported metrics are team and individual success rates, collision event count, collision episode fraction, mean minimum pairwise separation, mean team return, mean episode length, and timeout count. A timeout is an episode that reaches the configured horizon without the VMAS team completion condition. Collision reward magnitude in training logs is reported separately and is not used as a collision count.

Evaluation is deterministic by default. Each seed initializes a separate set of episodes. Explicit termination and truncation signals are used when the installed environment provides them; otherwise a full horizon without goal completion is counted as a timeout.

## TensorBoard

Start TensorBoard from the repository root:

```bash
tensorboard --logdir outputs
```

Then open the local address printed by TensorBoard. The evaluation video is stored as TensorBoard event data alongside the run scalars.

## Generate plots

Pass the timestamped experiment directory (the folder containing `.hydra/`):

```bash
python visualization/plot_training.py outputs/<date>/<run>
python visualization/plot_evaluation.py outputs/<date>/<run>
python visualization/plot_safety.py outputs/<date>/<run>
```

Plots are saved as high-resolution PNG and PDF files under `results/figures/<run>/`. The scripts discover BenchMARL scalar CSVs and map iteration steps through `counters_total_frames.csv`, so the x-axis uses environment frames. Evaluation success and actual collision frequency are shown only after checkpoint evaluation metrics are available. Missing metrics are reported as unavailable rather than filled with zero.

## Output and version control

Experiment outputs, checkpoints, generated videos, and figures are ignored by Git so local runs are preserved without adding large artifacts to source control. Source code, dependencies, and documentation remain tracked.

## Transfer experiments

The three transfer experiments share the same small VMAS task family in `transfer/navigation.py`. Goal layouts, fixed environment settings and each method's source/target task names live in `transfer/tasks.json`. T0, T1a, T1b, T1c and T2 change only the three agents' goal assignments; each reset adds seeded position jitter of at most 0.03. Agent count, observations, actions, reward, sensing and physics remain the navigation baseline.

Run source training before the transfer comparisons:

```bash
python experiments/train_transfer_task.py T1a --seed 0 --frames 600000
python experiments/train_transfer_task.py T1b --seed 0 --frames 600000
python experiments/train_transfer_task.py T2 --seed 0 --frames 600000
```

Use a competent T2 checkpoint for the MAPTF-style same-task experiment, and T1a/T1b checkpoints for Bayesian reuse:

```bash
python experiments/transfer_maptf.py outputs/T2/seed_0/<run>/checkpoints/checkpoint_600000.pt
python experiments/transfer_bayesian_reuse.py \
  outputs/T1a/seed_0/<run>/checkpoints/checkpoint_600000.pt \
  outputs/T1b/seed_0/<run>/checkpoints/checkpoint_600000.pt
python experiments/transfer_portal.py --mode portal
```

If a PORTAL run stops after training its initial T0 stage, continue from that saved stage checkpoint without retraining it:

```bash
python experiments/transfer_portal.py --mode portal \
  --resume-checkpoint outputs/portal/seed_0/stage_0_T0/<run>/checkpoints/checkpoint_600000.pt
```

The resumed run repeats the small T0 evaluation, selects the next task, and continues the remaining curriculum stages.

BenchMARL places each checkpoint in a generated run folder under the output directory. Find the exact path with `find outputs -name 'checkpoint_*.pt'`. All transfer runners accept `--frames`, `--seed`, `--output`, and `--device`; the Bayesian runner also accepts `--selection bayes|random|fixed-a|fixed-b|oracle`. The portal runner supports `portal`, `difficulty`, `manual`, and `scratch` comparisons. They write a `manifest.json` and a compact held-out `evaluation.json` beside their run logs.

The implementation is deliberately a first pilot. The MAPTF runner uses a frozen peer actor library, a simple online return bandit and a learned reward-change termination classifier. This is labeled MAPTF-style because it does not port the paper's SRO option-value update or execute temporally extended environment options. Bayesian reuse uses return/success/contact likelihoods and a BPR expected-improvement selector; it is not Bayes-ToMoP. PORTAL uses fixed-dimension Gaussian mixtures and actor-only transfer; ERFN is not included. Portal stages currently use fixed frame caps and record competence scores rather than implementing the paper's two-consecutive-evaluation stopping rule. These limits are included in each run manifest.

The default evaluation uses 100 held-out T2 episodes; set `--eval-episodes` in MAPTF/Bayesian/PORTAL commands to change that cost. `train_transfer_task.py` trains one source task. Use at least five training seeds for final comparisons; the default single seed and 600k frames are pilot settings, not final reported results.

For PORTAL main runs, first use a pilot to record the `difficulty_interval` in its manifest. Freeze those bounds for every main seed with `--difficulty-band LOW HIGH`. The CLI computes 25th/75th percentile bounds when this option is omitted.

After running all seed folders, compare their held-out metrics with:

```bash
python visualization/plot_transfer.py outputs
```

This writes one compact success, return and contact comparison figure in PNG and PDF under `results/figures/`.

## Research evaluation and plots

Every training evaluation now also writes per-episode validation rows to `evaluation/validation_episodes.jsonl`; held-out evaluation writes `evaluation/test_episodes.jsonl`. Run settings and dependency versions are saved in `manifest.json`. The shared records include return, team and per-agent success, timeout, collision-reward condition, contact, overlap, clearances, and trajectories.

Evaluate source policies on the same five-task test seed set. Pass multiple checkpoints per task for independent training seeds. To calculate zero-shot gain, also provide a separately trained scratch checkpoint for every target task:

```bash
python experiments/evaluate_source_target_matrix.py \
  --checkpoint T0=PATH --checkpoint T1a=PATH --checkpoint T1b=PATH \
  --checkpoint T1c=PATH --checkpoint T2=PATH \
  --scratch-checkpoint T0=PATH --scratch-checkpoint T1a=PATH \
  --scratch-checkpoint T1b=PATH --scratch-checkpoint T1c=PATH \
  --scratch-checkpoint T2=PATH
```

Generate the plots available from raw run data with:

```bash
python visualization/plot_research.py outputs --output results/figures/research
```

This writes available PNG and PDF learning curves, zero-shot matrices, BPR belief traces, safety distributions, and trajectory plots. Missing data are reported as unavailable. A valid transfer-gain matrix needs scratch baseline rows. The script uses the training-seed variation for intervals; episode variation from one policy is not treated as independent training runs.

The modules `transfer/epistemic.py` and `transfer/safety.py` are isolated research utilities: the first checks explicit finite possible-world sets, and the second scales nominal actions against a one-step geometric prediction. Neither is integrated into training. The latter is not a formal safety guarantee. MAPTF and PORTAL remain pilots; consult [the implementation audit](docs/implementation_audit.md), [algorithm mapping](docs/algorithm_mapping.md), [evaluation protocol](docs/evaluation_protocol.md), and [plot catalog](docs/plot_catalog.md) for exact limits and references.
