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
