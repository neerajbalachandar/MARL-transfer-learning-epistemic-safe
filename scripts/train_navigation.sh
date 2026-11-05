#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARL_ROOT="${BENCHMARL_ROOT:-${REPO_ROOT}/../BenchMARL}"
SEED="${SEED:-0}"
FRAMES="${FRAMES:-120000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
DEVICE="${DEVICE:-$(python - <<'PY'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PY
)}"

python "${REPO_ROOT}/experiments/train_navigation.py" "${BENCHMARL_ROOT}" \
    algorithm=mappo \
    task=vmas/navigation \
    task.n_agents=3 \
    task.collisions=True \
    task.shared_rew=True \
    task.agents_with_same_goal=1 \
    task.split_goals=False \
    task.observe_all_goals=False \
    task.max_steps=100 \
    task.agent_radius=0.1 \
    task.lidar_range=0.35 \
    experiment.share_policy_params=False \
    "experiment.sampling_device=${DEVICE}" \
    "experiment.train_device=${DEVICE}" \
    "experiment.max_n_frames=${FRAMES}" \
    experiment.on_policy_collected_frames_per_batch=6000 \
    experiment.evaluation=True \
    experiment.evaluation_interval=24000 \
    "experiment.evaluation_episodes=${EVAL_EPISODES}" \
    experiment.evaluation_deterministic_actions=True \
    experiment.render=True \
    experiment.loggers='[csv,tensorboard]' \
    experiment.checkpoint_at_end=True \
    "seed=${SEED}"
