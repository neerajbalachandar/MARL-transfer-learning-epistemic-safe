"""Shared fixed-team navigation tasks used by the transfer experiments."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, Optional

import torch
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.vmas.common import VmasClass
from vmas.scenarios.navigation import Scenario as NavigationScenario


with (Path(__file__).with_name("tasks.json")).open() as source:
    TASK_CONFIG = json.load(source)

TASK_GOALS = {name: tuple(values) for name, values in TASK_CONFIG["goal_y"].items()}
STARTS = tuple(tuple(position) for position in TASK_CONFIG["starts"])
EXPERIMENTS = TASK_CONFIG["experiments"]


class FixedNavigationScenario(NavigationScenario):
    """VMAS navigation with fixed, task-specific starts and goal assignments."""

    def make_world(self, batch_dim, device, **kwargs):
        self.task_name = kwargs.pop("task_name", "T2")
        self.position_jitter = float(kwargs.pop("position_jitter", TASK_CONFIG["position_jitter"]))
        if self.task_name not in TASK_GOALS:
            raise ValueError(f"Unknown transfer task {self.task_name!r}")
        return super().make_world(batch_dim, device, **kwargs)

    def reset_world_at(self, env_index=None):
        # Keep VMAS's seeded random generator in charge of small reset variation.
        device = self.world.device
        count = self.world.batch_dim if env_index is None else 1
        starts = torch.tensor(STARTS, device=device, dtype=torch.float32)
        goal_y = torch.tensor(TASK_GOALS[self.task_name], device=device)
        goals = torch.stack(
            (torch.full_like(goal_y, TASK_CONFIG["goal_x"]), goal_y), dim=-1
        )
        starts = starts.unsqueeze(0).expand(count, -1, -1).clone()
        goals = goals.unsqueeze(0).expand(count, -1, -1).clone()
        if self.position_jitter:
            starts += (torch.rand_like(starts) * 2 - 1) * self.position_jitter
            goals += (torch.rand_like(goals) * 2 - 1) * self.position_jitter

        for index, agent in enumerate(self.world.agents):
            agent.set_pos(starts[:, index], batch_index=env_index)
            agent.goal.set_pos(goals[:, index], batch_index=env_index)
            shaping = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos, dim=-1
            ) * self.pos_shaping_factor
            if env_index is None:
                agent.pos_shaping = shaping
            else:
                agent.pos_shaping[env_index] = shaping[env_index]


class NavigationTask(VmasClass):
    """BenchMARL task class that keeps VMAS's standard specs and transforms."""

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device,
    ) -> Callable:
        config = deepcopy(self.config)
        task_name = config.pop("task_name", self.name)
        jitter = config.pop("position_jitter", 0.03)
        max_steps = config.pop("max_steps")
        return lambda: VmasEnv(
            scenario=FixedNavigationScenario(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            max_steps=max_steps,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            task_name=task_name,
            position_jitter=jitter,
            **config,
        )


def make_task(task_name="T2"):
    """Build a task with the baseline navigation physics and sensing settings."""
    return NavigationTask(
        name=task_name,
        config={
            "task_name": task_name,
            "position_jitter": TASK_CONFIG["position_jitter"],
            **TASK_CONFIG["environment"],
        },
    )
