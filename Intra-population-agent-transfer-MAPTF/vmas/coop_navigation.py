"""Three-agent local-sensing cooperative navigation scenario for VMAS 1.5.x.

LiDAR values are normalized proximity readings in [0, 1]: 0 means no hit
within 0.35 world units, and 1 means the ray reaches an obstacle immediately.
VMAS's Lidar.measure() returns distance to the first hit, with max_range for
no hit; we convert it via 1 - distance / max_range.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar


class Scenario(BaseScenario):
    N_AGENTS = 3
    RADIUS = 0.1
    LIDAR_RANGE = 0.35
    N_RAYS = 12
    SUBSTEPS = 2
    HORIZON = 100
    GOAL_TOLERANCE = 0.1
    COLLISION_CLEARANCE = 0.005
    NOMINAL_STARTS = ((-0.75, -0.5), (-0.75, 0.0), (-0.75, 0.5))
    NOMINAL_GOALS = ((0.75, -0.5), (0.75, 0.0), (0.75, 0.5))

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.reset_jitter = float(kwargs.pop("reset_jitter", 0.03))
        self.control_scale = float(kwargs.pop("control_scale", 1.0))
        if kwargs:
            raise TypeError(f"Unexpected scenario arguments: {sorted(kwargs)}")
        if not 0.0 <= self.reset_jitter <= 0.03:
            raise ValueError("reset_jitter must be between 0 and 0.03")
        if self.control_scale <= 0:
            raise ValueError("control_scale must be positive")

        world = World(batch_dim, device, dt=0.1, substeps=self.SUBSTEPS)
        self.starts = torch.tensor(self.NOMINAL_STARTS, dtype=torch.float32, device=device)
        self.nominal_goals = torch.tensor(self.NOMINAL_GOALS, dtype=torch.float32, device=device)
        self.goals = []
        for i in range(self.N_AGENTS):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(radius=self.RADIUS),
                collide=True,
                rotatable=False,
                u_range=1.0,
                u_multiplier=self.control_scale,
                color=((0.22, 0.49, 0.72), (1.0, 0.50, 0.0), (0.30, 0.69, 0.29))[i],
                sensors=[Lidar(
                    world,
                    n_rays=self.N_RAYS,
                    max_range=self.LIDAR_RANGE,
                    entity_filter=lambda entity, holder=None: isinstance(entity, Agent)
                    and entity is not holder,
                    render=False,
                )],
            )
            # Bind the current agent into the sensor filter to exclude itself.
            agent.sensors[0].entity_filter = lambda entity, own=agent: isinstance(entity, Agent) and entity is not own
            world.add_agent(agent)
            goal = Landmark(name=f"goal_{i}", shape=Sphere(radius=0.04), collide=False, color=agent.color)
            world.add_landmark(goal)
            agent.goal = goal
            self.goals.append(goal)

        self.prev_distances = torch.zeros(batch_dim, self.N_AGENTS, device=device)
        self.progress = torch.zeros(batch_dim, device=device)
        self.collisions = torch.zeros(batch_dim, self.N_AGENTS, device=device)
        self.success = torch.zeros(batch_dim, dtype=torch.bool, device=device)
        self.min_center_distance = torch.full((batch_dim,), float("inf"), device=device)
        self.min_surface_clearance = torch.full((batch_dim,), float("inf"), device=device)
        self._recompute_distances()
        return world

    def _uniform_jitter(self, ref: Tensor, env_index: int | None) -> Tensor:
        if self.reset_jitter == 0:
            return torch.zeros_like(ref)
        return (torch.rand_like(ref) * 2.0 - 1.0) * self.reset_jitter

    def reset_world_at(self, env_index: int | None = None):
        # Each vectorized world receives independent position and goal samples.
        for i, agent in enumerate(self.world.agents):
            start = self.starts[i].expand(self.world.batch_dim, 2)
            goal = self.nominal_goals[i].expand(self.world.batch_dim, 2)
            start = start + self._uniform_jitter(start, env_index)
            goal = goal + self._uniform_jitter(goal, env_index)
            agent.set_pos(start, batch_index=env_index)
            self.goals[i].set_pos(goal, batch_index=env_index)
        self._recompute_distances(env_index)

    def _distances(self) -> Tensor:
        return torch.stack([
            torch.linalg.vector_norm(a.state.pos - a.goal.state.pos, dim=-1)
            for a in self.world.agents
        ], dim=-1)

    def _recompute_distances(self, env_index: int | None = None):
        distances = self._distances()
        if env_index is None:
            self.prev_distances.copy_(distances)
        else:
            self.prev_distances[env_index] = distances[env_index]

    def pre_step(self):
        self._pre_step_distances = self._distances().clone()

    def post_step(self):
        distances = self._distances()
        self.progress = (self._pre_step_distances - distances).sum(dim=-1)
        self.success = (distances <= self.GOAL_TOLERANCE).all(dim=-1)
        self.collisions.zero_()
        center_distances = []
        for i, a in enumerate(self.world.agents):
            for j in range(i + 1, self.N_AGENTS):
                b = self.world.agents[j]
                center = torch.linalg.vector_norm(a.state.pos - b.state.pos, dim=-1)
                clearance = center - 2 * self.RADIUS
                center_distances.append(center)
                # Count unordered pair once and each involved agent at most once per step.
                qualifying = self.world.collides(a, b) & (clearance <= self.COLLISION_CLEARANCE)
                self.collisions[:, i] = torch.maximum(self.collisions[:, i], qualifying.float())
                self.collisions[:, j] = torch.maximum(self.collisions[:, j], qualifying.float())
        if center_distances:
            centers = torch.stack(center_distances, dim=-1)
            self.min_center_distance = torch.minimum(self.min_center_distance, centers.min(dim=-1).values)
            self.min_surface_clearance = torch.minimum(
                self.min_surface_clearance, (centers - 2 * self.RADIUS).min(dim=-1).values
            )
        self.prev_distances = distances

    def observation(self, agent: Agent) -> Tensor:
        lidar_distance = agent.sensors[0].measure()
        lidar_proximity = (1.0 - lidar_distance / self.LIDAR_RANGE).clamp(0.0, 1.0)
        return torch.cat((agent.state.pos, agent.state.vel, agent.state.pos - agent.goal.state.pos, lidar_proximity), dim=-1)

    def reward(self, agent: Agent) -> Tensor:
        agent_id = self.world.agents.index(agent)
        return self.progress - self.collisions[:, agent_id] + 0.01 * self.success.float()

    def done(self) -> Tensor:
        return self.success

    def info(self, agent: Agent) -> dict[str, Tensor]:
        agent_id = self.world.agents.index(agent)
        return {
            "success": self.success,
            "agent_goal_success": torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1) <= self.GOAL_TOLERANCE,
            "collision": self.collisions[:, agent_id],
            "team_progress": self.progress,
            "min_center_distance": self.min_center_distance,
            "min_surface_clearance": self.min_surface_clearance,
        }
