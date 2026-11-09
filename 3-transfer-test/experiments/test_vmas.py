import torch
import vmas


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create 4 parallel simulation environments.
    # Each environment contains 3 cooperative agents.

    env = vmas.make_env(
        scenario="navigation",
        num_envs=4,
        n_agents=3,
        device=device,
        continuous_actions=True,
        max_steps=100,
        seed=0,

        # Collision-related configuration
        collisions=True,
        agent_radius=0.1,
        lidar_range=0.35,

        # Cooperative task configuration
        shared_rew=True,
        agents_with_same_goal=1,
        split_goals=False,
        observe_all_goals=False,
    )

    observations = env.reset()

    print("Number of agents:", env.n_agents)

    for i, obs in enumerate(observations):
        print(f"Agent {i} observation shape:", obs.shape)

    collision_agent_steps = 0

    # Test the simulator with random actions.
    for step in range(20):

        actions = env.get_random_actions()

        observations, rewards, dones, infos = env.step(actions)

        collision_rewards = torch.stack(
            [info["agent_collisions"] for info in infos],
            dim=1
        )

        collision_agent_steps += int(
            (collision_rewards < 0).sum().item()
        )

    print("Random-action test completed.")
    print("Collision-related agent-step count:",
          collision_agent_steps)

    print("Environment information keys:")
    print(infos[0].keys())


if __name__ == "__main__":
    main()