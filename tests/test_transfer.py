import torch
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from transfer.methods import (
    bayes_update,
    bpr_expected_improvement,
    continuous_policy_kl,
    freeze_policy,
    moderate_candidates,
    option_choice,
    portal_gmm_similarity,
    termination_probability,
)
from transfer.experiment import assert_policy_compatible, checkpoint_hash, copy_actor_weights
from transfer.navigation import TASK_GOALS, make_task


def test_peer_copy_is_frozen_and_does_not_change_source():
    source = torch.nn.Linear(3, 2)
    before = {key: value.clone() for key, value in source.state_dict().items()}
    peer = freeze_policy(source)
    assert all(not parameter.requires_grad for parameter in peer.parameters())
    for key, value in source.state_dict().items():
        assert torch.equal(value, before[key])


def test_continuous_policy_kl_has_action_dimension_and_receiver_gradients():
    source_loc = torch.zeros(4, 2)
    source_scale = torch.ones(4, 2)
    receiver_loc = torch.full((4, 2), 0.2, requires_grad=True)
    receiver_scale = torch.ones(4, 2, requires_grad=True)
    loss = continuous_policy_kl(source_loc, source_scale, receiver_loc, receiver_scale)
    loss.backward()
    assert loss.ndim == 0
    assert receiver_loc.grad is not None and receiver_loc.grad.abs().sum() > 0
    assert receiver_scale.grad is not None
    assert source_loc.grad is None and source_scale.grad is None


def test_termination_probability_and_deterministic_option_choice():
    features = torch.tensor([[-4.0], [0.0], [4.0]])
    probability = termination_probability(features, torch.tensor([[1.0]]), torch.tensor([0.0]))
    assert torch.all((probability >= 0) & (probability <= 1))
    assert option_choice(torch.tensor([1.0, 3.0, 2.0])) == 1
    assert option_choice(torch.tensor([3.0, 3.0])) == 0


def test_bayesian_update_matches_hand_calculation():
    prior = torch.tensor([0.5, 0.5])
    posterior = bayes_update(prior, torch.log(torch.tensor([0.8, 0.2])))
    assert torch.allclose(posterior, torch.tensor([0.8, 0.2]))
    assert torch.isclose(posterior.sum(), torch.tensor(1.0))
    assert bpr_expected_improvement(
        posterior,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.full((2, 2), 0.1),
    ) in (0, 1)


def test_custom_task_observation_spec_and_partial_reset():
    observed_shapes, action_shapes = set(), set()
    for task_name in TASK_GOALS:
        env = make_task(task_name).get_env_fun(2, True, 7, "cpu")()
        try:
            reset = env.reset()
            observed_shapes.add(tuple(reset.get(("agents", "observation")).shape[-2:]))
            action_shapes.add(tuple(env.full_action_spec_unbatched["agents", "action"].shape))
            if task_name == "T2":
                scenario = env._env.scenario
                before = [agent.state.pos[0].clone() for agent in scenario.world.agents]
                shaping_before = [agent.pos_shaping[0].clone() for agent in scenario.world.agents]
                scenario.reset_world_at(1)
                after = [agent.state.pos[0].clone() for agent in scenario.world.agents]
                assert all(torch.equal(old, new) for old, new in zip(before, after))
                for index, agent in enumerate(scenario.world.agents):
                    assert torch.equal(agent.pos_shaping[0], shaping_before[index])
                    expected = torch.linalg.vector_norm(
                        agent.state.pos[1] - agent.goal.state.pos[1]
                    ) * scenario.pos_shaping_factor
                    assert torch.allclose(agent.pos_shaping[1], expected)
        finally:
            env.close()
    assert observed_shapes == {(3, 18)}
    assert len(action_shapes) == 1


def test_actor_copy_checks_dimensions_and_leaves_source_checkpoint_untouched():
    source_policy = torch.nn.Linear(3, 2)
    target_policy = torch.nn.Linear(3, 2)
    source = SimpleNamespace(policy=source_policy)
    target = SimpleNamespace(policy=target_policy)
    with TemporaryDirectory() as folder:
        checkpoint = Path(folder) / "source.pt"
        torch.save(source_policy.state_dict(), checkpoint)
        before = checkpoint_hash(checkpoint)
        copy_actor_weights(source, target)
        assert checkpoint_hash(checkpoint) == before
        for left, right in zip(source_policy.parameters(), target_policy.parameters()):
            assert torch.equal(left, right)
    try:
        assert_policy_compatible(source_policy, torch.nn.Linear(4, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("incompatible actors must be rejected")


def test_portal_difficulty_fallback_and_gmm_alignment():
    candidates, interval = moderate_candidates({"a": -2.0, "b": 0.0, "c": 2.0, "d": 4.0})
    assert candidates
    assert "fallback" in interval
    observations = torch.arange(90, dtype=torch.float32).reshape(30, 3).numpy()
    score = portal_gmm_similarity(observations, observations, components=3)
    assert abs(score) < 1e-5
