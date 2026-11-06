"""Finite possible-world knowledge checks for controlled experiments.

This module is a small exact checker for an explicitly supplied finite world
set. It does not infer worlds from VMAS observations or model continuous sensor
uncertainty; callers must construct the conservative set themselves.
"""

from collections.abc import Callable, Iterable


def indistinguishable(world_a, world_b, observation):
    """Return whether two worlds give the same observation to one agent."""
    return observation(world_a) == observation(world_b)


def information_set(worlds: Iterable, actual_world, observation: Callable):
    """Select possible worlds matching the actual world's local observation."""
    worlds = tuple(worlds)
    if not any(world == actual_world for world in worlds):
        raise ValueError("actual_world must be included in worlds")
    actual_observation = observation(actual_world)
    return tuple(world for world in worlds if observation(world) == actual_observation)


def knows(worlds: Iterable, actual_world, observation: Callable, proposition: Callable):
    """Whether a proposition is true in every locally indistinguishable world."""
    possible = information_set(worlds, actual_world, observation)
    return all(bool(proposition(world)) for world in possible)


def action_is_known_safe(worlds, actual_world, observation, action, safe_after_action):
    """Epistemic action precondition: action is safe in every possible world."""
    return knows(
        worlds,
        actual_world,
        observation,
        lambda world: safe_after_action(world, action),
    )
