"""One-step geometric action filter for the pinned VMAS navigation dynamics.

The filter checks a finite nominal model. It is not a formal safety guarantee:
it does not account for state-estimation error, unmodelled contacts, wall
clamping, action noise, or errors in the discrete dynamics.
"""

from dataclasses import dataclass

import torch


@dataclass
class ShieldResult:
    actions: torch.Tensor
    intervened: bool
    feasible: bool
    scale: float
    minimum_predicted_clearance: float


def predict_positions(positions, velocities, actions, dt=0.1, substeps=2, drag=0.25, mass=1.0):
    """Predict one VMAS navigation step using semi-implicit Euler.

    Pinned scenario settings use dt=.1, two substeps, drag=.25. VMAS applies
    drag once at the first substep, then constant force (action) at each
    substep. Agents have mass 1 and holonomic action force in this scenario.
    """
    if positions.shape != velocities.shape or positions.shape != actions.shape:
        raise ValueError("positions, velocities and actions must have matching [agents, 2] shapes")
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("inputs must have shape [agents, 2]")
    sub_dt = dt / substeps
    velocity = velocities.clone()
    position = positions.clone()
    for step in range(substeps):
        if step == 0:
            velocity = velocity * (1.0 - drag)
        velocity = velocity + actions * (sub_dt / mass)
        position = position + velocity * sub_dt
    return position


def _minimum_clearance(positions, radii):
    if positions.shape[0] < 2:
        return float("inf")
    indices = torch.triu_indices(positions.shape[0], positions.shape[0], 1, device=positions.device)
    distances = torch.linalg.vector_norm(positions[indices[0]] - positions[indices[1]], dim=-1)
    radii = torch.as_tensor(radii, dtype=positions.dtype, device=positions.device)
    return float((distances - radii[indices[0]] - radii[indices[1]]).min().item())


def filter_actions(
    positions,
    velocities,
    proposed_actions,
    radii,
    margin=0.02,
    dt=0.1,
    substeps=2,
    drag=0.25,
    action_limit=1.0,
    scale_steps=21,
):
    """Scale all nominal actions until predicted pair clearances meet margin.

    The global scale search is intentionally small and understandable. If no
    candidate is feasible, it returns zero force and reports infeasibility;
    zero force is a fallback, not a guarantee that existing momentum is safe.
    """
    if margin < 0 or action_limit <= 0 or scale_steps < 2:
        raise ValueError("invalid shield settings")
    proposed = proposed_actions.clamp(-action_limit, action_limit)
    radii = torch.as_tensor(radii, dtype=positions.dtype, device=positions.device)
    for scale in torch.linspace(1.0, 0.0, scale_steps, device=positions.device, dtype=positions.dtype):
        candidate = proposed * scale
        predicted = predict_positions(positions, velocities, candidate, dt, substeps, drag)
        clearance = _minimum_clearance(predicted, radii)
        if clearance >= margin:
            factor = float(scale.item())
            return ShieldResult(candidate, factor < 1.0, True, factor, clearance)
    fallback = torch.zeros_like(proposed)
    predicted = predict_positions(positions, velocities, fallback, dt, substeps, drag)
    return ShieldResult(fallback, True, False, 0.0, _minimum_clearance(predicted, radii))
