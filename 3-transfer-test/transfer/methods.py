"""Compact algorithm helpers shared by the three transfer experiments."""

import math
from copy import deepcopy

import torch
from torch.distributions import Normal, kl_divergence

def freeze_policy(policy):
    """Return an immutable policy copy suitable for a source-policy library."""
    frozen = deepcopy(policy).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def option_choice(values, deterministic=True):
    """Choose the highest-valued peer option, with stable lowest-index ties."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty 1D tensor")
    if deterministic:
        return int(torch.argmax(values).item())
    return int(torch.distributions.Categorical(logits=values).sample().item())


def termination_probability(features, weight, bias):
    """Logistic option termination probability in [0, 1]."""
    return torch.sigmoid(torch.nn.functional.linear(features, weight, bias))


def continuous_policy_kl(source_loc, source_scale, receiver_loc, receiver_scale):
    """KL(source || receiver) for diagonal continuous action distributions."""
    if source_loc.shape != source_scale.shape:
        raise ValueError("source location and scale shapes differ")
    if receiver_loc.shape != receiver_scale.shape:
        raise ValueError("receiver location and scale shapes differ")
    if source_loc.shape != receiver_loc.shape:
        raise ValueError("source and receiver policy shapes differ")
    source = Normal(source_loc, source_scale.clamp_min(1e-6))
    receiver = Normal(receiver_loc, receiver_scale.clamp_min(1e-6))
    return kl_divergence(source, receiver).sum(-1).mean()


def bayes_update(prior, log_likelihood):
    """Stable normalized posterior update from per-hypothesis log likelihoods."""
    if prior.ndim != 1 or prior.shape != log_likelihood.shape:
        raise ValueError("prior and log_likelihood must be matching 1D tensors")
    if torch.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("prior must contain non-negative mass")
    log_prior = torch.where(prior > 0, prior.log(), -torch.inf)
    posterior = torch.softmax(log_prior + log_likelihood, dim=0)
    if not torch.isfinite(posterior).all():
        raise ValueError("likelihood assigned zero probability to every hypothesis")
    return posterior


def calibrated_bayes_update(prior, log_likelihood, probability_floor=1e-9):
    """Stable posterior update with a documented floor on prior mass.

    A zero prior remains impossible under the ordinary update. This calibrated
    variant instead floors and renormalizes the prior before observing the new
    evidence; the floor is explicit so a hypothesis cannot disappear forever.
    """
    prior = torch.as_tensor(prior, dtype=torch.float64)
    log_likelihood = torch.as_tensor(log_likelihood, dtype=torch.float64)
    if prior.ndim != 1 or prior.shape != log_likelihood.shape:
        raise ValueError("prior and log_likelihood must be matching 1D tensors")
    if probability_floor <= 0 or probability_floor >= 1 / prior.numel():
        raise ValueError("probability_floor must be positive and smaller than 1 / hypotheses")
    if torch.any(prior < 0) or not torch.isfinite(prior).all() or prior.sum() <= 0:
        raise ValueError("prior must be finite, non-negative, and have positive mass")
    if torch.isnan(log_likelihood).any() or torch.isposinf(log_likelihood).any():
        raise ValueError("log likelihoods may be finite or -inf, but not NaN or +inf")
    prior = prior.clamp_min(probability_floor)
    prior = prior / prior.sum()
    log_weight = prior.log() + log_likelihood
    if torch.isneginf(log_weight).all():
        raise ValueError("all hypotheses have zero likelihood for this observation")
    posterior = torch.softmax(log_weight, dim=0)
    return posterior / posterior.sum()


def gaussian_log_likelihood(value, mean, variance, variance_floor=1e-3):
    """Gaussian log likelihood with a floor for small validation variances."""
    variance = torch.as_tensor(variance).clamp_min(variance_floor)
    value, mean = torch.as_tensor(value), torch.as_tensor(mean)
    return -0.5 * (math.log(2 * math.pi) + variance.log() + (value - mean) ** 2 / variance)


def signal_log_likelihood(signal, model, variance_floor=1e-3):
    """Log likelihood for return (Gaussian), success and contact (Bernoulli)."""
    result = gaussian_log_likelihood(
        signal["return"], model["return_mean"], model["return_variance"], variance_floor
    )
    for key in ("success", "contact"):
        probability = torch.as_tensor(model[f"{key}_probability"]).clamp(1e-6, 1 - 1e-6)
        observed = torch.as_tensor(signal[key], dtype=probability.dtype)
        result = result + observed * probability.log() + (1 - observed) * torch.log1p(-probability)
    return result


def summarize_joint_signals(records, dirichlet_alpha=0.5, variance_floor=1e-3):
    """Fit a joint model of (return, success, contact) without independence.

    Success/contact patterns are modeled jointly as four categorical outcomes.
    Return is modeled by a Gaussian conditional on that joint outcome. Sparse
    patterns back off to the overall return distribution.
    """
    if not records:
        raise ValueError("at least one validation record is required")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")
    returns = torch.tensor([row["return"] for row in records], dtype=torch.float64)
    patterns = torch.tensor(
        [int(bool(row["success"])) * 2 + int(bool(row["contact"])) for row in records],
        dtype=torch.long,
    )
    counts = torch.bincount(patterns, minlength=4).to(torch.float64)
    pattern_probabilities = (counts + dirichlet_alpha) / (
        counts.sum() + 4 * dirichlet_alpha
    )
    global_mean = float(returns.mean())
    global_variance = float(returns.var(unbiased=False).clamp_min(variance_floor))
    conditional = []
    for pattern in range(4):
        values = returns[patterns == pattern]
        if values.numel() < 2:
            mean, variance = global_mean, global_variance
        else:
            mean = float(values.mean())
            variance = float(values.var(unbiased=False).clamp_min(variance_floor))
        conditional.append({"return_mean": mean, "return_variance": variance})
    return {
        "pattern_probabilities": pattern_probabilities.tolist(),
        "conditional_return": conditional,
        "dirichlet_alpha": float(dirichlet_alpha),
        "variance_floor": float(variance_floor),
        "model": "joint categorical(success,contact) and conditional Gaussian(return)",
    }


def joint_signal_log_likelihood(signal, model, probability_floor=1e-9):
    """Evaluate a signal under the correlated rollout-feature model."""
    pattern = int(bool(signal["success"])) * 2 + int(bool(signal["contact"]))
    probabilities = torch.as_tensor(model["pattern_probabilities"], dtype=torch.float64)
    probability = probabilities[pattern].clamp_min(probability_floor)
    conditional = model["conditional_return"][pattern]
    return probability.log() + gaussian_log_likelihood(
        signal["return"],
        conditional["return_mean"],
        conditional["return_variance"],
        model.get("variance_floor", 1e-3),
    )


def belief_information_gain(prior, posterior):
    """Return entropy reduction in nats from a probe batch."""
    prior = torch.as_tensor(prior, dtype=torch.float64).clamp_min(1e-12)
    posterior = torch.as_tensor(posterior, dtype=torch.float64).clamp_min(1e-12)
    if prior.shape != posterior.shape:
        raise ValueError("prior and posterior shapes differ")
    prior = prior / prior.sum()
    posterior = posterior / posterior.sum()
    entropy_before = -(prior * prior.log()).sum()
    entropy_after = -(posterior * posterior.log()).sum()
    return float((entropy_before - entropy_after).item())


def bpr_expected_improvement(belief, utility_mean, utility_std):
    """Select the source with greatest posterior probability of beating current best."""
    belief = torch.as_tensor(belief, dtype=torch.float64)
    means = torch.as_tensor(utility_mean, dtype=torch.float64)
    stds = torch.as_tensor(utility_std, dtype=torch.float64).clamp_min(1e-6)
    if means.ndim != 2 or means.shape[0] != belief.numel() or means.shape != stds.shape:
        raise ValueError("utility arrays must have shape [hypotheses, policies]")
    if not torch.isclose(belief.sum(), torch.tensor(1.0, dtype=belief.dtype)):
        raise ValueError("belief must be normalized")
    threshold = (belief[:, None] * means).sum(dim=0).max()
    z = (threshold - means) / stds
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return int(torch.argmax((belief[:, None] * (1 - cdf)).sum(dim=0)).item())


def moderate_candidates(scores, fixed_interval=None):
    """Return candidates in the interquartile return band, with nearest fallback."""
    names = list(scores)
    values = torch.tensor([scores[name] for name in names], dtype=torch.float64)
    if fixed_interval is None:
        low, high = torch.quantile(values, torch.tensor([0.25, 0.75], dtype=torch.float64))
    else:
        low, high = (torch.tensor(value, dtype=torch.float64) for value in fixed_interval)
        if low > high:
            raise ValueError("fixed difficulty interval lower bound exceeds upper bound")
    selected = [name for name, value in scores.items() if low <= value <= high]
    if selected:
        return selected, {"lower": float(low), "upper": float(high), "fallback": False}
    midpoint = (low + high) / 2
    closest = names[int(torch.argmin((values - midpoint).abs()).item())]
    return [closest], {"lower": float(low), "upper": float(high), "fallback": True}


def portal_aligned_mean_distance_similarity(candidate_observations, target_observations, components=5, reg=1e-4):
    """Negative aligned component-mean distance proxy (higher is more similar).

    This is not a GMM KL divergence. Mixture weights and covariance differences
    only affect component matching, not the final score.
    """
    from itertools import permutations
    import numpy as np

    candidate = np.asarray(candidate_observations, dtype=np.float64)
    target = np.asarray(target_observations, dtype=np.float64)
    if candidate.ndim != 2 or target.ndim != 2 or candidate.shape[1] != target.shape[1]:
        raise ValueError("candidate and target observations must have matching feature dims")
    combined = np.concatenate((candidate, target), axis=0)
    center = combined.mean(axis=0)
    scale = combined.std(axis=0).clip(min=1e-6)
    candidate = (candidate - center) / scale
    target = (target - center) / scale
    def fit_mixture(data):
        rng = np.random.default_rng(0)
        means = data[rng.choice(len(data), size=components, replace=False)].copy()
        global_covariance = np.cov(data, rowvar=False) + np.eye(data.shape[1]) * reg
        covariances = np.repeat(global_covariance[None], components, axis=0)
        weights = np.full(components, 1 / components)
        for _ in range(30):
            component_logs = []
            for weight, mean, covariance in zip(weights, means, covariances):
                sign, logdet = np.linalg.slogdet(covariance)
                if sign <= 0:
                    covariance = covariance + np.eye(data.shape[1]) * reg
                    sign, logdet = np.linalg.slogdet(covariance)
                inverse = np.linalg.inv(covariance)
                delta = data - mean
                distance = np.einsum("ni,ij,nj->n", delta, inverse, delta)
                component_logs.append(
                    np.log(max(weight, 1e-12))
                    - 0.5 * (data.shape[1] * np.log(2 * np.pi) + logdet + distance)
                )
            component_logs = np.stack(component_logs, axis=1)
            maximum = component_logs.max(axis=1, keepdims=True)
            log_sum = maximum + np.log(np.exp(component_logs - maximum).sum(axis=1, keepdims=True))
            responsibilities = np.exp(component_logs - log_sum)
            counts = responsibilities.sum(axis=0).clip(min=1e-8)
            weights = counts / counts.sum()
            means = responsibilities.T @ data / counts[:, None]
            fitted_covariances = []
            for index in range(components):
                delta = data - means[index]
                weighted = delta * responsibilities[:, index : index + 1]
                covariance = weighted.T @ delta / counts[index]
                fitted_covariances.append(covariance + np.eye(data.shape[1]) * reg)
            covariances = np.stack(fitted_covariances)
        return weights, means, covariances

    first_weights, first_means, first_covariances = fit_mixture(candidate)
    second_weights, second_means, second_covariances = fit_mixture(target)
    # Align components by minimum symmetric Mahalanobis cost, not by component index.
    costs = np.empty((components, components))
    for i in range(components):
        for j in range(components):
            covariance = (first_covariances[i] + second_covariances[j]) / 2
            delta = first_means[i] - second_means[j]
            costs[i, j] = delta @ np.linalg.solve(covariance, delta)
    assignment = min(
        permutations(range(components)),
        key=lambda order: sum(costs[i, j] for i, j in enumerate(order)),
    )
    rows, cols = np.arange(components), np.asarray(assignment)
    weights = (first_weights[rows] + second_weights[cols]) / 2
    distance = 0.0
    for i, j, weight in zip(rows, cols, weights):
        delta = first_means[i] - second_means[j]
        covariance = (first_covariances[i] + second_covariances[j]) / 2
        distance += weight * (delta @ np.linalg.solve(covariance, delta))
    return -float(distance)


# Kept for older experiment outputs and scripts. New code uses the precise name.
portal_gmm_similarity = portal_aligned_mean_distance_similarity
