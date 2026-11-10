from __future__ import annotations

from pathlib import Path


def diagnostic_availability(metric_records: list[dict]) -> dict[str, list[str]]:
    names = {row.get("metric_name", "") for row in metric_records}
    requirements = {
        "maptf": ["peer_option_selection_frequency", "option_duration", "termination_probability", "successor_prediction_loss", "embedding_loss", "peer_policy_kl"],
        "bayesian_task_reuse": ["source_policy_posterior", "probe_sample_count", "selection_confidence"],
        "portal": ["task_similarity", "selected_task_sequence", "stage_win_rate"],
        "gct_marl": ["smac_win_rate", "sgc_view_loss", "alignment_loss"],
        "zsc_eval": ["partner_conditioned_native_exports"],
        "bayes_tomop": ["opponent_strategy_posterior", "response_performance"],
        "epirl": ["wumpus_goal_completion"],
    }
    available = {}
    for method, signals in requirements.items():
        available[method] = [signal for signal in signals if any(signal in name for name in names)]
    return available


def plot_diagnostic_placeholder(availability: dict, output: Path) -> None:
    # No synthetic panels: missing method diagnostics are reported in JSON/HTML only.
    output.mkdir(parents=True, exist_ok=True)
