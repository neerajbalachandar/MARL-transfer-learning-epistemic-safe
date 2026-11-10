from __future__ import annotations

from transfer_eval.normalization.units import assert_same_semantics


def matched_budget_advantage(transfer: list[dict], scratch: list[dict], budget: float, metric_name: str) -> dict:
    left = [row for row in transfer if row.get("metric_name") == metric_name and row.get("checkpoint_step") == budget]
    right = [row for row in scratch if row.get("metric_name") == metric_name and row.get("checkpoint_step") == budget]
    if not left or not right:
        return {"value": None, "missing_reason": "matched transfer and scratch seed/checkpoint data are not both available"}
    assert_same_semantics(left + right)
    left_mean = sum(row["seed_value"] for row in left) / len(left)
    right_mean = sum(row["seed_value"] for row in right) / len(right)
    return {"value": left_mean - right_mean, "transfer_mean": left_mean, "scratch_mean": right_mean,
            "budget": budget, "native_unit": left[0].get("native_unit"),
            "missing_reason": None}
