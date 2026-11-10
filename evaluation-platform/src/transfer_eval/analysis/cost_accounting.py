from __future__ import annotations

from collections import defaultdict


def phase_costs(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in records:
        groups[(row.get("run_id"), row.get("seed"), row.get("interaction_unit"))].append(row)
    out = []
    for (run_id, seed, unit), rows in groups.items():
        interactions = [row["native_interactions"] for row in rows if row.get("native_interactions") is not None]
        seconds = [row["wall_clock_seconds"] for row in rows if row.get("wall_clock_seconds") is not None]
        out.append({"run_id": run_id, "seed": seed, "interaction_unit": unit,
                    "phase_interaction_total": sum(interactions) if interactions else None,
                    "phase_wall_time_seconds_total": sum(seconds) if seconds else None,
                    "phases": rows})
    return out
