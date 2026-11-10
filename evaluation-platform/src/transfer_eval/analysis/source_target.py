from __future__ import annotations


def source_target_matrix(records: list[dict], metric_name: str) -> dict:
    cells = {}
    for row in records:
        if row.get("metric_name") != metric_name or row.get("value") is None:
            continue
        source, target = row.get("source_condition"), row.get("target_condition")
        if source is None or target is None:
            continue
        cells.setdefault((source, target), []).append(float(row["value"]))
    return {f"{source}->{target}": {"mean": sum(values) / len(values), "n_seeds": len(values)}
            for (source, target), values in sorted(cells.items())}
