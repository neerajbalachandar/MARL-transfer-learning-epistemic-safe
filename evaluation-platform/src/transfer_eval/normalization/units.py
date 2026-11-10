from __future__ import annotations


def assert_same_semantics(records: list[dict], field: str = "benchmark_semantics_id") -> None:
    values = {row.get(field) for row in records if row.get(field) is not None}
    if len(values) > 1:
        raise ValueError(f"Refusing comparison across incompatible {field}: {sorted(values)}")


def assert_same_unit(records: list[dict], field: str = "native_unit") -> None:
    values = {row.get(field) for row in records if row.get(field) is not None}
    if len(values) > 1:
        raise ValueError(f"Refusing to combine unlike units in {field}: {sorted(values)}")


def safe_native_sum(records: list[dict], value_field: str, unit_field: str) -> float:
    assert_same_unit(records, unit_field)
    if any(row.get(value_field) is None for row in records):
        raise ValueError(f"Cannot sum {value_field}: one or more values are missing")
    return sum(float(row[value_field]) for row in records)
