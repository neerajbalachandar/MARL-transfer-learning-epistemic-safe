"""Small shared helpers for reading BenchMARL scalar CSV files."""

import csv
from pathlib import Path


def read_scalars(path):
    values = {}
    with path.open(newline="") as source:
        for row in csv.reader(source):
            if len(row) >= 2:
                try:
                    values[int(float(row[0]))] = float(row[1])
                except ValueError:
                    continue
    return values


def find_scalars(experiment_dir):
    return {path.stem: path for path in Path(experiment_dir).rglob("*.csv") if "scalars" in path.parts}


def frame_lookup(scalars):
    path = scalars.get("counters_total_frames")
    return read_scalars(path) if path else {}


def frame_series(path, frames):
    values = read_scalars(path)
    points = [(frames[step], value) for step, value in values.items() if step in frames]
    return sorted(points)


def evaluation_rows(experiment_dir):
    rows = []
    for path in Path(experiment_dir).rglob("*_metrics.csv"):
        with path.open(newline="") as source:
            rows.extend(csv.DictReader(source))
    return rows


def save_figure(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
