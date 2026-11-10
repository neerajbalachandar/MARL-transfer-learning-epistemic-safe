from __future__ import annotations

from pathlib import Path

from transfer_eval.plotting.dashboard import render_dashboard
from transfer_eval.reporting.export import write_pdf_summary
from transfer_eval.reporting.tables import write_csv, write_json


def build_report(rows: list[dict], output_dir: Path, formats: set[str], plot_paths: list[str] | None = None, overwrite: bool = False) -> list[Path]:
    paths = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if "html" in formats:
        path = render_dashboard(rows, output_dir / "dashboard.html", plot_paths, overwrite)
        paths.append(path)
    if "csv" in formats:
        path = output_dir / "cross_category_summary.csv"
        write_csv(path, rows, overwrite)
        paths.append(path)
    if "pdf" in formats:
        path = output_dir / "cross_category_summary.pdf"
        if path.exists() and not overwrite:
            # PDF output is not byte-stable across matplotlib versions; avoid silently replacing.
            raise FileExistsError(f"Refusing to replace existing PDF without --overwrite: {path}")
        write_pdf_summary(rows, path)
        paths.append(path)
    write_json(output_dir / "cross_category_summary.json", rows, overwrite)
    return paths
