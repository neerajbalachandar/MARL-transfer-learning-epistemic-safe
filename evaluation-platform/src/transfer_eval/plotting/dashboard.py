from __future__ import annotations

from html import escape
from pathlib import Path


def render_dashboard(rows: list[dict], output: Path, plot_paths: list[str] | None = None, overwrite: bool = False) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["method_id", "benchmark_id", "target_condition", "transfer_axis", "evidence_status", "native_simulator",
               "native_interaction_unit", "zero_shot_retention_or_degradation", "matched_budget_advantage_over_scratch",
               "normalized_within_benchmark_learning_auc", "interactions_to_competence",
               "end_to_end_wall_clock_seconds", "target_only_wall_clock_seconds", "robustness_safety_evidence",
               "n_independent_training_seeds", "confidence_interval", "primary_competence_mean",
               "primary_competence_std", "missing_data_flags", "reproducibility_flags"]
    head = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, list):
                value = "; ".join(str(v) for v in value)
            cells.append(f"<td>{escape(str(value if value is not None else 'unavailable'))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    plots = "".join(f'<li><a href="{escape("../" + Path(path).as_posix())}">{escape(Path(path).name)}</a></li>' for path in (plot_paths or []))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>MARL transfer evaluation</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#19232d}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd5df;padding:.45rem;text-align:left}}th{{background:#edf2f7}}.note{{padding:1rem;background:#fff4d6}}</style></head><body>
<h1>MARL transfer evaluation dashboard</h1><p class="note"><b>Interpretation:</b> cross-category rows are normalized and benchmark-scoped. Raw rewards and native interaction counts are not ranked or pooled across simulators. Missing runs are unavailable, never zero.</p>
<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>
<h2>Generated normalized-data plots</h2><ul>{plots or '<li>No plot has sufficient normalized data yet.</li>'}</ul>
<p>Detailed metrics, costs, uncertainty, missing fields and provenance are in the adjacent JSON/CSV exports.</p></body></html>"""
    if output.exists() and output.read_text(encoding="utf-8") == html:
        return output
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace existing dashboard without --overwrite: {output}")
    output.write_text(html, encoding="utf-8")
    return output
