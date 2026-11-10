from __future__ import annotations

import csv
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict], overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old == content:
            return
        if not overwrite:
            raise FileExistsError(f"Refusing to replace non-identical output without --overwrite: {path}")
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, value, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace non-identical output without --overwrite: {path}")
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict], overwrite: bool = False, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or sorted({key for row in rows for key in row})
    from io import StringIO
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    if columns:
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    content = stream.getvalue()
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to replace non-identical output without --overwrite: {path}")
    path.write_text(content, encoding="utf-8")
