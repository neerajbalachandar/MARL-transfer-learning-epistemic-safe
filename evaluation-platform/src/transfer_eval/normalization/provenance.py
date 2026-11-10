from pathlib import Path

from transfer_eval.manifest import sha256


def file_provenance(path: Path, root: Path) -> dict[str, str]:
    return {"source_file": path.resolve().relative_to(root.resolve()).as_posix(), "source_sha256": sha256(path)}
