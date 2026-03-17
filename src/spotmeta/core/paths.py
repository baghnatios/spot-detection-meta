
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional

def find_repo_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path.cwd()
    start = Path(start).resolve()
    markers = ["registries", "src", "notebooks", ".git"]
    for candidate in [start] + list(start.parents):
        if all((candidate / m).exists() for m in markers):
            return candidate
    raise FileNotFoundError(
        f"Could not locate repo root from {start}. "
        "Expected an ancestor containing registries/, src/, notebooks/, and .git/."
    )

def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path
