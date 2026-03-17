from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import getpass
import json
import os
import platform
import uuid
from typing import Any

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"

def build_provenance_record(
    *,
    run_id: str,
    notebook_name: str,
    repo_root: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "notebook_name": notebook_name,
        "repo_root": repo_root,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "user": getpass.getuser(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_executable": os.sys.executable,
        "config": config,
    }

def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")
    return path
