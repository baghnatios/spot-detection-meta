
from __future__ import annotations
import hashlib
import json
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

def make_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}"

def stable_hash_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def build_provenance_record(
    run_id: str,
    notebook_name: str,
    repo_root: str,
    config: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    rec = {
        "run_id": run_id,
        "notebook_name": notebook_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "repo_root": str(Path(repo_root).resolve()),
        "config_sha256": stable_hash_json(config),
        "config": config,
    }
    if extra:
        rec.update(extra)
    return rec
