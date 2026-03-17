
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import pandas as pd
from .loaders import load_image_shape_only

def choose_discovery_root(
    candidates: Sequence[str],
    override: Optional[str] = None,
) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Configured data root override does not exist: {p}")
        return p

    existing = []
    for c in candidates:
        p = Path(c).expanduser().resolve()
        if p.exists():
            existing.append(p)

    if not existing:
        raise FileNotFoundError(
            "None of the configured data-root candidates exist. "
            "Update CFG['data_root_candidates'] or set CFG['data_root_override']."
        )

    scored = []
    for p in existing:
        n = 0
        for ext in ("*.tif", "*.tiff"):
            n += sum(1 for _ in p.rglob(ext))
        scored.append((n, -len(p.parts), str(p), p))
    scored.sort(reverse=True)
    return scored[0][-1]

def _extract_first(patterns: Sequence[str], text: str):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None

def inventory_image_files(
    root: Path,
    inventory_extensions: Sequence[str],
    well_regexes: Sequence[str],
    cycle_regexes: Sequence[str],
    channel_regexes: Sequence[str],
    skip_hidden_paths: bool = True,
    max_inventory_files: Optional[int] = None,
) -> pd.DataFrame:
    root = Path(root).resolve()
    rows: List[Dict[str, object]] = []

    def hidden(p: Path) -> bool:
        return any(part.startswith(".") for part in p.parts)

    count = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffixes = "".join(p.suffixes).lower()
        if not any(suffixes.endswith(ext.lower()) for ext in inventory_extensions):
            continue
        if skip_hidden_paths and hidden(p):
            continue

        rel = p.relative_to(root)
        text = str(rel).replace("\\", "/")

        row = {
            "file_path": str(p),
            "relative_path": text,
            "file_name": p.name,
            "suffixes": suffixes,
            "file_size_bytes": p.stat().st_size,
            "well_id": _extract_first(well_regexes, text),
            "cycle_id_raw": _extract_first(cycle_regexes, text),
            "channel_id_raw": _extract_first(channel_regexes, text),
        }
        rows.append(row)
        count += 1
        if max_inventory_files is not None and count >= max_inventory_files:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No image files were discovered under {root}. "
            "Check the configured data root and allowed extensions."
        )

    def _as_int(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        try:
            return int(v)
        except Exception:
            return pd.NA

    df["cycle_id"] = df["cycle_id_raw"].map(_as_int).astype("Int64")
    df["channel_id"] = df["channel_id_raw"].map(_as_int).astype("Int64")
    return df

def enrich_inventory_with_shapes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    shape_rows = []
    for path in out["file_path"].tolist():
        info = load_image_shape_only(path)
        shape_rows.append(info)
    shape_df = pd.DataFrame(shape_rows)
    out = pd.concat([out.reset_index(drop=True), shape_df.reset_index(drop=True)], axis=1)
    return out
