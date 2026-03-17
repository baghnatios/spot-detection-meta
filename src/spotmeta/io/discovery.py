from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
import tifffile

WELL_PAT = re.compile(r"(?<![A-Z0-9])([A-H](?:[1-9]|1[0-2]))(?![A-Z0-9])", re.IGNORECASE)
CYCLE_PAT = re.compile(r"(?:cycle|round|r)(?:[_\-\s]?)(\d+)", re.IGNORECASE)
CHANNEL_PAT = re.compile(r"(?:channel|chan|ch|c)(?:[_\-\s]?)(\d+)", re.IGNORECASE)

def choose_discovery_root(candidates: list[str], override: str | None = None) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Configured data_root_override does not exist: {p}")
        return p

    existing = []
    for cand in candidates:
        p = Path(cand).expanduser().resolve()
        if p.exists():
            n_img = sum(1 for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".tif", ".tiff"})
            existing.append((n_img, p))
    if not existing:
        raise FileNotFoundError("None of the candidate data roots exists.")
    existing.sort(key=lambda t: (t[0], len(str(t[1]))), reverse=True)
    return existing[0][1]

def _extract_well(path: Path) -> str | None:
    joined = " ".join(path.parts[-4:]) + " " + path.name
    m = WELL_PAT.search(joined)
    return m.group(1).upper() if m else None

def _extract_cycle(path: Path) -> int | None:
    joined = " ".join(path.parts[-4:]) + " " + path.name
    m = CYCLE_PAT.search(joined)
    return int(m.group(1)) if m else None

def _extract_channel(path: Path) -> int | None:
    joined = " ".join(path.parts[-4:]) + " " + path.name
    m = CHANNEL_PAT.search(joined)
    return int(m.group(1)) if m else None

def inventory_image_files(
    root: str | Path,
    *,
    suffixes: Iterable[str] = (".tif", ".tiff"),
    recursive: bool = True,
    dataset_id: str = "dataset_default",
) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    suffixes = {s.lower() for s in suffixes}
    iterator = root.rglob("*") if recursive else root.glob("*")
    rows = []
    for path in iterator:
        if not path.is_file():
            continue
        if path.suffix.lower() not in suffixes:
            continue
        rows.append({
            "dataset_id": dataset_id,
            "file_path": str(path),
            "file_name": path.name,
            "suffix": path.suffix.lower(),
            "well_id": _extract_well(path),
            "cycle_id": _extract_cycle(path),
            "channel_id": _extract_channel(path),
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["dataset_id", "file_path", "file_name", "suffix", "well_id", "cycle_id", "channel_id"])
    return df.sort_values(["well_id", "cycle_id", "channel_id", "file_name"], na_position="last").reset_index(drop=True)

def load_image_array(path: str | Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)
    return arr

def _is_valid_tiff(path: str | Path) -> bool:
    try:
        import tifffile as _tf
        with _tf.TiffFile(str(path)):
            pass
        return True
    except Exception:
        return False

def _infer_yx(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim < 2:
        raise ValueError(f"Expected image with ndim>=2, got shape {arr.shape}")
    return int(arr.shape[-2]), int(arr.shape[-1])

def enrich_inventory_with_shapes(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        out = df.copy()
        out["image_shape_y"] = pd.Series(dtype="Int64")
        out["image_shape_x"] = pd.Series(dtype="Int64")
        out["image_ndim"] = pd.Series(dtype="Int64")
        out["image_dtype"] = pd.Series(dtype="object")
        return out

    rows = []
    skipped = []
    for rec in df.to_dict(orient="records"):
        try:
            arr = load_image_array(rec["file_path"])
            y, x = _infer_yx(arr)
            rec = dict(rec)
            rec["image_shape_y"] = y
            rec["image_shape_x"] = x
            rec["image_ndim"] = int(arr.ndim)
            rec["image_dtype"] = str(arr.dtype)
            rows.append(rec)
        except Exception as _e:
            skipped.append((rec["file_path"], str(_e)))
    if skipped:
        import warnings
        for _path, _reason in skipped:
            warnings.warn(f"Skipped non-readable file: {_path}\n  reason: {_reason}")
    return pd.DataFrame(rows)

def compute_projection(arr: np.ndarray, kind: str = "max") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        proj = arr
    elif arr.ndim >= 3:
        axes = tuple(range(arr.ndim - 2))
        if kind == "max":
            proj = np.max(arr, axis=axes)
        elif kind == "mean":
            proj = np.mean(arr, axis=axes)
        else:
            raise ValueError(f"Unsupported projection kind: {kind}")
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return np.asarray(proj, dtype=np.float32)
