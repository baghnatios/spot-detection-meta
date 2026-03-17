from __future__ import annotations

import pandas as pd

REQUIRED_CROP_COLS = [
    "crop_id", "dataset_id", "well_id",
    "well_ymin_px", "well_xmin_px", "well_ymax_px", "well_xmax_px",
    "selection_tags", "selection_rationale", "annotator_status", "crop_registry_version",
]
ALLOWED_STATUS = {"pending", "in_progress", "complete", "excluded"}

def normalize_registry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "selection_tags" in out.columns:
        out["selection_tags"] = out["selection_tags"].apply(
            lambda x: list(x) if isinstance(x, (list, tuple, set)) else ([] if pd.isna(x) else [str(x)])
        )
    for c in ["well_ymin_px", "well_xmin_px", "well_ymax_px", "well_xmax_px"]:
        if c in out.columns:
            out[c] = out[c].astype(int)
    return out

def validate_crop_registry(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CROP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Crop registry missing required columns: {missing}")
    d = normalize_registry_dataframe(df)
    if d["crop_id"].duplicated().any():
        raise ValueError("crop_id values must be unique")
    bad = d[~d["annotator_status"].isin(ALLOWED_STATUS)]
    if len(bad):
        raise ValueError(f"Invalid annotator_status values: {sorted(bad['annotator_status'].unique())}")
    for _, r in d.iterrows():
        if not (r["well_ymin_px"] < r["well_ymax_px"] and r["well_xmin_px"] < r["well_xmax_px"]):
            raise ValueError(f"Invalid crop bounds for crop_id={r['crop_id']}")
        if not isinstance(r["selection_tags"], list):
            raise ValueError(f"selection_tags must be list for crop_id={r['crop_id']}")

def assert_roundtrip_examples(df: pd.DataFrame, shape_lookup: dict[tuple[str, str], tuple[int, int]]) -> None:
    d = normalize_registry_dataframe(df)
    for _, r in d.iterrows():
        key = (r["dataset_id"], r["well_id"])
        if key not in shape_lookup:
            continue
        H, W = shape_lookup[key]
        if not (0 <= r["well_ymin_px"] < r["well_ymax_px"] <= H):
            raise ValueError(f"Crop y-bounds exceed image shape for {r['crop_id']}: {(r['well_ymin_px'], r['well_ymax_px'])} vs H={H}")
        if not (0 <= r["well_xmin_px"] < r["well_xmax_px"] <= W):
            raise ValueError(f"Crop x-bounds exceed image shape for {r['crop_id']}: {(r['well_xmin_px'], r['well_xmax_px'])} vs W={W}")
