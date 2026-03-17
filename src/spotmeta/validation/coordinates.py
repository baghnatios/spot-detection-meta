
from __future__ import annotations
from typing import Dict, Sequence, Iterable
import numpy as np
import pandas as pd

ALLOWED_SELECTION_TAGS = {
    "dense",
    "sparse",
    "dim",
    "bright",
    "close_pair",
    "edge_artifact",
    "detector_disagreement",
    "error_analysis",
}
ALLOWED_ANNOTATOR_STATUS = {
    "proposed",
    "reviewed",
    "accepted",
    "rejected",
    "annotated",
    "complete",
}

def crop_bounds_to_record(
    dataset_id: str,
    well_id: str,
    crop_id: str,
    y0: int,
    x0: int,
    h: int,
    w: int,
    image_shape_y: int,
    image_shape_x: int,
    selection_tags: Sequence[str],
    selection_rationale: str,
    crop_registry_version: str,
    annotator_status: str = "proposed",
) -> Dict[str, object]:
    y1 = int(y0) + int(h)
    x1 = int(x0) + int(w)
    tags = sorted({str(t) for t in selection_tags})
    return {
        "dataset_id": str(dataset_id),
        "well_id": str(well_id),
        "crop_id": str(crop_id),
        "well_ymin_px": int(y0),
        "well_xmin_px": int(x0),
        "well_ymax_px": int(y1),
        "well_xmax_px": int(x1),
        "crop_height_px": int(h),
        "crop_width_px": int(w),
        "coord_frame_primary": "well",
        "coord_units": "pixel",
        "image_shape_y": int(image_shape_y),
        "image_shape_x": int(image_shape_x),
        "selection_tags": tags,
        "selection_rationale": str(selection_rationale),
        "annotator_status": str(annotator_status),
        "crop_registry_version": str(crop_registry_version),
    }

def normalize_selection_tags(tags: object) -> list[str]:
    if isinstance(tags, str):
        items = [t.strip() for t in tags.split(",") if t.strip()]
    elif isinstance(tags, Iterable):
        items = [str(t).strip() for t in tags if str(t).strip()]
    else:
        raise TypeError(f"Unsupported selection_tags value: {type(tags)}")
    items = sorted(set(items))
    unknown = [t for t in items if t not in ALLOWED_SELECTION_TAGS]
    if unknown:
        raise ValueError(f"Unknown selection_tags values: {unknown}")
    return items

def validate_crop_registry(df: pd.DataFrame) -> None:
    required = [
        "dataset_id", "well_id", "crop_id",
        "well_ymin_px", "well_xmin_px", "well_ymax_px", "well_xmax_px",
        "crop_height_px", "crop_width_px",
        "coord_frame_primary", "coord_units",
        "image_shape_y", "image_shape_x",
        "selection_tags", "selection_rationale",
        "annotator_status", "crop_registry_version",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Crop registry missing required columns: {missing}")
    if df["crop_id"].duplicated().any():
        dupes = df.loc[df["crop_id"].duplicated(), "crop_id"].tolist()
        raise ValueError(f"Duplicate crop_id values found: {dupes[:10]}")
    if df[["dataset_id", "well_id", "crop_id"]].isna().any().any():
        raise ValueError("dataset_id, well_id, and crop_id must be non-null")
    if df["selection_rationale"].astype(str).str.len().eq(0).any():
        raise ValueError("selection_rationale must be non-empty for every crop")
    if not (df["coord_frame_primary"] == "well").all():
        raise ValueError("Notebook 01 crop registry must persist primary coordinates in well frame")
    if not (df["coord_units"] == "pixel").all():
        raise ValueError("coord_units must be 'pixel'")

    bad = df[
        (df["well_ymin_px"] < 0) |
        (df["well_xmin_px"] < 0) |
        (df["well_ymax_px"] > df["image_shape_y"]) |
        (df["well_xmax_px"] > df["image_shape_x"]) |
        (df["well_ymax_px"] <= df["well_ymin_px"]) |
        (df["well_xmax_px"] <= df["well_xmin_px"])
    ]
    if len(bad):
        raise ValueError(f"Found out-of-bounds or degenerate crops: {len(bad)} rows")

    h = df["well_ymax_px"] - df["well_ymin_px"]
    w = df["well_xmax_px"] - df["well_xmin_px"]
    if not (h == df["crop_height_px"]).all():
        raise ValueError("crop_height_px does not match ymax-ymin for all rows")
    if not (w == df["crop_width_px"]).all():
        raise ValueError("crop_width_px does not match xmax-xmin for all rows")

    for idx, row in df.iterrows():
        normalize_selection_tags(row["selection_tags"])
        status = str(row["annotator_status"])
        if status not in ALLOWED_ANNOTATOR_STATUS:
            raise ValueError(f"Invalid annotator_status at row {idx}: {status}")

def well_to_crop_coords(well_y_px, well_x_px, crop_origin_well_y_px, crop_origin_well_x_px):
    return (well_y_px - crop_origin_well_y_px, well_x_px - crop_origin_well_x_px)

def crop_to_well_coords(crop_y_px, crop_x_px, crop_origin_well_y_px, crop_origin_well_x_px):
    return (crop_y_px + crop_origin_well_y_px, crop_x_px + crop_origin_well_x_px)

def assert_roundtrip_examples(df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        points = [
            (row["well_ymin_px"], row["well_xmin_px"]),
            (row["well_ymax_px"] - 1, row["well_xmax_px"] - 1),
            ((row["well_ymin_px"] + row["well_ymax_px"]) // 2, (row["well_xmin_px"] + row["well_xmax_px"]) // 2),
        ]
        for wy, wx in points:
            cy, cx = well_to_crop_coords(wy, wx, row["well_ymin_px"], row["well_xmin_px"])
            wy2, wx2 = crop_to_well_coords(cy, cx, row["well_ymin_px"], row["well_xmin_px"])
            if (wy, wx) != (wy2, wx2):
                raise AssertionError(
                    f"Coordinate roundtrip failed for crop_id={row['crop_id']}: "
                    f"({wy}, {wx}) -> ({cy}, {cx}) -> ({wy2}, {wx2})"
                )

def normalize_registry_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "well_ymin_px","well_xmin_px","well_ymax_px","well_xmax_px",
        "crop_height_px","crop_width_px","image_shape_y","image_shape_x"
    ]:
        out[c] = out[c].astype(int)
    out["dataset_id"] = out["dataset_id"].astype(str)
    out["well_id"] = out["well_id"].astype(str)
    out["crop_id"] = out["crop_id"].astype(str)
    out["annotator_status"] = out["annotator_status"].astype(str)
    out["crop_registry_version"] = out["crop_registry_version"].astype(str)
    out["selection_rationale"] = out["selection_rationale"].astype(str)
    out["selection_tags"] = out["selection_tags"].map(normalize_selection_tags)
    return out
