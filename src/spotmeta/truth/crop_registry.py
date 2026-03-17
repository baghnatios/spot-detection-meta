from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import yaml

def summarize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["well_id", "n_files", "n_cycles", "n_channels", "image_shape_y", "image_shape_x"])
    grp = (
        df.groupby("well_id", dropna=False)
        .agg(
            n_files=("file_path", "size"),
            n_cycles=("cycle_id", lambda s: int(pd.Series(s).dropna().nunique())),
            n_channels=("channel_id", lambda s: int(pd.Series(s).dropna().nunique())),
            image_shape_y=("image_shape_y", "first"),
            image_shape_x=("image_shape_x", "first"),
        )
        .reset_index()
        .sort_values("well_id", na_position="last")
        .reset_index(drop=True)
    )
    return grp

def select_primary_well_images(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    work = df.copy()
    work["cycle_rank"] = work["cycle_id"].fillna(10**9)
    work["channel_rank"] = work["channel_id"].fillna(10**9)
    work = work.sort_values(["well_id", "cycle_rank", "channel_rank", "file_name"], na_position="last")
    primary = work.groupby("well_id", as_index=False).first()
    return primary.drop(columns=["cycle_rank", "channel_rank"])

def _local_stats(img: np.ndarray, y0: int, x0: int, h: int, w: int) -> dict:
    patch = img[y0:y0+h, x0:x0+w]
    if patch.size == 0:
        return {"mean": -np.inf, "std": -np.inf, "max": -np.inf, "q99": -np.inf, "edge": -np.inf, "closepair": -np.inf}
    gy, gx = np.gradient(patch.astype(np.float32), axis=(0, 1))
    edge = float(np.mean(np.hypot(gy, gx)))
    mx = float(np.max(patch))
    mean = float(np.mean(patch))
    std = float(np.std(patch))
    q99 = float(np.quantile(patch, 0.99))
    bright = patch >= np.quantile(patch, 0.995)
    closepair = float(np.sum(bright))
    return {"mean": mean, "std": std, "max": mx, "q99": q99, "edge": edge, "closepair": closepair}

def _candidate_grid(shape_y: int, shape_x: int, crop_h: int, crop_w: int, step: int):
    y_limit = max(1, shape_y - crop_h + 1)
    x_limit = max(1, shape_x - crop_w + 1)
    for y0 in range(0, y_limit, max(1, step)):
        for x0 in range(0, x_limit, max(1, step)):
            yield y0, x0
    if (shape_y - crop_h) not in range(0, y_limit, max(1, step)) or (shape_x - crop_w) not in range(0, x_limit, max(1, step)):
        yield max(0, shape_y - crop_h), max(0, shape_x - crop_w)

def _make_crop_id(dataset_id: str, well_id: str, y0: int, x0: int, y1: int, x1: int) -> str:
    raw = f"{dataset_id}|{well_id}|{y0}|{x0}|{y1}|{x1}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"crop_{well_id}_{digest}"

def propose_crops_for_projection(
    projection: np.ndarray,
    *,
    dataset_id: str,
    well_id: str,
    crop_size_yx: tuple[int, int] = (256, 256),
    max_crops: int = 12,
    edge_margin_px: int = 24,
    min_center_separation_px: int = 128,
    allow_overlap: bool = False,
) -> list[dict]:
    img = np.asarray(projection, dtype=np.float32)
    H, W = img.shape
    crop_h, crop_w = map(int, crop_size_yx)
    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)
    step = max(32, min(crop_h, crop_w) // 2)

    stats = []
    for y0, x0 in _candidate_grid(H, W, crop_h, crop_w, step):
        y1, x1 = y0 + crop_h, x0 + crop_w
        s = _local_stats(img, y0, x0, crop_h, crop_w)
        s.update({"y0": y0, "x0": x0, "y1": y1, "x1": x1})
        s["is_edge"] = int(y0 <= edge_margin_px or x0 <= edge_margin_px or y1 >= H - edge_margin_px or x1 >= W - edge_margin_px)
        stats.append(s)

    if not stats:
        return []

    sdf = pd.DataFrame(stats)
    pools = {
        "bright": sdf.sort_values(["q99", "max"], ascending=False),
        "dim": sdf.sort_values(["mean", "q99"], ascending=True),
        "dense": sdf.sort_values(["std", "q99"], ascending=False),
        "sparse": sdf.sort_values(["std", "mean"], ascending=True),
        "close_pair": sdf.sort_values(["closepair", "std"], ascending=False),
        "edge_artifact": sdf.sort_values(["is_edge", "edge", "q99"], ascending=False),
    }

    wanted_tags = ["bright", "dim", "dense", "sparse", "close_pair", "edge_artifact"]
    chosen = []

    def center(rec):
        return ((rec["y0"] + rec["y1"]) / 2.0, (rec["x0"] + rec["x1"]) / 2.0)

    def too_close(rec):
        cy, cx = center(rec)
        for prev in chosen:
            py, px = center(prev)
            if np.hypot(cy - py, cx - px) < min_center_separation_px:
                return True
            if not allow_overlap:
                if not (rec["x1"] <= prev["x0"] or rec["x0"] >= prev["x1"] or rec["y1"] <= prev["y0"] or rec["y0"] >= prev["y1"]):
                    return True
        return False

    for tag in wanted_tags:
        for _, rec in pools[tag].iterrows():
            rec = rec.to_dict()
            if too_close(rec):
                continue
            chosen.append(rec | {"primary_tag": tag})
            break

    sdf["composite"] = (
        sdf["q99"].rank(pct=True)
        + sdf["std"].rank(pct=True)
        + sdf["edge"].rank(pct=True)
        + sdf["closepair"].rank(pct=True)
    )
    for _, rec in sdf.sort_values("composite", ascending=False).iterrows():
        if len(chosen) >= max_crops:
            break
        rec = rec.to_dict()
        if too_close(rec):
            continue
        chosen.append(rec | {"primary_tag": "mixed"})

    records = []
    for rec in chosen[:max_crops]:
        y0, x0, y1, x1 = map(int, [rec["y0"], rec["x0"], rec["y1"], rec["x1"]])
        tags = sorted(set([rec["primary_tag"]]))
        if rec.get("is_edge", 0):
            tags.append("edge_artifact")
        if rec["q99"] >= sdf["q99"].quantile(0.8):
            tags.append("bright")
        if rec["mean"] <= sdf["mean"].quantile(0.2):
            tags.append("dim")
        if rec["std"] >= sdf["std"].quantile(0.8):
            tags.append("dense")
        if rec["std"] <= sdf["std"].quantile(0.2):
            tags.append("sparse")
        if rec["closepair"] >= sdf["closepair"].quantile(0.8):
            tags.append("close_pair")
        tags = sorted(set(tags))

        rationale = (
            f"image-driven initial crop; primary_tag={rec['primary_tag']}; "
            f"mean={rec['mean']:.3f}; std={rec['std']:.3f}; q99={rec['q99']:.3f}; edge={rec['edge']:.3f}; closepair={rec['closepair']:.1f}"
        )
        records.append({
            "crop_id": _make_crop_id(dataset_id, well_id, y0, x0, y1, x1),
            "dataset_id": dataset_id,
            "well_id": well_id,
            "well_ymin_px": y0,
            "well_xmin_px": x0,
            "well_ymax_px": y1,
            "well_xmax_px": x1,
            "selection_tags": tags,
            "selection_rationale": rationale,
        })
    return records

def deduplicate_crop_records(records: list[dict]) -> list[dict]:
    out = []
    seen = set()
    for rec in records:
        cid = rec["crop_id"]
        if cid in seen:
            continue
        seen.add(cid)
        out.append(rec)
    return out

def build_crop_registry(
    records: list[dict],
    *,
    crop_registry_version: str,
    annotator_status_default: str = "pending",
) -> pd.DataFrame:
    rows = []
    for rec in deduplicate_crop_records(records):
        row = dict(rec)
        row["annotator_status"] = row.get("annotator_status", annotator_status_default)
        row["crop_registry_version"] = row.get("crop_registry_version", crop_registry_version)
        rows.append(row)
    cols = [
        "crop_id", "dataset_id", "well_id",
        "well_ymin_px", "well_xmin_px", "well_ymax_px", "well_xmax_px",
        "selection_tags", "selection_rationale", "annotator_status", "crop_registry_version",
    ]
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return pd.DataFrame(columns=cols)
    return out[cols].sort_values(["well_id", "well_ymin_px", "well_xmin_px"]).reset_index(drop=True)

def append_manual_crops(df: pd.DataFrame, manual_records: list[dict], *, crop_registry_version: str) -> pd.DataFrame:
    if not manual_records:
        return df.copy()
    extra = build_crop_registry(
        manual_records,
        crop_registry_version=crop_registry_version,
        annotator_status_default="pending",
    )
    merged = pd.concat([df, extra], ignore_index=True)
    merged = merged.drop_duplicates(subset=["crop_id"], keep="first").reset_index(drop=True)
    return merged

def crop_registry_to_yaml_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for rec in df.to_dict(orient="records"):
        rec = dict(rec)
        rec["selection_tags"] = list(rec["selection_tags"])
        out.append(rec)
    return out

def write_crop_registry_yaml(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "spotmeta_crop_registry",
        "n_rows": int(len(df)),
        "records": crop_registry_to_yaml_records(df),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path

def read_crop_registry_yaml(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return pd.DataFrame(payload["records"])

def write_schema_contract_doc(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Crop registry schema contract",
        "",
        "Required fields:",
        "- crop_id",
        "- dataset_id",
        "- well_id",
        "- well_ymin_px",
        "- well_xmin_px",
        "- well_ymax_px",
        "- well_xmax_px",
        "- selection_tags",
        "- selection_rationale",
        "- annotator_status",
        "- crop_registry_version",
        "",
        "Coordinate contract:",
        "- bounds are in full-well pixel frame (well_y_px, well_x_px)",
        "- ymin/xmin inclusive, ymax/xmax exclusive",
        "- coord_frame_primary: well",
        "- coord_units: pixel",
        "- crop_origin_well_y_px = well_ymin_px",
        "- crop_origin_well_x_px = well_xmin_px",
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return path
