
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _normalize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        hi = float(img.max()) if img.size else 1.0
        lo = float(img.min()) if img.size else 0.0
    if hi <= lo:
        return np.zeros_like(img)
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)

def plot_crop_overlay(image2d: np.ndarray, crop_df: pd.DataFrame, well_id: str, title=None, figsize=(10,10)):
    img = _normalize(image2d)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap="gray", interpolation="nearest")
    for _, row in crop_df.iterrows():
        rect = Rectangle((row["well_xmin_px"], row["well_ymin_px"]), row["crop_width_px"], row["crop_height_px"], fill=False, linewidth=1.5)
        ax.add_patch(rect)
        label = f"{row['crop_id']}\n{','.join(row['selection_tags'])}"
        ax.text(row["well_xmin_px"], row["well_ymin_px"] - 2, label, fontsize=7, va="bottom", ha="left",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0))
    ax.set_title(title or f"Crop overlay — {well_id}")
    ax.set_axis_off()
    plt.tight_layout()
    return fig, ax

def plot_crop_gallery(image2d: np.ndarray, crop_df: pd.DataFrame, well_id: str, max_cols: int = 4, figsize_scale: float = 3.0):
    img = _normalize(image2d)
    n = len(crop_df)
    cols = min(max_cols, max(1, n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_scale * cols, figsize_scale * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    for ax, (_, row) in zip(axes, crop_df.iterrows()):
        y0, y1 = int(row["well_ymin_px"]), int(row["well_ymax_px"])
        x0, x1 = int(row["well_xmin_px"]), int(row["well_xmax_px"])
        tile = img[y0:y1, x0:x1]
        ax.imshow(tile, cmap="gray", interpolation="nearest")
        ax.set_title(f"{row['crop_id']}\n{','.join(row['selection_tags'])}", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Crop gallery — {well_id}", y=1.02)
    plt.tight_layout()
    return fig, axes
