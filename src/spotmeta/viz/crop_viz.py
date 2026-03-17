from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

def plot_crop_overlay(
    image: np.ndarray,
    crop_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    for _, r in crop_df.iterrows():
        rect = Rectangle(
            (r["well_xmin_px"], r["well_ymin_px"]),
            r["well_xmax_px"] - r["well_xmin_px"],
            r["well_ymax_px"] - r["well_ymin_px"],
            fill=False,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(r["well_xmin_px"], r["well_ymin_px"], r["crop_id"][-6:], fontsize=6)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax

def plot_crop_gallery(
    image: np.ndarray,
    crop_df: pd.DataFrame,
    *,
    title: str = "",
    save_path: str | Path | None = None,
):
    n = len(crop_df)
    ncols = 3
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.set_axis_off()
    for ax, (_, r) in zip(axes, crop_df.iterrows()):
        patch = image[r["well_ymin_px"]:r["well_ymax_px"], r["well_xmin_px"]:r["well_xmax_px"]]
        ax.imshow(patch, cmap="gray")
        ax.set_title(f"{r['well_id']} | {','.join(r['selection_tags'])}")
        ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, axes
