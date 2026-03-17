
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import tifffile

def load_image_shape_only(path: str | Path) -> Dict[str, object]:
    path = str(path)
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    ndim = arr.ndim
    shape = tuple(int(v) for v in arr.shape)
    if ndim < 2:
        raise ValueError(f"Expected image with at least 2 dimensions, got shape={shape} for {path}")
    return {
        "ndim": ndim,
        "shape": shape,
        "image_shape_y": int(shape[-2]),
        "image_shape_x": int(shape[-1]),
        "dtype": str(arr.dtype),
    }

def load_image_array(path: str | Path) -> np.ndarray:
    path = str(path)
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"Expected image with at least 2 dimensions, got {arr.shape} for {path}")
    return arr

def compute_projection(arr: np.ndarray, how: str = "max") -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3:
        if how == "max":
            return arr.max(axis=0).astype(np.float32)
        if how == "mean":
            return arr.mean(axis=0).astype(np.float32)
        raise ValueError(f"Unsupported projection mode: {how}")
    lead = int(np.prod(arr.shape[:-2]))
    arr2 = arr.reshape((lead, arr.shape[-2], arr.shape[-1]))
    if how == "max":
        return arr2.max(axis=0).astype(np.float32)
    if how == "mean":
        return arr2.mean(axis=0).astype(np.float32)
    raise ValueError(f"Unsupported projection mode: {how}")
