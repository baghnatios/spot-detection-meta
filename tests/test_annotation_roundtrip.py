
from __future__ import annotations
import pandas as pd
from spotmeta.validation.coordinates import (
    validate_crop_registry,
    assert_roundtrip_examples,
    normalize_registry_dataframe,
)

def test_crop_registry_roundtrip_examples():
    df = pd.DataFrame([{
        "dataset_id": "ds1",
        "well_id": "C8",
        "crop_id": "crop_test",
        "well_ymin_px": 10,
        "well_xmin_px": 20,
        "well_ymax_px": 42,
        "well_xmax_px": 68,
        "crop_height_px": 32,
        "crop_width_px": 48,
        "coord_frame_primary": "well",
        "coord_units": "pixel",
        "image_shape_y": 128,
        "image_shape_x": 128,
        "selection_tags": ["dense"],
        "selection_rationale": "unit test",
        "annotator_status": "proposed",
        "crop_registry_version": "vtest",
    }])
    df = normalize_registry_dataframe(df)
    validate_crop_registry(df)
    assert_roundtrip_examples(df)
