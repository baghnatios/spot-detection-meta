# Crop registry schema contract

Required fields:
- crop_id
- dataset_id
- well_id
- well_ymin_px
- well_xmin_px
- well_ymax_px
- well_xmax_px
- selection_tags
- selection_rationale
- annotator_status
- crop_registry_version

Coordinate contract:
- bounds are in full-well pixel frame (well_y_px, well_x_px)
- ymin/xmin inclusive, ymax/xmax exclusive
- coord_frame_primary: well
- coord_units: pixel
- crop_origin_well_y_px = well_ymin_px
- crop_origin_well_x_px = well_xmin_px
