# Schema Contracts

This document defines the canonical persisted schemas for the project.
It is normative for all parquet outputs.

## Global invariants

Every canonical table must include:

- `run_id`
- `dataset_id`
- `well_id`
- `code_git_commit`
- `created_at_utc`
- relevant registry versions

All persisted coordinate columns must be frame-typed.

## candidate_raw.parquet

Primary key: `raw_detection_id`

Required columns:
- `raw_detection_id: string`
- `run_id: string`
- `dataset_id: string`
- `well_id: string`
- `method_id: string`
- `method_family: string`
- `source_view_id: string`
- `score_type: string`
- `score_is_calibrated: boolean`
- `detection_score_raw: float64 | null`
- `detection_score_norm: float64 | null`
- `well_y_px: float64`
- `well_x_px: float64`
- `coord_frame_primary: string`
- `sigma_px: float64 | null`
- `timing_sec: float64`

## candidate_union.parquet

Primary key: `union_id`

Required columns:
- `union_id: string`
- `run_id: string`
- `dataset_id: string`
- `well_id: string`
- `union_n_members: int32`
- `union_centroid_well_y_px: float64`
- `union_centroid_well_x_px: float64`
- `union_medoid_well_y_px: float64`
- `union_medoid_well_x_px: float64`
- `union_radius_px: float64`
- `union_bbox_ymin_px: float64`
- `union_bbox_xmin_px: float64`
- `union_bbox_ymax_px: float64`
- `union_bbox_xmax_px: float64`
- `top_method_id: string | null`
- `top_method_score_raw: float64 | null`
- `top_method_score_norm: float64 | null`
- `proposer_support_count: int32`
- `proposer_support_fraction: float64`
- `proposer_set_canonical: string`
- `family_support_count: int32`
- `family_set_canonical: string`
- `cluster_id: string | null`

## candidate_union_membership.parquet

Composite key: `(union_id, raw_detection_id)`

Required columns:
- `union_id: string`
- `raw_detection_id: string`
- `member_rank_by_score: int32 | null`
- `member_is_medoid: boolean`
- `member_distance_to_centroid_px: float64`
- `member_distance_to_medoid_px: float64`

## candidate_clusters.parquet

Primary key: `cluster_id`

Required columns:
- `cluster_id: string`
- `dataset_id: string`
- `well_id: string`
- `n_union_candidates: int32`
- `n_raw_detections: int32`
- `cluster_centroid_well_y_px: float64`
- `cluster_centroid_well_x_px: float64`
- `cluster_bbox_ymin_px: float64`
- `cluster_bbox_xmin_px: float64`
- `cluster_bbox_ymax_px: float64`
- `cluster_bbox_xmax_px: float64`
- `max_contrast: float64 | null`
- `max_proposer_support: float64 | null`
- `score_gap_top2: float64 | null`
- `support_gap_top2: float64 | null`
- `is_bimodal: boolean | null`
- `parent_component_area_px: float64 | null`

## annotations.parquet

Primary key: `annotation_id`

Required columns:
- `annotation_id: string`
- `dataset_id: string`
- `well_id: string`
- `crop_id: string`
- `annotator: string`
- `timestamp_utc: string`
- `label: string`
- `confidence: float64 | null`
- `raw_click_crop_y_px: float64`
- `raw_click_crop_x_px: float64`
- `raw_click_well_y_px: float64`
- `raw_click_well_x_px: float64`
- `refined_click_crop_y_px: float64 | null`
- `refined_click_crop_x_px: float64 | null`
- `refined_click_well_y_px: float64 | null`
- `refined_click_well_x_px: float64 | null`
- `refinement_policy_id: string | null`
- `annotation_registry_version: string`

Allowed frozen labels:
- `TRUE_SPOT`
- `UNCERTAIN`

## candidate_to_truth_match.parquet

Primary key: `match_id`

Required columns:
- `match_id: string`
- `union_id: string | null`
- `annotation_id: string | null`
- `match_status: string`
- `gt_label: string | null`
- `gt_distance_px: float64 | null`
- `matching_algorithm: string`
- `truth_match_radius_px: float64`
- `matching_registry_version: string`

Allowed `match_status` values:
- `matched_positive`
- `unmatched_candidate_negative`
- `unmatched_truth_false_negative`
- `excluded_uncertain`

## mega_feature_table.parquet

Primary key: `union_id`

Required columns:
- `union_id: string`
- `dataset_id: string`
- `well_id: string`
- `cluster_id: string | null`
- `well_y_px: float64`
- `well_x_px: float64`
- `in_annotated_crop: boolean`
- `annotation_coverage_status: string`
- feature columns per `feature_registry.yaml`

## training_supervision_table.parquet

Primary key: `(union_id, split_id)`

Required columns:
- `union_id: string`
- `split_id: string`
- `supervision_status: string`
- `target_binary: int8 | null`
- `target_source: string`
- `sample_weight: float64`
- model-eligible feature columns

## splits.parquet

Primary key: `split_id`

Required columns:
- `split_id: string`
- `dataset_id: string`
- `well_id: string`
- `spatial_block_id: string`
- `split_role: string`
- `split_registry_version: string`

Allowed `split_role` values:
- `train`
- `calibration`
- `test`

## model_predictions.parquet

Primary key: `prediction_id`

Required columns:
- `prediction_id: string`
- `union_id: string`
- `model_id: string`
- `model_version: string`
- `split_id: string | null`
- `prob_true_spot: float64`
- `decision_threshold: float64`
- `predicted_label: int8`
- `calibration_version: string | null`

## feature_stats.parquet

Primary key: `(feature_name, feature_registry_version)`

Required columns:
- `feature_name: string`
- `feature_group: string`
- `is_core_frozen: boolean`
- `non_null_fraction: float64`
- `constant_flag: boolean`
- `univariate_score: float64 | null`
- `correlation_pruned_flag: boolean`
- `importance_metric: float64 | null`
- `retained_for_model: boolean`
- `feature_registry_version: string`
