# Registry Contracts

All registries are executable scientific contracts.
They must be machine-validated and versioned.

## Common fields for all registries

- `id`
- `family`
- `implementation`
- `parameters`
- `dependencies`
- `output_columns`
- `version`
- `status`
- `notes`

Allowed `status` values:
- `planned`
- `implemented`
- `validated`
- `deprecated`

## preprocessing_registry.yaml

Additional required fields:
- `input_modality`
- `output_semantics`
- `deterministic`
- `cache_policy`
- `valid_source_views`
- `comparable_across_crops`
- `comparable_across_wells`

## detector_registry.yaml

Additional required fields:
- `primitive_or_aggregator`
- `input_preprocessing_id`
- `input_projection_id`
- `score_type`
- `score_is_calibrated`
- `coordinate_semantics`
- `duplicate_policy`
- `expected_scale_px`
- `requires_cache`

## feature_registry.yaml

Additional required fields:
- `feature_name`
- `feature_group`
- `core_or_experimental`
- `source_view_id`
- `source_preprocessing_id`
- `statistic`
- `window_definition`
- `null_policy`
- `dtype`
- `units`
- `expected_directionality`

## annotation_registry.yaml

Additional required fields:
- `allowed_labels`
- `default_negative_policy`
- `refinement_policy`
- `exclusion_policy`
- `uncertain_policy`

## matching_registry.yaml

Additional required fields:
- `matching_algorithm`
- `truth_match_radius_px`
- `distance_metric`
- `one_to_one_enforced`
- `uncertain_handling`
- `tie_break_policy`
