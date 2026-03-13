# DEFINITIVE PLAN v6 — Research-Grade Contract Architecture + Concrete Implementation Inventory
## For reproducible candidate-union spot detection, manual truth matching, feature engineering, and GBDT-based local resolution

---

# 0. Purpose and authority

This document is the governing specification for the project.
It merges:

- the **contract architecture** from v5: repository semantics, immutable identities, frame-typed coordinates, canonical table schemas, deterministic matching and split rules, provenance, and notebook boundaries; and
- the **concrete implementation inventory** from v4: named preprocessing transforms, named detector universe, named feature families, approximate family-level column counts, and stage ordering.

This document is authoritative at three levels:

1. **Contract level** — identities, coordinate frames, schemas, provenance, matching semantics, split semantics, cache/rebuild logic.
2. **Implementation level** — concrete preprocessing maps, detector list, feature families, canonical artifacts, notebook responsibilities.
3. **Experimental level** — optional extensions beyond the frozen baseline.

If an implementation detail conflicts with the contract layer, the **contract layer wins**.

---

# 1. Repository architecture and top-level semantics

## 1.1 Canonical meaning of each top-level directory

- `src/spotmeta/` — all executable logic.
- `notebooks/` — orchestration, annotation, inspection, and reporting only.
- `registries/` — versioned declarative research objects describing preprocessing, detectors, features, annotation policy, and matching policy.
- `configs/` — runtime/project settings only; no registry objects belong here.
- `tables/` — canonical persisted machine-readable outputs.
- `artifacts/` — human-facing noncanonical outputs: galleries, overlays, reports, model cards, QA figures.
- `docs/` — architecture, contracts, protocols, experiment logs.
- `tests/` — deterministic contract tests and edge-case tests.

## 1.2 Required repository structure

```text
annotations/
  clicked_truth/
  crop_registry/
  exclusion_regions/

artifacts/
  galleries/
  manifests/
  mentor_reference/
  model_cards/
  reports/

configs/
  feature_config.yaml
  mentor_config.yaml
  model_config.yaml
  project_config.yaml

docs/
  DEFINITIVE_PLAN_v6.md
  annotation_protocol.md
  architecture.md
  experiment_log.md
  specs/
    schema_contracts.md
    registry_contracts.md
    evaluation_protocol.md
    runbook.md

notebooks/
  00_mentor_extraction_and_parity.ipynb
  01_crop_registry_and_data_loader.ipynb
  02_crop_truth_annotation.ipynb
  03_candidate_universe_generation.ipynb
  04_feature_engineering.ipynb
  05_truth_matching.ipynb
  06_model_training_and_local_resolution.ipynb
  07_full_well_inference.ipynb

registries/
  preprocessing_registry.yaml
  detector_registry.yaml
  feature_registry.yaml
  annotation_registry.yaml
  matching_registry.yaml

src/spotmeta/
  core/
  detect/
  features/
  io/
  mentor/
  model/
  pipeline/
  preprocess/
  schemas/
  truth/
  validation/
  viz/

tables/
  .gitkeep

tests/
  test_candidate_union.py
  test_feature_schema.py
  test_mentor_parity.py
  test_truth_matching.py
```

## 1.3 Naming rule

There must be exactly one registry namespace.
Any file named `*registry*.yaml` belongs in `registries/`, not `configs/`.

---

# 2. Identity model and coordinate contracts

## 2.1 Frame-typed coordinates

No canonical table may store unlabeled `x`/`y` columns as its only persisted coordinate representation.
All coordinates are frame-typed.

Required coordinate frames:

- `crop_y_px`, `crop_x_px` — coordinates local to an annotated crop.
- `well_y_px`, `well_x_px` — coordinates in the full-well reference frame.
- `panel_y_px`, `panel_x_px` — only if a detector or transform is panel-local before projection.

Required metadata fields wherever coordinates appear:

- `coord_frame_primary`
- `coord_units = pixel`
- `image_shape_y`
- `image_shape_x`
- `crop_origin_well_y_px` and `crop_origin_well_x_px` if crop-local coordinates are stored

## 2.2 Immutable identifiers

Every persisted entity has a stable primary key.

- `run_id`
- `raw_detection_id`
- `union_id`
- `cluster_id`
- `annotation_id`
- `match_id`
- `split_id`
- `prediction_id`

### Deterministic construction rules

- `run_id` is created once per pipeline execution and propagated everywhere.
- `raw_detection_id` is deterministic from `(run_id, dataset_id, well_id, method_id, source_view_id, sorted_output_index)`.
- `union_id` is deterministic from the sorted membership set of `raw_detection_id` values.
- `cluster_id` is deterministic from the sorted membership set of `union_id` values.
- `annotation_id` is deterministic from `(dataset_id, well_id, crop_id, annotator, timestamp_utc, raw_click_well_y_px, raw_click_well_x_px)`.
- `match_id` is deterministic from `(union_id, annotation_id, matching_registry_version)`.

Connected-component ordering must not define IDs.

## 2.3 Provenance and rebuild invariants

Every canonical table carries:

- `run_id`
- `dataset_id`
- `well_id`
- `project_config_version`
- `preprocessing_registry_version`
- `detector_registry_version`
- `feature_registry_version`
- `annotation_registry_version`
- `matching_registry_version`
- `code_git_commit`
- `source_image_fingerprint`
- `crop_registry_version` when applicable
- `merge_radius_px`
- `truth_match_radius_px` when applicable
- `created_at_utc`

Any change to an upstream governing object invalidates downstream derived artifacts.

---

# 3. Canonical table system

## 3.1 Canonical persisted tables

1. `candidate_raw.parquet`
2. `candidate_union.parquet`
3. `candidate_union_membership.parquet`
4. `candidate_clusters.parquet`
5. `annotations.parquet`
6. `candidate_to_truth_match.parquet`
7. `mega_feature_table.parquet`
8. `training_supervision_table.parquet`
9. `splits.parquet`
10. `model_predictions.parquet`
11. `feature_stats.parquet`
12. `run_manifest.json` or `run_manifest.parquet`

## 3.2 `candidate_raw.parquet`

One row per detector-emitted candidate point.

Required columns:

- `raw_detection_id`
- `run_id`
- `dataset_id`
- `well_id`
- `method_id`
- `method_family`
- `source_view_id`
- `score_type`
- `score_is_calibrated`
- `detection_score_raw`
- `detection_score_norm`
- `well_y_px`
- `well_x_px`
- `coord_frame_primary`
- `sigma_px` when scale-specific
- `timing_sec`
- provenance block from §2.3

## 3.3 `candidate_union.parquet`

One row per radius-graph merged union candidate.

Required columns:

- `union_id`
- `run_id`
- `dataset_id`
- `well_id`
- `union_n_members`
- `union_centroid_well_y_px`
- `union_centroid_well_x_px`
- `union_medoid_well_y_px`
- `union_medoid_well_x_px`
- `union_radius_px`
- `union_bbox_ymin_px`
- `union_bbox_xmin_px`
- `union_bbox_ymax_px`
- `union_bbox_xmax_px`
- `top_method_id`
- `top_method_score_raw`
- `top_method_score_norm`
- `proposer_support_count`
- `proposer_support_fraction`
- `proposer_set_canonical`
- `family_support_count`
- `family_set_canonical`
- `cluster_id` if assigned
- provenance block from §2.3

### Deterministic union algorithm

1. Build KD-tree over `candidate_raw` well coordinates.
2. Connect detections with Euclidean distance `<= merge_radius_px`.
3. Each connected component becomes one union candidate.
4. Membership is sorted by `raw_detection_id`.
5. `union_id` is deterministically hashed from the sorted membership set.
6. `union_medoid` is the member minimizing summed distance to all other members.
7. `union_centroid` is the arithmetic mean of member coordinates unless a future weighted policy is explicitly versioned.
8. `top_method_*` is selected by descending `detection_score_norm`, then descending `detection_score_raw`, then lexicographic `raw_detection_id`.

## 3.4 `candidate_union_membership.parquet`

One row per `(union_id, raw_detection_id)` relation.

Required columns:

- `union_id`
- `raw_detection_id`
- `member_rank_by_score`
- `member_is_medoid`
- `member_distance_to_centroid_px`
- `member_distance_to_medoid_px`

## 3.5 `candidate_clusters.parquet`

One row per higher-order local candidate cluster.

Required columns:

- `cluster_id`
- `dataset_id`
- `well_id`
- `n_union_candidates`
- `n_raw_detections`
- `cluster_centroid_well_y_px`
- `cluster_centroid_well_x_px`
- `cluster_bbox_ymin_px`
- `cluster_bbox_xmin_px`
- `cluster_bbox_ymax_px`
- `cluster_bbox_xmax_px`
- `max_contrast`
- `max_proposer_support`
- `score_gap_top2`
- `support_gap_top2`
- `is_bimodal`
- `parent_component_area_px`

## 3.6 `annotations.parquet`

One row per human annotation event.

Required columns:

- `annotation_id`
- `dataset_id`
- `well_id`
- `crop_id`
- `annotator`
- `timestamp_utc`
- `label`
- `confidence`
- `raw_click_crop_y_px`
- `raw_click_crop_x_px`
- `raw_click_well_y_px`
- `raw_click_well_x_px`
- `refined_click_crop_y_px`
- `refined_click_crop_x_px`
- `refined_click_well_y_px`
- `refined_click_well_x_px`
- `refinement_policy_id`
- `annotation_registry_version`

Frozen baseline labels:

- `TRUE_SPOT`
- `UNCERTAIN`

`NOT_SPOT` remains optional and noncanonical unless a registry version explicitly enables it.

## 3.7 `candidate_to_truth_match.parquet`

One row per candidate/annotation matching outcome.

Required columns:

- `match_id`
- `union_id`
- `annotation_id`
- `match_status`
- `gt_label`
- `gt_distance_px`
- `matching_algorithm`
- `truth_match_radius_px`
- `matching_registry_version`

Allowed `match_status` values:

- `matched_positive`
- `unmatched_candidate_negative`
- `unmatched_truth_false_negative`
- `excluded_uncertain`

## 3.8 `mega_feature_table.parquet`

One row per `union_id` with feature values plus provenance.

Required columns:

- `union_id`
- `dataset_id`
- `well_id`
- `cluster_id`
- `well_y_px`
- `well_x_px`
- `in_annotated_crop`
- `annotation_coverage_status`
- all frozen feature columns
- provenance block from §2.3

## 3.9 `training_supervision_table.parquet`

Derived join of `mega_feature_table`, `candidate_to_truth_match`, and `splits`.

Required columns:

- `union_id`
- `split_id`
- `supervision_status`
- `target_binary`
- `target_source`
- `sample_weight`
- all model-eligible feature columns

## 3.10 `splits.parquet`

Required columns:

- `split_id`
- `dataset_id`
- `well_id`
- `spatial_block_id`
- `split_role` in `{train, calibration, test}`
- `split_registry_version`

## 3.11 `model_predictions.parquet`

Required columns:

- `prediction_id`
- `union_id`
- `model_id`
- `model_version`
- `split_id`
- `prob_true_spot`
- `decision_threshold`
- `predicted_label`
- `calibration_version`

## 3.12 `feature_stats.parquet`

Required columns:

- `feature_name`
- `feature_group`
- `is_core_frozen`
- `non_null_fraction`
- `constant_flag`
- `univariate_score`
- `correlation_pruned_flag`
- `importance_metric`
- `retained_for_model`
- `feature_registry_version`

---

# 4. Registry system

Every registry entry is an executable contract, not a note.
Each entry must declare:

- `id`
- `family`
- `implementation`
- `parameters`
- `dependencies`
- `output_columns`
- `version`
- `status`
- `notes`

Allowed status values:

- `planned`
- `implemented`
- `validated`
- `deprecated`

---

# 5. Preprocessing registry

Every preprocessing transform produces a 2D image aligned to a declared coordinate frame and suitable for feature sampling and, where appropriate, detector input.

## 5.1 Existing transforms retained from v4

| ID | Function | Output class |
|----|----------|--------------|
| `raw` | identity | raw image |
| `norm01` | percentile normalization | normalized intensity |
| `white_local` | local whitening | whitened image |
| `peak_bg` | gaussian difference | spot-enhanced image |
| `z_local` | local z-score | locally normalized image |
| `highpass` | raw − gaussian blur | soft high-pass |
| `local_mad` | robust MAD scale | noise estimate |
| `white` | gaussian local whitening | clipped whiten |
| `norm01_ppi` | per-panel normalization | cached normalized panel image |

## 5.2 Existing response maps retained from v4

| Map ID | Implementation class |
|--------|----------------------|
| `log_response` | negative Gaussian Laplace |
| `dog_response` | Gaussian difference |
| `atrous_wavelet_bandpass` | multi-scale wavelet bandpass |
| `white_tophat_response` | morphological opening subtraction |
| `prominence_response` | blurred minus median background |

## 5.3 Additional preprocessing transforms retained from v4

| ID | Description |
|----|-------------|
| `rolling_ball_{R}` | rolling-ball background estimate / residual |
| `opening_by_recon` | opening-by-reconstruction subtraction |
| `wavelet_product` | multiplicative wavelet response |
| `radial_symmetry_map` | gradient-vote radial symmetry map |
| `hessian_blobness` | Hessian determinant / blobness map |
| `log_normalized` | sigma-squared normalized LoG |

## 5.4 Projection views

Each declared transform may generate the following projection views where scientifically justified:

- `maxproj`
- `meanproj`

Primary frozen views named in v4 and adopted here:

- `raw_maxproj`
- `raw_meanproj`
- `norm01_maxproj`
- `norm01_meanproj`
- `white_local_maxproj`
- `white_local_meanproj`
- `peak_bg_maxproj`
- `peak_bg_meanproj`
- `z_local_maxproj`
- `z_local_meanproj`

## 5.5 Contract rules for preprocessing

Each preprocessing entry must additionally declare:

- `input_modality`
- `output_semantics`
- `deterministic`
- `cache_policy`
- `valid_source_views`
- `comparable_across_crops`
- `comparable_across_wells`

---

# 6. Detector registry

All detectors emit candidates under a common output contract and are versioned as registry entries.

## 6.1 Existing detector universe retained from v4

- `mentor_v1`
- `mentor_v2`
- `consensus_v2`
- `matched_filter_v2`
- `local_expansion`
- `restrained_psf`
- `bright_rescue`
- `sk_log`
- `sk_dog`
- `sk_doh`
- `bigfish_style`
- `trackpy`
- `proj_local_max_raw`
- `proj_local_max_norm`
- `proj_log_norm`
- `proj_log_white`
- `proj_peakbg_max`
- `proj_zlocal_max`
- `atrous_wavelet`
- `morph_tophat`
- `multiscale_log`

## 6.2 Additional detector families retained from v4

| ID | Description |
|----|-------------|
| `radial_symmetry` | radial symmetry detector |
| `wavelet_product` | multiplicative wavelet detector |
| `hessian_blobness` | Hessian determinant blob detector |
| `rolling_ball_residual` | residual after rolling-ball background |
| `opening_recon_residual` | residual after opening-by-reconstruction |

## 6.3 Detector output contract

Each detector must emit:

- `pts : (N, 2)` candidate coordinates
- `method_id`
- `family`
- `source_view_id`
- `timing_sec`
- `score` if meaningful
- `score_type`
- `coordinate_semantics`

Allowed `coordinate_semantics` include:

- `local_maximum`
- `fit_centroid`
- `radial_center`
- `medoid_like`

## 6.4 Detector contract additions required for research-grade operation

Each detector registry entry must also declare:

- `primitive_or_aggregator`
- `input_preprocessing_id`
- `input_projection_id`
- `score_is_calibrated`
- `duplicate_policy`
- `expected_scale_px`
- `requires_cache`

---

# 7. Feature registry

The feature system is split into:

- `core_frozen` — minimal scientifically defensible set required for the first reproducible end-to-end baseline.
- `experimental` — additional families retained for later ablation and expansion.

Total feature count retained from v4 is approximately **~560 features before normalization / derived copies are fully resolved**. The family-level counts below are authoritative for planning, not a substitute for the exact registry columns.

## 7.1 Category 1 — Patch photometry

Base named quantities retained from v4:

- `center`
- `inner_mean`
- `inner_max`
- `inner_min`
- `inner_std`
- `inner_skew`
- `inner_kurtosis`
- `ring_mean`
- `ring_std`
- `ring_max`
- `ring_median`
- `contrast`
- `SNR_local`
- `SNR_mad`

Approximate total retained from v4: **~282 features**.

Interpretation:
These are instantiated across selected views, windows, and panels under the exact feature registry.

## 7.2 Category 2 — Detector response features

Retained response-feature families:

- `LoG`
- `normalized_LoG`
- `DoG`
- `wavelet_bandpass`
- `wavelet_detail_levels`
- `wavelet_product`
- `top_hat`
- `rolling_ball_residual`
- `opening_by_reconstruction_residual`
- `PSF_fit_metrics`
- `Hessian_eigenvalues`
- `radial_symmetry`
- `matched_filter_NCC`
- `prominence`

Approximate total retained from v4: **~78 features**.

## 7.3 Category 3 — Spatial / geometric / split / cluster features

Retained feature families:

- `neighbor_counts`
- `nearest_neighbor_distances`
- `local_density`
- `border_distance`
- `tile_context`
- `saddle_depth`
- `valley_contrast`
- `nms_margin`
- `nearest_stronger_neighbor`
- `local_maxima_flags`
- `watershed_basin_area`
- `dual_peak_separability`
- `line_profile_dip`
- cluster-derived summaries

Approximate total retained from v4: **~40 features**.

## 7.4 Category 4 — Multi-proposer consensus

Retained feature families:

- binary vote per proposer
- proposer support fraction
- family counts
- continuous proposer scores
- consensus centroid spread
- consensus radius

Approximate total retained from v4: **~41 features**.

## 7.5 Category 5 — Barcode / multi-channel features

Retained feature families:

- per-panel photometry
- barcode consistency metrics

Approximate total retained from v4: **~40 features**.

## 7.6 Category 6 — Crop / image regime features

Retained feature families:

- `crop_background_mean`
- `crop_background_std`
- `crop_dynamic_range`
- `crop_noise_sigma`
- `crop_entropy`
- `crop_candidate_density`
- `crop_edge_density`
- `crop_illumination_nonuniformity`

Approximate total retained from v4: **~40 features**.

## 7.7 Category 7 — Symmetry / shape

Retained feature families:

- `radial_symmetry_score`
- `radial_symmetry_residual`
- `Hu_moments`
- `eccentricity`
- `PSF_fit_parameters`

Approximate total retained from v4: **~16 features**.

## 7.8 Category 8 — Interaction / derived / rank features

Retained feature families:

- `cross_view_ratios`
- `feature_percentile_ranks`
- targeted interactions
- `MAD_standardized_copies`

Approximate total retained from v4: **~26 features**.

## 7.9 Feature contract rules

Every feature entry must declare:

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
- `dependencies`
- `expected_directionality` if scientifically meaningful

## 7.10 Immediate implementation rule

The exact registry must freeze a **minimal core_frozen subset first**. The broader ~560-feature universe remains available but must not be treated as a single monolithic baseline until validated by non-null checks, constant-column checks, leakage review, and ablation.

---

# 8. Annotation and crop registry

## 8.1 Crop selection strategy retained from v4

Crops are intentionally selected to cover:

- dense candidate regions
- sparse regions
- dim regions
- bright regions
- close-pair spots
- detector-disagreement regions
- edge artifacts

Expected truth density retained from v4: **~500–1500 truth points per well**.

## 8.2 Crop registry contract

Each crop entry must include:

- `crop_id`
- `dataset_id`
- `well_id`
- `well_ymin_px`
- `well_xmin_px`
- `well_ymax_px`
- `well_xmax_px`
- `selection_tags`
- `selection_rationale`
- `annotator_status`
- `crop_registry_version`

## 8.3 Annotation policy

Frozen baseline annotation labels:

- `TRUE_SPOT`
- `UNCERTAIN`

Negatives are derived automatically from unmatched candidates, not manually clicked by default.

Required stored fields retained and hardened from v4:

- `raw_click_y/x`
- `refined_click_y/x`
- `confidence`
- `annotator`
- `crop_id`
- `timestamp`

---

# 9. Truth matching

## 9.1 Frozen matching policy

Use one-to-one Hungarian assignment with distance threshold `<= truth_match_radius_px`.

Principles:

- one candidate ↔ one truth
- `UNCERTAIN` annotations are excluded from binary supervision unless a registry version explicitly states otherwise
- unmatched union candidates in annotated territory become negatives
- unmatched truth points become false negatives for evaluation

## 9.2 Matching outputs

Required outputs retained and hardened from v4:

- `gt_label`
- `gt_distance_px`
- `gt_click_well_y_px`
- `gt_click_well_x_px`

---

# 10. Training and evaluation plan

## 10.1 Dataset construction retained from v4

Training uses pooled cross-well data.

Report at minimum:

- pooled metrics
- per-well metrics
- per-crop metrics

## 10.2 Split strategy retained and hardened from v4

Spatial block cross-validation with frozen split proportions:

- train: `60%`
- calibration: `20%`
- test: `20%`

Splits must be stored in `splits.parquet` and reused rather than regenerated ad hoc.

## 10.3 Models retained from v4

Primary model:

- `GBDT`

Baselines:

- `RandomForest`
- `LogisticRegression`

Production target:

- feature-pruned calibrated `GBDT`

## 10.4 Evaluation metrics retained from v4

Primary metrics:

- precision
- recall
- F1
- ROC-AUC
- average precision

Required stratified analyses:

- isolated spots
- close pairs
- cluster edges

## 10.5 Additional research-grade requirements

Also persist:

- calibration curves
- precision-recall curves
- threshold selection reports
- error taxonomy summaries
- feature ablation comparisons

---

# 11. Compute budget, caching, and runtime expectations

These are planning estimates, not claimed benchmarks.
They are intended for a constrained laptop workflow and must be replaced by measured runtimes after the first dry run.

## 11.1 Planning assumptions

- dataset scope initially limited to annotated crops plus wells `C8` and `D8`
- preprocessing maps cached per well and per view
- raw detector outputs cached before union
- feature extraction performed on frozen union candidates
- no deep learning training in the baseline

## 11.2 Expected runtime envelope on a memory-constrained laptop

### Crop-only debug runs

- preprocessing maps: **minutes**
- detector sweep on a small crop set: **minutes to tens of minutes**
- union + feature table build for crop subset: **minutes**
- matching + training iteration: **minutes**

### Full-well candidate generation on a small number of wells

- detector sweep across all declared methods: **tens of minutes to multiple hours**, depending on cache reuse and detector implementation quality
- feature extraction for full candidate universe: **tens of minutes to hours**
- pooled model training with the frozen baseline feature set: **minutes to tens of minutes**

## 11.3 Operational rule

Until empirical timings are collected, the project must support two execution modes:

- `debug_small` — crop-limited, low-cost, contract-validation mode
- `full_research` — full candidate-universe mode with caches and persisted manifests

## 11.4 Required timing instrumentation

Persist timing at:

- preprocessing step
- per-detector execution
- union construction
- feature family computation
- matching
- training
- inference

---

# 12. Notebook responsibilities

## 12.1 `00_mentor_extraction_and_parity.ipynb`

Only:

- mentor reference extraction
- parity checks
- mentor reference artifacts

## 12.2 `01_crop_registry_and_data_loader.ipynb`

Only:

- crop registry creation and inspection
- image loading and coordinate validation

## 12.3 `02_crop_truth_annotation.ipynb`

Only:

- annotation interface
- annotation export validation

## 12.4 `03_candidate_universe_generation.ipynb`

Only:

- preprocessing cache generation
- detector execution
- `candidate_raw`
- `candidate_union`
- `candidate_clusters`

## 12.5 `04_feature_engineering.ipynb`

Only:

- feature map generation
- `mega_feature_table`

No labels are constructed here.

## 12.6 `05_truth_matching.ipynb`

Only:

- candidate-to-truth matching
- `candidate_to_truth_match`
- `training_supervision_table`

## 12.7 `06_model_training_and_local_resolution.ipynb`

Only:

- split loading
- model fitting
- calibration
- evaluation
- feature statistics and pruning

## 12.8 `07_full_well_inference.ipynb`

Only:

- frozen-model application to full wells
- prediction export
- inference reports

---

# 13. Tests

Minimum required tests:

- `test_preprocess_shape_and_finiteness.py`
- `test_detector_output_contract.py`
- `test_candidate_union.py`
- `test_union_determinism.py`
- `test_feature_schema.py`
- `test_truth_matching.py`
- `test_annotation_roundtrip.py`
- `test_split_leakage.py`
- `test_model_training_reproducibility.py`

---

# 14. What is frozen now versus later

## 14.1 Frozen now

- repository structure
- identity model
- coordinate contracts
- canonical tables
- provenance fields
- union algorithm semantics
- annotation/matching policy
- split policy
- notebook boundaries
- detector universe list from v4
- preprocessing inventory from v4
- feature family inventory from v4

## 14.2 To be frozen next

- exact feature column registry names for the `core_frozen` baseline
- exact detector-to-view mapping
- exact cache manifests
- exact measured runtimes and memory profiles

## 14.3 Explicit non-goals for the baseline

Retained from earlier planning discipline:

- no deep learning detector stack in the frozen baseline
- no denoising/training-heavy method dependency for the baseline
- no blind polynomial feature explosion
- no notebook-owned hidden schema logic

---

# 15. Immediate implementation actions

1. eliminate config/registry naming collisions
2. install schema and registry contract documents under `docs/specs/`
3. harden canonical tables with immutable IDs and provenance
4. create executable YAML registries with statuses and output contracts
5. freeze a `core_frozen` feature subset before using the broader ~560-feature universe as a model baseline
6. instrument runtimes and cache manifests from the first dry run onward
7. keep notebooks orchestration-only and move reusable logic into `src/spotmeta/`

---

# 16. Appendix A — concrete preprocessing inventory summary

Adopted concrete IDs:

- transforms: `raw`, `norm01`, `white_local`, `peak_bg`, `z_local`, `highpass`, `local_mad`, `white`, `norm01_ppi`
- response maps: `log_response`, `dog_response`, `atrous_wavelet_bandpass`, `white_tophat_response`, `prominence_response`
- additional transforms/maps: `rolling_ball_{R}`, `opening_by_recon`, `wavelet_product`, `radial_symmetry_map`, `hessian_blobness`, `log_normalized`
- projection views: `raw_maxproj`, `raw_meanproj`, `norm01_maxproj`, `norm01_meanproj`, `white_local_maxproj`, `white_local_meanproj`, `peak_bg_maxproj`, `peak_bg_meanproj`, `z_local_maxproj`, `z_local_meanproj`

---

# 17. Appendix B — concrete detector inventory summary

Adopted detector IDs:

- `mentor_v1`
- `mentor_v2`
- `consensus_v2`
- `matched_filter_v2`
- `local_expansion`
- `restrained_psf`
- `bright_rescue`
- `sk_log`
- `sk_dog`
- `sk_doh`
- `bigfish_style`
- `trackpy`
- `proj_local_max_raw`
- `proj_local_max_norm`
- `proj_log_norm`
- `proj_log_white`
- `proj_peakbg_max`
- `proj_zlocal_max`
- `atrous_wavelet`
- `morph_tophat`
- `multiscale_log`
- `radial_symmetry`
- `wavelet_product`
- `hessian_blobness`
- `rolling_ball_residual`
- `opening_recon_residual`

---

# 18. Appendix C — concrete feature family inventory summary

Adopted approximate feature family counts:

- patch photometry: ~282
- detector responses: ~78
- spatial / geometric / cluster: ~40
- multi-proposer consensus: ~41
- barcode / multi-channel: ~40
- crop / image regime: ~40
- symmetry / shape: ~16
- interaction / derived / rank: ~26

Approximate total retained from v4: **~560 features before full normalization/derived expansion is resolved**.

