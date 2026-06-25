# Karate Self-Calibration

`karate_selfcal` is the research workspace for multi-view karate reconstruction
without manual extrinsic calibration.

This project is intentionally separate from:

- `reconstruction_pipeline/algorithm_pipeline/`
- `learning/rumpl_fourview/`

because its core assumptions are different:

- no manual extrinsic calibration
- cross-view identity matching
- geometry-based identity selection
- self-calibration from tracked people
- rough 3D reconstruction followed by refinement

## Goal

The current research goal is:

`recover two-player 3D pose from fixed multi-view karate videos without manual extrinsic calibration`

The practical interpretation is:

1. detect people and estimate 2D pose in each view
2. track people within each view
3. generate cross-view identity hypotheses
4. use multi-view geometry to choose the correct hypothesis
5. estimate rough relative camera pose
6. continue to rough triangulation and later refinement

## Current Mainline

The current implemented mainline is:

```text
Multi-view videos
-> 2D detection + top-down pose
-> single-view tracking
-> cross-view pairwise scoring
-> global identity hypotheses
-> geometry-based hypothesis selection
-> relative pose estimation
-> rough triangulation
-> cheirality / scale checks
-> human-scale prior
-> joint-quality triangulation
-> bone prior + outlier filtering
-> pose-level candidate validation (next)
-> optimization-based pose refinement
-> transformer schema / dataset loader
-> SMPL (later)
```

## Current Recommended Order Before Transformer

The project should not move directly into full Transformer training yet. The
current blocker is unstable multi-view joint correspondence: people are mostly
matched, but individual joints can still be mismatched across views.

Recommended order:

1. GT correspondence diagnostic using dataset `poses3d` / `smpl` as evaluation-only references.
2. Convert GT findings into non-GT proxy rules for candidate scoring.
3. Stable joint quality / inlier-view / candidate-3D output.
4. Transformer token schema definition.
5. Dataset loader for sequence-level refinement.
6. Small Transformer baseline.
7. Full refinement training only after the input quality is stable enough.

The immediate next step is GT correspondence diagnostic. GT is not part of the
runtime pipeline; it is used to identify which joints, view subsets, and scoring
terms fail before those findings are translated into inference-time proxy rules.

Current GT-based findings from `09_karate/004_karate`:

- `oracle_scaled_subset_02_07_13` aligns well to GT, with mean joint error near
  5 cm after similarity alignment. This confirms that the dataset GT and the
  diagnostic alignment are usable.
- `pose_hypothesis_selection_v1` is much worse against GT, despite low
  reprojection error. This points to candidate / view-subset selection rather
  than a total failure of triangulation.
- Reprojection error alone is not reliable: many samples have low reprojection
  error but high GT error.
- Candidate selection should be refined with joint-specific risk, view-subset
  risk, ray/depth quality, bone plausibility, and temporal stability.

## Candidate Scoring V2 Findings

`run_pose_hypothesis_selection.py` supports `--use-candidate-scoring-v2`.
This scoring mode keeps GT out of runtime inference but uses lessons from GT
diagnostics as soft proxy terms:

- lower weight for reprojection-only quality
- stronger ray-angle and depth/scale sanity terms
- joint-specific risk priors for unstable joints
- view-subset risk penalties for combinations that produced low-reprojection
  but high-3D-error samples
- optional single-view reliability penalty for diagnostic experiments

Current result on `09_karate/004_karate`:

| Variant | Extrinsics | Mean GT Error | P90 GT Error | Bad Rate | Low-Reproj Bad |
|---|---|---:|---:|---:|---:|
| pose hypothesis v1 | selfcal rough | 0.353 m | 0.592 m | 69.37% | 604 |
| candidate scoring v2 | selfcal rough | 0.348 m | 0.598 m | 67.25% | 546 |
| candidate scoring v2b | selfcal rough | 0.348 m | 0.603 m | 66.70% | 535 |
| candidate scoring v2b | reference extrinsics | 0.039 m | 0.073 m | 0.59% | 7 |

Interpretation:

- Candidate scoring can reduce low-reprojection failure cases, but it cannot
  fully rescue noisy rough extrinsics.
- With reference extrinsics, the same scoring path produces a clean result,
  which makes rough extrinsic refinement the next major blocker before
  Transformer training.

## Layout

- `configs/`: stage-specific configuration files
- `data/`: lightweight manifests and JSON schemas
- `detection/`: person detection and 2D pose stage
- `tracking/`: single-view tracking stage
- `matching/`: cross-view association logic
- `selfcal/`: geometry validation and relative pose estimation
- `reconstruction/`: weighted triangulation and geometric constraints
- `refinement/`: post-triangulation pose refinement
- `smpl/`: SMPL fitting stage
- `stage_a/`: supervised warm-up dataset and dataloader
- `evaluation/`: metrics and visualization helpers
- `tools/`: utilities for inspection, conversion, and annotations

## Environment

Typical WSL environment:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
```

## Stage A: Supervised Warm-Up

Stage A starts from fixed four-view Harmony4D karate annotations and trains a
small supervised 3D pose baseline before attempting self-supervised
pose-camera refinement.

Current selected views:

- `cam02`
- `cam03`
- `cam07`
- `cam16`

Current precomputed dataset:

- Manifest: [manifest.json](/home/yp8700/amass/amass/outputs/karate_selfcal/stage_a_cam02_03_07_16/manifest.json)
- Selected view index: [selected_views_index.md](/home/yp8700/amass/amass/outputs/karate_selfcal/selected_views_cam02_03_07_16/selected_views_index.md)

Dataset summary:

| Split | Samples | Sequence rule |
|---|---:|---|
| train | 17478 | all selected sequences except validation/test |
| val | 120 | `09_karate/004_karate` |
| test | 680 | `11_karate3/008_karate3` |

Stage A sample tensor shapes:

| Tensor | Shape |
|---|---|
| `ray_tokens` | `[B, 17, 4, 7]` |
| `view_mask` | `[B, 4]` |
| `joint_view_mask` | `[B, 17, 4]` |
| `target_3d` | `[B, 17, 3]` |
| `target_3d_root_relative` | `[B, 17, 3]` |
| `target_confidence` | `[B, 17]` |

Build dataset:

```bash
python -m learning.karate_selfcal.stage_a.build_dataset \
  --selected-views-manifest outputs/karate_selfcal/selected_views_cam02_03_07_16/selected_views_manifest.json \
  --output-dir outputs/karate_selfcal/stage_a_cam02_03_07_16 \
  --val-sequences 09_karate/004_karate \
  --test-sequences 11_karate3/008_karate3
```

Check dataloader:

```bash
python -m learning.karate_selfcal.stage_a.check_dataset \
  --manifest outputs/karate_selfcal/stage_a_cam02_03_07_16/manifest.json \
  --batch-size 8
```

Smoke train:

```bash
python -m learning.karate_selfcal.stage_a.train \
  --config learning/karate_selfcal/configs/stage_a_smoke.yaml
```

Full first-pass train:

```bash
python -m learning.karate_selfcal.stage_a.train \
  --config learning/karate_selfcal/configs/stage_a.yaml
```

Evaluate checkpoint:

```bash
python -m learning.karate_selfcal.stage_a.evaluate \
  --checkpoint outputs/karate_selfcal/stage_a_cam02_03_07_16_run/checkpoints/best.pt \
  --manifest outputs/karate_selfcal/stage_a_cam02_03_07_16/manifest.json \
  --split test \
  --output-json outputs/karate_selfcal/stage_a_cam02_03_07_16_run/eval_test.json
```

## Stage A: A1/A2 Decoupled Validation

The current Stage A validation uses the ray-token dataset with true multi-view
geometry features:

- Manifest: `outputs/karate_selfcal/stage_a_cam02_03_07_16_ray_split_8_1_1/manifest.json`
- Token shape: `[B, 17, 4, 11]`
- Views: `cam02`, `cam03`, `cam07`, `cam16`
- Test sequence used for detailed review: `11_karate3/008_karate3`

The model keeps the same two-stage attention structure:

```text
view attention:  same joint across four camera views
joint attention: 17 fused joint tokens within one person
```

A1 and A2 are intentionally trained separately as a diagnostic and stabilization
step before A3 joint fine-tuning:

| Stage | Target | Config | Purpose |
|---|---|---|---|
| A1 | root-relative pose | `configs/stage_a_a1_pose_endpoint_8_1_1.yaml` | learn body shape, endpoint position, and limb extension |
| A2 | global pelvis translation | `configs/stage_a_a2_pelvis_8_1_1.yaml` | learn where the person is in the scene |

Training commands:

```bash
python -m learning.karate_selfcal.stage_a.train \
  --config learning/karate_selfcal/configs/stage_a_a1_pose_endpoint_8_1_1.yaml

python -m learning.karate_selfcal.stage_a.train \
  --config learning/karate_selfcal/configs/stage_a_a2_pelvis_8_1_1.yaml
```

A1 loss terms:

- base root-relative MPJPE
- endpoint loss on wrists and ankles: joints `[9, 10, 15, 16]`
- limb extension loss from pelvis to endpoints
- high-extension frame weighting for extended strike / kick poses

A2 loss terms:

- pelvis L2 / MPJPE loss only
- no full-body 3D loss, so the global translation task does not overwrite pose
  learning

Current composed validation on `11_karate3/008_karate3`:

```text
full_3d = A1 pred_pose_root_relative + A2 pred_pelvis
```

| Metric | Endpoint + old pelvis | Endpoint + A2 pelvis | Improvement |
|---|---:|---:|---:|
| Absolute MPJPE | 0.132 m | 0.114 m | 13.5% |
| Pelvis error | 0.109 m | 0.083 m | 24.1% |
| Endpoint error | 0.150 m | 0.136 m | 9.3% |
| Wrist error | 0.184 m | 0.171 m | 6.9% |
| Ankle error | 0.117 m | 0.102 m | 13.0% |
| Torso error | 0.115 m | 0.091 m | 20.5% |
| Two-person pelvis distance error | 0.062 m | 0.043 m | 30.5% |

This supports the working hypothesis that root-relative pose and global pelvis
translation interfere when trained naively as a single objective. The next step
is A3 staged joint fine-tuning, where A1 initializes the pose branch and A2
initializes the pelvis branch before producing one final checkpoint.
## Stage 1: Detection

The current preferred frontend is:

- `YOLO detector -> HRNet top-down pose`

This path preserves:

- person bounding boxes
- 2D keypoints
- per-joint confidence
- optional heatmaps

Quick example:

```bash
python -m learning.karate_selfcal.detection.run_detection \
  --input-videos camtest/cam1.mp4 camtest/cam2.mp4 \
  --view-ids cam1 cam2 \
  --output-dir outputs/karate_selfcal/detection_demo \
  --save-annotated-video
```

Outputs:

- `keypoints_<view>.json`
- `run_summary.json`
- `<view>_annotated.mp4`

Reference config:

- [detection_hrnet_w32.yaml](/home/yp8700/amass/amass/learning/karate_selfcal/configs/detection_hrnet_w32.yaml)

Installer helper:

- [install_mmpose_hrnet.sh](/home/yp8700/amass/amass/learning/karate_selfcal/tools/install_mmpose_hrnet.sh)

## Stage 2: Single-View Tracking

Tracking currently uses a pose-aware association baseline that mixes:

- bbox IoU
- pose similarity
- center continuity
- tracklet merge and short-track filtering

Quick example:

```bash
python -m learning.karate_selfcal.tracking.run_tracking \
  --input-jsons outputs/karate_selfcal/detection_demo/keypoints_cam1.json \
  --source-videos camtest/cam1.mp4 \
  --view-ids cam1 \
  --output-dir outputs/karate_selfcal/tracking_demo \
  --save-tracked-video
```

Outputs:

- `tracks_<view>.json`
- `run_summary.json`
- `tracked_<view>.mp4`

## Stage 3A: Learned Single-View 3D Lifting

The project keeps a learned single-view lifting baseline through MMPose
VideoPose3D. This is currently treated as a weak cue and debugging aid, not as
the identity decision maker.

Quick example:

```bash
python -m learning.karate_selfcal.matching.run_videopose3d_lifting \
  --input-tracks outputs/karate_selfcal/harmony4d_karate_004_tracking/tracks_karate004_cam05.json \
                 outputs/karate_selfcal/harmony4d_karate_004_tracking/tracks_karate004_cam19.json \
  --view-ids karate004_cam05 karate004_cam19 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_videopose3d
```

Outputs:

- `lifted_<view>.json`
- `run_summary.json`

Useful review tool:

- [visualize_coarse_lifting.py](/home/yp8700/amass/amass/learning/karate_selfcal/tools/visualize_coarse_lifting.py)

Despite the filename, this visualization tool is also used for learned
VideoPose3D outputs.

## Stage 3B-1: Cross-View Pairwise Scoring

This step compares tracklets across views and builds candidate scores.

The current score terms are:

- `pose_shape`
- `motion_prior`
- `visibility`
- `temporal`
- `skeleton_consistency`
- `appearance` placeholder

Quick example:

```bash
python -m learning.karate_selfcal.matching.run_cross_view_matching \
  --input-tracks outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam01.json \
                 outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam06.json \
                 outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam11.json \
                 outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam16.json \
  --input-lifted outputs/karate_selfcal/harmony4d_karate_004_fourview_videopose3d/lifted_karate004_cam01.json \
                 outputs/karate_selfcal/harmony4d_karate_004_fourview_videopose3d/lifted_karate004_cam06.json \
                 outputs/karate_selfcal/harmony4d_karate_004_fourview_videopose3d/lifted_karate004_cam11.json \
                 outputs/karate_selfcal/harmony4d_karate_004_fourview_videopose3d/lifted_karate004_cam16.json \
  --view-ids karate004_cam01 karate004_cam06 karate004_cam11 karate004_cam16 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_fourview_matching
```

Outputs:

- `candidate_matches.json`
- `matching_graph.json`
- `global_assignment_hypotheses.json`
- `run_summary.json`

Review helpers:

- [visualize_matches.py](/home/yp8700/amass/amass/learning/karate_selfcal/evaluation/visualize_matches.py)
- [visualize_matching_groups.py](/home/yp8700/amass/amass/learning/karate_selfcal/evaluation/visualize_matching_groups.py)
- [visualize_matching_hypotheses.py](/home/yp8700/amass/amass/learning/karate_selfcal/evaluation/visualize_matching_hypotheses.py)

## Stage 3B-2: Geometry-Based Identity Selection

This is now the official end of Stage 3B.

Instead of trusting pairwise score alone, the pipeline:

1. enumerates global two-person identity hypotheses
2. collects same-identity, same-frame, same-joint correspondences
3. estimates a fundamental matrix for each view pair
4. scores each hypothesis by inlier ratio and Sampson error
5. selects the final identity assignment automatically

Quick example:

```bash
python -m learning.karate_selfcal.selfcal.run_hypothesis_geometry \
  --hypotheses-json outputs/karate_selfcal/harmony4d_karate_004_fourview_matching/global_assignment_hypotheses.json \
  --track-jsons outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam01.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam06.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam11.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam16.json \
  --view-ids karate004_cam01 karate004_cam06 karate004_cam11 karate004_cam16 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_fourview_geometry_validation
```

Outputs:

- `hypothesis_geometry_scores.json`
- `selected_hypothesis.json`
- `run_summary.json`

`selected_hypothesis.json` is now the formal handoff from Stage 3B to Stage 4A.

## Stage 4A: Relative Pose Estimation

Stage 4A consumes the geometry-selected identity hypothesis and estimates
pairwise camera relative pose with an essential-matrix baseline.

Quick example:

```bash
python -m learning.karate_selfcal.selfcal.run_relative_pose \
  --selected-hypothesis-json outputs/karate_selfcal/harmony4d_karate_004_fourview_geometry_validation/selected_hypothesis.json \
  --track-jsons outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam01.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam06.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam11.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam16.json \
  --view-ids karate004_cam01 karate004_cam06 karate004_cam11 karate004_cam16 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_fourview_relative_pose
```

Optional:

- pass `--intrinsics` with existing calibration reports or `.npz`
- otherwise the current baseline falls back to an approximate pinhole model

Outputs:

- `relative_pose_results.json`
- `run_summary.json`

### Stage 4A-Aux: Pre-Refinement Camera Checks

The current self-calibration branch includes several diagnostic and
stabilization steps before full pose refinement:

- `run_cheirality_sign_correction.py`: tests the `t` vs `-t` ambiguity from
  essential-matrix recovery by checking positive-depth ratios.
- `run_translation_scale_refinement.py`: aligns pairwise translation directions
  in an anchor-camera graph. This improves direction consistency but does not
  solve metric scale by itself.
- `run_human_scale_prior.py`: applies a global scale prior from median human
  height. This is the current first-pass scale baseline.

Current conclusion:

- translation sign is not the dominant failure mode
- pairwise direction is usable
- absolute scale needs an explicit prior or later optimization
- these checks are sufficient before moving into pose-level refinement

## Stage 5: Rough Triangulation

Stage 5 consumes the selected cross-view identity hypothesis and camera
extrinsics. In the self-calibration mainline, those extrinsics come from Stage
4A. In reference experiments, they can come from dataset COLMAP cameras.

Quick example:

```bash
python -m learning.karate_selfcal.reconstruction.run_triangulation \
  --selected-hypothesis-json outputs/karate_selfcal/harmony4d_karate_004_fourview_geometry_validation/selected_hypothesis.json \
  --rough-extrinsics-json outputs/karate_selfcal/harmony4d_karate_004_fourview_relative_pose/rough_extrinsics.json \
  --track-jsons outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam01.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam06.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam11.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam16.json \
  --view-ids karate004_cam01 karate004_cam06 karate004_cam11 karate004_cam16 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_fourview_triangulation
```

The current triangulation baseline is inlier-aware:

- filters pairwise epipolar outliers with Sampson error
- triangulates in undistorted normalized camera coordinates
- reports pixel reprojection error with the original camera distortion model
- can select per-joint view subsets by reprojection error, triangulation angle,
  confidence, and dropped-view penalty

Outputs:

- `triangulated_3d.json`
- `run_summary.json`

Useful options:

- `--use-view-subset-selection`: choose a better view subset for each joint
- `--min-triangulation-angle-deg`: reject weak-baseline joint hypotheses

Current conclusion:

- joint-level view subset selection improves reprojection error
- lower reprojection error alone does not guarantee human-shaped 3D pose
- per-joint triangulation still needs pose-level constraints

## Stage 5-Aux: Bone Prior and Outlier Filtering

The project now includes two pre-refinement skeleton stabilizers:

- `run_bone_length_prior.py`: estimates per-identity median bone lengths and
  softly pulls unstable bones toward the identity's stable length profile.
- `run_bone_outlier_filter.py`: clamps occasional overlong bones and applies
  light temporal smoothing.

These steps are filters, not full pose refinement. They reduce exploding limbs
and spider-like artifacts, but they cannot reconstruct a coherent person when
the whole skeleton is structurally inconsistent.

Current conclusion:

- bone outlier filtering is useful as a safety pass
- it should not be extended indefinitely as the main solution
- the next required step is optimization-based pose refinement over the whole
  skeleton and a temporal window

## Reference Baseline: COLMAP Extrinsics

This is a validation branch, not part of the self-calibration method itself.
It uses dataset COLMAP intrinsics/extrinsics as an oracle camera baseline to
answer one debugging question:

`If the cameras are correct, can the current identity matching and triangulation backend produce usable 3D motion?`

Quick example:

```bash
python -m learning.karate_selfcal.evaluation.run_reference_baseline \
  --selected-hypothesis-json outputs/karate_selfcal/harmony4d_karate_004_fourview_geometry_validation_v2/selected_hypothesis.json \
  --track-jsons outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam01.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam06.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam11.json \
                outputs/karate_selfcal/harmony4d_karate_004_fourview_tracking/tracks_karate004_cam16.json \
  --view-ids karate004_cam01 karate004_cam06 karate004_cam11 karate004_cam16 \
  --colmap-cameras-txt /mnt/d/09_karate.zip::09_karate/004_karate/colmap/workplace/cameras.txt \
  --colmap-images-txt /mnt/d/09_karate.zip::09_karate/004_karate/colmap/workplace/images.txt \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_reference_baseline \
  --render-video
```

Outputs:

- `rough_extrinsics_colmap.json`
- `triangulated_3d_raw.json`
- `triangulated_3d_completed.json`
- `triangulation_summary.json`
- `completion_metrics.json`
- optional raw/completed review videos

The completion step is intentionally simple for now:

- short missing joint gaps are linearly interpolated
- finite samples are smoothed with a small moving average
- completeness metrics report joint coverage before and after completion

Current 4B status:

- the oracle branch confirms that the backend can produce lower reprojection
  error when camera geometry is correct
- it can show body shape and motion, but still has missing / unstable joints
- the remaining issue is not only camera extrinsics; it is also 2D observation
  quality, joint-level triangulation stability, and missing pose-level priors
- 4B is complete enough for its current purpose: validating the pre-refinement
  upper bound before Stage 4C

## What Is Stable Right Now

These parts are already usable as a baseline:

- 2D detection with YOLO + HRNet
- single-view tracking
- pairwise cross-view scoring
- global identity hypothesis generation
- geometry-based hypothesis selection
- pairwise relative pose estimation
- inlier-aware triangulation with known camera parameters
- COLMAP oracle reference baseline for backend validation
- cheirality sign check
- human-scale prior
- joint-quality triangulation
- bone-length prior and bone outlier filtering

## What Comes Next

The next implementation target is:

`Stage 4C: optimization-based pose refinement`

The current pre-refinement filters and constraints are at a reasonable baseline
limit. Continuing to add more filters may make the sequence smoother, but will
not reliably make it more human-shaped.

Stage 4C should optimize an entire identity skeleton over a short temporal
window with:

- reprojection loss
- bone length consistency
- left/right symmetry
- temporal smoothness
- joint confidence weighting
- robust outlier loss

In parallel, the reference baseline can continue to:

- camera bundle refinement
- SMPL fitting
- later transformer-based refinement

## Design Rules

- Keep this project self-contained as a research system.
- Treat calibrated reconstruction as a baseline and comparison target.
- Use geometry, not only 2D similarity, to finalize identity.
- Keep learned lifting as a weak cue unless later experiments prove otherwise.
- Do not let refinement models replace explainable front-end validation too early.
