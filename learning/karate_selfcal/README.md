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
-> optimization-based pose refinement (next)
-> SMPL (later)
```

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
- `evaluation/`: metrics and visualization helpers
- `tools/`: utilities for inspection, conversion, and annotations

## Environment

Typical WSL environment:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
```

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
