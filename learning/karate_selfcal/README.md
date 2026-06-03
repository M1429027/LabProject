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
-> rough triangulation (next)
-> refinement / SMPL (later)
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

## What Is Stable Right Now

These parts are already usable as a baseline:

- 2D detection with YOLO + HRNet
- single-view tracking
- pairwise cross-view scoring
- global identity hypothesis generation
- geometry-based hypothesis selection
- pairwise relative pose estimation

## What Comes Next

The next implementation target is:

`selected hypothesis + relative pose -> rough multi-view triangulation`

After triangulation becomes stable, the project can continue to:

- optimization-based refinement
- camera bundle refinement
- SMPL fitting
- later transformer-based refinement

## Design Rules

- Keep this project self-contained as a research system.
- Treat calibrated reconstruction as a baseline and comparison target.
- Use geometry, not only 2D similarity, to finalize identity.
- Keep learned lifting as a weak cue unless later experiments prove otherwise.
- Do not let refinement models replace explainable front-end validation too early.
