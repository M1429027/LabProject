# Karate Self-Calibration

`karate_selfcal` is the research workspace for multi-view karate reconstruction
without manual extrinsic calibration.

This project is intentionally separate from:

- `reconstruction_pipeline/algorithm_pipeline/`
- `learning/rumpl_fourview/`

because its core assumptions are different:

- no manual extrinsic calibration
- cross-view identity matching
- multi-frame self-calibration
- rough 3D reconstruction followed by refinement

## Scope

The first implementation phase focuses on:

1. single-view detection
2. single-view tracking
3. cross-view matching without extrinsics
4. multi-frame self-calibration
5. rough 3D reconstruction

Later phases may extend to:

- optimization-based 3D refinement
- SMPL fitting
- physics-based refinement

## Layout

- `configs/`: stage-specific configuration files
- `data/`: lightweight manifests and JSON schemas
- `detection/`: person detection and 2D pose stage
- `tracking/`: single-view tracking stage
- `matching/`: cross-view association logic
- `selfcal/`: relative pose estimation and bundle refinement
- `reconstruction/`: weighted triangulation and geometric constraints
- `refinement/`: post-triangulation pose refinement
- `smpl/`: SMPL fitting stage
- `evaluation/`: metrics and visualization helpers
- `tools/`: utilities for inspection, conversion, and annotations

## Design Rules

- Keep this project self-contained as a research system.
- Reuse ideas from older systems only through small utilities, not by coupling
  directory structures.
- Treat calibrated reconstruction as a baseline, not as the main execution path.
- Avoid introducing physics-specific modules until the non-physics baseline is
  stable.

## First Runnable Stage

The first executable module is the 2D detection stage under
`learning/karate_selfcal/detection/`.

Example run from your WSL environment:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
python -m learning.karate_selfcal.detection.run_detection \
  --input-videos camtest/cam1.mp4 camtest/cam2.mp4 \
  --view-ids cam1 cam2 \
  --output-dir outputs/karate_selfcal/detection_demo \
  --save-annotated-video
```

Outputs include:

- `keypoints_<view_id>.json`
- `run_summary.json`
- `<view_id>_annotated.mp4` when annotation export is enabled

## Available 2D Pose Modes

The detection stage currently supports two frontends:

- `detector.backend: yolo_pose`
  - single-stage YOLO pose baseline
  - returns per-joint confidence
- `detector.backend: yolo_hrnet_topdown`
  - YOLO person detection first
  - HRNet top-down single-person pose second
  - designed for later heatmap-aware refinement

For `yolo_hrnet_topdown`, you must provide:

- `pose.backend: hrnet`
- `pose.config_path`
- `pose.checkpoint_path`

and install the MMPose stack in your active environment:

- `mmpose`
- `mmengine`
- `mmcv`
- any extra runtime dependencies required by your chosen HRNet config

This repo also includes:

- [detection_hrnet_w32.yaml](/home/yp8700/amass/amass/learning/karate_selfcal/configs/detection_hrnet_w32.yaml)
- [install_mmpose_hrnet.sh](/home/yp8700/amass/amass/learning/karate_selfcal/tools/install_mmpose_hrnet.sh)

Recommended setup path for your current WSL environment:

```bash
cd /home/yp8700/amass/amass
bash learning/karate_selfcal/tools/install_mmpose_hrnet.sh
```

## First Runnable Tracking Stage

The next executable module is single-view tracking under
`learning/karate_selfcal/tracking/`.

Example run from your WSL environment:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
python -m learning.karate_selfcal.tracking.run_tracking \
  --input-jsons outputs/karate_selfcal/detection_hrnet_demo/keypoints_cam1.json \
  --source-videos camtest/cam1_demo.mp4 \
  --view-ids cam1 \
  --output-dir outputs/karate_selfcal/tracking_demo \
  --save-tracked-video
```

Outputs include:

- `tracks_<view_id>.json`
- `run_summary.json`
- `tracked_<view_id>.mp4` when tracked-video export is enabled

## Stage 3A: Coarse 3D Lifting

The next baseline module is a heuristic coarse lifting stage under
`learning/karate_selfcal/matching/`.

This stage is intentionally **not** the final 3D reconstruction. Instead, it
produces a lightweight pseudo-3D skeleton and torso-direction descriptor for
each single-view track. The purpose is to support later cross-view matching,
especially skeleton-direction consistency checks.

Example run from your WSL environment:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
python -m learning.karate_selfcal.matching.run_coarse_lifting \
  --input-tracks outputs/karate_selfcal/harmony4d_karate_004_tracking/tracks_karate004_cam05.json \
                 outputs/karate_selfcal/harmony4d_karate_004_tracking/tracks_karate004_cam19.json \
  --view-ids karate004_cam05 karate004_cam19 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_coarse_lifting
```

Outputs include:

- `lifted_<view_id>.json`
- `run_summary.json`

Each lifted track payload contains:

- per-frame pseudo-3D joints in root-centered coordinates
- a coarse `forward_vector`
- a coarse `spine_vector`
- per-track averaged direction descriptors

## Stage 3A: Learned 3D Lifting with VideoPose3D

When you need an actual 2D-to-3D lifting baseline instead of heuristic
pseudo-3D, use the MMPose `VideoPose3D` path:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
python -m learning.karate_selfcal.matching.run_videopose3d_lifting \
  --input-tracks outputs/karate_selfcal/harmony4d_karate_004_tracking/tracks_karate004_cam05.json \
                 outputs/karate_selfcal/harmony4d_karate_004_tracking/tracks_karate004_cam19.json \
  --view-ids karate004_cam05 karate004_cam19 \
  --output-dir outputs/karate_selfcal/harmony4d_karate_004_videopose3d
```

This path uses:

- config:
  - [lifting_videopose3d.yaml](/home/yp8700/amass/amass/learning/karate_selfcal/configs/lifting_videopose3d.yaml)
- checkpoint:
  - [videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth](/home/yp8700/amass/amass/learning/karate_selfcal/checkpoints/videopose3d_h36m/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth)

Outputs include:

- `lifted_<view_id>.json`
- `run_summary.json`

Each learned lifting payload contains:

- per-frame 3D joints from `VideoPose3D`
- `spine_vector`
- `shoulder_vector`
- `forward_vector`
- `track_descriptors`

## 3D Visualization Tool

You can preview either the heuristic or learned Stage 3A lifting result with:

```bash
cd /home/yp8700/amass/amass
source /home/yp8700/amass/.venv/bin/activate
python -m learning.karate_selfcal.tools.visualize_coarse_lifting \
  --input-json outputs/karate_selfcal/harmony4d_karate_004_videopose3d/lifted_karate004_cam19.json \
  --mode anchored
```

This tool writes:

- `<input_stem>_<mode>_3d.mp4`
- `<input_stem>_<mode>_3d.summary.json`

Supported modes:

- `root_centered`
  - show only the root-centered pseudo-3D pose
- `anchored`
  - place each pseudo-3D skeleton using its normalized image root, which is
    better for reviewing multi-person separation
