# LabProject

Multi-view human pose project for reconstructing 3D skeletons from 2D keypoints, with a separated camera-calibration domain for real-world deployment and a standalone four-view ray-based transformer learning system.

## Repository Layout

```text
LabProject/
|- camera_system/
|  |- camera_calibration/            # camera GUI / recording / ChArUco calibration
|  `- debug_tools/
|- reconstruction_pipeline/
|  |- algorithm_pipeline/
|  |  |- pipelines/
|  |  |  |- real_world_pipeline.py          # current 2-camera triangulation pipeline
|  |  |  `- transformer_inference_pipeline.py
|  |  |- modules/
|  |  |  `- transformer_adapter/            # thin real-world adapter for rumpl_fourview
|  |  |- evaluation/
|  |  |- experimental/
|  |  |- configs/
|  |  `- notebooks/
|  `- debug_tools/
|- learning/
|  `- rumpl_fourview/
|     |- configs/                           # model / training / inference yaml configs
|     |- data/                              # sample manifest + base rig schema
|     |- datasets/                          # camera sampling + synthetic projection + loaders
|     |- models/                            # ray encoder + VFT + heads
|     |- training/                          # train loop, losses, metrics, engine
|     |- inference/                         # checkpoint loading + precomputed prediction entrypoint
|     |- tools/                             # synthetic dataset build + sanity check tools
|     `- checkpoints/                       # optional checkpoint staging location
|- outputs/                                 # generated outputs only
|  |- calibration/
|  |- reconstruction/
|  |- debug/
|  `- reports/
|- support_data/                            # body models and support assets
|- docs/
|- scripts/
|- src/                                     # AMASS package code
|- requirements.txt
`- setup.py
```

## Dual-Track Workflows

### Workflow A: Existing triangulation / reconstruction

Use the current real-world pipeline when you want a direct calibrated multi-view baseline from YOLO keypoints.

```bash
python reconstruction_pipeline/algorithm_pipeline/pipelines/real_world_pipeline.py \
  --keypoints-cam1-json outputs/reconstruction/demo/keypoints_cam1.json \
  --keypoints-cam2-json outputs/reconstruction/demo/keypoints_cam2.json \
  --cam1-intrinsics outputs/calibration/calib_out_cam1/cam1_intrinsics.npz \
  --cam1-extrinsics outputs/calibration/calib_out_cam1/cam1_extrinsics.npz \
  --cam2-intrinsics outputs/calibration/calib_out_cam2/cam2_intrinsics.npz \
  --cam2-extrinsics outputs/calibration/calib_out_cam2/cam2_extrinsics.npz \
  --output-dir outputs/reconstruction/real_world_output
```

### Workflow B: `rumpl_fourview` synthetic training -> checkpoint -> real-world inference

Use this route when you want a four-view transformer that consumes rays instead of raw 2D pixel coordinates.

1. Prepare a base four-camera rig JSON matching `learning/rumpl_fourview/data/base_rig.schema.json`.
2. Prepare AMASS-derived 3D joint sequences in `.npz` files with shape `(T, 17, 3)` under a path you control.
3. Generate synthetic ray-token samples.
4. Train the transformer.
5. Run checkpoint-based real-world inference on aligned keypoint JSONs and calibration files.

Synthetic dataset generation:

```bash
python -m learning.rumpl_fourview.tools.build_synthetic_dataset \
  --config learning/rumpl_fourview/configs/synthetic_pretrain.yaml \
  --base-rig-json learning/rumpl_fourview/data/base_rig.schema.json \
  --source-glob "outputs/amass_prepared/*.npz" \
  --output-dir outputs/learning/rumpl_fourview/generated \
  --max-samples 500
```

Manifest sanity check:

```bash
python -m learning.rumpl_fourview.tools.sanity_check_dataset \
  --manifest outputs/learning/rumpl_fourview/generated/manifest.json
```

Training:

```bash
python -m learning.rumpl_fourview.training.train \
  --config learning/rumpl_fourview/configs/synthetic_pretrain.yaml
```

Real-world transformer inference:

```bash
python reconstruction_pipeline/algorithm_pipeline/pipelines/transformer_inference_pipeline.py \
  --checkpoint outputs/learning/rumpl_fourview/synthetic_pretrain/checkpoints/best.pt \
  --keypoints-cam1-json outputs/reconstruction/fourview/keypoints_cam1.json \
  --cam1-intrinsics outputs/calibration/calib_out_cam1/cam1_intrinsics.npz \
  --cam1-extrinsics outputs/calibration/calib_out_cam1/cam1_extrinsics.npz \
  --keypoints-cam2-json outputs/reconstruction/fourview/keypoints_cam2.json \
  --cam2-intrinsics outputs/calibration/calib_out_cam2/cam2_intrinsics.npz \
  --cam2-extrinsics outputs/calibration/calib_out_cam2/cam2_extrinsics.npz \
  --keypoints-cam3-json outputs/reconstruction/fourview/keypoints_cam3.json \
  --cam3-intrinsics outputs/calibration/calib_out_cam3/cam3_intrinsics.npz \
  --cam3-extrinsics outputs/calibration/calib_out_cam3/cam3_extrinsics.npz \
  --keypoints-cam4-json outputs/reconstruction/fourview/keypoints_cam4.json \
  --cam4-intrinsics outputs/calibration/calib_out_cam4/cam4_intrinsics.npz \
  --cam4-extrinsics outputs/calibration/calib_out_cam4/cam4_extrinsics.npz \
  --output-dir outputs/reconstruction/rumpl_fourview_inference
```

## Data Requirements

### Real-world calibration and keypoints

For production inference the transformer path expects:

- time-aligned `keypoints_cam*.json` files
- per-camera `cam*_intrinsics.npz`
- per-camera `cam*_extrinsics.npz`
- a trained checkpoint from `learning/rumpl_fourview/training/train.py`

The model still needs calibration at deployment time so it can turn 2D joints into world-space rays. The benefit is that the network does not need to implicitly memorize a fixed camera layout inside the model weights.

### AMASS and synthetic generation

The first implementation assumes you have already prepared AMASS-derived 3D joint sequences as `.npz` files shaped `(T, 17, 3)`. The synthetic builder projects those joints into a sampled four-camera rig, injects detector-style noise, and writes precomputed training samples containing:

- `ray_tokens` with shape `(17, 4, 7)`
- `view_mask` with shape `(4,)`
- `target_3d` with shape `(17, 3)`
- optional synthetic `observations_2d`

### Checkpoint outputs

Training outputs should be written under `outputs/learning/rumpl_fourview/...`, especially:

- `history.json`
- `resolved_config.json`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`

## Core Workflows Kept From The Existing Repo

Camera / calibration utilities still live under `camera_system/camera_calibration/`.

Camera GUI / recording:

```bash
python camera_system/camera_calibration/capture/cam_gui.py
```

ChArUco intrinsics calibration:

```bash
python camera_system/camera_calibration/charuco/calib_charuco_v2/calibrate_intrinsics.py \
  --config camera_system/camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam1.yaml
```

ChArUco extrinsics calibration:

```bash
python camera_system/camera_calibration/charuco/calib_charuco_v2/calibrate_extrinsics.py \
  --config camera_system/camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam1.yaml
```

Four-webcam synchronized recording (USB/DirectShow):

```bash
python camera_system/camera_calibration/capture/webcam_quad_gui.py \
  camera_system/camera_calibration/capture/config_camera_webcam_quad.yaml
```

AMASS simulation pipeline already present in the repo:

```bash
python reconstruction_pipeline/algorithm_pipeline/pipelines/main_pipeline.py
```

## Output Policy

All generated files should go under `outputs/` and not be committed unless explicitly needed for docs/examples.

## Documentation

- `docs/MEMORY.md` (session rules and execution policy)
- `docs/PROJECT_STRUCTURE.md` (folder structure details)
- `docs/2d3d_optimization_plan.md` (optimization roadmap)
- `docs/PUSH_CANDIDATES.md` (push scope notes)
- `scripts/RUN_COMMANDS.md` (quick command references)
