# LabProject

Multi-view human pose project for reconstructing 3D skeletons from 2D keypoints, with a separated camera-calibration domain for real-world deployment.

## Repository Layout

```text
LabProject/
|- camera_system/
|  |- camera_calibration/
|  |  |- capture/                     # camera GUI / recording
|  |  |- charuco/calib_charuco_v2/    # ChArUco intrinsics/extrinsics pipeline
|  |  |- debug/                       # calibration-local compatibility folder
|  |  `- legacy/                      # older calibration scripts (reference)
|  `- debug_tools/                    # camera debug scripts
|- reconstruction_pipeline/
|  |- algorithm_pipeline/
|  |  |- pipelines/                   # main runnable pipelines
|  |  |- modules/                     # reusable algorithm modules
|  |  |- evaluation/                  # evaluation/comparison tools
|  |  |- experimental/                # experiments
|  |  |- configs/                     # algorithm configs
|  |  `- notebooks/
|  `- debug_tools/                    # reconstruction debug tools
|- outputs/                           # generated outputs only
|  |- calibration/
|  |- reconstruction/
|  |- debug/
|  `- reports/
|- docs/                              # project docs and memory rules
|- scripts/                           # utility and consistency scripts
|- src/                               # AMASS package code
|- requirements.txt
`- setup.py
```

## Core Workflows

1. Real-world 2D -> 3D reconstruction
```bash
python reconstruction_pipeline/algorithm_pipeline/pipelines/real_world_pipeline.py
```

2. AMASS simulation pipeline
```bash
python reconstruction_pipeline/algorithm_pipeline/pipelines/main_pipeline.py
```

3. Camera GUI / recording
```bash
python camera_system/camera_calibration/capture/cam_gui.py
```

4. ChArUco intrinsics calibration
```bash
python camera_system/camera_calibration/charuco/calib_charuco_v2/calibrate_intrinsics.py \
  --config camera_system/camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam1.yaml
```

5. ChArUco extrinsics calibration
```bash
python camera_system/camera_calibration/charuco/calib_charuco_v2/calibrate_extrinsics.py \
  --config camera_system/camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam1.yaml
```

## Output Policy

All generated files should go under `outputs/` and not be committed unless explicitly needed for docs/examples.

## Documentation

- `docs/MEMORY.md` (session rules and execution policy)
- `docs/PROJECT_STRUCTURE.md` (folder structure details)
- `docs/2d3d_optimization_plan.md` (optimization roadmap)
- `docs/PUSH_CANDIDATES.md` (push scope notes)
- `scripts/RUN_COMMANDS.md` (quick command references)
