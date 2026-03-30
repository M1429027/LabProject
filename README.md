# LabProject

Human pose reconstruction project: from multi-view 2D keypoints to 3D skeleton, with a separate real-world camera calibration workflow.

## What This Repo Contains

- `algorithm_pipeline/`: 2D->3D reconstruction algorithms, evaluation, and AMASS-related pipeline code.
- `camera_calibration/`: camera capture app, ChArUco intrinsics/extrinsics calibration, and calibration debug tools.
- `docs/`: project structure and optimization notes.
- `scripts/`: quick command references.

## Project Structure

```text
LabProject/
├─ algorithm_pipeline/
│  ├─ pipelines/          # end-to-end runnable pipelines
│  ├─ modules/            # reusable algorithm modules
│  ├─ evaluation/         # comparison/evaluation scripts
│  ├─ experimental/       # non-core experiments
│  ├─ configs/            # algorithm configs
│  └─ notebooks/
├─ camera_calibration/
│  ├─ capture/            # camera GUI / recording
│  ├─ charuco/            # ChArUco calibration workflow
│  ├─ debug/              # calibration diagnostics + artifacts
│  ├─ legacy/             # older calibration scripts (reference)
│  └─ configs/            # reserved for centralized camera configs
├─ docs/
├─ scripts/
├─ src/                   # AMASS package code
├─ requirements.txt
├─ setup.py
└─ .gitignore
```

## Main Entry Points

### Real-world 2D->3D reconstruction
```bash
python algorithm_pipeline/pipelines/real_world_pipeline.py
```

### AMASS simulation pipeline
```bash
python algorithm_pipeline/pipelines/main_pipeline.py
```

### Camera preview/recording GUI
```bash
python camera_calibration/capture/cam_gui.py
```

### ChArUco intrinsics calibration
```bash
python camera_calibration/charuco/calib_charuco_v2/calibrate_intrinsics.py --config camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam1.yaml
```

### ChArUco extrinsics calibration
```bash
python camera_calibration/charuco/calib_charuco_v2/calibrate_extrinsics.py --config camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam1.yaml
```

## Notes for GitHub Cleanliness

- Large datasets, model weights, videos, and generated outputs are excluded by `.gitignore`.
- Core source code and configs are kept under `algorithm_pipeline/` and `camera_calibration/`.
- If files were renamed/moved, Git history will track them after commit and push.

## Documentation

- Structure guide: `docs/PROJECT_STRUCTURE.md`
- Optimization roadmap: `docs/2d3d_optimization_plan.md`
- Push audit reference: `docs/PUSH_CANDIDATES.md`
- Quick commands: `scripts/RUN_COMMANDS.md`
