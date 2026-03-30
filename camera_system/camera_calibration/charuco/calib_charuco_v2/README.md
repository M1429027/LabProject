# Charuco Calibration v2

This folder contains a clean, reproducible Charuco-based calibration workflow.
It separates intrinsics and extrinsics and writes debug outputs for verification.

## Quick Start
1) Edit `config_intrinsics.yaml` for each camera (video path, board params, ROI).
2) Run intrinsics:

```bash
python calib_charuco_v2/calibrate_intrinsics.py --config calib_charuco_v2/config_intrinsics.yaml
```

3) Edit `config_extrinsics.yaml` for each camera (video path + intrinsics path).
4) Run extrinsics:

```bash
python calib_charuco_v2/calibrate_extrinsics.py --config calib_charuco_v2/config_extrinsics.yaml
```

## Notes
- Use the correct ArUco dictionary for your printed board.
- If you enable ROI masking, you must use the same ROI consistently in your pipeline.
- Wide-angle lenses benefit from using frames that cover the full FOV at multiple angles.
