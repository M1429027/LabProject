# Run Commands

## Real-world 2D->3D pipeline
python algorithm_pipeline/pipelines/real_world_pipeline.py

## AMASS simulation pipeline
python algorithm_pipeline/pipelines/main_pipeline.py

## Camera GUI
python camera_calibration/capture/cam_gui.py

## ChArUco intrinsics
python camera_calibration/charuco/calib_charuco_v2/calibrate_intrinsics.py --config camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam1.yaml

## ChArUco extrinsics
python camera_calibration/charuco/calib_charuco_v2/calibrate_extrinsics.py --config camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam1.yaml

## Phase 0: Consistency check
python scripts/check_pipeline_consistency.py --out-json docs/phase0_consistency_report.json

## Phase 1: Time offset estimation
python scripts/estimate_time_offset.py \
  --keypoints-cam1 old/cleanup_20260326_182859/outputs/real_world_output_demo/keypoints_cam1.json \
  --keypoints-cam2 old/cleanup_20260326_182859/outputs/real_world_output_demo/keypoints_cam2.json \
  --camera-params old/cleanup_20260326_182859/outputs/real_world_output_demo/camera_params.json \
  --out-json docs/phase1_time_offset_report.json
