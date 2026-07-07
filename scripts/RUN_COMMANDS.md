# Run Commands

## Environment activation
cd /home/yp8700/amass
source .venv/bin/activate
cd /home/yp8700/amass/amass

## Real-world 2D->3D pipeline
python reconstruction_pipeline/algorithm_pipeline/pipelines/real_world_pipeline.py

## AMASS simulation pipeline
python reconstruction_pipeline/algorithm_pipeline/pipelines/main_pipeline.py

## RTSP camera GUI (2 cameras)
python camera_system/camera_calibration/capture/cam_gui.py

## USB webcam GUI (4 cameras, synchronized)
python camera_system/camera_calibration/capture/webcam_quad_gui.py \
  camera_system/camera_calibration/capture/config_camera_webcam_quad.yaml

## ChArUco intrinsics
python camera_system/camera_calibration/charuco/calib_charuco_v2/calibrate_intrinsics.py --config camera_system/camera_calibration/charuco/calib_charuco_v2/config_intrinsics_cam1.yaml

## ChArUco extrinsics
python camera_system/camera_calibration/charuco/calib_charuco_v2/calibrate_extrinsics.py --config camera_system/camera_calibration/charuco/calib_charuco_v2/config_extrinsics_cam1.yaml

## Phase 0: Consistency check
python scripts/check_pipeline_consistency.py --out-json outputs/reports/phase0_consistency_report.json

## Phase 1: Time offset estimation
python scripts/estimate_time_offset.py \
  --keypoints-cam1 outputs/reconstruction/real_world_output_demo/keypoints_cam1.json \
  --keypoints-cam2 outputs/reconstruction/real_world_output_demo/keypoints_cam2.json \
  --camera-params outputs/reconstruction/real_world_output_demo/camera_params.json \
  --out-json outputs/reports/phase1_time_offset_report.json
