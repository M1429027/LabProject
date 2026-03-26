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
