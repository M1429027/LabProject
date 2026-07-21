# camera_system

Domain boundary for camera-side workflows.

Scope:
- camera capture / GUI / recording
- intrinsics/extrinsics calibration
- camera and calibration debug tools

Canonical path:
- `camera_system/camera_calibration/`

Capture entry points:
- RTSP dual-camera GUI: `camera_system/camera_calibration/capture/cam_gui.py`
- USB webcam quad GUI: `camera_system/camera_calibration/capture/webcam_quad_gui.py`

Webcam quad config:
- `camera_system/camera_calibration/capture/config_camera_webcam_quad.yaml`
