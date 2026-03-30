# camera_system

Domain boundary for camera-side workflows.

Scope:
- capture / GUI / recording
- intrinsics/extrinsics calibration
- calibration debug tools

Safe split note:
- Existing code still lives under legacy paths during migration.
- New camera-side tools should be placed under this domain.

Canonical path:
- camera_system/camera_calibration/

Compatibility alias:
- root camera_calibration -> camera_system/camera_calibration (symlink)
