# Debug Workspace

This folder centralizes debugging/checking tools and outputs.

## Structure
- `scripts/`: debug/check/diagnostic scripts
- `artifacts/`: generated debug images/videos
- `logs/`: runtime logs (reserved)

## Moved Scripts
- `debug/scripts/debug_charuco.py`
- `debug/scripts/diagnose.py`
- `debug/scripts/diagnose_calib.py`
- `debug/scripts/verify_world_origin.py`
- `debug/scripts/analyze_output.py`
- `debug/scripts/find_env.py`

## Notes
- If you previously ran scripts from project root, update commands to new paths.
- Example:
  - `python debug/scripts/debug_charuco.py --video debug/artifacts/debug_camera2.mp4 --config calib_charuco_v2/config_extrinsics.yaml --frame 30`
