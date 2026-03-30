# Project Structure

This repository is organized by responsibility.

## 1) algorithm_pipeline
- `algorithm_pipeline/pipelines/`: end-to-end runnable pipelines
- `algorithm_pipeline/modules/`: reusable algorithm modules
- `algorithm_pipeline/evaluation/`: quality comparison/evaluation scripts
- `algorithm_pipeline/experimental/`: legacy experiments and quick tests
- `algorithm_pipeline/configs/`: algorithm configs

## 2) camera_calibration
- `camera_calibration/capture/`: camera GUI / recording tools
- `camera_calibration/charuco/`: ChArUco calibration workflow and configs
- `camera_calibration/legacy/`: old extrinsics scripts kept for reference
- `camera_calibration/debug/`: calibration/debug scripts and artifacts
- `camera_calibration/configs/`: reserved for future centralized camera configs

## Root stays minimal
- project metadata: `README.md`, `requirements.txt`, `setup.py`, `.gitignore`
- large datasets/outputs remain outside code folders and are ignored by Git.
