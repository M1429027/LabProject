# GitHub Push Candidate List (Audit)

Generated: 2026-03-26
Workspace: /home/yp8700/amass/amass

## A. High-value files to push now (core workflow)

### Project metadata
- README.md
- requirements.txt
- setup.py
- .gitignore (should be cleaned before final push)
- 2d3d_optimization_plan.md

### Real-world pipeline (current active path)
- cam_gui.py
- real_world_pipeline.py

### Calibration v2 (current active path)
- calib_charuco_v2/README.md
- calib_charuco_v2/common.py
- calib_charuco_v2/calibrate_intrinsics.py
- calib_charuco_v2/calibrate_extrinsics.py
- calib_charuco_v2/config_intrinsics.yaml
- calib_charuco_v2/config_intrinsics_cam1.yaml
- calib_charuco_v2/config_intrinsics_cam2.yaml
- calib_charuco_v2/config_extrinsics.yaml
- calib_charuco_v2/config_extrinsics_cam1.yaml
- calib_charuco_v2/config_extrinsics_cam2.yaml
- calib_charuco_v2/config_extrinsics_cam1_relaxed.yaml

### AMASS simulation path (keep if you still use simulation pipeline)
- main_pipeline.py
- src/** (python source only)

### Debug scripts (useful, lightweight)
- debug/README.md
- debug/scripts/debug_charuco.py
- debug/scripts/diagnose.py
- debug/scripts/diagnose_calib.py
- debug/scripts/verify_world_origin.py
- debug/scripts/analyze_output.py
- debug/scripts/find_env.py

---

## B. Optional / likely legacy (push only if you still use them)
- triangulation.py
- triangulation_smooth.py
- yolo.py
- yoloskeleton.py
- compute_extrinsics.py
- outer.py
- visualization.py
- four_cameraview.py
- compare.py
- comparerating.py
- compareresult.py
- videotest.py
- jointstest.py
- config.yaml
- notebooks/**

Reason: these look like experiment or earlier-stage scripts, overlapping with newer pipelines.

---

## C. Do NOT push (generated outputs / large data / binaries)

### Generated outputs (reproducible artifacts)
- real_world_output/**
- real_world_output_demo/**
- triangulation_output*/**
- comparison_frames_output*/**
- render_output_pro/**
- visualization_output_4cams/**
- mediaoutputs/**
- runs/**
- debug/artifacts/**
- calib_out_cam1/**
- calib_out_cam2/**

### Raw videos
- calib_charuco_v2/*.mp4
- mediavideos/**
- rendervideo/**

### Models / checkpoints
- *.pt

### Large datasets / external dependencies
- CMU/**
- support_data/**
- psbody-mesh/**

### Temp/system
- __pycache__/**
- *.Zone.Identifier

---

## D. Suggested next step before first clean push
1. keep only section A (and optionally B) in first commit
2. move section C under ignore rules
3. create a fresh README for your own pipeline (not upstream AMASS-only readme)
4. push in two commits:
   - commit 1: structure + pipeline code + configs
   - commit 2: cleanup/docs
