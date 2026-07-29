#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
baseline_dir="camtest/line/baselines/2026-07-28_global-axes_hrnet-motion-peak"
output_dir="camtest/line/demo_hrnet_enhanced_20260728/robust_3d_baseline_reproduction"

"$python_bin" camera_system/camera_calibration/charuco/calib_charuco_v2/run_demo_robust_pose_pipeline.py \
  --line-dir camtest/line \
  --pose-jsons \
    camtest/line/demo_hrnet_enhanced_20260728/pose_filtered/keypoints_cam1demo_filtered.json \
    camtest/line/demo_hrnet_enhanced_20260728/pose_filtered/keypoints_cam2demo_filtered.json \
    camtest/line/demo_hrnet_enhanced_20260728/pose_filtered/keypoints_cam3demo_filtered.json \
    camtest/line/demo_hrnet_enhanced_20260728/pose_filtered/keypoints_cam4demo_filtered.json \
  --cam-ids cam1demo cam2demo cam3demo cam4demo \
  --output-dir "$output_dir" \
  --reference cam1demo \
  --swap-lr-cameras \
  --fixed-sync-report "$baseline_dir/motion_peak_sync_report.json" \
  --refined-extrinsics-dir "$baseline_dir/extrinsics" \
  --skip-synced-videos
