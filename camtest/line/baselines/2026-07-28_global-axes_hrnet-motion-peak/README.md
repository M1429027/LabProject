# 2026-07-28 Correct Demo Baseline

This snapshot records the first verified usable four-camera 3D pose baseline.

## Frozen configuration

- Intrinsics:
  - cam1: `camtest/line/calib_out_cam1_hvflip/cam1_intrinsics.npz`
  - cam2: `camtest/line/calib_out_cam2_hvflip_recalc/cam2_intrinsics.npz`
  - cam3: `camtest/line/calib_out_cam3_hvflip_fast/cam3_intrinsics.npz`
  - cam4: `camtest/line/calib_out_cam4_hvflip_newintr_20260727/cam4_intrinsics.npz`
- Extrinsics: `extrinsics/cam1..4_line_extrinsics_refined.npz`
- Synchronization: explicit eight arm-motion peaks with affine frame mapping.
- 2D pose: HRNet filtered tracks.
- Left/right mapping: no camera-level COCO left/right swap.
- 3D: pose-level subset selection, strict joint RANSAC, temporal gate, smoothing and skeleton optimization.

## Verified results

- Global bilateral epipolar median: `5.7515 px`
- LR-invariant torso-center epipolar median: `4.94 px`
- Frames: `533`
- Empty camera-subset frames: `0`
- Temporal joint rejections: `0`
- Bad bones before/after projection: `0 / 0`
- Root Z range: `0.06196 m`
- Camera layout: four sides of the field, all optical axes pointing inward.

## Canonical outputs

- 3D JSON: `camtest/line/demo_hrnet_enhanced_20260728/robust_3d_global_axes_motion_peak_noswap_20260729/triangulated_enhanced_robust.json`
- 3D video: `camtest/line/demo_hrnet_enhanced_20260728/robust_3d_global_axes_motion_peak_noswap_20260729/triangulated_hrnet_global_axes_motion_peak_noswap_fixed_axis_95.mp4`
- Before/after video: `camtest/line/demo_hrnet_enhanced_20260728/robust_3d_global_axes_motion_peak_noswap_20260729/comparison_wrong_xy_vs_global_xy_3d.mp4`
- Camera layout: `camtest/line/line_refine_global_axes_20260729/camera_layout_before_after.png`

## Important findings

The earlier low per-camera floor reprojection RMS was misleading because each
camera used image-relative floor XY directions. Cam1/cam2/cam3/cam4 required
global rotations of 0/90/180/270 degrees. HRNet also must not reuse the fixed
cam3/cam4 left-right swap previously found for YOLO Pose.

## Known remaining work

- Cam2–cam4 epipolar median remains higher (`9.24 px`; torso center `11.82 px`).
- Fix the latent Viterbi recovery bug for future frames where every camera subset fails.
- Add bounded-gap interpolation and reject implausible learned bone priors.
- Optionally refine small residual extrinsic errors using reliable torso trajectories.
