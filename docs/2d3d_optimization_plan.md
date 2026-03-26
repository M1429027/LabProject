# 2D->3D Reconstruction Optimization Plan

Owner: yp8700 + Codex
Last updated: 2026-03-20
Workspace: /home/yp8700/amass/amass

## Goal
- Reduce final multi-view reprojection error and improve 3D skeleton stability.
- Keep implementation incremental and reversible.

## Current Baseline (real_world_output_demo)
- Reprojection error (overall): mean 51.57 px, median 41.48 px, p95 129.47 px
- Reprojection error (cam1): mean 52.07 px, median 42.74 px, p95 128.58 px
- Reprojection error (cam2): mean 51.07 px, median 40.45 px, p95 130.72 px
- Intrinsics RMS: cam1 1.3351 px, cam2 1.3713 px
- Extrinsics RMS: cam1 median 0.5575 px, cam2 median 0.7971 px

## Why baseline is high
- Likely time offset between two recorded videos.
- 2D confidence-only filtering is not enough for geometric consistency.
- No robust loss in triangulation refinement for outliers.
- Potential calibration generalization issue (especially cam2 intrinsics appears unstable).

## Implementation Roadmap

### Phase 0 - Consistency Check (do first)
Status: TODO

Deliverable:
- Script: `check_pipeline_consistency.py`

Checks:
- Same resolution across video/intrinsics/extrinsics/pipeline.
- Same ROI definition across intrinsics/extrinsics/inference.
- Correct cam1<->cam1 and cam2<->cam2 file pairing.
- ChArUco metadata consistency (dict, board size, marker/square size, legacy_pattern).
- Projection sanity (baseline length, principal point in expected range).

Acceptance:
- Script outputs PASS/WARN/FAIL summary before every pipeline run.

---

### Phase 1 - Offline Time Alignment
Status: TODO

Deliverable:
- Script: `estimate_time_offset.py`

Method:
- Coarse stage: motion-energy cross-correlation on 2D keypoints.
- Fine stage: evaluate candidate offsets by triangulation reprojection median.
- Output fixed frame offset and aligned frame range.

Acceptance:
- Estimated offset saved to JSON.
- Re-run with offset should reduce reprojection median.

---

### Phase 2 - 2D Quality Gating (without over-pruning)
Status: TODO

Deliverable:
- Add pre-triangulation gate in `real_world_pipeline.py`.

Rules:
- Keep existing confidence threshold.
- Add dual-view confidence gate (both cameras must pass min confidence for a joint).
- Add geometric gate (epipolar or reprojection gate).
- Add minimum ray-angle gate (avoid near-parallel rays).

Anti-over-prune policy:
- Use soft fallback: if strict gate fails, keep point only if robust refinement still converges.
- Track dropped-point ratio in report.

Acceptance:
- Missing-joint ratio does not explode.
- Reprojection p95 drops.

---

### Phase 3 - Robust Triangulation Refinement (Huber)
Status: TODO

Deliverable:
- Replace LM-only residual minimization with robust least squares (`loss='huber'`).

Method:
- Initialize by SVD triangulation.
- Non-linear refinement with Huber loss and tunable `delta`.
- Keep RANSAC inlier selection for >2 views (future-proof).

Acceptance:
- Outlier-heavy frames show lower error spikes.
- Mean/median reprojection error decreases.

---

### Phase 4 - Bone Length Constraint (lightweight)
Status: TODO

Deliverable:
- Optional post-process module: `enforce_bone_length.py`

Method:
- Estimate subject bone lengths from high-quality frames (median per bone).
- Per frame, apply lightweight correction to reduce bone-length jitter.
- No manual height/arm-span measurement required.

Acceptance:
- Bone-length coefficient of variation drops.
- Visual artifacts ("stretch/shrink") reduced.

---

### Phase 5 - Temporal Filter (postpone)
Status: POSTPONED

Note:
- Keep for final polishing (One-Euro/Kalman), after geometric issues are fixed.

## Metrics to Track Every Iteration
- overall reprojection: mean / median / p95 (px)
- per-camera reprojection: mean / median / p95 (px)
- valid joint ratio
- dropped joint ratio by each gate
- offset used (frames)
- runtime (optional)

## Execution Order for Next Sessions
1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5 (later)
