# Calibration Improvement Plan for Karate Capture Space

Owner: yp8700 + Codex
Last updated: 2026-04-15
Workspace: /home/yp8700/amass/amass

## Goal
- Define a practical external-calibration strategy for a traditional karate capture setup.
- Choose a method that matches the real capture volume instead of assuming a single board can solve every scale.
- Keep the workflow compatible with the current RGB camera + OpenCV + ChArUco based project.

## Problem Context
- The project target is karate motion capture / action detection.
- A single A1 ChArUco board is already large, but recent debug runs showed that "board visible" does not automatically mean "enough Charuco corners for stable extrinsics."
- For larger spaces, calibration quality is limited more by geometry coverage, viewing angle, blur, and board coverage than by raw board size alone.

## Recommended Method by Capture Area

### Small capture area: 4m x 4m to 6m x 6m
Best fit:
- Single ChArUco / AprilGrid large board
- Multi-frame extrinsics estimation

Use when:
- Single-person kihon
- Short striking combinations
- Limited translation in space

Why it works:
- The board can occupy enough image area in all cameras.
- One common board coordinate system is usually sufficient.
- Existing repo scripts already match this workflow.

### Medium capture area: 6m x 6m to 8m x 8m
Best fit:
- Multi-position A1 board
- Global extrinsics optimization across positions

Use when:
- Single-person kata
- Four-camera ring around the performance area
- Need better coverage across the whole tatami-like region

Why this is the recommended mainline:
- More realistic than making a single larger board.
- Lets each camera observe the board well at different locations.
- Keeps the current ChArUco / OpenCV toolchain.
- Fits the current project better than immediately switching to wand calibration.

### Large capture area: 8m x 8m active area or larger
Best fit:
- Wand + reference frame
or
- Fixed multi-marker layout + moving board hybrid

Use when:
- Kumite-like movement range
- Full competition-scale capture volume
- Large subject translation across the space

Why:
- A single board becomes inefficient for all cameras at all parts of the space.
- Large spaces benefit from calibration geometry spread across the whole volume.

## Recommended Choice for This Project
- Primary recommendation: multi-position A1 board + global optimization
- Secondary fallback: single-board multi-frame calibration only for early small-area tests
- Long-term upgrade path: multi-marker or wand-style calibration if the capture area grows to full kumite scale

Reason:
- The project is currently built around standard cameras, OpenCV-based ChArUco tools, and a custom 3D pose pipeline.
- The medium-area strategy gives the best balance between accuracy, implementation cost, and compatibility with the current repo.

## World Coordinate System Recommendation
- Do not keep the final world origin permanently at the board center.
- Use the board only to build a temporary shared coordinate frame.
- After calibration, redefine the final world frame to the floor center of the karate action area.

Recommended final axes:
- X: left-right across the action area
- Y: forward-back across the action area
- Z: vertical up

Benefits:
- Easier interpretation of stance, displacement, and technique trajectories
- Cleaner downstream use in motion analysis and transformer training

## Detailed Implementation Plan

### Option A: Single-board multi-frame calibration
Use for:
- Early testing
- Small capture volume

Workflow:
1. Calibrate intrinsics per camera using the intrinsics ChArUco board.
2. Place the extrinsics board near the center of the capture area.
3. Record short videos per camera while keeping the board visible and large in frame.
4. Run `calibrate_extrinsics.py` on each camera.
5. Use many frames, not one frame, to average out corner noise.
6. Convert the temporary board-centered world frame into a floor-centered action frame.

Acceptance:
- Each camera should consistently detect many Charuco corners.
- Frame count passing `min_corners` should be comfortably above the threshold.
- Reprojection RMS should stay stable across retained frames.

### Option B: Multi-position A1 board + global optimization
Use for:
- Main four-camera karate setup
- 6m x 6m to 8m x 8m action area

Workflow:
1. Mark several board positions on the floor across the action area.
   Suggested layout:
   - center
   - front-left
   - front-right
   - back-left
   - back-right
2. At each marked position, record a short board video for all cameras.
3. For each camera and board position, estimate a pose from multiple good frames.
4. Store all valid board poses and reprojection errors.
5. Solve for a single consistent camera extrinsics set across all board positions.
6. Define the final world frame at the action-area center on the floor.

What this improves:
- Better geometric coverage across the whole movement area
- Less dependence on any one camera seeing one board perfectly
- More stable calibration than a single-position board

Implementation note for this repo:
- Keep the current intrinsics pipeline.
- Extend the extrinsics process from one `outer_cameraX.mp4` to multiple position captures.
- Add a later aggregation step instead of overwriting calibration from one short clip.

### Option C: Fixed markers + moving board hybrid
Use for:
- Larger dojo space
- Need to keep OpenCV workflow without a dedicated mocap wand system

Workflow:
1. Place multiple fixed ArUco / AprilTag markers around the perimeter.
2. Measure or jointly estimate their layout in one shared world frame.
3. Use the moving A1 board inside the main action area to improve central accuracy.
4. Solve all marker and board observations together.

What this improves:
- Gives the calibration system spatial anchors across the whole room
- Avoids forcing one board to serve all distances and angles

### Option D: Wand + reference object
Use for:
- Full competition-scale or near competition-scale capture
- Long-term professional calibration workflow

Workflow:
1. Define origin and axes using an L-frame or equivalent reference object.
2. Move a calibrated wand through the whole capture volume.
3. Let all cameras observe the wand across many positions.
4. Solve camera extrinsics from the wand trajectories.
5. Align the final world frame with the karate floor center.

Tradeoff:
- Highest scalability for large spaces
- More setup and tooling complexity than current repo workflows

## Practical Capture Rules
- The board must occupy enough image area; "board visible" is not enough.
- Prefer more good corners across many frames over one dramatic angled frame.
- Avoid strong slant, motion blur, glare, and partial board visibility.
- If a frame only yields a few Charuco corners, treat it as visibility confirmation, not reliable calibration evidence.
- For larger spaces, improve geometric coverage before trying to print larger boards.

## Near-Term Action Items
1. Keep the updated board definitions:
   - intrinsics board: 9 x 6, 55mm / 41mm
   - extrinsics board: 9 x 6, 78mm / 58mm
2. Use `debug_charuco.py` first to verify board visibility and corner count.
3. Decide the effective karate action area to capture:
   - small technical zone
   - kata-sized zone
   - kumite-sized zone
4. If the target area is kata-sized, move forward with the multi-position A1 board plan.
5. Only consider wand or hybrid marker systems if the capture area expands beyond what the board-based method can stably cover.

## Bottom Line
- Do not scale the solution by endlessly enlarging the board.
- Scale the solution by improving calibration geometry.
- For this project, the best next step is to adopt a multi-position A1 board workflow and define the final world frame at the karate action-area floor center.
