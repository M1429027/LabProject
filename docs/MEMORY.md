# MEMORY

Last updated: 2026-03-30
Owner: POYI (M1429027)

## Session Bootstrap
- Mandatory first step for every new session: read this file (`docs/MEMORY.md`).
- After reading, report a short bootstrap summary:
  - memory version date
  - workspace path
  - active git workflow
  - active execution policy
- If this file and ad-hoc instructions conflict, follow this file unless user explicitly overrides.

## Workspace Domains
Monorepo strategy: **Two domains in one repo**.

### Domain A: `camera_system/`
Scope:
- camera capture, camera GUI, camera recording
- intrinsics/extrinsics calibration workflows
- calibration-focused debug tools

Current mapping (phase 2 started):
- canonical: `camera_system/camera_calibration/*`
- compatibility alias: root `camera_calibration` symlink -> `camera_system/camera_calibration`

### Domain B: `reconstruction_pipeline/`
Scope:
- 2D keypoint extraction
- time alignment, triangulation, 3D reconstruction
- reconstruction evaluation and debug tools

Current mapping (phase 2 started):
- canonical: `reconstruction_pipeline/algorithm_pipeline/*`
- compatibility alias: root `algorithm_pipeline` symlink -> `reconstruction_pipeline/algorithm_pipeline`

Compatibility note:
- During transition, legacy folders remain valid runtime paths.
- New code should follow domain boundaries and avoid cross-domain coupling.

## Tool Discovery
When a new debug/tool task starts, discover in this order:
1. `scripts/` (general reusable tools)
2. `camera_system/debug_tools/` (camera-specific)
3. `reconstruction_pipeline/debug_tools/` (reconstruction-specific)
4. `docs/` command notes or runbooks

Tool placement rules:
- General purpose tool: `scripts/`
- Camera-specific tool: `camera_system/debug_tools/`
- Reconstruction-specific tool: `reconstruction_pipeline/debug_tools/`
- One-off experiment should not stay in source roots without clear owner/purpose.

## Output Policy
Unified output root: `outputs/`
- `outputs/calibration/`
- `outputs/reconstruction/`
- `outputs/debug/`
- `outputs/reports/`

Run folder naming (mandatory):
- `YYYYMMDD_HHMMSS_<tag>`
- Example: `20260330_142500_shift46_compare`

Rules:
- New generated outputs must go under `outputs/`.
- Legacy roots (e.g., `real_world_output_*`) are considered transition/legacy and should be gradually retired.

## Code Style
Maintainability and debugability first:
- Keep functions focused and short; avoid one giant `main()`.
- Separate I/O orchestration from computational logic.
- Keep logging consistent with step labels and concise messages.
- Prefer explicit variable names over abbreviations.
- Comments should explain why, not restate obvious code.
- Use English for comments/logs to avoid encoding issues.

## Naming Template
- Python files/functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Config files: `config_<domain>_<target>.yaml`
- Reports: `<scope>_<metric>_report.json`
- Debug scripts: `debug_<topic>.py`
- Output run folder: `YYYYMMDD_HHMMSS_<tag>`

## Git Workflow
Fixed workflow:
- `feature branch -> PR -> merge to main`
- Branch prefix: `codex/<topic>`
- Do not push feature work directly to `main`.

Commit message template:
- `type(scope): summary`
- Examples: `feat(calibration): add charuco dict validation`, `fix(pipeline): correct shift alignment logging`

PR required sections:
- goal
- change summary
- test evidence
- risk and rollback note

## Execution Policy
Default behavior:
- Inspect/check class operations execute directly, then report results.
- Mutating operations (write/move/delete/sudo/git push): ask for confirmation first, unless the user explicitly requested that mutation in the current message.

Inspect/check class (always auto-run, no pre-ask):
- list/search/read files
- inspect configs/logs
- run diagnostics
- check/status/verify/find/grep-type commands
- any command whose primary purpose is inspection or validation, even if tooling does not strictly tag it as read-only

Reporting rule:
- run first, then summarize key findings in the same response.

Mutating examples:
- edit files
- move/delete files or folders
- privileged commands (`sudo`)
- branch/push/force operations
## Operating Environment
- Host OS: Windows
- Runtime: WSL Ubuntu (Linux)
- Working repo path: `/home/yp8700/amass/amass`
- Git remote: `https://github.com/M1429027/LabProject.git`

## Calibration Defaults In Use
- Video resolution: 1920x1080
- Effective ROI: x=192, y=54, w=1536, h=972
- Intrinsics board:
  - squares_x=10, squares_y=7
  - square_size_mm=25, marker_size_mm=18
  - aruco_dict=DICT_4X4_50
  - legacy_pattern=true
- Extrinsics board:
  - squares_x=9, squares_y=6
  - square_size_mm=55, marker_size_mm=41
  - aruco_dict=DICT_4X4_50
  - legacy_pattern=true

## Memory File Rules
- Update this file when user requests memory/rule changes.
- This is the single source of operational rules for this repository.


